#!/usr/bin/env python3
"""Read the bridge log and report what is actually going wrong, ranked.

The bridge compensates for a weaker upstream in a dozen places, and every one of those
compensations logs when it fires. That makes the log a record of which compensations are
earning their keep, which are firing so often they point at a real defect, and which
have never fired at all - and that is a far better guide to what needs fixing than
reading the code and guessing.

Usage:  ./scripts/diagnose.py [/tmp/agentaus-bridge.log]
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

# (label, pattern, severity, what it means, what to do about it)
SIGNALS = [
    ("upstream 5xx / gateway timeout",
     r"HTTP (5\d\d)|returned HTTP 52\d", "high",
     "Agentaus or Cloudflare failed the request outright.",
     "Expected under load. If frequent, lower AGENTAUS_SEARCH_CHUNK_TOKENS or "
     "AGENTAUS_MAX_CONCURRENCY - the adaptive ceiling handles bursts, not sustained load."),

    ("prompt rejected as too long",
     r"rejected the prompt as too long", "high",
     "The conversation exceeded Agentaus' window and had to be recompacted mid-turn.",
     "Lower AGENTAUS_COMPACT_THRESHOLD so compaction happens earlier."),

    ("tool round limit reached",
     r"tool round limit \(\d+\) reached", "high",
     "A turn used its whole tool budget without producing an answer.",
     "Raise AGENTAUS_TOOL_ROUNDS, or bound the task: the model explores indefinitely "
     "when the prompt does not say when to stop."),

    ("model invented a tool name",
     r"model invented \d+ tool name", "medium",
     "Agentaus called a tool nobody offered it. Corrected upstream, never surfaced.",
     "If frequent, the tool list in the prompt is being ignored - shorten it, or check "
     "the <tool_selection> block matches the wire."),

    ("model refused tools it has",
     r"refused to use tools it has", "high",
     "Agentaus claimed it cannot read files while holding tools that can.",
     "The re-ask usually recovers it. If frequent, strengthen the anti-refusal wording "
     "in TOOL_GUIDANCE."),

    ("repeated an identical tool call",
     r"repeated with identical arguments", "medium",
     "Agentaus re-ran a call it had already made. Answered from cache.",
     "Working as intended. High counts mean the tool ledger is not reaching the model."),

    ("chunk split after a timeout",
     r"timed out; splitting", "medium",
     "A search chunk was too slow and was halved rather than dropped.",
     "Working as intended. Sustained counts mean the configured chunk size is too big "
     "for this upstream's current load."),

    ("learned chunk ceiling lowered",
     r"capping search chunks at", "medium",
     "The bridge reduced its own chunk size after a capacity failure.",
     "Informational. If it never lifts again, the configured size is simply too high."),

    ("upstream call stalled",
     r"still waiting after", "high",
     "An upstream call produced nothing for long enough to be reported while it ran.",
     "A dead or wedged connection, not a slow answer. One of these held a run for 26 "
     "minutes while the upstream was healthy. If frequent, lower BRIDGE_READ_TIMEOUT so "
     "the retry happens sooner."),

    ("bridge tool stopped at its deadline",
     r"bridge tool \S+ exceeded", "high",
     "A search, inventory or zoom ran past AGENTAUS_TOOL_TIMEOUT and was stopped.",
     "Usually a path that is far too broad - a home directory, /Applications. The model "
     "was told to narrow it. If it fires on sensible paths, raise AGENTAUS_TOOL_TIMEOUT."),

    ("turn ended at BRIDGE_TURN_TIMEOUT",
     r"turn exceeded BRIDGE_TURN_TIMEOUT", "high",
     "A streamed Agentaus turn ran past its wall-clock ceiling and was ended with an error.",
     "Something inside the turn wedged - look at the phases logged for that request id."),

    ("file walk hit its cap",
     r"stopped at \d+ files \(AGENTAUS_SEARCH_MAX_FILES\)", "medium",
     "A search or inventory tree held more files than AGENTAUS_SEARCH_MAX_FILES.",
     "The result said so. Frequent hits mean the model is pointing tools at trees that "
     "are too broad."),

    ("passthrough stream died mid-response",
     r"passthrough stream failed mid-response", "medium",
     "A Claude response from Anthropic was cut off after it had started.",
     "Network or upstream, not the bridge. The client got an error event and can retry."),

    ("helper call abandoned on timeout",
     r"helper call exceeded", "high",
     "A bridge-initiated call hit AGENTAUS_HELPER_TIMEOUT and was abandoned.",
     "A stalled upstream. If frequent with AGENTAUS_STREAM_HELPERS=true, turn streaming off."),

    ("summarisation fell back to trimming",
     r"summarisation could not reach", "high",
     "Compaction could not summarise enough and dropped messages instead - real context lost.",
     "Lower AGENTAUS_KEEP_FRACTION, or compact earlier."),

    ("self-review revised the answer",
     r"self-review revised the answer", "low",
     "The review pass found a defect and rewrote the answer.",
     "Working as intended - this is the compensation paying off."),

    ("review verdict had to be adjudicated",
     r"review verdict was unstated", "low",
     "The reviewer ignored its output format and had to be asked again.",
     "One extra call. Frequent means REVIEW_INSTRUCTION's format is not landing."),

    ("planning pass failed",
     r"planning pass failed", "medium",
     "The turn proceeded without a plan.",
     "Usually an upstream blip. Frequent means the planning prompt is too large."),

    ("waited for a concurrency slot",
     r"waited [\d.]+s for a slot", "low",
     "A bridge call queued behind the bridge's own background work.",
     "Expected. Long waits mean call volume is too high for AGENTAUS_MAX_CONCURRENCY - "
     "prefer fewer, larger calls over raising the cap."),

    ("client disconnected mid-turn",
     r"client disconnected", "medium",
     "Claude Code gave up before the bridge answered.",
     "The turn was too slow. Check what preceded it - usually compaction or a long search."),

    ("stream closed early",
     r"stream closed early", "medium",
     "The response stream ended before it finished.",
     "Often the client hanging up. Paired with a disconnect, it is the same event."),

    ("in-band error from Agentaus",
     r"agentaus in-band error", "high",
     "Agentaus answered HTTP 200 with an error object instead of content.",
     "The bridge surfaces these correctly; read the message for the real cause."),

    ("distillation failed",
     r"distillation failed", "medium",
     "A tool result could not be condensed; the raw result was kept.",
     "Safe fallback. Frequent means AGENTAUS_DISTILL_CHUNK_TOKENS is too large."),

    ("search found nothing",
     r"search .*-> 0 of", "low",
     "A search read no chunks at all.",
     "Usually a path with nothing readable in it."),
]

TURN = re.compile(r"recv model=(\S+) route=(\w+)")
SEARCH = re.compile(r"bridge tool agentaus_\w+ ran in ([\d.]+)s")
COMPACT = re.compile(r"compaction done in ([\d.]+)s")

# The SIGNALS sweep above is one regex over the whole text, which cannot express three
# things this log actually says: density of 429s (a storm), a "compaction start" with no
# matching end (a hang), and one token modifying another (gemini plus an error). These
# detectors live in main() with line pairing instead. The exact strings this branch
# emits are matched first; the surrounding tokens are a safety net, because the running
# bridge at ../agentaus-bridge-run may log strings a fresh clone does not (gemini).
RATE_LIMIT = re.compile(r"\b429\b|rate[\s-]?limit|quota|exceeded your current quota", re.I)
COMPACT_START = re.compile(r"compaction start")
COMPACT_END = re.compile(r"compaction (?:done in|ended after) ([\d.]+)s")
GEMINI_HINT = re.compile(r"gemini", re.I)
GEMINI_ERROR = re.compile(r"\berror\b|exception|traceback|quota|\b429\b|5\d\d|timed? ?out", re.I)
# A compaction over this long is not slow, it is wedged: BRIDGE_TURN_TIMEOUT defaults to
# 900s, and a phase that beats it is exactly the hang AGENTAUS_COMPACT_TIMEOUT exists to
# bound. Mirrors _Phase's own contract: "a phase that never finishes only shows up in
# the log as a start with no matching end."
COMPACT_HANG_S = 900.0
# A "storm" is this many 429/rate/quota lines, each within this many lines of the next -
# dense retries against an endpoint that is not going to answer, not single blips.
STORM_SPAN_LINES = 60
STORM_MIN_HITS = 5


def _storms(hits: list[int], span: int = STORM_SPAN_LINES, min_hits: int = STORM_MIN_HITS) -> list[list[int]]:
    """Split matching line indices into bursts whose hits stay within `span` lines.

    Short-lived rate limits back off and stop; a storm is retries that keep coming in a
    tight cluster - the "85 x 429s" failure mode. Only clusters dense and large enough
    to matter are returned.
    """
    groups: list[list[int]] = []
    current: list[int] = []
    for i in hits:
        if current and i - current[-1] > span:
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return [g for g in groups if len(g) >= min_hits]


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agentaus-bridge.log")
    if not path.exists():
        print(f"no log at {path}")
        return 1
    text = path.read_text(errors="replace")
    lines = text.splitlines()

    turns = TURN.findall(text)
    routes = Counter(r for _m, r in turns)
    print(f"=== {path}  ({len(lines):,} lines) ===\n")
    print(f"turns: {routes.get('agentaus', 0):,} to Agentaus, "
          f"{routes.get('anthropic', 0):,} passed through to Anthropic\n")

    tool_times = [float(t) for t in SEARCH.findall(text)]
    if tool_times:
        tool_times.sort()
        print(f"bridge tool calls: {len(tool_times)}  "
              f"median {tool_times[len(tool_times)//2]:.0f}s  "
              f"p90 {tool_times[int(len(tool_times)*0.9)]:.0f}s  "
              f"max {tool_times[-1]:.0f}s")
    compactions = [float(t) for t in COMPACT.findall(text)]
    if compactions:
        compactions.sort()
        print(f"compactions      : {len(compactions)}  "
              f"median {compactions[len(compactions)//2]:.1f}s  max {compactions[-1]:.0f}s")
    print()

    found = []
    for label, pattern, severity, means, action in SIGNALS:
        count = len(re.findall(pattern, text))
        if count:
            found.append((count, severity, label, means, action))

    # --- 429 / rate-limit storm -----------------------------------------------------
    rate_hits = [i for i, ln in enumerate(lines) if RATE_LIMIT.search(ln)]
    storms = _storms(rate_hits)
    if storms:
        # One representative row per burst, so a day of scattered clusters reads clearly.
        for hits in storms:
            found.append((len(hits), "high", "rate-limit storm",
                          f"{len(hits)} x HTTP 429 / rate-limit / quota lines within {STORM_SPAN_LINES} "
                          f"lines (lines {hits[0] + 1}-{hits[-1] + 1}).",
                          "The tenant is being hammered and every hit only earned backoff - a 429 is "
                          "retryable, not free. Lower AGENTAUS_MAX_CONCURRENCY / "
                          "AGENTAUS_UPSTREAM_CONCURRENCY, or top up the quota if these are 402s."))

    # --- compaction that started but never finished ----------------------------------
    # Pair each "compaction start" with the next "done in / ended after" line between it
    # and the following start; a start with no such end before the next start (or EOF) is
    # a hang. _Phase logs exactly this contract: "a phase that never finishes only shows
    # up in the log as a start with no matching end."
    starts = [i for i, ln in enumerate(lines) if COMPACT_START.search(ln)]
    ends = [i for i, ln in enumerate(lines) if COMPACT_END.search(ln)]
    hangs = [
        i for idx, i in enumerate(starts)
        if not any(i < e < (starts[idx + 1] if idx + 1 < len(starts) else len(lines)) for e in ends)
    ]
    for i in hangs:
        found.append((1, "high", "compaction never finished",
                      f"a compaction started (line {i + 1}) with no matching 'done in' / "
                      "'ended after' line before the next compaction or end of log.",
                      "Compaction that never ends is the loop that beat BRIDGE_TURN_TIMEOUT. Lower "
                      "AGENTAUS_COMPACT_TIMEOUT so the turn falls back to trimming instead of hanging."))
    # A compaction that finished (or ended on an error) only after a wedged-length time is
    # the same class - it soaked up most of the turn and left nothing for the answer.
    for ln in lines:
        m = COMPACT_END.search(ln)
        if not m:
            continue
        dur = float(m.group(1))
        if dur >= COMPACT_HANG_S:
            found.append((1, "medium", "compaction took too long",
                          f"a compaction ran {dur:.0f}s - at the {COMPACT_HANG_S:.0f}s edge of "
                          f"BRIDGE_TURN_TIMEOUT.",
                          "Lower AGENTAUS_COMPACT_TIMEOUT (defaults to 180s, and 0 disables the "
                          "fallback) so a slow compaction yields to trimming before the turn dies."))

    # --- gemini lines that also carry an error ---------------------------------------
    gemini_bad = [i for i, ln in enumerate(lines)
                  if GEMINI_HINT.search(ln) and GEMINI_ERROR.search(ln)]
    if gemini_bad:
        found.append((len(gemini_bad), "high", "gemini errors",
                      f"{len(gemini_bad)} line(s) mention gemini alongside an error, quota, "
                      f"429 or 5xx (e.g. line {gemini_bad[0] + 1}).",
                      "This branch has no gemini support - the running bridge may log these with "
                      "the API key / quota it holds. Check AGENTAUS_API_KEY and the quota."))

    order = {"high": 0, "medium": 1, "low": 2}
    found.sort(key=lambda row: (order[row[1]], -row[0]))

    print("=== what is firing ===\n")
    for count, severity, label, means, action in found:
        print(f"[{severity.upper():6}] {count:>5} x  {label}")
        print(f"                  {means}")
        print(f"                  -> {action}\n")

    silent = [label for label, pattern, *_ in SIGNALS if not re.search(pattern, text)]
    if silent:
        print("=== never fired (either healthy, or not exercised) ===")
        for label in silent:
            print(f"  - {label}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
