"""Summarising compaction for conversations that outgrow Agentaus' context window.

Trimming the oldest messages keeps a turn alive but throws the early conversation
away - decisions, file paths, constraints the user stated once and expects to hold.
This module replaces that head of the conversation with a summary instead, so the
detail survives in compressed form.

The hard constraint is that the bridge is **stateless**: Claude Code re-sends the whole
conversation every turn, so a naive implementation would re-summarise the entire history
on every single request - slow, and paid for again each time. The fix is that the
summarised region is a stable *prefix*. Turn N and turn N+1 summarise the same head, so
the result is cached by content hash and computed once, not once per turn.

Layout of a compacted request:

    system prompt + "[earlier conversation summary] ..."     <- compressed head
    <most recent messages, verbatim>                         <- untouched tail

The summary goes in the system prompt rather than in a message so the message sequence
keeps its shape: injecting a synthetic turn risks two user messages in a row and
detaches tool_result blocks from the tool_use they answer.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import unicodedata
import logging
from collections import OrderedDict
from typing import Awaitable, Callable

from .gate import hold
from .translate import (as_sent, content_text, estimate_request_tokens, estimate_tokens,
                        has_unseen, unseen_note)
# Re-exported: these lived here first and several modules import them from here.
from .text import normalise_for_display, normalise_identifiers  # noqa: F401
from . import prompt_style

log = logging.getLogger("agentaus-bridge")

# What the summariser is asked to preserve. Written as an extraction task rather than
# "summarise this", because a prose summary of an engineering session reliably drops
# the specifics - paths, identifiers, versions - that later turns depend on.
SUMMARY_INSTRUCTION = """\
You are compacting the earlier part of a software engineering conversation so it can \
continue within a smaller context window. Produce a dense factual record, not prose.

Preserve, exactly as written wherever they appear:
- File paths, function, class and variable names, commands, flags and version numbers
- Decisions that were made, and the reason each one was chosen
- Constraints, preferences and requirements the user stated
- Bugs found, their root cause, and how each was fixed
- What has been completed versus what is still outstanding or was deferred
- Error messages and their resolutions
- Any values that later steps depend on: ports, URLs, identifiers, limits

Rules:
- Never invent detail that is not present. Omission is fine; fabrication is not.\n- Reproduce every identifier EXACTLY: file paths, names, flags, versions, region\n  codes. Use plain ASCII hyphens and quotes, never typographic ones, and never\n  wrap an identifier in $$. Put identifiers in backticks.
- Keep specifics over generalities: "set BRIDGE_PORT=9100 in .env" beats "changed config".
- Use terse bullet points grouped under short headings.
- Do not add commentary, preamble, or a closing summary.

The conversation is inside <conversation> tags below. Your output is the record itself,
with no tags around it.
"""


# A single summarising pass reliably drops specifics. This second pass shows the model
# the source again alongside the summary and asks only "what is missing" - a much
# easier question to answer well than "summarise this", and it recovers detail the
# first pass elided.
GAP_INSTRUCTION = """\
A <summary> of part of a software engineering conversation is below, followed by the \
<original> text it was made from.

List any concrete facts present in the ORIGINAL but missing from the SUMMARY. Focus on:
file paths, function and variable names, commands, flags, version numbers, ports, URLs, \
numeric limits, decisions and the reasons for them, bugs and their root causes, and \
anything stated as a requirement or constraint.

Output only the missing facts as terse bullet points. If nothing of substance is \
missing, output exactly: NONE

"""

# Merging concatenated per-chunk summaries produces repetition and loses ordering.
# Asking the model to reorganise them keeps the record readable as one account.
MERGE_INSTRUCTION = """\
Inside <summaries> below are summaries of consecutive parts of one software \
engineering conversation. Merge them into a single coherent record.

- Keep every specific fact: paths, identifiers, commands, numbers, decisions and reasons.
- Remove duplication, but never drop a detail that appears only once.
- Preserve chronological order where it matters, and group related points under headings.
- Do not add anything that is not present below.

"""


Summariser = Callable[[str], Awaitable[str]]


# The Markdown layout refers to its blocks by heading, so the three sentences that name a
# tag get their own wording there. The XML wording above is left exactly as it was.
_MD_WORDING = (
    ("The conversation is inside <conversation> tags below. Your output is the record itself,\n"
     "with no tags around it.",
     "The conversation is in the fenced block below. Your output is the record itself, with\n"
     "no fence or heading around it."),
    ("A <summary> of part of a software engineering conversation is below, followed by the "
     "<original> text it was made from.",
     "A summary of part of a software engineering conversation is below, followed by the "
     "original text it was made from."),
    ("Inside <summaries> below are summaries of consecutive parts of one software "
     "engineering conversation.",
     "Below are summaries of consecutive parts of one software engineering conversation."),
)


def _styled(instruction: str) -> str:
    if not prompt_style.markdown():
        return instruction
    for xml_wording, md_wording in _MD_WORDING:
        instruction = instruction.replace(xml_wording, md_wording)
    return prompt_style.headings(instruction)



# Models reformat punctuation when they write prose: ASCII hyphens become U+2011
# NON-BREAKING HYPHEN, quotes become curly, and identifiers sometimes arrive wrapped
# in $$...$$ as though they were mathematics. Observed from Agentaus:
#
#     EU-WEST-2        -> EU\u2011WEST\u20112
#     retry_budget_ms  -> $$retry_budget_ms$$
#
# In prose that is cosmetic. In a summary that a coding agent will read back and act
# on, it is corruption: a region code or file path carrying a typographic hyphen looks
# correct and is not, which is worse than it being missing. Normalising deterministically
# is more reliable than asking the model not to do it.
def _clean_turn_start(message: dict) -> bool:
    """Whether the conversation may validly begin at this message.

    A user message holding `tool_result` blocks answers a `tool_use` in the assistant
    turn before it. Starting there orphans the result, which Agentaus rejects.
    """
    if message.get("role") != "user":
        return False
    content = message.get("content")
    if isinstance(content, list):
        return not any(
            isinstance(b, dict) and b.get("type") == "tool_result" for b in content
        )
    return True


def split_head_tail(
    messages: list, keep_tokens: int, *, block: int = 0
) -> tuple[list, list]:
    """Split into (head to summarise, tail to keep verbatim).

    The tail grows backwards from the newest message until it reaches `keep_tokens`,
    then extends further until it starts on a clean turn. The newest message is always
    in the tail: it is the one being answered.

    "Until it starts on a clean turn" usually means moving the boundary forward, which
    keeps the tail small. When compaction lands mid-tool-loop there is no clean turn
    ahead to move to, and the boundary has to go backwards instead - see below.

    `block` snaps the boundary up to a multiple of that many messages. Without it the
    boundary advances by roughly one turn every turn, so the summary of the head is
    keyed on something that changes constantly and the cache never hits - every single
    turn re-summarises the entire history from scratch. Measured on a real session that
    was 173 seconds per turn. Snapping upward keeps the boundary still for many turns,
    and upward rather than downward so the tail can only get smaller, never overflow.
    """
    if not messages:
        return [], []

    tail: list = []
    used = 0
    for message in reversed(messages):
        # Costed as Agentaus receives it: an image is its note, not its base64, or one
        # recent screenshot overflows keep_tokens and pushes nearly everything into the head.
        sized = as_sent(message) if has_unseen(message) else message
        cost = estimate_tokens(json.dumps(sized, default=str))
        if tail and used + cost > keep_tokens:
            break
        tail.insert(0, message)
        used += cost

    idx = len(messages) - len(tail)

    if block > 1 and idx > 0:
        snapped = ((idx + block - 1) // block) * block
        idx = min(snapped, len(messages) - 1)

    # The tail must open on a clean user turn. Prefer the nearest one at or after the
    # boundary, because moving forward keeps the tail small.
    forward = idx
    while forward < len(messages) and not _clean_turn_start(messages[forward]):
        forward += 1

    if forward < len(messages):
        idx = forward
    else:
        # Nothing clean ahead: the conversation ends mid-tool-loop, so every message
        # from the boundary on belongs to an exchange that has not closed. Stopping at
        # the last message would open the tail on a `tool_result` whose `tool_use` had
        # just gone into the head, and Agentaus rejects that with a bare 400 - no
        # mention of prompt length, because length is not the problem. Walk back to the
        # last clean turn instead: it keeps the pair whole for a slightly larger tail.
        while idx > 0 and not _clean_turn_start(messages[idx]):
            idx -= 1
    return messages[:idx], messages[idx:]


def render_for_summary(messages: list) -> str:
    """Flatten messages into text a model can summarise.

    Tool traffic is included but labelled, because which tool ran and what it returned
    is often the substance of an engineering conversation.
    """
    lines: list[str] = []
    for message in messages:
        role = message.get("role", "?")
        content = message.get("content")
        if isinstance(content, str):
            lines.append(f"[{role}] {content}")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                lines.append(f"[{role}] {block.get('text', '')}")
            elif kind == "tool_use":
                args = json.dumps(block.get("input") or {}, default=str)[:600]
                lines.append(f"[{role} calls {block.get('name')}] {args}")
            elif kind == "tool_result":
                body = block.get("content")
                # An image becomes its note: the summary must not carry base64, or lose
                # that the model never saw it.
                text = (body if isinstance(body, str) else content_text(body)
                        if has_unseen(body) else json.dumps(body, default=str))
                lines.append(f"[tool result] {str(text)[:1500]}")
            elif kind in ("image", "document"):
                lines.append(f"[{role}] {unseen_note(kind)}")
    return "\n".join(lines)


def _chunk(text: str, budget_tokens: int, *, overlap_ratio: float = 0.08) -> list[str]:
    """Split text into pieces that each fit one summarisation call.

    Sizing is by characters throughout, never by line count. `render_for_summary`
    emits one line per message, and a single message can be tens of thousands of
    characters, so a line-counted overlap carries far more than intended: an earlier
    version overlapped 30 lines and produced chunks six times larger than the model's
    entire context window, which the API rejected outright.

    A line longer than the budget on its own is hard-split rather than emitted whole,
    for the same reason.
    """
    budget_chars = max(budget_tokens * 4, 2000)
    overlap_chars = int(budget_chars * overlap_ratio)

    units: list[str] = []
    for line in text.splitlines():
        if len(line) <= budget_chars:
            units.append(line)
        else:
            for i in range(0, len(line), budget_chars):
                units.append(line[i:i + budget_chars])

    chunks: list[str] = []
    current: list[str] = []
    size = 0
    for unit in units:
        cost = len(unit) + 1
        if current and size + cost > budget_chars:
            chunks.append("\n".join(current))
            # Carry back a bounded tail so a point made either side of the boundary
            # reaches one summariser whole.
            carry: list[str] = []
            carried = 0
            for previous in reversed(current):
                if carried + len(previous) + 1 > overlap_chars:
                    break
                carry.insert(0, previous)
                carried += len(previous) + 1
            current, size = carry, carried
        current.append(unit)
        size += cost
    if current:
        chunks.append("\n".join(current))
    return chunks


_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)
_NOTIFICATION = re.compile(r"<task-notification>.*?</task-notification>", re.S)
_VERBATIM_CHARS = 12_000


def _cut(text: str, limit: int = _VERBATIM_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[: limit * 2 // 3] + f"\n[... {len(text) - limit} characters ...]\n" + text[-limit // 3:]


def verbatim_record(head: list) -> str:
    """What compaction keeps whole from the messages it summarises: the first thing the user
    said, and every `<task-notification>` (a background workflow's result). "" if neither."""
    first, reports = "", []
    for message in head:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        texts = [content] if isinstance(content, str) else [
            b.get("text", "") for b in content or [] if isinstance(b, dict) and b.get("type") == "text"]
        for text in texts:
            reports += _NOTIFICATION.findall(text or "")
            if not first:
                first = _REMINDER.sub("", text or "").strip()
    parts = []
    if first:
        parts.append("The user's first message, word for word:\n" + _cut(first))
    parts += ["A background task's report, word for word:\n" + _cut(r) for r in reports]
    return ("\n\n[Kept whole from the summarised messages]\n" + "\n\n".join(parts)) if parts else ""


class ConversationCompactor:
    """Replaces the head of a conversation with a summary produced by the model.

    Quality is chosen over economy throughout: chunks are summarised concurrently, a
    second pass re-reads the source to recover specifics the first pass dropped, and
    partial summaries are merged by the model rather than concatenated. The expensive
    result is cached by content hash, so the cost lands once per boundary move rather
    than once per turn.
    """

    def __init__(
        self,
        summarise: Summariser,
        *,
        cache_size: int = 64,
        verify: bool = True,
        block: int = 20,
    ) -> None:
        self._summarise = summarise
        # Compaction itself blocks the turn, so it is not background work.
        self._priority = "normal"
        self._cache: OrderedDict[str, str] = OrderedDict()
        # Per-chunk summary cache: chunk-text hash -> summary. A cold compaction that dies at
        # the merge (or is cut by BRIDGE_TURN_TIMEOUT) otherwise re-runs all ~34 chunk and gap
        # calls on the client's retry. Cached chunks resume instantly, so retry is cheap.
        self._chunk_cache: OrderedDict[str, str] = OrderedDict()
        # Head-hashes that blew the summarisation budget (see compact()); an immediate retry with
        # the same head trims at once rather than re-entering the frozen loop (Step 3).
        self._timeout_mem: OrderedDict[str, bool] = OrderedDict()
        # (length, hash, summary) for prefixes already summarised.
        self._prefixes: list[tuple[int, str, str]] = []
        self._cache_size = cache_size
        self._verify = verify
        self._block = block
        self.hits = 0
        self.misses = 0
        self.calls = 0

    @staticmethod
    def _key(messages: list) -> str:
        return hashlib.sha256(
            json.dumps(messages, default=str, sort_keys=True).encode()
        ).hexdigest()

    def _remember(self, key: str, summary: str) -> None:
        self._cache[key] = summary
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    @staticmethod
    def _chunk_key(chunk: str) -> str:
        """Content hash of one chunk's rendered text, so a retry recognises already-done chunks."""
        return hashlib.sha256(chunk.encode("utf-8", "replace")).hexdigest()

    def _cached_chunk(self, chunk: str) -> str | None:
        key = self._chunk_key(chunk)
        summary = self._chunk_cache.get(key)
        if summary is not None:
            self._chunk_cache.move_to_end(key)
        return summary

    def _remember_chunk(self, chunk: str, summary: str) -> None:
        # Written after the first pass AND after the gap pass, so an interrupted run resumes
        # from however far it got instead of redoing every chunk upstream call.
        self._chunk_cache[self._chunk_key(chunk)] = summary
        self._chunk_cache.move_to_end(self._chunk_key(chunk))
        while len(self._chunk_cache) > self._cache_size:
            self._chunk_cache.popitem(last=False)

    async def _call(self, prompt: str) -> str:
        """One summariser call, bounded by the bridge-wide concurrency cap.

        The cap is shared with every other bridge-initiated call rather than private to
        the compactor: a long compaction and a search running at once must add up to
        one limit, not two.
        """
        async with hold("summarisation", self._priority):
            self.calls += 1
            return (await self._summarise(prompt)) or ""

    @staticmethod
    def _raw_fallback(chunk: str, limit: int = 6000) -> str:
        """Last resort when a chunk cannot be summarised at all.

        Passing the raw head through, truncated, keeps its facts in play. Returning
        nothing would silently delete that slice of the conversation, which is the
        failure this whole module exists to prevent.
        """
        if len(chunk) <= limit:
            body = chunk
        else:
            # Keep both ends: openings carry decisions, closings carry outcomes.
            half = limit // 2
            body = chunk[:half] + "\n[...]\n" + chunk[-half:]
        return "[Unsummarised excerpt - the summariser failed on this section]\n" + body

    async def _summarise_chunk(self, chunk: str, index: int, total: int) -> str:
        label = f" (part {index} of {total})" if total > 1 else ""
        cached = self._cached_chunk(chunk)
        if cached is not None:
            return cached
        # Remember one in-progress marker so a retry that dies mid-merge does not re-run this
        # specific call, and write the real summary as soon as it is known.
        summary = normalise_identifiers(
            (await self._call(
                _styled(SUMMARY_INSTRUCTION) + label
                + "\n" + prompt_style.data("conversation", chunk) + "\n"
            )).strip()
        )
        self._remember_chunk(chunk, summary)
        if not self._verify:
            return summary

        # Second pass: ask what the first pass missed, and fold it back in. This is
        # the single biggest fidelity win - "what is missing" is a far easier question
        # for the model than "summarise well".
        # The gap pass is a quality improvement: the summary is usable without it, so it
        # yields to anything a turn is blocked on.
        previous, self._priority = self._priority, "background"
        try:
            gaps = normalise_identifiers((await self._call(
                _styled(GAP_INSTRUCTION)
                + "\n" + prompt_style.data("summary", summary)
                + "\n\n" + prompt_style.data("original", chunk) + "\n"
            )).strip())
        finally:
            self._priority = previous
        if gaps and gaps.upper().strip().rstrip(".") != "NONE":
            summary = summary + "\nAdditional details:\n" + gaps
        # The gap pass finished: cache the final (first pass + gaps) result for this chunk so
        # a later retry resumes past both upstream calls, not just the first one.
        self._remember_chunk(chunk, summary)
        return summary

    def _find_prefix(self, head: list) -> tuple[int, str] | None:
        """Longest cached summary covering a prefix of `head`.

        The head grows by appending, so last turn's summary almost always covers most
        of this turn's head. Reusing it turns the cost from "summarise everything" into
        "summarise what arrived since", which is the difference between minutes and
        seconds on a long session.
        """
        best: tuple[int, str] | None = None
        for length, entry_hash, summary in self._prefixes:
            if length >= len(head) or (best and length <= best[0]):
                continue
            if self._key(head[:length]) == entry_hash:
                best = (length, summary)
        return best

    def _remember_prefix(self, head: list, summary: str) -> None:
        self._prefixes.append((len(head), self._key(head), summary))
        while len(self._prefixes) > self._cache_size:
            self._prefixes.pop(0)

    async def _merge_summaries(self, parts: list[str]) -> str:
        """Merge many summaries into one, in a bounded fan-in.

        The naive version merged every chunk summary in ONE prompt - a ~34-part merge that
        took 4-7 minutes serial and blew the turn deadline. Merging in small batches (each
        call handles at most 4 summaries) is log-depth, every call is small enough to stream
        back quickly, and the batches within a level run concurrently under the gate. Falls
        back to a raw join on any failure - a broken merge must not lose the record.
        """
        batch = 4
        while len(parts) > 1:
            groups = [parts[i:i + batch] for i in range(0, len(parts), batch)]

            async def _merge_group(group: list[str]) -> str:
                if len(group) == 1:
                    return group[0]
                try:
                    merged = await self._call(
                        _styled(MERGE_INSTRUCTION) + "\n"
                        + prompt_style.data("summaries", "\n\n---\n\n".join(group)) + "\n"
                    )
                    return merged.strip() or "\n\n---\n\n".join(group)
                except Exception as exc:  # noqa: BLE001 - a broken merge must not lose the record
                    log.warning("merge call failed (%s); raw-joining the batch", exc)
                    return "\n\n---\n\n".join(group)

            # The batches within a level run concurrently under the gate; levels wait on the
            # previous one only because a level needs every batch above it.
            parts = await asyncio.gather(*[_merge_group(g) for g in groups])
        return parts[0] if parts else ""

    async def summarise_head(
        self, head: list, *, chunk_budget: int, shrink_limit: int | None = None
    ) -> str:
        """Summarise `head`, reusing cached work wherever possible.

        Three paths, cheapest first: an exact cache hit; an extension of a previously
        summarised prefix, where only the new messages are read; and a cold summary of
        the whole head.

        `shrink_limit` is the most tokens the summary may occupy (the real room left in
        the context window after the verbatim tail and overhead). The pre-standard shrink
        loop compared the summary against the per-chunk size (6,000), so any real summary
        of a large head looked "too big", triggering up to 3 more sequential multi-minute
        merge calls - which is how a ~250k-token head blew past BRIDGE_TURN_TIMEOUT. It is
        only worth shrinking when the summary would not fit the room it actually needs.
        """
        key = self._key(head)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            self.hits += 1
            return cached

        self.misses += 1

        prefix = self._find_prefix(head)
        if prefix is not None:
            covered, prior = prefix
            delta = head[covered:]
            delta_text = render_for_summary(delta)
            if delta_text.strip():
                log.info(
                    "incremental compaction: reusing a summary of %d message(s), "
                    "summarising %d new", covered, len(delta),
                )
                pieces = _chunk(delta_text, chunk_budget)
                new_parts = await asyncio.gather(*[
                    self._summarise_chunk(piece, i, len(pieces))
                    for i, piece in enumerate(pieces, 1)
                ])
                new_parts = [s for s in new_parts if s.strip()]
                summary = await self._merge_summaries([prior] + new_parts)
            else:
                summary = prior
            self._remember(key, summary)
            self._remember_prefix(head, summary)
            return summary

        text = render_for_summary(head)
        pieces = _chunk(text, chunk_budget)

        # Concurrent: a long history is many independent calls, and doing them in
        # sequence is the difference between seconds and minutes of visible latency.
        summaries = await asyncio.gather(*[
            self._summarise_chunk(piece, i, len(pieces))
            for i, piece in enumerate(pieces, 1)
        ])
        summaries = [s for s in summaries if s.strip()]

        if not summaries:
            raise RuntimeError("summariser produced nothing")

        if len(summaries) == 1:
            summary = summaries[0]
        else:
            # Merged by the model rather than concatenated: consecutive chunks overlap,
            # so raw concatenation repeats itself and reads as several disjoint records.
            # Batched tree-merge (see _merge_summaries) so no single call carries ~34
            # summaries - the one that took 4-7 minutes serial and blew the turn deadline.
            summary = await self._merge_summaries(summaries)

        # Shrink only against what the summary actually has room for (the real window space
        # left after tail + overhead), not the per-chunk size. A summary of a large head is
        # allowed to be larger than one chunk; shrinking it down to 6,000 tokens wasted up to
        # three sequential multi-minute calls and blew the turn deadline for zero benefit.
        if shrink_limit and shrink_limit > 0:
            rounds = 0
            while estimate_tokens(summary) > shrink_limit and rounds < 3:
                summary = (await self._call(
                    _styled(MERGE_INSTRUCTION) + "\n" + prompt_style.data("summaries", summary) + "\n"
                )).strip()
                rounds += 1

        self._remember(key, summary)
        self._remember_prefix(head, summary)
        return summary

    async def compact(
        self,
        body: dict,
        *,
        limit: int,
        reserve: int,
        keep_fraction: float = 0.5,
        threshold: float = 1.0,
        chunk_tokens: int | None = None,
        timeout_s: float | None = None,
    ) -> dict:
        """Return a plan describing how to fit `body` into `limit`.

        `threshold` compacts before the window is actually full - at 0.8, compaction
        happens at 80% occupancy. Waiting for the hard limit means every turn near the
        boundary runs with almost no headroom for the reply, and one large tool result
        tips it over into a failure the user sees.

        Keys: `messages`, `summary` (or None), `summarised`, `method`
        ("none" | "summarised" | "trimmed").
        """
        messages = list(body.get("messages") or [])
        overhead = estimate_request_tokens(
            {"system": body.get("system"), "messages": [], "tools": body.get("tools")}
        )
        budget = limit - reserve - overhead

        def total(msgs: list, summary: str = "") -> int:
            return estimate_request_tokens(
                {"system": body.get("system"), "messages": msgs, "tools": body.get("tools")}
            ) + estimate_tokens(summary) + reserve

        trigger = int(limit * threshold)
        if total(messages) <= trigger:
            return {"messages": messages, "summary": None, "summarised": 0, "method": "none"}

        keep_tokens = max(int(budget * keep_fraction), 512)
        head, tail = split_head_tail(messages, keep_tokens, block=self._block)

        if not head:
            return {"messages": tail, "summary": None, "summarised": 0, "method": "none"}

        # The room the summary can actually occupy: the window minus the verbatim tail,
        # the system/tools overhead, and the reply reserve. This - not the per-chunk size -
        # is what the shrink loop should aim at; a summary of a large head that still leaves
        # the request under the window never needs shrinking to 6,000 tokens.
        tail_cost = estimate_request_tokens(
            {"system": body.get("system"), "messages": tail, "tools": body.get("tools")}
        ) + reserve
        shrink_limit = max(limit - tail_cost, chunk_tokens or max(budget // 4, 1000) // 2)

        head_key = self._key(head)
        summary = ""
        # Step 3 fast path: a head whose summarisation already blew the budget is trimmed
        # directly on an immediate retry, instead of re-entering the loop that just froze
        # for a whole compaction. The LRU bound keeps one timeout from poisoning later turns.
        if head_key in self._timeout_mem:
            self._timeout_mem.move_to_end(head_key)
            log.warning("compaction for this conversation already timed out; trimming directly")
        else:
            try:
                run = self.summarise_head(
                    head, chunk_budget=chunk_tokens or max(budget // 4, 1000),
                    shrink_limit=shrink_limit,
                )
                if timeout_s and timeout_s > 0:
                    summary = await asyncio.wait_for(run, timeout=timeout_s)
                else:
                    summary = await run
            except asyncio.TimeoutError:
                self._timeout_mem[head_key] = True
                self._timeout_mem.move_to_end(head_key)
                while len(self._timeout_mem) > self._cache_size:
                    self._timeout_mem.popitem(last=False)
                log.warning(
                    "summarisation exceeded %.0fs; falling back to trimming "
                    "(an immediate retry trims directly)", timeout_s,
                )
            except Exception as exc:  # fall back rather than fail the turn
                log.warning("summarisation failed (%s); falling back to trimming", exc)
            else:
                # It finished (possibly reusing cached chunks): the result is in the summary
                # cache, so a later identical retry is cheap and any earlier flag no longer applies.
                self._timeout_mem.pop(head_key, None)

        # The opening request and any background-task report, word for word, when they fit.
        # Measured (d12 c1913h): one compaction summarised the "ultracode:" request, the
        # Workflow call and its `<task-notification>` together, and the main loop worked on
        # from a paraphrase of both.
        pinned = verbatim_record(head)
        if summary and pinned and total(tail, summary + pinned) <= limit:
            summary = summary + pinned
        if summary and total(tail, summary) <= limit:
            return {
                "messages": tail,
                "summary": summary,
                "summarised": len(head),
                "method": "summarised",
            }

        # Summary absent or still too large: drop the head outright. Worse, but it
        # keeps the turn alive, which is the whole point.
        kept = list(tail)
        dropped = len(head)
        while total(kept) > limit:
            # Drop a whole turn at a time. Popping message by message is what lets the
            # list end up opening on a `tool_result` whose `tool_use` has just been
            # dropped, so advance to the next clean turn or stop.
            nxt = 1
            while nxt < len(kept) and not _clean_turn_start(kept[nxt]):
                nxt += 1
            if nxt >= len(kept):
                # Only the unfinished trailing exchange is left. Cutting into it would
                # orphan a tool_result, which fails the turn outright; going over the
                # limit only might, and Agentaus is the authority on that.
                break
            kept = kept[nxt:]
            dropped += nxt
        return {"messages": kept, "summary": None, "summarised": 0,
                "method": "trimmed", "dropped": dropped}
