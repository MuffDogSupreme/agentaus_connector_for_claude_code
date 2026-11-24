"""Single source of truth for "what is wrong right now" across Agentaus, Gemini and the
Anthropic passthrough.

The bridge reacts to a struggling upstream in many places - retries, adaptive limits,
compaction timeouts - and each reaction logs when it fires but nothing assembles them into
one picture. A user watching the chat only ever sees "Still waiting for Agentaus (Ns)". This
module is where every producer reports one line, and every consumer (heartbeat text, the
/status page, the Windows toast) reads the assembled picture.

Severity is about how much it hurts. A single 429 that clears in a few seconds is normal; a
turn killed at its deadline is not. Consumers decide what to do with that:
  * the heartbeat text shows whatever is most urgent right now;
  * /status shows the live incident plus recent history;
  * Windows toasts a "soft" incident only once it has lasted AGENTAUS_NOTIFY_AFTER_S, and a
    "hard" one immediately.

Consumers subscribe and are called on every transition. `notify.py` is the Windows-toast
consumer; the heartbeat and /status page poll `active()` / `describe()` on demand.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque, OrderedDict
from dataclasses import dataclass, field
from typing import Callable

from .config import settings

log = logging.getLogger("agentaus-bridge")

# incident kind -> severity. "hard" hurts a turn or the session (alert immediately),
# "soft" is a degraded upstream (alert only once it has lasted a while).
SEVERITY = {
    "rate_limited": "soft",
    "no_connection": "soft",
    "slow_upstream": "soft",
    "quota": "hard",
    "key_rejected": "hard",
    "turn_timeout": "hard",
    "compaction_fallback": "hard",
    "compacting": "soft",
    "settings_drift": "hard",
}

_ORDER = {"hard": 0, "soft": 1}

# Each kind's one-line human phrase. `detail` carries the specifics (status code, attempt,
# elapsed) that vary per occurrence.
_FRASES = {
    "rate_limited": "The upstream is rate-limiting (HTTP 429).",
    "no_connection": "No connection to the upstream (network/DNS error).",
    "slow_upstream": "The upstream is slow or timing out.",
    "quota": "Upstream quota exhausted (HTTP 402) - top up before turns succeed.",
    "key_rejected": "The API key was rejected (HTTP 401/403).",
    "turn_timeout": "A turn hit BRIDGE_TURN_TIMEOUT and was ended.",
    "compaction_fallback": "Compaction could not finish in time and trimmed the "
                          "conversation instead - early context lost.",
    "compacting": "Compacting the earlier conversation to fit the window.",
    "settings_drift": "Settings changed on disk after the bridge started.",
}


@dataclass
class Incident:
    kind: str
    detail: str = ""
    started: float = field(default_factory=time.monotonic)
    occurrences: int = 1

    @property
    def severity(self) -> str:
        return SEVERITY.get(self.kind, "soft")

    def describe(self) -> str:
        base = _FRASES.get(self.kind, self.kind)
        return f"{base} {self.detail}".strip()


_lock = threading.Lock()
# kind -> Incident (unresolved). One live incident per kind; a 429 storm does not spawn one
# Incident per 429, it bumps the occurrences counter.
_active: dict[str, Incident] = {}
# (started, severity, kind, detail, resolved) for recent history, newest first.
_history: deque[tuple[float, str, str, str, bool]] = deque(maxlen=200)
# Consumers told about every incident transition (start = incident begins, resolve = ends).
_subscribers: list[Callable[[Incident, bool], None]] = []


def subscribe(callback: Callable[[Incident, bool], None]) -> None:
    """Register a consumer. Called with (incident, is_resolve). Called under `_lock`, so a
    consumer must not block - a toast launcher hands off to a thread/subprocess."""
    with _lock:
        _subscribers.append(callback)


def _worst_active() -> Incident | None:
    best: Incident | None = None
    for inc in _active.values():
        if best is None or (_ORDER[inc.severity], inc.started) < (_ORDER[best.severity], best.started):
            best = inc
    return best


def record(kind: str, detail: str = "") -> None:
    """Report an incident. Call whenever a producer sees one: throttle failures, 429s,
    connection errors, compaction timeouts. Repeated reports of a live kind bump its
    occurrence counter rather than starting a new incident."""
    now = time.monotonic()
    with _lock:
        inc = _active.get(kind)
        new = inc is None
        if inc is None:
            inc = Incident(kind, detail, started=now)
            _active[kind] = inc
        else:
            inc.occurrences += 1
            if detail and detail != inc.detail:
                inc.detail = detail
        _history.appendleft((now, inc.severity, kind, detail or inc.detail, False))
        if inc.occurrences in (1, 5, 20, 50) or inc.occurrences % 100 == 0:
            log.warning("status: %s (%s), occurrence %d", kind, inc.describe(), inc.occurrences)
        notify = list(_subscribers)
        snapshot = inc
    for callback in notify:
        try:
            callback(snapshot, False)
        except Exception as exc:  # noqa: BLE001 - a broken consumer must not break a turn
            log.warning("status subscriber failed (%s)", exc)


def resolve(kind: str) -> None:
    """Mark an incident resolved (a retry succeeded, the upstream is back)."""
    with _lock:
        inc = _active.pop(kind, None)
        if inc is None:
            return
        _history.appendleft((time.monotonic(), inc.severity, kind, inc.detail, True))
        notify = list(_subscribers)
        snapshot = inc
    for callback in notify:
        try:
            callback(snapshot, True)
        except Exception as exc:  # noqa: BLE001
            log.warning("status subscriber failed (%s)", exc)


def active() -> list[Incident]:
    """The currently-unresolved incidents, worst first. The caller must not mutate them."""
    with _lock:
        return sorted(_active.values(), key=lambda i: (_ORDER[i.severity], i.started))


def describe() -> str:
    """One line for the chat heartbeat, or "" when nothing is wrong right now."""
    worst = _worst_active()
    if worst is None:
        return ""
    return worst.describe()


def recent(limit: int = 50) -> list[dict]:
    with _lock:
        return [{"time": t, "severity": sev, "kind": k, "detail": d, "resolved": r}
                for t, sev, k, d, r in list(_history)[:limit]]


def reset() -> None:
    """For tests: forget every incident and subscriber."""
    with _lock:
        _active.clear()
        _history.clear()
        _subscribers.clear()


def snapshot() -> dict:
    """The whole picture, for /status.json. Cheap, read-only, no model calls."""
    incs = _worst_active()
    return {
        "incident": {"kind": incs.kind, "severity": incs.severity,
                     "detail": incs.detail, "occurrences": incs.occurrences}
        if incs else None,
        "active": [{"kind": i.kind, "severity": i.severity, "detail": i.detail,
                    "occurrences": i.occurrences} for i in active()],
        "recent": recent(50),
    }


# --------------------------------------------------------------------------------------
# Settings drift: .env edited under a running bridge is the bug that started this branch
# (the window was raised to 1M but the process kept the old 131k in memory). Env is read
# once at startup, so a change on disk is not effective until restart - and the user should
# be told rather than silently running with a stale window.
# --------------------------------------------------------------------------------------
_env_file: str | None = None
_env_mtime: float | None = None
_drift_checked = False


def set_env_file(path: str | None) -> None:
    """The resolved .env path, for drift detection. Set by __main__.py after argument parsing."""
    global _env_file
    with _lock:
        _env_file = path


def snapshot_settings() -> None:
    """Record .env's mtime at startup so later edits can be spotted."""
    global _env_mtime, _drift_checked
    with _lock:
        _drift_checked = False
        if not _env_file:
            return
        try:
            _env_mtime = os.stat(_env_file).st_mtime
        except OSError:
            _env_mtime = None


def check_settings_drift() -> None:
    """If .env changed on disk since startup, record a hard incident. Runs once; a series of
    edits does not re-alert forever, so a restart to apply the change also clears it."""
    global _drift_checked
    with _lock:
        if _drift_checked or not _env_file:
            return
        _drift_checked = True
        try:
            now = os.stat(_env_file).st_mtime
        except OSError:
            return
    if _env_mtime is not None and now > _env_mtime + 1.0:
        record("settings_drift",
               "The .env file changed after the bridge started. Settings are read once at "
               "startup, so the change is not active - restart the bridge to apply it.")
