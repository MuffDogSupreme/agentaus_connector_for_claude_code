"""Windows desktop toasts for upstream incidents - the consumer of agentaus_bridge.status.

Alerts when the bridge is having a bad time, so the user hears about it even when they are
not watching the chat. Two levels, matching status.SEVERITY:

  * "hard" incidents (quota 402, rejected key, turn timeout, compaction fallback, settings
    drift) toast as soon as they start;
  * "soft" incidents (429, no connection, slow upstream) toast only once they have lasted
    AGENTAUS_NOTIFY_AFTER_S seconds, so a blip that clears on its own stays silent.

Every kind is cooldowned (AGENTAUS_NOTIFY_COOLDOWN_S) so a flapping upstream cannot spam,
and a "recovered" toast fires when a kind we actually toasted resolves.

Deliberately hands off to a worker thread: the README's rule is that nothing may block the
event loop, and a toast that hangs (PowerShell blocked, session weirdness) must not cost a
turn. A notify failure is logged and swallowed - it must never break the request.

Off by default (AGENTAUS_NOTIFY=off). Set AGENTAUS_NOTIFY=toast to turn on. Notification is
best effort on top of a working bridge; it never acquires a position in the request path.
"""

from __future__ import annotations

import logging
import subprocess
import textwrap
import threading
import time

from .config import settings
from . import status

log = logging.getLogger("agentaus-bridge")

# Kinds that hurt a turn or the session - toast immediately on start.
_HARD = {"quota", "key_rejected", "turn_timeout", "compaction_fallback", "settings_drift"}

# kind -> monotonic time the current (unresolved) run of that kind first started.
_first_seen: dict[str, float] = {}
# kind -> monotonic time the last toast (or a recovery toast) was fired for it.
_last_toast: dict[str, float] = {}
_state_lock = threading.Lock()


def _reset_state() -> None:
    with _state_lock:
        _first_seen.clear()
        _last_toast.clear()


def _record_start(kind: str) -> None:
    with _state_lock:
        _first_seen.setdefault(kind, time.monotonic())


def _mature(kind: str) -> bool:
    """A soft incident has lasted long enough to toast about."""
    window = settings.agentaus_notify_after_s
    if window <= 0:
        return True
    with _state_lock:
        started = _first_seen.get(kind)
        return started is not None and time.monotonic() - started >= window


def _cooldown_ok(kind: str) -> bool:
    with _state_lock:
        last = _last_toast.get(kind)
        return last is None or time.monotonic() - last >= settings.agentaus_notify_cooldown_s


def _mark_toasted(kind: str) -> None:
    with _state_lock:
        _last_toast[kind] = time.monotonic()


def _fire(title: str, body: str) -> None:
    """Launch one Windows notification. Runs in a thread - never the event loop. Any failure
    is logged and swallowed; a toast that cannot happen must not cost a turn."""
    body = textwrap.shorten(body.replace("\n", " "), width=220, placeholder="...")
    # WinRT toast through PowerShell. Single-quoted PS string; our payload is short text so
    # doubling any apostrophe keeps it literal. Precomputed so the f-string holds no backslashes.
    # Best effort - if the machine has no toast delivery the call just no-ops or fails quietly.
    title_ps = title.replace("'", "''")
    body_ps = body.replace("'", "''")
    script = (
        f"$n = New-Object Windows.UI.Notifications.Toast; "
        f"$n.AppId = 'Agentaus bridge'; "
        f"$n.Title = '{title_ps}'; "
        f"$n.Body = '{body_ps}'; "
        f"$n.Expiration = 8000; $n.OnClicked = {{ exit }}; $n.Show()"
    )
    try:
        subprocess.Popen(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception as exc:  # noqa: BLE001 - a broken notifier must not break a turn
        log.warning("toast failed to launch (%s)", exc)


def _on_incident(incident, resolved: bool) -> None:
    """status.subscribe callback, run under status's lock. Must not block - hands off to a thread."""
    if settings.agentaus_notify != "toast":
        return
    kind = incident.kind

    if not resolved:
        # Record when this run of the kind started, then toast if it is hard or has lasted
        # the soft threshold, subject to the per-kind cooldown.
        _record_start(kind)
        if kind in _HARD or _mature(kind):
            if _cooldown_ok(kind):
                _mark_toasted(kind)
                title = f"Agentaus bridge: {kind.replace('_', ' ')}"
                threading.Thread(target=_fire,
                                 kwargs={"title": title, "body": incident.describe()},
                                 daemon=True).start()
        return

    # Recovery: only when this kind was actually toasted (so the user got the bad news),
    # and cooldowned on its own key so a quick down/up does not spam toasts.
    with _state_lock:
        was_toasted = kind in _last_toast
    if was_toasted and _cooldown_ok(f"{kind}:recovered"):
        _mark_toasted(f"{kind}:recovered")
        threading.Thread(target=_fire, kwargs={
            "title": "Agentaus bridge recovered",
            "body": f"The {kind} issue has cleared."}, daemon=True).start()


def install() -> None:
    """Register with status. No-op unless notifications are enabled."""
    if settings.agentaus_notify != "toast":
        return
    status.subscribe(_on_incident)
    log.info("Windows toasts enabled (after %.0fs for soft incidents, %.0fs cooldown)",
             settings.agentaus_notify_after_s, settings.agentaus_notify_cooldown_s)
