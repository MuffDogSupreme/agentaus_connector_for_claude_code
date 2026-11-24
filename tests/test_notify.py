"""Unit tests for agentaus_bridge.notify - Windows toasts for upstream incidents.

Notification is the status consumer. "hard" incidents (quota, rejected key, turn
timeout, compaction fallback, settings drift) toast as soon as they start; "soft" ones
(429, no connection, slow upstream) toast only once they have lasted
AGENTAUS_NOTIFY_AFTER_S. Every kind is cooldowned so a flapping upstream cannot spam,
and a recovery toast fires only for a kind we actually toasted.

These tests drive the decision logic (`_on_incident`) with a controlled clock and mock
the toast launcher `_fire` - they never run the real PowerShell toast. The clock is
patched onto `notify.time`, and the `threading.Thread` a toast is dispatched to is
replaced with a synchronous stand-in, so every assertion is deterministic rather than
racing a daemon thread.
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentaus_bridge import notify  # noqa: E402
from agentaus_bridge import status  # noqa: E402


class _Clock:
    """A stand-in for the time module with a caller-controlled monotonic clock."""

    def __init__(self, now: float = 0.0):
        self.now = now

    def monotonic(self) -> float:
        return self.now


class _SyncThread:
    """Thread stand-in whose `start()` runs the target synchronously, so tests that
    mock `_fire` see every toast decision without racing a real daemon thread."""

    def __init__(self, *args, target=None, kwargs=None, daemon=None):
        self._target = target
        self._kwargs = kwargs or {}

    def start(self) -> None:
        if self._target is not None:
            self._target(**self._kwargs)


class NotifyTestBase(unittest.TestCase):
    def setUp(self):
        from agentaus_bridge.config import settings

        self.settings = settings
        self.clock = _Clock()
        # `status.reset()` forgets every incident and subscriber; the notify state is
        # cleared separately. The notify toggles are saved and restored per test.
        status.reset()
        notify._reset_state()
        self._saved = (
            settings.agentaus_notify,
            settings.agentaus_notify_after_s,
            settings.agentaus_notify_cooldown_s,
        )
        settings.agentaus_notify = "toast"
        settings.agentaus_notify_after_s = 30.0
        settings.agentaus_notify_cooldown_s = 300.0
        mock.patch("agentaus_bridge.notify.time", self.clock).start()
        self.addCleanup(mock.patch.stopall)
        # Route every toast through the synchronous stand-in for deterministic asserts.
        mock.patch("agentaus_bridge.notify.threading.Thread", _SyncThread).start()

    def tearDown(self):
        (self.settings.agentaus_notify,
         self.settings.agentaus_notify_after_s,
         self.settings.agentaus_notify_cooldown_s) = self._saved
        notify._reset_state()

    @staticmethod
    def incident(kind: str) -> status.Incident:
        return status.Incident(kind)


class TestWhenDisabled(NotifyTestBase):
    def test_disabled_notification_never_fires_a_toast(self):
        self.settings.agentaus_notify = "off"
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("turn_timeout"), False)
        fire.assert_not_called()


class TestHardIncidentsToastImmediately(NotifyTestBase):
    def test_a_hard_incident_toasts_straight_away(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("turn_timeout"), False)
        fire.assert_called_once_with(
            title="Agentaus bridge: turn timeout",
            body=self.incident("turn_timeout").describe(),
        )

    def test_a_hard_incident_ignores_the_soft_delay(self):
        self.settings.agentaus_notify_after_s = 10_000.0
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("quota"), False)
        fire.assert_called_once()


class TestSoftIncidentsWait(NotifyTestBase):
    def test_a_brief_429_stays_silent(self):
        """The first sighting records the run's start; it cannot already be mature."""
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("rate_limited"), False)
        fire.assert_not_called()

    def test_a_soft_incident_toasts_once_mature(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("rate_limited"), False)  # t=0: start
            self.clock.now = 45.0
            notify._on_incident(self.incident("rate_limited"), False)  # t=45: 45s in
        fire.assert_called_once()

    def test_a_soft_incident_is_still_silent_just_under_the_threshold(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("slow_upstream"), False)  # t=0: start
            self.clock.now = 29.9
            notify._on_incident(self.incident("slow_upstream"), False)  # 29.9s < 30
        fire.assert_not_called()


class TestRecoveryToast(NotifyTestBase):
    def test_no_recovery_toast_when_nothing_was_toasted(self):
        """A soft 429 that clears before 30s stays entirely silent - no start, no recovery."""
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("rate_limited"), False)  # t=0: not mature
            notify._on_incident(self.incident("rate_limited"), True)   # clears
        fire.assert_not_called()

    def test_a_recovered_toasted_incident_reports_recovery(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("rate_limited"), False)  # t=0: start
            self.clock.now = 45.0
            notify._on_incident(self.incident("rate_limited"), False)  # t=45: toast (1)
            self.clock.now = 50.0
            notify._on_incident(self.incident("rate_limited"), True)   # recovery (2)
        self.assertEqual(fire.call_count, 2)
        self.assertEqual(fire.call_args.kwargs["title"], "Agentaus bridge recovered")


class TestCooldown(NotifyTestBase):
    def test_a_flapping_hard_incident_does_not_spam(self):
        """A recurrence during the cooldown period yields exactly the start and recovery."""
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("quota"), False)       # t=0: toast (1)
            self.clock.now = 10.0
            notify._on_incident(self.incident("quota"), True)        # recovery (2)
            self.clock.now = 20.0
            notify._on_incident(self.incident("quota"), False)       # back within cooldown
        self.assertEqual(fire.call_count, 2, "a flapping upstream spammed toasts")

    def test_last_toast_ok_again_after_the_cooldown(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("turn_timeout"), False)  # t=0: toast (1)
            self.clock.now = 305.0
            notify._on_incident(self.incident("turn_timeout"), False)  # new run, past 300s
        self.assertEqual(fire.call_count, 2)

    def test_recovery_toast_respects_its_own_cooldown(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify._on_incident(self.incident("quota"), False)       # t=0: toast (1)
            self.clock.now = 10.0
            notify._on_incident(self.incident("quota"), True)        # recovery (2)
            self.clock.now = 605.0
            notify._on_incident(self.incident("quota"), True)        # past recovery cooldown
        self.assertEqual(fire.call_count, 3)


class TestThreadedHandoff(NotifyTestBase):
    def test_the_toast_is_launched_on_a_thread_not_the_event_loop(self):
        """`_on_incident` runs under status's lock; the toast must not run in it."""
        with mock.patch("agentaus_bridge.notify._fire") as fire, \
             mock.patch("agentaus_bridge.notify.threading.Thread") as thread:
            notify._on_incident(self.incident("quota"), False)
        thread.assert_called_once()
        self.assertEqual(thread.call_args.kwargs["target"], fire)
        thread.return_value.start.assert_called_once()
        self.assertEqual(fire.call_count, 0, "_fire ran synchronously on the calling thread")

    def test_a_soft_toast_is_deferred_to_a_thread_when_mature(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire, \
             mock.patch("agentaus_bridge.notify.threading.Thread") as thread:
            notify._on_incident(self.incident("rate_limited"), False)  # t=0: start
            self.clock.now = 45.0
            notify._on_incident(self.incident("rate_limited"), False)  # mature
        thread.assert_called_once()
        self.assertEqual(fire.call_count, 0)


class TestInstall(NotifyTestBase):
    def test_install_wires_status_records_to_toasts(self):
        with mock.patch("agentaus_bridge.notify._fire") as fire:
            notify.install()
            status.record("quota")
        fire.assert_called_once()


if __name__ == "__main__":
    unittest.main()
