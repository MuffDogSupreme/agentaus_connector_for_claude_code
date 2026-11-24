"""Unit tests for agentaus_bridge.status - the single source of truth for what is wrong.

The bridge reacts to a struggling upstream in many places - retries, adaptive limits,
compaction timeouts - and each reports one line into `status`, which consumers (the
heartbeat text, the /status page, the Windows toast) read. This suite pins down which
incident kinds exist, how severity is assigned, how repeated reports deduplicate onto one
live incident, the bounded recent history, the recovery/clear path, subscriber
notifications, and the settings-drift detector.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentaus_bridge import status  # noqa: E402


class StatusTestBase(unittest.TestCase):
    def setUp(self):
        # `reset()` forgets every incident, subscriber and the history; the drift
        # detector state is cleared too so earlier tests cannot pin `_drift_checked`.
        status.reset()
        status.set_env_file(None)
        status.snapshot_settings()

    def kinds(self) -> list[str]:
        return [i.kind for i in status.active()]


class TestIncidentTracking(StatusTestBase):
    """Every kind a producer can report survives into `active()`/`describe()`."""

    def test_rate_limited_is_tracked_as_soft(self):
        status.record("rate_limited", "HTTP 429 - retrying with backoff")
        inc = status.active()[0]
        self.assertEqual(inc.kind, "rate_limited")
        self.assertEqual(inc.severity, "soft")
        self.assertIn("rate-limiting", inc.describe())
        self.assertIn("429", inc.describe())

    def test_no_connection_is_tracked_as_soft(self):
        status.record("no_connection", "network or DNS failure")
        self.assertEqual(status.active()[0].kind, "no_connection")
        self.assertEqual(status.active()[0].severity, "soft")

    def test_slow_upstream_is_tracked_as_soft(self):
        status.record("slow_upstream", "HTTP 524")
        self.assertEqual(status.active()[0].kind, "slow_upstream")
        self.assertEqual(status.active()[0].severity, "soft")

    def test_quota_is_tracked_as_hard(self):
        status.record("quota", "top up the Agentaus quota")
        inc = status.active()[0]
        self.assertEqual(inc.kind, "quota")
        self.assertEqual(inc.severity, "hard")

    def test_turn_timeout_is_tracked_as_hard(self):
        status.record("turn_timeout", "hit BRIDGE_TURN_TIMEOUT")
        self.assertEqual(status.active()[0].severity, "hard")

    def test_compacting_is_tracked_as_soft(self):
        status.record("compacting", "compressing the head")
        self.assertEqual(status.active()[0].severity, "soft")

    def test_settings_drift_is_tracked_as_hard(self):
        status.record("settings_drift", "the .env file changed")
        self.assertEqual(status.active()[0].severity, "hard")


class TestChildren(StatusTestBase):
    def test_unknown_incident_kinds_default_to_soft(self):
        status.record("some_other_kind")
        self.assertEqual(status.active()[0].severity, "soft")

    def test_repeated_reports_bump_occurrences_not_new_incidents(self):
        for _ in range(4):
            status.record("rate_limited", "HTTP 429")
        active = status.active()
        self.assertEqual(len(active), 1, "a 429 storm spawned one incident per 429")
        self.assertEqual(active[0].occurrences, 4)

    def test_a_repeated_report_refreshes_the_detail(self):
        status.record("slow_upstream", "HTTP 524")
        status.record("slow_upstream", "HTTP 504")
        self.assertIn("504", status.active()[0].detail)

    def test_unknown_kinds_describe_by_their_name(self):
        status.record("mystery", "extra")
        self.assertIn("mystery", status.active()[0].describe())

    def test_every_kind_resolves_independently(self):
        status.record("rate_limited")
        status.record("quota")
        status.resolve("quota")
        self.assertEqual(self.kinds(), ["rate_limited"])


class TestSeverityOrdering(StatusTestBase):
    def test_hard_incidents_sort_before_soft(self):
        status.record("rate_limited")     # soft, started first
        status.record("turn_timeout")     # hard, started later
        order = self.kinds()
        self.assertEqual(order, ["turn_timeout", "rate_limited"], "hard did not lead")

    def test_describe_reports_the_worst_active_incident(self):
        status.record("rate_limited")
        status.record("turn_timeout")
        self.assertIn("BRIDGE_TURN_TIMEOUT", status.describe())


class TestDescribeEmpty(StatusTestBase):
    def test_describe_is_empty_when_nothing_is_wrong(self):
        self.assertEqual(status.describe(), "")


class TestRecentHistory(StatusTestBase):
    def test_recent_is_newest_first(self):
        status.record("rate_limited", "first")
        status.record("quota", "second")
        recent = status.recent()
        self.assertEqual([r["kind"] for r in recent], ["quota", "rate_limited"])

    def test_recent_marks_a_resolved_incident(self):
        status.record("quota")
        status.resolve("quota")
        recent = status.recent()
        self.assertEqual(recent[0]["kind"], "quota")
        self.assertTrue(recent[0]["resolved"])

    def test_recent_is_bounded_by_limit(self):
        for i in range(60):
            status.record("rate_limited", f"n{i}")
        self.assertEqual(len(status.recent(10)), 10)

    def test_history_keeps_the_occurrence_detail_fresh(self):
        status.record("slow_upstream", "HTTP 524")
        status.record("slow_upstream", "HTTP 504")
        self.assertEqual(status.recent(1)[0]["detail"], "HTTP 504")


class TestResolution(StatusTestBase):
    def test_resolve_clears_the_live_incident(self):
        status.record("turn_timeout")
        status.resolve("turn_timeout")
        self.assertEqual(status.active(), [])

    def test_resolving_an_unknown_kind_is_a_noop(self):
        status.resolve("rate_limited")
        self.assertEqual(status.active(), [])

    def test_recovery_then_recurrence_starts_a_fresh_incident(self):
        status.record("rate_limited")
        status.resolve("rate_limited")
        status.record("rate_limited")
        self.assertEqual(status.active()[0].occurrences, 1,
                        "a recovered-and-recurred kind did not start a new run")


class TestSubscribers(StatusTestBase):
    def test_subscribers_hear_start_and_resolve(self):
        seen = []

        def waiter(incident, resolved):
            seen.append((incident.kind, resolved))

        status.subscribe(waiter)
        status.record("quota", "x")
        status.resolve("quota")
        self.assertIn(("quota", False), seen)
        self.assertIn(("quota", True), seen)

    def test_a_broken_subscriber_does_not_break_the_turn(self):
        def broken(incident, resolved):
            raise RuntimeError("consumer exploded")

        status.subscribe(broken)
        status.record("quota")  # must not raise
        self.assertEqual(status.active()[0].kind, "quota")

    def test_subscribers_are_called_once_per_transition(self):
        seen = []
        status.subscribe(lambda i, r: seen.append(r))
        status.record("rate_limited")
        self.assertEqual(seen, [False])


class TestSnapshot(StatusTestBase):
    def test_snapshot_reports_nothing_when_healthy(self):
        snap = status.snapshot()
        self.assertIsNone(snap["incident"])
        self.assertEqual(snap["active"], [])

    def test_snapshot_lists_every_active_incident(self):
        status.record("rate_limited")
        status.record("turn_timeout")
        snap = status.snapshot()
        self.assertEqual(len(snap["active"]), 2)
        self.assertEqual(snap["incident"]["kind"], "turn_timeout")


class TestSettingsDrift(StatusTestBase):
    """The .env-drift detector: one hard incident, recorded once per run."""

    def _env(self) -> str:
        fd, path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.unlink, path)
        return path

    def test_no_env_file_is_safe(self):
        status.set_env_file(None)
        status.snapshot_settings()
        status.check_settings_drift()
        self.assertEqual(status.active(), [])

    def test_no_drift_when_the_env_is_unchanged(self):
        path = self._env()
        status.set_env_file(path)
        status.snapshot_settings()
        base = os.stat(path).st_mtime
        os.utime(path, (base + 0.5, base + 0.5))  # under the 1.0s guard
        status.check_settings_drift()
        self.assertEqual(status.active(), [])

    def test_a_changed_env_records_one_hard_incident(self):
        path = self._env()
        status.set_env_file(path)
        status.snapshot_settings()
        base = os.stat(path).st_mtime
        os.utime(path, (base + 5.0, base + 5.0))
        status.check_settings_drift()
        self.assertEqual(status.active()[0].kind, "settings_drift")
        self.assertEqual(status.active()[0].severity, "hard")

    def test_drift_is_reported_once_per_run(self):
        path = self._env()
        status.set_env_file(path)
        status.snapshot_settings()
        base = os.stat(path).st_mtime
        os.utime(path, (base + 5.0, base + 5.0))
        status.check_settings_drift()
        status.resolve("settings_drift")
        os.utime(path, (base + 9.0, base + 9.0))
        status.check_settings_drift()  # already checked this run - no re-alert
        self.assertEqual(status.active(), [])


if __name__ == "__main__":
    unittest.main()
