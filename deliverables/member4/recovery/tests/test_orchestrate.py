"""Negative controls for the orchestration integrity gate (no GPU, no inference): correction 4."""
import unittest
from unittest import mock
from recovery import orchestrate as O


def manifest(all_ok=True, same=True):
    runs = [dict(model=m, split="val", name=n, integrity_ok=all_ok) for m in O.MODELS for n in ("a", "d05", "d10", "d20")]
    return dict(all_runs_ok=all_ok, val_schedule_identical_across_models=same, runs=runs)


class Gate(unittest.TestCase):
    def test_failed_runs_stop_before_any_analysis(self):
        calls = []
        with self.assertRaises(SystemExit) as c:
            O.all_(status_fn=lambda: manifest(all_ok=False), runner=lambda s: calls.append(s))
        self.assertNotEqual(c.exception.code, 0); self.assertEqual(calls, [])

    def test_schedule_mismatch_stops_before_any_analysis(self):
        calls = []
        with self.assertRaises(SystemExit):
            O.all_(status_fn=lambda: manifest(same=False), runner=lambda s: calls.append(s))
        self.assertEqual(calls, [])

    def test_both_false_and_missing_required_run(self):
        calls = []
        with self.assertRaises(SystemExit):
            O.all_(status_fn=lambda: manifest(False, False), runner=lambda s: calls.append(s))
        m = manifest(); m["runs"] = [r for r in m["runs"] if not (r["model"] == "sparsebev" and r["name"] == "d20")]
        with self.assertRaises(SystemExit):
            O.integrity_gate(m)
        self.assertEqual(calls, [])

    def test_good_manifest_runs_all_four_steps(self):
        calls = []
        O.all_(status_fn=lambda: manifest(), runner=lambda s: calls.append(s))
        self.assertEqual(len(calls), 4)


class Execute(unittest.TestCase):
    def test_invalid_existing_run_is_a_problem_not_completion(self):
        with mock.patch("os.path.isdir", return_value=True), mock.patch("os.path.isfile", return_value=True), \
                mock.patch.object(O, "inspect_run", return_value=dict(integrity_ok=False, checks=dict(audit_consistent=False))):
            r = O.execute(run=False)
        self.assertEqual(r["todo"], []); self.assertEqual(len(r["problems"]), 12)

    def test_subprocess_failure_is_propagated(self):
        class P_:
            returncode = 3
            def poll(self): return 3
            def wait(self): return 3
        with mock.patch("os.path.isdir", return_value=False):
            r = O.execute(run=True, runner=lambda cmd: P_())
        self.assertEqual(len(r["failed"]), 12)


class Pinned(unittest.TestCase):
    def test_pins_are_full_sha256(self):
        for m, c in O.MODELS.items():
            self.assertIsNotNone(c["expected_ckpt"], m); self.assertEqual(len(c["expected_ckpt"]), 64)

    def test_pin_check_is_real_for_every_model_and_fails_on_mismatch(self):
        for m in O.MODELS:
            r = O.inspect_run(m, "pilot", "clean", "a")
            self.assertTrue(r["checks"]["checkpoint_matches_pinned_record"], m)
        with mock.patch.dict(O.MODELS["streampetr"], {"expected_ckpt": "0" * 64}):
            self.assertFalse(O.inspect_run("streampetr", "pilot", "clean", "a")["checks"]["checkpoint_matches_pinned_record"])
        with mock.patch.dict(O.MODELS["bevformer"], {"expected_ckpt": "7305046d"}):                  # a prefix is no longer accepted
            self.assertFalse(O.inspect_run("bevformer", "pilot", "clean", "a")["checks"]["checkpoint_matches_pinned_record"])


if __name__ == "__main__":
    unittest.main()
