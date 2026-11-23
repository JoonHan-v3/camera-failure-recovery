"""Controlled overlap examples for the same-object diagnostic (correction 3): a sample that is both the last 'during' sample (front exposure black) and bin 0.0
(evaluation time) must never be compared with itself; a bin-0.0 sample whose front image is still black is unavailable; a restored later sample is used."""
import unittest
import numpy as np
from recovery import samplelevel as sl

US = 1_000_000
K = {n: i for i, n in enumerate(sl.SO_KEYS)}


def sample(front_s, lidar_s, inst, eligible=None):
    n = len(inst)
    return dict(ts_front_us=int(front_s * US), ts_lidar_us=int(lidar_s * US), gt=dict(instance=np.array(inst), eligible=np.ones(n, bool) if eligible is None else eligible))


def rec(matched):
    return dict(matched_instance=list(matched))


class SameObject(unittest.TestCase):
    def run_case(self, samples, labels, t1_s, matched_clean, matched_out):
        gt = {k: v for k, v in samples.items()}
        sc = "scene"
        ph = {sc: labels}
        clean = {k: rec(matched_clean.get(k, [])) for k in gt}; out = {k: rec(matched_out.get(k, [])) for k in gt}
        detail = []
        X = sl.sameobject([sc], {sc: 0}, ph, gt, clean, out, {sc: int(t1_s * US)}, detail)
        return X[0], detail

    def test_same_sample_is_unavailable_not_a_persisting_miss(self):
        # sample 'a': front black at 9.9 s (< t1 = 10.0) but lidar time 10.05 s -> BOTH 'during' and 'b0.0'. Object 7 is missed in outage and clean-detected.
        samples = {"a": sample(9.9, 10.05, [7, 8])}
        labels = {"a": ["during", "b0.0"]}
        x, det = self.run_case(samples, labels, 10.0, {"a": [7, 8]}, {"a": [8]})
        self.assertEqual(x[K["events_unavailable_same_sample_as_bin0"]], 1)
        self.assertEqual(x[K["events_unavailable_no_restored_later_sample"]], 1)
        self.assertEqual(x[K["cont_missed_d_missed_after"]], 0)                      # the old code counted object 7 here (comparison with itself)
        self.assertEqual(x[K["events_with_last_during_and_later_restored"]], 0); self.assertIsNone(det[0]["after"])

    def test_bin0_with_black_front_is_unavailable_and_later_restored_sample_is_used(self):
        samples = {"a": sample(9.7, 9.8, [1, 2]), "b": sample(9.95, 10.1, [1, 2]), "c": sample(10.45, 10.6, [1, 2, 3])}
        labels = {"a": ["during"], "b": ["b0.0"], "c": ["b0.5"]}                      # b: front exposure 9.95 < 10.0 -> still black, so not a recovery observation
        x, det = self.run_case(samples, labels, 10.0, {"a": [1, 2], "b": [1, 2], "c": [1, 2, 3]}, {"a": [2], "b": [2], "c": [1, 2, 3]})
        self.assertEqual(x[K["events_unavailable_bin0_front_still_black"]], 1); self.assertEqual(x[K["events_unavailable_same_sample_as_bin0"]], 0)
        self.assertEqual(det[0]["after"], "c"); self.assertEqual(det[0]["after_bin"], 0.5)
        self.assertEqual(x[K["events_after_is_later_bin"]], 1); self.assertEqual(x[K["sum_after_bin_tenths_s"]], 5)
        self.assertGreater(det[0]["lidar_separation_s"], 0)
        self.assertEqual(x[K["cont_missed_d_detected_after"]], 1)                      # object 1: missed during, detected after
        self.assertEqual(x[K["cont_missed_d_missed_after"]], 0)

    def test_normal_restored_bin0(self):
        samples = {"a": sample(9.7, 9.8, [1]), "b": sample(10.2, 10.3, [1])}
        x, det = self.run_case(samples, {"a": ["during"], "b": ["b0.0"]}, 10.0, {"a": [1], "b": [1]}, {"a": [], "b": []})
        self.assertEqual(x[K["events_after_is_bin0"]], 1); self.assertEqual(x[K["cont_missed_d_missed_after"]], 1)   # genuine persisting miss on two distinct samples


if __name__ == "__main__":
    unittest.main()
