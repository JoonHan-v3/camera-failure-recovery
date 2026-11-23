"""SparseBEV adapter tests (uint8 raw images, float-second timestamps -> ts_mode='abs_s'), synthetic, numpy only."""
import copy, unittest
import numpy as np
from outage.adapter import CorruptFrontCamera
from outage.schedule import SceneOutage, US

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
T0 = 1_533_153_000_123_456          # us, realistic magnitude (1.5e15) so float-second rounding is exercised


def results(ts_us, scene="s"):
    rng = np.random.RandomState(0)
    return dict(img=[rng.randint(1, 255, (9, 16, 3)).astype(np.uint8) for _ in CAMS],
                img_filename=[f"data/nuscenes/samples/{c}/x__{c}__{ts_us}.jpg" for c in CAMS],
                img_timestamp=[ts_us / 1e6] * 6, scene_token=scene, timestamp=ts_us / 1e6, eval_ts_us=ts_us, sample_idx="smp")


def step(d):
    return CorruptFrontCamera({"s": SceneOutage("s", T0, 6, 2 * US, round(4.5 * US))}, d, ts_mode="abs_s")


class SparseBEVAdapter(unittest.TestCase):
    def test_half_open_window_exact_microseconds(self):
        s = step(1.0)
        for ts, black in ((T0 - 1, False), (T0, True), (T0 + 999_999, True), (T0 + 1_000_000, False)):
            out = s(results(ts))
            self.assertEqual(not out["img"][0].any(), black, ts)

    def test_dtype_shape_preserved_uint8(self):
        out = step(1.0)(results(T0))
        self.assertEqual(out["img"][0].dtype, np.uint8); self.assertEqual(out["img"][0].shape, (9, 16, 3))

    def test_other_cameras_untouched_and_input_not_mutated(self):
        r = results(T0); before = copy.deepcopy(r["img"]); orig = list(r["img"])      # orig: the arrays the loader produced
        out = step(2.0)(r)
        for k in range(1, 6):
            np.testing.assert_array_equal(out["img"][k], before[k])
        np.testing.assert_array_equal(orig[0], before[0])               # the loader's array itself is not modified in place
        self.assertFalse(out["img"][0].any())

    def test_clean_run_never_corrupts(self):
        out = step(0.0)(results(T0 + 100))
        self.assertTrue(all(i.any() for i in out["img"]))

    def test_float_seconds_roundtrip_is_exact(self):
        for k in range(-3, 4):
            ts = T0 + k * 33_333 + 17
            self.assertEqual(round((ts / 1e6) * 1e6), ts)

    def test_malformed_inputs_rejected(self):
        r = results(T0); del r["scene_token"]
        with self.assertRaises(ValueError): step(1.0)(r)
        r = results(T0); r["img_timestamp"] = r["img_timestamp"][:5]
        with self.assertRaises(ValueError): step(1.0)(r)

    def test_unscheduled_scene_is_clean(self):
        out = step(2.0)(results(T0 + 10, scene="other"))
        self.assertTrue(all(i.any() for i in out["img"]))


if __name__ == "__main__":
    unittest.main()
