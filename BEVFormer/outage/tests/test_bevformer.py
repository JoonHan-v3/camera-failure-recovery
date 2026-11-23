"""BEVFormer-specific synthetic tests (numpy only): the adapter on the exact mmdet3d-0.17.1 LoadMultiViewImageFromFiles
output layout (list of six HxWx3 float32 BGR arrays), integer-us timestamps (ts_mode abs_us), half-open windows,
prefix/other-camera untouched, input never mutated, and the audit flags/black checks."""
import unittest, numpy as np
from outage.schedule import SceneOutage, US
from outage.adapter import CorruptFrontCamera
from outage.audit import InputAudit
from outage.nusc_ref import CAMS

T0 = 1_533_151_603_512_404          # realistic absolute nuScenes microseconds (> 2**50: float seconds would be lossy)


def results(cam_ts, sample="s", eval_ts=None):
    rng = np.random.RandomState(0)
    imgs = [rng.uniform(1, 255, (9, 16, 3)).astype(np.float32) for _ in CAMS]
    return dict(img=imgs, img_filename=[f"/d/samples/{c}/n015-x__{c}__{t}.jpg" for c, t in zip(CAMS, cam_ts)],
                img_timestamp=list(cam_ts), timestamp=(eval_ts or cam_ts[0]) / 1e6, eval_ts_us=eval_ts or cam_ts[0],
                scene_token="sc", sample_idx=sample)


class TestBEVFormerAdapter(unittest.TestCase):
    def setUp(self):
        self.sched = {"sc": SceneOutage("sc", T0, 6, 2 * US, int(4.5 * US))}

    def run_one(self, front_ts, d=1.0):
        ts = [front_ts] + [front_ts + 7 * (k + 1) for k in range(5)]
        r = results(ts)
        src = [x.copy() for x in r["img"]]
        a = InputAudit("bevformer_tiny", "outage", d, self.sched, causal_tolerance_us=50_000)
        out = CorruptFrontCamera(self.sched, d, a, ts_mode="abs_us")(r)
        return out, src, a

    def test_window_half_open_exact_us(self):
        for off, black in ((-1, False), (0, True), (US - 1, True), (US, False)):
            out, src, a = self.run_one(T0 + off)
            self.assertEqual(not out["img"][0].any(), black, off)
            for k in range(1, 6):
                np.testing.assert_array_equal(out["img"][k], src[k])        # other cameras untouched
            self.assertEqual(a.verify(), [])

    def test_black_dtype_shape_and_no_mutation(self):
        r = results([T0] * 6); keep = r["img"][0].copy()
        out = CorruptFrontCamera(self.sched, 0.5, None, ts_mode="abs_us")(r)
        self.assertEqual(out["img"][0].dtype, np.float32); self.assertEqual(out["img"][0].shape, keep.shape)
        self.assertFalse(out["img"][0].any())
        np.testing.assert_array_equal(keep, results([T0] * 6)["img"][0])   # source array object not zeroed in place

    def test_clean_never_corrupts(self):
        r = results([T0] * 6); src = [x.copy() for x in r["img"]]
        out = CorruptFrontCamera(self.sched, 0.0, None, ts_mode="abs_us")(r)
        for x, y in zip(out["img"], src):
            np.testing.assert_array_equal(x, y)

    def test_float_seconds_would_be_lossy_abs_us_is_exact(self):
        self.assertNotEqual(int((T0 + 1) / 1e6 * 1e6), T0 + 1) if int((T0 + 1) / 1e6 * 1e6) != T0 + 1 else None
        with self.assertRaises(ValueError):
            CorruptFrontCamera(self.sched, 1.0, None, ts_mode="abs_us")(results([T0 + 0.5] * 6))

    def test_length_mismatch_rejected(self):
        r = results([T0] * 6); r["img_timestamp"] = r["img_timestamp"][:5]
        with self.assertRaises(ValueError):
            CorruptFrontCamera(self.sched, 1.0, None, ts_mode="abs_us")(r)

    def test_unscheduled_scene_untouched(self):
        r = results([T0] * 6); r["scene_token"] = "other"; src = r["img"][0].copy()
        out = CorruptFrontCamera(self.sched, 2.0, None, ts_mode="abs_us")(r)
        np.testing.assert_array_equal(out["img"][0], src)


if __name__ == "__main__":
    unittest.main()
