import unittest, numpy as np
from outage.schedule import build_schedule, US
from outage.corruption import is_corrupted, maybe_corrupt
from outage.audit import InputAudit, compare_runs, key
from outage.state import SceneStateGuard, StateResetError, predictions_equal
from outage.adapter import CorruptFrontCamera, cam_from_path

def scene(n=40, t0=1_000_000_000):
    return [t0 + i * 500_000 for i in range(n)]

class Sched(unittest.TestCase):
    def test_rule_and_skips(self):
        s, sk = build_schedule({"a": scene(), "short": scene(6), "tight": scene(18)})
        self.assertEqual(s["a"].t_start_us, scene()[6])
        self.assertIn("short", sk); self.assertIn("tight", sk)
    def test_boundary_exact(self):
        # start idx 6, need 2.0+4.5=6.5s = 13 keyframes after -> last idx 19 is OK, 18 not
        s, _ = build_schedule({"x": scene(20)}); self.assertIn("x", s)
        s, _ = build_schedule({"x": scene(19)}); self.assertNotIn("x", s)
    def test_same_start_across_durations(self):
        s, _ = build_schedule({"a": scene()})
        starts = {s["a"].window(d)[0] for d in (0.5, 1, 2)}
        self.assertEqual(len(starts), 1)
    def test_bad_input(self):
        with self.assertRaises(ValueError): build_schedule({"a": [3, 2, 1]})

class Corr(unittest.TestCase):
    def test_half_open_and_sweeps(self):
        t = 10 * US
        self.assertTrue(is_corrupted("CAM_FRONT", t, t, 0.5))
        self.assertTrue(is_corrupted("CAM_FRONT", t + 499_999, t, 0.5))   # sweep inside
        self.assertFalse(is_corrupted("CAM_FRONT", t + 500_000, t, 0.5))
        self.assertFalse(is_corrupted("CAM_FRONT", t - 1, t, 2.0))
        self.assertFalse(is_corrupted("CAM_FRONT_LEFT", t, t, 2.0))
        self.assertFalse(is_corrupted("CAM_FRONT", t, t, 0.0))
    def test_no_mutation(self):
        im = np.full((4, 4, 3), 9, np.uint8)
        out, f = maybe_corrupt(im, "CAM_FRONT", 5, 5, 1.0)
        self.assertTrue(f and not out.any() and im.all())

def mk_results(scene_tok, base_s, front_offsets, shape=(4, 4, 3)):
    fn, ts, imgs = [], [], []
    for cam in ["CAM_FRONT", "CAM_BACK"]:
        for off in front_offsets:
            fn.append(f"samples/{cam}/n_{cam}_{off}.jpg"); ts.append(off); imgs.append(np.full(shape, 7, np.uint8))
    return dict(scene_token=scene_tok, timestamp=base_s, img_timestamp=ts, img_filename=fn, img=imgs, sample_idx=f"s{round(base_s * 1e6)}")

class Adapter(unittest.TestCase):
    def setUp(self):
        self.sched, _ = build_schedule({"a": scene()})
        self.o = self.sched["a"]
    def run_scene(self, dur, cond):
        audit = InputAudit("StreamPETR", cond, dur, self.sched)
        guard = SceneStateGuard()
        step = CorruptFrontCamera(self.sched, dur, audit, guard)
        for t in scene():
            step(mk_results("a", t / 1e6, [0.0, -0.05, -0.1]))
        return audit
    def test_outage_audit_passes(self):
        a = self.run_scene(1.0, "outage")
        self.assertEqual(a.verify(), [])
        n_bad = sum(r["corrupted"] for r in a.records)
        self.assertGreater(n_bad, 0)
        self.assertTrue(all(r["cam"] == "CAM_FRONT" for r in a.records if r["corrupted"]))
        self.assertTrue(all(r["all_black"] for r in a.records if r["corrupted"]))
    def test_prefix_unchanged_and_clean(self):
        clean, out = self.run_scene(0, "clean"), self.run_scene(2.0, "outage")
        self.assertEqual(clean.verify(), [])
        self.assertEqual(compare_runs(clean, out), [])
        self.assertIsNone(clean.last_corrupted_ts("a"))
    def test_last_corrupted_leaves_window(self):
        a = self.run_scene(0.5, "outage")
        self.assertLess(a.last_corrupted_ts("a"), self.o.t_start_us + 500_000)
    def test_audit_detects_tamper(self):
        a = self.run_scene(1.0, "outage")
        for r in a.records:
            if r["corrupted"]: r["corrupted"] = False; break
        self.assertTrue(a.verify())
    def test_cam_parse(self):
        self.assertEqual(cam_from_path("samples/CAM_FRONT_LEFT/x.jpg"), "CAM_FRONT_LEFT")
        with self.assertRaises(ValueError): cam_from_path("x.jpg")

class State(unittest.TestCase):
    def test_ok_and_reset(self):
        g = SceneStateGuard()
        self.assertTrue(g.step("a", 1, True)); self.assertFalse(g.step("a", 2, False)); self.assertTrue(g.step("b", 1, True))
    def test_stale_memory(self):
        g = SceneStateGuard(); g.step("a", 1, True)
        with self.assertRaises(StateResetError): g.step("b", 1, False)
    def test_interleave_and_backwards(self):
        g = SceneStateGuard(); g.step("a", 1, True); g.step("b", 1, True)
        with self.assertRaises(StateResetError): g.step("a", 2, True)
        g = SceneStateGuard(); g.step("a", 5, True)
        with self.assertRaises(StateResetError): g.step("a", 5, True)
    def test_repeatability(self):
        p = {"t": [[1, 2, .9]]}
        self.assertTrue(predictions_equal(p, {"t": [[1, 2, .9 + 1e-7]]}))
        self.assertFalse(predictions_equal(p, {"t": [[1, 2, .5]]}))


from outage.schedule import save_schedule, load_schedule
from outage.state import check_scene_isolation
import tempfile, os

class Regress(unittest.TestCase):
    def setUp(self):
        self.sched, _ = build_schedule({"a": scene()})
    def full(self, dur, cond, tamper=None):
        a = InputAudit("m", cond, dur, self.sched)
        step = CorruptFrontCamera(self.sched, dur, a) if cond != "clean" else None
        for t in scene():
            r = mk_results("a", t / 1e6, [0.0, -0.05])
            if step: step(r)
            else:
                for i, (im, fn, ts) in enumerate(zip(r["img"], r["img_filename"], r["img_timestamp"])):
                    a.log("a", f"s{t}", fn.split("/")[1], round((t/1e6+ts)*1e6), None, fn, im, im, False, slot=i)
        return a
    def test_empty_audit_fails(self):
        self.assertTrue(InputAudit("m", "clean", 0, self.sched).verify())
        self.assertTrue(compare_runs(self.full(0, "clean"), InputAudit("m", "outage", 1.0, self.sched)))
    def test_pair_ok_and_missing(self):
        c, o = self.full(0, "clean"), self.full(1.0, "outage")
        self.assertEqual(compare_runs(c, o), [])
        o.records.pop(0)
        self.assertTrue(any("clean only" in e for e in compare_runs(c, o)))
    def test_unflagged_modification(self):
        o = self.full(1.0, "outage")
        r = next(r for r in o.records if r["cam"] == "CAM_BACK"); r["sha1"] = "x"
        self.assertTrue(any("unflagged" in e for e in o.verify()))
    def test_manifest_and_unobserved(self):
        o = self.full(1.0, "outage")
        man = {(r["scene"], r["sample"], r["cam"], r["ts_us"], r["slot"]) for r in o.records}
        self.assertEqual(o.verify(man), [])
        self.assertTrue(o.verify(man | {("a", "s", "CAM_FRONT", 1, 9)}))
        self.assertTrue(o.verify(require_scenes=["a", "zz"]))
    def test_adapter_length_mismatch(self):
        r = mk_results("a", scene()[0] / 1e6, [0.0, -0.05]); r["img_filename"] = r["img_filename"][:3]
        with self.assertRaises(ValueError): CorruptFrontCamera(self.sched, 1.0)(r)
    def test_abs_us_mode(self):
        a = InputAudit("m", "outage", 1.0, self.sched)
        t0 = self.sched["a"].t_start_us
        r = dict(scene_token="a", timestamp=t0 / 1e6, img=[np.ones((2, 2, 3), np.uint8)] * 2,
                 img_filename=["x/CAM_FRONT/a.jpg", "x/CAM_FRONT/b.jpg"], img_timestamp=[t0 + 999_999, t0 + 1_000_000])
        CorruptFrontCamera(self.sched, 1.0, a, ts_mode="abs_us")(r)
        self.assertEqual([x["corrupted"] for x in a.records], [True, False])
    def test_schedule_validation_and_persist(self):
        with self.assertRaises(ValueError): build_schedule({"a": scene()}, durations_s=(0,))
        with self.assertRaises(ValueError): build_schedule({"a": scene()}, warmup_kf=-1)
        with self.assertRaises(TypeError): build_schedule({"a": [1.5, 2.5]})
        s, sk = build_schedule({"a": scene(), "b": scene(8)})
        p = os.path.join(tempfile.mkdtemp(), "s.json"); save_schedule(p, s, sk)
        s2, sk2, params = load_schedule(p)
        self.assertEqual(s, s2); self.assertEqual(sk, sk2); self.assertIn("b", sk2)
    def test_restoration_offsets(self):
        o = InputAudit("m", "outage", 1.0, self.sched); t0 = self.sched["a"].t_start_us
        z = np.ones((2, 2, 3), np.uint8)
        o.log("a", "k", "CAM_FRONT", t0 + 1_480_000, True, "f", z, z, False, eval_ts_us=t0 + 1_500_000)
        self.assertEqual(o.restoration_offsets()["a"], [("k", 0.5)])
    def test_predictions_rtol_and_order(self):
        self.assertFalse(predictions_equal({"t": [[1000.0, .9]]}, {"t": [[1000.001, .9]]}))
        self.assertTrue(predictions_equal({"t": [[1, .5], [2, .9]]}, {"t": [[2, .9], [1, .5]]}))
        self.assertFalse(predictions_equal({"t": [[1, .5]]}, {"u": [[1, .5]]}))
        self.assertFalse(predictions_equal({"t": [[1, .5]]}, {"t": [[1, .5], [1, .4]]}))
    def test_unknown_memory_rejected(self):
        with self.assertRaises(StateResetError): SceneStateGuard().step("a", 1)
        SceneStateGuard().step("a", 1, require_memory=False)
    def test_scene_isolation_toy_model(self):
        class M:  # fixed-size buffer + 'valid' flag; leaks across scenes if leak=True
            def __init__(s, leak): s.leak, s.mem, s.fresh = leak, 0.0, True
            def reset(s): s.mem, s.fresh = 0.0, True
            def run(s, sc):
                if not s.leak: s.reset()
                out = []
                for i in range(3):
                    s.mem = s.mem * .5 + sc; out.append([s.mem, 1.0]); s.fresh = False
                return {sc: out}
        for leak in (False, True):
            m = M(leak); res = check_scene_isolation(m.run, m.reset, lambda: m.fresh, 3.0, 5.0)
            self.assertTrue(res["repeatable"]); self.assertEqual(res["isolated"], not leak)


class Round3(unittest.TestCase):
    def setUp(self):
        self.sched, _ = build_schedule({"a": scene()})
        self.z = np.ones((2, 2, 3), np.uint8)
    def test_future_image_rejected(self):
        a = InputAudit("m", "clean", 0, self.sched, causal_tolerance_us=50_000)
        a.log("a", "s", "CAM_FRONT", 1000 * 10**6 + 10 * 10**6, None, "f", self.z, self.z, False, eval_ts_us=1000 * 10**6)
        self.assertTrue(any("future" in e for e in a.verify()))
    def test_causal_needs_eval_ts_and_status(self):
        a = InputAudit("m", "clean", 0, self.sched)
        a.log("a", "s", "CAM_FRONT", 5, None, "f", self.z, self.z, False)
        self.assertTrue(a.status().startswith("NOT VERIFIED"))
        a.causal_tolerance_us = 0
        self.assertTrue(a.verify())  # missing eval_ts
    def test_manifest_retained_and_saved(self):
        a = InputAudit("m", "clean", 0, self.sched)
        a.log("a", "s", "CAM_FRONT", 5, None, "f", self.z, self.z, False, eval_ts_us=5)
        a.set_manifest({("a", "s", "CAM_FRONT", 5, 0), ("a", "s", "CAM_BACK", 5, 0)}, "m1")
        self.assertEqual(a.status(), "FAIL")
        p = os.path.join(tempfile.mkdtemp(), "a.json"); a.save(p)
        import json
        with open(p) as fh: d = json.load(fh)
        self.assertEqual(d["status"], "FAIL"); self.assertTrue(d["violations"]); self.assertEqual(d["manifest_id"], "m1")
    def test_pass_requires_both(self):
        a = InputAudit("m", "clean", 0, self.sched, causal_tolerance_us=0)
        a.log("a", "s", "CAM_FRONT", 5, None, "f", self.z, self.z, False, eval_ts_us=5)
        a.set_manifest({("a", "s", "CAM_FRONT", 5, 0)}, "m1")
        self.assertEqual(a.status(), "CONSISTENT")
    def test_prefix_only_run_not_ready(self):
        a = InputAudit("m", "outage", 1.0, self.sched, causal_tolerance_us=0)
        t = self.sched["a"].t_start_us - 500_000
        a.log("a", "s", "CAM_FRONT", t, None, "f", self.z, self.z, False, eval_ts_us=t)
        a.set_manifest({("a", "s", "CAM_FRONT", t, 0)}, "m1")
        self.assertEqual(a.status(), "CONSISTENT")
        r = a.readiness()
        self.assertTrue(r.startswith("NOT READY")); self.assertIn("corrupted", r); self.assertIn("follow-up", r)
    def _valid_outage(self, offs, d=1.0):
        a = InputAudit("m", "outage", d, self.sched, causal_tolerance_us=0)
        o = self.sched["a"]; tr = o.restoration_us(d); meta = {}
        ev0 = o.t_start_us
        a.log("a", "in", "CAM_FRONT", ev0, None, "f", self.z, np.zeros_like(self.z), True, eval_ts_us=ev0)
        meta[("a", "in")] = ev0
        for off in offs:
            ev = tr + int(off * 1e6)
            a.log("a", f"s{off}", "CAM_FRONT", ev, None, "f", self.z, self.z, False, eval_ts_us=ev)
            meta[("a", f"s{off}")] = ev
        a.set_manifest({key(r) for r in a.records}, "m1"); a.set_sample_metadata(meta)
        return a
    def test_sparse_midpoint_bins(self):
        a = self._valid_outage((0.25, 1.25, 2.25, 3.25, 4.25))
        self.assertEqual(a.verify(), []); self.assertEqual(a.status(), "CONSISTENT")
        self.assertEqual(a.missing_bins()["a"], [0.5, 1.5, 2.5, 3.5, 4.5])
        r = a.readiness(); self.assertTrue(r.startswith("NOT READY")); self.assertIn("follow-up bins", r)
        self.assertEqual(self._valid_outage([i * 0.5 for i in range(10)]).readiness(), "READY")
    def test_save_preserves_manifest_and_metadata(self):
        import json
        a = self._valid_outage([i * 0.5 for i in range(10)])
        p = os.path.join(tempfile.mkdtemp(), "a.json"); a.save(p)
        with open(p) as fh: d = json.load(fh)
        self.assertEqual(len(d["expected_manifest"]), len(a.manifest))
        self.assertEqual(len(d["sample_metadata"]), len(a.sample_meta))
        self.assertEqual(d["hashes"], a.artifact_hashes()); self.assertTrue(d["hashes"]["manifest_sha1"])
        b = InputAudit("m", "clean", 0, self.sched); self.assertEqual(b.artifact_hashes()["manifest_sha1"], None)
    def test_duplicate_labels_same_timestamp_rejected(self):
        a = self._valid_outage((0.25, 1.25, 2.25, 3.25, 4.25))
        tr = self.sched["a"].restoration_us(1.0)
        for off in (0.25, 1.25, 2.25, 3.25, 4.25):
            ev = tr + int(off * 1e6)
            a.log("a", f"dup{off}", "CAM_FRONT", ev, None, "f", self.z, self.z, False, slot=1, eval_ts_us=ev)
        self.assertTrue(any("multiple samples" in e for e in a.verify()))
        self.assertEqual(a.status(), "FAIL"); self.assertTrue(a.readiness().startswith("NOT READY"))
    def test_conflicting_eval_ts_not_overwritten(self):
        a = self._valid_outage([i * 0.5 for i in range(10)])
        a.log("a", "s0.0", "CAM_BACK", 5, None, "f", self.z, self.z, False, eval_ts_us=5)
        self.assertTrue(any("conflicting eval" in e for e in a.verify()))
        self.assertEqual(a.eval_samples()[("a", "s0.0")], self.sched["a"].restoration_us(1.0))  # first kept
        self.assertTrue(a.readiness().startswith("NOT READY"))
    def test_missing_identity_and_metadata_mismatch(self):
        a = self._valid_outage([i * 0.5 for i in range(10)])
        a.sample_meta[("a", "s0.5")] += 1
        self.assertTrue(any("!= metadata" in e for e in a.verify()))
        b = self._valid_outage([i * 0.5 for i in range(10)]); b.sample_meta = None
        self.assertIn("no independent sample metadata", b.readiness())
        b.log("a", "", "CAM_BACK", 5, None, "f", self.z, self.z, False, eval_ts_us=5)
        self.assertTrue(any("missing sample identity" in e for e in b.verify()))
    def test_full_run_ready(self):
        o = self.sched["a"]; a = InputAudit("m", "outage", 1.0, self.sched, causal_tolerance_us=0)
        step = CorruptFrontCamera(self.sched, 1.0, a)
        for t in scene():
            r = mk_results("a", t / 1e6, [0.0, -0.05]); r["sample_idx"] = f"s{t}"
            step(r)
        a.set_manifest({(r["scene"], r["sample"], r["cam"], r["ts_us"], r["slot"]) for r in a.records}, "self")
        self.assertTrue(a.readiness().startswith("NOT READY"))   # no independent sample metadata
        a.set_sample_metadata({(r["scene"], r["sample"]): r["eval_ts_us"] for r in a.records})
        self.assertEqual(a.readiness(), "READY")
    def test_compare_rejects_duplicates_and_schedule_mismatch(self):
        c = InputAudit("m", "clean", 0, self.sched); o = InputAudit("m", "outage", 1.0, {})
        for x in (c, o, o):
            x.log("a", "s", "CAM_BACK", 5, None, "f", self.z, self.z, False)
        c.log("a", "s", "CAM_BACK", 5, None, "f", self.z, self.z, False)
        e = compare_runs(c, o)
        self.assertTrue(any("duplicate" in x for x in e)); self.assertTrue(any("schedule mismatch" in x for x in e))
    def test_numpy_bool_and_commit_after_validation(self):
        with self.assertRaises(StateResetError): SceneStateGuard().step("a", 1, np.bool_(False))
        self.assertTrue(SceneStateGuard().step("a", 1, np.bool_(True)))
        with self.assertRaises(TypeError): SceneStateGuard().step("a", 1, 0)
        g = SceneStateGuard()
        with self.assertRaises(StateResetError): g.step("a", 1, False)
        self.assertTrue(g.step("a", 1, True))  # corrected retry is not 'interleaving'
    def test_restoration_eval_once_and_bins(self):
        o = self.sched["a"]; d = 1.0
        a = InputAudit("m", "outage", d, self.sched)
        tr = o.restoration_us(d)
        for ev in [tr + int(i * 500_000) + 20_000 for i in range(10)]:
            for slot, cam_ts in enumerate([ev - 30_000, ev - 530_000]):   # async front ts + repeated history
                a.log("a", f"s{ev}", "CAM_FRONT", cam_ts, None, "f", self.z, self.z, False, slot=slot, eval_ts_us=ev)
        offs = a.restoration_offsets()["a"]
        self.assertEqual(len(offs), 10)                       # once per evaluated sample
        self.assertAlmostEqual(offs[0][1], 0.02)
        self.assertEqual(a.missing_bins(), {})
        a.records = [r for r in a.records if r["sample"] != f"s{tr + 4_500_000 + 20_000}"]
        self.assertEqual(a.missing_bins(), {"a": [4.5]})

if __name__ == "__main__":
    unittest.main()


class RealLoaderContract(unittest.TestCase):
    """Contracts observed on StreamPETR 95f6470 / nuScenes: absolute-second img_timestamp, camera sensor
    timestamps ~35 ms BEFORE the lidar sample timestamp, outage start defined on the front sensor ts."""
    def test_abs_s_mode_and_front_skew(self):
        t_front = 1533151603512404                       # CAM_FRONT keyframe sensor ts (us), real value
        sched, _ = build_schedule({"a": [t_front + i * 500_000 for i in range(40)]})
        o = sched["a"]; self.assertEqual(o.t_start_us, t_front + 6 * 500_000)
        a = InputAudit("m", "outage", 0.5, sched, causal_tolerance_us=0)
        step = CorruptFrontCamera(sched, 0.5, a, ts_mode="abs_s")
        z = np.full((4, 4, 3), 7, np.float32)
        ts = o.t_start_us
        res = dict(scene_token="a", sample_idx="s", timestamp=(ts + 36_000) / 1e6,        # lidar sample ts
                   img=[z.copy(), z.copy()], img_filename=["x/CAM_FRONT/f.jpg", "x/CAM_BACK/b.jpg"],
                   img_timestamp=[ts / 1e6, (ts - 20_000) / 1e6])
        out = step(res)
        self.assertFalse(out["img"][0].any()); self.assertTrue(out["img"][1].any())
        self.assertEqual([r["ts_us"] for r in a.records], [ts, ts - 20_000])           # float s -> exact us
        self.assertEqual(a.records[0]["eval_ts_us"], ts + 36_000)
        # a front image 1 us before the start stays clean (half-open, uses sensor ts not sample ts)
        res2 = dict(res, img=[z.copy(), z.copy()], img_timestamp=[(ts - 1) / 1e6, ts / 1e6])
        self.assertTrue(CorruptFrontCamera(sched, 0.5, None, ts_mode="abs_s")(res2)["img"][0].any())
    def test_audit_save_load_roundtrip(self):
        from outage.audit import load_audit
        sched, _ = build_schedule({"a": scene()})
        a = InputAudit("m", "clean", 0, sched, causal_tolerance_us=0)
        z = np.zeros((2, 2, 3), np.uint8)
        a.log("a", "s", "CAM_FRONT", 5, None, "f", z, z, False, eval_ts_us=5)
        a.set_manifest({("a", "s", "CAM_FRONT", 5, 0)}, "m1"); a.set_sample_metadata({("a", "s"): 5})
        p = os.path.join(tempfile.mkdtemp(), "a.json"); a.save(p)
        b = load_audit(p)
        self.assertEqual(b.status(), a.status()); self.assertEqual(b.verify(), a.verify())
        self.assertEqual(b.artifact_hashes(), a.artifact_hashes())


class Comparator(unittest.TestCase):
    def test_pred_diff_rejects_nonfinite_and_bad_schema(self):
        from outage.verify_runs import pred_diff
        ok = np.zeros((2, 11)); ok[:, 9] = [1, 2]
        self.assertEqual(pred_diff({"s": ok}, {"s": ok.copy()}), 0.0)
        bad = ok.copy(); bad[0, 0] = np.nan
        self.assertEqual(pred_diff({"s": bad}, {"s": ok}), float("inf"))
        self.assertEqual(pred_diff({"s": bad}, {"s": bad.copy()}), float("inf"))     # nan==nan must not pass
        self.assertEqual(pred_diff({"s": ok[:, :10]}, {"s": ok[:, :10]}), float("inf"))
        frac = ok.copy(); frac[0, 9] = 1.5
        self.assertEqual(pred_diff({"s": frac}, {"s": frac.copy()}), float("inf"))
        self.assertEqual(pred_diff({"s": ok}, {"t": ok}), float("inf"))
    def test_causal_policy_is_frozen_constant(self):
        from outage.audit import CAUSAL_TOL_US
        self.assertEqual(CAUSAL_TOL_US, 50_000)
        a = InputAudit("m", "clean", 0, {}, causal_tolerance_us=CAUSAL_TOL_US); z = np.zeros((2, 2, 3), np.uint8)
        a.log("a", "s", "CAM_BACK_LEFT", 1_000_049_992, None, "f", z, z, False, eval_ts_us=1_000_000_000)
        self.assertFalse(any("future" in e for e in a.verify()))
        a.log("a", "s2", "CAM_BACK_LEFT", 1_000_050_001, None, "f", z, z, False, eval_ts_us=1_000_000_000)
        self.assertTrue(any("future image" in e for e in a.verify()))
