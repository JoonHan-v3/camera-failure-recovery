"""Controlled examples for the front-region evaluator (study plan p.4 'Evaluator correctness'): projection, matches, ignored GT,
empty bins, threshold selection, recovery confirmation. Synthetic, numpy only."""
import unittest
import numpy as np
from outage.evaluator import (match_sample, counts_at, select_threshold, recovery_onset, damaged, assign_bins, TP, FP, IGN, OUT,
                              recall, precision, BINS_S)
from outage.gt_tables import project_front, quat_to_rot, make_T, inv_T, apply_T, ground_xy
from outage.audit import InputAudit
from outage.schedule import SceneOutage, US


def P(x, y, cls=0, score=0.9, z=0.0, h=1.5):
    return [x, y, z, 2.0, 4.0, h, 0.0, 0.0, 0.0, cls, score]


def gt(*items):
    """items: (x, y, cls, eligible) — x, y are already GROUND-PLANE coordinates (aligned synthetic frame: ego == LiDAR plane)."""
    return dict(gt_cls=[i[2] for i in items], gt_xy=[[i[0], i[1]] for i in items], gt_eligible=[i[3] for i in items])


def run(preds, g, in_region=None, pred_xy=None):
    preds = np.array(preds, float).reshape(-1, 11)
    reg = np.ones(len(preds), bool) if in_region is None else np.array(in_region, bool)
    pxy = preds[:, :2] if pred_xy is None else pred_xy
    order, st, gi = match_sample(preds, reg, pxy, **g)
    return preds[order, 10], st, gi, order


class Projection(unittest.TestCase):
    def setUp(self):   # ego = LiDAR frame = global; camera looks along +x of the ego frame (cam z = ego x, cam x = -ego y, cam y = -ego z)
        R = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]], float)  # ego -> camera rotation
        self.T = np.eye(4); self.T[:3, :3] = R
        self.K = np.array([[1000., 0, 800], [0, 1000., 450], [0, 0, 1]])
        self.ego = np.eye(4)

    def test_in_front_and_pixel(self):
        ok, uv, d, gd = project_front(self.T, self.K, self.ego, [[20.0, 0, 0]])
        self.assertTrue(ok[0]); np.testing.assert_allclose(uv[0], [800, 450]); self.assertAlmostEqual(d[0], 20.0)

    def test_behind_camera(self):
        self.assertFalse(project_front(self.T, self.K, self.ego, [[-10.0, 0, 0]])[0][0])

    def test_outside_image_sideways(self):     # camera x = -ego y: u = 800 + 1000*(-y)/x ; y=+25 at 20 m -> u = -450 (outside); y=+15 -> u = 50 (inside)
        self.assertFalse(project_front(self.T, self.K, self.ego, [[20.0, 25.0, 0]])[0][0])
        self.assertTrue(project_front(self.T, self.K, self.ego, [[20.0, 15.0, 0]])[0][0])
        self.assertFalse(project_front(self.T, self.K, self.ego, [[20.0, -25.0, 0]])[0][0])
        self.assertTrue(project_front(self.T, self.K, self.ego, [[20.0, 5.0, 0]])[0][0])    # u = 800 - 250 = 550

    def test_distance_limit_inclusive_and_beyond(self):
        self.assertTrue(project_front(self.T, self.K, self.ego, [[50.0, 0, 0]])[0][0])
        self.assertFalse(project_front(self.T, self.K, self.ego, [[50.5, 0, 0]])[0][0])

    def test_ground_distance_ignores_height(self):
        ok, _, _, gd = project_front(self.T, self.K, self.ego, [[30.0, 0, 5.0]])
        self.assertAlmostEqual(gd[0], 30.0)

    def test_vertical_bounds(self):
        self.assertFalse(project_front(self.T, self.K, self.ego, [[20.0, 0, 12.0]])[0][0])   # v = 450 - 600 < 0

    def test_empty(self):
        ok, uv, d, gd = project_front(self.T, self.K, self.ego, np.zeros((0, 3)))
        self.assertEqual(len(ok), 0)

    def test_transform_inverse_and_quaternion(self):
        q = [0.7071068, 0, 0, 0.7071068]    # +90 deg about z
        R = quat_to_rot(q); np.testing.assert_allclose(R @ [1, 0, 0], [0, 1, 0], atol=1e-6)
        T = make_T(q, [1, 2, 3]); np.testing.assert_allclose(inv_T(T) @ T, np.eye(4), atol=1e-9)
        np.testing.assert_allclose(apply_T(inv_T(T), apply_T(T, [[4, 5, 6]])), [[4, 5, 6]], atol=1e-9)

    def test_camera_time_ego_motion_changes_projection(self):
        # camera-time ego pose 1 m ahead of the sample-time pose -> a point 20 m ahead is 19 m from the camera, same pixel on axis
        T_l2cam = self.T @ inv_T(make_T([1, 0, 0, 0], [1.0, 0, 0]))
        _, uv, d, _ = project_front(T_l2cam, self.K, self.ego, [[20.0, 0, 0]])
        self.assertAlmostEqual(d[0], 19.0)


class Matching(unittest.TestCase):
    def test_one_to_one_two_preds_one_gt(self):
        s, st, gi, o = run([P(10, 0, score=0.9), P(10.5, 0, score=0.8)], gt((10, 0, 0, True)))
        self.assertEqual(list(st), [TP, FP])

    def test_descending_confidence_gets_the_gt(self):
        s, st, gi, o = run([P(10.5, 0, score=0.5), P(11.5, 0, score=0.95)], gt((10, 0, 0, True)))
        self.assertEqual(list(o), [1, 0]); self.assertEqual(list(st), [TP, FP])       # the higher score matched, although farther

    def test_nearest_unmatched_eligible(self):
        s, st, gi, o = run([P(10, 0, score=0.9)], gt((11, 0, 0, True), (10.2, 0, 0, True)))
        self.assertEqual(gi[0], 1)

    def test_tie_lowest_gt_index(self):
        s, st, gi, o = run([P(10, 0)], gt((11, 0, 0, True), (9, 0, 0, True)))
        self.assertEqual(gi[0], 0)

    def test_class_aware(self):
        s, st, gi, o = run([P(10, 0, cls=1)], gt((10, 0, 0, True)))
        self.assertEqual(st[0], FP)

    def test_distance_boundary_inclusive(self):
        self.assertEqual(run([P(12.0, 0)], gt((10, 0, 0, True)))[1][0], TP)
        self.assertEqual(run([P(12.01, 0)], gt((10, 0, 0, True)))[1][0], FP)

    def test_ignored_gt_absorbs_prediction(self):
        s, st, gi, o = run([P(10, 0)], gt((10.5, 0, 0, False)))          # same-class GT, not eligible -> ignored, not FP
        self.assertEqual(st[0], IGN)

    def test_ignored_other_class_does_not_absorb(self):
        self.assertEqual(run([P(10, 0, cls=1)], gt((10.5, 0, 0, False)))[1][0], FP)

    def test_eligible_matched_first_then_ignored_check(self):
        s, st, gi, o = run([P(10, 0, score=0.9), P(10.2, 0, score=0.8)], gt((10, 0, 0, True), (10.4, 0, 0, False)))
        self.assertEqual(list(st), [TP, IGN])                              # second pred: eligible GT taken, ignored GT absorbs

    def test_fp_only_when_prediction_in_region(self):
        self.assertEqual(run([P(10, 0)], gt(), in_region=[True])[1][0], FP)
        self.assertEqual(run([P(10, 0)], gt(), in_region=[False])[1][0], OUT)

    def test_empty_prediction_and_empty_gt(self):
        s, st, gi, o = run(np.zeros((0, 11)), gt((1, 1, 0, True)))
        self.assertEqual(len(st), 0)
        self.assertEqual(counts_at(s, st, 0.0), dict(tp=0, fp=0, ign=0, n_pred_region=0))

    def test_threshold_prefix_property(self):
        rng = np.random.RandomState(0)
        for _ in range(40):
            n, m = rng.randint(1, 12), rng.randint(0, 8)
            preds = [P(*rng.uniform(0, 20, 2), cls=int(rng.randint(0, 2)), score=float(rng.randint(1, 8)) / 8) for _ in range(n)]
            g = gt(*[(*rng.uniform(0, 20, 2), int(rng.randint(0, 2)), bool(rng.randint(0, 2))) for _ in range(m)])
            reg = rng.rand(n) > 0.3
            s, st, gi, o = run(preds, g, reg)
            for thr in (0.2, 0.5, 0.75):
                keep = [i for i in range(n) if preds[i][10] >= thr]
                if not keep:
                    continue
                s2, st2, _, _ = run([preds[i] for i in keep], g, [reg[i] for i in keep])
                self.assertEqual(counts_at(s, st, thr), counts_at(s2, st2, thr))

    def test_recall_precision_undefined(self):
        self.assertIsNone(recall(0, 0)); self.assertIsNone(precision(0, 0))
        self.assertEqual(recall(1, 4), 0.25); self.assertEqual(precision(3, 1), 0.75)


class GroundPlane(unittest.TestCase):
    """Geometry v2: matching distance is measured between GEOMETRIC centres in the sample-time EGO plane, not the tilted LiDAR XY plane."""
    def setUp(self):
        a = np.deg2rad(2.6)      # LiDAR pitched w.r.t. ego (the real calibration has ~2.6 deg): rotation about the y axis
        R = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        self.T = np.eye(4); self.T[:3, :3] = R; self.T[:3, 3] = [0.9, 0.0, 1.8]
        roll = np.deg2rad(1.0)   # plus a little roll
        Rr = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
        self.T[:3, :3] = R @ Rr

    def centres(self, gt_c, pred_c):
        return ground_xy(self.T, [gt_c]), ground_xy(self.T, [pred_c])

    # pitch about +y: ego_x offset = cos(a)*dx + sin(a)*dz, so a prediction centre LOWER than the GT centre shrinks the ego-plane offset
    # and a HIGHER one enlarges it (dz = +-1.5 m, a = 2.6 deg -> +-0.068 m); the sensor-plane XY ignores dz.
    def _dists(self, gc, pc):
        gxy, pxy = self.centres(gc, pc)
        return gxy, pxy, float(np.hypot(*(gxy[0] - pxy[0]))), float(np.hypot(*(gc[:2] - pc[:2])))

    def test_sensor_plane_says_fp_but_ego_plane_says_tp(self):
        gc, pc = np.array([10.0, 0.0, 2.4]), np.array([12.04, 0.0, 0.9])        # prediction lower than GT
        gxy, pxy, d_ego, d_lidar = self._dists(gc, pc)
        self.assertGreater(d_lidar, 2.0); self.assertLessEqual(d_ego, 2.0)
        row = P(*pc[:2], z=pc[2] - 0.75, h=1.5)
        self.assertEqual(run([row], dict(gt_cls=[0], gt_xy=gxy, gt_eligible=[True]), pred_xy=pxy)[1][0], TP)     # geometry v2
        self.assertEqual(run([row], gt((10.0, 0.0, 0, True)))[1][0], FP)                                         # old sensor-plane rule

    def test_sensor_plane_says_tp_but_ego_plane_says_fp(self):
        gc, pc = np.array([10.0, 0.0, 0.9]), np.array([11.97, 0.0, 2.4])        # prediction higher than GT
        gxy, pxy, d_ego, d_lidar = self._dists(gc, pc)
        self.assertLessEqual(d_lidar, 2.0); self.assertGreater(d_ego, 2.0)
        self.assertEqual(run([P(*pc[:2])], dict(gt_cls=[0], gt_xy=gxy, gt_eligible=[True]), pred_xy=pxy)[1][0], FP)
        self.assertEqual(run([P(*pc[:2])], gt((10.0, 0.0, 0, True)))[1][0], TP)

    def test_ignored_gt_distance_uses_ego_plane(self):
        gc, pc = np.array([10.0, 0.0, 2.4]), np.array([12.04, 0.0, 0.9])
        gxy, pxy, d_ego, d_lidar = self._dists(gc, pc)
        out = run([P(*pc[:2])], dict(gt_cls=[0], gt_xy=gxy, gt_eligible=[False]), pred_xy=pxy)[1][0]
        self.assertEqual(out, IGN)                                                          # absorbed: ego-plane distance <= 2 m
        self.assertEqual(run([P(*pc[:2])], gt((10.0, 0.0, 0, False)))[1][0], FP)             # sensor-plane XY would have left it as FP

    def test_ground_xy_is_the_ego_frame_xy(self):
        c = np.array([[3.0, -2.0, 1.0]])
        np.testing.assert_allclose(ground_xy(self.T, c)[0], (c @ self.T[:3, :3].T + self.T[:3, 3])[0, :2])
        self.assertEqual(ground_xy(self.T, np.zeros((0, 3))).shape, (0, 2))


class Threshold(unittest.TestCase):
    def test_max_precision_meeting_recall(self):
        # scores desc: TP .9, TP .8, FP .7, TP .6, FP .5 ; 4 eligible GT
        sc = [.9, .8, .7, .6, .5]; st = [TP, TP, FP, TP, FP]
        r = select_threshold(sc, st, 4, recall_target=0.5, min_pred=1)
        self.assertEqual(r["tau"], .8); self.assertEqual(r["precision"], 1.0); self.assertEqual(r["recall"], .5)

    def test_recall_target_forces_lower_threshold(self):
        sc = [.9, .8, .7, .6, .5]; st = [TP, TP, FP, TP, FP]
        r = select_threshold(sc, st, 4, recall_target=0.75, min_pred=1)
        self.assertEqual(r["tau"], .6); self.assertAlmostEqual(r["precision"], 0.75)

    def test_min_pred_count(self):
        sc = [.9, .8, .7, .6, .5]; st = [TP, TP, FP, TP, FP]
        r = select_threshold(sc, st, 4, recall_target=0.25, min_pred=4)
        self.assertEqual(r["tau"], .6)                                     # only >=4 predictions allowed

    def test_infeasible(self):
        r = select_threshold([.9, .8], [TP, FP], 10, recall_target=0.5, min_pred=1)
        self.assertIsNone(r["tau"])

    def test_zero_prediction_precision_cannot_win(self):
        self.assertIsNone(select_threshold([.9], [OUT], 1, 0.0, 0)["tau"])
        self.assertIsNone(select_threshold([], [], 3)["tau"]); self.assertIsNone(select_threshold([.5], [TP], 0)["tau"])

    def test_tie_group_and_tiebreak(self):
        # equal precision 1.0 at tau .9 (recall .25) and tau .8 (recall .5): the higher recall wins
        r = select_threshold([.9, .8], [TP, TP], 4, recall_target=0.1, min_pred=1)
        self.assertEqual(r["tau"], .8)
        # tied scores form one group: tau=.5 includes both rows (TP and FP)
        r = select_threshold([.5, .5], [TP, FP], 2, recall_target=0.1, min_pred=1)
        self.assertEqual(r["tau"], .5); self.assertEqual(r["n_pred"], 2)

    def test_ignored_not_counted_in_precision(self):
        r = select_threshold([.9, .8], [TP, IGN], 2, recall_target=0.1, min_pred=1)
        self.assertEqual(r["tau"], .8)         # ign rows add nothing: .9 and .8 tie on precision 1.0 and recall 0.5 -> lower tau
        self.assertEqual(r["precision"], 1.0)


class Recovery(unittest.TestCase):
    ok = {b: True for b in BINS_S}

    def test_first_bin_confirmed(self):
        self.assertEqual(recovery_onset({b: 0.0 for b in BINS_S}, self.ok), 0.0)

    def test_later_recovery(self):
        loss = {b: (0.3 if b < 1.0 else 0.0) for b in BINS_S}
        self.assertEqual(recovery_onset(loss, self.ok), 1.0)

    def test_single_good_bin_not_confirmed(self):
        loss = {b: 0.3 for b in BINS_S}; loss[1.0] = 0.0
        self.assertIsNone(recovery_onset(loss, self.ok))

    def test_margin_inclusive(self):
        loss = {b: 0.05 for b in BINS_S}
        self.assertEqual(recovery_onset(loss, self.ok), 0.0)
        loss[0.5] = 0.0501
        self.assertEqual(recovery_onset(loss, self.ok), 1.0)

    def test_missing_bin_breaks_pair_later_pair_ok(self):
        loss = {b: 0.0 for b in BINS_S}; del loss[0.5]
        self.assertEqual(recovery_onset(loss, self.ok), 1.0)                # 0.0 cannot be confirmed by a missing 0.5

    def test_insufficient_bin_breaks_pair(self):
        ok = dict(self.ok); ok[0.5] = False
        self.assertEqual(recovery_onset({b: 0.0 for b in BINS_S}, ok), 1.0)

    def test_last_bin_cannot_be_onset(self):
        loss = {b: 0.3 for b in BINS_S}; loss[4.5] = 0.0
        self.assertIsNone(recovery_onset(loss, self.ok))
        loss[4.0] = 0.0
        self.assertEqual(recovery_onset(loss, self.ok), 4.0)

    def test_none_loss_is_not_recovery(self):
        loss = {b: None for b in BINS_S}
        self.assertIsNone(recovery_onset(loss, self.ok))

    def test_adjacency_not_next_available(self):
        loss = {b: 0.3 for b in BINS_S}; loss[1.0] = 0.0; loss[2.0] = 0.0; del loss[1.5]
        self.assertIsNone(recovery_onset(loss, self.ok))                    # 1.0 and 2.0 are not adjacent scheduled bins

    def test_damage_rules(self):
        self.assertIsNone(damaged(0.4, 3, 5)); self.assertIsNone(damaged(None, 10, 5))
        self.assertTrue(damaged(0.0501, 5, 5)); self.assertFalse(damaged(0.05, 5, 5)); self.assertFalse(damaged(-0.2, 9, 5))


class Bins(unittest.TestCase):
    def test_assign_matches_audit_convention(self):
        rng = np.random.RandomState(1)
        o = SceneOutage("s", 10 * US, 6, 2 * US, round(4.5 * US))
        for _ in range(30):
            ts = np.sort(rng.uniform(0, 5.2, rng.randint(0, 12)))
            audit = InputAudit("m", "outage", 1.0, {"s": o})
            rest = o.restoration_us(1.0)
            for i, t in enumerate(ts):
                audit.log("s", f"x{i}", "CAM_FRONT", rest + int(t * US), True, f"CAM_FRONT/{i}.jpg", np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8), False, slot=0, eval_ts_us=rest + int(t * US))
            miss = audit.missing_bins().get("s", [])
            offs = sorted(audit.restoration_offsets().get("s", []), key=lambda x: x[1])
            got = assign_bins(offs)
            self.assertEqual([b for b in BINS_S if b not in got], miss)

    def test_one_sample_serves_one_bin(self):
        self.assertEqual(assign_bins([("a", 0.25)]), {0.0: "a"})


if __name__ == "__main__":
    unittest.main()
