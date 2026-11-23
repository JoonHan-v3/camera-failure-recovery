"""Independent re-implementation check of the custom evaluator's per-sample counts (plan: 'checked per-scene detection statistics').
For every sample of a split, recompute (tp, fp, ign, n_pred_region, n_eligible_gt) with a deliberately different implementation:
  * geometry through the nuScenes devkit / pyquaternion (not gt_tables.quat_to_rot): global boxes, ego pose at LIDAR_TOP time, ego pose at
    CAM_FRONT time, calibrated sensor, devkit `view_points`;
  * MATCHING PLANE (geometry v2): both centres are derived INDEPENDENTLY in the sample-time EGO frame from the devkit records
    (GT: global -> ego via the LIDAR_TOP-keyframe ego pose; prediction: LiDAR -> ego via the calibrated sensor, geometric centre z_bottom + h/2)
    and the 2 m distance is the XY norm there - NOT the LiDAR-frame XY used by the sensor-plane rule of geometry v1;
  * predictions are FILTERED by the threshold first and matched with plain python loops (no prefix trick, no vectorisation);
and compare with event_stats.sample_status + counts_at. Also verifies invariants: TP <= eligible GT, one-to-one GT use, matched GT eligible.
Usage: python -m outage.check_stats <gt.pkl> <preds.pkl> <tau.json> <out.json> [stride]"""
import json, os, pickle, sys
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.eval.detection.utils import category_to_detection_name
from .event_stats import sample_status
from .evaluator import counts_at, TP
from .gt_tables import CLASSES


def to_cam(nusc, sample, p_lidar):
    """LiDAR-frame point (sample time) -> CAM_FRONT frame, plus ground distance from the sample-time ego origin (devkit/pyquaternion only)."""
    sl = nusc.get("sample_data", sample["data"]["LIDAR_TOP"]); sc = nusc.get("sample_data", sample["data"]["CAM_FRONT"])
    cl, cc = nusc.get("calibrated_sensor", sl["calibrated_sensor_token"]), nusc.get("calibrated_sensor", sc["calibrated_sensor_token"])
    es, ec = nusc.get("ego_pose", sl["ego_pose_token"]), nusc.get("ego_pose", sc["ego_pose_token"])
    p = Quaternion(cl["rotation"]).rotate(p_lidar) + np.array(cl["translation"])              # ego (sample time)
    gd = float(np.hypot(p[0], p[1]))
    p = Quaternion(es["rotation"]).rotate(p) + np.array(es["translation"])                     # global
    p = Quaternion(ec["rotation"]).inverse.rotate(p - np.array(ec["translation"]))             # ego (camera time)
    p = Quaternion(cc["rotation"]).inverse.rotate(p - np.array(cc["translation"]))             # camera
    return p, np.array(cc["camera_intrinsic"]), gd


def lidar_to_ego_xy(nusc, sample, p_lidar):
    sl = nusc.get("sample_data", sample["data"]["LIDAR_TOP"]); cl = nusc.get("calibrated_sensor", sl["calibrated_sensor_token"])
    p = Quaternion(cl["rotation"]).rotate(np.asarray(p_lidar, float)) + np.array(cl["translation"])
    return p[:2]


def global_to_ego_xy(nusc, sample, p_global):
    sl = nusc.get("sample_data", sample["data"]["LIDAR_TOP"]); es = nusc.get("ego_pose", sl["ego_pose_token"])
    return Quaternion(es["rotation"]).inverse.rotate(np.asarray(p_global, float) - np.array(es["translation"]))[:2]


def in_region(nusc, sample, p_lidar):
    p, K, gd = to_cam(nusc, sample, p_lidar)
    if p[2] <= 1e-6:
        return False
    uv = view_points(p[:, None], K, normalize=True)[:2, 0]
    return bool(0 <= uv[0] < 1600 and 0 <= uv[1] < 900 and gd <= 50.0)


def main(gt_pkl, preds_pkl, tau_json, out, stride=1):
    gt = pickle.load(open(gt_pkl, "rb")); preds = pickle.load(open(preds_pkl, "rb")); tau = json.load(open(tau_json))["tau"]
    nusc = NuScenes("v1.0-trainval", os.path.expanduser("~/storage/nuscenes"), verbose=False)
    bad, n, inv_bad = [], 0, 0
    tot = dict(tp=0, fp=0, ign=0, n_pred_region=0, n_gt=0)
    extra = [t for t in os.environ.get("CHECK_STATS_EXTRA", "").split(",") if t in preds]          # e.g. the Inspector's boundary-crossing sample
    for st in sorted(set(sorted(preds)[::stride]) | set(extra)):      # stride > 1: evenly spaced subset of the sorted sample tokens
        s = nusc.get("sample", st); g = gt[st]["gt"]
        # independent GT: eligibility straight from the devkit records
        cls, cen, elig, cen_ego = [], [], [], []
        for at in g["ann"]:
            a = nusc.get("sample_annotation", at)
            name = category_to_detection_name(nusc.get("instance", a["instance_token"])["category_name"] if "category_name" in nusc.get("instance", a["instance_token"]) else a["category_name"])
            # centre in LiDAR frame via the devkit chain (global -> ego(sample time) -> lidar)
            sl = nusc.get("sample_data", s["data"]["LIDAR_TOP"]); es = nusc.get("ego_pose", sl["ego_pose_token"]); cl = nusc.get("calibrated_sensor", sl["calibrated_sensor_token"])
            p = Quaternion(es["rotation"]).inverse.rotate(np.array(a["translation"]) - np.array(es["translation"]))
            p = Quaternion(cl["rotation"]).inverse.rotate(p - np.array(cl["translation"]))
            cls.append(CLASSES.index(name)); cen.append(p); cen_ego.append(global_to_ego_xy(nusc, s, a["translation"]))
            elig.append(in_region(nusc, s, p) and a["num_lidar_pts"] + a["num_radar_pts"] >= 1)
        rows = np.asarray(preds[st], float).reshape(-1, 11)
        keep = [i for i in np.argsort(-rows[:, 10], kind="stable") if rows[i, 10] >= tau]       # filter FIRST, then match
        matched, tp, fp, ign, npr = set(), 0, 0, 0, 0
        for i in keep:
            r = rows[i]; c = int(round(r[9]))
            pxy = lidar_to_ego_xy(nusc, s, [r[0], r[1], r[2] + r[5] / 2.0])            # prediction geometric centre in the ego frame
            best, bd = None, 1e9
            for j in range(len(cls)):
                if cls[j] == c and elig[j] and j not in matched:
                    d = float(np.hypot(cen_ego[j][0] - pxy[0], cen_ego[j][1] - pxy[1]))
                    if d <= 2.0 and d < bd:
                        best, bd = j, d
            if best is not None:
                matched.add(best); tp += 1; npr += 1; continue
            if any(cls[j] == c and not elig[j] and float(np.hypot(cen_ego[j][0] - pxy[0], cen_ego[j][1] - pxy[1])) <= 2.0 for j in range(len(cls))):
                ign += 1; continue
            ctr = np.array([r[0], r[1], r[2] + r[5] / 2.0])
            if in_region(nusc, s, ctr):
                fp += 1; npr += 1
        ref = dict(tp=tp, fp=fp, ign=ign, n_pred_region=npr)
        ss = sample_status(preds[st], gt[st]); mine = counts_at(ss["score"], ss["status"], tau)
        n += 1
        for k in ref:
            tot[k] += ref[k]
        tot["n_gt"] += int(sum(elig))
        if ref != mine or sum(elig) != int(g["eligible"].sum()):
            bad.append(dict(sample=st, ref=ref, mine=mine, ref_gt=int(sum(elig)), mine_gt=int(g["eligible"].sum())))
        m_idx = [j for j, st_ in zip(ss["gt_idx"], ss["status"]) if st_ == TP and ss["score"][0] is not None]
        inv_bad += int(len(set(m_idx)) != len(m_idx) or any(not g["eligible"][j] for j in m_idx) or len(m_idx) > int(g["eligible"].sum()))
    rep = dict(n_samples=n, stride=stride, extra_samples_forced=extra, tau=tau, totals_reference=tot, mismatching_samples=len(bad), invariant_violations=inv_bad, examples=bad[:5], PASS=bool(not bad and not inv_bad))
    json.dump(rep, open(out, "w"), indent=1); print({k: v for k, v in rep.items() if k != "examples"})


if __name__ == "__main__":
    main(*sys.argv[1:5], *[int(x) for x in sys.argv[5:6]])
