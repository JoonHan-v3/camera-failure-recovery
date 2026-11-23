"""Independent check of gt_tables.py against the official nuScenes devkit: for sampled annotations, the CAM_FRONT-frame centre
(devkit: global -> ego(camera-time pose) -> camera, `get_sample_data`) and pixel must equal what gt_tables computed through the
LiDAR frame, and the eligibility flags must equal a direct re-derivation from the devkit objects.
Usage: python -m outage.check_projection_devkit <gt.pkl> <out.json> [n_samples]"""
import json, os, pickle, sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
from nuscenes.eval.detection.utils import category_to_detection_name
from .gt_tables import project_front, apply_T


def main(gt_pkl, out, n=150, seed=0):
    gt = pickle.load(open(gt_pkl, "rb"))
    nusc = NuScenes("v1.0-trainval", os.path.expanduser("~/storage/nuscenes"), verbose=False)
    rng = np.random.RandomState(seed)
    toks = sorted(gt)
    pick = [toks[i] for i in rng.choice(len(toks), min(n, len(toks)), replace=False)]
    worst_xyz, worst_uv, n_boxes, mism_region, mism_elig, n_elig = 0.0, 0.0, 0, 0, 0, 0
    for st in pick:
        s = nusc.get("sample", st)
        cam_sd = nusc.get("sample_data", s["data"]["CAM_FRONT"])
        _, boxes, K = nusc.get_sample_data(cam_sd["token"], box_vis_level=BoxVisibility.NONE, selected_anntokens=s["anns"])
        by_ann = {b.token: b for b in boxes}
        g = gt[st]["gt"]
        for k, at in enumerate(g["ann"]):
            ann = nusc.get("sample_annotation", at)
            b = by_ann[at]                                           # devkit box in the CAM_FRONT frame (camera-time ego pose)
            mine = apply_T(gt[st]["T_l2cam"], g["center"][k][None])[0]
            worst_xyz = max(worst_xyz, float(np.abs(mine - b.center).max()))
            uv = view_points(b.center[:, None], np.array(K), normalize=True)[:2, 0]
            mine_uv = (gt[st]["K"] @ mine)[:2] / mine[2]
            if b.center[2] > 0:
                worst_uv = max(worst_uv, float(np.abs(uv - mine_uv).max()))
            # eligibility re-derived from devkit quantities: front, inside image, <=50 m ground distance (ego pose at SAMPLE time), >=1 point
            lid = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lid["ego_pose_token"])
            d = np.hypot(*(np.array(ann["translation"][:2]) - np.array(ego["translation"][:2])))
            ref_region = bool(b.center[2] > 1e-6 and 0 <= uv[0] < 1600 and 0 <= uv[1] < 900 and d <= 50.0)
            ref_elig = ref_region and (ann["num_lidar_pts"] + ann["num_radar_pts"] >= 1) and category_to_detection_name(b.name) is not None
            mism_region += int(ref_region != bool(g["in_region"][k]))
            mism_elig += int(ref_elig != bool(g["eligible"][k]))
            n_elig += int(g["eligible"][k]); n_boxes += 1
    rep = dict(n_samples=len(pick), n_boxes=n_boxes, n_eligible=n_elig, max_abs_center_diff_m=worst_xyz, max_abs_pixel_diff=worst_uv,
               region_flag_mismatches=mism_region, eligible_flag_mismatches=mism_elig,
               note="devkit transform: global -> ego(CAM_FRONT sample_data pose) -> camera; ground distance from LIDAR_TOP-time ego origin")
    json.dump(rep, open(out, "w"), indent=1); print(rep)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(int(x) for x in sys.argv[3:4]))
