"""Per-sample geometry + ground truth for the front-region event analysis, built from the RAW nuScenes JSON tables
(model-independent: the same tables serve BEVFormer, StreamPETR and SparseBEV predictions).

For every sample of the requested scenes:
  * T_l2cam : LiDAR(sample time) -> CAM_FRONT, through global, using the ego pose of the LIDAR_TOP keyframe (sample time) and the
              ego pose of the CAM_FRONT keyframe (camera time). Object motion is NOT compensated to exposure time (plan p.3).
  * K       : CAM_FRONT intrinsics;  T_l2ego : LiDAR -> ego(sample time) (ground-plane distance is measured from the ego origin).
  * gt      : sample annotations of the ten detection classes (nuScenes devkit class mapping) in the LiDAR frame, with
              instance_token and num_lidar_pts + num_radar_pts.
Clean calibration is used (never the corrupted images). Output pickle keyed by sample token."""
import json, os, pickle, sys
import numpy as np
from nuscenes.eval.detection.utils import category_to_detection_name

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
IMG_W, IMG_H = 1600, 900


def quat_to_rot(q):
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n
    return np.array([[1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                     [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                     [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]])


def make_T(rot_q, trans):
    T = np.eye(4)
    T[:3, :3] = quat_to_rot(rot_q)
    T[:3, 3] = trans
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def apply_T(T, p):
    p = np.atleast_2d(np.asarray(p, float))
    return p @ T[:3, :3].T + T[:3, 3]


def project_front(T_l2cam, K, T_l2ego, centers_l, max_dist_m=50.0, img_w=IMG_W, img_h=IMG_H):
    """centers_l: (n,3) LiDAR-frame gravity centers. Returns (in_region bool (n,), uv (n,2), depth (n,), ground_dist (n,)).
    in_region = center in front of the camera (depth > 0) AND inside [0,w) x [0,h) AND ground-plane distance <= max_dist_m."""
    c = np.asarray(centers_l, float).reshape(-1, 3)
    if len(c) == 0:
        return np.zeros(0, bool), np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    pc = apply_T(T_l2cam, c)
    depth = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc @ K.T)[:, :2] / depth[:, None]
    pe = apply_T(T_l2ego, c)
    gd = np.hypot(pe[:, 0], pe[:, 1])
    ok = (depth > 1e-6) & (uv[:, 0] >= 0) & (uv[:, 0] < img_w) & (uv[:, 1] >= 0) & (uv[:, 1] < img_h) & (gd <= max_dist_m)
    return ok, uv, depth, gd


def ground_xy(T_l2ego, centers_lidar):
    """XY (sample-time ego frame) of LiDAR-frame centres: THE ground plane for matching and for the 50 m region rule (geometry version 2)."""
    c = np.asarray(centers_lidar, float).reshape(-1, 3)
    return apply_T(T_l2ego, c)[:, :2] if len(c) else np.zeros((0, 2))


def _load(root, version, name):
    with open(os.path.join(root, version, name + ".json")) as f:
        return json.load(f)


def build(root, version, scene_tokens, out_pkl=None):
    sample = {s["token"]: s for s in _load(root, version, "sample")}
    scenes = {s["token"]: s for s in _load(root, version, "scene")}
    want = {}                                           # sample_token -> scene token
    for tok in scene_tokens:
        t = scenes[tok]["first_sample_token"]
        while t:
            want[t] = tok; t = sample[t]["next"]
    calib = {c["token"]: c for c in _load(root, version, "calibrated_sensor")}
    sensor = {s["token"]: s["channel"] for s in _load(root, version, "sensor")}
    sd_of = {}                                          # (sample, channel) -> sample_data (keyframes)
    for sd in _load(root, version, "sample_data"):
        if sd["is_key_frame"] and sd["sample_token"] in want:
            ch = sensor[calib[sd["calibrated_sensor_token"]]["sensor_token"]]
            if ch in ("LIDAR_TOP", "CAM_FRONT"):
                sd_of[(sd["sample_token"], ch)] = sd
    need_pose = {sd["ego_pose_token"] for sd in sd_of.values()}
    pose = {p["token"]: p for p in _load(root, version, "ego_pose") if p["token"] in need_pose}
    cat = {c["token"]: c["name"] for c in _load(root, version, "category")}
    inst = {i["token"]: cat[i["category_token"]] for i in _load(root, version, "instance")}
    anns = {}
    for a in _load(root, version, "sample_annotation"):
        if a["sample_token"] in want:
            anns.setdefault(a["sample_token"], []).append(a)

    out = {}
    for st, scene in want.items():
        sl, sc = sd_of[(st, "LIDAR_TOP")], sd_of[(st, "CAM_FRONT")]
        cl, cc = calib[sl["calibrated_sensor_token"]], calib[sc["calibrated_sensor_token"]]
        T_l2e = make_T(cl["rotation"], cl["translation"])
        T_es2g = make_T(pose[sl["ego_pose_token"]]["rotation"], pose[sl["ego_pose_token"]]["translation"])
        T_ec2g = make_T(pose[sc["ego_pose_token"]]["rotation"], pose[sc["ego_pose_token"]]["translation"])
        T_c2e = make_T(cc["rotation"], cc["translation"])
        T_l2cam = inv_T(T_c2e) @ inv_T(T_ec2g) @ T_es2g @ T_l2e
        T_g2l = inv_T(T_es2g @ T_l2e)
        K = np.array(cc["camera_intrinsic"], float)
        rows = []
        for a in anns.get(st, []):
            name = category_to_detection_name(inst[a["instance_token"]])
            if name is None:
                continue
            rows.append((a["instance_token"], CLASSES.index(name), apply_T(T_g2l, a["translation"])[0],
                         int(a["num_lidar_pts"]) + int(a["num_radar_pts"]), a["token"]))
        centers = np.array([r[2] for r in rows], float).reshape(-1, 3)
        T_l2ego = T_l2e
        in_reg, uv, depth, gd = project_front(T_l2cam, K, T_l2ego, centers)
        npts = np.array([r[3] for r in rows], int)
        out[st] = dict(scene=scene, ts_lidar_us=int(sl["timestamp"]), ts_front_us=int(sc["timestamp"]),
                       T_l2cam=T_l2cam, K=K, T_l2ego=T_l2ego,
                       gt=dict(instance=[r[0] for r in rows], ann=[r[4] for r in rows], cls=np.array([r[1] for r in rows], int),
                               center=centers, num_pts=npts, in_region=in_reg, depth=depth, ground_dist=gd,
                               eligible=in_reg & (npts >= 1)))
    if out_pkl:
        pickle.dump(out, open(out_pkl, "wb"))
    return out


if __name__ == "__main__":
    # usage: python -m outage.gt_tables <val|pilot> <out.pkl>   (scenes = those of the frozen schedule)
    from .schedule import load_schedule
    which, out_pkl = sys.argv[1:3]
    P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sched, _, _ = load_schedule(os.path.join(P, "artifacts", "schedule_trainval_%s.json" % ("val" if which == "val" else "train")))
    scenes = sorted(sched)
    if which == "pilot":
        names = json.load(open(os.path.join(os.path.dirname(P), "StreamPETR/artifacts/trainval_infos/pilot_scenes.json")))
        root = os.path.expanduser("~/storage/nuscenes")
        sc = {s["name"]: s["token"] for s in _load(root, "v1.0-trainval", "scene")}
        scenes = sorted(sc[n] for n in names)
    t = build(os.path.expanduser("~/storage/nuscenes"), "v1.0-trainval", scenes, out_pkl)
    print(which, len(scenes), "scenes", len(t), "samples", sum(len(v["gt"]["instance"]) for v in t.values()), "GT boxes",
          sum(int(v["gt"]["eligible"].sum()) for v in t.values()), "eligible")
