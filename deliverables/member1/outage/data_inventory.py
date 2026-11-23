"""Shared data inventory (no GPU): reconcile every sample_data file named in the raw v1.0-trainval JSON against the extracted
blobs (~/storage/nuscenes_dl/v1.0-trainval??_blobs) and the keyframe copy used by the BEVFormer/StreamPETR loaders
(~/storage/nuscenes_trainval_keyframes), per channel and keyframe/sweep, with on-disk bytes, plus val-scene totals.
The EXACT per-loader manifests (val AND pilot, per model) are produced by outage/loader_manifests.py; the camera-sweep totals
here are the available-scene inventory (upper bound), not files selected by any loader.
Usage: python3 -m outage.data_inventory <out.json>"""
import hashlib, json, os, random, sys, collections
from .nusc_ref import NuscRef
H = os.path.expanduser("~")
DL = os.path.join(H, "storage/nuscenes_dl")
KF = os.path.join(H, "storage/nuscenes_trainval_keyframes")
META = os.path.join(H, "storage/nuscenes")
CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def du(p):
    tot = 0
    for r, _, fs in os.walk(p):
        for f in fs:
            fp = os.path.join(r, f)
            if not os.path.islink(fp):
                tot += os.path.getsize(fp)
    return tot


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def main(out):
    js = lambda n: json.load(open(os.path.join(META, "v1.0-trainval", n + ".json")))
    sd = js("sample_data"); calib = {c["token"]: c for c in js("calibrated_sensor")}
    sensor = {s["token"]: s["channel"] for s in js("sensor")}
    blobs = sorted(d for d in os.listdir(DL) if d.startswith("v1.0-trainval") and d.endswith("_blobs") and os.path.isdir(os.path.join(DL, d)))
    where = {}                                            # relative filename -> blob dir (first hit)
    for b in blobs:
        for top in ("samples", "sweeps"):
            base = os.path.join(DL, b, top)
            if not os.path.isdir(base):
                continue
            for ch in os.listdir(base):
                with os.scandir(os.path.join(base, ch)) as it:
                    for e in it:
                        where.setdefault(f"{top}/{ch}/{e.name}", (b, e.stat().st_size))
    stats = collections.defaultdict(lambda: dict(n_json=0, n_blob=0, bytes_blob=0, n_kfcopy=0, kfcopy_size_mismatch=0))
    for r in sd:
        ch = sensor[calib[r["calibrated_sensor_token"]]["sensor_token"]]
        k = f"{ch}|{'keyframe' if r['is_key_frame'] else 'sweep'}"
        s = stats[k]; s["n_json"] += 1
        hit = where.get(r["filename"])
        if hit:
            s["n_blob"] += 1; s["bytes_blob"] += hit[1]
        if r["is_key_frame"] and ch in CAMS + ["LIDAR_TOP"]:
            p = os.path.join(KF, r["filename"])
            if os.path.exists(p):
                s["n_kfcopy"] += 1
                if hit and os.path.getsize(p) != hit[1]:
                    s["kfcopy_size_mismatch"] += 1
    json_names = {r["filename"] for r in sd}
    extra = [f for f in where if f not in json_names]
    # sampled byte-identity of the keyframe copy vs blobs
    random.seed(0); kfs = [r["filename"] for r in sd if r["is_key_frame"] and r["filename"].startswith("samples/CAM")]
    samp = random.sample(kfs, 300)
    md5_same = sum(md5(os.path.join(KF, f)) == md5(os.path.join(DL, where[f][0], f)) for f in samp)
    # what each loader consumes for the frozen evaluation set (150 val scenes) and pilot (10 train scenes)
    ref = NuscRef(META, "v1.0-trainval")
    val = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts/schedule_trainval_val.json")))
    from nuscenes.utils import splits
    vtok = [t for t, v in ref.scenes.items() if v["name"] in set(splits.val)]
    nkf = sum(len(ref.scenes[t]["samples"]) for t in vtok)
    sample_scene = {s: t for t in vtok for s in ref.scenes[t]["samples"]}
    cam_sweeps_val = collections.Counter()
    tok2sample = {}
    for r in sd:
        ch = sensor[calib[r["calibrated_sensor_token"]]["sensor_token"]]
        if ch in CAMS and not r["is_key_frame"] and r["sample_token"] in sample_scene:
            cam_sweeps_val[ch] += 1
    kf_bytes_val = sum(where[ref.kf[(s, c)]["filename"]][1] for t in vtok for s in ref.scenes[t]["samples"] for c in CAMS)
    sw_bytes_val = sum(where[r["filename"]][1] for r in sd if not r["is_key_frame"] and r["sample_token"] in sample_scene
                       and sensor[calib[r["calibrated_sensor_token"]]["sensor_token"]] in CAMS and r["filename"] in where)
    rep = dict(
        blob_dirs=blobs, per_channel=dict(sorted(stats.items())),
        all_json_files_present_in_blobs=all(v["n_json"] == v["n_blob"] for v in stats.values()),
        files_on_disk_not_in_json=len(extra), extra_examples=extra[:5],
        keyframe_copy_md5_identical_sampled=f"{md5_same}/300",
        disk_bytes=dict(nuscenes_dl_tgz=sum(os.path.getsize(os.path.join(DL, f)) for f in os.listdir(DL) if f.endswith(".tgz")),
                        nuscenes_dl_extracted_blobs=sum(du(os.path.join(DL, b)) for b in blobs),
                        keyframe_copy=du(KF), metadata_trainval=du(os.path.join(META, "v1.0-trainval")),
                        can_bus=du(os.path.join(H, "storage/nuscenes_can_bus"))),
        val_scenes=len(vtok), val_scenes_scheduled=len(val["scenes"]), val_keyframe_samples=nkf,
        val_keyframe_cam_images=nkf * 6, val_keyframe_cam_bytes=kf_bytes_val,
        val_scene_cam_sweeps_available_UPPER_BOUND=dict(cam_sweeps_val), val_scene_cam_sweep_bytes_UPPER_BOUND=sw_bytes_val,
        loaders={"BEVFormer-tiny": "6 keyframe cams/sample; no camera sweeps; recurrent prev_bev (+ CAN-bus pose, can_bus.zip)",
                 "StreamPETR": "6 keyframe cams/sample; no camera sweeps (Member 2 verified)",
                 "SparseBEV": "r50_nuimg_704x256 @ d341052 test mode: current + 7 previous KEYFRAMES, 0 sweeps (replicated; see loader_manifests.json)"},
        exact_loader_manifests="artifacts/loader_manifests.json")
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_channel"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
