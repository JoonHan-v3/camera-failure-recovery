"""Merge the 8 sliced converter outputs into pilot_train_infos.pkl / val_infos.pkl and cross-check every sample against the
raw nuScenes JSON (lidar timestamp, camera order, per-camera timestamp, absolute existing image path).
Usage (project root): python artifacts/trainval_infos/merge_infos.py [out_dir=artifacts/trainval_infos]"""
import glob, os, pickle, sys
sys.path.insert(0, ".")
from outage.nusc_ref import NuscRef, CAMS

ROOT = os.path.expanduser("~/storage/nuscenes_trainval_root")


def main(out="artifacts/trainval_infos"):
    res = {}
    for sp in ("train", "val"):
        inf = []
        for f in sorted(glob.glob(f"{ROOT}/w*_temporal_infos_{sp}.pkl")):
            inf += pickle.load(open(f, "rb"))["infos"]
        res[sp] = sorted(inf, key=lambda e: e["timestamp"])
        print(sp, len(inf), "samples,", len({i["scene_token"] for i in inf}), "scenes")
    ref = NuscRef(os.path.expanduser("~/storage/nuscenes"), "v1.0-trainval")
    bad = n = 0
    for sp in res:
        for i in res[sp]:
            n += 1
            bad += int(i["timestamp"]) != ref.lidar_ts(i["token"])
            bad += list(i["cams"]) != CAMS
            for cam in CAMS:
                c = i["cams"][cam]
                bad += int(c["timestamp"]) != int(ref.kf[(i["token"], cam)]["timestamp"])
                bad += (not os.path.isabs(c["data_path"])) or (not os.path.exists(c["data_path"]))
    print("samples checked", n, "mismatches/missing/relative", bad)
    assert bad == 0, "infos disagree with the raw nuScenes tables"
    for sp, name in (("train", "pilot_train"), ("val", "val")):
        pickle.dump(dict(infos=res[sp], metadata=dict(version="v1.0-trainval")), open(os.path.join(out, f"{name}_infos.pkl"), "wb"))


if __name__ == "__main__":
    main(*sys.argv[1:2])
