"""Official nuScenes detection eval of saved SparseBEV predictions (rows [x,y,z,w,l,h,yaw,vx,vy,label,score], native decoded output:
the released config's NMSFreeCoder keeps the top-300 queries with score > 0.05 inside post_center_range) through the dataset's own
format_results/evaluate (the same conversion as upstream val.py: dataset.evaluate(results, jsonfile_prefix=...)).
Run from sparsebev_src. Usage: python -m outage.eval_official <infos.pkl> <preds.pkl> <out_dir>"""
import os, pickle, sys, json
import numpy as np, torch
from . import compat; compat.install()
from .run_sparsebev import build_test_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def main(infos, preds_pkl, out_dir):
    cfg, ds = build_test_dataset(infos, None)
    preds = pickle.load(open(preds_pkl, "rb"))
    res = []
    for info in ds.data_infos:
        r = torch.tensor(np.asarray(preds[info["token"]], np.float32)).reshape(-1, 11)
        res.append(dict(pts_bbox=dict(boxes_3d=LiDARInstance3DBoxes(r[:, :9], 9), scores_3d=r[:, 10], labels_3d=r[:, 9].long())))
    os.makedirs(out_dir, exist_ok=True)
    m = ds.evaluate(res, jsonfile_prefix=out_dir)
    json.dump({k: float(v) for k, v in m.items()}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print({k: round(float(v), 4) for k, v in m.items() if k.endswith(("NDS", "mAP"))})


if __name__ == "__main__":
    main(*sys.argv[1:4])
