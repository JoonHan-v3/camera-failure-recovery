"""Official nuScenes detection eval of saved predictions (rows [x,y,z,w,l,h,yaw,vx,vy,label,score]) through
the dataset's own format_results/evaluate (same conversion as StreamPETR's tools/test.py)."""
import os, pickle, sys, json
import numpy as np, torch
sys.path[:0] = ["streampetr_src"]
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def main(cfg_path, ann_pkl, preds_pkl, out_dir):
    cfg = Config.fromfile(cfg_path)
    import importlib; importlib.import_module("projects.mmdet3d_plugin")
    t = cfg.data.test; t.ann_file, t.test_mode = ann_pkl, True
    ds = build_dataset(t)
    preds = pickle.load(open(preds_pkl, "rb"))
    res = []
    for info in ds.data_infos:
        r = torch.tensor(preds[info["token"]])
        res.append(dict(pts_bbox=dict(boxes_3d=LiDARInstance3DBoxes(r[:, :9], box_dim=9), scores_3d=r[:, 10], labels_3d=r[:, 9].long())))
    ev = dict(cfg.get("evaluation", {})); [ev.pop(k, None) for k in ("interval", "tmpdir", "start", "gpu_collect", "save_best", "rule", "pipeline")]
    os.makedirs(out_dir, exist_ok=True)
    m = ds.evaluate(res, metric="bbox", jsonfile_prefix=out_dir, **ev)
    json.dump({k: float(v) for k, v in m.items()}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print({k: round(float(v), 4) for k, v in m.items() if k.endswith(("NDS", "mAP"))})


if __name__ == "__main__":
    main(*sys.argv[1:5])
