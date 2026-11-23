"""Official nuScenes detection eval of saved BEVFormer predictions (rows [x,y,z,w,l,h,yaw,vx,vy,label,score], z = bottom
centre as produced by the head) through the dataset's own format_results/evaluate (same path as upstream tools/test.py --eval bbox).
Run from BEVFormer/bevformer_src:  python -m outage.eval_official <infos.pkl> <preds.pkl> <out_dir>"""
import os, pickle, sys, json
import torch
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from .run_bevformer import CFG, SRC


def main(ann_pkl, preds_pkl, out_dir):
    os.chdir(SRC)
    cfg = Config.fromfile(CFG)
    import importlib; importlib.import_module("projects.mmdet3d_plugin")
    t = cfg.data.test; t.ann_file, t.test_mode = os.path.abspath(ann_pkl), True
    ds = build_dataset(t)
    preds = pickle.load(open(preds_pkl, "rb"))
    assert set(preds) == {i["token"] for i in ds.data_infos}, "predictions must cover exactly the info samples"
    res = []
    for info in ds.data_infos:
        r = torch.tensor(preds[info["token"]])
        res.append(dict(pts_bbox=dict(boxes_3d=LiDARInstance3DBoxes(r[:, :9], box_dim=9), scores_3d=r[:, 10], labels_3d=r[:, 9].long())))
    os.makedirs(out_dir, exist_ok=True)
    m = ds.evaluate(res, metric="bbox", jsonfile_prefix=os.path.abspath(out_dir))
    json.dump({k: float(v) for k, v in m.items()}, open(os.path.join(out_dir, "metrics.json"), "w"), indent=1)
    print({k: round(float(v), 4) for k, v in m.items() if k.endswith(("NDS", "mAP"))})


if __name__ == "__main__":
    main(*[os.path.abspath(x) for x in sys.argv[1:4]])
