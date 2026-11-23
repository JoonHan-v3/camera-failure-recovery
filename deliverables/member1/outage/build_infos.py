"""Build BEVFormer info pkls (upstream converter code, unmodified) for the frozen scene subsets only:
  pilot : the 10 frozen pilot TRAIN-split scenes (same as Member 2: StreamPETR/artifacts/trainval_infos/pilot_scenes.json)
  val   : all 150 nuScenes validation scenes
The upstream `_fill_trainval_infos` iterates `nusc.sample`; we restrict that list to complete chosen scenes (scene order
preserved), so frame_idx / can_bus / cams are exactly what upstream would write. Usage (from BEVFormer/bevformer_src):
  python -m outage.build_infos <pilot|val> <out.pkl>   (PYTHONPATH=bevformer_src:..)"""
import json, os, sys, pickle
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from nuscenes.utils import splits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.expanduser("~/storage/nuscenes_trainval_root")
CAN = os.path.expanduser("~/storage/nuscenes_can_bus")
PILOT = os.path.join(os.path.dirname(ROOT), "StreamPETR/artifacts/trainval_infos/pilot_scenes.json")


def main(which, out):
    sys.path.insert(0, os.path.join(ROOT, "bevformer_src/tools"))
    from data_converter.nuscenes_converter import _fill_trainval_infos
    nusc = NuScenes("v1.0-trainval", DATA, verbose=False)
    names = splits.val if which == "val" else json.load(open(PILOT))
    scenes = {s["token"] for s in nusc.scene if s["name"] in set(names)}
    assert len(scenes) == len(names), (len(scenes), len(names))
    import data_converter.nuscenes_converter as conv      # filter only the ITERATED list (nusc.get indices stay valid)
    conv.mmcv.track_iter_progress = lambda seq: [s for s in seq if s["scene_token"] in scenes]
    can = NuScenesCanBus(dataroot=CAN)
    tr, va = _fill_trainval_infos(nusc, can, scenes if which == "pilot" else set(), scenes if which == "val" else set(), False, max_sweeps=10)
    infos = tr if which == "pilot" else va
    pickle.dump(dict(infos=infos, metadata=dict(version="v1.0-trainval")), open(out, "wb"))
    print(which, len(infos), "samples", len(scenes), "scenes")
    from .canbus_audit import audit                       # guard: CAN values must respect the frozen sample+50 ms allowance
    rep = audit(out)
    json.dump({k: v for k, v in rep.items()}, open(out[:-4] + "_canbus_audit.json", "w"), indent=1)
    if not rep["PASS"]:
        os.rename(out, out + ".REJECTED")
        raise SystemExit(f"CAN-bus causal audit failed ({len(rep['violations'])} violations); info file rejected")


if __name__ == "__main__":
    main(*sys.argv[1:3])
