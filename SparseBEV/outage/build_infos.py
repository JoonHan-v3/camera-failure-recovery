"""Build SparseBEV info pkls for the frozen scene subsets, with the upstream code unmodified:
  1. mmdet3d v1.0.0rc6 `_fill_trainval_infos` (what `tools/create_data.py nuscenes` writes = nuscenes_infos_{train,val}.pkl),
     restricted to complete chosen scenes by filtering only the ITERATED list (nusc.get indices stay valid);
  2. SparseBEV d341052 `gen_sweep_info.add_sweep_info` (what `gen_sweep_info.py` writes = *_sweep.pkl).
We add one key, `scene_token` (needed by the outage schedule; not read by any upstream code).
  pilot : the 10 frozen pilot TRAIN-split scenes (same as Members 1/2: StreamPETR/artifacts/trainval_infos/pilot_scenes.json)
  val   : all 150 nuScenes validation scenes
Usage (from SparseBEV/sparsebev_src, env sparsebev): PYTHONPATH=$PWD:.. python -m outage.build_infos <pilot|val> <out.pkl>"""
import json, os, sys, pickle
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PILOT = os.path.join(os.path.dirname(ROOT), "StreamPETR/artifacts/trainval_infos/pilot_scenes.json")
DATA = "data/nuscenes"      # = ~/storage/nuscenes_trainval_root (symlink); upstream data_path prefix


def main(which, out):
    assert os.path.basename(os.getcwd()) == "sparsebev_src", "run from SparseBEV/sparsebev_src (relative data paths)"
    sys.path.insert(0, os.path.join(ROOT, "mmdetection3d/tools"))
    import data_converter.nuscenes_converter as conv
    nusc = NuScenes("v1.0-trainval", DATA, verbose=False)
    names = splits.val if which == "val" else json.load(open(PILOT))
    scenes = {s["token"] for s in nusc.scene if s["name"] in set(names)}
    assert len(scenes) == len(names), (len(scenes), len(names))
    conv.mmcv.track_iter_progress = lambda seq: [s for s in seq if s["scene_token"] in scenes]
    tr, va = conv._fill_trainval_infos(nusc, scenes if which == "pilot" else set(),
                                       scenes if which == "val" else set(), False, max_sweeps=10)
    infos = tr if which == "pilot" else va
    for i in infos:
        i["scene_token"] = nusc.get("sample", i["token"])["scene_token"]
    sys.argv = ["gen_sweep_info.py", "--data-root", DATA, "--version", "v1.0-trainval"]
    import gen_sweep_info                                     # upstream module (argparse at import; main guarded)
    data = gen_sweep_info.add_sweep_info(nusc, dict(infos=infos, metadata=dict(version="v1.0-trainval")))
    pickle.dump(data, open(out, "wb"))
    print(which, len(data["infos"]), "samples", len(scenes), "scenes")


if __name__ == "__main__":
    main(*sys.argv[1:3])
