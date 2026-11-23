"""Model-input probe (CPU only, no model): build the exact uint8 image tensors SparseBEV's pipeline hands to the model
(LoadMultiViewImageFromFiles -> OutageFrontCamera -> LoadMultiViewImageFromMultiSweeps[online] -> RandomTransformImage(resize/crop to
256x704) -> FormatBundle) for a clean and an outage condition and check, for every sample of the given scenes: corrupted CAM_FRONT slot
is all zero over the full 256x704 area; every other slot and every uncorrupted frame is bit-identical to the clean tensor. The GPU-side
normalization/padding is covered by backbone_hook_probe.py. Also saves a contact sheet of the front inputs around the outage.
Usage (from sparsebev_src): python -m outage.input_probe <infos.pkl> <schedule.json> <duration_s> <out_dir> <scene_token>[,...]"""
import json, os, sys
import numpy as np, mmcv
from .run_sparsebev import build_test_dataset
from .adapter import OutageFrontCamera, CorruptFrontCamera
from .schedule import load_schedule
from .corruption import is_corrupted


def tensors(ds, step):
    OutageFrontCamera.STEP = step
    out = []
    for i in range(len(ds)):
        d = ds[i]
        img = d["img"][0].data if isinstance(d["img"], list) else d["img"].data
        out.append(np.asarray(img))                   # (6, 3, 256, 704) uint8 (current six images only in single-GPU test mode)
    return out


def main(infos, sched_path, dur, out_dir, scenes):
    os.makedirs(out_dir, exist_ok=True)
    scenes = scenes.split(",")
    sched, _, _ = load_schedule(sched_path)
    cfg, ds = build_test_dataset(infos, scenes)
    sch = {s: sched[s] for s in scenes}
    C = tensors(ds, CorruptFrontCamera(sch, 0.0, None, ts_mode="abs_s"))
    O = tensors(ds, CorruptFrontCamera(sch, float(dur), None, ts_mode="abs_s"))
    rep = dict(n_samples=len(ds), duration_s=float(dur), tensor_shape=list(C[0].shape), dtype=str(C[0].dtype),
               corrupted_samples=0, corrupted_slot_all_zero=0, other_slots_identical=True, uncorrupted_frames_identical=True, frames=[])
    for i, info in enumerate(ds.data_infos):
        so = sch[info["scene_token"]]
        ts = int(info["cams"]["CAM_FRONT"]["timestamp"])
        bad = is_corrupted("CAM_FRONT", ts, so.t_start_us, float(dur))
        rep["frames"].append(dict(sample=info["token"], front_ts_us=ts, offset_from_start_s=(ts - so.t_start_us) / 1e6, black=bad))
        if not np.array_equal(C[i][1:], O[i][1:]):
            rep["other_slots_identical"] = False
        if bad:
            rep["corrupted_samples"] += 1
            rep["corrupted_slot_all_zero"] += bool(not O[i][0].any())
            assert C[i][0].any(), "clean front image is itself black?"
        elif not np.array_equal(C[i], O[i]):
            rep["uncorrupted_frames_identical"] = False
    rep["PASS"] = bool(rep["corrupted_samples"] > 0 and rep["corrupted_slot_all_zero"] == rep["corrupted_samples"]
                       and rep["other_slots_identical"] and rep["uncorrupted_frames_identical"])
    idx = [k for k, f in enumerate(rep["frames"]) if -1.1 <= f["offset_from_start_s"] <= float(dur) + 1.1]
    tiles = [mmcv.imrescale(np.ascontiguousarray(O[k][0].transpose(1, 2, 0)), 0.5) for k in idx]       # BGR uint8
    if tiles:
        mmcv.imwrite(np.concatenate(tiles, 1), os.path.join(out_dir, f"front_inputs_d{dur}_{scenes[0][:8]}.jpg"))
    json.dump(rep, open(os.path.join(out_dir, f"input_probe_d{dur}.json"), "w"), indent=1)
    print({k: v for k, v in rep.items() if k != "frames"})


if __name__ == "__main__":
    main(*sys.argv[1:6])
