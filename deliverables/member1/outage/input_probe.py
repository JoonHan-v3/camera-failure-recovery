"""Model-input probe (CPU only, no model): build the exact tensors BEVFormer receives (after OutageFrontCamera ->
NormalizeMultiviewImage -> x0.5 RandomScale -> Pad/32 -> FormatBundle) for a clean and an outage condition and check, for
every sample of the given scenes: corrupted CAM_FRONT slot == normalized black ((0-mean)/std per RGB channel) over the
whole 450x800 image area (pad = 0); every other slot and every uncorrupted frame bit-identical to the clean tensor.
Also saves PNG contact sheets of the raw (pre-normalization) front images around the outage for visual inspection.
Usage (from bevformer_src): python -m outage.input_probe <infos.pkl> <schedule.json> <duration_s> <out_dir> <scene_token>[,...]"""
import json, os, sys
import numpy as np, mmcv
from .run_bevformer import build_test_dataset
from .mm_pipeline import OutageFrontCamera
from .adapter import CorruptFrontCamera
from .schedule import load_schedule
from .corruption import is_corrupted


def tensors(ds, step):
    OutageFrontCamera.STEP = step
    out = []
    for i in range(len(ds)):
        d = ds[i]
        img = d["img"][0].data if isinstance(d["img"], list) else d["img"].data
        out.append(img.numpy())                       # (6, 3, H, W) float32
    return out


def main(infos, sched_path, dur, out_dir, scenes):
    os.makedirs(out_dir, exist_ok=True)
    scenes = scenes.split(",")
    sched, _, _ = load_schedule(sched_path)
    cfg, ds = build_test_dataset(infos, scenes)
    norm = [t for t in cfg.data.test.pipeline if t["type"] == "NormalizeMultiviewImage"][0]
    black = (0 - np.array(norm["mean"], np.float32)) / np.array(norm["std"], np.float32)   # to_rgb: channel order RGB
    sch = {s: sched[s] for s in scenes}
    C = tensors(ds, CorruptFrontCamera(sch, 0.0, None, ts_mode="abs_us"))
    O = tensors(ds, CorruptFrontCamera(sch, float(dur), None, ts_mode="abs_us"))
    rep = dict(n_samples=len(ds), duration_s=float(dur), tensor_shape=list(C[0].shape), normalized_black=black.tolist(),
               corrupted_samples=0, corrupted_slot_exact_black=0, pad_zero=0, other_slots_identical=True,
               uncorrupted_frames_identical=True, frames=[])
    H, W = 450, 800
    for i, info in enumerate(ds.data_infos):
        so = sch[info["scene_token"]]
        ts = int(info["cams"]["CAM_FRONT"]["timestamp"])
        bad = is_corrupted("CAM_FRONT", ts, so.t_start_us, float(dur))
        rep["frames"].append(dict(sample=info["token"], front_ts_us=ts, offset_from_start_s=(ts - so.t_start_us) / 1e6, black=bad))
        if not np.array_equal(C[i][1:], O[i][1:]):
            rep["other_slots_identical"] = False
        if bad:
            rep["corrupted_samples"] += 1
            f = O[i][0]
            rep["corrupted_slot_exact_black"] += bool(all(np.all(f[c, :H, :W] == black[c]) for c in range(3)))
            rep["pad_zero"] += bool(np.all(f[:, H:, :] == 0) and np.all(f[:, :, W:] == 0))
        elif not np.array_equal(C[i], O[i]):
            rep["uncorrupted_frames_identical"] = False
    rep["PASS"] = bool(rep["corrupted_samples"] > 0 and rep["corrupted_slot_exact_black"] == rep["corrupted_samples"] ==
                       rep["pad_zero"] and rep["other_slots_identical"] and rep["uncorrupted_frames_identical"])
    # contact sheet: raw front images of the frames around the outage, as the model would see them (post-corruption)
    idx = [k for k, f in enumerate(rep["frames"]) if -1.1 <= f["offset_from_start_s"] <= float(dur) + 1.1]
    tiles = []
    for k in idx:
        x = O[k][0][:, :H, :W].transpose(1, 2, 0) * np.array(norm["std"]) + np.array(norm["mean"])    # back to RGB 0..255
        tiles.append(mmcv.imrescale(np.clip(x, 0, 255).astype(np.uint8)[:, :, ::-1], 0.25))
    if tiles:
        mmcv.imwrite(np.concatenate(tiles, 1), os.path.join(out_dir, f"front_inputs_d{dur}_{scenes[0][:8]}.jpg"))
    json.dump(rep, open(os.path.join(out_dir, f"input_probe_d{dur}.json"), "w"), indent=1)
    print({k: v for k, v in rep.items() if k != "frames"})


if __name__ == "__main__":
    main(*sys.argv[1:6])
