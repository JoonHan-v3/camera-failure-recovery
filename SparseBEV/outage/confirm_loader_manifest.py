"""Confirm the REAL SparseBEV loader's image selection (d341052, executed code, no GPU / no image decoding) against the
replicated manifest in BEVFormer/artifacts/loader_manifests.json (study plan p.2: count historical sweeps required by the loader).
For every sample the real dataset.get_data_info + LoadMultiViewImageFromMultiSweeps.load_online(test_mode, sweeps_num=7) (single-GPU
test path) AND load_offline (multi-GPU path; image decoding stubbed) produce the list of filenames the model would use. We check:
  * the 8 frame groups = current keyframe + the 7 previous KEYFRAMES of the same scene (clamped to the scene's oldest keyframe;
    the first sample repeats its own keyframe), cameras in converter order;
  * zero filenames outside samples/ (no camera sweeps) -> the union of unique files = the six keyframe cameras of the evaluated samples;
  * online and offline paths choose identical filenames.
Usage (from sparsebev_src): python -m outage.confirm_loader_manifest <infos.pkl> <out.json>"""
import json, os, sys
import numpy as np
from unittest import mock
from .run_sparsebev import build_test_dataset

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def main(infos, out):
    cfg, ds = build_test_dataset(infos, None)
    from loaders.pipelines.loading import LoadMultiViewImageFromMultiSweeps as L
    loader = L(sweeps_num=7, test_mode=True)
    by_scene = {}
    for i, info in enumerate(ds.data_infos):
        by_scene.setdefault(info["scene_token"], []).append(i)
    rep = dict(n_samples=len(ds.data_infos), mismatching_samples=0, sweep_files_used=0, online_offline_mismatch=0, boundary_repeat_own=0,
               boundary_clamped=0, n_slots=0)
    uniq = set()
    for scene, idxs in by_scene.items():
        for pos, i in enumerate(idxs):
            d = ds.get_data_info(i)
            d["filename"] = list(d["img_filename"]); d["img"] = []
            on = loader.load_online(dict(d, filename=list(d["filename"]), img_timestamp=list(d["img_timestamp"]), lidar2img=list(d["lidar2img"])))
            with mock.patch("mmcv.imread", lambda p, c: np.zeros((1, 1, 3), np.uint8)):
                dd = dict(d, filename=list(d["filename"]), img_timestamp=list(d["img_timestamp"]), lidar2img=list(d["lidar2img"]),
                          img=[np.zeros((1, 1, 3), np.uint8)] * 6)
                off = loader.load_offline(dd)
            fn_on, fn_off = on["filename"], off["filename"]
            rep["n_slots"] += len(fn_on)
            if fn_on != fn_off:
                rep["online_offline_mismatch"] += 1
            expect = []
            for m in range(8):                                   # frame group m = keyframe max(0, pos - m) of the same scene
                src = ds.data_infos[idxs[max(0, pos - m)]]
                expect += [os.path.relpath(src["cams"][c]["data_path"]) for c in CAMS]
            if fn_on != expect:
                rep["mismatching_samples"] += 1
            rep["boundary_repeat_own"] += int(pos == 0)
            rep["boundary_clamped"] += int(1 <= pos < 7)
            for f in fn_on:
                uniq.add(f)
                if "/sweeps/" in f:
                    rep["sweep_files_used"] += 1
    keyframes = {os.path.relpath(c["data_path"]) for i in ds.data_infos for c in i["cams"].values()}
    rep.update(unique_files=len(uniq), unique_files_not_keyframes_of_evaluated_samples=len(uniq - keyframes),
               keyframes_of_evaluated_samples=len(keyframes), first_of_scene_samples=rep.pop("boundary_repeat_own"),
               samples_clamped_to_oldest=rep.pop("boundary_clamped"))
    rep["PASS"] = bool(rep["mismatching_samples"] == 0 and rep["sweep_files_used"] == 0 and rep["online_offline_mismatch"] == 0
                       and rep["unique_files_not_keyframes_of_evaluated_samples"] == 0 and rep["unique_files"] == rep["keyframes_of_evaluated_samples"])
    json.dump(rep, open(out, "w"), indent=1); print(rep)


if __name__ == "__main__":
    main(*sys.argv[1:3])
