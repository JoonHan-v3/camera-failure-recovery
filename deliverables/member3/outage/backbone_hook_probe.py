"""Backbone-hook probe (real GPU inference): forward_pre_hook on SparseBEV's img_backbone records the exact tensor the ResNet receives
(after GPU BGR->RGB, mean/std normalization, padding to /32, GridMask = identity in eval) for every frame in a clean and an outage run
of the same scenes (same Runner as the study runs). Per frame: CAM_FRONT view of corrupted frames == normalized black ((0-mean)/std,
RGB) over the whole 256x704 area; all other views, and every uncorrupted frame, bit-identical to the clean run's backbone input.
Each frame triggers exactly one backbone call (current six images; history comes from the cache), which is asserted.
Usage (from sparsebev_src): python -m outage.backbone_hook_probe <infos.pkl> <schedule.json> <duration_s> <out.json> <scene,...>"""
import json, sys
import numpy as np, torch
from .run_sparsebev import Runner
from .adapter import CorruptFrontCamera
from .schedule import load_schedule
from .state import SceneStateGuard
from .corruption import is_corrupted


def main(infos, sched_path, dur, out, scenes):
    from .numerics import set_study_numerics
    set_study_numerics()
    scenes = scenes.split(",")
    sched, _, _ = load_schedule(sched_path)
    sch = {s: sched[s] for s in scenes}
    r = Runner(infos, scenes)
    cap = []
    h = r.det.img_backbone.register_forward_pre_hook(lambda m, inp: cap.append(inp[0].detach().float().cpu().numpy()))

    def run(d):
        cap.clear()
        r.run(CorruptFrontCamera(sch, d, None, guard=SceneStateGuard(), ts_mode="abs_s"))
        return list(cap)
    C, O = run(0.0), run(float(dur))
    h.remove()
    assert len(C) == len(O) == len(r.dataset), (len(C), len(O), len(r.dataset))
    nm = r.det.data_aug["img_norm_cfg"]
    black = (0 - np.array(nm["mean"], np.float32)) / np.array(nm["std"], np.float32)
    rep = dict(n_frames=len(C), backbone_input_shape=list(C[0].shape), duration_s=float(dur), corrupted=0, exact_black=0,
               other_views_identical=True, uncorrupted_identical=True)
    for i, info in enumerate(r.dataset.data_infos):
        so = sch[info["scene_token"]]
        bad = is_corrupted("CAM_FRONT", int(info["cams"]["CAM_FRONT"]["timestamp"]), so.t_start_us, float(dur))
        c, o = C[i], O[i]                                  # (6, 3, 256, 704): views x channels in converter order
        if not np.array_equal(c[1:], o[1:]):
            rep["other_views_identical"] = False
        if bad:
            rep["corrupted"] += 1
            rep["exact_black"] += bool(all(np.all(o[0, k] == black[k]) for k in range(3)))
        elif not np.array_equal(c, o):
            rep["uncorrupted_identical"] = False
    rep["PASS"] = bool(rep["corrupted"] > 0 and rep["corrupted"] == rep["exact_black"]
                       and rep["other_views_identical"] and rep["uncorrupted_identical"])
    json.dump(rep, open(out, "w"), indent=1)
    print(rep)


if __name__ == "__main__":
    main(*sys.argv[1:6])
