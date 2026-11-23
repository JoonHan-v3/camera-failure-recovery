"""Backbone-hook probe (real GPU inference): registers a forward_pre_hook on the model's img_backbone and records the exact
tensor the ResNet receives for every frame, in a clean and an outage run of the same scenes (same Runner as the study runs).
Checks, per frame: CAM_FRONT view of corrupted frames == normalized black ((0-mean)/std, RGB) over the 450x800 image area and 0
in the padding; all other views, and every uncorrupted frame, bit-identical to the clean run's backbone input.
Usage (from bevformer_src): python -m outage.backbone_hook_probe <infos.pkl> <schedule.json> <duration_s> <out.json> <scene,...>"""
import json, sys
import numpy as np, torch
from .run_bevformer import Runner
from .adapter import CorruptFrontCamera
from .schedule import load_schedule
from .state import SceneStateGuard
from .corruption import is_corrupted


def main(infos, sched_path, dur, out, scenes):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scenes = scenes.split(",")
    sched, _, _ = load_schedule(sched_path)
    sch = {s: sched[s] for s in scenes}
    r = Runner(infos, scenes)
    cap = []
    h = r.det.img_backbone.register_forward_pre_hook(lambda m, inp: cap.append(inp[0].detach().float().cpu().numpy()))

    def run(d):
        cap.clear()
        r.run(CorruptFrontCamera(sch, d, None, guard=SceneStateGuard(), ts_mode="abs_us"))
        return list(cap)
    C, O = run(0.0), run(float(dur))
    h.remove()
    norm = dict(mean=np.array([123.675, 116.28, 103.53], np.float32), std=np.array([58.395, 57.12, 57.375], np.float32))
    black = (0 - norm["mean"]) / norm["std"]
    rep = dict(n_frames=len(C), backbone_input_shape=list(C[0].shape), duration_s=float(dur), corrupted=0, exact_black=0,
               pad_zero=0, other_views_identical=True, uncorrupted_identical=True)
    H, W = 450, 800
    for i, info in enumerate(r.dataset.data_infos):
        so = sch[info["scene_token"]]
        bad = is_corrupted("CAM_FRONT", int(info["cams"]["CAM_FRONT"]["timestamp"]), so.t_start_us, float(dur))
        c, o = C[i], O[i]                                  # (6, 3, 480, 800): views x channels in converter order
        if not np.array_equal(c[1:], o[1:]):
            rep["other_views_identical"] = False
        if bad:
            rep["corrupted"] += 1
            rep["exact_black"] += bool(all(np.all(o[0, k, :H, :W] == black[k]) for k in range(3)))
            rep["pad_zero"] += bool(np.all(o[0, :, H:, :] == 0) and np.all(o[0, :, :, W:] == 0))
        elif not np.array_equal(c, o):
            rep["uncorrupted_identical"] = False
    rep["PASS"] = bool(rep["corrupted"] > 0 and rep["corrupted"] == rep["exact_black"] == rep["pad_zero"]
                       and rep["other_views_identical"] and rep["uncorrupted_identical"])
    json.dump(rep, open(out, "w"), indent=1)
    print(rep)


if __name__ == "__main__":
    main(*sys.argv[1:6])
