"""Real-model state checks (StreamPETR recurrent memory) on saved infos.
 1. every scene run alone from a reset model == the same scene inside the continuous all-scene run
    (so each scene B after any scene A is isolated; scene-boundary reset works)
 2. fresh B twice (explicit reset) identical
 3. NEGATIVE CONTROLS: (a) leaking A's state into B (token forced) must change B and be rejected by the
    model-state guard; (b) interleaved scene order must be rejected by the ordering guard.
Usage: python -m outage.isolation <infos_abs.pkl> <continuous_preds.pkl> <out.json> [version]"""
import json, os, pickle, sys
import numpy as np
from .run_streampetr import Runner, NUSC
from .nusc_ref import NuscRef
from .state import SceneStateGuard, StateResetError
from .streampetr_adapter import CorruptFrontCamera
from .verify_runs import pred_diff


def sub(preds, toks):
    return {k: v for k, v in preds.items() if k in toks}


def main(infos, cont_pkl, out, version="v1.0-mini"):
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False      # FROZEN numerics (see README): TF32 off, real flash-attn
    torch.backends.cudnn.allow_tf32 = False
    ref = NuscRef(NUSC, version)
    scenes = sorted({i["scene_token"] for i in pickle.load(open(infos, "rb"))["infos"]})
    r = Runner(version, infos, scenes)
    all_infos = list(r.dataset.data_infos)
    cont = pickle.load(open(cont_pkl, "rb"))

    def run(scene_list, reset=True, guard=True):
        keep = set(scene_list)
        r.dataset.data_infos = [i for i in all_infos if i["scene_token"] in keep]
        step = CorruptFrontCamera({}, 0.0, None, guard=SceneStateGuard(), ts_mode="abs_s")
        return r.run(None, step, do_reset=reset, use_model_guard=guard)

    import flash_attn
    rep = dict(per_scene_alone_vs_continuous={}, n_scenes=len(scenes), flash_attn=os.path.abspath(flash_attn.__file__), tf32=False)
    for s in scenes:
        p, info = run([s])
        toks = {i["token"] for i in all_infos if i["scene_token"] == s}
        rep["per_scene_alone_vs_continuous"][s] = dict(
            max_abs_diff=pred_diff(p, sub(cont, toks)), fresh_at_start=[f for _, _, f in info["fresh_log"] if f is not None])
    s0, s1 = scenes[0], scenes[1]
    pa, _ = run([s1]); pb, _ = run([s1])
    rep["fresh_B_twice_max_abs_diff"] = pred_diff(pa, pb)
    # A then B continuous, no manual reset in between (model's own boundary reset)
    pab, _ = run([s0, s1]); tb = {i["token"] for i in all_infos if i["scene_token"] == s1}
    rep["B_after_A_vs_fresh_B_max_abs_diff"] = pred_diff(sub(pab, tb), pa)
    # negative control (a): leak A's memory into B by forcing the model to think it is still in scene B
    run([s0]); r.det.prev_scene_token = s1
    try:
        pl, _ = run([s1], reset=False)
        rep["leak_control"] = dict(rejected_by_guard=False, max_abs_diff_vs_fresh=pred_diff(pl, pa))
    except StateResetError as e:
        rep["leak_control"] = dict(rejected_by_guard=True, message=str(e))
    # sensitivity: same leak WITHOUT the guard must change B's predictions (else equality above is vacuous)
    run([s0]); r.det.prev_scene_token = s1
    pl, _ = run([s1], reset=False, guard=False)
    rep["leak_unguarded_max_abs_diff_vs_fresh"] = pred_diff(pl, pa)
    # negative control (b): interleaved frames of two scenes
    g = SceneStateGuard(); inter = [i for i in all_infos if i["scene_token"] in (s0, s1)]
    seq = [x for pair in zip([i for i in inter if i["scene_token"] == s0][:5], [i for i in inter if i["scene_token"] == s1][:5]) for x in pair]
    try:
        for i in seq + seq[:1]:
            g.step(i["scene_token"], int(i["timestamp"]), require_memory=False)
        rep["interleave_control"] = dict(rejected=False)
    except StateResetError as e:
        rep["interleave_control"] = dict(rejected=True, message=str(e))
    rep["all_isolated"] = (all(v["max_abs_diff"] == 0.0 and v["fresh_at_start"] == [True] for v in rep["per_scene_alone_vs_continuous"].values())
                           and rep["leak_control"].get("rejected_by_guard") is True and rep["interleave_control"].get("rejected") is True and rep["fresh_B_twice_max_abs_diff"] == 0.0 and rep["leak_unguarded_max_abs_diff_vs_fresh"] > 0.0 and rep["B_after_A_vs_fresh_B_max_abs_diff"] == 0.0)
    json.dump(rep, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_scene_alone_vs_continuous"}, indent=1, default=str))
    print("per-scene max diffs:", [v["max_abs_diff"] for v in rep["per_scene_alone_vs_continuous"].values()])


if __name__ == "__main__":
    main(*sys.argv[1:5])
