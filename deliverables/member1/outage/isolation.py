"""Real-model state checks for BEVFormer's recurrent BEV cache (model.prev_frame_info) on saved infos.
 1. every scene run alone from a reset model == the same scene inside the continuous all-scene run (boundary reset works)
 2. fresh B twice (explicit reset) identical; B right after A (model's own boundary reset only) == fresh B
 3. NEGATIVE CONTROLS: (a) leak A's prev_bev into B (scene token forced to B): must change B and be rejected by the guard;
    (b) interleaved scene order rejected by the ordering guard; (c) stale-run hazard: WITHOUT the explicit run reset, a run
    whose first scene equals the previous run's last scene reuses the old prev_bev (shows why Runner.reset() is required).
Usage (from bevformer_src): python -m outage.isolation <infos.pkl> <continuous_preds.pkl> <out.json>"""
import json, pickle, sys
import torch
from .run_bevformer import Runner
from .state import SceneStateGuard, StateResetError
from .adapter import CorruptFrontCamera
from .verify_runs import pred_diff


def sub(preds, toks):
    return {k: v for k, v in preds.items() if k in toks}


def main(infos, cont_pkl, out):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scenes = sorted({i["scene_token"] for i in pickle.load(open(infos, "rb"))["infos"]})
    r = Runner(infos, scenes)
    all_infos = list(r.dataset.data_infos)
    cont = pickle.load(open(cont_pkl, "rb"))

    def run(scene_list, reset=True, guard=True):
        keep = set(scene_list)
        r.dataset.data_infos = [i for i in all_infos if i["scene_token"] in keep]
        step = CorruptFrontCamera({}, 0.0, None, guard=SceneStateGuard(), ts_mode="abs_us")
        return r.run(step, do_reset=reset, use_model_guard=guard)

    toks = lambda s: {i["token"] for i in all_infos if i["scene_token"] == s}
    rep = dict(per_scene_alone_vs_continuous={}, n_scenes=len(scenes), tf32=False)
    for s in scenes:
        p, info = run([s])
        rep["per_scene_alone_vs_continuous"][s] = dict(max_abs_diff=pred_diff(p, sub(cont, toks(s))),
                                                       fresh_at_start=info["fresh_log"][0][2])
    s0, s1 = scenes[0], scenes[1]
    pa, _ = run([s1]); pb, _ = run([s1])
    rep["fresh_B_twice_max_abs_diff"] = pred_diff(pa, pb)
    pab, _ = run([s0, s1])
    rep["B_after_A_vs_fresh_B_max_abs_diff"] = pred_diff(sub(pab, toks(s1)), pa)
    run([s0]); r.det.prev_frame_info["scene_token"] = s1            # (a) leak with guard
    try:
        pl, _ = run([s1], reset=False)
        rep["leak_control"] = dict(rejected_by_guard=False, max_abs_diff_vs_fresh=pred_diff(pl, pa))
    except StateResetError as e:
        rep["leak_control"] = dict(rejected_by_guard=True, message=str(e))
    run([s0]); r.det.prev_frame_info["scene_token"] = s1            # same leak, unguarded: must change predictions
    pl, _ = run([s1], reset=False, guard=False)
    rep["leak_unguarded_max_abs_diff_vs_fresh"] = pred_diff(pl, pa)
    run([s1])                                                        # (c) natural hazard: rerun same scene, no reset
    pn, info = run([s1], reset=False, guard=False)
    rep["no_reset_rerun_same_scene"] = dict(prev_bev_reused_at_first_frame=not info["fresh_log"][0][2],
                                            max_abs_diff_vs_fresh=pred_diff(pn, pa))
    g = SceneStateGuard(); inter = [i for i in all_infos if i["scene_token"] in (s0, s1)]   # (b) interleave
    seq = [x for pair in zip([i for i in inter if i["scene_token"] == s0][:5], [i for i in inter if i["scene_token"] == s1][:5]) for x in pair]
    try:
        for i in seq:
            g.step(i["scene_token"], int(i["timestamp"]), require_memory=False)
        rep["interleave_control"] = dict(rejected=False)
    except StateResetError as e:
        rep["interleave_control"] = dict(rejected=True, message=str(e))
    rep["all_isolated"] = bool(all(v["max_abs_diff"] == 0.0 and v["fresh_at_start"] for v in rep["per_scene_alone_vs_continuous"].values())
                               and rep["leak_control"].get("rejected_by_guard") is True and rep["interleave_control"]["rejected"]
                               and rep["fresh_B_twice_max_abs_diff"] == 0.0 and rep["B_after_A_vs_fresh_B_max_abs_diff"] == 0.0
                               and rep["leak_unguarded_max_abs_diff_vs_fresh"] > 0.0)
    json.dump(rep, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_scene_alone_vs_continuous"}, indent=1, default=str))
    print("per-scene max diffs:", [v["max_abs_diff"] for v in rep["per_scene_alone_vs_continuous"].values()])


if __name__ == "__main__":
    main(*sys.argv[1:4])
