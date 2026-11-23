"""Real-model state checks for SparseBEV's filename-keyed feature cache (SparseBEV.memory / .queue) on saved infos.
 1. every scene run alone from a reset model == the same scene inside the continuous all-scene run (scene boundaries are isolated)
 2. fresh B twice identical; B right after A (no reset between scenes, as in a continuous run) == fresh B
 3. NEGATIVE CONTROLS:
    (a) stale-cache hazard (the reason Runner.reset() and the cache guard exist). The cache is a FIFO of 8 filenames, so after a FULL-scene
        run the leftover entries are the scene's LAST keyframes and are evicted before a rerun reaches them (shown first: 0 hits). A hit
        needs leftovers that start at the scene's first frames, e.g. an interrupted/short run: run only the first 7 samples of scene B
        with a 2 s outage (the outage starts at keyframe 6, so sample 6's cached features are black), then run the CLEAN full scene
        WITHOUT clearing the cache: the study guard must reject it (frame 0 already cached); unguarded, the clean run silently reuses
        the black features at sample 6 (that sample's predictions == the outage run's, != the fresh clean run's);
    (b) interleaved scene order rejected by the ordering guard.
Usage (from sparsebev_src): python -m outage.isolation <infos.pkl> <continuous_clean_preds.pkl> <schedule.json> <out.json>"""
import json, pickle, sys
import torch
from .run_sparsebev import Runner
from .schedule import load_schedule
from .state import SceneStateGuard, StateResetError
from .adapter import CorruptFrontCamera
from .verify_runs import pred_diff as _pred_diff


def pred_diff(a, b):
    """Max |a-b|; differing row counts / schema problems (the shared helper returns inf) are encoded as None (strict-JSON friendly) and
    reported via `shape_mismatch` fields: they are NOT numerical differences."""
    d = _pred_diff(a, b)
    return None if d == float("inf") else d


def differs(d):
    return d is None or d > 0.0


def sub(preds, toks):
    return {k: v for k, v in preds.items() if k in toks}


def main(infos, cont_pkl, sched_path, out):
    from .numerics import set_study_numerics
    set_study_numerics()
    sched, _, _ = load_schedule(sched_path)
    scenes = sorted({i["scene_token"] for i in pickle.load(open(infos, "rb"))["infos"]})
    r = Runner(infos, scenes)
    all_infos = list(r.dataset.data_infos)
    cont = pickle.load(open(cont_pkl, "rb"))

    def run(scene_list, dur=0.0, reset=True, guard=True):
        keep = set(scene_list)
        r.dataset.data_infos = [i for i in all_infos if i["scene_token"] in keep]
        step = CorruptFrontCamera({s: sched[s] for s in scene_list if s in sched}, dur, None, guard=SceneStateGuard(), ts_mode="abs_s")
        return r.run(step, do_reset=reset, use_model_guard=guard)

    rep = dict(per_scene_alone_vs_continuous={}, n_scenes=len(scenes), tf32=False)
    for s in scenes:
        p, info = run([s])
        toks = {i["token"] for i in all_infos if i["scene_token"] == s}
        o = info["obs_log"]
        rep["per_scene_alone_vs_continuous"][s] = dict(max_abs_diff=pred_diff(p, sub(cont, toks)),
                                                       fresh_at_start=not o[0]["cur_cached_before"], max_cache_size=max(x["cache_size_before"] for x in o))
    s0, s1 = scenes[0], scenes[1]
    pa, _ = run([s1]); pb, _ = run([s1])
    pa_full = pa
    rep["fresh_B_twice_max_abs_diff"] = pred_diff(pa, pb)
    pab, _ = run([s0, s1]); tb = {i["token"] for i in all_infos if i["scene_token"] == s1}
    rep["B_after_A_vs_fresh_B_max_abs_diff"] = pred_diff(sub(pab, tb), pa)
    # (a) stale-cache hazard on scene s1
    full_out, _ = run([s1], dur=2.0)                                      # full-scene outage run, then clean rerun WITHOUT reset
    try:
        run([s1], dur=0.0, reset=False)
        rep["full_run_then_rerun_without_reset"] = dict(rejected_by_guard=False, note="no stale hit: FIFO-8 evicts the leftover tail first")
    except StateResetError as e:
        rep["full_run_then_rerun_without_reset"] = dict(rejected_by_guard=True, message=str(e))
    first7 = sorted([i for i in all_infos if i["scene_token"] == s1], key=lambda i: i["timestamp"])[:7]

    def run_short(dur, reset, guard):
        r.dataset.data_infos = first7
        step = CorruptFrontCamera({s1: sched[s1]}, dur, None, guard=SceneStateGuard(), ts_mode="abs_s")
        return r.run(step, do_reset=reset, use_model_guard=guard)
    p_short_out, _ = run_short(2.0, True, True)                           # leaves cache entries for samples 0..6 (sample 6 black)
    try:
        run([s1], dur=0.0, reset=False)
        rep["stale_cache_control"] = dict(rejected_by_guard=False)
    except StateResetError as e:
        rep["stale_cache_control"] = dict(rejected_by_guard=True, message=str(e))
    run_short(2.0, True, True)
    p_stale, _ = run([s1], dur=0.0, reset=False, guard=False)
    tok6 = first7[6]["token"]
    rep["stale_unguarded_sample6_equals_outage_run"] = pred_diff({tok6: p_stale[tok6]}, {tok6: p_short_out[tok6]}) == 0.0
    d6 = pred_diff({tok6: p_stale[tok6]}, {tok6: pa_full[tok6]})
    rep["stale_unguarded_sample6_vs_fresh_clean_max_abs_diff"] = d6               # None = different row counts (shape mismatch), not a numeric value
    rep["stale_unguarded_sample6_vs_fresh_clean_shape_mismatch"] = d6 is None
    rep["outage_vs_clean_changed_samples_in_short_run"] = sum(differs(pred_diff({k: p_short_out[k]}, {k: pa_full[k]})) for k in p_short_out)
    # (b) interleaving
    seq = [i for pair in zip([x for x in all_infos if x["scene_token"] == s0], [x for x in all_infos if x["scene_token"] == s1]) for i in pair]
    g = SceneStateGuard()
    try:
        for i in seq:
            g.step(i["scene_token"], int(i["timestamp"]), require_memory=False)
        rep["interleave_control"] = dict(rejected=False)
    except StateResetError as e:
        rep["interleave_control"] = dict(rejected=True, message=str(e))
    rep["all_isolated"] = bool(all(v["max_abs_diff"] == 0.0 and v["fresh_at_start"] for v in rep["per_scene_alone_vs_continuous"].values())
                               and rep["fresh_B_twice_max_abs_diff"] == 0.0 and rep["B_after_A_vs_fresh_B_max_abs_diff"] == 0.0
                               and rep["stale_cache_control"].get("rejected_by_guard") is True
                               and rep["stale_unguarded_sample6_equals_outage_run"] and differs(rep["stale_unguarded_sample6_vs_fresh_clean_max_abs_diff"])
                               and rep["interleave_control"]["rejected"])
    json.dump(rep, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "per_scene_alone_vs_continuous"}, indent=1, default=str))
    print("per-scene max diffs:", [v["max_abs_diff"] for v in rep["per_scene_alone_vs_continuous"].values()])


if __name__ == "__main__":
    main(*sys.argv[1:5])
