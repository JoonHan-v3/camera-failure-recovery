"""Finite-window temporal-contamination check (study plan, 'Temporal contamination': for finite-window inputs identify when the last
corrupted image leaves history and check that predictions then return to the clean run within numerical tolerance).
SparseBEV (test mode) sees the current keyframe + the 7 previous keyframes (indices j-7..j, clamped at the scene start). A sample j is
CONTAMINATED iff one of its window keyframes has a black CAM_FRONT image. Expectation (no recurrent state exists):
  * every NON-contaminated sample (before the outage and from k_last + 8 onwards) equals the clean run within tolerance;
  * every contaminated sample differs from the clean run (the outage reached the model through the cache/history).
Everything is derived from the frozen schedule + raw timestamps (gt_tables), then compared with the saved predictions.
Usage: python -m outage.window_check <gt.pkl> <schedule.json> <runs_dir> <clean> <out.json> <outage_run:duration_s>..."""
import json, os, pickle, sys
import numpy as np
from .schedule import load_schedule, US
from .verify_runs import pred_diff

WINDOW = 8          # current + 7 previous keyframes (num_frames=8, test interval 6 sweeps = one keyframe)
TOL = 1e-4


def contaminated_samples(gt_tab, sched, dur):
    by_scene = {}
    for s, g in gt_tab.items():
        by_scene.setdefault(g["scene"], []).append(s)
    out, leave = {}, {}
    for scene, ss in by_scene.items():
        ss = sorted(ss, key=lambda s: gt_tab[s]["ts_lidar_us"])
        o = sched[scene]
        t0, t1 = o.window(dur)
        bad = [t0 <= gt_tab[s]["ts_front_us"] < t1 for s in ss]
        k_last = max((k for k, b in enumerate(bad) if b), default=None)
        for j, s in enumerate(ss):
            win = {max(0, j - m) for m in range(WINDOW)}
            out[s] = any(bad[k] for k in win)
        if k_last is not None:
            jl = k_last + WINDOW
            leave[scene] = dict(k_last=k_last, first_clean_window_index=jl, n_samples=len(ss),
                                leave_offset_s_after_restoration=(gt_tab[ss[jl]]["ts_lidar_us"] - t1) / US if jl < len(ss) else None)
    return out, leave


def main(gt_pkl, sched_path, runs, clean, out, *specs):
    gt = pickle.load(open(gt_pkl, "rb"))
    sched, _, _ = load_schedule(sched_path)
    C = pickle.load(open(os.path.join(runs, clean, "preds.pkl"), "rb"))
    rep = dict(window_keyframes=WINDOW, tol=TOL, runs={})
    for spec in specs:
        name, d = spec.split(":"); d = float(d)
        O = pickle.load(open(os.path.join(runs, name, "preds.pkl"), "rb"))
        cont, leave = contaminated_samples(gt, sched, d)
        nc_diffs, c_diffs = [], []
        for s, flag in cont.items():
            diff = pred_diff({s: C[s]}, {s: O[s]})
            (c_diffs if flag else nc_diffs).append(diff)
        nc, cc = np.array(nc_diffs), np.array(c_diffs)
        offs = [v["leave_offset_s_after_restoration"] for v in leave.values() if v["leave_offset_s_after_restoration"] is not None]
        rep["runs"][name] = dict(duration_s=d, n_noncontaminated=len(nc), noncontaminated_max_abs_diff=float(nc.max()),
                                 noncontaminated_all_within_tol=bool((nc <= TOL).all()), n_contaminated=len(cc),
                                 contaminated_changed=int((cc > TOL).sum()), contaminated_unchanged=int((cc <= TOL).sum()),
                                 scenes_with_leave_time=len(offs), leave_offset_s_after_restoration=dict(
                                     min=float(np.min(offs)), median=float(np.median(offs)), max=float(np.max(offs))) if offs else None,
                                 scenes_whose_window_never_clears_before_scene_end=sum(1 for v in leave.values() if v["leave_offset_s_after_restoration"] is None))
    rep["PASS"] = bool(all(r["noncontaminated_all_within_tol"] for r in rep["runs"].values()))
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:6], *sys.argv[6:])
