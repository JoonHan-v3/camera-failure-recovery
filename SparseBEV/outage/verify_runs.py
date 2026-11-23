"""Cross-run verification of saved StreamPETR runs (no GPU): paired input audits, independent
recomputation of which front images must be black, clean-twice repeatability, prefix prediction identity,
restoration offsets. Usage: python -m outage.verify_runs <runs_dir> <tag> <out.json> [clean_a clean_b d05 d10 d20]"""
import json, os, pickle, sys
import numpy as np
from .audit import load_audit, compare_runs, CAUSAL_TOL_US, CAUSAL_POLICY
from .corruption import is_corrupted
from .nusc_ref import NuscRef, CAMS
from .schedule import CAMERA, US


def pred_diff(a, b):
    """Max |a-b| over matching samples. Any schema problem (different samples, shapes, non-2D, wrong column count,
    non-finite values, labels that are not integral) returns inf, never a finite 'equal'."""
    if a.keys() != b.keys():
        return float("inf")
    d = 0.0
    for k in a:
        x, y = np.asarray(a[k], float), np.asarray(b[k], float)
        if x.shape != y.shape or x.ndim != 2 or (x.size and x.shape[1] != 11):
            return float("inf")
        if x.size:
            if not (np.isfinite(x).all() and np.isfinite(y).all()):
                return float("inf")
            if not (np.array_equal(x[:, 9], np.round(x[:, 9])) and np.array_equal(y[:, 9], np.round(y[:, 9]))):
                return float("inf")
            d = max(d, float(np.abs(x - y).max()))
    return d


def main(runs, version, out, names=("a", "b", "d05", "d10", "d20")):
    ref = NuscRef(os.path.expanduser("~/storage/nuscenes"), version)  # raw JSON tables
    A = {n: load_audit(os.path.join(runs, n, "audit.json")) for n in names}
    for a in A.values():
        a.causal_tolerance_us = CAUSAL_TOL_US          # re-apply the FROZEN policy to every saved audit
    P = {n: pickle.load(open(os.path.join(runs, n, "preds.pkl"), "rb")) for n in names}
    rep = dict(version=version, runs=list(names), causal_policy=CAUSAL_POLICY)
    rep["max_cam_minus_eval_skew_us"] = max(r["ts_us"] - r["eval_ts_us"] for a in A.values() for r in a.records)
    rep["min_cam_minus_eval_skew_us"] = min(r["ts_us"] - r["eval_ts_us"] for a in A.values() for r in a.records)
    rep["clean_twice_max_abs_diff"] = pred_diff(P[names[0]], P[names[1]])
    rep["clean_twice_identical_samples"] = len(P[names[0]])
    rep["clean_audit_a"], rep["clean_audit_b"] = A[names[0]].status(), A[names[1]].status()
    rep["clean_a_vs_b_input_errors"] = len(compare_runs(A[names[0]], A[names[1]]))
    clean = A[names[0]]
    rep["conditions"] = {}
    for n in names[2:]:
        o = A[n]; d = o.duration_s
        c = dict(duration_s=d, status=o.status(), readiness=o.readiness(),
                 paired_input_errors=compare_runs(clean, o)[:10], n_paired_errors=len(compare_runs(clean, o)))
        # independent recomputation from raw nuScenes tables of what must be black
        exp_black, got_black, per_scene = set(), set(), {}
        for tok, so in o.schedule.items():
            for s in ref.scenes[tok]["samples"]:
                ts = int(ref.kf[(s, CAMERA)]["timestamp"])
                if is_corrupted(CAMERA, ts, so.t_start_us, d):
                    exp_black.add((tok, s, CAMERA, ts))
        for r in o.records:
            if r["corrupted"]:
                got_black.add((r["scene"], r["sample"], r["cam"], r["ts_us"]))
        c["black_inputs_expected_vs_logged_equal"] = exp_black == got_black
        c["n_black_images"] = len(got_black)
        for tok, so in o.schedule.items():
            per_scene[tok] = sum(1 for g in got_black if g[0] == tok)
        c["black_images_per_scene_min_max"] = [min(per_scene.values()), max(per_scene.values())]
        c["all_black_pixels_verified"] = all(r["all_black"] for r in o.records if r["corrupted"])
        c["other_cameras_corrupted"] = sum(1 for r in o.records if r["corrupted"] and r["cam"] != CAMERA)
        # predictions: identical before the outage, different after
        pre_same = pre_n = post_diff = post_n = 0
        for tok, so in o.schedule.items():
            for s in ref.scenes[tok]["samples"]:
                if s not in P[n] or s not in P[names[0]]:
                    continue
                dd = pred_diff({s: P[names[0]][s]}, {s: P[n][s]})
                if ref.lidar_ts(s) <= so.t_start_us - 0 and int(ref.kf[(s, CAMERA)]["timestamp"]) < so.t_start_us:
                    pre_n += 1; pre_same += dd == 0.0
                elif int(ref.kf[(s, CAMERA)]["timestamp"]) >= so.t_start_us:
                    post_n += 1; post_diff += dd > 0.0
        c["prefix_samples"], c["prefix_predictions_identical"] = pre_n, pre_same
        c["post_start_samples"], c["post_start_predictions_changed"] = post_n, post_diff
        c["missing_follow_up_bins"] = o.missing_bins()
        offs = o.restoration_offsets()
        c["restoration_offsets_s_example"] = {k: [round(x[1], 3) for x in v[:12]] for k, v in list(offs.items())[:1]}
        rep["conditions"][n] = c
    json.dump(rep, open(out, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in rep.items() if k != "conditions"}, indent=1, default=str))
    for n, c in rep["conditions"].items():
        print(n, {k: c[k] for k in ("duration_s", "status", "readiness", "n_paired_errors", "black_inputs_expected_vs_logged_equal",
              "n_black_images", "black_images_per_scene_min_max", "all_black_pixels_verified", "other_cameras_corrupted",
              "prefix_samples", "prefix_predictions_identical", "post_start_samples", "post_start_predictions_changed")}, "missing_bins", len(c["missing_follow_up_bins"]))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *([sys.argv[4:]] if len(sys.argv) > 4 else []))
