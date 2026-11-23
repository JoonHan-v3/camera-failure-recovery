"""Recovery analysis driver: pooled recall/loss curves, aggregate degradation rule, per-event categories, confirmed-onset curves, common-event
comparison and pairwise paired differences, all with paired scene-bootstrap 95% intervals. Inputs: Member 3's per-scene statistics (frozen thresholds,
frozen event rules). Output: artifacts/analysis/recovery_summary.json + per_event_categories.csv.
Usage (from RecoveryAnalysis/): python -m recovery.analysis [B]"""
import csv, json, os, sys, time
import numpy as np
from . import stats
from .data import MODELS, DURATIONS, PHASES, load_all, STATS, LOSS_MARGIN, MIN_GT
from .events import classify, CATS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "artifacts/analysis")


def crosscheck_member3(ms, curves):
    """Pooled point estimates recomputed here (from the CSV) must equal Member 3's own pooled summaries (summary_<tag>.json): max abs diff."""
    worst = 0.0
    for m, v in MODELS.items():
        ref = json.load(open(os.path.join(STATS, f"summary_{v['tag']}.json")))["summary"]
        for d in DURATIONS:
            for ph, rec in ref[str(d)]["pooled"].items():
                for k in ("recall_clean", "recall_outage", "loss"):
                    a, b = curves[m][str(d)][ph][k], rec[k]
                    if a is None or b is None:
                        assert a is None and b is None, (m, d, ph, k)
                        continue
                    worst = max(worst, abs(a - b))
    return worst


def inputs_manifest():
    from .data import S3, PROJECTS, sha256
    files = {}
    for m, v in MODELS.items():
        for pre in ("per_scene_bins_", "summary_", "per_sample_"):
            ext = ".pkl" if pre == "per_sample_" else (".csv" if pre.startswith("per_scene") else ".json")
            files[f"event_stats/{pre}{v['tag']}{ext}"] = os.path.join(STATS, f"{pre}{v['tag']}{ext}")
        files[f"calibration/threshold_{m}.json"] = os.path.join(S3, f"artifacts/calibration/threshold_{m}.json")
    files.update({"gt_val.pkl": os.path.join(S3, "artifacts/gt_val.pkl"), "schedule_trainval_val.json": os.path.join(S3, "artifacts/schedule_trainval_val.json"),
                  "EVENT_RULES_DECLARATION.txt": os.path.join(S3, "artifacts/provenance/EVENT_RULES_DECLARATION.txt")})
    return {k: dict(path=os.path.relpath(p, PROJECTS), sha256=sha256(p)) for k, p in files.items()}


def main(B=stats.B_DEFAULT):
    os.makedirs(OUT, exist_ok=True)
    ms = load_all()
    S = len(ms["sparsebev"]["scenes"])
    W = stats.weights(S, B)
    rep = dict(bootstrap=dict(B=B, seed=stats.SEED, resample="complete scenes with replacement; one weight matrix shared by all models and durations (paired)",
                              interval="percentile 2.5/97.5"), n_scenes=S, rules=dict(loss_margin=LOSS_MARGIN, min_gt=MIN_GT, bins_s=[0.5 * i for i in range(10)]),
               curves={}, aggregate={}, events={}, common_events={}, pairwise={}, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    rows = []
    cls_all = {}
    for m in MODELS:
        rep["curves"][m], rep["aggregate"][m], rep["events"][m] = {}, {}, {}
        for d in DURATIONS:
            arr = ms[m]["by_duration"][d]
            cur, cbs = stats.curve_summary(arr, W)
            rep["curves"][m][str(d)] = cur
            rep["aggregate"][m][str(d)] = stats.aggregate_summary(arr, W, cbs)
            cl = classify(arr); cls_all[(m, d)] = cl
            ev, _ = stats.event_summary(cl, W)
            rep["events"][m][str(d)] = ev
            for s, scene in enumerate(ms[m]["scenes"]):
                rows.append(dict(model=m, duration_s=d, scene=scene, category=CATS[cl["cat"][s]], n_gt_during=int(cl["n_gt_during"][s]),
                                 loss_during="" if np.isnan(cl["loss_during"][s]) else round(float(cl["loss_during"][s]), 6),
                                 first_observed_bin_s="" if np.isnan(cl["first_observed"][s]) else cl["first_observed"][s],
                                 confirmed_onset_s="" if np.isnan(cl["onset"][s]) else cl["onset"][s], broken_adjacent_pairs=int(cl["n_broken_pairs"][s])))
    for d in DURATIONS:
        cls = {m: cls_all[(m, d)] for m in MODELS}
        rep["common_events"][str(d)] = stats.common_event_summary(cls, W)
        rep["pairwise"][str(d)] = stats.pairwise_summary({m: ms[m]["by_duration"][d] for m in MODELS}, cls, W)
    rep["crosscheck_member3_pooled_max_abs_diff"] = crosscheck_member3(ms, rep["curves"])
    rep["inputs"] = inputs_manifest()
    json.dump(rep, open(os.path.join(OUT, "recovery_summary.json"), "w"), indent=1)
    with open(os.path.join(OUT, "per_event_categories.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0])); w.writeheader(); w.writerows(rows)
    print("crosscheck vs Member 3 pooled summaries, max |diff| =", rep["crosscheck_member3_pooled_max_abs_diff"])
    for m in MODELS:
        for d in DURATIONS:
            a, e = rep["aggregate"][m][str(d)], rep["events"][m][str(d)]
            f = lambda x: "n/a" if x is None else f"{x:.3f}"
            print(f"{m:10s} {d}s loss_during {f(a['pooled_loss_during'])} CI {[f(x) for x in a['ci95']]} detected={a['aggregate_degradation_detected']} "
                  f"agg_onset={a['aggregate_recovery_onset_s']} | events suff {e['n_sufficient']} damaged {e['n_damaged']} "
                  f"onset<=4s {e['confirmed_onset_by_4s_among_damaged']['numerator']}/{e['confirmed_onset_by_4s_among_damaged']['denominator']} "
                  f"cats {list(e['category_counts'].values())}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else stats.B_DEFAULT)
