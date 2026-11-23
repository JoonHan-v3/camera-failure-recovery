"""Per-class recall loss and the same-object diagnostic, with the same paired scene bootstrap (same weight matrix as analysis.py).
Usage (from RecoveryAnalysis/): python -m recovery.deep_dive [B]  -> artifacts/analysis/classes_sameobject.json"""
import json, os, sys
import numpy as np
from . import stats, samplelevel as sl
from .data import MODELS, DURATIONS, PHASES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHOW = ["pre", "during", "b0.0", "b0.5", "b1.0"]


def per_class(b, W):
    out = {}
    for d in DURATIONS:
        r = b["by_duration"][d]; out[str(d)] = {}
        for ph in SHOW:
            j = PHASES.index(ph)
            n = np.einsum("bs,sc->bc", W, r["n_c"][:, j, :].astype(float)); tc = np.einsum("bs,sc->bc", W, r["tp_c"][:, j, :].astype(float)); to = np.einsum("bs,sc->bc", W, r["tp_oc"][:, j, :].astype(float))
            n1, c1, o1 = r["n_c"][:, j, :].sum(0), r["tp_c"][:, j, :].sum(0), r["tp_oc"][:, j, :].sum(0)
            rec = {}
            for k, cname in enumerate(sl.CLASSES):
                if n1[k] == 0:
                    rec[cname] = dict(n_gt=0, loss=None, loss_ci95=[None, None]); continue
                loss = c1[k] / n1[k] - o1[k] / n1[k]
                rec[cname] = dict(n_gt=int(n1[k]), recall_clean=float(c1[k] / n1[k]), recall_outage=float(o1[k] / n1[k]), loss=float(loss),
                                  loss_ci95=stats.ci(stats.ratio(tc[:, k] - to[:, k], n[:, k])), small_denominator=bool(n1[k] < 50))
            out[str(d)][ph] = rec
    return out


def same_object(b, W):
    out = {}
    for d in DURATIONS:
        X = b["by_duration"][d]["sameobject"].astype(float); k = {n: i for i, n in enumerate(sl.SO_KEYS)}
        pt, bs = X.sum(0), W @ X
        def frac(num, den):
            return dict(value=float(stats.ratio(pt[k[num]], pt[k[den]])), ci95=stats.ci(stats.ratio(bs[:, k[num]], bs[:, k[den]])),
                        numerator=int(pt[k[num]]), denominator=int(pt[k[den]]))
        out[str(d)] = dict(
            counts={n: int(pt[i]) for n, i in k.items()},
            pair="last during-outage sample -> first STRICTLY LATER sample with a restored CAM_FRONT exposure and a post-restoration bin (same-sample / still-black pairs are unavailable, never recovery observations); objects = eligible GT instance tokens",
            event_accounting={n: int(pt[k[n]]) for n in ("events_with_during_sample", "events_with_last_during_and_later_restored", "events_unavailable_same_sample_as_bin0", "events_unavailable_bin0_front_still_black", "events_unavailable_no_restored_later_sample", "events_after_is_bin0", "events_after_is_later_bin")},
            mean_after_bin_s=float(stats.ratio(pt[k["sum_after_bin_tenths_s"]] / 10.0, pt[k["events_with_last_during_and_later_restored"]])),
            continuing_clean_detected_outage_missed_during_then_detected_after=frac("cont_missed_d_detected_after", "cont_missed_d_detected_after") if False else None,
            recovered_among_missed_during=dict(
                value=float(stats.ratio(pt[k["cont_missed_d_detected_after"]], pt[k["cont_missed_d_detected_after"]] + pt[k["cont_missed_d_missed_after"]])),
                ci95=stats.ci(stats.ratio(bs[:, k["cont_missed_d_detected_after"]], bs[:, k["cont_missed_d_detected_after"]] + bs[:, k["cont_missed_d_missed_after"]])),
                numerator=int(pt[k["cont_missed_d_detected_after"]]), denominator=int(pt[k["cont_missed_d_detected_after"]] + pt[k["cont_missed_d_missed_after"]]),
                reading="of continuing objects the clean run detected at both samples and the outage run missed during the outage: fraction detected again at the restored later observation"),
            persisting_miss_fraction_of_continuing_clean_detected=frac("cont_missed_d_missed_after", "continuing_clean_detected_both"),
            lost_after_restoration_fraction=frac("cont_detected_d_missed_after", "continuing_clean_detected_both"),
            entering_objects_missed_by_outage_given_clean_detected=frac("entering_outage_missed", "entering_clean_detected"))
        out[str(d)].pop("continuing_clean_detected_outage_missed_during_then_detected_after")
    return out


def main(B=stats.B_DEFAULT):
    gt, sched = sl.load_gt(), sl.load_schedule(os.path.join(sl.S3, "artifacts/schedule_trainval_val.json"))
    W = stats.weights(150, B)
    rep = dict(bootstrap=dict(B=B, seed=stats.SEED), classes=list(sl.CLASSES), per_class={}, same_object={})
    for m in MODELS:
        b = sl.build_model(m, gt, sched)
        rep["per_class"][m] = per_class(b, W); rep["same_object"][m] = same_object(b, W)
    json.dump(rep, open(os.path.join(HERE, "artifacts/analysis/classes_sameobject.json"), "w"), indent=1)
    for m in MODELS:
        for d in DURATIONS:
            s = rep["same_object"][m][str(d)]; r = s["recovered_among_missed_during"]
            print(m, d, "continuing(clean-det both)", s["counts"]["continuing_clean_detected_both"], "recovered|missed-during %.2f (%d/%d)" % (r["value"], r["numerator"], r["denominator"]))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else stats.B_DEFAULT)
