"""Threshold calibration on the pilot (calibration) scenes from the model's CLEAN predictions (study plan p.3): maximise pooled
precision subject to recall >= 0.30 and >= MIN_PRED predictions in the region; frozen BEFORE any validation event evaluation.
Usage: python -m outage.calibrate <gt_pilot.pkl> <pilot_clean_preds.pkl> <out.json> [recall_target] [min_pred]"""
import json, pickle, sys, hashlib
import numpy as np
from .evaluator import select_threshold, TP, FP
from .event_stats import run_statuses


def main(gt_pkl, preds_pkl, out, recall_target=0.30, min_pred=100):
    gt = pickle.load(open(gt_pkl, "rb"))
    preds = pickle.load(open(preds_pkl, "rb"))
    assert set(preds) == set(gt), "calibration predictions must cover exactly the calibration samples"
    st = run_statuses(preds, gt)
    scores = np.concatenate([v["score"] for v in st.values()])
    status = np.concatenate([v["status"] for v in st.values()])
    n_gt = int(sum(g["gt"]["eligible"].sum() for g in gt.values()))
    sel = select_threshold(scores, status, n_gt, recall_target, min_pred)
    res = dict(sel, recall_target=recall_target, min_pred=min_pred, n_eligible_gt=n_gt, n_samples=len(gt),
               n_pred_total=int(len(scores)), native_score_floor=float(scores.min()),
               calibration_preds_sha256=hashlib.sha256(open(preds_pkl, "rb").read()).hexdigest(),
               calibration_scenes="pilot (train split): reserved calibration scenes, may have been seen by the released models")
    json.dump(res, open(out, "w"), indent=1)
    print(res)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], *[float(x) if i == 0 else int(x) for i, x in enumerate(sys.argv[4:6])])
