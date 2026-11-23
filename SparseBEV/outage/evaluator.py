"""Custom front-region evaluator (study plan p.3-4). Pure numpy; model-independent (common prediction format
[x,y,z,w,l,h,yaw,vx,vy,label,score], z = box bottom, LiDAR frame, native decoded predictions BEFORE official range filtering).

Rules implemented (and unit-tested on controlled examples in tests/test_evaluator.py):
 * Eligible GT: one of the ten detection classes, centre in the CAM_FRONT region (in front of the camera, inside the original
   1600x900 image, ground-plane distance <= 50 m from the ego origin) and >= 1 annotated LiDAR or radar point (gt_tables.py).
 * Matching (per sample, class-aware, one-to-one, ground-plane centre distance <= 2.0 m, inclusive): predictions in descending score
   (ties: input order); each is assigned to the NEAREST UNMATCHED ELIGIBLE GT of its class (ties: lowest GT index). Otherwise it is
   checked against IGNORED same-class GT (any same-class GT that is not eligible, nearest-within-2 m test, matched or not): such a
   prediction is IGNORED (neither TP nor FP). Remaining unmatched predictions count as FALSE POSITIVES only if their own centre lies
   in the region; the rest are outside the evaluated population.
 * Threshold prefix property: greedy matching in descending score means the matching for threshold tau is exactly the matching of
   the prefix {score >= tau}, so one pass gives the statuses for every tau.
 * Threshold selection (calibration scenes): maximise pooled precision TP/(TP+FP) subject to recall >= recall_target and
   n_pred >= min_pred; precision undefined (TP+FP = 0) never wins; ties -> higher recall -> lower tau (deterministic).
 * Damage / recovery: loss L = R_clean - R_outage per bin on the same eligible objects; damaged iff pooled loss during the outage
   > 0.05 and the frozen minimum object count is met; recovery onset = first post-restoration bin with loss <= 0.05 whose NEXT
   ADJACENT scheduled bin also has loss <= 0.05; both bins need sufficient data; a missing / insufficient bin breaks the pair.
"""
import numpy as np

TP, FP, IGN, OUT = 1, 2, 3, 0          # per-prediction status codes
MATCH_DIST_M = 2.0
LOSS_MARGIN = 0.05
BINS_S = tuple(i * 0.5 for i in range(10))     # post-restoration bins 0.0 ... 4.5 s


def match_sample(pred, pred_in_region, pred_xy, gt_cls, gt_xy, gt_eligible, dist=MATCH_DIST_M):
    """pred: (n,>=11) rows (score, label used); pred_in_region: (n,) bool; pred_xy: (n,2) and gt_xy: (m,2) GEOMETRIC-centre XY in the SAME ground
    plane (sample-time ego frame, see geometry version 2). gt_*: arrays over ALL same-sample GT of the ten classes.
    Returns (order, status, gt_index) for predictions in descending-score order: status code per prediction, matched eligible GT
    index (-1 if none). Threshold-free; see the prefix property above."""
    pred = np.asarray(pred, float).reshape(-1, 11)
    n = len(pred)
    order = np.argsort(-pred[:, 10], kind="stable")
    status = np.full(n, OUT, int)
    gidx = np.full(n, -1, int)
    gt_cls, gt_eligible = np.asarray(gt_cls, int), np.asarray(gt_eligible, bool)
    gt_xy = np.asarray(gt_xy, float).reshape(-1, 2)
    pred_xy = np.asarray(pred_xy, float).reshape(-1, 2)
    matched = np.zeros(len(gt_cls), bool)
    for k, i in enumerate(order):
        c = int(round(pred[i, 9]))
        d = np.hypot(*(gt_xy - pred_xy[i]).T) if len(gt_xy) else np.zeros(0)
        cand = np.flatnonzero((gt_cls == c) & gt_eligible & ~matched & (d <= dist))
        if len(cand):
            j = cand[np.argmin(d[cand])]          # argmin returns the first minimum -> ties: lowest GT index
            matched[j] = True
            status[k], gidx[k] = TP, j
            continue
        if np.any((gt_cls == c) & ~gt_eligible & (d <= dist)):
            status[k] = IGN
        elif pred_in_region[i]:
            status[k] = FP
        else:
            status[k] = OUT
    return order, status, gidx


def counts_at(scores_sorted, status, thr):
    """Counts for threshold thr on statuses produced by match_sample (scores_sorted descending)."""
    m = scores_sorted >= thr
    s = status[m]
    return dict(tp=int((s == TP).sum()), fp=int((s == FP).sum()), ign=int((s == IGN).sum()), n_pred_region=int((s == TP).sum() + (s == FP).sum()))


def recall(tp, n_gt):
    return None if n_gt == 0 else tp / n_gt


def precision(tp, fp):
    return None if tp + fp == 0 else tp / (tp + fp)


def select_threshold(scores, status, n_gt, recall_target=0.30, min_pred=100):
    """Pooled calibration selection. scores/status: concatenated over all calibration samples (any order); n_gt: pooled eligible GT.
    Candidates = every distinct score. Returns dict(tau, precision, recall, n_pred, tp, fp, floor_score) or dict(tau=None, reason)."""
    scores, status = np.asarray(scores, float), np.asarray(status, int)
    if len(scores) == 0 or n_gt == 0:
        return dict(tau=None, reason="no predictions or no eligible GT in calibration set")
    o = np.argsort(-scores, kind="stable")
    sc, st = scores[o], status[o]
    ctp, cfp = np.cumsum(st == TP), np.cumsum(st == FP)
    last = np.r_[sc[1:] != sc[:-1], True]            # a threshold includes whole tie groups: evaluate at the group's last row
    best = None
    for k in np.flatnonzero(last):
        tp, fp = int(ctp[k]), int(cfp[k])
        if tp + fp == 0 or tp + fp < min_pred:      # undefined precision / too few predictions cannot win
            continue
        r = tp / n_gt
        if r < recall_target:
            continue
        key = (tp / (tp + fp), r, -sc[k])
        if best is None or key > best[0]:
            best = (key, dict(tau=float(sc[k]), precision=tp / (tp + fp), recall=r, n_pred=tp + fp, tp=tp, fp=fp))
    if best is None:
        return dict(tau=None, reason=f"no threshold reaches recall >= {recall_target} with >= {min_pred} predictions")
    return best[1]


def assign_bins(offsets, bins=BINS_S, tol_s=0.25):
    """Same one-to-one convention as audit.missing_bins. offsets: list of (sample, offset_s >= 0) for ONE scene. Returns {bin: sample}."""
    pairs = sorted((abs(h - b), b, h, i) for b in bins for i, (_, h) in enumerate(offsets) if abs(h - b) <= tol_s + 1e-9)
    used_b, used_s, out = set(), set(), {}
    for _, b, h, i in pairs:
        if b not in used_b and i not in used_s:
            used_b.add(b); used_s.add(i); out[b] = offsets[i][0]
    return out


def damaged(loss_during, n_gt_during, min_gt_during, margin=LOSS_MARGIN):
    """None = insufficient data (too few eligible objects during the outage); else bool (loss strictly above margin)."""
    if loss_during is None or n_gt_during < min_gt_during:
        return None
    return loss_during > margin


def recovery_onset(loss_by_bin, ok_by_bin, bins=BINS_S, margin=LOSS_MARGIN):
    """loss_by_bin/ok_by_bin: dicts bin -> loss / sufficient-data flag (missing bin = absent or ok False). Returns the first bin b with
    loss <= margin whose next ADJACENT scheduled bin is also present, sufficient and <= margin; None if never confirmed.
    The last bin can never be an onset (nothing confirms it); a missing or insufficient bin breaks the pair."""
    def good(b):
        return b in loss_by_bin and ok_by_bin.get(b, False) and loss_by_bin[b] is not None and loss_by_bin[b] <= margin + 1e-12
    for i, b in enumerate(bins[:-1]):
        if good(b) and good(bins[i + 1]):
            return b
    return None
