"""Paired scene bootstrap (study plan p.4): complete scenes are resampled WITH replacement; every replicate uses ONE set of scene weights for all three
models and all three durations (so clean/outage pairing, within-scene dependence and cross-model pairing are preserved); statistics are recomputed in
each replicate from saved per-scene statistics (no further inference). 95% intervals = percentile (2.5, 97.5). Fixed seed, B recorded."""
import numpy as np
from .data import BINS, POST, PHASES, DURATIONS, LOSS_MARGIN, MIN_GT
from .events import CATS, aggregate_onset, classify

B_DEFAULT, SEED = 10000, 535


def weights(S, B=B_DEFAULT, seed=SEED):
    """W (B, S): multiplicity of each scene in each replicate (rows sum to S)."""
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, S, size=(B, S))
    W = np.zeros((B, S))
    for b in range(B):
        W[b] = np.bincount(idx[b], minlength=S)
    return W


def ci(x, q=(2.5, 97.5)):
    x = np.asarray(x, float)
    if np.isnan(x).all():
        return [None, None]
    lo, hi = np.nanpercentile(x, q, axis=0)
    return [None if np.isnan(lo) else float(lo), None if np.isnan(hi) else float(hi)]


def ratio(num, den):
    with np.errstate(divide="ignore", invalid="ignore"):
        return num / den


def pooled(arr, W):
    """Pooled sums over scenes for each phase with weights W (R, S) -> dict of (R, P) arrays. W=ones((1,S)) gives the point estimate."""
    out = {k: W @ arr[k].astype(float) for k in ("n_gt", "tp", "tp_o", "fp", "fp_o", "n_pred_region", "n_pred_region_o")}
    out["n_scenes"] = W @ arr["present"].astype(float)
    out["recall_clean"] = ratio(out["tp"], out["n_gt"]); out["recall_outage"] = ratio(out["tp_o"], out["n_gt"])
    out["loss"] = out["recall_clean"] - out["recall_outage"]
    out["relative_loss"] = ratio(out["loss"], out["recall_clean"])                    # share of the clean recall that is lost (0 = none, 1 = all)
    out["precision_clean"] = ratio(out["tp"], out["tp"] + out["fp"]); out["precision_outage"] = ratio(out["tp_o"], out["tp_o"] + out["fp_o"])
    return out


def curve_summary(arr, W):
    pt, bs = pooled(arr, np.ones((1, arr["tp"].shape[0]))), pooled(arr, W)
    res = {}
    for j, ph in enumerate(PHASES):
        res[ph] = dict(n_gt=int(pt["n_gt"][0, j]), n_scenes_present=int(pt["n_scenes"][0, j]))
        for k in ("recall_clean", "recall_outage", "loss", "relative_loss", "precision_clean", "precision_outage"):
            v = pt[k][0, j]
            res[ph][k] = None if np.isnan(v) else float(v)
            res[ph][k + "_ci95"] = ci(bs[k][:, j])
        for k in ("fp", "fp_o"):
            res[ph]["false_positives_" + ("clean" if k == "fp" else "outage")] = int(pt[k][0, j])
    return res, bs


def aggregate_summary(arr, W, curve_bs):
    """Aggregate degradation rule (plan p.4): pooled loss DURING > 0.05 AND the scene-bootstrap interval excludes zero; otherwise the wording is
    'no statistically detected aggregate degradation' (absence of evidence, not of an effect). Aggregate-curve recovery onset (separate from per-event)."""
    j = PHASES.index("during")
    pt = pooled(arr, np.ones((1, arr["tp"].shape[0])))
    loss, lo_hi = float(pt["loss"][0, j]), ci(curve_bs["loss"][:, j])
    detected = bool(loss > LOSS_MARGIN and lo_hi[0] is not None and lo_hi[0] > 0.0)
    pidx = [PHASES.index(p) for p in POST]
    onset_pt = aggregate_onset(pt["loss"][:, pidx])[0]
    onset_bs = aggregate_onset(curve_bs["loss"][:, pidx])
    dist = {f"{b:.1f}": float(np.mean(onset_bs == b)) for b in BINS}; dist["none"] = float(np.mean(np.isnan(onset_bs)))
    return dict(pooled_loss_during=loss, ci95=lo_hi, rule="pooled loss during > 0.05 and 95% scene-bootstrap interval excludes 0",
                aggregate_degradation_detected=detected,
                wording="aggregate degradation detected (pooled loss > 5 points and interval excludes zero)" if detected else
                        "no statistically detected aggregate degradation (does not establish absence of an effect)",
                aggregate_recovery_onset_s=None if np.isnan(onset_pt) else float(onset_pt), aggregate_onset_bootstrap_distribution=dist)


def median_summary(m_pt, m_bs, q=(2.5, 97.5)):
    """Unconditional percentile interval for a recovery-time median. Unreached replicates (nan) are +infinity (order statistics, 'lower'/'higher' so no
    inf arithmetic). An endpoint that is +infinity is reported as None with `*_not_reached=True`. A conditional finite-replicate interval is supplemental."""
    m_bs = np.asarray(m_bs, float)
    x = np.where(np.isnan(m_bs), np.inf, m_bs)
    lo = float(np.percentile(x, q[0], method="lower")); hi = float(np.percentile(x, q[1], method="higher"))
    fin = m_bs[~np.isnan(m_bs)]
    cond = [float(np.percentile(fin, q[0])), float(np.percentile(fin, q[1]))] if len(fin) else [None, None]
    return dict(value=None if np.isnan(m_pt) else float(m_pt), reached=bool(not np.isnan(m_pt)), B=int(len(m_bs)),
                n_replicates_unreached=int(np.isnan(m_bs).sum()), fraction_of_replicates_reached=float(np.mean(~np.isnan(m_bs))),
                ci95=[None if not np.isfinite(lo) else lo, None if not np.isfinite(hi) else hi],
                ci95_lower_not_reached=bool(not np.isfinite(lo)), ci95_upper_not_reached=bool(not np.isfinite(hi)),
                ci95_conditional_on_reached_supplemental=cond)


def event_summary(cl, W):
    """cl = events.classify(...) for one model/duration. Per-event category counts and fractions with bootstrap intervals."""
    cat, onset = cl["cat"], cl["onset"]
    onehot = np.stack([(cat == k).astype(float) for k in range(len(CATS))], 1)                     # (S, 6)
    cnt_pt, cnt_bs = onehot.sum(0), W @ onehot
    S = len(cat)
    dam_pt, dam_bs = cnt_pt[2:].sum(), cnt_bs[:, 2:].sum(1)
    suf_pt, suf_bs = S - cnt_pt[0], S - cnt_bs[:, 0]
    rec_pt, rec_bs = cnt_pt[2] + cnt_pt[3], cnt_bs[:, 2] + cnt_bs[:, 3]
    out = dict(n_events=S, category_counts={c: int(cnt_pt[k]) for k, c in enumerate(CATS)},
               category_fractions={c: dict(value=float(cnt_pt[k] / S), ci95=ci(cnt_bs[:, k] / S)) for k, c in enumerate(CATS)},
               n_sufficient=int(suf_pt), n_damaged=int(dam_pt),
               damage_rate_among_sufficient=dict(value=float(ratio(dam_pt, suf_pt)), ci95=ci(ratio(dam_bs, suf_bs))),
               confirmed_onset_by_4s_among_damaged=dict(value=float(ratio(rec_pt, dam_pt)), ci95=ci(ratio(rec_bs, dam_bs)),
                                                         numerator=int(rec_pt), denominator=int(dam_pt), censored=int(dam_pt - rec_pt)))
    # follow-up completeness: unconfirmed events are almost all right-censored by missing / insufficient bins, a property of the shared schedule and the
    # keyframe spacing (identical for every model), NOT of the model. Recovery among events with COMPLETE follow-up is reported as a SELECTED subset (every bin has >= MIN_GT objects): it is an observed association, not an isolation of model behaviour.
    comp = ((cat >= 2) & (cl["n_broken_pairs"] == 0)).astype(float); comp_rec = comp * ((cat == 2) | (cat == 3))
    out["damaged_with_complete_followup"] = dict(n=int(comp.sum()), confirmed_onset=int(comp_rec.sum()),
                                                 fraction_confirmed=dict(value=float(ratio(comp_rec.sum(), comp.sum())), ci95=ci(ratio(W @ comp_rec, W @ comp))),
                                                 share_of_damaged_with_incomplete_followup=dict(value=float(1 - ratio(comp.sum(), dam_pt)), ci95=ci(1 - ratio(W @ comp, dam_bs))))
    # cumulative confirmed-onset curve over damaged events (censored events stay in the denominator: never average only recovered events)
    cum_pt, cum_bs = {}, {}
    dmask = (cat >= 2).astype(float)
    for b in BINS[:-1]:
        hit = ((onset <= b + 1e-9) & (cat >= 2)).astype(float)
        cum_pt[f"{b:.1f}"] = dict(value=float(ratio(hit.sum(), dmask.sum())), ci95=ci(ratio(W @ hit, W @ dmask)))
    out["cumulative_confirmed_onset_fraction_by_time_s"] = cum_pt
    # recovery-time median, reported ONLY if reached (more than half of the damaged events have a confirmed onset; censored events count as not reached)
    def med(wvec):
        d = float(wvec @ dmask)
        if d == 0:
            return np.nan
        order = np.argsort(np.where(np.isnan(onset), np.inf, onset), kind="stable")
        o, w, m = np.where(np.isnan(onset), np.inf, onset)[order], wvec[order], dmask[order]
        c = np.cumsum(w * m) / d
        k = np.searchsorted(c, 0.5)
        return o[k] if k < len(o) and np.isfinite(o[k]) else np.nan
    m_pt = med(np.ones(S))
    m_bs = np.array([med(W[b]) for b in range(W.shape[0])])                                      # ALL B replicates (correction 1; was every 10th)
    out["median_recovery_time_s"] = median_summary(m_pt, m_bs)
    out["median_recovery_time_s"]["note"] = ("censored/unconfirmed damaged events are kept in the denominator; a median exists only if more than half of damaged "
                                             "events have a confirmed onset. Replicates in which the median is not reached stay in the distribution as +infinity: "
                                             "an upper endpoint beyond the last confirmable bin is reported as 'not reached', never dropped")
    return out, dict(damaged=dmask, onset=onset)


def common_event_summary(cls, W):
    """Events damaged in ALL models (same scene & duration): recovery on a common population. cls: {model: classify(...)}; returns summary + pairwise paired differences."""
    models = list(cls)
    dam = np.stack([cls[m]["cat"] >= 2 for m in models], 0)
    common = dam.all(0).astype(float)
    n_pt, n_bs = common.sum(), W @ common
    out = dict(n_common_damaged_events=int(n_pt), models=models, per_model={}, pairwise={})
    rec = {m: ((cls[m]["cat"] == 2) | (cls[m]["cat"] == 3)).astype(float) * common for m in models}
    for m in models:
        out["per_model"][m] = dict(confirmed_onset_by_4s=dict(value=float(ratio(rec[m].sum(), n_pt)), ci95=ci(ratio(W @ rec[m], n_bs)), numerator=int(rec[m].sum()), denominator=int(n_pt)))
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            d_pt = ratio(rec[a].sum(), n_pt) - ratio(rec[b].sum(), n_pt)
            d_bs = ratio(W @ rec[a], n_bs) - ratio(W @ rec[b], n_bs)
            out["pairwise"][f"{a}_minus_{b}"] = dict(onset_fraction_difference=float(d_pt), ci95=ci(d_bs), interval_excludes_zero=bool(ci(d_bs)[0] is not None and (ci(d_bs)[0] > 0 or ci(d_bs)[1] < 0)))
    return out


def pairwise_summary(arrs, cls, W):
    """Paired differences between models (same scenes, same bootstrap weights): pooled loss during the outage and damage rate among sufficient events."""
    models = list(arrs)
    j = PHASES.index("during")
    pooled_bs = {m: pooled(arrs[m], W)["loss"][:, j] for m in models}
    pooled_pt = {m: float(pooled(arrs[m], np.ones((1, arrs[m]["tp"].shape[0])))["loss"][0, j]) for m in models}
    dam = {m: (cls[m]["cat"] >= 2).astype(float) for m in models}
    suf = {m: (cls[m]["cat"] >= 1).astype(float) for m in models}
    out = {}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            dl = pooled_bs[a] - pooled_bs[b]
            dr_pt = ratio(dam[a].sum(), suf[a].sum()) - ratio(dam[b].sum(), suf[b].sum())
            dr_bs = ratio(W @ dam[a], W @ suf[a]) - ratio(W @ dam[b], W @ suf[b])
            out[f"{a}_minus_{b}"] = dict(pooled_loss_during_difference=dict(value=pooled_pt[a] - pooled_pt[b], ci95=ci(dl)),
                                         damage_rate_difference=dict(value=float(dr_pt), ci95=ci(dr_bs)))
    return out
