"""Per-event (scene x duration) damage / recovery classification — the study plan's rules, FROZEN before validation (EVENT_RULES_DECLARATION):
 * loss L(bin) = R_clean(bin) - R_outage(bin) on the same eligible objects of the event's own samples;
 * damaged iff pooled loss DURING the outage > 0.05 and >= MIN_GT eligible objects during the outage; fewer objects -> insufficient data;
 * a post-restoration bin is usable iff it exists (assigned sample, one-to-one rule) and has >= MIN_GT eligible objects;
 * recovery ONSET = first post-restoration bin with L <= 0.05 whose next ADJACENT scheduled bin is also usable with L <= 0.05 (a missing or
   insufficient bin breaks the pair); the 4.5 s bin only confirms, so onsets are <= 4.0 s;
 * categories (plan p.4): 0 insufficient data | 1 no measured degradation | 2 damaged, recovery at the first observed bin |
   3 damaged, later recovery | 4 damaged, no confirmed recovery with COMPLETE follow-up (every adjacent pair usable, none confirmed) |
   5 damaged, no confirmed recovery with INCOMPLETE follow-up (at least one pair broken by a missing/insufficient bin: right-censored, unknown;
   NEVER read as permanent failure).
Independent re-implementation (does not import Member 3's evaluator); tests/test_events.py cross-checks it against evaluator.recovery_onset."""
import numpy as np
from .data import BINS, POST, PHASES, LOSS_MARGIN, MIN_GT

CATS = ["insufficient_data", "no_measured_degradation", "damaged_recovered_first_observed_bin", "damaged_recovered_later",
        "damaged_no_confirmed_recovery_complete_followup", "damaged_unconfirmed_incomplete_followup"]
POST_IDX = [PHASES.index(p) for p in POST]
EPS = 1e-12


def classify(arr, margin=LOSS_MARGIN, min_gt=MIN_GT):
    """arr: per-duration field arrays (S, P). Returns dict of per-scene arrays: cat (S,), onset (S,) float (nan if none), loss_during, n_gt_during,
    usable (S,10) bool, loss_bins (S,10), first_observed (S,) float (nan if none), n_broken_pairs (S,)."""
    S = arr["tp"].shape[0]
    j = PHASES.index("during")
    n_d = arr["n_gt"][:, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_d = (arr["tp"][:, j] - arr["tp_o"][:, j]) / n_d
        n_b = arr["n_gt"][:, POST_IDX]
        loss_b = (arr["tp"][:, POST_IDX] - arr["tp_o"][:, POST_IDX]) / n_b
    usable = arr["present"][:, POST_IDX] & (n_b >= min_gt)
    good = usable & (loss_b <= margin + EPS)
    cat = np.zeros(S, int)
    onset = np.full(S, np.nan)
    first_obs = np.full(S, np.nan)
    broken = np.zeros(S, int)
    for s in range(S):
        if n_d[s] < min_gt:
            cat[s] = 0
            continue
        if not loss_d[s] > margin:
            cat[s] = 1
            continue
        obs = np.flatnonzero(usable[s])
        first_obs[s] = BINS[obs[0]] if len(obs) else np.nan
        pairs_ok = usable[s, :-1] & usable[s, 1:]
        broken[s] = int((~pairs_ok).sum())
        hit = np.flatnonzero(good[s, :-1] & good[s, 1:])
        if len(hit):
            onset[s] = BINS[hit[0]]
            cat[s] = 2 if BINS[hit[0]] == first_obs[s] else 3
        else:
            cat[s] = 5 if broken[s] > 0 else 4
    return dict(cat=cat, onset=onset, loss_during=loss_d, n_gt_during=n_d, usable=usable, loss_bins=loss_b, first_observed=first_obs, n_broken_pairs=broken)


def aggregate_onset(loss_by_bin, margin=LOSS_MARGIN):
    """Aggregate-curve recovery (kept separate from per-event recovery): loss_by_bin (B, 10) pooled losses. Returns (B,) onset seconds or nan."""
    good = loss_by_bin <= margin + EPS
    pair = good[:, :-1] & good[:, 1:]
    first = np.where(pair.any(1), pair.argmax(1), -1)
    return np.where(first >= 0, np.array(BINS)[np.clip(first, 0, 8)], np.nan)
