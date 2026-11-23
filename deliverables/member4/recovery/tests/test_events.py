import unittest, sys, os
import numpy as np
from recovery.data import PHASES, POST, BINS, MIN_GT
from recovery.events import classify, aggregate_onset, CATS


def mk(S=1, n_during=10, tp_during=(8, 3), n_bin=10, losses=None, present=None):
    """One-scene arrays: during recall 8/10 clean vs 3/10 outage by default; per-bin (tp, tp_o) list or None -> equal."""
    P = len(PHASES)
    a = {k: np.zeros((S, P), np.int64) for k in ("n_gt", "tp", "tp_o")}
    a["present"] = np.zeros((S, P), bool)
    j = PHASES.index("during"); a["n_gt"][:, j] = n_during; a["tp"][:, j], a["tp_o"][:, j] = tp_during
    for i, p in enumerate(POST):
        k = PHASES.index(p); a["n_gt"][:, k] = n_bin; a["present"][:, k] = True if present is None else present[i]
        c, o = (8, 8) if losses is None else losses[i]
        a["tp"][:, k], a["tp_o"][:, k] = c, o
    return a


class Events(unittest.TestCase):
    def test_insufficient_and_no_degradation(self):
        self.assertEqual(classify(mk(n_during=4, tp_during=(3, 0)))["cat"][0], 0)
        self.assertEqual(classify(mk(tp_during=(8, 8)))["cat"][0], 1)
        self.assertEqual(classify(mk(n_during=10, tp_during=(8, 7)))["cat"][0], 2)                # loss 0.1 > 0.05 -> damaged (recovered at first bin)
    def test_margin_strict_for_damage(self):
        a = mk(n_during=20, tp_during=(11, 10))                                                  # loss exactly 0.05 -> NOT damaged
        self.assertEqual(classify(a)["cat"][0], 1)
        a = mk(n_during=20, tp_during=(12, 10)); self.assertEqual(classify(a)["cat"][0], 2)       # 0.10 > 0.05, recovered at first bin
    def test_first_bin_recovery(self):
        c = classify(mk()); self.assertEqual(c["cat"][0], 2); self.assertEqual(c["onset"][0], 0.0)
    def test_later_recovery(self):
        losses = [(8, 2)] * 3 + [(8, 8)] * 7
        c = classify(mk(losses=losses)); self.assertEqual(c["cat"][0], 3); self.assertEqual(c["onset"][0], 1.5)
    def test_single_good_bin_not_confirmed_complete_followup(self):
        losses = [(8, 2)] * 10; losses[4] = (8, 8)
        self.assertEqual(classify(mk(losses=losses))["cat"][0], 4)
    def test_missing_bin_breaks_pair_incomplete(self):
        losses = [(8, 8)] * 10; present = [True] * 10; present[1] = False
        c = classify(mk(losses=losses, present=present))
        self.assertEqual(c["onset"][0], 2.0 / 2); self.assertEqual(c["cat"][0], 3)             # 0.0 cannot be confirmed by missing 0.5 -> onset 1.0
    def test_never_recovers_with_missing_bin_is_censored(self):
        losses = [(8, 2)] * 10; present = [True] * 10; present[6] = False
        self.assertEqual(classify(mk(losses=losses, present=present))["cat"][0], 5)
    def test_insufficient_bin_counts_as_broken(self):
        a = mk(losses=[(8, 2)] * 10); a["n_gt"][:, PHASES.index("b2.0")] = MIN_GT - 1
        self.assertEqual(classify(a)["cat"][0], 5)
    def test_last_bin_cannot_be_onset(self):
        losses = [(8, 2)] * 10; losses[9] = (8, 8)
        self.assertEqual(classify(mk(losses=losses))["cat"][0], 4)
    def test_first_observed_bin_skips_missing(self):
        present = [True] * 10; present[0] = False
        c = classify(mk(present=present)); self.assertEqual(c["first_observed"][0], 0.5); self.assertEqual(c["cat"][0], 2)
    def test_negative_loss_counts_as_recovered(self):
        c = classify(mk(losses=[(8, 9)] * 10)); self.assertEqual(c["cat"][0], 2)

    def test_agrees_with_member3_recovery_onset(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "SparseBEV"))
        from outage.evaluator import recovery_onset
        rng = np.random.RandomState(0)
        for _ in range(300):
            loss = {b: float(rng.choice([0.0, 0.04, 0.05, 0.06, 0.3, -0.1])) for b in BINS if rng.rand() > 0.2}
            ok = {b: bool(rng.rand() > 0.15) for b in BINS}
            # build arrays that realise exactly these losses with n=20 (loss in 1/20 steps: map to tp counts)
            losses, present = [], []
            for b in BINS:
                present.append(b in loss)
                l = loss.get(b, 0.0); losses.append((20, int(round(20 - l * 20))))
            a = mk(n_during=20, tp_during=(20, 5), n_bin=20, losses=losses, present=present)
            for i, b in enumerate(BINS):
                if not ok[b]: a["n_gt"][:, PHASES.index(POST[i])] = MIN_GT - 1
            c = classify(a)
            ref = recovery_onset({b: (round(l * 20) / 20) for b, l in loss.items()}, {b: ok[b] for b in BINS}, margin=0.05) if "margin" in recovery_onset.__code__.co_varnames else \
                recovery_onset({b: (round(l * 20) / 20) for b, l in loss.items()}, {b: ok[b] for b in BINS})
            got = None if np.isnan(c["onset"][0]) else float(c["onset"][0])
            self.assertEqual(got, ref)

    def test_aggregate_onset(self):
        L = np.array([[0.3, 0.2, 0.04, 0.03, 0.3, 0, 0, 0, 0, 0], [0.3] * 10, [0.0] * 10])
        np.testing.assert_array_equal(np.nan_to_num(aggregate_onset(L), nan=-1), [1.0, -1, 0.0])


if __name__ == "__main__":
    unittest.main()
