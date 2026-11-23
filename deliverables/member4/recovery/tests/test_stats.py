import os, sys, unittest
import numpy as np
from recovery import stats
from recovery.data import PHASES, POST, BINS, FIELDS
from recovery.events import classify
from recovery import samplelevel as sl


def arrays(S, during=(10, 8, 4), post=(10, 8, 8), present=True):
    """S identical scenes: n_gt, tp(clean), tp_o(outage) per phase; during differs, post bins equal."""
    P = len(PHASES)
    a = {f: np.zeros((S, P), np.int64) for f in FIELDS}; a["present"] = np.zeros((S, P), bool)
    j = PHASES.index("during"); a["n_gt"][:, j], a["tp"][:, j], a["tp_o"][:, j] = during; a["present"][:, j] = True
    for p in POST:
        k = PHASES.index(p); a["n_gt"][:, k], a["tp"][:, k], a["tp_o"][:, k] = post; a["present"][:, k] = present
    a["fp"][:] = 1; a["fp_o"][:] = 2
    return a


class Bootstrap(unittest.TestCase):
    def test_weights_rows_sum_and_reproducible(self):
        W = stats.weights(20, 50, seed=1)
        np.testing.assert_array_equal(W.sum(1), np.full(50, 20)); np.testing.assert_array_equal(W, stats.weights(20, 50, seed=1))
        self.assertFalse(np.array_equal(W, stats.weights(20, 50, seed=2)))

    def test_whole_scene_resampling_keeps_pairs(self):
        # clean/outage counts live in the SAME scene row, so any weighting keeps tp, tp_o aligned: recompute pooled loss and check identity
        a = arrays(30); W = stats.weights(30, 200)
        p = stats.pooled(a, W); j = PHASES.index("during")
        np.testing.assert_allclose(p["tp"][:, j] - p["tp_o"][:, j], W @ (a["tp"][:, j] - a["tp_o"][:, j]))
        np.testing.assert_allclose(p["loss"][:, j], 0.4)                                         # identical scenes: every replicate has loss 0.4

    def test_point_estimate_and_ci_degenerate(self):
        a = arrays(15); cur, bs = stats.curve_summary(a, stats.weights(15, 100))
        self.assertAlmostEqual(cur["during"]["loss"], 0.4); self.assertEqual(cur["during"]["loss_ci95"], [0.4, 0.4])
        self.assertAlmostEqual(cur["b0.0"]["loss"], 0.0); self.assertEqual(cur["b0.0"]["false_positives_outage"], 15 * 2)

    def test_ci_covers_mean_for_heterogeneous_scenes(self):
        rng = np.random.RandomState(0); S = 80
        a = arrays(S); j = PHASES.index("during")
        a["tp"][:, j] = rng.randint(4, 10, S); a["tp_o"][:, j] = rng.randint(0, 6, S)
        cur, bs = stats.curve_summary(a, stats.weights(S, 2000))
        lo, hi = cur["during"]["loss_ci95"]; self.assertLess(lo, cur["during"]["loss"]); self.assertGreater(hi, cur["during"]["loss"])
        self.assertLess(hi - lo, 0.2)

    def test_aggregate_rule(self):
        a = arrays(60); W = stats.weights(60, 500)
        cur, bs = stats.curve_summary(a, W); ag = stats.aggregate_summary(a, W, bs)
        self.assertTrue(ag["aggregate_degradation_detected"]); self.assertEqual(ag["aggregate_recovery_onset_s"], 0.0)
        a2 = arrays(60, during=(10, 8, 8)); cur2, bs2 = stats.curve_summary(a2, W); ag2 = stats.aggregate_summary(a2, W, bs2)
        self.assertFalse(ag2["aggregate_degradation_detected"]); self.assertIn("no statistically detected", ag2["wording"])
        # loss above the margin but interval including zero -> NOT detected
        rng = np.random.RandomState(3); S = 20; a3 = arrays(S); j = PHASES.index("during")
        a3["tp"][:, j] = 8; a3["tp_o"][:, j] = np.where(rng.rand(S) > 0.5, 0, 8)                 # mean loss 0.4 but very noisy at S=20? still >0
        a3["tp_o"][:, j] = rng.choice([4, 12], S)                                                  # mean 0 with big spread
        c3, b3 = stats.curve_summary(a3, stats.weights(S, 800)); ag3 = stats.aggregate_summary(a3, stats.weights(S, 800), b3)
        self.assertFalse(ag3["aggregate_degradation_detected"])

    def test_event_summary_counts_and_censoring(self):
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 8))                                       # all damaged, all recovered at first bin
        cl = classify(a); ev, aux = stats.event_summary(cl, stats.weights(10, 300))
        self.assertEqual(ev["n_damaged"], 10); self.assertEqual(ev["confirmed_onset_by_4s_among_damaged"]["numerator"], 10)
        self.assertEqual(ev["median_recovery_time_s"]["value"], 0.0)
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 2))                                       # never recovers, complete follow-up
        cl = classify(a); ev, _ = stats.event_summary(cl, stats.weights(10, 300))
        self.assertEqual(ev["confirmed_onset_by_4s_among_damaged"]["numerator"], 0); self.assertFalse(ev["median_recovery_time_s"]["reached"])
        self.assertEqual(ev["category_counts"]["damaged_no_confirmed_recovery_complete_followup"], 10)
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 2), present=False)                        # no follow-up at all -> censored, unconfirmed
        ev, _ = stats.event_summary(classify(a), stats.weights(10, 300))
        self.assertEqual(ev["category_counts"]["damaged_unconfirmed_incomplete_followup"], 10)

    def test_complete_followup_and_relative_loss(self):
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 2))                                       # complete follow-up, never recovers
        ev, _ = stats.event_summary(classify(a), stats.weights(10, 200))
        self.assertEqual(ev["damaged_with_complete_followup"]["n"], 10); self.assertEqual(ev["damaged_with_complete_followup"]["confirmed_onset"], 0)
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 8)); a["present"][:5, PHASES.index("b2.0")] = False   # half the events lose one bin
        ev, _ = stats.event_summary(classify(a), stats.weights(10, 200))
        self.assertEqual(ev["damaged_with_complete_followup"]["n"], 5); self.assertAlmostEqual(ev["damaged_with_complete_followup"]["share_of_damaged_with_incomplete_followup"]["value"], 0.5)
        cur, _ = stats.curve_summary(arrays(10), stats.weights(10, 100)); self.assertAlmostEqual(cur["during"]["relative_loss"], 0.5)   # (8-4)/8

    def test_median_requires_more_than_half(self):
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 8))
        for s in range(6):                                                                       # 6 of 10 never recover -> median not reached
            for p in POST: a["tp_o"][s, PHASES.index(p)] = 2
        ev, _ = stats.event_summary(classify(a), stats.weights(10, 300))
        self.assertEqual(ev["n_damaged"], 10); self.assertEqual(ev["confirmed_onset_by_4s_among_damaged"]["numerator"], 4); self.assertFalse(ev["median_recovery_time_s"]["reached"])

    def test_median_interval_keeps_unreached_mass(self):
        # 10000 replicates: 3% unreached (>2.5%) -> the upper endpoint must be 'not reached', not the largest finite value (correction 1)
        m = np.full(10000, 1.0); m[:300] = np.nan; m[300:1300] = 2.0; m[1300:2300] = 3.0
        r = stats.median_summary(1.0, m)
        self.assertEqual(r["B"], 10000); self.assertEqual(r["n_replicates_unreached"], 300)
        self.assertTrue(r["ci95_upper_not_reached"]); self.assertIsNone(r["ci95"][1]); self.assertEqual(r["ci95"][0], 1.0)
        self.assertEqual(r["ci95_conditional_on_reached_supplemental"][1], 3.0)                 # conditional interval is supplemental only
        # 1% unreached (<2.5%) -> upper endpoint finite and equals the unconditional order statistic
        m = np.full(10000, 1.0); m[:100] = np.nan; m[100:1100] = 2.0
        r = stats.median_summary(1.0, m)
        self.assertFalse(r["ci95_upper_not_reached"]); self.assertEqual(r["ci95"][1], 2.0)
        # exactly mixed case through event_summary: all B replicates used
        a = arrays(10, during=(10, 9, 3), post=(10, 8, 8)); a["tp_o"][:4, PHASES.index("b0.0")] = 2; a["tp_o"][:4, PHASES.index("b0.5")] = 2
        ev, _ = stats.event_summary(classify(a), stats.weights(10, 400))
        self.assertEqual(ev["median_recovery_time_s"]["B"], 400)

    def test_common_events_and_pairwise(self):
        S = 12; W = stats.weights(S, 300)
        A = arrays(S, during=(10, 9, 3), post=(10, 8, 8)); B = arrays(S, during=(10, 9, 3), post=(10, 8, 8))
        for s in range(6):
            for p in POST: B["tp_o"][s, PHASES.index(p)] = 2                                     # model B fails to recover in half of the events
        ca, cb = classify(A), classify(B)
        c = stats.common_event_summary({"a": ca, "b": cb}, W)
        self.assertEqual(c["n_common_damaged_events"], 12)
        self.assertEqual(c["per_model"]["a"]["confirmed_onset_by_4s"]["value"], 1.0); self.assertEqual(c["per_model"]["b"]["confirmed_onset_by_4s"]["value"], 0.5)
        self.assertAlmostEqual(c["pairwise"]["a_minus_b"]["onset_fraction_difference"], 0.5)
        pw = stats.pairwise_summary({"a": A, "b": B}, {"a": ca, "b": cb}, W)
        self.assertAlmostEqual(pw["a_minus_b"]["pooled_loss_during_difference"]["value"], 0.0)

    def test_independent_bin_assignment_matches_member3(self):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "SparseBEV"))
        from outage.evaluator import assign_bins
        rng = np.random.RandomState(5)
        for _ in range(100):
            offs = sorted([(f"s{i}", float(x)) for i, x in enumerate(rng.uniform(0, 5.2, rng.randint(0, 12)))], key=lambda t: t[1])
            self.assertEqual(sl.assign(offs), {s: b for b, s in assign_bins(offs).items()})


if __name__ == "__main__":
    unittest.main()
