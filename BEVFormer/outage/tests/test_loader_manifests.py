"""Synthetic tests of the replicated SparseBEV test-mode history selection (outage/loader_manifests.py)."""
import unittest
from outage.loader_manifests import sparsebev_consumed, CAMS_LD, CAMS_SB


def mk(n, sweeps_per=5):
    """n samples of one scene, 2 Hz keyframes at t=k*500 ms, 5 sweeps (12 Hz) before each keyframe except the first."""
    infos = []
    for k in range(n):
        t = k * 500_000
        cams = {c: dict(path=f"samples/{c}/k{k}", ts=t) for c in CAMS_SB}
        sw = [] if k == 0 else [{c: dict(path=f"sweeps/{c}/k{k}_s{j}", ts=t - (j + 1) * 83_000) for c in CAMS_SB} for j in range(sweeps_per)]
        infos.append(dict(token=f"s{k}", scene="A", timestamp=t, cams=cams, sweeps=sw))
    return infos


class TestSparseBEVSelection(unittest.TestCase):
    def test_first_sample_repeats_keyframe(self):
        cons, kind = sparsebev_consumed(mk(10), 0)
        self.assertEqual(kind, "no_history_keyframe_repeated")
        self.assertEqual(len(cons), 48)
        self.assertEqual([p for _, p, _ in cons[6:12]], [f"samples/{c}/k0" for c in CAMS_LD])

    def test_interval6_hits_previous_keyframes_only(self):
        infos = mk(12)
        cons, kind = sparsebev_consumed(infos, 11)
        self.assertEqual(kind, "full")
        hist = [p for _, p, _ in cons[6:]]
        self.assertTrue(all(p.startswith("samples/") for p in hist))
        self.assertEqual(hist[::6], [f"samples/CAM_FRONT/k{k}" for k in range(10, 3, -1)])   # k-1 ... k-7

    def test_short_history_clamped_to_oldest(self):
        cons, kind = sparsebev_consumed(mk(12), 3)
        self.assertEqual(kind, "history_clamped_to_oldest")
        hist = [p for c, p, _ in cons[6:] if c == "CAM_FRONT"]
        self.assertEqual(hist, ["samples/CAM_FRONT/k2", "samples/CAM_FRONT/k1", "samples/CAM_FRONT/k0"] + ["samples/CAM_FRONT/k0"] * 4)

    def test_slot_order_is_loader_order(self):
        cons, _ = sparsebev_consumed(mk(5), 4)
        self.assertEqual([c for c, _, _ in cons[:6]], CAMS_LD)


if __name__ == "__main__":
    unittest.main()
