"""Label-free temporal-contamination diagnostic for BEVFormer's recurrent BEV cache (no GT, no thresholds; NOT a recall
curve - Members 3/4 own event analysis). For each outage run vs the clean run, per evaluated sample, d = mean |sorted top-K
score(clean) - sorted top-K score(outage)| (K=50). Reported by restoration-offset bin (the audit's offsets, 0..4.5 s, +-0.25 s)
plus prefix (before the outage start) and during-outage samples. A recurrent state never returns exactly to clean: the point
is to show whether the effect decays after restoration and to document that 'returns to clean within tolerance' (the plan's
finite-window check) does not apply to BEVFormer. Usage: python -m outage.state_contamination <runs_dir> <clean> <out.json> <outage runs...>"""
import json, os, pickle, sys
import numpy as np
from .audit import load_audit

K = 50


def score_gap(a, b):
    sa, sb = np.sort(a[:, 10])[::-1][:K], np.sort(b[:, 10])[::-1][:K]
    return float(np.abs(sa - sb).mean())


def main(runs, clean, out, *names):
    C = pickle.load(open(os.path.join(runs, clean, "preds.pkl"), "rb"))
    rep = dict(K=K, clean=clean, runs={})
    for n in names:
        O = pickle.load(open(os.path.join(runs, n, "preds.pkl"), "rb"))
        a = load_audit(os.path.join(runs, n, "audit.json"))
        if a.duration_s <= 0:
            raise SystemExit(f"{n} is a clean run; pass outage runs only")
        ev = a.eval_samples()
        bins = {}
        for (scene, s), t in ev.items():
            so = a.schedule.get(scene)
            if so is None:
                continue
            g = score_gap(C[s], O[s])
            if t < so.t_start_us:
                key = "prefix"
            elif t < so.restoration_us(a.duration_s):
                key = "during"
            else:
                off = (t - so.restoration_us(a.duration_s)) / 1e6
                b = round(off * 2) / 2
                if abs(off - b) > 0.25 + 1e-9 or b > 4.5:
                    continue
                key = f"{b:.1f}"
            bins.setdefault(key, []).append(g)
        order = ["prefix", "during"] + [f"{0.5 * i:.1f}" for i in range(10)]
        rep["runs"][n] = {k: dict(n=len(bins[k]), median=float(np.median(bins[k])), mean=float(np.mean(bins[k])),
                                  frac_exact_zero=float(np.mean(np.array(bins[k]) == 0))) for k in order if k in bins}
    json.dump(rep, open(out, "w"), indent=1)
    for n, r in rep["runs"].items():
        print(n, {k: round(v["median"], 4) for k, v in r.items()})


if __name__ == "__main__":
    main(*sys.argv[1:4], *sys.argv[4:])
