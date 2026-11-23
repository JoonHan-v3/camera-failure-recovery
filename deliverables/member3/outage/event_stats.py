"""Per-scene / per-bin detection statistics for the front-region event analysis (model-independent).

Inputs: common-format predictions (preds.pkl) for the clean run and each outage duration, the frozen schedule, GT tables
(gt_tables.py) and a frozen score threshold per model (calibrate.py). Output per (scene, duration): rows by phase
  pre    : samples whose CAM_FRONT keyframe is before t_start
  during : samples whose CAM_FRONT keyframe lies in [t_start, t_start + d)   (the image the model consumed was black)
  b0.0 ... b4.5 : post-restoration bins, assigned one-to-one with the SAME convention as audit.missing_bins
                  (offset = lidar/evaluation time - restoration time, tolerance 0.25 s); missing bins are kept as missing
with n_samples, n_gt (eligible GT), and tp / fp / ign / n_pred_region for the clean run and the outage run on the SAME GT objects.
A sample can carry both `during` and a bin when its CAM_FRONT image was black but its evaluation (lidar) time is already after
restoration (<= 43 ms camera lead); such samples are counted in both places and reported in the summary.
Same-object join (schema): per_sample[condition][sample] = dict(score, status, gt_idx, matched_instance), all aligned with the predictions in
DESCENDING-SCORE order; `gt_idx[k]` indexes the FULL gt_<split>.pkl[sample]['gt']['instance'] list (all ten-class GT, eligible or not), and
`matched_instance[k]` is that instance token for TP rows (None otherwise). The pickle's `eligible_gt_instances[sample]` is the FILTERED list of eligible
instances only — a count/denominator helper that is NOT aligned with gt_idx and must not be indexed with it."""
import argparse, csv, json, os, pickle
import numpy as np
from .evaluator import match_sample, counts_at, recall, precision, assign_bins, BINS_S, TP
from .gt_tables import project_front, ground_xy
from .schedule import load_schedule, US


def sample_status(rows, g):
    """rows: predictions of one sample (common format); g: gt_tables entry. Returns dict(score, status, matched_gt_idx) in descending score."""
    rows = np.asarray(rows, float).reshape(-1, 11)
    c = rows[:, :3].copy()
    c[:, 2] += rows[:, 5] / 2.0                       # box bottom -> gravity centre (same convention as nuScenes annotations)
    in_reg, _, _, _ = project_front(g["T_l2cam"], g["K"], g["T_l2ego"], c)
    gt = g["gt"]
    # geometry v2: match in the sample-time EGO ground plane between GEOMETRIC centres (same plane as the 50 m region rule)
    order, status, gidx = match_sample(rows, in_reg, ground_xy(g["T_l2ego"], c), gt["cls"], ground_xy(g["T_l2ego"], gt["center"]), gt["eligible"])
    inst = gt["instance"]
    return dict(score=rows[order, 10], status=status, gt_idx=gidx,
                matched_instance=[inst[j] if (st == TP and j >= 0) else None for st, j in zip(status, gidx)])


def run_statuses(preds, gt_tab):
    return {s: sample_status(preds[s], gt_tab[s]) for s in preds}


def phases(gt_tab, sched, duration_s):
    """{scene: {sample: dict(phase flags)}} from schedule + raw timestamps only (independent of any run)."""
    by_scene = {}
    for s, g in gt_tab.items():
        by_scene.setdefault(g["scene"], []).append(s)
    out = {}
    for scene, samples in by_scene.items():
        o = sched.get(scene)
        if o is None:
            continue
        t0, t1 = o.window(duration_s)
        rest = o.restoration_us(duration_s)
        offs = [(s, (gt_tab[s]["ts_lidar_us"] - rest) / US) for s in samples if gt_tab[s]["ts_lidar_us"] - rest >= 0]
        bins = assign_bins(sorted(offs, key=lambda x: x[1]))
        sample_bin = {v: b for b, v in bins.items()}
        out[scene] = dict(
            samples={s: dict(pre=gt_tab[s]["ts_front_us"] < t0, during=t0 <= gt_tab[s]["ts_front_us"] < t1, bin=sample_bin.get(s))
                     for s in samples},
            missing_bins=[b for b in BINS_S if b not in bins])
    return out


def _acc():
    return dict(n_samples=0, n_gt=0, tp=0, fp=0, ign=0, n_pred_region=0, tp_o=0, fp_o=0, ign_o=0, n_pred_region_o=0)


def scene_table(gt_tab, st_clean, st_out, sched, duration_s, tau):
    """Rows per scene x phase for one duration, paired clean/outage at the same threshold tau."""
    ph = phases(gt_tab, sched, duration_s)
    rows = []
    for scene in sorted(ph):
        acc = {}
        for s, f in ph[scene]["samples"].items():
            labels = [k for k in ("pre", "during") if f[k]] + (["b%.1f" % f["bin"]] if f["bin"] is not None else [])
            if not labels:
                labels = ["other"]
            n_gt = int(gt_tab[s]["gt"]["eligible"].sum())
            cc, co = counts_at(st_clean[s]["score"], st_clean[s]["status"], tau), counts_at(st_out[s]["score"], st_out[s]["status"], tau)
            for lab in labels:
                a = acc.setdefault(lab, _acc())
                a["n_samples"] += 1; a["n_gt"] += n_gt
                for k in ("tp", "fp", "ign", "n_pred_region"):
                    a[k] += cc[k]; a[k + "_o"] += co[k]
        for lab in ["pre", "during"] + ["b%.1f" % b for b in BINS_S] + ["other"]:
            a = acc.get(lab)
            if a is None:
                rows.append(dict(scene=scene, duration_s=duration_s, phase=lab, **_acc(), present=False))
            else:
                rows.append(dict(scene=scene, duration_s=duration_s, phase=lab, **a, present=True))
    return rows, {k: v["missing_bins"] for k, v in ph.items()}


def pooled_curve(rows):
    """Aggregate (pooled over scenes) clean/outage recall + loss per phase; plain aggregation, no inference (bootstrap is Member 4's)."""
    agg = {}
    for r in rows:
        a = agg.setdefault(r["phase"], _acc()); a["n_scenes"] = a.get("n_scenes", 0) + int(r["present"])
        for k in _acc():
            a[k] += r[k]
    out = {}
    for lab, a in agg.items():
        rc, ro = recall(a["tp"], a["n_gt"]), recall(a["tp_o"], a["n_gt"])
        out[lab] = dict(a, recall_clean=rc, recall_outage=ro, loss=None if rc is None else rc - ro,
                        precision_clean=precision(a["tp"], a["fp"]), precision_outage=precision(a["tp_o"], a["fp_o"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True); ap.add_argument("--schedule", required=True); ap.add_argument("--tau-json", required=True)
    ap.add_argument("--clean", required=True); ap.add_argument("--d05", required=True); ap.add_argument("--d10", required=True)
    ap.add_argument("--d20", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    gt_tab = pickle.load(open(a.gt, "rb"))
    sched, _, _ = load_schedule(a.schedule)
    tau = json.load(open(a.tau_json))["tau"]
    assert tau is not None, "no calibrated threshold"
    os.makedirs(a.out, exist_ok=True)
    cl = run_statuses(pickle.load(open(a.clean, "rb")), gt_tab)
    allrows, summ, per_sample = [], {}, {"clean": cl}
    for d, p in ((0.5, a.d05), (1.0, a.d10), (2.0, a.d20)):
        st = run_statuses(pickle.load(open(p, "rb")), gt_tab)
        per_sample[str(d)] = st
        rows, miss = scene_table(gt_tab, cl, st, sched, d, tau)
        allrows += rows
        summ[str(d)] = dict(pooled=pooled_curve(rows), scenes_with_missing_bins=sum(1 for v in miss.values() if v), missing_bins=miss)
    cols = ["scene", "duration_s", "phase", "present", "n_samples", "n_gt", "tp", "fp", "ign", "n_pred_region", "tp_o", "fp_o", "ign_o", "n_pred_region_o"]
    with open(os.path.join(a.out, f"per_scene_bins_{a.tag}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); [w.writerow({k: r[k] for k in cols}) for r in allrows]
    json.dump(dict(tag=a.tag, tau=tau, summary=summ, n_eligible_gt_total=int(sum(g["gt"]["eligible"].sum() for g in gt_tab.values()))),
              open(os.path.join(a.out, f"summary_{a.tag}.json"), "w"), indent=1)
    pickle.dump(dict(tau=tau, per_sample=per_sample, eligible_gt_instances={s: [g["gt"]["instance"][i] for i in np.flatnonzero(g["gt"]["eligible"])] for s, g in gt_tab.items()}),
                open(os.path.join(a.out, f"per_sample_{a.tag}.pkl"), "wb"))
    print(a.tag, "tau", tau, "rows", len(allrows))


if __name__ == "__main__":
    main()
