"""Geometry v1 -> v2 regression record (Inspector finding: matching used LiDAR-sensor XY, the region rule used sample-time ego XY).
 1. Reproducible example: SparseBEV clean val, sample 8c394cbb..., prediction row 23 vs eligible GT index 15 at the frozen tau: LiDAR-XY distance
    2.0266 m (v1: FP) vs ego-plane distance between geometric centres 1.9997 m (v2: TP); the sample's counts change TP/FP 3/1 -> 4/0.
 2. Affected-count comparison v1 (superseded per-sample statistics) vs v2 (current) for every model, condition and threshold in use: number of
    samples whose (tp, fp, ign, n_pred_region) changed, pooled TP/FP deltas, and pooled recall/precision before/after.
Usage (from SparseBEV/): python -m outage.geometry_regression <out.json>"""
import json, os, pickle, sys
import numpy as np
from .evaluator import counts_at
from .gt_tables import ground_xy

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(P, "artifacts")
SAMPLE, ROW = "8c394cbb9706486aae5c0edc1d704178", 23


def example():
    from .event_stats import sample_status
    gt = pickle.load(open(os.path.join(A, "gt_val.pkl"), "rb")); pr = pickle.load(open(os.path.join(A, "runs/val_a/preds.pkl"), "rb"))
    tau = json.load(open(os.path.join(A, "calibration/threshold_sparsebev.json")))["tau"]
    g, rows = gt[SAMPLE], pr[SAMPLE]
    j = 15
    gc = g["gt"]["center"][j]; pc = rows[ROW, :3].copy(); pc[2] += rows[ROW, 5] / 2
    d_lidar = float(np.hypot(*(gc[:2] - rows[ROW, :2])))
    d_ego = float(np.hypot(*(ground_xy(g["T_l2ego"], gc[None])[0] - ground_xy(g["T_l2ego"], pc[None])[0])))
    ss = sample_status(rows, g); c2 = counts_at(ss["score"], ss["status"], tau)
    return dict(sample=SAMPLE, prediction_row=ROW, score=float(rows[ROW, 9 + 1]), eligible_gt_index=j, gt_instance=g["gt"]["instance"][j], tau=tau,
                lidar_xy_distance_m=d_lidar, ego_plane_geometric_centre_distance_m=d_ego, v1_status="FP (2.0266 m > 2 m)" if d_lidar > 2 else "TP",
                v2_status="TP" if d_ego <= 2 else "FP", v2_counts_at_tau=c2, v2_counts_expected_tp4_fp0=bool(c2["tp"] == 4 and c2["fp"] == 0),
                matched_gt_in_v2=int(j in set(ss["gt_idx"][ss["status"] == 1].tolist())))


def compare(tag):
    old = pickle.load(open(os.path.join(A, f"superseded_geometry_v1/event_stats/per_sample_{tag}.pkl"), "rb"))
    new = pickle.load(open(os.path.join(A, f"event_stats/per_sample_{tag}.pkl"), "rb"))
    gt = pickle.load(open(os.path.join(A, "gt_val.pkl" if tag.endswith("val") else "gt_pilot.pkl"), "rb"))
    out = dict(tau_old=old["tau"], tau_new=new["tau"], runs={})
    n_gt = int(sum(g["gt"]["eligible"].sum() for g in gt.values()))
    for cond in old["per_sample"]:
        ch, tot_o, tot_n = 0, dict(tp=0, fp=0), dict(tp=0, fp=0)
        changed = []
        for s, so in old["per_sample"][cond].items():
            sn = new["per_sample"][cond][s]
            co, cn = counts_at(so["score"], so["status"], old["tau"]), counts_at(sn["score"], sn["status"], new["tau"])
            for k in ("tp", "fp"):
                tot_o[k] += co[k]; tot_n[k] += cn[k]
            if co != cn:
                ch += 1; changed.append(dict(sample=s, v1=co, v2=cn))
        out["runs"][cond] = dict(n_samples=len(old["per_sample"][cond]), samples_with_changed_counts=ch, tp_v1=tot_o["tp"], tp_v2=tot_n["tp"], fp_v1=tot_o["fp"], fp_v2=tot_n["fp"],
                                 recall_v1=tot_o["tp"] / n_gt, recall_v2=tot_n["tp"] / n_gt,
                                 precision_v1=tot_o["tp"] / max(1, tot_o["tp"] + tot_o["fp"]), precision_v2=tot_n["tp"] / max(1, tot_n["tp"] + tot_n["fp"]), changed_examples=changed[:5])
    return out


def main(out):
    rep = dict(geometry="v2: sample-time ego-plane XY between geometric centres (v1 superseded: LiDAR-sensor XY, prediction z = box bottom)", example=example(), comparison={})
    for tag in ("sparsebev_val", "bevformer_val", "streampetr_val", "sparsebev_pilot"):
        rep["comparison"][tag] = compare(tag)
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep["example"], indent=1))
    for tag, c in rep["comparison"].items():
        print(tag, "tau old/new", round(c["tau_old"], 4), round(c["tau_new"], 4), {k: (v["samples_with_changed_counts"], v["tp_v2"] - v["tp_v1"], v["fp_v2"] - v["fp_v1"]) for k, v in c["runs"].items()})


if __name__ == "__main__":
    main(sys.argv[1])
