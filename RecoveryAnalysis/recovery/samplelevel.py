"""Sample-level re-derivation from Member 3's per-sample statuses (score-sorted `status`, `gt_idx`, `matched_instance`) + the packaged GT tables:
  (1) an INDEPENDENT recomputation of the per-scene x per-phase table (own phase assignment, own counting) -> must equal Member 3's CSV exactly;
  (2) per-class pooled counts (eligible GT / matched) per scene x phase, for per-class recall loss with scene-bootstrap intervals;
  (3) the same-object diagnostic: eligible GT INSTANCE tokens followed from the last during-outage sample to the first STRICTLY LATER sample whose CAM_FRONT
      exposure is restored (front timestamp >= restoration) and that carries a post-restoration bin, to separate continuing errors from objects entering /
      leaving the region (plan p.4). A zero-time pair (the same sample is both the last 'during' sample by front exposure and bin 0.0 by evaluation time) or a
      pair whose later sample still has a black front image is NEVER a recovery observation: such events are counted as unavailable (correction 3).
Phases (plan p.2-3 / Member 3): pre = front keyframe before t_start; during = front keyframe in [t_start, t_start+d); bin b = evaluation-time offset
(lidar time - restoration) assigned ONE-TO-ONE to the scheduled bins 0.0..4.5 s (tolerance 0.25 s; ties: smaller distance, smaller bin, smaller offset)."""
import json, os, pickle
import numpy as np
from .data import S3, STATS, MODELS, DURATIONS, PHASES, BINS, POST, FIELDS, sha256

CLASSES = ["car", "truck", "construction_vehicle", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "traffic_cone"]
TP_, FP_, IGN_ = 1, 2, 3
US = 1_000_000


def load_schedule(path):
    d = json.load(open(path))
    return {s["scene_token"]: s for s in d["scenes"]}


def assign(offsets, tol=0.25):
    """offsets: list of (sample, offset_s >= 0). One-to-one bin assignment -> {sample: bin}."""
    pairs = sorted((abs(h - b), b, h, i) for b in BINS for i, (_, h) in enumerate(offsets) if abs(h - b) <= tol + 1e-9)
    ub, us, out = set(), set(), {}
    for _, b, h, i in pairs:
        if b not in ub and i not in us:
            ub.add(b); us.add(i); out[offsets[i][0]] = b
    return out


def phase_table(gt_tab, sched, d):
    """{scene: {sample: [phase labels]}} from schedule + raw timestamps only."""
    by = {}
    for s, g in gt_tab.items():
        by.setdefault(g["scene"], []).append(s)
    out = {}
    for scene, samples in by.items():
        o = sched[scene]
        t0, t1 = o["t_start_us"], o["t_start_us"] + round(d * US)
        offs = sorted(((s, (gt_tab[s]["ts_lidar_us"] - t1) / US) for s in samples if gt_tab[s]["ts_lidar_us"] >= t1), key=lambda x: x[1])
        sb = assign(offs)
        lab = {}
        for s in samples:
            f = gt_tab[s]["ts_front_us"]
            L = (["pre"] if f < t0 else []) + (["during"] if t0 <= f < t1 else []) + ([f"b{sb[s]:.1f}"] if s in sb else [])
            lab[s] = L or ["other"]
        out[scene] = lab
    return out


def load_gt():
    return pickle.load(open(os.path.join(S3, "artifacts/gt_val.pkl"), "rb"))


def sample_stats(rec, tau, g):
    """Counts for one sample at the frozen threshold (rec = per-sample record of one condition)."""
    m = rec["score"] >= tau
    st = rec["status"][m]
    elig = np.flatnonzero(g["gt"]["eligible"])
    tp_idx = rec["gt_idx"][m][st == TP_]
    mi = [x for x, s_, k in zip(rec["matched_instance"], rec["status"], rec["score"]) if s_ == TP_ and k >= tau]
    return dict(tp=int((st == TP_).sum()), fp=int((st == FP_).sum()), ign=int((st == IGN_).sum()), n_pred_region=int(((st == TP_) | (st == FP_)).sum()),
                n_gt=len(elig), tp_idx=tp_idx, matched_instance=mi)


def build_model(model, gt_tab=None, sched=None):
    """Returns dict(duration -> dict(table fields (S,P), per-class tp/tp_o/n_gt (S,P,C), sameobject X (S,K))) + scenes."""
    gt_tab = gt_tab or load_gt()
    sched = sched or load_schedule(os.path.join(S3, "artifacts/schedule_trainval_val.json"))
    pk = pickle.load(open(os.path.join(STATS, f"per_sample_{MODELS[model]['tag']}.pkl"), "rb"))
    tau = pk["tau"]
    scenes = sorted({g["scene"] for g in gt_tab.values()}); si = {s: i for i, s in enumerate(scenes)}
    S, P, C = len(scenes), len(PHASES), len(CLASSES)
    clean = {s: sample_stats(r, tau, gt_tab[s]) for s, r in pk["per_sample"]["clean"].items()}
    res = {}
    for d in DURATIONS:
        out_ps = {s: sample_stats(r, tau, gt_tab[s]) for s, r in pk["per_sample"][str(d)].items()}
        ph = phase_table(gt_tab, sched, d)
        T = {f: np.zeros((S, P), np.int64) for f in FIELDS}; pres = np.zeros((S, P), bool)
        n_c = np.zeros((S, P, C), np.int64); tp_c = np.zeros((S, P, C), np.int64); tp_oc = np.zeros((S, P, C), np.int64)
        for scene, lab in ph.items():
            i = si[scene]
            for s, labels in lab.items():
                g, cs, os_ = gt_tab[s]["gt"], clean[s], out_ps[s]
                elig = np.flatnonzero(g["eligible"])
                cls_e = g["cls"][elig]
                ncl = np.bincount(cls_e, minlength=C)
                tcl = np.bincount(g["cls"][cs["tp_idx"]], minlength=C); tol_ = np.bincount(g["cls"][os_["tp_idx"]], minlength=C)
                for L in labels:
                    j = PHASES.index(L)
                    T["n_samples"][i, j] += 1; T["n_gt"][i, j] += cs["n_gt"]; pres[i, j] = True
                    for k in ("tp", "fp", "ign", "n_pred_region"):
                        T[k][i, j] += cs[k]; T[k + "_o"][i, j] += os_[k]
                    n_c[i, j] += ncl; tp_c[i, j] += tcl; tp_oc[i, j] += tol_
        t1 = {sc: sched[sc]["t_start_us"] + round(d * US) for sc in ph}
        res[d] = dict(table=T, present=pres, n_c=n_c, tp_c=tp_c, tp_oc=tp_oc, sameobject=sameobject(scenes, si, ph, gt_tab, clean, out_ps, t1))
    return dict(scenes=scenes, tau=tau, by_duration=res)


SO_KEYS = ["events_with_last_during_and_later_restored", "continuing_objects", "continuing_clean_detected_both", "cont_missed_d_missed_after", "cont_missed_d_detected_after",
           "cont_detected_d_detected_after", "cont_detected_d_missed_after", "entering_objects", "entering_clean_detected", "entering_outage_missed",
           "leaving_objects",
           # event accounting (correction 3): what the 'after' observation was, and why events have none
           "events_with_during_sample", "events_unavailable_same_sample_as_bin0", "events_unavailable_bin0_front_still_black", "events_unavailable_no_restored_later_sample",
           "events_after_is_bin0", "events_after_is_later_bin", "sum_after_bin_tenths_s"]


def sameobject(scenes, si, ph, gt_tab, clean, out_ps, t1, detail=None):
    """Per scene counts X (S, len(SO_KEYS)). Pair = (LAST during-outage sample sd, FIRST sample after) where 'after' is a sample STRICTLY LATER than sd
    (evaluation time) whose CAM_FRONT exposure is restored (front timestamp >= restoration time t1) and that has a post-restoration bin label.
    sd itself can never be 'after'; a bin-0.0 sample that is sd (zero-time) or whose front image is still black is not a recovery observation and is counted
    as unavailable instead (the frozen primary bin assignment is untouched). `detail` (optional list) receives one record per event with actual timestamps.
    Instances are GT instance tokens of ELIGIBLE objects. 'detected' = matched at the frozen threshold (outage run for the cont_* rows). The cont_* rows
    are restricted to objects the CLEAN run detected at both samples, which isolates the outage effect from objects that are missed anyway."""
    X = np.zeros((len(scenes), len(SO_KEYS)), np.int64); K = {n: i for i, n in enumerate(SO_KEYS)}
    for scene, lab in ph.items():
        during = sorted([s for s, L in lab.items() if "during" in L], key=lambda s: gt_tab[s]["ts_lidar_us"])
        if not during:
            continue
        x = X[si[scene]]; x[K["events_with_during_sample"]] = 1
        sd = during[-1]
        b0 = [s for s, L in lab.items() if "b0.0" in L]
        bin_of = lambda s: min(float(L[1:]) for L in lab[s] if L.startswith("b") and L != "b")
        cands = sorted([s for s, L in lab.items() if gt_tab[s]["ts_lidar_us"] > gt_tab[sd]["ts_lidar_us"] and gt_tab[s]["ts_front_us"] >= t1[scene]
                        and any(l.startswith("b") for l in L)], key=lambda s: gt_tab[s]["ts_lidar_us"])
        s0 = cands[0] if cands else None
        if b0 and b0[0] == sd:
            x[K["events_unavailable_same_sample_as_bin0"]] = 1
        elif b0 and gt_tab[b0[0]]["ts_front_us"] < t1[scene]:
            x[K["events_unavailable_bin0_front_still_black"]] = 1
        if s0 is None:
            x[K["events_unavailable_no_restored_later_sample"]] = 1
            if detail is not None:
                detail.append(dict(scene=scene, last_during=sd, after=None))
            continue
        bn = bin_of(s0)
        x[K["events_after_is_bin0"] if bn == 0.0 else K["events_after_is_later_bin"]] = 1; x[K["sum_after_bin_tenths_s"]] = int(round(bn * 10))
        if detail is not None:
            detail.append(dict(scene=scene, last_during=sd, after=s0, after_bin=bn, front_ts_last_during_us=gt_tab[sd]["ts_front_us"], front_ts_after_us=gt_tab[s0]["ts_front_us"],
                               restoration_us=t1[scene], lidar_separation_s=(gt_tab[s0]["ts_lidar_us"] - gt_tab[sd]["ts_lidar_us"]) / US))
        E = {}
        for s in (sd, s0):
            g = gt_tab[s]["gt"]; E[s] = {g["instance"][k] for k in np.flatnonzero(g["eligible"])}
        cont = E[sd] & E[s0]
        c_d, c_0 = set(clean[sd]["matched_instance"]), set(clean[s0]["matched_instance"])
        o_d, o_0 = set(out_ps[sd]["matched_instance"]), set(out_ps[s0]["matched_instance"])
        both = cont & c_d & c_0
        x[0] = 1; x[1] = len(cont); x[2] = len(both)
        x[3] = len([i for i in both if i not in o_d and i not in o_0]); x[4] = len([i for i in both if i not in o_d and i in o_0])
        x[5] = len([i for i in both if i in o_d and i in o_0]); x[6] = len([i for i in both if i in o_d and i not in o_0])
        ent = E[s0] - E[sd]
        x[7] = len(ent); x[8] = len([i for i in ent if i in c_0]); x[9] = len([i for i in ent if i in c_0 and i not in o_0])
        x[10] = len(E[sd] - E[s0])
    return X
