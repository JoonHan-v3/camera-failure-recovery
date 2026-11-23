"""Qualitative gallery (plan p.4): CAM_FRONT and BEV overlays of detections and GT before / during / after selected 2 s outages.
Events are chosen by FIXED RULES computed from the saved per-event table (not by eye): for the study's headline outage (2 s), SparseBEV events:
  recovered_first_bin    : category 'damaged, recovered at first observed bin' with the most eligible objects during the outage
  late_recovery          : category 'damaged, later recovery' with the LATEST confirmed onset
  false_positives        : the event with the largest increase in region false positives (outage - clean) over bins 0.0-1.0 s
  no_measured_degradation: category 'no measured degradation' with the most eligible objects during the outage   (a counter-example)
(ties: smallest scene token). Markers (frozen per-model threshold tau; eligible GT only):
  GT circle  green = detected in clean and outage | red = detected in clean, MISSED in outage | grey = missed in both | blue = detected only in outage
  orange x   = outage-run prediction >= tau in the region that matched no eligible/ignored GT (false positive)
CAM_FRONT tiles show what the model consumed (black while the outage is active); BEV is the sample-time ego frame (x forward, y left), camera FoV wedge ±35°.
Usage (from RecoveryAnalysis/): python -m recovery.qualitative"""
import csv, json, os, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from . import samplelevel as sl
from .data import MODELS, S3, STATS, PROJECTS
from .figures import J, FIG

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_IMG = os.path.expanduser("~/storage/nuscenes_trainval_root")
D = 2.0


def project_uv(g, centers_lidar):
    c = np.asarray(centers_lidar, float).reshape(-1, 3)
    pc = c @ g["T_l2cam"][:3, :3].T + g["T_l2cam"][:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = (pc @ g["K"].T)[:, :2] / pc[:, 2:3]
    return uv, pc[:, 2]


def ego_xy(g, centers_lidar):
    c = np.asarray(centers_lidar, float).reshape(-1, 3)
    return (c @ g["T_l2ego"][:3, :3].T + g["T_l2ego"][:3, 3])[:, :2]


def choose_events():
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "artifacts/analysis/per_event_categories.csv"))) if r["model"] == "sparsebev" and float(r["duration_s"]) == D]
    byc = {}
    for r in rows:
        byc.setdefault(r["category"], []).append(r)
    pick = lambda lst, key, rev=True: sorted(lst, key=lambda r: ((-key(r)) if rev else key(r), r["scene"]))[0]
    sel = {}
    sel["recovered_first_bin"] = pick(byc["damaged_recovered_first_observed_bin"], lambda r: int(r["n_gt_during"]))
    sel["late_recovery"] = pick(byc["damaged_recovered_later"], lambda r: float(r["confirmed_onset_s"]))
    sel["no_measured_degradation"] = pick(byc["no_measured_degradation"], lambda r: int(r["n_gt_during"]))
    tab = [r for r in csv.DictReader(open(os.path.join(STATS, "per_scene_bins_sparsebev_val.csv"))) if float(r["duration_s"]) == D]
    dfp = {}
    for r in tab:
        if r["phase"] in ("b0.0", "b0.5", "b1.0") and r["present"] == "True":
            dfp[r["scene"]] = dfp.get(r["scene"], 0) + int(r["fp_o"]) - int(r["fp"])
    best = sorted(dfp.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    sel["false_positives"] = dict(scene=best[0], fp_increase_bins_0_to_1s=best[1])
    return {k: v["scene"] for k, v in sel.items()}, sel


def load_state():
    gt = sl.load_gt(); sched = sl.load_schedule(os.path.join(S3, "artifacts/schedule_trainval_val.json"))
    infos = pickle.load(open(os.path.join(S3, "artifacts/infos/val_infos.pkl"), "rb"))["infos"]
    img = {i["token"]: os.path.join(ROOT_IMG, i["cams"]["CAM_FRONT"]["data_path"].split("data/nuscenes/")[-1]) for i in infos}
    st, preds = {}, {}
    run = dict(sparsebev=os.path.join(S3, "artifacts/runs/val_%s/preds.pkl"), bevformer=os.path.join(PROJECTS, "BEVFormer/artifacts/runs/val_%s/preds.pkl"),
               streampetr=os.path.join(PROJECTS, "StreamPETR/artifacts/runs/rf_val_%s/preds.pkl"))
    for m, v in MODELS.items():
        pk = pickle.load(open(os.path.join(STATS, f"per_sample_{v['tag']}.pkl"), "rb"))
        st[m] = pk
        preds[m] = dict(clean=pickle.load(open(run[m] % "a", "rb")), outage=pickle.load(open(run[m] % "d20", "rb")))
    return gt, sched, img, st, preds


def sample_view(m, s, gt, st, preds):
    g = gt[s]; tau = st[m]["tau"]; gg = g["gt"]
    out = {}
    for cond, key in (("clean", "clean"), ("outage", str(D))):
        rec = st[m]["per_sample"][key][s]; rows = np.asarray(preds[m]["clean" if cond == "clean" else "outage"][s], float).reshape(-1, 11)
        order = np.argsort(-rows[:, 10], kind="stable"); assert np.allclose(rows[order, 10], rec["score"])
        keep = rec["score"] >= tau
        out[cond] = dict(rows=rows[order][keep], status=rec["status"][keep], gt_idx=rec["gt_idx"][keep])
    matched = {c: set(out[c]["gt_idx"][out[c]["status"] == 1].tolist()) for c in out}
    elig = np.flatnonzero(gg["eligible"])
    cen = gg["center"]
    col = {}
    for j in elig:
        c_, o_ = j in matched["clean"], j in matched["outage"]
        col[j] = "#2ca02c" if (c_ and o_) else ("#d62728" if c_ else ("#1f77b4" if o_ else "#9e9e9e"))
    r = out["outage"]["rows"]; fp = out["outage"]["status"] == 2
    gc = r[:, :3].copy(); gc[:, 2] += r[:, 5] / 2
    return dict(elig=elig, col=col, centers=cen, fp_centers=gc[fp], g=g)


def draw_tile(ax_img, ax_bev, s, label, view, img_path, black):
    g = view["g"]
    if ax_img is not None:
        im = np.zeros((900, 1600, 3), np.uint8) if black else plt.imread(img_path)
        ax_img.imshow(im); ax_img.set_xticks([]); ax_img.set_yticks([]); ax_img.set_title(label, fontsize=8)
        uv, z = project_uv(g, view["centers"][view["elig"]])
        for (u, v_), j, zz in zip(uv, view["elig"], z):
            if zz > 0: ax_img.scatter([u], [v_], s=28, facecolors="none", edgecolors=view["col"][j], linewidths=1.6)
        if len(view["fp_centers"]):
            uv2, z2 = project_uv(g, view["fp_centers"]); m = (z2 > 0) & (uv2[:, 0] >= 0) & (uv2[:, 0] < 1600) & (uv2[:, 1] >= 0) & (uv2[:, 1] < 900)
            ax_img.scatter(uv2[m, 0], uv2[m, 1], marker="x", s=28, c="#ff7f0e", linewidths=1.4)
        ax_img.set_xlim(0, 1600); ax_img.set_ylim(900, 0)
    xy = ego_xy(g, view["centers"][view["elig"]])
    ax_bev.add_patch(Wedge((0, 0), 50, 90 - 35, 90 + 35, color="#fff3b0", alpha=0.5, lw=0))
    for (x, y), j in zip(xy, view["elig"]):
        ax_bev.scatter([-y], [x], s=34, facecolors="none", edgecolors=view["col"][j], linewidths=1.6)
    if len(view["fp_centers"]):
        f = ego_xy(g, view["fp_centers"]); ax_bev.scatter(-f[:, 1], f[:, 0], marker="x", s=26, c="#ff7f0e", linewidths=1.3)
    ax_bev.scatter([0], [0], marker="^", c="k", s=30); ax_bev.set_xlim(-30, 30); ax_bev.set_ylim(-8, 55); ax_bev.set_aspect("equal"); ax_bev.set_xticks([]); ax_bev.set_yticks([]); ax_bev.grid(False)


def event_samples(gt, sched, scene):
    ph = sl.phase_table(gt, sched, D)[scene]
    t = lambda s: gt[s]["ts_lidar_us"]
    by = lambda lab: sorted([s for s, L in ph.items() if lab in L], key=t)
    pre, dur = by("pre"), by("during")
    cols = [("last pre-outage", pre[-1:]), ("first during", dur[:1]), ("last during", dur[-1:]), ("restoration bin 0.0 s", by("b0.0")[:1]), ("bin 1.0 s", by("b1.0")[:1]), ("bin 3.5 s", by("b3.5")[:1])]
    return [(lab, s[0]) for lab, s in cols if s], ph


def main():
    chosen, detail = choose_events()
    print("selected events (SparseBEV, 2 s):", chosen)
    gt, sched, img, st, preds = load_state()
    os.makedirs(os.path.join(FIG, "qualitative"), exist_ok=True)
    rep = dict(rule="see module docstring", duration_s=D, events=chosen, detail=detail)
    for name, scene in chosen.items():
        cols, ph = event_samples(gt, sched, scene)
        fig, axs = plt.subplots(2, len(cols), figsize=(2.6 * len(cols), 5.6), squeeze=False)
        for c, (lab, s) in enumerate(cols):
            v = sample_view("sparsebev", s, gt, st, preds)
            black = "during" in ph[s]
            draw_tile(axs[0, c], axs[1, c], s, f"{lab}\n{'(black: outage)' if black else ''}", v, img[s], black)
        axs[0, 0].set_ylabel("CAM_FRONT (as consumed)"); axs[1, 0].set_ylabel("BEV (ego frame)")
        fig.suptitle(f"SparseBEV, 2 s outage, event '{name}' — scene {scene[:8]}…  (green: detected clean+outage, red: missed only in outage, grey: missed in both, blue: only outage detects, orange x: false positive)", fontsize=7.5)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "qualitative", f"qual_sparsebev_{name}.png")); plt.close(fig)
        # three-model BEV comparison for the same event
        pick = [c for c in cols if c[0] in ("last during", "restoration bin 0.0 s", "bin 1.0 s")]
        fig, axs = plt.subplots(3, len(pick), figsize=(2.9 * len(pick), 8.2), squeeze=False)
        for r_, (m, v_) in enumerate(MODELS.items()):
            for c, (lab, s) in enumerate(pick):
                view = sample_view(m, s, gt, st, preds)
                draw_tile(None, axs[r_, c], s, "", view, None, False)
                if r_ == 0: axs[r_, c].set_title(lab, fontsize=8)
            axs[r_, 0].set_ylabel(v_["label"])
        fig.suptitle(f"Same event, three models (BEV, 2 s outage, event '{name}', scene {scene[:8]}…; each model at its frozen threshold)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "qualitative", f"qual_models_{name}.png")); plt.close(fig)
    json.dump(rep, open(os.path.join(HERE, "artifacts/analysis/qualitative_selection.json"), "w"), indent=1)
    print(sorted(os.listdir(os.path.join(FIG, "qualitative"))))


if __name__ == "__main__":
    main()
