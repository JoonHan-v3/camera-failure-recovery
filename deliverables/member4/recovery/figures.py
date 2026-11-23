"""Figures from the saved analysis (no inference, no recomputation of statistics): artifacts/figures/*.png.
Usage (from RecoveryAnalysis/): python -m recovery.figures"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .data import MODELS, DURATIONS, BINS, POST, PHASES
from .events import CATS
from .orchestrate import inspect_run, CONDS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = os.path.join(HERE, "artifacts")
FIG = os.path.join(A, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False})
CAT_COLORS = ["#bdbdbd", "#c7e9c0", "#41ab5d", "#006d2c", "#cb181d", "#fd8d3c"]
CAT_LABELS = ["insufficient data", "no measured degradation", "damaged, recovered at first observed bin", "damaged, later recovery",
              "damaged, no confirmed recovery (complete follow-up)", "damaged, unconfirmed (incomplete follow-up; right-censored)"]
X_POST = np.array(BINS)


def J(p):
    return json.load(open(os.path.join(A, p)))


def band(ax, x, mid, lo, hi, color, label=None, ls="-", marker="o"):
    mid, lo, hi = (np.array([np.nan if v is None else v for v in a], float) for a in (mid, lo, hi))
    ax.plot(x, mid, ls, color=color, marker=marker, ms=3, lw=1.4, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0)


def series(rep, m, d, key):
    c = rep["curves"][m][str(d)]
    mid = [c[p][key] for p in POST]; lo = [c[p][key + "_ci95"][0] for p in POST]; hi = [c[p][key + "_ci95"][1] for p in POST]
    return mid, lo, hi, c["during"]


def fig_loss(rep):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    for ax, d in zip(axs, DURATIONS):
        for m, v in MODELS.items():
            mid, lo, hi, dur = series(rep, m, d, "loss")
            band(ax, X_POST, mid, lo, hi, v["color"], v["label"])
            ax.errorbar([-0.7], [dur["loss"]], yerr=[[dur["loss"] - dur["loss_ci95"][0]], [dur["loss_ci95"][1] - dur["loss"]]], fmt="s", color=v["color"], ms=5, capsize=2)
        ax.axhline(0.05, color="k", ls="--", lw=0.8); ax.axhline(0, color="k", lw=0.6)
        ax.axvline(3.5, color=MODELS["sparsebev"]["color"], ls=":", lw=1.0)
        ax.set_xticks([-0.7] + list(X_POST[::2])); ax.set_xticklabels(["during"] + [f"{b:.1f}" for b in X_POST[::2]])
        ax.set_title(f"{d:g} s outage"); ax.set_xlabel("evaluation time after restoration (s)")
    axs[0].set_ylabel("pooled recall loss  L = R_clean − R_outage"); axs[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Front-region recall loss during and after the outage (pooled over 150 scenes, 95% scene-bootstrap bands; dashed = 0.05 margin; dotted = SparseBEV history window clears ≈3.5 s)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_loss_curves.png")); plt.close(fig)


def fig_recall(rep):
    fig, axs = plt.subplots(3, 3, figsize=(12, 8), sharey="row")
    for r, (m, v) in enumerate(MODELS.items()):
        for c, d in enumerate(DURATIONS):
            ax = axs[r, c]
            for key, col, lab, ls in (("recall_clean", "#555555", "clean", "-"), ("recall_outage", v["color"], "outage", "-")):
                mid, lo, hi, dur = series(rep, m, d, key)
                band(ax, X_POST, mid, lo, hi, col, lab, ls)
                ax.errorbar([-0.7], [dur[key]], yerr=[[dur[key] - dur[key + "_ci95"][0]], [dur[key + "_ci95"][1] - dur[key]]], fmt="s", color=col, ms=5, capsize=2)
            ax.set_xticks([-0.7] + list(X_POST[::2])); ax.set_xticklabels(["during"] + [f"{b:.1f}" for b in X_POST[::2]])
            if r == 0: ax.set_title(f"{d:g} s outage")
            if c == 0: ax.set_ylabel(f"{v['label']}\nfront-region recall")
            if r == 2: ax.set_xlabel("evaluation time after restoration (s)")
            if r == 0 and c == 0: ax.legend(fontsize=8)
    fig.suptitle("Absolute pooled recall on the same eligible objects (clean vs outage), 95% scene-bootstrap bands", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_recall_clean_vs_outage.png")); plt.close(fig)


def fig_events(rep):
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, d in zip(axs, DURATIONS):
        x = np.arange(len(MODELS)); bottom = np.zeros(len(MODELS))
        for k, (cn, col, lab) in enumerate(zip(CATS, CAT_COLORS, CAT_LABELS)):
            h = np.array([rep["events"][m][str(d)]["category_counts"][cn] for m in MODELS], float)
            ax.bar(x, h, bottom=bottom, color=col, label=lab, edgecolor="white", lw=0.5)
            for xi, (b, hh) in enumerate(zip(bottom, h)):
                if hh >= 8: ax.text(xi, b + hh / 2, f"{int(hh)}", ha="center", va="center", fontsize=7, color="white" if k in (3, 4) else "black")
            bottom += h
        ax.set_xticks(x); ax.set_xticklabels([MODELS[m]["label"] for m in MODELS]); ax.set_title(f"{d:g} s outage (150 events)")
        for xi, m in enumerate(MODELS):
            e = rep["events"][m][str(d)]["confirmed_onset_by_4s_among_damaged"]; cf = rep["events"][m][str(d)]["damaged_with_complete_followup"]
            ax.text(xi, 152, f"onset ≤4 s: {e['numerator']}/{e['denominator']}\ncomplete follow-up: {cf['confirmed_onset']}/{cf['n']}", ha="center", fontsize=6.5)
        ax.set_ylim(0, 178); ax.grid(False)
    axs[0].set_ylabel("events (scenes)")
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.suptitle("Per-event categories (frozen rules: damage if pooled loss during > 0.05 with ≥5 eligible objects; 2-bin confirmed recovery)", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(os.path.join(FIG, "fig_event_categories.png")); plt.close(fig)


def fig_cumulative(rep):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    xs = list(BINS[:-1])
    for ax, d in zip(axs, DURATIONS):
        for m, v in MODELS.items():
            c = rep["events"][m][str(d)]["cumulative_confirmed_onset_fraction_by_time_s"]
            band(ax, xs, [c[f"{b:.1f}"]["value"] for b in xs], [c[f"{b:.1f}"]["ci95"][0] for b in xs], [c[f"{b:.1f}"]["ci95"][1] for b in xs], v["color"], v["label"], marker="s")
        ax.set_title(f"{d:g} s outage"); ax.set_xlabel("confirmed recovery onset time (s after restoration)"); ax.set_ylim(0, 1.02)
    axs[0].set_ylabel("fraction of damaged events\nwith confirmed onset by t"); axs[0].legend(fontsize=8)
    fig.suptitle("Cumulative confirmed-onset curve (censored events stay in the denominator; 95% scene-bootstrap bands)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_cumulative_onset.png")); plt.close(fig)


def fig_common(rep):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    ax = axs[0]; w = 0.25
    for i, m in enumerate(MODELS):
        vals, lo, hi = [], [], []
        for d in DURATIONS:
            e = rep["common_events"][str(d)]["per_model"][m]["confirmed_onset_by_4s"]
            vals.append(e["value"]); lo.append(e["value"] - e["ci95"][0]); hi.append(e["ci95"][1] - e["value"])
        ax.bar(np.arange(3) + (i - 1) * w, vals, w, yerr=[lo, hi], color=MODELS[m]["color"], capsize=2, label=MODELS[m]["label"])
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{d:g} s\n(n={rep['common_events'][str(d)]['n_common_damaged_events']} common events)" for d in DURATIONS])
    ax.set_ylabel("confirmed onset by 4 s"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.set_title("Common events (damaged in all three models)")
    ax = axs[1]; pairs = list(next(iter(rep["common_events"].values()))["pairwise"]); k = 0
    for d in DURATIONS:
        for p in pairs:
            e = rep["common_events"][str(d)]["pairwise"][p]
            ax.errorbar([e["onset_fraction_difference"]], [k], xerr=[[e["onset_fraction_difference"] - e["ci95"][0]], [e["ci95"][1] - e["onset_fraction_difference"]]], fmt="o", color="k", capsize=2, ms=3)
            k += 1
    ax.axvline(0, color="k", lw=0.7); ax.set_yticks(range(k)); ax.set_yticklabels([f"{p.replace('_minus_', ' − ')} ({d:g} s)" for d in DURATIONS for p in pairs], fontsize=6.5)
    ax.set_xlabel("difference in confirmed-onset fraction (paired scene bootstrap, 95%)"); ax.set_title("Pairwise differences"); ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_common_events.png")); plt.close(fig)


def fig_precision(rep):
    fig, axs = plt.subplots(2, 3, figsize=(12, 6), sharex=True)
    for c, d in enumerate(DURATIONS):
        for m, v in MODELS.items():
            for key, ls in (("precision_clean", "--"), ("precision_outage", "-")):
                mid, lo, hi, dur = series(rep, m, d, key)
                axs[0, c].plot(X_POST, mid, ls, color=v["color"], marker="o", ms=2.5, lw=1.2, label=f"{v['label']} {key.split('_')[1]}" if c == 0 else None)
            fc = [rep["curves"][m][str(d)][p]["false_positives_clean"] for p in POST]; fo = [rep["curves"][m][str(d)][p]["false_positives_outage"] for p in POST]
            axs[1, c].plot(X_POST, np.array(fo) - np.array(fc), color=v["color"], marker="o", ms=2.5, lw=1.2)
        axs[0, c].set_title(f"{d:g} s outage"); axs[1, c].set_xlabel("evaluation time after restoration (s)"); axs[1, c].axhline(0, color="k", lw=0.6)
    axs[0, 0].set_ylabel("pooled precision (dashed clean, solid outage)"); axs[1, 0].set_ylabel("false positives: outage − clean (count)")
    axs[0, 0].legend(fontsize=6, ncol=2)
    fig.suptitle("Region precision and false-positive change after restoration (frozen per-model thresholds; operating points differ across models)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_precision_false_positives.png")); plt.close(fig)


def fig_class():
    cs = J("analysis/classes_sameobject.json"); classes = cs["classes"]
    fig, axs = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    for ax, ph in zip(axs, ("during", "b0.0")):
        w = 0.27
        for i, (m, v) in enumerate(MODELS.items()):
            vals, lo, hi = [], [], []
            for cn in classes:
                r = cs["per_class"][m]["2.0"][ph][cn]
                if r["loss"] is None: vals.append(np.nan); lo.append(0); hi.append(0)
                else: vals.append(r["loss"]); lo.append(r["loss"] - r["loss_ci95"][0]); hi.append(r["loss_ci95"][1] - r["loss"])
            ax.bar(np.arange(len(classes)) + (i - 1) * w, vals, w, yerr=[lo, hi], color=v["color"], capsize=1.5, label=v["label"], error_kw=dict(lw=0.8))
        ax.axhline(0, color="k", lw=0.6); ax.set_ylabel(f"recall loss ({'during outage' if ph == 'during' else 'first post-restoration bin'})")
    n = {cn: cs["per_class"]["sparsebev"]["2.0"]["during"][cn]["n_gt"] for cn in classes}
    axs[1].set_xticks(range(len(classes))); axs[1].set_xticklabels([f"{cn}\n(n={n[cn]})" for cn in classes], fontsize=7, rotation=0)
    axs[0].legend(fontsize=8); fig.suptitle("Per-class recall loss, 2 s outage (eligible objects n during the outage; wide intervals = few objects)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_per_class_loss.png")); plt.close(fig)


def fig_sameobject():
    cs = J("analysis/classes_sameobject.json")
    fig, ax = plt.subplots(figsize=(9, 3.8)); x = 0
    labels = []
    for m, v in MODELS.items():
        for d in DURATIONS:
            c = cs["same_object"][m][str(d)]["counts"]; tot = c["continuing_clean_detected_both"]
            parts = [c["cont_detected_d_detected_after"], c["cont_missed_d_detected_after"], c["cont_missed_d_missed_after"], c["cont_detected_d_missed_after"]]
            bottom = 0
            for p_, col, lab in zip(parts, ["#a1d99b", "#31a354", "#de2d26", "#fdae6b"], ["detected both (outage unaffected)", "missed during → detected at restored later sample (recovered)", "missed during → missed at restored later sample (persisting)", "detected during → missed later"]):
                ax.bar(x, p_ / tot, bottom=bottom, color=col, label=lab if x == 0 else None, width=0.8); bottom += p_ / tot
            labels.append(f"{v['label'][:6]}\n{d:g} s\nn={tot}"); x += 1
        x += 0.5
    ax.set_xticks([i + (i // 3) * 0.5 for i in range(9)]); ax.set_xticklabels(labels, fontsize=6.5); ax.set_ylabel("fraction of continuing objects\n(clean run detected at both samples)")
    ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, 1.28), ncol=2); ax.grid(False)
    fig.suptitle("Same-object diagnostic: last during-outage sample → first strictly later sample with restored front image", fontsize=9, y=1.0)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_same_object.png")); plt.close(fig)


def fig_profile():
    pr = {m: J(f"profile/{m}.json") for m in MODELS}
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
    names = [MODELS[m]["label"] for m in MODELS]; cols = [MODELS[m]["color"] for m in MODELS]
    L = [pr[m]["latency_ms"] for m in MODELS]
    axs[0].bar(names, [l["steady_median"] for l in L], color=cols, yerr=[[0] * 3, [l["steady_p95"] - l["steady_median"] for l in L]], capsize=3)
    for i, l in enumerate(L): axs[0].text(i, l["steady_median"] + 4, f"{l['steady_median']:.0f} ms\n{1000 / l['steady_median']:.1f} FPS", ha="center", fontsize=7, va="bottom")
    axs[0].set_ylabel("batch-one model latency, ms (median; bar = p95)"); axs[0].set_title("Model latency (synchronized, warm)")
    axs[1].bar(names, [pr[m]["data_loading_cpu_ms"]["median"] for m in MODELS], color=cols, alpha=0.55, label="CPU decode + pipeline")
    axs[1].bar(names, [l["steady_median"] for l in L], bottom=[pr[m]["data_loading_cpu_ms"]["median"] for m in MODELS], color=cols, label="model")
    for i, m in enumerate(MODELS): axs[1].text(i, pr[m]["end_to_end_ms_paired_median"] + 4, f"{pr[m]['end_to_end_ms_paired_median']:.0f} ms", ha="center", fontsize=7)
    axs[1].set_ylabel("end-to-end per frame, ms"); axs[1].set_title("End-to-end (paired per-frame totals, single process)"); axs[1].legend(fontsize=7)
    x = np.arange(3)
    axs[2].bar(x - 0.2, [pr[m]["peak_allocated_mb"] for m in MODELS], 0.4, color=cols, label="peak allocated")
    axs[2].bar(x + 0.2, [pr[m]["peak_reserved_mb"] for m in MODELS], 0.4, color=cols, alpha=0.5, label="peak reserved")
    axs[2].axhline(12288 - 3000, color="r", ls=":", lw=0.8); axs[2].text(2.4, 12288 - 2800, "12 GB card − desktop ≈ 9 GB", fontsize=6, color="r", ha="right")
    axs[2].set_xticks(x); axs[2].set_xticklabels(names); axs[2].set_ylabel("GPU memory, MB"); axs[2].set_title("Peak GPU memory (profile, batch 1)"); axs[2].legend(fontsize=7)
    for a in axs[:2]: a.grid(False)
    fig.suptitle("Resource use on an RTX 3060 12 GB (clean, 2 pilot scenes = 80 frames × 3 measured passes after a discarded warm-up pass; TF32 off)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_resource_profile.png")); plt.close(fig)


def fig_official():
    man = J("run_manifest.json")["runs"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for k, ax in zip(("NDS", "mAP"), axs):
        for m, v in MODELS.items():
            ys = []
            for cond, name in CONDS:
                r = [x for x in man if x["model"] == m and x["split"] == "val" and x["name"] == name][0]
                ys.append(r["official"][k])
            ax.plot(["clean", "0.5 s", "1 s", "2 s"], ys, "o-", color=v["color"], label=v["label"])
        ax.set_title(f"official {k}, whole validation sequence"); ax.set_ylabel(k)
    axs[0].legend(fontsize=8)
    fig.suptitle("Official nuScenes metrics per condition (diluted by the 150-scene average; no significance claimed)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_official_metrics.png")); plt.close(fig)


def main():
    rep = J("analysis/recovery_summary.json")
    fig_loss(rep); fig_recall(rep); fig_events(rep); fig_cumulative(rep); fig_common(rep); fig_precision(rep); fig_class(); fig_sameobject(); fig_profile(); fig_official()
    print(sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
