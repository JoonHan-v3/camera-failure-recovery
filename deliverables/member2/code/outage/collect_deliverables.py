"""Collect the Member 2 deliverables into a self-contained directory with a results summary and checksums.
Usage (project root, env streampetr_fa): python -m outage.collect_deliverables [out_dir=../deliverables/member2]
Default location: ~/Documents/UM/rob535/projects/deliverables/member2 (shared folder next to the StreamPETR project; rebuilt from scratch each run).
Copies ONLY valid (real flash-attn, TF32 off) materials; shim-based artifacts are deliberately excluded (listed in MANIFEST.md)."""
import hashlib, json, os, shutil, subprocess, sys

RUNS = ["rf_pilot_a", "rf_pilot_b", "rf_pilot_d05", "rf_pilot_d10", "rf_pilot_d20",
        "rf_val_a", "rf_val_b", "rf_val_d05", "rf_val_d10", "rf_val_d20"]
EXCLUDED = ["mini_*", "pilot_a/b/d05/d10/d20", "pilot_notf32", "pilot_shimfp16", "val_a", "val_d05", "val_notf32",
            "ref_test raw outputs (only comparison.json is included, labelled SHIM_STACK under baseline_gate/)", "isolation_{mini,pilot}.json (shim stack)", "verification_{mini,pilot}.json (shim stack)"]


def cp(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        shutil.copytree(src, dst, symlinks=False, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "fa_src"))
    else:
        shutil.copy2(src, dst)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.load(open(p))


def main(out="../deliverables/member2"):
    if os.path.exists(out):
        shutil.rmtree(out)
    os.makedirs(out)
    # ---- code, docs, scripts
    cp("README.md", f"{out}/README.md"); cp("outage", f"{out}/code/outage"); cp("process.txt", f"{out}/process.txt")
    for s in ("run_matrix_realflash.sh", "eval_runs_official.sh", "extract_trainval_keyframes.sh"):
        cp(f"artifacts/{s}", f"{out}/scripts/{s}")
    for s in ("build_infos_par.py", "merge_infos.py", "pilot_scenes.json"):
        cp(f"artifacts/trainval_infos/{s}", f"{out}/infos/{s}")
    for s in ("pilot_train_infos.pkl", "val_infos.pkl"):
        cp(f"artifacts/trainval_infos/{s}", f"{out}/infos/{s}")
    # ---- frozen schedules
    for s in ("schedule_trainval_val.json", "schedule_trainval_train.json", "schedule_mini.json"):
        cp(f"artifacts/{s}", f"{out}/schedules/{s}")
    # ---- runs (preds, audit, run.json) ; rf_val_a is a symlink to val_realflash -> copied as a real directory
    for r in RUNS:
        src = f"artifacts/runs/{r}"
        for f in ("preds.pkl", "audit.json", "run.json"):
            cp(f"{src}/{f}", f"{out}/runs/{r}/{f}")
    # ---- official metrics
    for r in ("rf_val_a", "rf_val_b", "rf_val_d05", "rf_val_d10", "rf_val_d20"):
        cp(f"artifacts/eval/{r}/metrics.json", f"{out}/official_eval/{r}/metrics.json")
        if os.path.exists(f"artifacts/eval/{r}/pts_bbox/metrics_summary.json"):
            cp(f"artifacts/eval/{r}/pts_bbox/metrics_summary.json", f"{out}/official_eval/{r}/metrics_summary.json")
    # ---- verification, isolation, controls, provenance, baseline gate evidence
    for f in ("verification_pilot_rf.json", "verification_val_rf.json", "isolation_pilot_rf.json"):
        cp(f"artifacts/{f}", f"{out}/verification/{f}")
    cp("artifacts/controls", f"{out}/verification/controls")
    cp("artifacts/flashbuild", f"{out}/provenance/flashbuild")
    cp("artifacts/upstream", f"{out}/provenance/upstream_training_log_and_config")
    cp("artifacts/ref_test/comparison.json", f"{out}/baseline_gate/upstream_control_comparison_SHIM_STACK.json")
    cp("artifacts/eval/val_realflash/metrics.json", f"{out}/baseline_gate/val_realflash_metrics.json")
    cp("artifacts/runs/val_realflash/run.json", f"{out}/baseline_gate/val_realflash_run.json")
    # ---- unit tests (fresh run)
    t = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", ".", "-t", ".", "-p", "test_*.py", "-v"],
                       capture_output=True, text=True, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    open(f"{out}/verification/unittest_output.txt", "w").write(t.stdout + t.stderr)
    tests_ok = t.returncode == 0

    # ---- results summary
    vp, vv, iso = (load(f"artifacts/{f}") for f in ("verification_pilot_rf.json", "verification_val_rf.json", "isolation_pilot_rf.json"))
    ev = {r: load(f"artifacts/eval/{r}/metrics.json") for r in ("rf_val_a", "rf_val_b", "rf_val_d05", "rf_val_d10", "rf_val_d20")}
    runinfo = {r: load(f"artifacts/runs/{r}/run.json") for r in RUNS}
    ok_prov = all(v["flash_attn"]["kind"] == "REAL flash-attn" and v["tf32_matmul"] is False and v["tf32_cudnn"] is False for v in runinfo.values())
    ext = {v["flash_attn"].get("cuda_extension_sha256") for v in runinfo.values()} - {None}
    key = lambda m, k: m["pts_bbox_NuScenes/" + k]
    summary = dict(
        frozen_config=dict(flash_attn="REAL flash-attn 0.2.2 (conda env streampetr_fa)", tf32="off", precision="fp16 per config",
                           streampetr_commit=runinfo["rf_val_b"]["streampetr_commit"], mmdet3d_commit=runinfo["rf_val_b"]["mmdet3d_commit"],
                           checkpoint_sha256=runinfo["rf_val_b"]["ckpt_sha256"], cuda_extension_sha256=sorted(ext),
                           all_runs_real_kernel_tf32_off=ok_prov),
        baseline_gate=dict(target_NDS=0.537, target_mAP=0.432, tolerance=0.010, NDS=key(ev["rf_val_a"], "NDS"), mAP=key(ev["rf_val_a"], "mAP"),
                           passed=abs(key(ev["rf_val_a"], "NDS") - 0.537) <= 0.010 and abs(key(ev["rf_val_a"], "mAP") - 0.432) <= 0.010),
        unit_tests_passed=tests_ok,
        official_whole_sequence_metrics={r: {k: key(m, k) for k in ("NDS", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE")} for r, m in ev.items()},
        pilot=dict(scenes=10, samples=402, clean_twice_max_abs_diff=vp["clean_twice_max_abs_diff"], clean_twice_samples=vp["clean_twice_identical_samples"],
                   conditions={k: dict({f: v[f] for f in ("status", "readiness", "n_paired_errors", "black_inputs_expected_vs_logged_equal", "n_black_images",
                                                     "other_cameras_corrupted", "prefix_samples", "prefix_predictions_identical", "post_start_samples",
                                                     "post_start_predictions_changed")}, scenes_with_missing_bins=len(v["missing_follow_up_bins"]))
                               for k, v in vp["conditions"].items()}),
        validation=dict(scenes=150, samples=6019, clean_twice_max_abs_diff=vv["clean_twice_max_abs_diff"], clean_twice_samples=vv["clean_twice_identical_samples"],
                        conditions={k: dict({f: v[f] for f in ("status", "readiness", "n_paired_errors", "black_inputs_expected_vs_logged_equal", "n_black_images",
                                                          "other_cameras_corrupted", "prefix_samples", "prefix_predictions_identical", "post_start_samples",
                                                          "post_start_predictions_changed")}, scenes_with_missing_bins=len(v["missing_follow_up_bins"]))
                                    for k, v in vv["conditions"].items()},
                        cam_minus_eval_skew_us=[vv["min_cam_minus_eval_skew_us"], vv["max_cam_minus_eval_skew_us"]]),
        isolation_pilot_real_model=dict({k: iso[k] for k in ("all_isolated", "fresh_B_twice_max_abs_diff", "B_after_A_vs_fresh_B_max_abs_diff",
                                                         "leak_unguarded_max_abs_diff_vs_fresh", "n_scenes")},
            leak_guard_rejected=iso["leak_control"]["rejected_by_guard"], interleave_rejected=iso["interleave_control"]["rejected"],
            per_scene_alone_vs_continuous_max_diff=max(v["max_abs_diff"] for v in iso["per_scene_alone_vs_continuous"].values())),
        causal_policy=vv["causal_policy"],
    )
    json.dump(summary, open(f"{out}/results_summary.json", "w"), indent=1)

    # ---- human-readable summary
    g = summary["baseline_gate"]; L = []
    L += ["# Member 2 deliverables — results summary", "",
          "Generated by `python -m outage.collect_deliverables`. Frozen config: real flash-attn 0.2.2, TF32 off, fp16, StreamPETR "
          f"`{summary['frozen_config']['streampetr_commit'][:7]}`, mmdet3d `{summary['frozen_config']['mmdet3d_commit'][:7]}`; "
          f"all {len(RUNS)} runs real-kernel + TF32-off: **{ok_prov}**; unit tests passed: **{tests_ok}**.", "",
          "## Baseline gate (declared before the result; tolerance unchanged)",
          f"NDS {g['NDS']:.4f} / mAP {g['mAP']:.4f} vs released 0.537 / 0.432 (tolerance ±0.010) -> **{'PASS' if g['passed'] else 'FAIL'}**", "",
          "## Official nuScenes metrics, full validation (150 scenes), whole-sequence",
          "| condition | NDS | mAP | mATE | mASE | mAOE | mAVE | mAAE |", "|---|---|---|---|---|---|---|---|"]
    for r, lab in (("rf_val_a", "clean (a)"), ("rf_val_b", "clean (b)"), ("rf_val_d05", "0.5 s outage"), ("rf_val_d10", "1.0 s outage"), ("rf_val_d20", "2.0 s outage")):
        m = summary["official_whole_sequence_metrics"][r]
        L.append(f"| {lab} | {m['NDS']:.4f} | {m['mAP']:.4f} | {m['mATE']:.4f} | {m['mASE']:.4f} | {m['mAOE']:.4f} | {m['mAVE']:.4f} | {m['mAAE']:.4f} |")
    L += ["", "Whole-sequence scores dilute a brief single-camera outage (the plan says to read them with the event curves, which are Members 3/4's work);",
          "differences of ~0.001 are within noise and no significance is claimed.", ""]
    for title, key_ in (("Pilot (10 training-split scenes, 402 samples)", "pilot"), ("Full validation (150 scenes, 6019 samples)", "validation")):
        s = summary[key_]
        L += [f"## {title}", f"Clean twice: max |diff| = {s['clean_twice_max_abs_diff']} over {s['clean_twice_samples']} samples.", "",
              "| outage | audit | paired-input errors | black images = independent recomputation | black images | other cameras touched | prefix preds identical | post-start preds changed | scenes with missing follow-up bins |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, v in s["conditions"].items():
            L.append(f"| {v.get('scenes_with_missing_bins') is not None and k} | {v['status']} | {v['n_paired_errors']} | {v['black_inputs_expected_vs_logged_equal']} | {v['n_black_images']} | "
                     f"{v['other_cameras_corrupted']} | {v['prefix_predictions_identical']}/{v['prefix_samples']} | {v['post_start_predictions_changed']}/{v['post_start_samples']} | {v['scenes_with_missing_bins']} |")
        L.append("")
    i = summary["isolation_pilot_real_model"]
    L += ["## Real-model state isolation (pilot, 10 scenes)",
          f"all_isolated = {i['all_isolated']}; each scene alone == continuous run (max diff {i['per_scene_alone_vs_continuous_max_diff']}); fresh B twice diff {i['fresh_B_twice_max_abs_diff']}; "
          f"B after A diff {i['B_after_A_vs_fresh_B_max_abs_diff']}; unguarded forced leak changes predictions by {i['leak_unguarded_max_abs_diff_vs_fresh']:.1f} "
          f"and the guard rejects it ({i['leak_guard_rejected']}); interleaved scenes rejected ({i['interleave_rejected']}).", "",
          "## Limitations (see README.md)",
          "* Missing follow-up bins are kept and reported (mostly bin 0.0; rare later bins from irregular keyframe spacing); a missing bin breaks only the adjacent recovery pair containing it.",
          f"* Causal policy: {summary['causal_policy']}",
          "* Pixel-level independent decoding was a sampled check plus the raw-JSON recomputation of expected black images.",
          "* Shim-based artifacts are invalid for study accuracy and excluded from this package.",
          "* Recovery curves, event analysis and bootstrap statistics are not part of this deliverable (Members 3/4).", ""]
    open(f"{out}/SUMMARY.md", "w").write("\n".join(L))

    # ---- manifest + checksums
    files = sorted(os.path.join(d, f) for d, _, fs in os.walk(out) for f in fs if f not in ("SHA256SUMS", "MANIFEST.md"))
    with open(f"{out}/SHA256SUMS", "w") as f:
        for p in files:
            f.write(f"{sha(p)}  {os.path.relpath(p, out)}\n")
    top = {}
    for p in files:
        top.setdefault(os.path.relpath(p, out).split("/")[0], []).append(os.path.getsize(p))
    M = ["# Package manifest", "", "| directory / file | files | size |", "|---|---|---|"]
    for k, v in sorted(top.items()):
        M.append(f"| `{k}` | {len(v)} | {sum(v) / 2**20:.1f} MiB |")
    M += ["", "Verify: `cd <this directory> && sha256sum -c SHA256SUMS`", "",
          "Built from the StreamPETR project by `python -m outage.collect_deliverables`; paths mentioned in `README.md` (a copy of the project README) are relative to that project root (`../StreamPETR/`), not to this package.", "",
          "Deliberately EXCLUDED (invalid or diagnostic, shim stack): " + ", ".join(EXCLUDED) + ".", "",
          "Not included (too large / external): raw nuScenes data (`~/storage/nuscenes*`), the StreamPETR source tree (`streampetr_src/`, pinned commit in SUMMARY), "
          "the checkpoint (`ckpts/`, sha256 in run.json), the flash-attn source (`fa_src`, commit in provenance)."]
    open(f"{out}/MANIFEST.md", "w").write("\n".join(M) + "\n")
    print("collected", len(files), "files;", f"{sum(sum(v) for v in top.values()) / 2**20:.0f} MiB;", "tests_ok", tests_ok, "gate", g["passed"], "real-kernel", ok_prov)


if __name__ == "__main__":
    main(*sys.argv[1:2])
