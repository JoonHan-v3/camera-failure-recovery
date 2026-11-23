"""Run orchestration for the 12-run study matrix (3 models x clean / 0.5 / 1 / 2 s) and the reproducible run summaries.

  python -m recovery.orchestrate status                 verify every run's integrity and write artifacts/run_manifest.json + RUN_SUMMARIES.md
  python -m recovery.orchestrate execute [--run]        list (default: dry run) or launch the missing runs via each project's own run script,
                                                        at most 2 GPU processes at once (a 12 GB card shared with the desktop fits two)
  python -m recovery.orchestrate reproduce <model> <split> <cond>   re-run ONE result from the saved configuration into a scratch directory and
                                                        compare predictions bit-for-bit and audits with the saved run
  python -m recovery.orchestrate all                    status -> verify_tables -> analysis -> deep_dive -> figures (the single evaluation command)
No inference code is duplicated here: runs are always produced by the member projects' own runners (Members 1/2/3); this module only checks,
schedules and summarizes them.
"""
import hashlib, json, os, pickle, subprocess, sys, time
import numpy as np
from .data import PROJECTS, sha256

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(HERE, "artifacts")
P = lambda *a: os.path.join(PROJECTS, *a)
CONDS = [("clean", "a"), ("0.5", "d05"), ("1", "d10"), ("2", "d20")]
MODELS = {
    "bevformer": dict(label="BEVFormer-tiny", env="bevformer", root="BEVFormer", cwd="bevformer_src", mod="outage.run_bevformer", pp="$PWD:$PWD/..",
                      run=lambda s, n: f"runs/{s}_{n}", eval=lambda n: f"eval/val_{n}", ckpt="ckpts/bevformer_tiny_epoch_24.pth", extra=[],
                      sched=lambda s: f"artifacts/schedule_trainval_{'val' if s == 'val' else 'train'}.json", expected_ckpt="7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12"),
    "streampetr": dict(label="StreamPETR R50", env="streampetr_fa", root="StreamPETR", cwd=".", mod="outage.run_streampetr", pp="streampetr_src:.",
                       run=lambda s, n: f"runs/rf_{s}_{n}", eval=lambda n: f"eval/rf_val_{n}", ckpt="ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth", extra=[],
                       sched=lambda s: f"artifacts/schedule_trainval_{'val' if s == 'val' else 'train'}.json", expected_ckpt="e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451"),
    "sparsebev": dict(label="SparseBEV R50", env="sparsebev", root="SparseBEV", cwd="sparsebev_src", mod="outage.run_sparsebev", pp="$PWD:$PWD/..",
                      run=lambda s, n: f"runs/{s}_{n}", eval=lambda n: f"eval/val_{n}", ckpt="ckpts/r50_nuimg_704x256.pth", extra=[],
                      sched=lambda s: f"artifacts/schedule_trainval_{'val' if s == 'val' else 'train'}.json", expected_ckpt="1fa2e196973f717407cbbe71a4381731ab21e5152528cdbe2e6586efcb6c0478"),
}
EXPECT = dict(val=6019, pilot=402)
BLACK = {"0.5": 168, "1": 326, "2": 635}            # val black CAM_FRONT images (independent recomputation by all members); pilot 11/21/42
BLACK_PILOT = {"0.5": 11, "1": 21, "2": 42}
K = "pts_bbox_NuScenes/"


def run_dir(m, split, name):
    return P(MODELS[m]["root"], "artifacts", MODELS[m]["run"](split, name))


def cmd_for(m, split, cond, name, out=None):
    c = MODELS[m]
    out = out or run_dir(m, split, name)
    return (f"cd {P(c['root'], c['cwd'])} && export PATH=$HOME/anaconda3/envs/{c['env']}/bin:$PATH PYTHONPATH={c['pp']} && "
            f"python -m {c['mod']} --split {split} --condition {cond} --out {out} --no-tf32")


def inspect_run(m, split, cond, name):
    d = run_dir(m, split, name)
    r = dict(model=m, split=split, condition=cond, name=name, dir=d, real_dir=os.path.realpath(d), exists=os.path.isdir(d))
    if not r["exists"]:
        return r
    files = {f: os.path.join(d, f) for f in ("preds.pkl", "audit.json", "run.json")}
    r["files_present"] = {f: os.path.isfile(p) for f, p in files.items()}
    if not all(r["files_present"].values()):
        return r
    rj = json.load(open(files["run.json"]))
    c = MODELS[m]
    ck = P(c["root"], c["ckpt"])
    r.update(sha256={f: sha256(p) for f, p in files.items()}, ckpt_sha256_recorded=rj["ckpt_sha256"], ckpt_sha256_on_disk=sha256(ck),
             n_samples=rj["n_samples"], audit_status=rj["audit_status"], readiness=rj["readiness"], tf32=[rj["tf32_matmul"], rj["tf32_cudnn"]],
             runtime_s=rj["runtime_s"], latency_median_s=rj["latency_median_s"], peak_gpu_mb=rj["peak_gpu_mb"], torch=rj["torch"], gpu=rj["gpu"],
             config=rj["config"], argv=rj["argv"], commit={k: rj[k] for k in rj if k.endswith("_commit")},
             schedule_sha256=sha256(P(c["root"], c["sched"](split))), run_json=rj)
    a = json.load(open(files["audit.json"]))
    r["n_audit_records"] = len(a["records"]); r["n_black_images"] = sum(bool(x["corrupted"]) for x in a["records"])
    mj = P(c["root"], "artifacts", c["eval"](name), "metrics.json") if split == "val" else None
    if mj and os.path.isfile(mj):
        mm = json.load(open(mj)); r["official"] = {k: mm[K + k] for k in ("NDS", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE")}; r["official_metrics_sha256"] = sha256(mj)
    checks = {}
    checks["files_present"] = True
    checks["audit_consistent"] = rj["audit_status"] == "CONSISTENT"
    checks["n_samples"] = rj["n_samples"] == EXPECT[split]
    checks["tf32_off"] = rj["tf32_matmul"] is False and rj["tf32_cudnn"] is False
    checks["checkpoint_matches_disk"] = rj["ckpt_sha256"] == r["ckpt_sha256_on_disk"]
    exp = c["expected_ckpt"]
    # full 64-hex SHA-256 identities, as accepted in the Member 1/2/3 packages (correction 4: no prefix match, no unconditional True)
    checks["checkpoint_matches_pinned_record"] = bool(exp and len(exp) == 64 and rj["ckpt_sha256"] == exp)
    want = (BLACK if split == "val" else BLACK_PILOT).get(cond, 0)
    checks["black_image_count"] = r["n_black_images"] == want
    r["checks"] = checks; r["integrity_ok"] = all(checks.values())
    return r


def status():
    os.makedirs(ART, exist_ok=True)
    runs = [inspect_run(m, s, c, n) for m in MODELS for s in ("val", "pilot") for c, n in CONDS]
    # clean-twice partner runs (repeatability)
    for m in MODELS:
        for s in ("val", "pilot"):
            runs.append(inspect_run(m, s, "clean", "b"))
    sched = {m: sha256(P(MODELS[m]["root"], MODELS[m]["sched"]("val"))) for m in MODELS}
    man = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), runs=[{k: v for k, v in r.items() if k != "run_json"} for r in runs],
               val_schedule_sha256=sched, val_schedule_identical_across_models=len(set(sched.values())) == 1,
               all_runs_ok=all(r.get("integrity_ok") for r in runs), n_runs=len(runs))
    json.dump(man, open(os.path.join(ART, "run_manifest.json"), "w"), indent=1)
    L = ["# Reproducible run summaries — 3 models x (clean, 0.5 s, 1 s, 2 s), full validation (150 scenes, 6,019 samples) + 10-scene pilot\n",
         "Generated by `python -m recovery.orchestrate status` from each run's `run.json`, `audit.json`, `preds.pkl` and official metrics (hashes in `artifacts/run_manifest.json`).\n",
         f"Val schedule sha256 identical across models: **{man['val_schedule_identical_across_models']}** (`{list(sched.values())[0][:12]}…`); every run integrity-checked: **{man['all_runs_ok']}**.\n",
         "| model | split | run | condition | samples | black images | audit | checkpoint sha256 | TF32 | runtime s | median e2e latency s | peak GPU MB | NDS | mAP | preds.pkl sha256 |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        if not r.get("integrity_ok") and "sha256" not in r:
            L.append(f"| {r['model']} | {r['split']} | {r['name']} | {r['condition']} | MISSING |||||||||||"); continue
        o = r.get("official", {})
        L.append(f"| {MODELS[r['model']]['label']} | {r['split']} | {r['name']} | {r['condition']} | {r['n_samples']} | {r['n_black_images']} | {r['audit_status']} | `{r['ckpt_sha256_recorded'][:10]}…` | "
                 f"{'off' if r['tf32']==[False, False] else 'ON'} | {r['runtime_s']:.0f} | {r['latency_median_s']:.3f} | {r['peak_gpu_mb']:.0f} | "
                 f"{o.get('NDS', float('nan')):.4f} | {o.get('mAP', float('nan')):.4f} | `{r['sha256']['preds.pkl'][:12]}…` |")
    L.append("\nRuntimes are end-to-end wall clock of the whole run (JPEG decode + pipeline + model, 1-2 concurrent processes); latencies are per-frame end-to-end, NOT the isolated batch-one latencies of `artifacts/profile/`.")
    open(os.path.join(ART, "RUN_SUMMARIES.md"), "w").write("\n".join(L) + "\n")
    bad = [r for r in runs if not r.get("integrity_ok")]
    print(f"{len(runs)} runs inspected; integrity ok: {man['all_runs_ok']}; schedules identical: {man['val_schedule_identical_across_models']}")
    for r in bad:
        print("  PROBLEM", r["model"], r["split"], r["name"], r.get("checks"))
    return man


def is_complete(m, split, cond, name):
    """A run counts as complete only if inspect_run's integrity checks all pass (a bare preds.pkl is NOT completion)."""
    r = inspect_run(m, split, cond, name)
    return bool(r.get("integrity_ok")), r


def execute(run=False, runner=None):
    """Lists (default) or launches the validation runs that are missing OR invalid. Returns dict(todo, problems, failed). Existing runs that fail their integrity
    checks are reported as PROBLEMS and are never overwritten or treated as complete; a launched subprocess with a nonzero exit status is reported as failed."""
    runner = runner or (lambda cmd: subprocess.Popen(["bash", "-lc", cmd]))
    todo, problems = [], []
    for m in MODELS:
        for c, n in CONDS:
            d = run_dir(m, "val", n)
            if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "preds.pkl")):
                todo.append((m, "val", c, n)); continue
            ok, r = is_complete(m, "val", c, n)
            if not ok:
                problems.append((m, "val", c, n, r.get("checks") or r.get("files_present")))
    for pr in problems:
        print("  PROBLEM existing run is invalid or incomplete (not overwritten):", pr)
    if not todo:
        print("no missing validation runs" if not problems else "no missing validation runs, but invalid existing runs need attention")
        return dict(todo=[], problems=problems, failed=[])
    print(f"{len(todo)} missing validation runs:")
    for m, s_, c, n in todo:
        print("  ", cmd_for(m, s_, c, n))
    if not run:
        print("(dry run; pass --run to launch, at most 2 at a time)"); return dict(todo=todo, problems=problems, failed=[])
    procs = []
    for m, s_, c, n in todo:
        while len([p for _, p in procs if p.poll() is None]) >= 2:
            time.sleep(15)
        procs.append(((m, s_, c, n), runner(cmd_for(m, s_, c, n))))
    failed = []
    for key, p in procs:
        if p.wait() != 0:
            failed.append((key, p.returncode))
    for key, ok_ in ((k, is_complete(*k)[0]) for k, _ in procs):
        if not ok_ and key not in [f[0] for f in failed]:
            failed.append((key, "integrity checks failed after run"))
    for f in failed:
        print("  FAILED", f)
    return dict(todo=todo, problems=problems, failed=failed)


def reproduce(m, split, cond):
    name = dict(CONDS)[cond]
    saved = run_dir(m, split, name)
    scratch = os.path.join(ART, "reproduction", f"{m}_{split}_{name}")
    os.makedirs(os.path.dirname(scratch), exist_ok=True)
    if os.path.isdir(scratch):
        import shutil; shutil.rmtree(scratch)
    t0 = time.time()
    log = os.path.join(ART, "logs", f"reproduce_{m}_{split}_{name}.log")
    rc = subprocess.run(["bash", "-lc", cmd_for(m, split, cond, name, out=scratch)], stdout=open(log, "w"), stderr=subprocess.STDOUT).returncode
    rep = dict(model=m, split=split, condition=cond, saved_run=saved, reproduced_run=scratch, returncode=rc, wall_s=time.time() - t0, command=cmd_for(m, split, cond, name, out=scratch))
    if rc == 0:
        a, b = pickle.load(open(os.path.join(saved, "preds.pkl"), "rb")), pickle.load(open(os.path.join(scratch, "preds.pkl"), "rb"))
        same_keys = a.keys() == b.keys()
        d = 0.0 if same_keys else float("inf")
        n_shape = 0
        for k in a:
            if k in b:
                x, y = np.asarray(a[k], float), np.asarray(b[k], float)
                if x.shape != y.shape:
                    n_shape += 1
                elif x.size:
                    d = max(d, float(np.abs(x - y).max()))
        aa, bb = json.load(open(os.path.join(saved, "audit.json"))), json.load(open(os.path.join(scratch, "audit.json")))
        FIELDS_ = ("eval_ts_us", "scene", "sample", "cam", "ts_us", "slot", "keyframe", "path", "corrupted", "all_black", "sha1_src", "sha1")
        key = lambda a_: sorted(tuple(r[f_] for f_ in FIELDS_) for r in a_["records"])       # every record field, incl. pixel SHA-1 before (src) and after corruption
        if not all(f_ in aa["records"][0] for f_ in FIELDS_):
            raise RuntimeError("audit schema changed: update FIELDS_")
        ra, rb = json.load(open(os.path.join(saved, "run.json"))), json.load(open(os.path.join(scratch, "run.json")))
        rep.update(n_samples=len(a), same_sample_set=same_keys, audit_records_compared=len(aa["records"]), audit_fields_compared=list(FIELDS_), audit_comparison="all record fields incl. pixel SHA-1 before (sha1_src) and after (sha1) corruption", rows_shape_mismatch_samples=n_shape, max_abs_pred_diff=d, bit_identical=bool(same_keys and n_shape == 0 and d == 0.0),
                   audit_records_identical=key(aa) == key(bb), ckpt_sha256_same=ra["ckpt_sha256"] == rb["ckpt_sha256"], audit_status=rb["audit_status"],
                   preds_sha256_saved=sha256(os.path.join(saved, "preds.pkl")), preds_sha256_reproduced=sha256(os.path.join(scratch, "preds.pkl")))
    os.makedirs(os.path.join(ART, "reproduction"), exist_ok=True)
    json.dump(rep, open(os.path.join(ART, "reproduction", f"reproduction_{m}_{split}_{name}.json"), "w"), indent=1)
    print({k: v for k, v in rep.items() if k not in ("command",)})
    return rep


def recompare(m, split, cond):
    """Re-run ONLY the comparison for an existing scratch reproduction (no inference)."""
    name = dict(CONDS)[cond]
    saved, scratch = run_dir(m, split, name), os.path.join(ART, "reproduction", f"{m}_{split}_{name}")
    rep = json.load(open(os.path.join(ART, "reproduction", f"reproduction_{m}_{split}_{name}.json")))
    aa, bb = json.load(open(os.path.join(saved, "audit.json"))), json.load(open(os.path.join(scratch, "audit.json")))
    FIELDS_ = ("eval_ts_us", "scene", "sample", "cam", "ts_us", "slot", "keyframe", "path", "corrupted", "all_black", "sha1_src", "sha1")
    key = lambda a_: sorted(tuple(r[f_] for f_ in FIELDS_) for r in a_["records"])
    rep.update(audit_records_identical=key(aa) == key(bb), audit_records_compared=len(aa["records"]), audit_fields_compared=list(FIELDS_),
               audit_comparison="all record fields incl. pixel SHA-1 before (sha1_src) and after (sha1) corruption")
    json.dump(rep, open(os.path.join(ART, "reproduction", f"reproduction_{m}_{split}_{name}.json"), "w"), indent=1)
    print(m, rep["audit_records_identical"], rep["audit_records_compared"])


def integrity_gate(man):
    """Raise SystemExit (nonzero) unless every run passes its checks, the validation schedules agree, and every analysis input exists (correction 4)."""
    errs = []
    if not man.get("all_runs_ok"):
        errs.append("not all runs pass their integrity checks: " + ", ".join(f"{r['model']}/{r['split']}/{r['name']}" for r in man["runs"] if not r.get("integrity_ok")))
    if not man.get("val_schedule_identical_across_models"):
        errs.append("validation schedules differ across models")
    need = {c for c in ("a", "d05", "d10", "d20")}
    have = {(r["model"], r["name"]) for r in man["runs"] if r["split"] == "val" and r.get("integrity_ok")}
    for m in MODELS:
        miss = [n for n in need if (m, n) not in have]
        if miss:
            errs.append(f"{m}: required validation runs missing/invalid: {sorted(miss)}")
        pk = os.path.join(P("SparseBEV", "artifacts/event_stats"), f"per_scene_bins_{m}_val.csv")
        if not os.path.isfile(pk):
            errs.append(f"missing analysis input {pk}")
    if errs:
        for e in errs:
            print("INTEGRITY GATE FAILED:", e, file=sys.stderr)
        raise SystemExit(2)
    return True


def all_(status_fn=None, runner=None):
    man = (status_fn or status)()
    integrity_gate(man)                                     # stop BEFORE any analysis if a run failed or the schedules disagree
    run = runner or (lambda s: subprocess.run(s, cwd=HERE, check=True))
    steps = [["python", "-m", "recovery.verify_tables", "artifacts/analysis/tables_verification.json"], ["python", "-m", "recovery.analysis"],
             ["python", "-m", "recovery.deep_dive"], ["python", "-m", "recovery.figures"]]
    for s in steps:
        print(">>", " ".join(s)); run(s)
    return man


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a or a[0] == "status":
        m_ = status()
        sys.exit(0 if (m_["all_runs_ok"] and m_["val_schedule_identical_across_models"]) else 1)
    elif a[0] == "execute":
        r = execute("--run" in a)
        sys.exit(1 if (r["failed"] or r["problems"]) else 0)
    elif a[0] == "reproduce":
        reproduce(a[1], a[2], a[3])
    elif a[0] == "recompare":
        recompare(a[1], a[2], a[3])
    elif a[0] == "all":
        all_()
    else:
        sys.exit(__doc__)
