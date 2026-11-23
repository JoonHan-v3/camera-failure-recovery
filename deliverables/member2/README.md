# StreamPETR front-camera outage study — Member 2 deliverable

Camera Failure and Recovery study plan (`Camera_Failure_and_Recovery_Study_Plan.pdf`), **Member 2**:
*StreamPETR integration; shared timestamp-corruption code and state-reset checks; a tested outage schedule and input audit.*

This directory contains the shared outage utilities (`outage/`), the pinned real StreamPETR source/checkpoint, the frozen
schedules, and every real-model run, audit and verification report produced so far.

**Status (2026-11-21).** The scoped Member 2 handoff (integration, corruption, schedule, input audit, state-reset/isolation) was
accepted by the Inspector (see `process.txt`, 19th review) under the **frozen configuration below**. Full-validation clean
baseline gate: **PASSED** (NDS 0.5370 / mAP 0.4332 vs released 0.537 / 0.432, tolerance 0.010). Not part of this deliverable:
recovery curves, event analysis and bootstrap statistics (Members 3/4).

> **Frozen configuration — do not change without re-passing the baseline gate:**
> real **flash-attn 0.2.2** (conda env `streampetr_fa`), **TF32 off** (`--no-tf32`), fp16 per config, StreamPETR `95f6470`,
> mmdet3d `v1.0.0rc6`, config `stream_petr_r50_flash_704_bs2_seq_90e`, checkpoint `ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth`.
> The pure-PyTorch attention shim (`outage/shims/`) is **not** equivalent (NDS 0.509 / mAP 0.389); every shim-based artifact
> (`mini_*`, `pilot_*`, `val_a`, `val_d05`, `val_notf32`, `pilot_notf32`, `pilot_shimfp16`, `ref_test`) is **invalid for study
> accuracy** and kept only as diagnostic evidence.

## Methodology
The procedure, in the order it was carried out. Details are in the numbered sections below; the full evaluation method is §3.

1. **Freeze the inputs.**
   * Model: the released StreamPETR R50 checkpoint (`stream_petr_r50_flash_704_bs2_seq_90e`), with no retraining, at StreamPETR
     `95f6470` and mmdet3d v1.0.0rc6.
   * Scenes: the full nuScenes validation set (150 scenes, 6,019 samples), plus a pilot of 10 training scenes chosen by a fixed
     rule before any result was seen.
   * Info files: built from the raw tables and cross-checked against the raw nuScenes JSON (6,421/6,421 samples).
   * Hashes: checkpoint sha256, commits and the flash-attn extension hash are recorded in every `run.json` (§2.3, §3.1–3.2).
2. **Build the shared outage schedule from raw timestamps.**
   * Start: t_start = CAM_FRONT sensor timestamp of keyframe 6.
   * Durations: 0.5, 1 and 2 s, all from the same start, and the same start for every model.
   * Eligibility: a scene is kept only if t_start + 2 s + 4.5 s of follow-up fits before its last front keyframe. All 150 val
     scenes qualify.
   * Frozen: the schedule regenerates byte-identically and is never changed using results (§2.4, §3.3).
3. **Write the shared corruption, audit and state code.** Member 1 reuses this code byte-identical, and it is intended for the other models as well.
   * Window: every CAM_FRONT image with sensor timestamp in the half-open window `[t_start, t_start+d)` is replaced by
     black, compared in exact integer µs.
   * Placement: applied to raw pixels right after image loading, before resize, normalization and features. Other cameras,
     calibration and ego motion are untouched.
   * Tests: 44 unit tests (§2.2, §3.4).
4. **Control the recurrent memory.** StreamPETR carries memory within a scene.
   * Reset: the runner clears both the head memory and `prev_scene_token` before each run, because the model's own reset
     leaves the token.
   * Guards: an ordering guard rejects revisited or interleaved scenes, and a model-state guard requires fresh memory at every
     scene start.
   * Separation: each condition runs in its own process, in timestamp order, with batch 1 (§3.5).
5. **Run the matrix.** On each split there are 5 runs: clean twice, then 0.5, 1 and 2 s outages.
   * Settings: the frozen stack (real flash-attn 0.2.2, TF32 off, fp16 per the config).
   * Predictions: 300 boxes per sample are kept, in the common prediction format.
   * Audit: every image the model consumed is logged (§2.5, §2.5b).
6. **Audit the inputs independently of the model** (§3.6):
   * Coverage: matches a manifest built from the raw nuScenes JSON (36,114 val images).
   * Black set: equals an independent recomputation (168/326/635 on val), with no other camera touched.
   * Paired inputs: everything outside the window is hash-identical between clean and outage runs.
   * Timing: the 50 ms causal allowance holds.
   * Follow-up bins: restoration offsets and bins are computed, and missing bins are kept and reported.
7. **Verify the model's behaviour** (§3.7):
   * Repeatability: the two clean runs must be bit-identical.
   * Prefix and post-start: predictions before the outage must be identical to clean, and predictions after the start must
     change.
   * Isolation: real-model tests show each scene run alone equals the continuous run, with negative controls (a forced state
     leak and interleaved scenes) that must be rejected.
8. **Pass the baseline gate before trusting outage results.** The pass/fail criterion was written down before the result was
   seen: |ΔNDS| and |ΔmAP| ≤ 0.010 against the released 0.537 / 0.432.
   * Clean run: scored with the official nuScenes evaluation.
   * Discarded stack: a pure-PyTorch attention stand-in failed the gate and was thrown out.
   * Real kernel: passed. The repo's own `tools/test.py` reproduces the runner box-for-box (§2.6, §3.8).
9. **Report the official metrics.** Whole-sequence NDS, mAP and TP errors are reported for each condition as context only,
   with no significance claim (§3.9).

   Event-level recovery, thresholds and bootstrap are not part of this method (Members 3/4). The method's limits are in §3.10.

---

## 1. Project structure

```
StreamPETR/
├── README.md                      this file
├── Camera_Failure_and_Recovery_Study_Plan.pdf
├── process.txt                    Generator/Inspector review log (19 inspections)
├── outage/                        shared outage code (python package) + its detailed engineering log
│   ├── README.md                  full round-by-round log: design decisions, evidence, limitations (read this for details)
│   ├── schedule.py                frozen schedule rule (integer-us, half-open windows), save/load
│   ├── corruption.py              black CAM_FRONT in [t_start, t_start+d) by sensor timestamp
│   ├── streampetr_adapter.py      CorruptFrontCamera: validated, audited corruption step (ts_mode abs_s for StreamPETR)
│   ├── mm_pipeline.py             registers OutageFrontCamera in the mmdet3d pipeline (right after image loading)
│   ├── audit.py                   InputAudit: per-image records, manifest/sample-metadata reconciliation, causal policy,
│   │                              restoration offsets, missing bins, compare_runs (paired clean/outage), save/load
│   ├── state.py                   scene-order / state-reset guard, prediction comparison helpers
│   ├── nusc_ref.py                independent reference: manifest, sample->eval-time, keyframe timestamps from raw nuScenes JSON
│   ├── make_schedules.py          freezes the shared schedules from raw nuScenes tables
│   ├── run_streampetr.py          REAL runner: builds dataset+model, hook, audit, predictions, run.json provenance
│   ├── verify_runs.py             cross-run verification (clean-twice, black sets, prefix identity, bins, offsets)
│   ├── isolation.py               real-model state tests (alone==continuous, fresh/after-A, leak + interleave negative controls)
│   ├── eval_official.py           official nuScenes detection eval of saved predictions
│   ├── controls/                  shim_fidelity.py, decode_inventory.py (reproducible sanity controls)
│   ├── shims/flash_attn/          DIAGNOSTIC pure-PyTorch flash-attn stand-in (NOT for study use)
│   └── tests/test_outage.py       44 unit tests (numpy only, synthetic)
├── streampetr_src/                pinned StreamPETR clone (95f6470) + mmdetection3d (v1.0.0rc6) + data/ symlinks
├── ckpts/                         released R50 checkpoint (sha256 recorded in every run.json)
└── artifacts/
    ├── schedule_{mini,trainval_val,trainval_train}.json   FROZEN schedules (val: 150 scenes, train: 700, none skipped)
    ├── trainval_infos/            StreamPETR info pkls (pilot_train 402 samples/10 scenes, val 6019/150), pilot_scenes.json,
    │                              build_infos_par.py (8-way slicing), merge_infos.py (merge + cross-check vs raw JSON)
    ├── runs/<name>/               one dir per inference run: preds.pkl, audit.json, run.json (+ infos_abs.pkl for mini)
    │     rf_pilot_{a,b,d05,d10,d20}   REAL-kernel pilot: clean twice + 0.5/1/2 s       (VALID)
    │     rf_val_{a,b,d05,d10,d20}     REAL-kernel full val; rf_val_a -> val_realflash  (VALID)
    │     mini_*, pilot_*, val_a, val_d05, val_notf32, ...   shim-based                  (INVALID, diagnostic only)
    ├── eval/<run>/               official nuScenes metrics (val_realflash = baseline gate; others diagnostic)
    ├── verification_{pilot,val}_rf.json    cross-run verification of the valid runs
    ├── isolation_pilot_rf.json    real-model state/isolation report (valid); isolation_{mini,pilot}.json are shim-based
    ├── controls/                  shim_fidelity.json, decode_inventory.json
    ├── flashbuild/                real flash-attn build: PROVENANCE.txt, build.log, env exports, patch (empty)
    ├── upstream/, ref_test/       released training log + config diff; upstream tools/test.py control (bit-identical to runner)
    └── run_matrix_realflash.sh    THE runner script (fail-fast). Other run_*.sh / rf_val_*.sh / eval_when_ready.sh are historical;
                                   run_matrix.sh, run_matrix_mini.sh, run_matrix_pilot_val.sh use the shim env — do NOT use them
```

The checksummed **deliverables package is outside this directory**: `~/Documents/UM/rob535/projects/deliverables/member2/`
(a copy of the valid materials above plus `SUMMARY.md`, `results_summary.json`, `MANIFEST.md`, `SHA256SUMS`; see section 2.6b).

External data (not in this directory): `~/storage/nuscenes_dl` (downloaded tarballs, 298 GB), `~/storage/nuscenes`
(v1.0-mini, trainval metadata), `~/storage/nuscenes_trainval_keyframes` (52 GB, extracted CAM_*/LIDAR_TOP keyframes),
`~/storage/nuscenes_trainval_root` (symlink root used by the converter).

---

## 2. How to run

Hardware used: 1× RTX 3060 12 GB, 128 CPU threads, 219 GB RAM. One val run ≈ 45 min alone (≈0.45 s/frame, **CPU/decode bound**).
Each process holds ≈3.5 GB of GPU memory: **run at most 3 inference processes at once** (5 crashed with CUDA OOM).

### 2.1 One-time environment (already done on this machine)

```bash
# env 1: conda env "streampetr" (py3.8): torch==1.9.0+cu111 torchvision==0.10.0+cu111 numpy==1.23.5 timm==0.6.12 einops fvcore ipython
#        mmcv-full==1.6.0 (cu111/torch1.9.0 wheel) mmdet==2.28.2 mmsegmentation==0.30.0 nuscenes-devkit ; mmdet3d v1.0.0rc6 (pip install -e)
# env 2 (USED FOR ALL STUDY RUNS): clone it and add the real kernel
conda create -n streampetr_fa --clone streampetr
conda install -n streampetr_fa -c nvidia/label/cuda-11.8.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvrtc-dev
pip install ninja      # (in streampetr_fa)
git clone https://github.com/HazyResearch/flash-attention fa_src && cd fa_src && git checkout v0.2.2 && git submodule update --init --recursive
CUDA_HOME=$HOME/anaconda3/envs/streampetr_fa CC=gcc-11 CXX=g++-11 CUDAHOSTCXX=g++-11 MAX_JOBS=24 \
  PATH=$HOME/anaconda3/envs/streampetr_fa/bin:$PATH pip install -v --no-build-isolation .
```
Exact commands, toolchain versions, extension hash and env exports: `artifacts/flashbuild/PROVENANCE.txt`,
`pip_freeze_streampetr_fa.txt`, `conda_export_streampetr_fa.txt`. (System `g++-11` is required; conda gcc conflicted on binutils.)

Common shell setup for everything below (run from this directory):
```bash
export PATH=$HOME/anaconda3/envs/streampetr_fa/bin:$PATH
export PYTHONPATH=streampetr_src:.          # never use `python -I` (it ignores PYTHONPATH); do NOT put outage/shims on the path
```

### 2.2 Unit tests (no GPU, numpy only)
```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -t . -p 'test_*.py'      # 44 tests, ~0.05 s
```

### 2.3 Data preparation (only if rebuilding from the downloaded tarballs)
```bash
# metadata + mini (small):
tar -xzf ~/storage/nuscenes_dl/v1.0-trainval_meta.tgz -C ~/storage/nuscenes
tar -xzf ~/storage/nuscenes_dl/v1.0-mini.tgz          -C ~/storage/nuscenes
# trainval keyframe images only, into a NEW empty dir (untrusted archives), ~20 min:
./artifacts/extract_trainval_keyframes.sh
mkdir ~/storage/nuscenes_trainval_root && cd ~/storage/nuscenes_trainval_root \
  && ln -s ../nuscenes_trainval_keyframes/samples samples && ln -s ../nuscenes/v1.0-trainval v1.0-trainval \
  && ln -s ../nuscenes/maps maps && cd -
# StreamPETR info files for the 10 frozen pilot scenes (every 70th training scene by name) + all 150 val scenes, 8 slices (~3 min):
cd streampetr_src/tools && for w in 0 1 2 3 4 5 6 7; do python ../../artifacts/trainval_infos/build_infos_par.py $w 8 & done; wait; cd ../..
python artifacts/trainval_infos/merge_infos.py        # merges + asserts agreement with raw nuScenes JSON (6421/6421 samples)
# evaluator needs these links inside streampetr_src/data/nuscenes: v1.0-trainval -> ~/storage/nuscenes/v1.0-trainval
# (mini infos: tools/create_data_nusc.py --root-path ../data/nuscenes --version v1.0-mini --out-dir ../data/nuscenes)
```
(`artifacts/trainval_infos/build_infos.py` is the superseded single-process builder — do not use.)

### 2.4 Freeze the shared outage schedules (no model needed; deterministic, byte-identical regeneration verified)
```bash
python -m outage.make_schedules artifacts          # writes schedule_{mini,trainval_val,trainval_train}.json
```
Rule: `t_start` = CAM_FRONT **sensor** timestamp of keyframe 6; scene kept iff `t_start + 2.0 s + 4.5 s ≤` last front keyframe.
Outage `d ∈ {0.5, 1, 2} s` blacks every CAM_FRONT image with sensor ts in `[t_start, t_start + d)` (half-open), before resize/normalize.
Same start for every duration and (by design) every model. Never change the schedule from detection outcomes.

### 2.5 Pilot (≈10 training-split scenes): clean twice + 0.5 / 1 / 2 s  (~20 min)
```bash
./artifacts/run_matrix_realflash.sh pilot          # writes artifacts/runs/rf_pilot_{a,b,d05,d10,d20}; asserts REAL flash-attn + TF32 off
python -m outage.verify_runs artifacts/runs v1.0-trainval artifacts/verification_pilot_rf.json \
       rf_pilot_a rf_pilot_b rf_pilot_d05 rf_pilot_d10 rf_pilot_d20
python -m outage.isolation $PWD/artifacts/trainval_infos/pilot_train_infos.pkl $PWD/artifacts/runs/rf_pilot_a/preds.pkl \
       $PWD/artifacts/isolation_pilot_rf.json v1.0-trainval          # real-model state/isolation, ~8 min
```

### 2.5b Full evaluation: all 150 validation scenes (6019 samples), clean twice + 0.5 / 1 / 2 s
Five runs × ≈45 min; start **at most 3 at a time**:
```bash
rm -f artifacts/runs/rf_val_a                       # it is a symlink to val_realflash (the run that passed the baseline gate)
./artifacts/run_matrix_realflash.sh val clean a &   # -> artifacts/runs/rf_val_a   (or keep the symlink and skip this one)
./artifacts/run_matrix_realflash.sh val 0.5   d05 &
./artifacts/run_matrix_realflash.sh val 1     d10 &
wait
./artifacts/run_matrix_realflash.sh val 2     d20 &
./artifacts/run_matrix_realflash.sh val clean b &
wait
python -m outage.verify_runs artifacts/runs v1.0-trainval artifacts/verification_val_rf.json \
       rf_val_a rf_val_b rf_val_d05 rf_val_d10 rf_val_d20
```

### 2.6 Official nuScenes detection metrics (baseline gate; and per-condition metrics if wanted)
`eval_official` must run with `streampetr_src/` as the working directory and absolute paths:
```bash
R=$PWD; cd streampetr_src
PYTHONPATH=$R/streampetr_src:$R python -m outage.eval_official \
  $R/streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
  $R/artifacts/trainval_infos/val_infos.pkl  $R/artifacts/runs/rf_val_a/preds.pkl  $R/artifacts/eval/rf_val_a
cd $R
# per-condition whole-sequence metrics for the other runs (done; 4 in parallel is fine, CPU only, ~4 min):
./artifacts/eval_runs_official.sh rf_val_b rf_val_d05 rf_val_d10 rf_val_d20     # -> artifacts/eval/<run>/metrics.json
```
**Baseline gate** (declared before the result was seen; unchanged): `|ΔNDS| ≤ 0.010` and `|ΔmAP| ≤ 0.010` vs released **0.537 / 0.432**
→ `val_realflash`: 0.53704 / 0.43324 **PASS**. Upstream control: `artifacts/ref_test/` (the repo's own `tools/test.py` reproduces the
runner box-for-box).

### 2.6b Collect the deliverables package (checksummed, self-contained, valid runs only)
```bash
python -m outage.collect_deliverables               # -> ../deliverables/member2/ i.e. ~/Documents/UM/rob535/projects/deliverables/member2/
                                                    #    (README, SUMMARY.md, results_summary.json, MANIFEST.md, SHA256SUMS, ...)
(cd ../deliverables/member2 && sha256sum -c SHA256SUMS)
```
`~/Documents/UM/rob535/projects/deliverables/member2/SUMMARY.md` holds the headline tables (baseline gate, per-condition official metrics, pilot/val audit results,
isolation). Shim-based artifacts are excluded on purpose.

### 2.7 Optional controls
```bash
python -m outage.controls.decode_inventory artifacts/trainval_infos/val_infos.pkl artifacts/controls/decode_inventory.json
PYTHONPATH=streampetr_src:. python -m outage.controls.shim_fidelity artifacts/controls/shim_fidelity.json   # diagnostic only
```

### 2.8 Single run (any condition) and its options
```bash
python -m outage.run_streampetr --split {mini|pilot|val} --condition {clean|0.5|1|2} --out artifacts/runs/<name> --no-tf32 [--scenes tok1,tok2]
```
Always pass `--no-tf32`. Each run writes `preds.pkl` (`{sample_token: rows[x,y,z,w,l,h,yaw,vx,vy,label,score]}`), `audit.json`,
`run.json` (provenance: checkpoint sha256, commits, flash-attn module/version/extension hash, TF32 flags, GPU, argv, timings, states).

---

## 3. Evaluation methodology

**Question the evaluation serves.** Does a controlled black-image outage of CAM_FRONT change StreamPETR's outputs, and can the
input pipeline be trusted to apply that outage exactly where intended (and nowhere else)? Member 2's part is the *physical outage
and its audit*; downstream recovery statistics are Members 3/4's. Every comparison is **paired**: the same scenes, same frames,
same model, same settings, differing only in the outage.

**3.1 Data and scene sets**
* nuScenes v1.0-trainval, keyframe camera images only (StreamPETR consumes the six keyframe cameras per sample and no image sweeps).
* **Full evaluation set:** all **150 validation scenes, 6019 samples** (36,114 images), complete scenes, no subset.
* **Pilot set:** **10 training-split scenes** chosen by a fixed rule before any result was seen (every 70th scene of the sorted training
  list, `artifacts/trainval_infos/pilot_scenes.json`); used only for pilot decisions/checks, never for the evaluation numbers.
* nuScenes mini was used only for early debugging (shim stack, now invalid).

**3.2 Model and frozen numerics.** Released StreamPETR R50, 900 queries, 704×256 input (`stream_petr_r50_flash_704_bs2_seq_90e`),
**no retraining**. Inference is fixed to real flash-attn 0.2.2, TF32 off, fp16 per the config; batch size 1; deterministic (no test-time
randomness), so repeated runs must be bit-identical. These settings were frozen only after the baseline gate (3.8) passed.

**3.3 Outage schedule (shared, frozen, built from raw timestamps).**
* One outage per scene. `t_start` = the **CAM_FRONT sensor timestamp** (integer µs) of keyframe index 6 (six clean keyframes of warm-up).
  Camera sensor time is used, not the sample (lidar) time, because cameras are not synchronous with it (front camera is 33–43 ms earlier).
* Durations d ∈ {0.5, 1.0, 2.0} s, all starting at the **same** `t_start` (and, by design, the same for every model).
* A scene is eligible iff `t_start + 2.0 s + 4.5 s` fits before the last front keyframe (4.5 s of follow-up after the longest outage). All
  150 val scenes (and 700 train scenes) qualify; skipped scenes would be stored with reasons. Schedules are persisted
  (`artifacts/schedule_*.json`) and regenerate byte-identically (`outage.make_schedules`). The schedule is never altered using outcomes.

**3.4 Corruption rule.** Every CAM_FRONT image whose sensor timestamp lies in the **half-open window `[t_start, t_start + d)`** is replaced by an
all-black image (pixel value 0, same shape and dtype). It is applied to the raw decoded pixels **directly after image loading and
before resize, normalization or feature extraction**, so no cached or resized copy can leak the original. Other cameras, calibration and ego
motion are untouched. Images keep the status of their own timestamps, so the outage cannot reach back into the clean prefix. (The shared
corruption code also handles timestamped image sweeps for models that use them; StreamPETR has none.)

**3.5 Inference protocol and state handling.**
* Complete scenes, strictly in timestamp order, one process per condition (clean ×2, 0.5, 1, 2 s), so no state or cache is shared between conditions.
* StreamPETR's recurrent memory persists within a scene. It is reset at every scene boundary (by the model) and **explicitly before each run**
  (the runner clears both the head memory and `prev_scene_token`, which the model's own reset does not clear).
* Two guards run during inference: an ordering guard in the data pipeline (rejects revisited/interleaved scenes and non-increasing time) and
  a model-state guard that observes the real memory buffers and requires them fresh (zero) at each scene start.
* The only temporal state is the model memory (no feature cache exists), so condition separation = separate processes + explicit reset.

**3.6 Input audit (what was consumed, and was it right).** Every consumed image is logged: scene, sample, camera, sensor timestamp, slot,
file path, evaluation timestamp, SHA-1 of the pixels **before** and **after** corruption, a corrupted flag and an all-black flag. Checks:
1. **Coverage:** records are reconciled against an *independently* built manifest from the raw nuScenes JSON tables (not from the run): no missing,
   extra or duplicate images (36,114 expected for val); the manifest and the sample→evaluation-time map are saved with their hashes.
2. **Black set:** the images that must be black are recomputed independently from the raw tables and must equal the logged set; flagged images must be
   truly all black; no non-target camera may be flagged; no unflagged image may differ from its source.
3. **Paired clean vs outage:** every image outside the window (all cameras, prefix and restored inputs) is hash-identical, and the raw source pixels of
   corrupted images equal those of the clean run.
4. **Causal alignment:** evaluation time is the sample (lidar) time; an image may be exposed at most **50 ms after** it (a frozen alignment
   allowance; the largest skew in all 204,894 trainval camera images is +49.99 ms). Later images are violations. Transport/processing delay is unmeasured.
5. **Restoration offsets and follow-up bins:** offsets are computed once per unique evaluated sample (evaluation time − restoration time). The 10 scheduled
   bins 0.0, 0.5, …, 4.5 s are assigned one-to-one to distinct evaluated samples within ±0.25 s (tie rule documented in `audit.missing_bins`). A bin with
   no sample is **kept as missing**, never filled or dropped; it breaks only the adjacent recovery pair that contains it.
6. **Status:** `CONSISTENT` = records are internally consistent and complete; `READY` additionally requires all bins present, so runs with missing
   bins read "NOT READY" by design.

**3.7 Repeatability, state-isolation and prediction checks.**
* **Clean twice:** two clean runs must give bit-identical predictions (max |diff| = 0).
* **Prefix:** predictions for samples before the outage must equal the clean run exactly; samples after the outage start must differ (confirms the outage reached the model).
* **Isolation (real model):** each scene run alone equals the same scene inside the continuous run; fresh-B twice equal; B-after-A equals fresh B; a **negative control**
  forces a state leak (it must change predictions and be rejected by the guard) and interleaved scenes must be rejected.

**3.8 Baseline gate (is the inference stack eligible?).** Before trusting any outage result, the clean full-val run is scored with the official nuScenes
detection evaluation and must match the released numbers (**NDS 0.537, mAP 0.432**) within **±0.010 each**; the tolerance was declared before seeing the result
and never relaxed. A first stack (pure-PyTorch attention stand-in) failed (0.509 / 0.389) despite passing micro-checks and was discarded; the real flash-attn kernel
passed (0.5370 / 0.4332). The repo's own `tools/test.py` was shown to give box-for-box identical output to the custom runner.

**3.9 Official metrics.** Saved predictions (`[x,y,z,w,l,h,yaw,vx,vy,label,score]` per sample token, 300 rows each) are converted by the dataset's own
`format_results` and scored with the official nuScenes detection code (`detection_cvpr_2019`: class-dependent ranges and filtering retained): NDS, mAP and the
five TP errors, once per condition on the whole validation sequence. These whole-sequence scores **dilute** a brief outage of one camera, so they are
reported as context only; no significance is claimed from differences of ~0.001. The plan's event-based analysis (front-region recall curves, damage/recovery
rules, scene-bootstrap intervals) uses the saved predictions and audits and is **outside this deliverable**.

**3.10 Limits of the methodology.** Single deterministic run per condition (repeatability checked, no run-to-run variance to estimate); sampled (24-image) independent
pixel decoding plus the raw-table recomputation rather than exhaustive re-decoding; kernel-level equivalence between the stand-in and real attention is unexplained
(the stand-in is excluded from all results); missing follow-up bins are properties of the data (irregular keyframe spacing, last black frame evaluated just before restoration).

---

## 4. Required results and materials (what must exist, where, and what "good" looks like)

| # | Deliverable (study plan: "tested outage schedule and input audit") | Location | Acceptance / observed |
|---|---|---|---|
| 1 | Pinned source, config, checkpoint, env record | `streampetr_src/` (95f6470, mmdet3d 47285b3), `ckpts/`, `artifacts/flashbuild/PROVENANCE.txt`, `run.json` | commits + ckpt sha256 recorded; real flash-attn ext sha256 `46a7049f…` |
| 2 | Frozen shared schedule from real timestamps | `artifacts/schedule_trainval_{val,train}.json`, `schedule_mini.json` | 150 val / 700 train scenes, 0 skipped; regenerates byte-identically |
| 3 | Independent consumed-input manifest + sample metadata | built per run from raw nuScenes JSON (`outage/nusc_ref.py`), saved in each `audit.json` (`expected_manifest`, `sample_metadata`, sha1 hashes) | 36,114 val images = 6019 × 6 cameras, exact coverage, 0 errors |
| 4 | Clean and 0.5/1/2 s input audits (actual timestamps, corrupted files, offsets) | `artifacts/runs/rf_*/audit.json` | status CONSISTENT; black sets = independent recomputation (168/326/635 val images); other cameras untouched; prefix inputs and predictions unchanged (900/900) |
| 5 | Repeatability / fresh state | `verification_{pilot,val}_rf.json` | clean-twice bit-identical (402 pilot, 6019 val samples, max diff 0.0) |
| 6 | State-reset / isolation evidence (real model) | `artifacts/isolation_pilot_rf.json` | alone == continuous (10 scenes, diff 0.0); fresh-B twice == B-after-A; unguarded leak changes preds (102.4) and is rejected; interleaving rejected |
| 7 | Restoration-to-evaluation offsets and follow-up bins | in each outage `audit.json` (`restoration_offsets`, `missing_bins`) | missing bins kept and reported: 14 / 26 / 34 of 150 val scenes (0.5/1/2 s), mostly bin 0.0 |
| 8 | Clean baseline check of the exact inference stack | `artifacts/eval/val_realflash/metrics.json`, `artifacts/runs/val_realflash/run.json` | NDS 0.5370 / mAP 0.4332 — gate PASSED |
| 8b | Official whole-sequence metrics per condition (full val) | `artifacts/eval/rf_val_{a,b,d05,d10,d20}/metrics.json` | clean a == b exactly; degradation small and growing with duration (see SUMMARY.md) |
| 9 | Common prediction format for the evaluator | `preds.pkl` of every run | rows `[x,y,z,w,l,h,yaw,vx,vy,label,score]` per sample token, finite, 300 rows/sample |
| 10 | Tests | `outage/tests/test_outage.py` | 44 pass |
| 11 | Engineering log with all limitations | `outage/README.md`, `process.txt` | read before reusing results |
| 12 | Checksummed deliverables package | `~/Documents/UM/rob535/projects/deliverables/member2/` (outside this project dir; 98 files, 635 MiB; `SUMMARY.md`, `MANIFEST.md`, `SHA256SUMS`) | `sha256sum -c SHA256SUMS` all OK |

### Known limitations (preserve in any downstream use)
* **Missing follow-up bins** are real data properties (the last black frame can be evaluated just before restoration; keyframe gaps up
  to 0.6 s). Rule: a missing bin breaks only the adjacent recovery pair that contains it; later valid pairs may still establish
  onset; the event is neither discarded nor read as permanent failure. Runs therefore report readiness `NOT READY` by design.
* **Causal policy:** evaluation time = sample (lidar) time; the frozen 50 ms bound (`audit.CAUSAL_TOL_US`) is a sample-alignment
  allowance (max observed trainval skew +49,992 µs), not zero-latency availability; transport/processing delay is unmeasured.
* StreamPETR consumes only the six keyframe camera images per sample (no image sweeps); history lives in recurrent head memory.
  `reset_memory()` does not clear `prev_scene_token`; the runner clears both. Each condition is a separate process with explicit reset.
* The shim-vs-real-kernel model-level discrepancy is **unexplained**; do not mix shim and real-kernel predictions.
* Pixel-level independent decoding was a **sampled** check (24 images by the Inspector), plus the raw-JSON recomputation of which
  images must be black; not every image byte was independently re-decoded.
* Official whole-sequence metrics per condition ARE produced (`artifacts/eval/rf_val_*`; clean 0.5370/0.4332, 0.5 s 0.5374/0.4326, 1 s 0.5368/0.4319, 2 s 0.5338/0.4273 NDS/mAP — whole-sequence scores dilute a brief single-camera outage; no significance claimed). Recovery curves, event analysis and bootstrap intervals are **not** produced here (Members 3/4).

### Gotchas
* Do not `pkill -f <pattern>` with a pattern that appears in your own command line (it kills the shell); use the `[b]racket` trick.
* `tools/` image paths in the original mini infos are relative; the runner rewrites them to absolute (`infos_abs.pkl`).
* Check `run.json → flash_attn.kind == "REAL flash-attn"` and `tf32_* == false` before using any run (the script asserts this).
