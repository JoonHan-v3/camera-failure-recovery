# Camera Failure and Recovery in Three Camera-Only 3D Detectors on nuScenes

ROB535 course project (Option 3: run three published methods on common scenes, generate our own predictions, compare strengths and limitations).
Plan: [Camera_Failure_and_Recovery_Study_Plan.pdf](Camera_Failure_and_Recovery_Study_Plan.pdf) (proposal: [Camera_Failure_and_Recovery_Project_Proposal.pdf](Camera_Failure_and_Recovery_Project_Proposal.pdf)).

This file is the entry point. It covers the goal, the repository layout, the methodology, how to run everything, and where the results are.
Each sub-project has its own, more detailed README ([BEVFormer](BEVFormer/README.md), [StreamPETR](StreamPETR/README.md), [SparseBEV](SparseBEV/README.md), [RecoveryAnalysis](RecoveryAnalysis/README.md); each starts with a *Methodology* section), and each has a checksummed package under [deliverables/](deliverables/).

---

## Quick start

This GitHub repository holds the code, frozen declarations, saved result tables and the paper and poster. It does **not** hold the multi-GB model checkpoints, predictions or nuScenes data (see [Repository contents](#repository-contents-and-what-is-not-included)), so the quickest useful checks are the ones that need only the saved results:

```bash
git clone https://github.com/JoonHan-v3/camera-failure-recovery.git
cd camera-failure-recovery

# 1. verify the packaged Member 4 deliverable (analysis, figures, profiles; 54 files)
(cd deliverables/member4 && sha256sum --check --quiet SHA256SUMS) && echo "member4 package verified"

# 2. regenerate every figure and table of the poster and the paper from the saved results (CPU, about a minute)
pip install numpy pandas matplotlib nbconvert ipykernel      # any Python 3.8+ environment
jupyter nbconvert --to notebook --execute --inplace Report/report.ipynb

# 3. rebuild the paper (IEEE RAS ieeeconf template, four pages plus references)
(cd Report && pdflatex paper && pdflatex paper)
```
Results: [deliverables/member4/SUMMARY.md](deliverables/member4/SUMMARY.md) (tables), [RecoveryAnalysis/artifacts/figures/](RecoveryAnalysis/artifacts/figures/) (figures), [Report/paper.pdf](Report/paper.pdf), [Report/poster.pdf](Report/poster.pdf). Re-running the models themselves needs the GPU and the data: see [§6](#6-how-to-run).

## Repository contents and what is not included

| included | not included (too large or third-party) |
|---|---|
| `outage/` code of the three model projects (shared outage schedule, corruption, audit, runners, evaluator), `recovery/` analysis code, unit tests | upstream code clones (below), released checkpoints (sources and SHA-256 in each `MODELS.md`) |
| frozen declarations, schedules, calibration thresholds, verification and summary JSON, per-scene tables (`event_stats/*.csv`) | predictions (`preds.pkl`), info files and ground-truth tables (`*.pkl`), per-image audit logs of the full validation runs (up to 23 MB each) |
| `deliverables/member4` (complete, checksummed), documents of members 1–3 | nuScenes data (see `BEVFormer/STORAGE_PLAN.md`), flash-attn build tree, logs |
| `Report/`: paper (LaTeX source and PDF), poster (PNG, PDF, design source), `report.ipynb` | |

The `SHA256SUMS` files of `deliverables/member1..3` list files that are not in this repository, so they verify only against a full local copy; `member4` verifies as cloned.

Upstream code is pinned at these commits and expected at these paths (unmodified):

| path | repository | commit |
|---|---|---|
| `BEVFormer/bevformer_src` | fundamentalvision/BEVFormer | `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` |
| `BEVFormer/mmdetection3d` | open-mmlab/mmdetection3d | `f1107977dfd26155fc1f83779ee6535d2468f449` |
| `BEVFormer/detectron2_src` | facebookresearch/detectron2 | `5aeb252b194b93dc2879b4ac34bc51a31b5aee13` |
| `StreamPETR/streampetr_src` | exiawsh/StreamPETR | `95f64702306ccdb7a78889578b2a55b5deb35b2a` |
| `SparseBEV/sparsebev_src` | MCG-NJU/SparseBEV | `d34105285f6a50aeb49dd62ef1486860650d1aab` |
| `SparseBEV/mmdetection3d` | open-mmlab/mmdetection3d (v1.0.0rc6) | `47285b3f1e9dba358e98fcd12e523cfd0769c876` |

Re-running the analysis from raw predictions therefore means: clone those repositories at the pinned commits, fetch the checkpoints and nuScenes, build the info files and run the model scripts (§6.3). The saved analysis outputs (`RecoveryAnalysis/artifacts/analysis/*.json`, `profile/`, `run_manifest.json`) and the per-scene tables (`SparseBEV/artifacts/event_stats/*.csv`) are included, and `Report/report.ipynb` reads only the saved outputs. Re-running `python3 -m recovery.analysis` needs a full local copy, because it also hashes the per-sample pickles (not in this repository) as recorded inputs.

---

## 1. Goal

> After a front-camera outage, how quickly does 3D detection recover — and how much does the answer depend on the model's temporal history?

A camera stream can go dark (hardware fault, communication loss). When it returns, a temporal detector may still carry features or object queries built from
the damaged inputs. Clean-data benchmarks say little about this transition. The study injects a **controlled black-image outage on CAM_FRONT** and measures
(a) the detection loss during the outage and (b) the recovery after the camera is restored.

Three released, pretrained detectors (no retraining), each with a different form of temporal history:

| detector | released configuration | temporal information |
|---|---|---|
| **BEVFormer** | R50 *tiny* | recurrent cached BEV representation (`prev_bev`, CAN-bus alignment) |
| **StreamPETR** | R50, 704, flash-attn, seq 90 ep | propagated object-centric memory (recurrent) |
| **SparseBEV** | original ICCV 2023 R50 `r50_nuimg_704x256` | finite window: current keyframe plus seven previous keyframes (`num_frames = 8`; filename-keyed feature cache) |

Scope: a carefully implemented, event-based **protocol** — not a new detector. The comparison is between *complete released systems*: pretraining, input
resolution, query count and history all differ, so rankings are descriptive and architecture is not isolated. The black-image outage is a controlled input
failure, not a model of dirt, packet loss or sensor electronics.

## 2. Team split (study plan) and where each part lives

| member | task | folder | package |
|---|---|---|---|
| 1 | BEVFormer integration, shared data inventory, checkpoint records, storage plan | [BEVFormer/](BEVFormer/) | [deliverables/member1/](deliverables/member1/) |
| 2 | StreamPETR integration, shared outage code, schedule, input audit | [StreamPETR/](StreamPETR/) | [deliverables/member2/](deliverables/member2/) |
| 3 | SparseBEV integration, projection, matching, official + custom evaluators, per-scene detection statistics | [SparseBEV/](SparseBEV/) | [deliverables/member3/](deliverables/member3/) |
| 4 | Run orchestration, bootstrap analysis, resource profiling, figures, recovery curves, uncertainty, run summaries | [RecoveryAnalysis/](RecoveryAnalysis/) | [deliverables/member4/](deliverables/member4/) |

Work was done and reviewed with a **Generator / Inspector protocol**: each folder has an append-only `process.txt`. The Generator appends a completion
entry ending in `Generator: task finished.#`; the Inspector appends an independent review ending in `#`; the Generator then fixes what is reasonable and
appends a response. **All four members are accepted.** Member 4 was accepted after a first review that required four corrections (see §8), with measurement and population qualifications recorded in its `process.txt`.

## 3. Project structure

```
camera-failure-recovery/
├── README.md                                   ← this file
├── Camera_Failure_and_Recovery_Study_Plan.pdf  study plan (protocol, rules, member tasks)
├── Camera_Failure_and_Recovery_Project_Proposal.pdf
├── BEVFormer/                                  Member 1
│   ├── bevformer_src/      pinned upstream BEVFormer (unmodified)      ckpts/   released checkpoint (bevformer_tiny_epoch_24.pth)
│   ├── mmdetection3d/ detectron2_src/          pinned dependencies
│   ├── outage/             adapter + shared outage code, runner (run_bevformer.py), audits, probes, tests
│   ├── artifacts/          infos, schedules, runs/, eval/, provenance/, verification, run_matrix.sh, run_queue.sh
│   ├── MODELS.md  STORAGE_PLAN.md  README.md  process.txt
├── StreamPETR/                                 Member 2
│   ├── streampetr_src/     pinned upstream StreamPETR (unmodified)     ckpts/   released checkpoint
│   ├── outage/             SHARED outage code (schedule, corruption, audit, state, nusc_ref, verify_runs, make_schedules), run_streampetr.py, tests
│   ├── artifacts/          trainval infos, schedules, runs/ (rf_*), eval/, flash-attn build provenance, run_matrix_realflash.sh
│   └── README.md  process.txt
├── SparseBEV/                                  Member 3
│   ├── sparsebev_src/ (commit d341052, unmodified)  mmdetection3d/ (v1.0.0rc6)  ckpts/r50_nuimg_704x256.pth
│   ├── outage/             shared outage code (byte-identical to Member 2) + adapter, runner, numerics/compat shims,
│   │                       evaluator.py (projection, matching, thresholds, recovery rules), event_stats.py, calibrate.py, checks, tests (96)
│   ├── artifacts/          runs/, eval/, calibration/ (thresholds), event_stats/ (per-scene × phase tables), provenance/ (declarations), gt_{pilot,val}.pkl
│   └── MODELS.md  README.md  process.txt
├── RecoveryAnalysis/                           Member 4
│   ├── recovery/           data, events, stats (bootstrap), analysis, samplelevel, deep_dive, orchestrate, profile_model, figures,
│   │                       qualitative, verify_tables, collect_deliverables, tests/ (35)
│   ├── artifacts/          analysis/ figures/ profile/ reproduction/ run_manifest.json RUN_SUMMARIES.md review_corrections.json logs/
│   └── README.md  process.txt
└── deliverables/member{1,2,3,4}/               checksummed packages: README, SUMMARY.md (headline tables), results_summary.json, SHA256SUMS, code + evidence
```

Outside the repository (large data): `~/storage/nuscenes/` (metadata, CAN bus), `~/storage/nuscenes_trainval_keyframes/` (52 GB keyframe images),
`~/storage/nuscenes_trainval_root/` (symlink root used by the loaders), `~/storage/nuscenes_dl/` (download tarballs). See `BEVFormer/STORAGE_PLAN.md`.

## 4. Methodology

### 4.1 Data and scenes
* nuScenes v1.0-trainval. **Evaluation = all 150 validation scenes (6,019 samples).** A **pilot** of 10 training-split scenes (402 samples) is used only for
  threshold calibration and pilot decisions (the released models may have seen these scenes in training — disclosed). nuScenes mini is for debugging only.
* Pretrained weights only; checkpoints, commits and SHA-256 hashes are recorded in each `MODELS.md` and in every run's `run.json`.

### 4.2 One physical outage schedule for every model
* One outage per scene. `t_start` = **CAM_FRONT sensor timestamp of keyframe 6** (six clean keyframes of warm-up); a scene is kept only if
  `t_start + 2.0 s + 4.5 s` fits before the last front keyframe. Same start for every duration and every model (schedule sha256 identical in all projects).
* Durations **0.5 / 1 / 2 s**. For duration *d*, every CAM_FRONT image with sensor timestamp in the half-open window `[t_start, t_start + d)` is replaced by an
  all-black image **before** resize / normalization. Causal allowance: an image counts as available to a sample if its timestamp ≤ sample timestamp + 50 ms.
* Every consumed image is audited (scene, sample, camera, timestamp, slot, path, pixel SHA-1 before/after, corrupted flag). Black counts on val: 168 / 326 / 635.
* Never change the schedule, thresholds, margin or scene list from detection outcomes.

### 4.3 Inference
* Each model is run **by its own project's runner** in its own conda environment: batch 1, TF32 off, deterministic cuDNN. Runs per model: clean twice
  (repeatability — bit-identical) + 0.5 / 1 / 2 s, on the pilot and the full validation set.
* Predictions share one format: `{sample_token: rows [x, y, z, w, l, h, yaw, vx, vy, label, score]}` in the LiDAR frame, so a single evaluator serves all models.
* **Baseline gate** (declared before the result was seen): clean full-val NDS within ±0.010 of the released number, and mAP too where the release publishes it.
  BEVFormer-tiny NDS 0.3545 / mAP 0.2519 (released 0.354 / 0.252) and StreamPETR 0.53704 / 0.43324 (released 0.537 / 0.432) were gated on **NDS and mAP**. **SparseBEV was gated on NDS only** (0.55598 vs released 0.556): the SparseBEV release does not publish mAP for this setting, so its mAP is reported but not gated. All three PASS.
* Model-state controls: recurrent state / feature caches are verified to contain no outage contamination before the outage and to restart per scene. SparseBEV's filename-keyed cache is
  cleared before every run; it is not emptied at scene boundaries (unused entries from the previous scene may remain until evicted), but scene histories are disjoint and those entries are never read
  (verified per frame, and by an alone-vs-continuous equality test). Its 8-keyframe window clears ≈ 3.54 s after restoration (window check).

### 4.4 Event-based evaluation (rules frozen before validation event statistics; geometry revised once afterwards)
* **Region:** objects within 50 m inside the CAM_FRONT field of view (geometry v2: a single ego ground plane at sample time; 2 m centre-distance matching, one-to-one greedy by score).
  **Protocol history:** the event rules (loss margin, ≥ 5 objects, onset rule, bins) and the baseline gate were declared before any validation event statistic existed. Geometry **v2 was *not*** — it was a documented correction made after validation statistics had already been computed with geometry v1 (v1 matched in LiDAR XY, v2 uses one ego-plane for region and matching, following an Inspector finding). The pilot thresholds were recalibrated under v2 and came out unchanged; v1 outputs are kept in `SparseBEV/artifacts/superseded_geometry_v1/`, and the disclosure is in `SparseBEV/artifacts/provenance/EVENT_RULES_DECLARATION.txt` (addendum).
* **Operating point:** per-model score threshold calibrated on the pilot's clean predictions (recall ≥ 0.30, ≥ 100 region predictions, maximum precision):
  SparseBEV 0.5422, BEVFormer 0.8793, StreamPETR 0.7578.
* **Phases:** `pre`, `during`, and 0.5 s **post-restoration bins** 0.0 … 4.5 s (±0.25 s, one-to-one assignment). Loss L = R_clean − R_outage, on the same eligible objects.
* **Per-event rules (as frozen):** damaged if pooled loss during > 0.05 with ≥ 5 eligible objects; **recovery onset** = first post-restoration bin with L ≤ 0.05 whose next adjacent bin
  also qualifies (a missing/insufficient bin breaks the pair). Categories: insufficient data / no measured degradation / damaged-recovered at first bin / damaged-recovered later /
  damaged with no confirmed recovery (complete follow-up) / damaged and **right-censored** (incomplete follow-up — never read as failure).
* **Aggregate rule:** degradation is "detected" only if pooled loss during > 0.05 **and** the bootstrap interval excludes 0.

### 4.5 Statistics
* Paired **complete-scene bootstrap**: B = 10,000, seed 535, one weight matrix shared across models and durations, percentile 95% intervals; everything recomputed from saved
  per-scene counts. Recovery-time medians use all replicates, with unreached replicates kept as +∞.
* Reported separately: aggregate curves vs per-event recovery, censored counts, insufficient-data counts, common-event comparison (events damaged in all three models),
  paired model differences, per-class and same-object diagnostics.

### 4.6 Verification (independent of the code that produced the numbers)
Independent re-implementations / checks: stat tables rebuilt from per-sample statuses (64,350 cells per model, exact match), projection vs the nuScenes devkit, geometry regression,
cross-model check of the outage set, pilot 2 s outage reproduced bit-for-bit by each runner, run-integrity gate before any analysis, unit tests in every project.

### 4.7 Resource profile
Batch-one latency (warm-up discarded, CUDA-synchronized, fresh input copy for every forward), peak allocated GPU memory and CPU decode time are measured directly. The reported "end-to-end" time is a **composition** — the median over frames of each frame's separately measured decode time plus that frame's forward time; it excludes copy/check overhead and scheduling and is *not* a directly timed pipeline wall-clock. Whole-run wall-clock time is measured separately in each run's `run.json` (`runtime_s`, concurrent processes) and listed in `RUN_SUMMARIES.md`. Each model is profiled in its own environment.

## 5. Environment and prerequisites

* Hardware used: 1× RTX 3060 12 GB (shared with the desktop), 128 CPU threads, 219 GB RAM. Inference is CPU/decode bound (~45 min per full-val run).
* Conda environments (all under `~/anaconda3/envs/`): `bevformer` (BEVFormer), `streampetr_fa` (StreamPETR, cloned from `streampetr` + real flash-attn 0.2.2, system `g++-11`),
  `sparsebev` (SparseBEV; also fine for RecoveryAnalysis — numpy / scipy / matplotlib only).
* GPU memory: ≈ 3.5 GB per StreamPETR process (≤ 3 at once), ≈ 4.4 GB per SparseBEV process (≤ 2 at once); more → CUDA OOM.
* Build details and exact package versions: `StreamPETR/README.md` §2.1 (+ `artifacts/flashbuild/PROVENANCE.txt`), `SparseBEV/README.md` §2 (CUDA extension build), `BEVFormer/README.md` §2.
* **Gotchas:** never use `python -I` (ignores `PYTHONPATH`); do not put `StreamPETR/outage/shims` on the path; always pass `--no-tf32`; `pkill -f <pattern>` can match your own shell.

## 6. How to run

All commands are copy-paste runnable from the repository root (after restoring the excluded data as above), and use real arguments (a pilot 2 s outage and the full-validation clean run as examples; change the
condition/name for other runs). Most of this is **already done on this machine**. Every command that needs the GPU says so. Full option lists: the "How to run" section of each project README
([BEVFormer](BEVFormer/README.md#2-how-to-run), [StreamPETR](StreamPETR/README.md#2-how-to-run), [SparseBEV](SparseBEV/README.md#2-how-to-run), [RecoveryAnalysis](RecoveryAnalysis/README.md#2-how-to-run)).

```bash
cd camera-failure-recovery
export ENVS=$HOME/anaconda3/envs         # conda envs: bevformer, streampetr_fa, sparsebev
```

### 6.1 Unit tests (no GPU)
```bash
(cd BEVFormer      && PATH=$ENVS/bevformer/bin:$PATH     PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -t . -p 'test_*.py')   # 54 tests
(cd StreamPETR     && PATH=$ENVS/streampetr_fa/bin:$PATH PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -t . -p 'test_*.py')   # 44 tests
(cd SparseBEV      && PATH=$ENVS/sparsebev/bin:$PATH     PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -t . -p 'test_*.py')   # 96 tests
(cd RecoveryAnalysis && PATH=$ENVS/sparsebev/bin:$PATH   PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s . -t . -p 'test_*.py')   # 35 tests
```

### 6.2 Shared schedules (deterministic; regeneration is byte-identical)
```bash
(cd StreamPETR && PATH=$ENVS/streampetr_fa/bin:$PATH PYTHONPATH=streampetr_src:. python -m outage.make_schedules artifacts)
sha256sum */artifacts/schedule_trainval_val.json          # must be identical in all three projects
```

### 6.3 Model runs (GPU)
Each model has its own runner script. Run at most 3 StreamPETR or 2 SparseBEV processes at once (more → CUDA OOM). A run writes `preds.pkl`, `audit.json`, `run.json`.
```bash
# BEVFormer-tiny      -> BEVFormer/artifacts/runs/pilot_d20
(cd BEVFormer  && artifacts/run_matrix.sh pilot 2 d20)
# StreamPETR R50      -> StreamPETR/artifacts/runs/rf_pilot_*   (whole pilot matrix: clean a, clean b, 0.5, 1, 2 s)
(cd StreamPETR && ./artifacts/run_matrix_realflash.sh pilot)
# StreamPETR, one full-validation run (here the 2 s outage) -> StreamPETR/artifacts/runs/rf_val_d20
(cd StreamPETR && ./artifacts/run_matrix_realflash.sh val 2 d20)
# SparseBEV R50       -> SparseBEV/artifacts/runs/pilot_d20
(cd SparseBEV  && artifacts/run_matrix.sh pilot 2 d20)
```
Conditions: `clean`, `0.5`, `1`, `2` (outage seconds); names used downstream: `a` (clean), `b` (clean again), `d05`, `d10`, `d20`. Logs: `<project>/artifacts/logs/`.
After the runs, verify audit and prediction integrity (CPU, per project):
```bash
(cd BEVFormer  && PATH=$ENVS/bevformer/bin:$PATH python -m outage.verify_runs artifacts/runs v1.0-trainval artifacts/verification_val.json val_a val_b val_d05 val_d10 val_d20)
(cd StreamPETR && PATH=$ENVS/streampetr_fa/bin:$PATH PYTHONPATH=streampetr_src:. python -m outage.verify_runs artifacts/runs v1.0-trainval artifacts/verification_val_rf.json rf_val_a rf_val_b rf_val_d05 rf_val_d10 rf_val_d20)
(cd SparseBEV  && PATH=$ENVS/sparsebev/bin:$PATH python -m outage.verify_runs artifacts/runs v1.0-trainval artifacts/verification_val.json val_a val_b val_d05 val_d10 val_d20)
```

### 6.4 Official nuScenes metrics and the baseline gate (CPU)
The three projects need **different invocations** — StreamPETR in particular takes an extra config argument and must be run from `streampetr_src/` with absolute paths.
```bash
R=$PWD
# BEVFormer (run from bevformer_src)
(cd BEVFormer/bevformer_src && PATH=$ENVS/bevformer/bin:$PATH PYTHONPATH=$PWD:$PWD/.. \
   python -m outage.eval_official ../artifacts/infos/val_infos.pkl ../artifacts/runs/val_a/preds.pkl ../artifacts/eval/val_a)
# StreamPETR (run from streampetr_src; CONFIG ARGUMENT FIRST; absolute paths)
(cd StreamPETR/streampetr_src && PATH=$ENVS/streampetr_fa/bin:$PATH PYTHONPATH=$R/StreamPETR/streampetr_src:$R/StreamPETR \
   python -m outage.eval_official $R/StreamPETR/streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
     $R/StreamPETR/artifacts/trainval_infos/val_infos.pkl $R/StreamPETR/artifacts/runs/rf_val_a/preds.pkl $R/StreamPETR/artifacts/eval/rf_val_a)
# SparseBEV (run from sparsebev_src; absolute paths)
(cd SparseBEV/sparsebev_src && PATH=$ENVS/sparsebev/bin:$PATH PYTHONPATH=$PWD:$PWD/.. \
   python -m outage.eval_official $R/SparseBEV/artifacts/infos/val_infos.pkl $R/SparseBEV/artifacts/runs/val_a/preds.pkl $R/SparseBEV/artifacts/eval/val_a)
```
Each writes `metrics.json` into its output directory. Gate: BEVFormer and StreamPETR NDS **and** mAP within ±0.010 of the release; SparseBEV NDS only (see §4.3).

### 6.5 Custom evaluator and per-scene statistics (CPU)
Thresholds and per-scene tables for all three models are produced with SparseBEV's evaluator (it consumes any model's common-format predictions). Example for BEVFormer:
```bash
cd SparseBEV; export PATH=$ENVS/sparsebev/bin:$PATH PYTHONPATH=$PWD
python -m outage.gt_tables pilot artifacts/gt_pilot.pkl                       # once; also: gt_tables val artifacts/gt_val.pkl
python -m outage.calibrate artifacts/gt_pilot.pkl ../BEVFormer/artifacts/runs/pilot_a/preds.pkl artifacts/calibration/threshold_bevformer.json
python -m outage.event_stats --gt artifacts/gt_val.pkl --schedule artifacts/schedule_trainval_val.json \
       --tau-json artifacts/calibration/threshold_bevformer.json \
       --clean ../BEVFormer/artifacts/runs/val_a/preds.pkl --d05 ../BEVFormer/artifacts/runs/val_d05/preds.pkl \
       --d10 ../BEVFormer/artifacts/runs/val_d10/preds.pkl --d20 ../BEVFormer/artifacts/runs/val_d20/preds.pkl \
       --out artifacts/event_stats --tag bevformer_val
cd ..
```
For StreamPETR use `StreamPETR/artifacts/runs/rf_pilot_a` / `rf_val_{a,d05,d10,d20}` with `threshold_streampetr.json` and `--tag streampetr_val`; for SparseBEV use `SparseBEV/artifacts/runs/pilot_a` / `val_*` with `threshold_sparsebev.json` and `--tag sparsebev_val`.
(Re-running this overwrites the accepted per-scene tables in `SparseBEV/artifacts/event_stats/`; do it into a scratch `--out` directory unless you intend to replace them.)

### 6.6 Recovery analysis, figures and profiling (Member 4)
```bash
cd RecoveryAnalysis; export PATH=$ENVS/sparsebev/bin:$PATH
python3 -m recovery.orchestrate status      # integrity of all 30 runs → artifacts/run_manifest.json, RUN_SUMMARIES.md (nonzero exit on failure)
python3 -m recovery.orchestrate execute     # dry run: lists missing/invalid validation runs; add --run to launch missing ones (GPU, ≤ 2 processes)
python3 -m recovery.orchestrate all         # integrity gate → verify_tables → analysis → deep_dive → figures   (CPU, ≈ 1 min)
python3 -m recovery.qualitative             # rule-selected qualitative gallery
artifacts/run_profile.sh                    # resource profile of all three models (GPU; must be idle)
python3 -m recovery.orchestrate reproduce sparsebev pilot 2    # GPU: re-run the pilot 2 s outage from its saved configuration; compare bit-for-bit (also: bevformer, streampetr)
cd ..
```

### 6.7 Build and verify a deliverables package
```bash
(cd BEVFormer  && PATH=$ENVS/bevformer/bin:$PATH     python3 -m outage.collect_deliverables)     # → deliverables/member1
(cd StreamPETR && PATH=$ENVS/streampetr_fa/bin:$PATH python3 -m outage.collect_deliverables)     # → deliverables/member2
(cd SparseBEV  && PATH=$ENVS/sparsebev/bin:$PATH     python3 -m outage.collect_deliverables)     # → deliverables/member3
(cd RecoveryAnalysis && PATH=$ENVS/sparsebev/bin:$PATH python3 -m recovery.collect_deliverables) # → deliverables/member4
for m in 1 2 3 4; do
  (cd "deliverables/member$m" &&
    sha256sum --check --quiet SHA256SUMS) || exit 1
done
echo "All four packages verified."
```

## 7. Where to find the results

| what | where |
|---|---|
| headline tables of each member | [member1](deliverables/member1/SUMMARY.md), [member2](deliverables/member2/SUMMARY.md), [member3](deliverables/member3/SUMMARY.md), [member4](deliverables/member4/SUMMARY.md) (`SUMMARY.md`; also `results_summary.json` next to each) |
| pooled recall-loss curves, aggregate rule, per-event categories, common events, paired differences | [recovery_summary.json](RecoveryAnalysis/artifacts/analysis/recovery_summary.json), [per_event_categories.csv](RecoveryAnalysis/artifacts/analysis/per_event_categories.csv) |
| per-class and same-object diagnostics | [classes_sameobject.json](RecoveryAnalysis/artifacts/analysis/classes_sameobject.json) |
| figures (loss curves, event categories, cumulative onset, common events, per-class, resource profile, official metrics) | [figures/](RecoveryAnalysis/artifacts/figures/) — e.g. [fig_loss_curves.png](RecoveryAnalysis/artifacts/figures/fig_loss_curves.png), [fig_event_categories.png](RecoveryAnalysis/artifacts/figures/fig_event_categories.png); qualitative gallery in [figures/qualitative/](RecoveryAnalysis/artifacts/figures/qualitative/) |
| resource profiles | [artifacts/profile/](RecoveryAnalysis/artifacts/profile/) |
| run inventory with hashes | [run_manifest.json](RecoveryAnalysis/artifacts/run_manifest.json), [RUN_SUMMARIES.md](RecoveryAnalysis/artifacts/RUN_SUMMARIES.md) |
| frozen declarations (baseline gate, event rules incl. the geometry-v2 addendum) | [SparseBEV/artifacts/provenance/](SparseBEV/artifacts/provenance/), [BEVFormer/artifacts/provenance/](BEVFormer/artifacts/provenance/) |
| model / checkpoint provenance | [BEVFormer/MODELS.md](BEVFormer/MODELS.md), [SparseBEV/MODELS.md](SparseBEV/MODELS.md) |
| review history and corrections | each project's `process.txt` ([BEVFormer](BEVFormer/process.txt), [StreamPETR](StreamPETR/process.txt), [SparseBEV](SparseBEV/process.txt), [RecoveryAnalysis](RecoveryAnalysis/process.txt)); [review_corrections.json](RecoveryAnalysis/artifacts/review_corrections.json) |

### Headline results (150 validation scenes; descriptive — see limitations)
* **Every model loses front-region recall during the outage**, increasing with duration. Pooled recall loss (0.5 / 1 / 2 s): SparseBEV 0.145 / 0.171 / 0.213,
  StreamPETR 0.093 / 0.118 / 0.149, BEVFormer 0.168 / 0.194 / 0.224; all 95% intervals exclude zero.
* **Recovery in the pooled curve is fast:** within the 0.05 margin from the first post-restoration bin, except SparseBEV and StreamPETR at 0.0 s after 2 s outages (0.072 / 0.074);
  afterwards |loss| ≤ 0.031. No step appears when SparseBEV's 8-keyframe window clears.
* **Per-event damage rate** (events with ≥ 5 objects): SparseBEV 0.66 / 0.71 / 0.74, StreamPETR 0.49 / 0.57 / 0.63, BEVFormer 0.79 / 0.79 / 0.80. StreamPETR is lowest;
  for SparseBEV vs BEVFormer there is no statistically detected difference (the interval reaches zero), which does not establish equivalence.
* **Per-event recovery is censoring-heavy:** 43–77% of damaged events lack complete follow-up, so unconfirmed events are reported as right-censored, never as failures.
* Whole-sequence official NDS barely moves (e.g. SparseBEV 0.556 → 0.552 after 2 s): whole-sequence metrics dilute a brief single-camera outage.
* Resources (RTX 3060, batch 1): median latency SparseBEV 190 ms, BEVFormer 131 ms, StreamPETR 79 ms; peak allocated GPU memory 2,652 / 617 / 375 MB.

## 8. Limitations (carry into any report)

* Systems differ in clean recall, thresholds, resolution, pretraining, query count and history → the model ranking is descriptive, not causal.
* Pilot scenes (threshold calibration) are training scenes the released models may have seen.
* Per-event analysis is noisy: ~1 sample per bin per scene and ~5 objects per sample; the ≥ 5 object rule leaves 80 / 51 / 25 of 150 events insufficient (0.5 / 1 / 2 s). Pooled curves are the more reliable summary.
* "Region" = projection of GT centres inside CAM_FRONT within 50 m: other cameras overlap, objects may be occluded, motion is not compensated to exposure time.
* Intervals measure scene-sampling variability only (one deterministic run per condition; no training or run-to-run variance).
* Recall at a fixed threshold says nothing about score or box quality.
* Member 4 review history: the first Inspector review required four corrections (unconditional median interval, profiling with reused mutated inputs, same-sample comparisons in the
  same-object diagnostic, orchestration gate / checkpoint pin). They are fixed; before/after values are in [review_corrections.json](RecoveryAnalysis/artifacts/review_corrections.json) and
  superseded outputs are preserved in `RecoveryAnalysis/artifacts/superseded_before_inspector_review2/`.

## 9. Conventions

* `process.txt` files are append-only logs; never truncate them.
* Deliverables are produced only by each project's `collect_deliverables` script and are verified with `sha256sum --check --quiet SHA256SUMS`.
* Upstream code (`*_src/`, `mmdetection3d/`) is pinned and unmodified; all changes live in `outage/` or `recovery/`.
