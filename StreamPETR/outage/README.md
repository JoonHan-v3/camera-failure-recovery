# Member 2: outage schedule + input audit (StreamPETR)
Tests: `python3 -m unittest discover -s . -t . -p "test_*.py"` from StreamPETR/ (numpy only).

## Done and unit-tested (synthetic data)
- schedule.py: validated, integer-us schedule; persisted with params + skip reasons (save/load_schedule).
- corruption.py: black CAM_FRONT incl. sweeps, half-open window, before resize/normalize.
- audit.py: manifest reconciliation (missing/extra/duplicate/unobserved scene), pre-corruption hashes
  (detect unflagged changes), clean-vs-outage pairing over ALL cameras, restoration offsets, saved run metadata.
- state.py: guard rejects unknown memory status; reset_between_runs; A->B isolation + repeat check (callables);
  predictions_equal with explicit rtol and canonical ordering.
- adapter: validates lengths/types/units; clean runs log-only (duration 0); ts_mode 'rel_s' | 'abs_us'.

## Round 2 additions
- Causality: audit records eval_ts_us; `causal_tolerance_us` must be set from the pinned loader's verified
  sensor/sample timing (unset => status NOT VERIFIED, no generic rule imposed).
- `status()` returns CONSISTENT (never PASS) only with a retained coverage manifest AND causal policy; saved JSON carries status,
  violations, manifest id, missing 0-4.5 s bins. compare_runs also checks duplicates, schedule/model match.
- Restoration offsets are per evaluated sample (evaluation time), `missing_bins()` checks 0..4.5 s coverage (one-to-one since Round 4).
- Guard accepts only bool/np.bool_ freshness; state commits only after validation.

## Round 3
- `status()` = record consistency only (CONSISTENT/FAIL/NOT VERIFIED). `readiness()` = study-protocol gate
  (all scheduled scenes observed, corrupted front images consumed, 0..4.5 s bins evaluated). A prefix-only
  run is CONSISTENT but NOT READY (regression test). Skipped scenes keep reasons in the schedule JSON.
- Finite-window history checks (SparseBEV) are NOT a substitute for StreamPETR recurrent-memory checks.

## Round 4
- Bin coverage is one-to-one (documented tie rule in `missing_bins`); the downstream evaluator must match it.

## Round 5
- Evaluation identity is validated (`eval_conflicts`, part of `verify()`): one immutable eval ts per (scene, sample),
  no missing sample ids, no two samples at one eval ts, and agreement with `set_sample_metadata()` (independent
  dataset sample table). `readiness()` requires that metadata; conflicts are never silently overwritten.
- Fixtures now use unique sample ids; midpoint test uses valid corrupted + unchanged follow-up records.

## Round 6
- `save()` now persists `expected_manifest`, `sample_metadata` and their sha1 `hashes`, so coverage and
  sample-time checks can be re-run by another member.

## Reproduction procedure once dependencies exist (not executed here)
Missing: StreamPETR repo + pinned commit, checkpoint, torch/mmcv/mmdet3d env, nuScenes (v1.0-mini or trainval)
with sample_data tables. Steps: (1) build per-scene keyframe ts from nuScenes `sample`; run
`build_schedule` + `save_schedule` once; (2) insert `CorruptFrontCamera` after image loading, before resize/normalize,
with `ts_mode='abs_us'`; (3) build the expected manifest and `set_sample_metadata` from nuScenes tables (not from the run);
(4) run clean, 0.5, 1, 2 s with explicit model-memory reset per scene/run; `save()` each audit; `compare_runs` clean vs outage;
(5) run clean twice on ~10 scenes and A->B isolation via `state.py`, hooked to the real model's memory.

## Round 7: REAL StreamPETR integration (verified on v1.0-mini and the 10-scene training-split pilot)
Pinned: StreamPETR `95f6470`, mmdet3d `v1.0.0rc6` (`47285b3`), mmcv-full 1.6.0, mmdet 2.28.2, torch 1.9.0+cu111
(conda env `streampetr`), config `stream_petr_r50_flash_704_bs2_seq_90e`, checkpoint sha256 in `artifacts/runs/*/run.json`.
Deviation: flash-attn cannot be built here (nvcc 12 vs torch cu111) -> `outage/shims/flash_attn` (exact fp32 softmax
attention). Clean-baseline numbers must be checked against the published ones before the shim is accepted for the study.
Verified loader contract (CustomNuScenesDataset): `img_timestamp` = ABSOLUTE seconds, `timestamp` = lidar keyframe
seconds, `sample_idx` = sample token, six KEYFRAME cameras per sample in converter order, NO image sweeps; history lives
only in the head's recurrent memory (no finite window, no boundary padding). Camera-minus-sample (lidar) time differs per
camera: CAM_FRONT is always 32.6-42.5 ms BEFORE it, other cameras range from -117.9 ms to +50.0 ms (all 204,894 trainval
keyframe images; pilot records: -46.1 ms..+1.9 ms). The outage start is therefore defined on the CAM_FRONT sensor
timestamp of keyframe 6 (`nusc_ref.py`); evaluation time = sample (lidar) time.
- Hook: `mm_pipeline.OutageFrontCamera` right after `LoadMultiViewImageFromFiles`; runner `run_streampetr.py`.
- Reset lifecycle: `reset_memory()` sets buffers to None (re-allocated at fixed size, zero-filled, refreshed with
  prev_exists=0) and does NOT clear `prev_scene_token`; the runner clears both and observes real buffers (`observe_reset_state`).
- Frozen schedules: `artifacts/schedule_mini.json`, `schedule_trainval_{val,train}.json` (150 val / 700 train scenes, 0 skipped).
- Mini (10 scenes, 404 samples; DEBUG set per plan): clean twice bit-identical (max |diff| 0.0); every audit CONSISTENT;
  black front images = 1/2/4-5 per scene for 0.5/1/2 s and equal an independent recomputation from raw nuScenes tables;
  prefix predictions identical to clean (60/60); other cameras never touched; 2 s run NOT READY: bin 0.0 has no
  observation in 2 scenes (the 5th black front image at +1.95 s is evaluated at +1.986 s, before restoration, and the next
  evaluation is at +0.485 s). This is reported, not hidden: the evaluator must label it insufficient data.
- Isolation (`isolation.py`, `artifacts/isolation_mini.json`): all 10 scenes alone == inside the continuous run (diff 0.0,
  memory fresh at start); fresh-B twice == B-after-A; negative controls: forced state leak changes predictions
  (max diff 102.4 when unguarded) and is rejected by the guard; interleaved scenes rejected.
- PILOT (10 training-split scenes, frozen rule: every 70th by name -> `artifacts/trainval_infos/pilot_scenes.json`; 402
  samples; infos cross-checked against raw nuScenes JSON: 6421/6421 timestamps, camera order, files OK):
  clean twice bit-identical; all audits CONSISTENT; black sets match the independent recomputation (1-2 / 2-3 / 4-5 front
  images per scene for 0.5/1/2 s; all-black pixels verified); prefix predictions identical 60/60; isolation test identical
  to the mini one (all scenes alone == continuous, unguarded leak changes predictions by 102.4, guard rejects it).
  READY is NOT reached for every outage run: bin 0.0 has no observation in 1-2 of 10 scenes (frame-timing jitter, see
  above). `artifacts/verification_pilot.json`, `isolation_pilot.json`, `runs/pilot_*/{audit,run}.json`.
- Clean accuracy sanity: 2 mini-val scenes, official nuScenes eval via the dataset's own converter: NDS 0.469 / mAP 0.409
  (81 frames; noisy; published full-val 0.537 / 0.432). Full-val check = `artifacts/runs/val_a` (running when this was
  written; see `artifacts/run_matrix_pilot_val.log`) + `python -m outage.eval_official`.
- Info files: `artifacts/trainval_infos/build_infos_par.py` (StreamPETR converter, 8 parallel slices) over
  ~/storage/nuscenes_trainval_keyframes (CAM_* + LIDAR_TOP keyframes extracted from the downloaded blobs).
- Reproduce: `artifacts/run_matrix_mini.sh`, `python -m outage.verify_runs ...`, `python -m outage.isolation ...`.

## Round 8 (Inspector 8th review)
- Frozen causal ALIGNMENT policy (`audit.CAUSAL_TOL_US = 50000`, text in `CAUSAL_POLICY`, saved in run.json):
  evaluation/annotation time = sample (lidar keyframe) time (predictions are scored against sample-time annotations);
  input-bundle availability is >= max(sample time, latest consumed exposure) with transport/processing delay UNMEASURED;
  the 50 ms bound is a fixed sample+50 ms inference-budget allowance for images assigned to a sample, not a claim that a
  later-exposed image existed at lidar time; pixel assignment is unchanged (max observed over ALL trainval keyframes: +49,992 us, CAM_BACK_LEFT); anything later is a future-image
  violation. It is NOT derived from the run's own manifest and must not be raised to silence violations; the runner aborts
  if a manifest exceeds it. `verify_runs` re-applies it to saved audits.
- Missing-bin rule (downstream evaluator): a missing/insufficient bin (e.g. 0.0) breaks only the adjacent recovery pair
  that contains it; later valid adjacent pairs may still establish onset; the event is kept, never auto-discarded, never
  read as permanent failure. The schedule is never changed from detection outcomes.
- `verify_runs.pred_diff` rejects non-finite / wrong-schema / non-integral-label predictions (returns inf); isolation
  `all_isolated` now also requires both negative controls to have been rejected (regression tests added).
- Handoff: `artifacts/run_matrix.sh <mini|pilot|val>` is fail-fast (pipefail, artifact + audit-status checks). The
  earlier `run_matrix_*.sh` are what actually produced the saved runs. Never use `python -I` with PYTHONPATH.
- Pixel-level check by the Inspector (24 decoded pilot_d05 images) is accepted as sampled, not exhaustive, evidence.
- OPEN GATE: full-validation clean result + official eval vs released baseline (tolerance to be declared) for the
  attention shim. Outage val runs are part of the broader run matrix, reported separately from the pilot evidence.

## BASELINE GATE (acceptance declared 2026-11-21 BEFORE the full-val result was seen)
Compare official nuScenes full-val clean metrics of `val_a` (shimmed attention, fp16) with the released R50 90e numbers
(NDS 0.537, mAP 0.432 from the StreamPETR README): accept the shim if |NDS diff| <= 0.010 and |mAP diff| <= 0.010; a larger
difference is reported and investigated (shim numerics vs env/data) before the study configuration is frozen.

### BASELINE GATE, SHIM CONFIGURATION (2026-11-21): FAILED (superseded by the real-kernel result in the next section)
`val_a` (full val, 150 scenes, 6019 samples, clean, fp16 + attention shim), official nuScenes eval (`artifacts/eval/val_a/metrics.json`):
NDS 0.5086 / mAP 0.3894 vs released 0.537 / 0.432 -> differences -0.028 / -0.043, both outside the declared 0.010 tolerance.
Ruled out so far: devkit version (1.1.11 and 1.1.9 give identical numbers); checkpoint/config mismatch (591/591 keys, 0 missing,
0 unexpected, 0 shape mismatches); fp32 comparison is impossible (the repo's FlashAttention asserts fp16/bf16).
Investigation evidence (`artifacts/ref_test/`, `artifacts/upstream/`):
- UPSTREAM CONTROL DONE: the repo's own unmodified `tools/test.py` (`python -m torch.distributed.launch --nproc_per_node=1
  tools/test.py <config> <ckpt> --launcher pytorch --eval bbox --cfg-options data.test.ann_file=artifacts/trainval_infos/val_infos.pkl`,
  same frozen infos/checkpoint/env/shim) gives NDS 0.5085751858 / mAP 0.3894301993, and its submission file equals my runner's
  `val_a` submission BOX FOR BOX (6019 samples, 1,148,071 boxes, 0 count/class mismatches, max |translation/size/score| diff 0).
  => a custom-runner-only discrepancy is ruled out (same official submissions under the same checkpoint, infos, env and
  shim). Components SHARED by both paths (data/preprocessing, numerical kernels, config, official conversion assumptions)
  are NOT cleared by this control; parity with the authors' released run is still unproven.
- Released training log (`artifacts/upstream/*.log`): final val NDS 0.5369 / mAP 0.4323 on 8x RTX 2080 Ti, torch 1.9.0+cu111,
  real flash-attn. Its embedded config equals the repo config for model/head/test numerics; differences are only the authors'
  internal data loader (nori), training-time `seq_mode` and batch bookkeeping. Multi-GPU val uses contiguous per-GPU slices
  (hypothesis only: its effect on the metric is untested; direction not established).
- Decode path: OpenCV vs PIL decode of real keyframes is pixel-identical (max diff 0).
- Attention shim numerics: fp32-softmax vs fp16-softmax shim give bit-identical predictions (pilot, 402 samples) while zeroing
  attention collapses predictions (top score 0.98 -> 0.25): the model uses attention, and is insensitive to shim precision.
  This does not prove equivalence with real flash-attn kernels (no real flash build is possible here: gcc 13 / nvcc 12).
- TF32 hypothesis (tested alone; it did not close the gap, see result below): torch 1.9 enables TF32 matmul/cuDNN by default on Ampere (this RTX 3060); the authors' Turing GPUs have no
  TF32. TF32 on vs off changes pilot predictions materially (top-20 score mean abs diff 0.012; `artifacts/runs/pilot_notf32`).
  TF32 RESULT (2026-11-21): full-val TF32-off run `val_notf32`: NDS 0.50779 / mAP 0.38741 vs TF32-on 0.50858 / 0.38943
  (changes -0.00079 / -0.00202). Disabling TF32 ALONE did not recover the released accuracy under this shared stack; this
  says nothing about possible TF32 interactions with other factors. Gate still FAILED (diffs -0.0292 / -0.0446 vs 0.537 / 0.432).
  fp16-vs-fp32 shim agreement does NOT establish FlashAttention-kernel equivalence (both are replacements).
  Settings of each numerical experiment: `run.json` now records FLASH_SHIM_MODE, shim sha256, TF32 flags, GPU and argv
  (val_notf32 started before this logging: FLASH_SHIM_MODE unset = fp32-softmax, --no-tf32, RTX 3060; see `artifacts/runs/val_notf32.settings.txt`).
- Shim fidelity (limited implementation control; script `outage/controls/shim_fidelity.py`, report
  `artifacts/controls/shim_fidelity.json`): the shim-backed `FlashMHA` of all 6 decoder layers with the released weights vs
  PyTorch's `multi_head_attention_forward`, seed 0, random fp32 inputs; criterion declared in the script before running
  (max abs <= 0.05 and max rel-to-scale <= 0.01): PASSED, per-layer max abs 0.0088-0.0216, rel-to-scale 0.0023-0.0042. Supports functional correctness of the replacement; real FlashAttention kernels remain
  untested (not buildable here), so kernel-level equivalence is still NOT shown.
- Data integrity (script `outage/controls/decode_inventory.py`, report `artifacts/controls/decode_inventory.json`: 36,114
  images = 6019 samples x 6 cameras, shapes all (900,1600,3), 0 failures, cv2 5.1.0, file bytes 54,586-313,126); extraction rc=0 for all 10 archives;
  infos agree with raw nuScenes JSON (6421/6421 timestamps, camera order, files).
- STATUS: baseline mismatch UNRESOLVED (NDS 0.5086 vs 0.537, mAP 0.3894 vs 0.432). Observed: the car AP shortfall is present at every official match
  distance (e.g. AP@4m 0.786 vs 0.845 in the released log); whether it is detection, confidence or localization related is
  UNKNOWN until an isolating diagnostic exists.
  Remaining untested shared-stack candidates: real flash-attn kernels, authors' data pipeline/nori decode path, library
  versions beyond the pinned set (numpy/Pillow/OpenCV 5.1), a different checkpoint than the logged run. Configuration NOT frozen.
All TF32-on artifacts (mini/pilot/val_a/val_d05) are labeled UNVALIDATED (`artifacts/runs/README_TF32.txt`); the old val
outage matrix was stopped after val_d05 so GPU time is not spent on a possibly invalid configuration.

### BASELINE GATE, REAL flash-attn (2026-11-21): **PASSED** -- configuration FROZEN
Real flash-attn 0.2.2 was built in a CLONE of the env (`streampetr_fa`; nvcc 11.8 + system gcc-11 host compiler; the build
script defines a CUDA-minor-version check (nvcc 11.8 != torch cu111) but it is NOT invoked in this build path, so NO patch was
needed or applied (an earlier note here saying it was patched was wrong; `artifacts/flashbuild/setup_py.patch` is empty);
sm80 SASS, no PTX; torch runtime stays 1.9.0+cu111; build log `artifacts/flashbuild/build.log`; exact commands, toolchain,
extension sha256 and env exports: `artifacts/flashbuild/PROVENANCE.txt`, `pip_freeze_*.txt`, `conda_export_*.txt`). Full val (150 scenes, 6019 samples), clean,
`--no-tf32`, no shim on PYTHONPATH (`run.json` flash_attn.kind = "REAL flash-attn", module path recorded):
NDS 0.53704 / mAP 0.43324 -> differences vs the declared 0.537 / 0.432: +0.0000 / +0.0012 (vs the released training log
0.5369 / 0.4323: +0.0001 / +0.0009; car AP@4m 0.8447 = released log) -> within the UNCHANGED 0.010 tolerance. Artifacts:
`artifacts/runs/val_realflash`, `artifacts/eval/val_realflash/metrics.json`, `artifacts/val_realflash.log`.
FROZEN numerical settings: real flash-attn kernel, TF32 off (matches the authors' Turing GPUs; TF32 had no effect on accuracy),
fp16 per config, checkpoint sha256 in run.json. Run everything with `artifacts/run_matrix_realflash.sh` (fail-fast; asserts REAL
flash-attn and TF32 off per run).
CONSEQUENCES: the pure-PyTorch shim is NOT equivalent at model level (NDS -0.029 / mAP -0.045) although it passed the
micro-controls (random-input FlashMHA vs torch MHA, real-vs-shim kernel on random fp16 inputs: max diff 1.2e-4). The cause of
that discrepancy is NOT identified (not investigated further). Therefore ALL shim-based prediction artifacts (mini_*, pilot_*,
val_a, val_d05, val_notf32, pilot_notf32, pilot_shimfp16, ref_test) are INVALID for study accuracy and are retained only as
failed/diagnostic evidence. The pixel-assignment, audit-logic, schedule and state-guard tests do not depend on the attention
kernel, but ALL real-model evidence (clean-twice, isolation, paired clean/outage audits, prefix predictions) is being
regenerated under the frozen settings (`artifacts/runs/rf_*`; results below when complete).

### REAL-KERNEL PILOT (frozen settings: real flash-attn 0.2.2, TF32 off, fp16; 10 frozen training-split scenes, 402 samples)
Runs `artifacts/runs/rf_pilot_{a,b,d05,d10,d20}` (`artifacts/run_matrix_realflash.sh pilot`, each run.json asserts REAL
flash-attn + TF32 off); reports `artifacts/verification_pilot_rf.json`, `artifacts/isolation_pilot_rf.json`.
- Clean twice: predictions bit-identical (max |diff| 0.0 over 402 samples); both clean audits CONSISTENT, 0 paired-input errors.
- Outage audits (0.5 / 1 / 2 s): all CONSISTENT, 0 paired-input errors; black front images 1-2 / 2-3 / 4-5 per scene
  (11 / 21 / 42 total) equal the independent raw-nuScenes recomputation; all-black pixels verified; other cameras corrupted: 0;
  prefix predictions identical to clean 60/60 in every condition; post-start predictions changed 342/342.
- Restoration offsets/bins: bin 0.0 has no observation in 1 / 1 / 2 scenes of 10 (0.5 / 1 / 2 s), kept and reported under the
  documented missing-bin rule (so readiness stays NOT READY for those runs by design, status CONSISTENT).
- State/isolation (`outage/isolation.py`, TF32 off, real kernel): all 10 scenes alone == inside the continuous run (diff 0.0,
  memory observed fresh at each start); fresh B twice == B after A (diff 0.0); NEGATIVE CONTROLS: forced state leak without the
  guard changes predictions (max diff 102.4) and is rejected by the guard; interleaved scenes rejected. `all_isolated` = true.
- Feature caches: none outside model memory; every condition is a separate process with explicit reset.
### REAL-KERNEL FULL VALIDATION (150 scenes, 6019 samples, same frozen settings)
Runs `rf_val_{a (=val_realflash, passed the baseline gate), b, d05, d10, d20}`; report `artifacts/verification_val_rf.json`
(`python -m outage.verify_runs artifacts/runs v1.0-trainval artifacts/verification_val_rf.json rf_val_a rf_val_b rf_val_d05 rf_val_d10 rf_val_d20`).
- Clean twice (a vs b): predictions bit-identical over all 6019 samples (max |diff| 0.0); both audits CONSISTENT, 0 paired-input errors.
- Outage audits 0.5 / 1 / 2 s: all CONSISTENT, 0 paired-input errors; black front images 168 / 326 / 635 (1-2 / 2-3 / 4-5 per
  scene) equal the independent raw-nuScenes recomputation; all-black pixels verified; other cameras corrupted: 0; prefix predictions
  identical to clean 900/900 (6 samples x 150 scenes); post-start predictions changed 5119/5119 in every condition.
- Missing follow-up bins (kept, reported under the agreed rule; status stays CONSISTENT, readiness NOT READY by design):
  scenes with >=1 missing bin 14 / 26 / 34 of 150 (9% / 17% / 23%) for 0.5 / 1 / 2 s. Bin 0.0 dominates (12 / 22 / 29 scenes: the
  last black frame is evaluated just before restoration); later bins are rare (<= 2 scenes each: 3.5, 4.5 for 0.5 s; 3.0, 4.0, 4.5
  for 1 s; 2.0, 3.0, 3.5, 4.5 for 2 s) and come from irregular keyframe spacing (gaps up to 0.6 s push the nearest evaluated
  sample outside the +-0.25 s bin tolerance). Downstream rule: a missing bin breaks only the adjacent recovery pair containing it.
- Max camera-minus-evaluation skew in these records: -49961..+2106 us (within the frozen 50 ms alignment allowance).
- Provenance: rf_val_b and rf_val_d20 run.json carry flash_attn.dist_version 0.2.2 + extension hash; rf_val_a / d05 / d10 started
  before those fields existed (same env/extension; see artifacts/flashbuild/PROVENANCE.txt). All five: REAL flash-attn, TF32 off.
- Official whole-sequence nuScenes metrics per condition ARE produced (`artifacts/eval/rf_val_*`, `eval_runs_official.sh`): NDS/mAP clean
  0.5370/0.4332 (a and b identical), 0.5 s 0.5374/0.4326, 1 s 0.5368/0.4319, 2 s 0.5338/0.4273 (whole-sequence scores dilute a brief
  outage; no significance claimed). NOT done here (Members 3/4): recovery curves, event analysis, bootstrap intervals.
- The checksummed deliverables package is collected into `~/Documents/UM/rob535/projects/deliverables/member2/` (`outage/collect_deliverables.py`).

## STILL UNVERIFIED / OPEN
- Full-validation clean accuracy: gate PASSED with real flash-attn (above); shim discrepancy cause unidentified.
- Per-scene bin gaps follow the documented missing-bin rule (Round 8); no further schedule decision is pending.
- Image-byte verification: recorded hashes plus independent re-decoding of a SAMPLE (24 pilot_d05 images, Inspector, 8th
  review) and the raw-JSON recomputation of which images must be black; not exhaustive re-decoding of every image.
- Feature caches: StreamPETR has none outside model memory; condition runs are separate processes with explicit reset.
- Val outage runs (150 scenes) belong to the broader run matrix and are reported separately from the pilot evidence.
