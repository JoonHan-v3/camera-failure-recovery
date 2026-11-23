# MODELS.md — shared checkpoint and environment records (Member 1)

Study plan p.1: *"MODELS.md will record configuration paths, code commits, checkpoint sources, file hashes, pretraining, and
stated detector-training splits. We will audit the available provenance for evaluation-scene overlap and verify causal
inference. Any uncertainty will be disclosed."* All hashes below were computed on this machine (sha256).

## 1. BEVFormer-tiny (Member 1) — VERIFIED, integrated, baseline gate in SUMMARY.md

| item | value |
|---|---|
| code | https://github.com/fundamentalvision/BEVFormer @ `66b65f3a1f58caf0507cb2a971b9c0e7f842376c` (`BEVFormer/bevformer_src`, unmodified) |
| config | `bevformer_src/projects/configs/bevformer/bevformer_tiny.py` (R50, BEV 50x50, 3 encoder layers, input 1600x900 -> x0.5 = 800x450, single scale C5, `video_test_mode=True`, `queue_length=3`, fp32) |
| checkpoint | `ckpts/bevformer_tiny_epoch_24.pth` from https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_epoch_24.pth (README link) |
| sha256 | `7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12` (400,785,466 B); recorded in every `run.json` |
| released training log | `dl/bevformer_tiny_epoch_24.log` (same release); epoch-24 val: NDS 0.3540, mAP 0.2524 |
| checkpoint meta | epoch 24, iter 84408, seed 0, 2022-06-18; built with mmcv 1.4.0, mmdet 2.14.0, mmseg 0.14.1, mmdet3d 0.17.1, torch 1.9.0+cu111 (= our stack) |
| embedded vs repo config | identical except bookkeeping (`gpu_ids`, `work_dir`): `artifacts/provenance/config_diff_ckpt_vs_repo.json`; embedded copy `ckpt_embedded_config.py` |
| pretraining | `img_backbone` init `torchvision://resnet50` (ImageNet-1k); `load_from = None` (from the log) — no nuScenes/nuImages pretraining |
| detector-training split | `data/nuscenes/nuscenes_infos_temporal_train.pkl` (nuScenes **train**, 700 scenes); val infos used only for per-epoch evaluation (from the embedded config and the log) |
| evaluation-scene overlap | none known: the 150 val scenes are not in the training infos. **Pilot/calibration scenes are train-split and WERE seen in training** (study plan p.2 states this) |
| temporal mechanism | recurrent cached BEV (`prev_frame_info['prev_bev']`) + CAN-bus/ego-pose shift; reset only on scene-token change; **no reset between runs upstream** (we add one, see README §4) |
| images consumed | 6 keyframe cameras per sample; no camera sweeps; image sensor ts ≤ sample ts + 50 ms (audited per image, observed max +2.1 ms) |
| CAN-bus timing | the 9 auxiliary CAN values (accel, rotation rate, velocity → `can_bus_mlp`) come from the first pose message AFTER the sample time (upstream converter); offsets audited for every info: pilot max +37.7 ms, val max **+49.9 ms**, 0 over the frozen sample+50 ms allowance, all values match the selected message (`artifacts/canbus_audit_{pilot,val}.json`, tied to info sha256). Inference is therefore compatible with the sample+50 ms availability allowance, **not** with strict availability at the sample time |
| extra data | CAN-bus expansion `can_bus.zip` sha256 `3c68b94c001e8bd05a19886ecb2c6854e0cd69d7005ed9a94d13d45d2951e83f` (https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip) -> `~/storage/nuscenes_can_bus/can_bus` |
| environment | conda `bevformer` (py3.8, clone of `streampetr_fa` then re-pinned): torch 1.9.0+cu111, torchvision 0.10.0+cu111, mmcv-full 1.4.0 (openmmlab cu111/torch1.9.0 wheel), mmdet 2.14.0, mmsegmentation 0.14.1, mmdet3d v0.17.1 `f1107977` (built from source, nvcc 11.8, g++-11, sm_86), detectron2 `5aeb252` (source build; imported by the BEVFormer plugin only for DD3D, unused by tiny), numpy 1.19.5, numba 0.48.0, nuscenes-devkit 1.1.9, pillow<10. Full list: `artifacts/provenance/pip_freeze_bevformer.txt` |
| numerics | fp32, TF32 disabled (`--no-tf32`), deterministic: clean-twice max diff 0.0 (pilot and val) |

## 1b. BEVFormer backup configuration (plan p.5: "Name a compatible backup configuration")

| item | value |
|---|---|
| backup | **BEVFormer-tiny_fp16** — same architecture/resolution/temporal mechanism as tiny, trained with fp16 (detector class `BEVFormer_fp16` inherits BEVFormer's test path unchanged) |
| config | `bevformer_src/projects/configs/bevformer_fp16/bevformer_tiny_fp16.py` (diff vs tiny: detector type, batch/workers, lr, video runner, `fp16=dict(loss_scale=512.)`, TransferWeight hook — no model-shape or test-pipeline change) |
| checkpoint | `ckpts/bevformer_tiny_fp16_epoch_24.pth` from https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_tiny_fp16_epoch_24.pth, sha256 `cbfc7cc13d2610b68a87874842bbdf0e4f3ec0526a3b98ee21c013e6dec729e8`; released log `artifacts/provenance/bevformer_tiny_fp16_epoch_24.log` (epoch 24: NDS 0.3588, mAP 0.2571; README 35.9/25.7) |
| status | **VERIFIED on this machine** (not just a documented candidate): gate declared before the result (`artifacts/provenance/BACKUP_GATE_DECLARATION.txt`, ±0.010 vs 0.359/0.257); full-val clean with the study adapter: NDS **0.3581** / mAP **0.2567** → PASS; audit CONSISTENT, fresh state at 150/150 scene starts; pilot clean + 2 s outage verified (`artifacts/runs_backup/`). fp16 inference, TF32 off |
| how | `python -m outage.run_bevformer --model tiny_fp16 ...` (same adapter, audit, reset; fp16 inference via `wrap_fp16_model` as upstream test.py); `artifacts/run_backup.sh` |

## 2. StreamPETR (Member 2) — fields from Member 2's accepted package (`deliverables/member2/`), hash re-verified here

| item | value |
|---|---|
| code | https://github.com/exiawsh/StreamPETR @ `95f64702306ccdb7a78889578b2a55b5deb35b2a`; mmdetection3d v1.0.0rc6 `47285b3f1e9dba358e98fcd12e523cfd0769c876` |
| config | `StreamPETR/streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py`, sha256 `391745f495546f21db2a91431dac52f2cb74caf39551abb28c730fd86357e40d` (R50, 900 queries, 704x256, 90 ep) |
| checkpoint source | https://github.com/exiawsh/storage/releases/download/v1.0/stream_petr_r50_flash_704_bs2_seq_90e.pth (StreamPETR README model table, "R50 - 900q, ImageNet, 90ep, 53.7/43.2") → `StreamPETR/ckpts/` |
| checkpoint sha256 | `e6323ae5c31adf1eedd46d6dd4fd3c73d95aa26f18cc8aa23c196494b7de3451` (re-hashed today = every Member 2 `run.json`) |
| pretraining | `img_backbone.pretrained='torchvision://resnet50'` (ImageNet-1k), `load_from=None` — from the released training config/log (`deliverables/member2/provenance/upstream_training_log_and_config/log_config.py`, lines 527 / 500) |
| detector-training split | `nuscenes2d_temporal_infos_train.pkl` (nuScenes train); val infos only for evaluation (same file, line 171 / 273) |
| evaluation-scene overlap | none known for val; pilot scenes are train-split (seen in training) |
| temporal mechanism | propagated object-query memory, reset at scene start (Member 2 state checks) |
| env / numerics | conda `streampetr_fa`, real flash-attn 0.2.2, fp16, TF32 off (`deliverables/member2/provenance/flashbuild/`) |
| images consumed | 6 keyframe cams/sample, no sweeps |
| baseline gate | NDS 0.5370 / mAP 0.4332 vs 0.537 / 0.432: PASS (Member 2) |

## 3. SparseBEV (Member 3) — frozen model IDENTITY recorded here; integration/baseline NOT run by Member 1

| item | value |
|---|---|
| code (FROZEN) | https://github.com/MCG-NJU/SparseBEV @ **`d34105285f6a50aeb49dd62ef1486860650d1aab`** ("Add new setting: r50_nuimg_704x256", 2023-08-31) — the commit that introduced this config; the checkpoint file is dated 2023-08-31 (gdrive). Earlier commits (`9b9ffc4` code release, `29ee3c9` bug fix) do not contain this config. Later commits change `models/sparsebev*.py`, `models/utils.py`, `models/csrc/wrapper.py` (native sampling op, comments, eval OOM fix) and add a future-frame loader class; `loaders/pipelines/loading.py` has **no changed/removed lines** for `LoadMultiViewImageFromMultiSweeps` after d341052, and `collect_sweeps` only changes `into_future` (unused by the past-only loader). Member 3 must run d341052 unless a clean-baseline check justifies otherwise |
| config | `configs/r50_nuimg_704x256.py` at d341052, sha256 `c6ac620a3231ccdf74d500c1c911ea81d876846a6bfc3f8f9253424aeea867b8` (copy: `artifacts/provenance/sparsebev_r50_nuimg_704x256_at_d341052.py`); HEAD differs only in `stop_prev_grad=False→0` and the test ann_file name (HEAD sha256 `5db43061…6d4036`). `num_frames=8`, 900 queries, 704x256 |
| checkpoint | `r50_nuimg_704x256.pth` (README gdrive id `1ft34-pxLpHGo2Aw-jowEtCxyXcqszHNn`) at `~/storage/ckpts_shared/sparsebev/`, sha256 `1fa2e196973f717407cbbe71a4381731ab21e5152528cdbe2e6586efcb6c0478` (178,604,339 B). Holds only `state_dict` (385 tensors): **no embedded meta/config, so the training split cannot be confirmed from the file**; the config trains on `nuscenes_infos_train_sweep.pkl` (README: 55.6 NDS val) |
| pretraining | `load_from` nuImages Cascade Mask R-CNN R50 (`cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth`, sha256 `4096396018c0cf59fbe0eb1afe6e269f4676b34460bed5eedde5d7680d58bb4e`, prefix = openmmlab tag `40963960`). **Disclosure:** nuImages is a separate dataset from the same fleet/cities; image-level overlap with nuScenes val is not documented and cannot be excluded |
| images consumed (replicated, not executed) | test mode picks prev indices 5,11,…,41 of [5 sweeps, prev keyframe, 5 sweeps, …] → **exactly the 7 previous KEYFRAMES; zero camera sweeps** for all 6,019 val / 402 pilot samples (`artifacts/loader_manifests.json`); boundary padding: first sample of each scene repeats its own keyframe 7x (150 val), next 6 samples clamp to the oldest keyframe (900 val). Member 3 must confirm against the real loader |
| execution path / feature cache (source review, d341052) | `LoadMultiViewImageFromMultiSweeps.__call__` uses **`load_online` when world_size == 1 and test_mode** (single-GPU `val.py`): it decodes only the 6 CURRENT images and passes history filenames; `SparseBEV.simple_test_online` reuses history features from `self.memory`, a cache **keyed by image filename** (FIFO-evicted). Consequences for Member 3: (i) actual JPEG reads ≈ 6/sample, not the 48 replicated slots (`load_offline`, multi-GPU); (ii) a blacked frame's features are cached under the clean file's name, which correctly keeps its outage status as history *within* a run, but the cache **must be cleared (new model state) at every run/condition start** or a clean cache would leak into an outage run (and vice versa) — plan p.2 "cached features separated by condition"; (iii) online and offline paths must be checked to give the same predictions before use |
| temporal mechanism | finite window of 8 keyframes (current + 7 previous; oldest observed 3.9 s back), no recurrent state → after the last black front keyframe leaves the window (7 keyframes ≈ 3.5 s after restoration) predictions should return to clean (plan's finite-window check applies here) |

## 4. Provenance uncertainty to disclose in the report
* BEVFormer and StreamPETR were trained on the nuScenes train split; the pilot scenes (train split) were seen in training.
* SparseBEV: no metadata in the checkpoint; split taken from the released config/README; nuImages pretraining overlap unknown;
  frozen identity d341052 recorded by Member 1, integration and baseline gate pending (Member 3).
* The comparison is between complete released systems (pretraining, resolution 800x450 vs 704x256, queries, history differ).
