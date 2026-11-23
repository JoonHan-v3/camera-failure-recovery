# MODELS.md — SparseBEV record (Member 3)

Study plan: *"MODELS.md will record configuration paths, code commits, checkpoint sources, file hashes, pretraining, and stated detector-training
splits. We will audit the available provenance for evaluation-scene overlap and verify causal inference."* Machine-readable copy:
`artifacts/ckpt_records.json`. Member 1 froze this identity before integration (`../BEVFormer/MODELS.md` §3); this file records what was
actually run and what was verified. BEVFormer and StreamPETR records: `../BEVFormer/MODELS.md`, `../StreamPETR/README.md`.

## 1. SparseBEV (the study model)
| item | value |
|---|---|
| paper / release | SparseBEV, ICCV 2023, "Original ICCV 2023 R50 release" (`r50_nuimg_704x256`); later variants (e.g. `r50_nuimg_704x256_400q_36ep`) are **not** substituted |
| code | https://github.com/MCG-NJU/SparseBEV @ `d34105285f6a50aeb49dd62ef1486860650d1aab` ("Add new setting: r50_nuimg_704x256"), working tree clean, **unmodified** (`git status` empty) |
| config | `sparsebev_src/configs/r50_nuimg_704x256.py` sha256 `c6ac620a…aeea867b8` (copy in `artifacts/provenance/`); 8 frames, 900 queries, 704x256, ResNet-50 + FPN |
| checkpoint | `r50_nuimg_704x256.pth`, sha256 `1fa2e196973f717407cbbe71a4381731ab21e5152528cdbe2e6586efcb6c0478` (178,604,339 B), README gdrive id `1ft34-pxLpHGo2Aw-jowEtCxyXcqszHNn`; stored at `~/storage/ckpts_shared/sparsebev/`, linked from `ckpts/` |
| pretraining | nuImages Cascade Mask R-CNN R50 (`load_from` in the config), sha256 `40963960…d5bb4e` |
| detector training split | `nuscenes_infos_train_sweep.pkl` (nuScenes **train** split) per the released config; the checkpoint holds only a `state_dict` (no embedded meta/config), so the split cannot be confirmed from the file |
| released result | README model zoo: NDS 55.6 (no mAP published for this setting); noise ~0.3 NDS |
| mmdet3d | v1.0.0rc6 (`47285b3`), own clone `mmdetection3d/`, built from source; mmcv-full 1.6.0, mmdet 2.28.2, mmseg 0.30.0, torch 1.9.0+cu111, numpy 1.23.5 (conda env `sparsebev`, cloned from `streampetr`, CUDA 11.8 toolchain, `artifacts/provenance/{pip_freeze,conda_export}_sparsebev.txt`) |
| CUDA extension | `models/csrc` `_msmv_sampling_cuda` built with nvcc 11.8 / g++-11 for sm_86 (sha256 in `ckpt_records.json`) |
| numerics | fp32, TF32 off, `cudnn.benchmark=False`, `set_random_seed(0, deterministic=True)`, batch 1, no workers |

## 2. Deviations from the upstream install, and why they do not change results
1. **torch 1.9.0 instead of 2.0.0 / 1.10.2** (the env was cloned from the StreamPETR stack; mmcv-full 1.6.0 has a cu111/torch1.9 wheel).
   The only incompatibility is `torch.meshgrid(..., indexing='ij')` (`models/sparsebev_head.py:59`); in torch 1.9 `meshgrid` has no
   `indexing` argument and *is* `'ij'`. `outage/compat.py` drops the keyword (any other value is rejected); upstream source untouched.
2. **cudnn.benchmark False** (upstream `val.py` sets True) so repeated runs are bit-identical; the baseline gate checks accuracy.
3. **Own info generation** (`outage/build_infos.py`): upstream mmdet3d `_fill_trainval_infos` + SparseBEV's own `gen_sweep_info.add_sweep_info`,
   restricted to the frozen pilot/val scenes (the Google-Drive info files were not used); one added key `scene_token`.
4. `turbojpeg`/`pillow-simd` (optional speed-ups) are not used; decoding goes through the mmcv fallback backend.
The clean-baseline gate (NDS 0.556 ± 0.010, declared before the result: `artifacts/provenance/BASELINE_GATE_DECLARATION.txt`) is the check
that these deviations do not matter. Result: see README §6 / `../deliverables/member3/SUMMARY.md`.

## 3. Temporal mechanism and execution path (source review at d341052, then verified by execution)
* **Finite window**: current keyframe + 7 previous keyframes (`num_frames = 8`; test interval 6 sweeps = one keyframe), oldest observed 3.9 s
  back; **no recurrent state**. After the last black front keyframe has left the window (7 keyframes ≈ 3.5 s after restoration) the
  predictions return to the clean run — verified bit-exactly (README §6, `artifacts/window_check_*.json`).
* **Images consumed** (executed, not replicated: `outage/confirm_loader_manifest.py`, `artifacts/loader_confirm_{pilot,val}.json`): for every
  sample the real `get_data_info` + `LoadMultiViewImageFromMultiSweeps` select the current keyframe and the 7 previous *keyframes* of the same
  scene (clamped at the scene start; first sample repeats its own keyframe): val 288,912 slots, **36,114 unique keyframe files** (= the six
  cameras of the 6,019 evaluated samples), **0 camera sweeps**, 150 first-of-scene samples, 900 clamped; pilot 19,296 slots / 2,412 files. The
  online (single-GPU) and offline (multi-GPU) loaders pick identical filenames. This confirms Member 1's replicated manifest.
* **Execution path**: with one GPU, `LoadMultiViewImageFromMultiSweeps.__call__` uses `load_online` and `SparseBEV.simple_test` uses
  `simple_test_online`: only the six CURRENT images are decoded; history features come from `SparseBEV.memory`, a dict **keyed by image
  filename** (FIFO of 8). Consequences handled by the study runner: (i) a corrupted frame's features are cached under the clean file's
  name, which keeps its outage status as history within a run (the intended behaviour); (ii) the cache is cleared at every run start
  and the runner **rejects any frame whose own filename is already cached** (the cache is cleared per run, not physically emptied at scene boundaries; other-scene entries are keyed by other filenames and never read) (stale features from another run/condition would silently be
  reused — demonstrated in `artifacts/isolation_pilot.json`); (iii) online vs offline predictions compared in `artifacts/online_offline_check.json`.
* **Native decoding**: `NMSFreeCoder` keeps the top-300 queries with score > 0.05 inside `post_center_range`, so SparseBEV emits 0–300
  rows per sample (val: mean 215.2, one sample with none; pilot mean 217.5), unlike BEVFormer/StreamPETR (always 300, no score floor). The common format is otherwise identical.

## 4. Provenance uncertainty to disclose in the report
* SparseBEV: no metadata in the checkpoint; split taken from the released config/README; nuImages pretraining overlap with nuScenes val
  cannot be excluded (a separate dataset from the same fleet/cities, image-level overlap undocumented).
* The pilot/calibration scenes come from the nuScenes **train** split and were seen by all three released models during training (BEVFormer,
  StreamPETR, SparseBEV); calibrated thresholds therefore reflect train-set behaviour and are reported with their achieved precision/recall.
* The comparison is between complete released systems (pretraining, resolution 704x256 vs 800x450, queries, history differ).
