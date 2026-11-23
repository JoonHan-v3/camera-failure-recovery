# STORAGE_PLAN.md — shared data inventory and storage plan (Member 1)

Study plan p.2: *"Before downloading, we will count the unique metadata, keyframe images, and historical sweeps required by the
selected loaders, along with environments, checkpoints, predictions, and peak extraction space. External storage is a
contingency, not an assumed resource."* Every number below is measured on this machine (2026-11-22) by
`python -m outage.data_inventory artifacts/data_inventory.json` (file-by-file reconciliation of the raw v1.0-trainval JSON
against disk) or `du`/`df`; GB = 10^9 bytes.

## 1. What exists (measured)

| item | location | size | status |
|---|---|---|---|
| downloaded archives (10 trainval blobs + meta + mini) | `~/storage/nuscenes_dl/*.tgz` | 319.1 GB | complete |
| extracted trainval blobs (samples + sweeps, all sensors) | `~/storage/nuscenes_dl/v1.0-trainval??_blobs/` | 417.1 GB | **every sample_data file in the JSON present (keyframes and sweeps, all channels); 0 files not in the JSON** |
| keyframe copy used by BEVFormer/StreamPETR loaders (6 CAM + LIDAR_TOP keyframes) | `~/storage/nuscenes_trainval_keyframes` (via `~/storage/nuscenes_trainval_root/samples`) | 54.5 GB | 34,149 files per channel; byte-identical to the blobs (md5, 300/300 random sample; sizes all equal) |
| metadata v1.0-trainval (+ maps, mini) | `~/storage/nuscenes/` | 2.6 GB (tables) | complete |
| CAN-bus expansion (BEVFormer only) | `~/storage/nuscenes_can_bus/can_bus` (zip 0.78 GB in `BEVFormer/dl`) | 4.6 GB | 7,832 files |
| shared checkpoints | `BEVFormer/ckpts` 0.40 GB; `StreamPETR/ckpts` 0.15 GB; `~/storage/ckpts_shared/sparsebev` 0.18 + 0.31 GB pretrain | ~1.1 GB | hashes in MODELS.md |
| conda envs | `bevformer` 11 GB, `streampetr` 5.7 GB, `streampetr_fa` 5.8 GB | 22.5 GB | SparseBEV env: Member 3 (est. ~6 GB) |
| BEVFormer infos (pilot 10 scenes + val 150 scenes) | `BEVFormer/artifacts/infos` | 83 MB | built by upstream converter, restricted to frozen scenes |

Hardware: `/` (NVMe) 916 GB, 487 GB free; `/mnt/storage` (= `~/storage`, RAID md127) 33 TB, 31 TB free; 219 GB RAM;
128 threads; RTX 3060 12 GB. **The plan's 159 GB disk figure is obsolete: external storage is now the primary data location
and has ample space; nothing in this plan depends on it growing.**

## 2. What each loader needs: exact manifests for the frozen evaluation and pilot scenes

Computed by `python -m outage.loader_manifests artifacts/loader_manifests.json` from the raw JSON (file paths + on-disk sizes).
BEVFormer/StreamPETR rows are the loaders' verified behaviour (the real runs' audits contain exactly these files); the SparseBEV
row is a **replication of its pinned test-mode selection logic (d341052), not an execution of SparseBEV code** — Member 3 must
confirm it against the real loader's filename list.

| split | model | samples | consumed image slots | unique files | bytes | camera sweeps | boundary padding |
|---|---|---|---|---|---|---|---|
| val (150 scenes) | BEVFormer-tiny, StreamPETR | 6,019 | 36,114 | 36,114 keyframes | 5.35 GB | 0 | — (recurrent state starts fresh) |
| val | SparseBEV r50 (8 frames) | 6,019 | 288,912 | **36,114 keyframes (same files)** | 5.35 GB | **0** | 150 first-of-scene samples repeat own keyframe 7x; 900 samples clamp to the oldest keyframe |
| pilot (10 train scenes) | BEVFormer-tiny, StreamPETR | 402 | 2,412 | 2,412 keyframes | 0.35 GB | 0 | — |
| pilot | SparseBEV r50 | 402 | 19,296 | 2,412 keyframes (same) | 0.35 GB | 0 | 10 repeat / 60 clamp |
| **union, all three models** | | | | **val 36,114 + pilot 2,412 files** | **5.71 GB** | 0 | |

Slots vs reads: the 48 SparseBEV slots/sample are the offline (multi-GPU) path; single-GPU test mode at d341052 uses the online
path, which decodes only the 6 current images and takes history features from a filename-keyed cache — so actual JPEG reads are
~6/sample and the cache must be reset per run/condition (MODELS.md §3). The unique-file set is the same either way.

Why SparseBEV reads no sweeps: its sweep info holds exactly 5 sweeps before each keyframe, and `collect_sweeps` appends the previous
keyframe after them, so the history list is [5 sweeps, keyframe k-1, 5 sweeps, keyframe k-2, …]; test mode takes indices
5, 11, …, 41 = keyframes k-1 … k-7 (oldest observed 3.9 s back). CAM_FRONT files inside the outage windows are therefore the same
168 / 326 / 635 (val) and 11 / 21 / 42 (pilot) keyframes for all three models, with 0 sweeps.

**Upper bound, not a loader manifest:** the val scenes contain 172,420 camera sweeps (25.6 GB; FRONT 28,929, FRONT_RIGHT
28,845, FRONT_LEFT 28,914, BACK 28,479, BACK_LEFT 28,328, BACK_RIGHT 28,925). That is the available-scene inventory (all sweeps
whose sample_token is a val sample), relevant only if a different SparseBEV setting (e.g. training-mode random intervals,
or a future-frame config) is chosen. All of them are present in the extracted blobs.
Extra for BEVFormer: CAN-bus pose JSON (4.6 GB extracted; only `*_pose.json` read).

## 3. Outputs per run (measured on BEVFormer)

| artifact | pilot (402 samples) | val (6,019 samples) |
|---|---|---|
| `preds.pkl` (300 boxes x 11 floats per sample, all queries kept) | 5.1 MB | 77 MB (measured) |
| `audit.json` (one record per consumed image + manifest) | 1.5 MB | 22 MB (measured) |
| official eval output (`results_nusc.json` 612 MB + metrics/plots) | — | 0.59 GB per evaluated run (measured; JSON of all 300 boxes/sample) |

Study matrix (3 models x 4 conditions + 1 repeat clean each = 15 val runs) ≈ 15 x (0.1 GB preds/audit + 0.6 GB eval json)
≈ **11 GB**; pilots and diagnostics < 1 GB. `results_nusc.json` is regenerable from `preds.pkl` (~3 min CPU) and is excluded
from packages (only `metrics.json`/`metrics_summary.json` are kept). StreamPETR measured 937 MB for its run directory.

## 4. Plan (decisions)

1. **Keep** the 10 `.tgz` archives (319 GB) as the immutable source; they are the only local source that does not require a new download. Do not
   re-extract: the extracted blobs (417 GB) already contain every file, including all sweeps (none needed by the frozen loaders).
2. **Use** `~/storage/nuscenes_trainval_root` (symlink root: keyframe copy + trainval tables + maps) for all three models — the
   replicated SparseBEV manifest needs only keyframes. Its info generator (`gen_sweep_info.py`) still writes sweep *paths*
   into the infos; if Member 3 wants those paths to resolve, add `sweeps/` symlinks into the extracted blobs (~0 extra space)
   rather than extracting another 25+ GB.
3. Peak extraction space if anything must be re-extracted: largest blob 41.7 GB compressed → ≤ ~55 GB extracted per blob;
   extract one blob at a time into a NEW empty directory (untrusted archives) — `/mnt/storage` has 31 TB free.
4. The duplicate keyframe copy (54.5 GB) may be deleted later to save space only after a symlink root into the blobs passes the
   same md5 reconciliation; not needed now (31 TB free).
5. Predictions, audits and eval JSON for all members go under each project's `artifacts/` and are packaged with SHA256SUMS
   (`deliverables/member{1,2,..}`); raw data, envs and checkpoints are referenced by path + hash, never copied into packages.
6. Subset contingency (plan p.2) is **not triggered**: storage and runtime fit the full 150-scene validation for BEVFormer
   (5 val runs ≈ 3.5 h with 3 concurrent processes).

## 5. GPU / runtime budget observed (BEVFormer, end-to-end incl. JPEG decode)
* peak allocated GPU memory 617 MB; ~2.1 GB per process incl. CUDA context and cache; **8 concurrent processes → CUDA OOM**
  (with ~3 GB used by the desktop); 3-4 concurrent processes are safe.
* ~0.5 s/frame alone, ~0.9 s/frame with 4 concurrent processes (decode/CPU bound); full val ≈ 90-100 min per run concurrently.
  Isolated batch-one GPU latency with CUDA synchronization is Member 4's profiling task.
