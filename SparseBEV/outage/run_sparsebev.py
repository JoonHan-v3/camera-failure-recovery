"""Real SparseBEV (r50_nuimg_704x256) runner with the front-camera outage hook, input audit and model-state checks.

Run from SparseBEV/sparsebev_src (cwd matters: config data_root is relative) in conda env `sparsebev`:
  PYTHONPATH=$PWD:$PWD/.. python -m outage.run_sparsebev --split pilot --condition clean --out ../artifacts/runs/pilot_a --no-tf32
(--condition: clean | 0.5 | 1 | 2 ; --split: pilot | val).
Pinned: SparseBEV d341052, mmdet3d v1.0.0rc6 (47285b3), mmcv-full 1.6.0, mmdet 2.28.2, torch 1.9.0+cu111, config
configs/r50_nuimg_704x256.py, checkpoint r50_nuimg_704x256.pth (released; README 55.6 NDS). fp32, TF32 off, batch 1, no workers.
Execution path: single GPU test mode -> LoadMultiViewImageFromMultiSweeps.load_online + SparseBEV.simple_test_online, i.e. the six
current images are decoded and history features come from SparseBEV.memory (FIFO of 8, keyed by image filename)."""
import argparse, hashlib, importlib, json, os, pickle, queue, subprocess, sys, time
import numpy as np, torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from . import compat; compat.install()
from .audit import InputAudit, CAUSAL_TOL_US, CAUSAL_POLICY
from .adapter import OutageFrontCamera, CorruptFrontCamera
from .nusc_ref import NuscRef
from .schedule import load_schedule
from .state import SceneStateGuard, StateResetError, reset_between_runs

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # SparseBEV/
SRC = os.path.join(PROJ, "sparsebev_src")
CFG = os.path.join(SRC, "configs/r50_nuimg_704x256.py")
CKPT = os.path.join(PROJ, "ckpts/r50_nuimg_704x256.pth")
NUSC = os.path.expanduser("~/storage/nuscenes")
SPLITS = dict(
    pilot=("artifacts/schedule_trainval_train.json", "artifacts/infos/pilot_infos.pkl"),
    val=("artifacts/schedule_trainval_val.json", "artifacts/infos/val_infos.pkl"))


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head(d):
    return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def build_test_dataset(infos_pkl, scene_tokens=None, online=True):
    """Upstream test dataset + OutageFrontCamera right after LoadMultiViewImageFromFiles, restricted to complete scenes.
    Dataset order = timestamp order, scenes contiguous (asserted): iteration is scene-by-scene exactly as upstream val.py."""
    os.chdir(SRC)
    cfg = Config.fromfile(CFG)
    importlib.import_module("models"); importlib.import_module("loaders")
    val = cfg.data.val
    val.ann_file, val.test_mode = infos_pkl, True
    assert val.pipeline[0]["type"] == "LoadMultiViewImageFromFiles"
    val.pipeline = [val.pipeline[0], dict(type="OutageFrontCamera")] + list(val.pipeline[1:])
    ds = build_dataset(val)
    if scene_tokens is not None:
        keep = set(scene_tokens)
        ds.data_infos = [i for i in ds.data_infos if i["scene_token"] in keep]
    ds.flag = np.zeros(len(ds.data_infos), dtype=np.int64)
    sw = [a["scene_token"] for a in ds.data_infos]
    assert sum(x != y for x, y in zip(sw, sw[1:])) == len(set(sw)) - 1, "scenes interleaved in dataset order"
    orig = ds.get_data_info

    def gdi(index):   # upstream dict + scene_token / eval_ts_us (needed by the shared outage step); img_timestamp stays float seconds
        d = orig(index)
        info = ds.data_infos[index]
        d["scene_token"], d["eval_ts_us"] = info["scene_token"], int(info["timestamp"])
        cams = list(info["cams"].values())
        assert [round(t * 1e6) for t in d["img_timestamp"]] == [int(c["timestamp"]) for c in cams], "abs_s round-trip not exact"
        return d
    ds.get_data_info = gdi
    return cfg, ds


class Runner:
    def __init__(self, infos_pkl, scene_tokens=None):
        cfg, self.dataset = build_test_dataset(infos_pkl, scene_tokens)
        cfg.model.pretrained, cfg.model.train_cfg = None, None
        self.model = build_model(cfg.model)
        load_checkpoint(self.model, CKPT, map_location="cpu", strict=True)
        self.model.CLASSES = self.dataset.CLASSES
        self.dp = MMDataParallel(self.model.cuda().eval(), device_ids=[0])
        self.det = self.dp.module
        self.ctx, self.model_guard, self.obs_log = None, None, []
        orig = self.det.simple_test_online

        def observed(img_metas, img=None, rescale=False):     # observe the REAL feature cache before every frame
            fns = img_metas[0]["filename"]
            cur, hist = fns[0], [fns[6 * k] for k in range(1, len(fns) // 6)]
            mem = self.det.memory
            self.obs_log.append(dict(scene=self.ctx[0], ts=self.ctx[1], cur_cached_before=cur in mem, cache_size_before=len(mem),
                                     n_hist=len(hist), n_hist_cached_before=sum(h in mem for h in hist if h != cur),
                                     n_hist_own=sum(h == cur for h in hist)))
            return orig(img_metas, img=img, rescale=rescale)
        self.det.simple_test_online = observed

    # -- state lifecycle ----------------------------------------------------------------------------
    def observe_reset_state(self):
        return len(self.det.memory) == 0 and self.det.queue.qsize() == 0

    def reset(self):
        def _reset():
            self.det.memory = {}; self.det.queue = queue.Queue()
        reset_between_runs(_reset, self.observe_reset_state)

    @torch.no_grad()
    def run(self, step, do_reset=True, use_model_guard=True):
        if do_reset:
            self.reset()
        OutageFrontCamera.STEP = step
        self.model_guard = SceneStateGuard() if use_model_guard else None
        self.obs_log = []
        preds, lat = {}, []
        for i in range(len(self.dataset)):
            info = self.dataset.data_infos[i]
            self.ctx = (info["scene_token"], int(info["timestamp"]))
            torch.cuda.synchronize(); t0 = time.time()
            data = mmcv.parallel.collate([self.dataset[i]], 1)
            res = self.dp(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize(); lat.append(time.time() - t0)
            o = self.obs_log[-1]
            if self.model_guard is not None:
                if o["cur_cached_before"]:     # a frame whose features are already cached = stale features from another run/scene
                    raise StateResetError(f"stale feature cache: current frame of {o['scene']}@{o['ts']} was cached before its own step")
                new = self.model_guard.prev_scene != o["scene"]
                self.model_guard.step(o["scene"], o["ts"], memory_fresh=(not o["cur_cached_before"]) if new else None,
                                      require_memory=True if new else False)
            b = res[0]["pts_bbox"]
            rows = torch.cat([b["boxes_3d"].tensor.float().cpu(), b["labels_3d"].float().cpu()[:, None],
                              b["scores_3d"].float().cpu()[:, None]], 1).numpy()
            preds[info["token"]] = rows        # [x,y,z,w,l,h,yaw,vx,vy,label,score]; z = box bottom (LiDAR frame)
        return preds, dict(latency_s=lat, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20, obs_log=self.obs_log,
                           model_guard_log=self.model_guard.log if self.model_guard else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--condition", required=True, help="clean | 0.5 | 1 | 2 (outage seconds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--scenes", default="")
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import set_random_seed
    set_random_seed(0, deterministic=True)            # as upstream val.py, but cudnn.benchmark stays False (repeatability)
    torch.backends.cudnn.benchmark = False
    out = os.path.abspath(a.out); os.makedirs(out, exist_ok=True)
    version = "v1.0-trainval"
    sched_path, info_src = SPLITS[a.split]
    ref = NuscRef(NUSC, version)
    sched_all, skipped, params = load_schedule(os.path.join(PROJ, sched_path))
    infos = os.path.join(PROJ, info_src)
    all_scenes = sorted({i["scene_token"] for i in pickle.load(open(infos, "rb"))["infos"]})
    scenes = a.scenes.split(",") if a.scenes else all_scenes
    sched = {t: sched_all[t] for t in scenes if t in sched_all}
    dur = 0.0 if a.condition == "clean" else float(a.condition)
    r = Runner(infos, scenes)
    keys, meta = ref.manifest(scenes)
    skew = max(k[3] - meta[(k[0], k[1])] for k in keys)
    if skew > CAUSAL_TOL_US:
        raise SystemExit(f"manifest has images {skew} us after the sample time > frozen policy {CAUSAL_TOL_US}")
    audit = InputAudit("sparsebev_r50_nuimg_704x256", "clean" if dur == 0 else "outage", dur, sched, run_id=a.out, causal_tolerance_us=CAUSAL_TOL_US)
    audit.set_manifest(keys, f"nusc-json:{version}:{len(scenes)}scenes")
    audit.set_sample_metadata(meta)
    step = CorruptFrontCamera(sched, dur, audit, guard=SceneStateGuard(), ts_mode="abs_s")
    t0 = time.time()
    preds, info = r.run(step)
    audit.save(os.path.join(out, "audit.json"))
    pickle.dump(preds, open(os.path.join(out, "preds.pkl"), "wb"))
    ol = info["obs_log"]
    first = [k == 0 or ol[k - 1]["scene"] != ol[k]["scene"] for k in range(len(ol))]
    starts = [(ol[k]["scene"], not ol[k]["cur_cached_before"]) for k in range(len(ol)) if first[k]]
    hist_ok = all(ol[k]["n_hist"] == 7 and ol[k]["n_hist_cached_before"] + ol[k]["n_hist_own"] == 7 for k in range(len(ol)) if not first[k])
    json.dump(dict(condition=a.condition, scenes=scenes, n_samples=len(preds), runtime_s=time.time() - t0,
                   latency_median_s=float(np.median(info["latency_s"])), peak_gpu_mb=info["peak_gpu_mb"],
                   audit_status=audit.status(), readiness=audit.readiness(), violations=audit.verify()[:20],
                   causal_tolerance_us=audit.causal_tolerance_us, causal_policy=CAUSAL_POLICY, manifest_max_skew_us=skew,
                   model_state_fresh_at_scene_starts=starts, never_cached_before_own_frame=all(not o["cur_cached_before"] for o in ol),
                   history_served_from_cache_after_first_frame=hist_ok, max_cache_size_before=max(o["cache_size_before"] for o in ol),
                   sparsebev_commit=git_head(SRC), mmdet3d_commit=git_head(os.path.join(PROJ, "mmdetection3d")),
                   ckpt_sha256=sha(CKPT), infos_sha256=sha(infos), config=os.path.relpath(CFG, PROJ), config_sha256=sha(CFG),
                   outage_code_sha256={f: sha(os.path.join(PROJ, "outage", f)) for f in sorted(os.listdir(os.path.join(PROJ, "outage"))) if f.endswith(".py")},
                   torch=torch.__version__, mmcv=mmcv.__version__, cuda_runtime=torch.version.cuda, argv=sys.argv, cwd_root=PROJ,
                   cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                   gpu=torch.cuda.get_device_name(0)), open(os.path.join(out, "run.json"), "w"), indent=1)
    print(a.condition, audit.status(), audit.readiness(), "samples", len(preds))


if __name__ == "__main__":
    main()
