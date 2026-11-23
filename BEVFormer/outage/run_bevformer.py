"""Real BEVFormer-tiny runner with the front-camera outage hook, input audit and model-state checks.

Run from BEVFormer/bevformer_src (cwd matters: config data_root is relative) in conda env `bevformer`:
  PYTHONPATH=$PWD:$PWD/.. python -m outage.run_bevformer --split pilot --condition clean --out ../artifacts/runs/pilot_a --no-tf32
(--condition: clean | 0.5 | 1 | 2 ; --split: pilot | val).
Pinned: BEVFormer 66b65f3, mmdet3d v0.17.1 (f110797), mmcv-full 1.4.0, mmdet 2.14.0, config bevformer/bevformer_tiny.py,
checkpoint bevformer_tiny_epoch_24.pth (released; NDS 0.354 / mAP 0.252). fp32, TF32 off."""
import argparse, hashlib, json, os, pickle, subprocess, sys, time
import numpy as np, torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from .audit import InputAudit, CAUSAL_TOL_US, CAUSAL_POLICY
from .mm_pipeline import OutageFrontCamera
from .nusc_ref import NuscRef, CAMS
from .schedule import load_schedule
from .state import SceneStateGuard, reset_between_runs
from .adapter import CorruptFrontCamera

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # BEVFormer/
SRC = os.path.join(PROJ, "bevformer_src")
MODELS = {  # name -> (config, checkpoint). 'tiny' = THE study model; 'tiny_fp16' = pinned compatible BACKUP (MODELS.md)
    "tiny": (os.path.join(SRC, "projects/configs/bevformer/bevformer_tiny.py"), os.path.join(PROJ, "ckpts/bevformer_tiny_epoch_24.pth")),
    "tiny_fp16": (os.path.join(SRC, "projects/configs/bevformer_fp16/bevformer_tiny_fp16.py"),
                  os.path.join(PROJ, "ckpts/bevformer_tiny_fp16_epoch_24.pth"))}
CFG, CKPT = MODELS["tiny"]
NUSC = os.path.expanduser("~/storage/nuscenes")
SPLITS = dict(  # name -> (schedule, info pkl)   schedules are Member 2's FROZEN shared schedules (copied, sha in README)
    pilot=("artifacts/schedule_trainval_train.json", "artifacts/infos/pilot_infos.pkl"),
    val=("artifacts/schedule_trainval_val.json", "artifacts/infos/val_infos.pkl"))
DEFAULT_PREV = lambda: {"prev_bev": None, "scene_token": None, "prev_pos": 0, "prev_angle": 0}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head(d):
    return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def build_test_dataset(infos_pkl, scene_tokens, cfg_path=None):
    """Upstream test dataset + OutageFrontCamera right after LoadMultiViewImageFromFiles, restricted to complete scenes.
    NuScenesDataset sorts infos by timestamp; scenes are contiguous in that order (verified: 149 switches / 150 val scenes),
    so iteration is scene-by-scene in timestamp order, exactly as upstream tools/test.py with one GPU."""
    os.chdir(SRC)
    cfg = Config.fromfile(cfg_path or CFG)
    import importlib; importlib.import_module("projects.mmdet3d_plugin")
    test = cfg.data.test
    test.ann_file, test.test_mode = infos_pkl, True
    test.pipeline = [test.pipeline[0], dict(type="OutageFrontCamera")] + list(test.pipeline[1:])
    ds = build_dataset(test)
    keep = set(scene_tokens)
    ds.data_infos = [i for i in ds.data_infos if i["scene_token"] in keep]
    ds.flag = np.zeros(len(ds.data_infos), dtype=np.int64)
    sw = [a["scene_token"] for a in ds.data_infos]
    assert sum(x != y for x, y in zip(sw, sw[1:])) == len(set(sw)) - 1, "scenes interleaved in dataset order"
    orig = ds.get_data_info

    def gdi(index):   # upstream get_data_info drops camera timestamps: add them (integer us) from the info itself
        d = orig(index)
        info = ds.data_infos[index]
        d["img_timestamp"] = [int(c["timestamp"]) for c in info["cams"].values()]
        assert [os.path.basename(f) for f in d["img_filename"]] == [os.path.basename(c["data_path"]) for c in info["cams"].values()]
        d["eval_ts_us"] = int(info["timestamp"])
        return d
    ds.get_data_info = gdi
    return cfg, ds


class Runner:
    def __init__(self, infos_pkl, scene_tokens, model="tiny"):
        cfg_path, self.ckpt = MODELS[model]
        cfg, self.dataset = build_test_dataset(infos_pkl, scene_tokens, cfg_path)
        cfg.model.pretrained, cfg.model.train_cfg = None, None
        self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16") is not None:               # backup tiny_fp16: fp16 inference as released (upstream test.py does the same)
            from mmcv.runner import wrap_fp16_model
            wrap_fp16_model(self.model)
        load_checkpoint(self.model, self.ckpt, map_location="cpu")
        self.model.CLASSES = self.dataset.CLASSES
        self.dp = MMDataParallel(self.model.cuda().eval(), device_ids=[0])
        self.det = self.dp.module
        self.ctx, self.model_guard, self.fresh_log = None, None, []
        orig = self.det.simple_test

        def observed(img_metas, img=None, prev_bev=None, rescale=False):   # observe the REAL recurrent state per frame
            scene, ts = self.ctx
            self.fresh_log.append((scene, ts, prev_bev is None))
            return orig(img_metas, img=img, prev_bev=prev_bev, rescale=rescale)
        self.det.simple_test = observed

    # -- state lifecycle ----------------------------------------------------------------------------
    def observe_reset_state(self):
        """True iff nothing of a previous run can survive: prev_bev freed, scene token cleared, ego deltas zero."""
        p = self.det.prev_frame_info
        return p["prev_bev"] is None and p["scene_token"] is None and p["prev_pos"] == 0 and p["prev_angle"] == 0

    def reset(self):
        def _reset():
            self.det.prev_frame_info = DEFAULT_PREV()   # upstream never resets between runs: a run starting on the
        reset_between_runs(_reset, self.observe_reset_state)  # previous run's last scene would silently reuse prev_bev

    @torch.no_grad()
    def run(self, step, do_reset=True, use_model_guard=True):
        if do_reset:
            self.reset()
        OutageFrontCamera.STEP = step
        self.model_guard = SceneStateGuard() if use_model_guard else None
        self.fresh_log = []
        preds, lat, hist = {}, [], []
        for i in range(len(self.dataset)):
            info = self.dataset.data_infos[i]
            self.ctx = (info["scene_token"], int(info["timestamp"]))
            torch.cuda.synchronize(); t0 = time.time()
            data = mmcv.parallel.collate([self.dataset[i]], 1)
            res = self.dp(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize(); lat.append(time.time() - t0)
            scene, ts, fresh = self.fresh_log[-1]
            if self.model_guard is not None:                 # observed AFTER the model's own scene-boundary check
                new = self.model_guard.prev_scene != scene
                self.model_guard.step(scene, ts, memory_fresh=fresh if new else None, require_memory=True if new else False)
            hist.append(self.det.prev_frame_info["prev_bev"] is not None)
            b = res[0]["pts_bbox"]
            rows = torch.cat([b["boxes_3d"].tensor.float().cpu(), b["labels_3d"].float().cpu()[:, None],
                              b["scores_3d"].float().cpu()[:, None]], 1).numpy()
            preds[info["token"]] = rows        # [x,y,z,w,l,h,yaw,vx,vy,label,score]
        return preds, dict(latency_s=lat, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
                           fresh_log=self.fresh_log, bev_cached_after_frame=hist,
                           model_guard_log=self.model_guard.log if self.model_guard else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="pilot")
    ap.add_argument("--condition", required=True, help="clean | 0.5 | 1 | 2 (outage seconds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tf32", action="store_true")
    ap.add_argument("--scenes", default="")
    ap.add_argument("--model", default="tiny", choices=sorted(MODELS))
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
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
    r = Runner(infos, scenes, a.model)
    keys, meta = ref.manifest(scenes)
    skew = max(k[3] - meta[(k[0], k[1])] for k in keys)
    if skew > CAUSAL_TOL_US:
        raise SystemExit(f"manifest has images {skew} us after the sample time > frozen policy {CAUSAL_TOL_US}")
    audit = InputAudit("bevformer_" + a.model, "clean" if dur == 0 else "outage", dur, sched, run_id=a.out, causal_tolerance_us=CAUSAL_TOL_US)
    audit.set_manifest(keys, f"nusc-json:{version}:{len(scenes)}scenes")
    audit.set_sample_metadata(meta)
    step = CorruptFrontCamera(sched, dur, audit, guard=SceneStateGuard(), ts_mode="abs_us")
    t0 = time.time()
    preds, info = r.run(step)
    audit.save(os.path.join(out, "audit.json"))
    pickle.dump(preds, open(os.path.join(out, "preds.pkl"), "wb"))
    fl = info["fresh_log"]
    first = [k == 0 or fl[k - 1][0] != fl[k][0] for k in range(len(fl))]
    starts = [(fl[k][0], fl[k][2]) for k in range(len(fl)) if first[k]]            # prev_bev None at every scene start
    history_ok = all(not fl[k][2] for k in range(len(fl)) if not first[k])         # prev_bev present on every later frame
    json.dump(dict(condition=a.condition, scenes=scenes, n_samples=len(preds), runtime_s=time.time() - t0,
                   latency_median_s=float(np.median(info["latency_s"])), peak_gpu_mb=info["peak_gpu_mb"],
                   audit_status=audit.status(), readiness=audit.readiness(), violations=audit.verify()[:20],
                   causal_tolerance_us=audit.causal_tolerance_us, causal_policy=CAUSAL_POLICY, manifest_max_skew_us=skew,
                   model_state_fresh_at_scene_starts=starts, bev_history_present_after_first_frame=history_ok,
                   bevformer_commit=git_head(SRC), mmdet3d_commit=git_head(os.path.join(PROJ, "mmdetection3d")),
                   model=a.model, ckpt_sha256=sha(r.ckpt), infos_sha256=sha(infos), outage_code_sha256={f: sha(os.path.join(PROJ, "outage", f)) for f in sorted(os.listdir(os.path.join(PROJ, "outage"))) if f.endswith(".py")}, config=os.path.relpath(MODELS[a.model][0], PROJ), torch=torch.__version__,
                   mmcv=mmcv.__version__, cuda_runtime=torch.version.cuda, argv=sys.argv, cwd_root=PROJ,
                   tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32,
                   gpu=torch.cuda.get_device_name(0)), open(os.path.join(out, "run.json"), "w"), indent=1)
    print(a.condition, audit.status(), audit.readiness(), "samples", len(preds))


if __name__ == "__main__":
    main()
