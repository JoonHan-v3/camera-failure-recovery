"""Real StreamPETR runner with the front-camera outage hook, input audit and model-state checks.

Run from the StreamPETR project root with the `streampetr` conda env (never `python -I`: it ignores PYTHONPATH):
  PYTHONPATH=outage/shims:streampetr_src:. python -m outage.run_streampetr --split mini --condition clean --out artifacts/runs/mini_clean_a
  (--condition: clean | 0.5 | 1 | 2 ; --split: mini | pilot | val)
Pinned: StreamPETR 95f6470, mmdet3d v1.0.0rc6 (47285b3), config stream_petr_r50_flash_704_bs2_seq_90e.
flash-attn is replaced by outage/shims/flash_attn (exact fp32 softmax attention; see README)."""
import argparse, copy, hashlib, json, os, pickle, sys, time
import numpy as np, torch
import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from .audit import InputAudit, CAUSAL_TOL_US, CAUSAL_POLICY
from .mm_pipeline import OutageFrontCamera
from .nusc_ref import NuscRef, CAMS
from .schedule import load_schedule
from .state import SceneStateGuard, reset_between_runs, StateResetError
from .streampetr_adapter import CorruptFrontCamera

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = os.path.join(ROOT, "streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py")
CKPT = os.path.join(ROOT, "ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth")
NUSC = os.path.expanduser("~/storage/nuscenes")
SPLITS = dict(  # name -> (nuScenes version, frozen schedule, info pkl(s))
    mini=("v1.0-mini", "artifacts/schedule_mini.json",
          [f"streampetr_src/data/nuscenes/nuscenes2d_temporal_infos_{s}.pkl" for s in ("train", "val")]),
    pilot=("v1.0-trainval", "artifacts/schedule_trainval_train.json", "artifacts/trainval_infos/pilot_train_infos.pkl"),
    val=("v1.0-trainval", "artifacts/schedule_trainval_val.json", "artifacts/trainval_infos/val_infos.pkl"))


def _flash_info():
    import flash_attn
    from importlib import metadata
    f = os.path.abspath(flash_attn.__file__)
    shim = "/outage/shims/" in f
    info = dict(module=f, kind="pure-PyTorch shim" if shim else "REAL flash-attn")
    try:
        info["dist_version"] = metadata.version("flash_attn")
    except Exception:
        info["dist_version"] = None
    if not shim:
        import flash_attn_cuda
        ext = os.path.abspath(flash_attn_cuda.__file__)
        info.update(cuda_extension=ext, cuda_extension_sha256=sha(ext))
    return info


def sha(path, algo="sha256"):
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_head(d):
    import subprocess
    return subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


class Runner:
    def __init__(self, version, infos_pkl, scene_tokens, fp16=True):
        os.chdir(os.path.join(ROOT, "streampetr_src"))
        cfg = Config.fromfile(CFG)
        import importlib; importlib.import_module("projects.mmdet3d_plugin")
        test = cfg.data.test
        test.ann_file, test.test_mode = infos_pkl, True
        test.pipeline = [test.pipeline[0], dict(type="OutageFrontCamera")] + list(test.pipeline[1:])
        self.dataset = build_dataset(test)
        keep = set(scene_tokens)
        self.dataset.data_infos = [i for i in self.dataset.data_infos if i["scene_token"] in keep]
        self.dataset.flag = np.zeros(len(self.dataset.data_infos), dtype=np.int64)
        cfg.model.pretrained, cfg.model.train_cfg = None, None
        self.model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
        if cfg.get("fp16") is not None and fp16:
            wrap_fp16_model(self.model)
        load_checkpoint(self.model, CKPT, map_location="cpu")
        self.model.CLASSES = self.dataset.CLASSES
        self.dp = MMDataParallel(self.model.cuda().eval(), device_ids=[0])
        self.det = self.dp.module
        self.head = self.det.pts_bbox_head
        self.ctx = None
        self.model_guard = None
        orig = self.head.pre_update_memory

        def observed(data):                       # observe the REAL memory after refresh at each frame
            orig(data)
            scene, ts = self.ctx
            fresh = None
            if float(data["prev_exists"].flatten()[0]) == 0.0:
                h = self.head
                fresh = bool(all(float(t.abs().max()) == 0.0 for t in
                                 (h.memory_embedding, h.memory_timestamp, h.memory_velo)))
            self.fresh_log.append((scene, ts, fresh))
            if self.model_guard is not None:
                self.model_guard.step(scene, ts, memory_fresh=fresh)
        self.head.pre_update_memory = observed
        self.fresh_log = []

    # -- state lifecycle ----------------------------------------------------------------------------
    def observe_reset_state(self):
        """True iff nothing of the previous run can survive: scene token cleared and every recurrent
        buffer deallocated (reset_memory sets them to None; they are reallocated, zero-filled and
        refreshed with prev_exists=0 on the next frame)."""
        h = self.head
        return (self.det.prev_scene_token is None and h.memory_embedding is None and
                h.memory_reference_point is None and h.memory_timestamp is None and
                h.memory_egopose is None and h.memory_velo is None)

    def reset(self):
        def _reset():
            self.det.prev_scene_token = None      # NOT cleared by reset_memory(): without this a run that
            self.head.reset_memory()              # starts on the previous run's last scene would not reset
        reset_between_runs(_reset, self.observe_reset_state)

    # -- one pass -------------------------------------------------------------------------------------
    @torch.no_grad()
    def run(self, audit, step, scene_order=None, do_reset=True, use_model_guard=True):
        if do_reset:
            self.reset()
        OutageFrontCamera.STEP = step
        self.model_guard = SceneStateGuard() if use_model_guard else None
        self.fresh_log = []
        preds, lat = {}, []
        for i in range(len(self.dataset)):
            info = self.dataset.data_infos[i]
            self.ctx = (info["scene_token"], int(info["timestamp"]))
            torch.cuda.synchronize(); t0 = time.time()
            data = mmcv.parallel.collate([self.dataset[i]], 1)
            res = self.dp(return_loss=False, rescale=True, **data)
            torch.cuda.synchronize(); lat.append(time.time() - t0)
            b = res[0]["pts_bbox"]
            rows = torch.cat([b["boxes_3d"].tensor.float().cpu(), b["labels_3d"].float().cpu()[:, None],
                              b["scores_3d"].float().cpu()[:, None]], 1).numpy()
            preds[info["token"]] = rows        # [x,y,z,w,l,h,yaw,vx,vy,label,score]
        return preds, dict(latency_s=lat, peak_gpu_mb=torch.cuda.max_memory_allocated() / 2**20,
                           fresh_log=self.fresh_log, model_guard_log=self.model_guard.log if self.model_guard else None)


def make_infos(version, tokens_split_pkls, out_pkl):
    """Combine info pkls (all mini scenes), rewriting relative image paths to absolute."""
    infos = []
    for p in tokens_split_pkls:
        infos += pickle.load(open(p, "rb"))["infos"]
    base = os.path.join(ROOT, "streampetr_src/tools")
    for inf in infos:
        for c in inf["cams"].values():
            c["data_path"] = os.path.abspath(os.path.join(base, c["data_path"]))
    pickle.dump(dict(infos=infos, metadata=dict(version=version)), open(out_pkl, "wb"))
    return out_pkl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mini")
    ap.add_argument("--condition", required=True, help="clean | 0.5 | 1 | 2 (outage seconds)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-tf32", action="store_true", help="disable TF32 matmul/cuDNN (torch 1.9 enables it by default on Ampere)")
    ap.add_argument("--scenes", default="", help="comma-separated scene tokens (default: all in split)")
    a = ap.parse_args()
    if a.no_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    os.makedirs(os.path.join(ROOT, a.out), exist_ok=True)
    out = os.path.join(ROOT, a.out)
    version, sched_path, info_src = SPLITS[a.split]
    ref = NuscRef(NUSC, version)
    sched_all, skipped, params = load_schedule(os.path.join(ROOT, sched_path))
    if isinstance(info_src, list):
        infos = make_infos(version, [os.path.join(ROOT, x) for x in info_src], os.path.join(out, "infos_abs.pkl"))
        all_scenes = sorted(ref.scenes)
    else:
        infos = os.path.join(ROOT, info_src)             # already absolute-path infos (cross-checked vs nuScenes JSON)
        all_scenes = sorted({i["scene_token"] for i in pickle.load(open(infos, "rb"))["infos"]})
    scenes = a.scenes.split(",") if a.scenes else all_scenes
    sched = {t: sched_all[t] for t in scenes if t in sched_all}
    dur = 0.0 if a.condition == "clean" else float(a.condition)
    r = Runner(version, infos, scenes)
    keys, meta = ref.manifest(scenes)
    skew = max(k[3] - meta[(k[0], k[1])] for k in keys)          # observed cam-minus-sample skew (us), this manifest
    if skew > CAUSAL_TOL_US:
        raise SystemExit(f"manifest has images {skew} us after the sample time > frozen policy {CAUSAL_TOL_US}")
    audit = InputAudit("streampetr_r50_704_90e", "clean" if dur == 0 else "outage", dur, sched,
                       run_id=a.out, causal_tolerance_us=CAUSAL_TOL_US)
    audit.set_manifest(keys, f"nusc-json:{version}:{len(scenes)}scenes")
    audit.set_sample_metadata(meta)
    pipe_guard = SceneStateGuard()
    step = CorruptFrontCamera(sched, dur, audit, guard=pipe_guard, ts_mode="abs_s")
    t0 = time.time()
    preds, info = r.run(audit, step)
    audit.save(os.path.join(out, "audit.json"))
    pickle.dump(preds, open(os.path.join(out, "preds.pkl"), "wb"))
    json.dump(dict(condition=a.condition, scenes=scenes, n_samples=len(preds), runtime_s=time.time() - t0,
                   latency_median_s=float(np.median(info["latency_s"])), peak_gpu_mb=info["peak_gpu_mb"],
                   audit_status=audit.status(), readiness=audit.readiness(), violations=audit.verify()[:20],
                   causal_tolerance_us=audit.causal_tolerance_us, causal_policy=CAUSAL_POLICY, manifest_max_skew_us=skew,
                   model_state_fresh_at_scene_starts=[(s, f) for s, _, f in info["fresh_log"] if f is not None],
                   streampetr_commit=git_head(os.path.join(ROOT, "streampetr_src")),
                   mmdet3d_commit=git_head(os.path.join(ROOT, "streampetr_src/mmdetection3d")),
                   ckpt_sha256=sha(CKPT), config=os.path.relpath(CFG, ROOT), torch=torch.__version__,
                   flash_attn=_flash_info(),
                   flash_shim_mode=(os.environ.get("FLASH_SHIM_MODE", "fp32-softmax(default)") if "/outage/shims/" in os.path.abspath(sys.modules["flash_attn"].__file__) else "unused (real kernel)"),
                   flash_shim_sha256=(sha(os.path.join(ROOT, "outage/shims/flash_attn/flash_attn_interface.py")) if "/outage/shims/" in os.path.abspath(sys.modules["flash_attn"].__file__) else "unused (real kernel)"),
                   cuda_runtime=torch.version.cuda, nvcc_used_for_flash="11.8 (conda streampetr_fa, gcc-11 host)",
                   argv=sys.argv, cwd_root=ROOT, tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                   tf32_cudnn=torch.backends.cudnn.allow_tf32, gpu=torch.cuda.get_device_name(0)), open(os.path.join(out, "run.json"), "w"), indent=1)
    print(a.condition, audit.status(), audit.readiness(), "samples", len(preds))


if __name__ == "__main__":
    main()
