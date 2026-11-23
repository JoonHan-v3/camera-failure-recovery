"""Batch-one resource profile of one released detector under the STUDY configuration (study plan, 'Resource use': measure batch-one latency
with warm-up / CUDA synchronization and peak allocated GPU memory; record end-to-end runtime separately).

Each model is profiled in ITS OWN conda env with ITS OWN study Runner (same dataset pipeline, same numerics, same wrappers as the study runs):
  sparsebev  : SparseBEV/outage/run_sparsebev.Runner          (env sparsebev)
  bevformer  : BEVFormer/outage/run_bevformer.Runner          (env bevformer)
  streampetr : StreamPETR/outage/run_streampetr.Runner        (env streampetr_fa)
Protocol (clean condition, two pilot scenes = 80 frames, scenes in timestamp order, batch size 1, TF32 off):
  1. all frames are decoded/pre-processed ONCE up front into PRISTINE templates (data_loading_s = per-frame CPU time of dataset[i]: JPEG decode + pipeline);
     the templates are never passed to a forward: every forward (warm-up included) gets a FRESH deep copy made OUTSIDE the timed region, because upstream
     forwards edit their metadata in place (BEVFormer rewrites img_metas[0][0]['can_bus'] from absolute pose to deltas; correction 2);
  2. model state is reset and one full pass is run and DISCARDED (CUDA/cudnn lazy init, allocator warm-up);
  3. `repeats` measured passes; per frame: fresh copy (untimed); torch.cuda.synchronize(); t0; forward (host->device copy of the inputs, detector, decode,
     result to CPU is NOT included); torch.cuda.synchronize(); latency = t1 - t0;  state is reset before every pass (streaming state restarts);
     an input-identity digest of every fresh copy (must equal the pristine template's), whether the forward mutated its copy, and an output digest / max
     difference of the detections across passes are recorded (`input_output_checks` in the JSON);
  4. peak allocated / reserved memory is reset after the warm-up pass and read after the measured passes.
`first_frame_of_scene` latencies (no history yet) are reported separately from steady-state frames.
Usage (cwd/env as the project's run_matrix.sh): python <this file> <model> <out.json> [repeats]"""
import copy, hashlib, json, os, statistics, subprocess, sys, time
import numpy as np

ROOT = os.path.expanduser("~/Documents/UM/rob535/projects")
SCENES = "129304d659464b0fb3a51c99eaec114a,73030fb67d3c46cfb5e590168088ae39".split(",")      # first two pilot scenes in the probe list
PROJ = dict(sparsebev=("SparseBEV", "sparsebev_src", "artifacts/infos/pilot_infos.pkl"),
            bevformer=("BEVFormer", "bevformer_src", "artifacts/infos/pilot_infos.pkl"),
            streampetr=("StreamPETR", "streampetr_src", "artifacts/trainval_infos/pilot_train_infos.pkl"))


def q(x, p):
    return float(np.percentile(np.asarray(x, float), p))


def digest(o, h=None):
    """SHA-1 over every array / tensor / scalar / string reachable from o (DataContainer, dict, list, tuple), in a deterministic order."""
    import torch
    top = h is None
    h = h or hashlib.sha1()
    if hasattr(o, "data") and o.__class__.__name__ == "DataContainer":
        digest(o.data, h)
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            h.update(str(k).encode()); digest(o[k], h)
    elif isinstance(o, (list, tuple)):
        for v in o:
            digest(v, h)
    elif isinstance(o, torch.Tensor):
        h.update(str(tuple(o.shape)).encode() + str(o.dtype).encode()); h.update(o.detach().cpu().contiguous().numpy().tobytes())
    elif isinstance(o, np.ndarray):
        h.update(str(o.shape).encode() + str(o.dtype).encode()); h.update(np.ascontiguousarray(o).tobytes())
    else:
        h.update(repr(o).encode())
    return h.hexdigest() if top else None


def out_arrays(res):
    """-> list of float arrays (boxes, scores, labels) of the detections returned by one forward, for the repeatability comparison."""
    import torch
    r = res[0] if isinstance(res, (list, tuple)) else res
    r = r.get("pts_bbox", r)
    arrs = []
    for k in ("boxes_3d", "scores_3d", "labels_3d"):
        v = r[k]; v = getattr(v, "tensor", v)
        arrs.append(v.detach().cpu().double().numpy() if isinstance(v, torch.Tensor) else np.asarray(v, float))
    return arrs


def main(model, out, repeats=3):
    import torch, mmcv
    proj, src, infos_rel = PROJ[model]
    P = os.path.join(ROOT, proj)
    sys.path[:0] = [os.path.join(P, src), P]
    infos = os.path.join(P, infos_rel)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if model == "sparsebev":
        from outage import compat; compat.install()
        from outage.numerics import set_study_numerics; set_study_numerics()
        from outage.run_sparsebev import Runner
        from outage.adapter import OutageFrontCamera, CorruptFrontCamera
        r = Runner(infos, SCENES); ts_mode = "abs_s"
        reset = r.reset
        def ctx_reset(): r.obs_log = []
    elif model == "bevformer":
        from outage.run_bevformer import Runner
        from outage.mm_pipeline import OutageFrontCamera
        from outage.adapter import CorruptFrontCamera
        r = Runner(infos, SCENES, "tiny"); ts_mode = "abs_us"
        reset = r.reset
        def ctx_reset(): r.fresh_log = []
    else:
        from outage.run_streampetr import Runner
        from outage.mm_pipeline import OutageFrontCamera
        from outage.streampetr_adapter import CorruptFrontCamera
        r = Runner("v1.0-trainval", infos, SCENES); ts_mode = "abs_s"
        reset = r.reset
        def ctx_reset(): r.fresh_log = []
    from outage.state import SceneStateGuard
    OutageFrontCamera.STEP = CorruptFrontCamera({}, 0.0, None, guard=SceneStateGuard(), ts_mode=ts_mode)
    r.model_guard = None
    ds = r.dataset
    n = len(ds.data_infos)
    # 1. decode + pre-process every frame once (also the end-to-end CPU cost)
    data, load_s = [], []
    for i in range(n):
        t0 = time.perf_counter(); d = mmcv.parallel.collate([ds[i]], 1); load_s.append(time.perf_counter() - t0); data.append(d)
    pristine = [digest(d) for d in data]                 # identity of the templates; checked again after all passes
    chk = dict(fresh_copy_equals_pristine=True, forward_mutated_its_input_copy_frames=0, n_forwards=0, pass_outputs=[])
    scene_of = [ds.data_infos[i]["scene_token"] for i in range(n)]
    first = [i == 0 or scene_of[i] != scene_of[i - 1] for i in range(n)]

    def one_pass():
        reset(); ctx_reset(); lat, outs = [], []
        with torch.no_grad():
            for i in range(n):
                r.ctx = (ds.data_infos[i]["scene_token"], int(ds.data_infos[i]["timestamp"]))
                fresh = copy.deepcopy(data[i])            # fresh mutable metadata for EVERY forward (untimed)
                d0 = digest(fresh)
                chk["fresh_copy_equals_pristine"] &= (d0 == pristine[i])
                torch.cuda.synchronize(); t0 = time.perf_counter()
                res = r.dp(return_loss=False, rescale=True, **fresh)
                torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
                chk["n_forwards"] += 1; chk["forward_mutated_its_input_copy_frames"] += int(digest(fresh) != d0)
                outs.append(out_arrays(res))
        chk["pass_outputs"].append(outs)
        return lat
    one_pass()                                            # 2. warm-up pass (discarded)
    torch.cuda.reset_peak_memory_stats()
    passes = [one_pass() for _ in range(repeats)]        # 3. measured passes
    outs = chk.pop("pass_outputs")                        # outs[0] = warm-up, outs[1:] = measured passes
    def maxdiff(a, b):
        return max((float(np.abs(x - y).max()) if x.shape == y.shape and x.size else (0.0 if x.shape == y.shape else float("inf"))) for fa, fb in zip(a, b) for x, y in zip(fa, fb))
    chk["output_max_abs_diff_warmup_vs_measured"] = [maxdiff(outs[0], o) for o in outs[1:]]
    chk["output_max_abs_diff_between_measured_passes"] = [maxdiff(outs[1], o) for o in outs[2:]]
    chk["pristine_templates_unchanged_after_all_passes"] = all(digest(d) == p for d, p in zip(data, pristine))
    chk["interpretation"] = ("forward_mutated_its_input_copy_frames > 0 means the model's forward edits its inputs in place, which is why every forward gets a fresh copy; "
                             "output differences are bitwise repeatability of detections across passes (0 = identical)")
    lat = np.array(passes)                                # (repeats, n)
    steady = lat[:, [not f for f in first]].ravel(); firstf = lat[:, [bool(f) for f in first]].ravel()
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,clocks.sm,clocks.max.sm,power.draw", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip().splitlines()
    nparam = sum(p.numel() for p in r.det.parameters())
    rep = dict(model=model, n_frames=n, scenes=SCENES, repeats=repeats, batch_size=1, condition="clean", tf32=False,
               latency_ms=dict(steady_median=1e3 * float(np.median(steady)), steady_mean=1e3 * float(steady.mean()), steady_p95=1e3 * q(steady, 95),
                               steady_p99=1e3 * q(steady, 99), steady_min=1e3 * float(steady.min()), steady_max=1e3 * float(steady.max()),
                               first_frame_of_scene_median=1e3 * float(np.median(firstf)), per_pass_median=[1e3 * float(np.median(p)) for p in lat]),
               fps_steady_median=1.0 / float(np.median(steady)),
               data_loading_cpu_ms=dict(median=1e3 * float(np.median(load_s)), mean=1e3 * float(np.mean(load_s)), p95=1e3 * q(load_s, 95)),
               end_to_end_ms_paired_median=1e3 * float(np.median(np.array([[l + load_s[i] for i, l in enumerate(p)] for p in lat.tolist()]))),
               end_to_end_ms_estimate_median_forward_plus_median_load=1e3 * (float(np.median(steady)) + float(np.median(load_s))),
               input_output_checks=chk,
               peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20, peak_reserved_mb=torch.cuda.max_memory_reserved() / 2**20,
               n_parameters=int(nparam), gpu_smi=smi, gpu_compute_apps_during_profile=apps, torch=torch.__version__, cuda_runtime=torch.version.cuda,
               fp16_wrapped=bool(getattr(r.det, "fp16_enabled", False)) or any(getattr(m, "fp16_enabled", False) for m in r.det.modules()),
               protocol="warm-up pass discarded; synchronized per-frame forward incl. host->device copy of FRESH copies of pre-built inputs (copy untimed); CPU decode excluded (reported separately); "
                        "end_to_end_ms_paired_median = median over passes and frames of (that frame's CPU decode time + that frame's forward time); the *_estimate_* field is median(forward)+median(load)",
               timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("model", "n_frames", "latency_ms", "fps_steady_median", "data_loading_cpu_ms", "peak_allocated_mb", "peak_reserved_mb", "n_parameters")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
