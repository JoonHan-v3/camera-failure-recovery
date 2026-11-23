"""Glue for the BEVFormer (mmdet3d 0.17.1) test pipeline: shared CorruptFrontCamera step (same code as Member 2's).

VERIFIED on BEVFormer 66b65f3 / mmdet3d v0.17.1: LoadMultiViewImageFromFiles yields results['img'] = list of six HxWx3
float32 BGR arrays (RAW pixels, 0..255, to_float32=True) in converter camera order (CAM_FRONT first), results['img_filename']
the six absolute keyframe paths. The upstream dataset does not forward camera timestamps; outage/mm_pipeline.py adds
img_timestamp (integer sensor microseconds from info['cams'][cam]['timestamp'], i.e. nuScenes sample_data.timestamp) and
eval_ts_us. Hence ts_mode='abs_us' (exact integers). BEVFormer consumes keyframes only (no camera sweeps / temporal queue
images in test mode: history is the recurrent prev_bev). The corruption is applied right after loading and BEFORE
normalization (NormalizeMultiviewImage), scaling (x0.5) and the backbone, so black = raw pixel 0 -> normalized black.
Expected input keys are validated (mismatches raise ValueError).
"""
import re
import numpy as np
from .corruption import maybe_corrupt

_CAM_RE = re.compile(r"(CAM_[A-Z_]+)")


def cam_from_path(p):
    m = _CAM_RE.search(str(p))
    if not m:
        raise ValueError(f"cannot infer camera from {p}")
    return m.group(1)


class CorruptFrontCamera:
    def __init__(self, schedule, duration_s, audit=None, guard=None, ts_mode="rel_s"):
        if ts_mode not in ("rel_s", "abs_us", "abs_s"):
            raise ValueError("ts_mode must be 'rel_s', 'abs_us' or 'abs_s'")
        if duration_s < 0:
            raise ValueError("duration must be >= 0 (0 = clean run: log only, never corrupt)")
        self.schedule, self.duration_s, self.audit, self.guard = schedule, duration_s, audit, guard
        self.ts_mode = ts_mode

    def _abs_us(self, results, t):
        if self.ts_mode == "abs_us":
            if int(t) != t:
                raise ValueError(f"abs_us timestamp not integral: {t}")
            return int(t)
        if self.ts_mode == "abs_s":
            return round(t * 1e6)
        return round((results["timestamp"] + t) * 1e6)

    def __call__(self, results):
        for k in ("img", "img_filename", "img_timestamp", "scene_token"):
            if k not in results:
                raise ValueError(f"missing key {k}")
        imgs, fns, tss = results["img"], results["img_filename"], results["img_timestamp"]
        n = len(imgs)
        if n == 0 or len(fns) != n or len(tss) != n:
            raise ValueError(f"inconsistent input lengths: img={n} filename={len(fns)} timestamp={len(tss)}")
        kf = results.get("img_is_keyframe")
        if kf is not None and len(kf) != n:
            raise ValueError("img_is_keyframe length mismatch")
        for im in imgs:
            if not isinstance(im, np.ndarray) or im.ndim != 3:
                raise ValueError("each image must be an HxWxC ndarray (raw decoded pixels)")
        scene = results["scene_token"]
        out = self.schedule.get(scene)
        if self.guard:  # ordering-only: model memory is observed in the inference process
            self.guard.step(scene, round(results["timestamp"] * 1e6), require_memory=False)
        eval_ts = results.get("eval_ts_us", round(results["timestamp"] * 1e6))
        new = []
        for i, (img, fn, t) in enumerate(zip(imgs, fns, tss)):
            cam, ts = cam_from_path(fn), self._abs_us(results, t)
            if out is not None:
                img2, flag = maybe_corrupt(img, cam, ts, out.t_start_us, self.duration_s)
            else:
                img2, flag = img, False
            if self.audit:
                self.audit.log(scene, results.get("sample_idx"), cam, ts,
                               None if kf is None else kf[i], fn, img, img2, flag, slot=i, eval_ts_us=eval_ts)
            new.append(img2)
        results["img"] = new
        return results
