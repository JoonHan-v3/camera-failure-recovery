"""Shared timestamp-based CAM_FRONT corruption (applies to keyframes AND sweeps)."""
import numpy as np
from .schedule import CAMERA, US


def is_corrupted(cam: str, ts_us: int, t_start_us: int, duration_s: float,
                 target_cam: str = CAMERA) -> bool:
    """True iff cam is the target camera and ts in [t_start, t_start + d). d<=0 means clean."""
    if cam != target_cam or duration_s <= 0:
        return False
    return t_start_us <= ts_us < t_start_us + round(duration_s * US)


def black_like(img: np.ndarray) -> np.ndarray:
    """All-black image (raw pixel value 0), same shape/dtype, applied BEFORE resize/normalize."""
    return np.zeros_like(img)


def maybe_corrupt(img, cam, ts_us, t_start_us, duration_s):
    """Returns (image, corrupted_flag). Never mutates the input."""
    if is_corrupted(cam, ts_us, t_start_us, duration_s):
        return black_like(img), True
    return img, False
