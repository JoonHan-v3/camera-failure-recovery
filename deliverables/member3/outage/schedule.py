"""Outage schedule: one CAM_FRONT outage per scene, same start across models/durations.

Timestamps are integer microseconds (nuScenes sensor timestamps) to avoid float drift.
Rule: t_start = timestamp of keyframe index `warmup_kf` (>= clean-history requirement of the
loader; default 6). A scene is eligible only if t_start + max(durations) + followup_s fits
before the last keyframe. Skipped scenes keep their reasons. Schedules persist to JSON.
"""
import json
from dataclasses import dataclass, asdict
from typing import Dict, Sequence

US = 1_000_000
DURATIONS_S = (0.5, 1.0, 2.0)
WARMUP_KF = 6
FOLLOWUP_S = 4.5
CAMERA = "CAM_FRONT"


@dataclass(frozen=True)
class SceneOutage:
    scene_token: str
    t_start_us: int
    warmup_kf: int
    max_duration_us: int
    followup_us: int

    def window(self, duration_s: float):
        """Half-open corruption window [t_start, t_start + d) in microseconds."""
        if duration_s <= 0:
            raise ValueError("duration must be positive")
        return self.t_start_us, self.t_start_us + round(duration_s * US)

    def restoration_us(self, duration_s: float) -> int:
        return self.window(duration_s)[1]


def _check_int_ts(tok, kts):
    for t in kts:
        if isinstance(t, bool) or not isinstance(t, int):
            raise TypeError(f"scene {tok}: timestamps must be integer microseconds, got {t!r}")


def build_schedule(scenes: Dict[str, Sequence[int]], durations_s=DURATIONS_S,
                   warmup_kf=WARMUP_KF, followup_s=FOLLOWUP_S):
    """scenes: scene_token -> sorted keyframe timestamps (us) from nuScenes sample metadata.
    Returns (schedule, skipped) where skipped maps token -> reason."""
    if not durations_s or any(d <= 0 for d in durations_s):
        raise ValueError("durations must be a non-empty set of positive values")
    if not isinstance(warmup_kf, int) or warmup_kf < 0:
        raise ValueError("warmup_kf must be a non-negative int")
    if followup_s < 0:
        raise ValueError("followup_s must be non-negative")
    max_d = round(max(durations_s) * US)
    follow = round(followup_s * US)
    sched, skipped = {}, {}
    for tok in sorted(scenes):
        kts = list(scenes[tok])
        _check_int_ts(tok, kts)
        if kts != sorted(kts) or len(set(kts)) != len(kts):
            raise ValueError(f"scene {tok}: keyframe timestamps not strictly increasing")
        if len(kts) <= warmup_kf:
            skipped[tok] = f"too short: {len(kts)} keyframes <= warmup {warmup_kf}"
            continue
        t_start = kts[warmup_kf]
        if t_start + max_d + follow > kts[-1]:
            skipped[tok] = "insufficient follow-up after longest outage"
            continue
        sched[tok] = SceneOutage(tok, t_start, warmup_kf, max_d, follow)
    return sched, skipped


def save_schedule(path, sched, skipped, durations_s=DURATIONS_S, warmup_kf=WARMUP_KF,
                  followup_s=FOLLOWUP_S, dataset_version=None):
    """Persist the frozen schedule with parameters and skip reasons (shared by all adapters)."""
    with open(path, "w") as f:
        json.dump(dict(params=dict(durations_s=list(durations_s), warmup_kf=warmup_kf,
                                   followup_s=followup_s, camera=CAMERA,
                                   dataset_version=dataset_version),
                       scenes=[asdict(sched[k]) for k in sorted(sched)],
                       skipped=skipped), f, indent=1, sort_keys=True)


def load_schedule(path):
    with open(path) as f:
        d = json.load(f)
    return {s["scene_token"]: SceneOutage(**s) for s in d["scenes"]}, d["skipped"], d["params"]
