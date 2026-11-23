"""Scene-order / state-reset checks for streaming (StreamPETR-style) inference."""
import numpy as np


class StateResetError(AssertionError):
    pass


class SceneStateGuard:
    """Feed frames in order via .step() BEFORE the model processes the frame.
    memory_fresh: True/False observed from the real model (e.g. 'no valid entries / reset flag
    set'); fixed-size allocated memory is fine as long as it is reported fresh. None = unknown
    and is rejected unless require_memory=False (ordering-only use from the data pipeline)."""

    def __init__(self):
        self.prev_scene, self.prev_ts, self.seen, self.log = None, None, set(), []

    def step(self, scene_token, ts_us, memory_fresh=None, require_memory=True):
        if memory_fresh is not None:
            if not isinstance(memory_fresh, (bool, np.bool_)):
                raise TypeError(f"memory_fresh must be bool/np.bool_ or None, got {type(memory_fresh)}")
            memory_fresh = bool(memory_fresh)
        new_scene = scene_token != self.prev_scene
        if new_scene:
            if scene_token in self.seen:
                raise StateResetError(f"scene {scene_token} revisited: interleaved batching/workers")
            if memory_fresh is None and require_memory:
                raise StateResetError(f"memory state unobserved at start of {scene_token}")
            if memory_fresh is False:
                raise StateResetError(f"stale memory at start of {scene_token}")
            self.seen.add(scene_token)  # committed only after validation
        elif self.prev_ts is not None and ts_us <= self.prev_ts:
            raise StateResetError(f"non-increasing timestamp in {scene_token}: {ts_us} <= {self.prev_ts}")
        self.log.append((scene_token, ts_us, new_scene, memory_fresh))
        self.prev_scene, self.prev_ts = scene_token, ts_us
        return new_scene


def _canon(rows):
    """Canonical order: descending score (last column), ties broken by remaining columns."""
    x = np.asarray(rows, float)
    if x.size == 0:
        return x.reshape(0, x.shape[-1] if x.ndim == 2 else 0)
    order = np.lexsort(tuple(x[:, i] for i in range(x.shape[1] - 1)) + (-x[:, -1],))
    return x[order]


def predictions_equal(a, b, atol=1e-5, rtol=0.0):
    """Compare {sample_token: rows(..., score)}: same samples, shapes, values within atol/rtol
    (rtol explicit, default absolute-only). Rows are canonically ordered first."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = _canon(a[k]), _canon(b[k])
        if x.shape != y.shape or not np.allclose(x, y, atol=atol, rtol=rtol):
            return False
    return True


def reset_between_runs(reset_fn, observe_fresh_fn):
    """Explicitly reset model state (and caches) and verify the model reports it fresh."""
    reset_fn()
    f = observe_fresh_fn()
    if not isinstance(f, (bool, np.bool_)) or not f:
        raise StateResetError("model did not report fresh state after reset")


def check_scene_isolation(run_scene, reset_fn, observe_fresh_fn, scene_a, scene_b, atol=1e-6):
    """A->B isolation: predictions for B run fresh must equal B run right after A (with the
    model's own scene-boundary reset, no manual reset between A and B). run_scene(scene)->preds.
    Also verifies repeat of the fresh run."""
    reset_between_runs(reset_fn, observe_fresh_fn)
    fresh = run_scene(scene_b)
    reset_between_runs(reset_fn, observe_fresh_fn)
    repeat = run_scene(scene_b)
    reset_between_runs(reset_fn, observe_fresh_fn)
    run_scene(scene_a)
    after_a = run_scene(scene_b)
    return dict(repeatable=predictions_equal(fresh, repeat, atol),
                isolated=predictions_equal(fresh, after_a, atol))
