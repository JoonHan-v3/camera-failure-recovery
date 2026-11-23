"""Independent nuScenes reference: reads the raw JSON tables (NOT the StreamPETR info pkl) to produce
per-scene keyframe timestamps, the expected consumed-image manifest and sample->eval-time metadata.

StreamPETR consumes, per evaluated sample, one keyframe image from each of the six cameras in the
converter's order (CAMS below, = slot index) and no image sweeps. The outage start is defined on the
CAM_FRONT *sensor* timestamp (cameras are asynchronous with the lidar sample timestamp)."""
import json, os
from .schedule import CAMERA

CAMS = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]


def _load(root, version, name):
    with open(os.path.join(root, version, name + ".json")) as f:
        return json.load(f)


class NuscRef:
    def __init__(self, root, version):
        self.version = version
        sample = {s["token"]: s for s in _load(root, version, "sample")}
        scene = _load(root, version, "scene")
        calib = {c["token"]: c for c in _load(root, version, "calibrated_sensor")}
        sensor = {s["token"]: s["channel"] for s in _load(root, version, "sensor")}
        kf = {}   # (sample_token, channel) -> sample_data (keyframes only)
        for sd in _load(root, version, "sample_data"):
            if sd["is_key_frame"]:
                kf[(sd["sample_token"], sensor[calib[sd["calibrated_sensor_token"]]["sensor_token"]])] = sd
        self.scenes = {}   # scene_token -> dict(name, samples=[sample tokens in order])
        for sc in scene:
            toks, t = [], sc["first_sample_token"]
            while t:
                toks.append(t); t = sample[t]["next"]
            self.scenes[sc["token"]] = dict(name=sc["name"], samples=toks)
        self.sample, self.kf = sample, kf

    def lidar_ts(self, sample_token):
        """Evaluation timestamp the model uses (LIDAR_TOP keyframe sample_data timestamp)."""
        return int(self.kf[(sample_token, "LIDAR_TOP")]["timestamp"])

    def front_keyframe_ts(self, scene_token):
        return [int(self.kf[(s, CAMERA)]["timestamp"]) for s in self.scenes[scene_token]["samples"]]

    def scene_ts_table(self, scene_tokens=None):
        return {t: self.front_keyframe_ts(t) for t in (scene_tokens or self.scenes)}

    def manifest(self, scene_tokens):
        """Expected consumed-image keys (scene, sample, cam, ts_us, slot) + sample metadata."""
        keys, meta = set(), {}
        for tok in scene_tokens:
            for s in self.scenes[tok]["samples"]:
                meta[(tok, s)] = self.lidar_ts(s)
                for slot, cam in enumerate(CAMS):
                    keys.add((tok, s, cam, int(self.kf[(s, cam)]["timestamp"]), slot))
        return keys, meta

    def filenames(self, scene_tokens):
        return {(tok, s, cam): self.kf[(s, cam)]["filename"] for tok in scene_tokens
                for s in self.scenes[tok]["samples"] for cam in CAMS}
