"""Inputs of the recovery analysis: Member 3's per-scene / per-phase statistics (CSV) for the three models, loaded into aligned arrays.
Everything here is READ-ONLY; every input file is hashed into artifacts/inputs.json by run_summary.py."""
import csv, hashlib, json, os
import numpy as np

PROJECTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S3 = os.path.join(PROJECTS, "SparseBEV")
STATS = os.path.join(S3, "artifacts/event_stats")
MODELS = {"sparsebev": dict(label="SparseBEV", tag="sparsebev_val", color="#1b9e77"),
          "streampetr": dict(label="StreamPETR", tag="streampetr_val", color="#d95f02"),
          "bevformer": dict(label="BEVFormer", tag="bevformer_val", color="#7570b3")}
DURATIONS = (0.5, 1.0, 2.0)
BINS = tuple(i * 0.5 for i in range(10))                                  # post-restoration bins, seconds
POST = [f"b{b:.1f}" for b in BINS]
PHASES = ["pre", "during"] + POST + ["other"]
FIELDS = ("n_samples", "n_gt", "tp", "fp", "ign", "n_pred_region", "tp_o", "fp_o", "ign_o", "n_pred_region_o")
# FROZEN event rules (SparseBEV/artifacts/provenance/EVENT_RULES_DECLARATION.txt): declared before any validation event statistic
LOSS_MARGIN = 0.05
MIN_GT = 5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_model(model, stats_dir=STATS):
    """-> dict(scenes=[...], by_duration={d: {field: array (S, len(PHASES))}, 'present': bool array}). Scene order = sorted scene tokens."""
    path = os.path.join(stats_dir, f"per_scene_bins_{MODELS[model]['tag']}.csv")
    rows = list(csv.DictReader(open(path)))
    scenes = sorted({r["scene"] for r in rows})
    si = {s: i for i, s in enumerate(scenes)}
    out = {}
    for d in DURATIONS:
        arr = {f: np.zeros((len(scenes), len(PHASES)), np.int64) for f in FIELDS}
        arr["present"] = np.zeros((len(scenes), len(PHASES)), bool)
        seen = 0
        for r in rows:
            if float(r["duration_s"]) != d:
                continue
            i, j = si[r["scene"]], PHASES.index(r["phase"])
            for f in FIELDS:
                arr[f][i, j] = int(r[f])
            arr["present"][i, j] = r["present"] == "True"
            seen += 1
        assert seen == len(scenes) * len(PHASES), (model, d, seen)
        out[d] = arr
    return dict(scenes=scenes, by_duration=out, path=path)


def load_all():
    ms = {m: load_model(m) for m in MODELS}
    ref = ms["sparsebev"]["scenes"]
    assert all(v["scenes"] == ref for v in ms.values()), "models must share the scene list"
    # the eligible-GT denominators are model-independent (GT + geometry only): verify, since pairing across models relies on it
    for d in DURATIONS:
        for m in ("streampetr", "bevformer"):
            assert np.array_equal(ms[m]["by_duration"][d]["n_gt"], ms["sparsebev"]["by_duration"][d]["n_gt"]), (m, d)
    return ms
