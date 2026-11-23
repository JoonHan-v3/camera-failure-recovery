"""Cross-model physical-outage check: for every split/condition the three models (BEVFormer, StreamPETR, SparseBEV) must have consumed the
IDENTICAL set of (scene, sample, camera, sensor-ts) images and blacked the IDENTICAL set (one physical outage schedule), from their saved audits.
Usage (from SparseBEV/): python -m outage.cross_model_check artifacts/cross_model_outage_check.json"""
import json, os, sys
from .audit import load_audit

P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(P)
MODELS = {"sparsebev": os.path.join(P, "artifacts/runs/{s}_{c}"), "bevformer": os.path.join(ROOT, "BEVFormer/artifacts/runs/{s}_{c}"),
          "streampetr_pilot": os.path.join(ROOT, "StreamPETR/artifacts/runs/rf_{s}_{c}"), "streampetr_val": os.path.join(ROOT, "StreamPETR/artifacts/runs/rf_{s}_{c}")}


def sets(path):
    a = load_audit(os.path.join(path, "audit.json"))
    consumed = {(r["scene"], r["sample"], r["cam"], r["ts_us"]) for r in a.records}
    black = {(r["scene"], r["sample"], r["cam"], r["ts_us"]) for r in a.records if r["corrupted"]}
    return consumed, black, len(a.records)


def main(out):
    rep, ok = {}, True
    for split in ("pilot", "val"):
        for c in ("a", "d05", "d10", "d20"):
            res = {}
            for m in ("sparsebev", "bevformer", "streampetr_" + split):
                p = MODELS[m].format(s=split, c=c)
                if m == "bevformer":      # BEVFormer's val_a is named val_a (same pattern)
                    p = MODELS[m].format(s=split, c=c)
                res[m] = sets(p)
            ref = res["sparsebev"]
            for m, v in res.items():
                same = (v[0] == ref[0], v[1] == ref[1])
                ok &= all(same)
                rep.setdefault(f"{split}_{c}", {})[m] = dict(n_records=v[2], n_unique_images=len(v[0]), n_black=len(v[1]),
                                                              consumed_set_equals_sparsebev=same[0], black_set_equals_sparsebev=same[1])
    rep["ALL_IDENTICAL"] = bool(ok)
    json.dump(rep, open(out, "w"), indent=1); print("ALL_IDENTICAL", ok)
    for k, v in rep.items():
        if k != "ALL_IDENTICAL":
            print(k, {m: (x["n_unique_images"], x["n_black"]) for m, x in v.items()})


if __name__ == "__main__":
    main(sys.argv[1])
