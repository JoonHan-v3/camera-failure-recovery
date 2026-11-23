"""Independent verification of Member 3's per-scene tables: rebuild every (scene, duration, phase) field (n_samples, n_gt, tp, fp, ign, n_pred_region,
clean and outage, plus the `present` flag) from the per-sample statuses with this module's own phase assignment and counting, and compare to the CSV.
Usage (from RecoveryAnalysis/): python -m recovery.verify_tables artifacts/analysis/tables_verification.json"""
import json, os, sys, time
import numpy as np
from .data import MODELS, DURATIONS, PHASES, FIELDS, load_all, sha256
from . import samplelevel as sl


def main(out):
    ms = load_all()
    gt, sched = sl.load_gt(), sl.load_schedule(os.path.join(sl.S3, "artifacts/schedule_trainval_val.json"))
    rep = dict(models={}, generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    allok = True
    for m in MODELS:
        b = sl.build_model(m, gt, sched)
        assert b["scenes"] == ms[m]["scenes"]
        mism = 0; cells = 0
        for d in DURATIONS:
            ref, mine = ms[m]["by_duration"][d], b["by_duration"][d]["table"]
            for f in FIELDS:
                mism += int((ref[f] != mine[f]).sum()); cells += ref[f].size
            mism += int((ref["present"] != b["by_duration"][d]["present"]).sum()); cells += ref["present"].size
        rep["models"][m] = dict(cells_compared=cells, mismatching_cells=mism, tau=b["tau"],
                                csv_sha256=sha256(ms[m]["path"]), per_sample_pickle_sha256=sha256(os.path.join(sl.STATS, f"per_sample_{MODELS[m]['tag']}.pkl")))
        allok &= mism == 0
        print(m, "cells", cells, "mismatches", mism)
    rep["ALL_MATCH"] = bool(allok)
    json.dump(rep, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
