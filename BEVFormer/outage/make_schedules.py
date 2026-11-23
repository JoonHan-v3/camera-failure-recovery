"""Freeze the shared physical outage schedules from raw nuScenes tables (no model, no images needed).
Usage: python -m outage.make_schedules [out_dir=artifacts]
Writes schedule_mini.json, schedule_trainval_val.json (150 val scenes), schedule_trainval_train.json (700 train scenes).
Rule: t_start = CAM_FRONT sensor timestamp (us) of keyframe index 6; scene kept iff t_start + 2.0 s + 4.5 s <= last front keyframe."""
import os, sys
from nuscenes.utils import splits
from .nusc_ref import NuscRef
from .schedule import build_schedule, save_schedule

ROOT = os.path.expanduser("~/storage/nuscenes")      # holds v1.0-mini and v1.0-trainval JSON tables


def main(out="artifacts"):
    os.makedirs(out, exist_ok=True)
    r = NuscRef(ROOT, "v1.0-trainval")
    name2tok = {v["name"]: k for k, v in r.scenes.items()}
    for split, names in (("val", splits.val), ("train", splits.train)):
        toks = [name2tok[n] for n in names]
        sched, skip = build_schedule(r.scene_ts_table(toks))
        save_schedule(os.path.join(out, f"schedule_trainval_{split}.json"), sched, skip, dataset_version="v1.0-trainval/" + split)
        print(split, len(toks), "scheduled", len(sched), "skipped", len(skip))
    m = NuscRef(ROOT, "v1.0-mini")
    s, k = build_schedule(m.scene_ts_table())
    save_schedule(os.path.join(out, "schedule_mini.json"), s, k, dataset_version="v1.0-mini")
    print("mini", len(s), k)


if __name__ == "__main__":
    main(*sys.argv[1:2])
