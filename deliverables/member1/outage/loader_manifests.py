"""Exact per-loader image manifests for the frozen evaluation (150 val scenes) and pilot (10 train scenes), from the raw
v1.0-trainval JSON only (no model, no GPU).

* BEVFormer-tiny / StreamPETR: the 6 keyframe camera images of every sample (verified in the real runs' audits).
* SparseBEV r50_nuimg_704x256 @ d341052 (pinned, see MODELS.md): a REPLICATION of its test-mode selection logic, copied
  line-for-line in behaviour from gen_sweep_info.add_sweep_info (5 camera sweeps before each keyframe, cam order and the
  'no prev -> repeat last sweep_info' rule), CustomNuScenesDataset.collect_sweeps (into_past=60, walk back over
  timestamp-sorted infos appending each sample's sweeps then the previous keyframe 'cams', stop at a sample with no sweeps)
  and LoadMultiViewImageFromMultiSweeps.load_offline (test_mode, sweeps_num=7, interval 6 -> prev indices 5,11,...,41,
  clamped to len-1, incomplete sweep -> previous entry; no history -> the current keyframe repeated 7x = boundary padding).
  It is NOT an execution of SparseBEV's own code; Member 3 must confirm it against the real loader (compare the filename
  lists of its first pipeline step) once integrated.
Outputs per (split, model): unique files, consumed image slots, bytes, boundary-padding counts, CAM_FRONT files inside each
outage window (the images the outage must black, sweeps included), and the oldest history offset (how far back a corrupted
image can stay in a finite window).
Usage: python -m outage.loader_manifests <out.json>"""
import json, os, sys, collections
from .schedule import load_schedule, US
from .corruption import is_corrupted

H = os.path.expanduser("~")
META = os.path.join(H, "storage/nuscenes")
DL = os.path.join(H, "storage/nuscenes_dl")
P = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMS_SB = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']   # gen_sweep_info order
CAMS_LD = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']   # loader order
SWEEPS_NUM, TEST_INTERVAL, INTO_PAST = 7, 6, 60


class Tables:
    def __init__(self):
        js = lambda n: json.load(open(os.path.join(META, "v1.0-trainval", n + ".json")))
        self.sd = {r["token"]: r for r in js("sample_data")}
        calib = {c["token"]: c for c in js("calibrated_sensor")}
        sensor = {s["token"]: s["channel"] for s in js("sensor")}
        self.chan = {t: sensor[calib[r["calibrated_sensor_token"]]["sensor_token"]] for t, r in self.sd.items()}
        self.sample = {s["token"]: s for s in js("sample")}
        self.scene = {s["token"]: s for s in js("scene")}
        self.kf = {}
        for t, r in self.sd.items():
            if r["is_key_frame"]:
                self.kf[(r["sample_token"], self.chan[t])] = r

    def cam(self, r):
        return dict(path=r["filename"], ts=int(r["timestamp"]))

    def scene_samples(self, tok):
        out, t = [], self.scene[tok]["first_sample_token"]
        while t:
            out.append(t); t = self.sample[t]["next"]
        return out


def sparsebev_infos(T, scenes):
    infos = []
    for sc in scenes:
        for s in T.scene_samples(sc):
            sample = T.sample[s]
            cams = {c: T.cam(T.kf[(s, c)]) for c in CAMS_SB}
            curr = {c: T.kf[(s, c)] for c in CAMS_SB}
            sweeps = []
            if sample["prev"] != "":                      # add_sweep_info
                for _ in range(5):
                    info = {}
                    for c in CAMS_SB:
                        if curr[c]["prev"] == "":
                            info = sweeps[-1]
                            break
                        r = T.sd[curr[c]["prev"]]
                        curr[c] = r
                        info[c] = T.cam(r)
                    sweeps.append(info)
            infos.append(dict(token=s, scene=sc, timestamp=sample["timestamp"], cams=cams, sweeps=sweeps))
    return sorted(infos, key=lambda e: e["timestamp"])   # NuScenesDataset.load_annotations


def sparsebev_consumed(infos, i):
    """Returns ([(cam, path, ts)] in slot order, padded_kind)."""
    prev, k = [], i                                       # collect_sweeps(into_past=60)
    while len(prev) < INTO_PAST:
        cs = infos[k]["sweeps"]
        if len(cs) == 0:
            break
        prev.extend(cs)
        prev.append(infos[k - 1]["cams"])
        k -= 1
    cur = [(c, infos[i]["cams"][c]["path"], infos[i]["cams"][c]["ts"]) for c in CAMS_LD]
    out = list(cur)
    if len(prev) == 0:
        return out + cur * SWEEPS_NUM, "no_history_keyframe_repeated"
    choices = [(j + 1) * TEST_INTERVAL - 1 for j in range(SWEEPS_NUM)]
    clamped = False
    for idx in sorted(choices):
        si = min(idx, len(prev) - 1)
        clamped |= si != idx
        sw = prev[si]
        if len(sw.keys()) < len(CAMS_LD):
            sw = prev[si - 1]
        out += [(c, sw[c]["path"], sw[c]["ts"]) for c in CAMS_LD]
    return out, ("history_clamped_to_oldest" if clamped else "full")


def main(out):
    T = Tables()
    from nuscenes.utils import splits
    name2tok = {v["name"]: k for k, v in T.scene.items()}
    pilot_names = json.load(open(os.path.join(os.path.dirname(P), "StreamPETR/artifacts/trainval_infos/pilot_scenes.json")))
    size = {}
    for b in sorted(os.listdir(DL)):
        if b.startswith("v1.0-trainval") and b.endswith("_blobs"):
            for top in ("samples", "sweeps"):
                base = os.path.join(DL, b, top)
                for ch in (os.listdir(base) if os.path.isdir(base) else []):
                    with os.scandir(os.path.join(base, ch)) as it:
                        for e in it:
                            size.setdefault(f"{top}/{ch}/{e.name}", e.stat().st_size)
    rep = {}
    for split, names, sched_file in (("val", splits.val, "schedule_trainval_val.json"), ("pilot", pilot_names, "schedule_trainval_train.json")):
        scenes = sorted(name2tok[n] for n in names)
        sched, _, _ = load_schedule(os.path.join(P, "artifacts", sched_file))
        kf = [(c, T.kf[(s, c)]["filename"], int(T.kf[(s, c)]["timestamp"]), sc) for sc in scenes for s in T.scene_samples(sc) for c in CAMS_LD]
        n_samples = len(kf) // 6
        r = {}
        kfiles = {x[1] for x in kf}
        r["BEVFormer-tiny | StreamPETR"] = dict(samples=n_samples, consumed_slots=len(kf), unique_files=len(kfiles),
                                                bytes=sum(size[f] for f in kfiles), sweeps_needed=0, all_present=all(f in size for f in kfiles))
        infos = sparsebev_infos(T, scenes)
        slots, kinds, files, front, oldest = 0, collections.Counter(), set(), {d: set() for d in (0.5, 1.0, 2.0)}, 0.0
        for i, inf in enumerate(infos):
            cons, kind = sparsebev_consumed(infos, i)
            kinds[kind] += 1; slots += len(cons)
            so = sched.get(inf["scene"])
            for c, p, ts in cons:
                files.add(p)
                oldest = max(oldest, (inf["timestamp"] - ts) / US)
                if c == "CAM_FRONT" and so is not None:
                    for d in front:
                        if is_corrupted(c, ts, so.t_start_us, d):
                            front[d].add(p)
        sweep_files = {f for f in files if f.startswith("sweeps/")}
        r["SparseBEV r50_nuimg_704x256 (replicated selection)"] = dict(
            samples=len(infos), consumed_slots=slots, unique_files=len(files), unique_sweep_files=len(sweep_files),
            unique_keyframe_files=len(files) - len(sweep_files), bytes=sum(size[f] for f in files),
            sweep_bytes=sum(size[f] for f in sweep_files), all_present=all(f in size for f in files),
            boundary_padding=dict(kinds), oldest_history_s=round(oldest, 3),
            front_files_inside_outage_window={str(d): dict(total=len(v), sweeps=sum(f.startswith("sweeps/") for f in v)) for d, v in front.items()})
        kb = {d: sum(1 for c, p, ts, sc in kf if c == "CAM_FRONT" and sc in sched and is_corrupted(c, ts, sched[sc].t_start_us, d)) for d in (0.5, 1.0, 2.0)}
        r["BEVFormer-tiny | StreamPETR"]["front_files_inside_outage_window"] = {str(d): dict(total=v, sweeps=0) for d, v in kb.items()}
        rep[split] = dict(scenes=len(scenes), scheduled=sum(1 for s in scenes if s in sched), models=r,
                          union_unique_files=len(kfiles | files), union_bytes=sum(size[f] for f in kfiles | files))
    rep["note"] = __doc__
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
