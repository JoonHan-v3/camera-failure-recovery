"""Input audit: log every image a model consumes; reconcile against a manifest and a clean run.

Each record carries sha1_src (raw decoded pixels BEFORE corruption) and sha1 (pixels handed on),
so unflagged modification is detectable independently of the self-reported flag.
Key of a consumption = (scene, sample, cam, ts_us, slot); `slot` is the position within the
frame's image list, which preserves repeated consumption of the same file (history reuse).
"""
import hashlib, json
import numpy as np
from .corruption import is_corrupted
from .schedule import CAMERA, US, SceneOutage


# FROZEN causal-ALIGNMENT policy (identified before validation acceptance; do not raise to silence violations).
# Two times are kept separate: (1) evaluation/annotation time = the sample (lidar keyframe) time; predictions are scored
# against sample-time annotations. (2) input-bundle availability = at least max(sample time, latest consumed sensor
# exposure time); transport/processing delay is NOT measured and is disclosed as unmeasured. nuScenes camera triggers are
# not synchronous with the lidar sample time: over ALL 34,149 v1.0-trainval keyframes (204,894 camera images)
# sensor-minus-sample time is in [-117,877, +49,992] us (max positive: CAM_BACK_LEFT). The frozen 50 ms bound is an
# ALIGNMENT ALLOWANCE (an image assigned to a sample may be exposed up to 50 ms after the sample time, i.e. the model
# is given a fixed sample+50 ms inference budget); it does not claim a later-exposed image existed at lidar time. An
# image exposed later than that is a future-image violation. Pixel assignment of clean/outage runs is unchanged.
CAUSAL_TOL_US = 50_000
CAUSAL_POLICY = ("alignment allowance: image sensor ts <= sample (lidar keyframe) ts + 50000 us (fixed sample+50ms inference "
                 "budget; bundle availability >= max(sample time, latest exposure), delay unmeasured); eval time = sample time; "
                 "observed trainval max skew +49992 us")


def _sha(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def key(r):
    return (r["scene"], r["sample"], r["cam"], r["ts_us"], r["slot"])


class InputAudit:
    def __init__(self, model, condition, duration_s, schedule, run_id="", causal_tolerance_us=None):
        if condition == "clean":
            duration_s = 0.0
        elif duration_s <= 0:
            raise ValueError("outage condition needs a positive duration")
        self.model, self.condition, self.duration_s = model, condition, duration_s
        self.schedule, self.run_id = schedule, run_id
        self.records = []
        # Causal policy: image ts must be <= eval_ts + tolerance. None = policy not yet verified
        # against the pinned loader (status NOT VERIFIED for causality, no generic rule imposed).
        self.causal_tolerance_us = causal_tolerance_us
        self.manifest, self.manifest_id = None, None
        self.sample_meta = None  # independent {(scene, sample): eval_ts_us} from dataset metadata

    def set_manifest(self, keys, identity):
        """Independently constructed expected consumed-image keys + an identity string."""
        self.manifest, self.manifest_id = set(keys), str(identity)

    def set_sample_metadata(self, mapping):
        """Independent sample identity -> evaluation timestamp map (e.g. nuScenes sample table)."""
        self.sample_meta = {(str(k[0]), str(k[1])): int(v) for k, v in dict(mapping).items()}

    def eval_conflicts(self):
        """Diagnostics for evaluation identity/timestamp problems (never silently overwritten):
        missing sample identity, >1 eval ts per (scene, sample), >1 sample per (scene, eval ts),
        and disagreement with independent sample metadata."""
        errs, ts_of, samples_at = [], {}, {}
        for r in self.records:
            ev = r["eval_ts_us"]
            if ev is None:
                continue
            if r["sample"] in (None, ""):
                errs.append(f"missing sample identity {key(r)}")
                continue
            sk = (r["scene"], r["sample"])
            if ts_of.setdefault(sk, ev) != ev:
                errs.append(f"conflicting eval timestamps for {sk}: {ts_of[sk]} vs {ev}")
            samples_at.setdefault((r["scene"], ev), set()).add(r["sample"])
        errs += [f"multiple samples {sorted(v)} at eval ts {k[1]} in scene {k[0]}"
                 for k, v in sorted(samples_at.items(), key=str) if len(v) > 1]
        if self.sample_meta is not None:
            for sk, ev in sorted(ts_of.items(), key=str):
                if sk not in self.sample_meta:
                    errs.append(f"sample not in independent metadata {sk}")
                elif self.sample_meta[sk] != ev:
                    errs.append(f"eval ts {ev} != metadata {self.sample_meta[sk]} for {sk}")
        return errs

    def log(self, scene_token, sample_token, cam, ts_us, is_keyframe, path, img_before, img_after,
            corrupted_flag, slot=0, eval_ts_us=None):
        self.records.append(dict(eval_ts_us=None if eval_ts_us is None else int(eval_ts_us),
            scene=scene_token, sample=sample_token, cam=cam, ts_us=int(ts_us), slot=int(slot),
            keyframe=None if is_keyframe is None else bool(is_keyframe), path=str(path),
            corrupted=bool(corrupted_flag), all_black=bool(not np.any(img_after)),
            sha1_src=_sha(img_before), sha1=_sha(img_after)))

    def verify(self, expected_manifest=None, require_scenes=None):
        """Return violations (empty == pass). Fails on empty audit, duplicates, missing/extra
        records vs expected_manifest (set of keys), and unobserved scheduled scenes."""
        errs = []
        if expected_manifest is None:
            expected_manifest = self.manifest
        if not self.records:
            return ["empty audit: no consumed images recorded"]
        seen = set()
        for r in self.records:
            k = key(r)
            if k in seen:
                errs.append(f"duplicate record {k}")
            seen.add(k)
        if expected_manifest is not None:
            em = set(expected_manifest)
            errs += [f"missing record {k}" for k in sorted(em - seen, key=str)]
            errs += [f"extra record {k}" for k in sorted(seen - em, key=str)]
        for tok in (require_scenes if require_scenes is not None else self.schedule):
            if not any(r["scene"] == tok for r in self.records):
                errs.append(f"scheduled scene never observed: {tok}")
        errs += self.eval_conflicts()
        for r in self.records:
            if self.causal_tolerance_us is not None:
                if r["eval_ts_us"] is None:
                    errs.append(f"causality check needs eval_ts_us {key(r)}")
                elif r["ts_us"] > r["eval_ts_us"] + self.causal_tolerance_us:
                    errs.append(f"future image consumed {key(r)}: {r['ts_us'] - r['eval_ts_us']} us after eval")
            out = self.schedule.get(r["scene"])
            if out is None:
                expect = False
            else:
                expect = is_corrupted(r["cam"], r["ts_us"], out.t_start_us, self.duration_s)
            if r["corrupted"] != expect:
                errs.append(f"flag mismatch {key(r)}: got {r['corrupted']} want {expect}")
            if expect and not r["all_black"]:
                errs.append(f"expected black input not black: {r['path']}")
            if not expect and r["sha1"] != r["sha1_src"]:
                errs.append(f"unflagged modification {key(r)}")
            if not expect and r["cam"] == CAMERA and out is not None and r["ts_us"] < out.t_start_us \
                    and r["corrupted"]:
                errs.append(f"prefix modified {key(r)}")
            if r["cam"] != CAMERA and r["corrupted"]:
                errs.append(f"non-front camera corrupted {key(r)}")
        return errs

    def status(self):
        """RECORD-CONSISTENCY only: 'FAIL' | 'CONSISTENT' | 'NOT VERIFIED (...)'.
        CONSISTENT never implies a complete outage/recovery run; see readiness()."""
        if self.verify():
            return "FAIL"
        missing = []
        if self.manifest is None:
            missing.append("coverage manifest")
        if self.causal_tolerance_us is None:
            missing.append("causal policy")
        return "CONSISTENT" if not missing else "NOT VERIFIED (no " + ", ".join(missing) + ")"

    def readiness(self):
        """Study-protocol readiness: consistency + every scheduled scene observed + (outage runs)
        consumed corrupted front images in each scene + all 0..4.5 s follow-up bins evaluated.
        Returns 'READY' or 'NOT READY: reasons'. Scenes skipped by the schedule are NOT in
        self.schedule; their reasons live in the persisted schedule file (never silently dropped)."""
        why = []
        if self.status() != "CONSISTENT":
            why.append(f"records {self.status()}")
        if any(r["eval_ts_us"] is None or r["sample"] in (None, "") for r in self.records):
            why.append("records lack sample identity / eval timestamp")
        if self.sample_meta is None:
            why.append("no independent sample metadata for eval identity/timestamps")
        unseen = [t for t in self.schedule if not any(r["scene"] == t for r in self.records)]
        if unseen:
            why.append(f"{len(unseen)} scheduled scenes unobserved")
        if self.duration_s > 0:
            noout = [t for t in self.schedule if t not in self.consumed_corrupted_per_scene()]
            if noout:
                why.append(f"{len(noout)} scenes never consumed a corrupted image")
            mb = self.missing_bins()
            if mb:
                why.append(f"follow-up bins missing in {len(mb)} scenes")
        return "READY" if not why else "NOT READY: " + "; ".join(why)

    def consumed_corrupted_per_scene(self):
        out = {}
        for r in self.records:
            if r["corrupted"]:
                out.setdefault(r["scene"], []).append(r["ts_us"])
        return out

    def last_corrupted_ts(self, scene):
        t = self.consumed_corrupted_per_scene().get(scene)
        return max(t) if t else None

    def eval_samples(self):
        """Unique evaluated samples {(scene, sample): eval_ts_us}; history re-consumption and
        extra cameras/sweeps do not create new observations."""
        out = {}
        for r in self.records:
            if r["eval_ts_us"] is not None:
                out.setdefault((r["scene"], r["sample"]), r["eval_ts_us"])  # conflicts: see eval_conflicts()
        return out

    def restoration_offsets(self):
        """Per scene: sorted [(sample, eval_ts - t_restore seconds)] for evaluated samples at/after
        restoration, once per evaluated sample (evaluation time, not camera exposure time)."""
        out = {}
        if self.duration_s <= 0:
            return out
        for (scene, sample), ev in self.eval_samples().items():
            o = self.schedule.get(scene)
            if o:
                off = (ev - o.restoration_us(self.duration_s)) / US
                if off >= 0:
                    out.setdefault(scene, []).append((sample, off))
        return {k: sorted(v, key=lambda x: x[1]) for k, v in out.items()}

    def missing_bins(self, bins=tuple(i * 0.5 for i in range(10)), tol_s=0.25):
        """Scheduled post-restoration bins (0..4.5 s) with no DISTINCT evaluated sample.
        One-to-one assignment: candidate (bin, sample) pairs within tol_s are taken in order of
        |offset - bin|, ties broken by smaller bin, then smaller offset; each sample and each bin
        is used at most once (so a 0.25 s sample serves bin 0.0, never also 0.5).
        The downstream evaluator must use this same convention. Downstream rule for a missing bin (incl. 0.0): it
        breaks only the adjacent recovery pair containing it; later valid adjacent pairs may still establish onset;
        the event is NOT discarded and is NOT treated as permanent failure. Never change the schedule from outcomes."""
        miss = {}
        offs = self.restoration_offsets()
        for scene in self.schedule:
            have = sorted(o for _, o in offs.get(scene, []))
            pairs = sorted((abs(h - b), b, h, i) for b in bins for i, h in enumerate(have)
                           if abs(h - b) <= tol_s + 1e-9)
            used_b, used_s = set(), set()
            for _, b, h, i in pairs:
                if b not in used_b and i not in used_s:
                    used_b.add(b); used_s.add(i)
            m = [b for b in bins if b not in used_b]
            if m:
                miss[scene] = m
        return miss

    def artifact_hashes(self):
        """Stable sha1 of the retained expected manifest and sample metadata (None if absent)."""
        h = lambda o: None if o is None else hashlib.sha1(
            json.dumps(sorted(map(list, o)), default=str).encode()).hexdigest()
        meta = None if self.sample_meta is None else [[*k, v] for k, v in self.sample_meta.items()]
        return dict(manifest_sha1=h(self.manifest), sample_meta_sha1=h(meta))

    def save(self, path):
        sched = {k: v.__dict__ for k, v in self.schedule.items()}
        with open(path, "w") as f:
            json.dump(dict(model=self.model, run_id=self.run_id, condition=self.condition,
                           duration_s=self.duration_s, schedule=sched,
                           status=self.status(), readiness=self.readiness(), violations=self.verify(),
                           manifest_id=self.manifest_id, causal_tolerance_us=self.causal_tolerance_us,
                           expected_manifest=None if self.manifest is None else sorted(map(list, self.manifest)),
                           sample_metadata=None if self.sample_meta is None else [[*k, v] for k, v in sorted(self.sample_meta.items())],
                           hashes=self.artifact_hashes(), missing_bins=self.missing_bins(), eval_conflicts=self.eval_conflicts(),
                           restoration_offsets=self.restoration_offsets(),
                           corrupted_files=sorted({r["path"] for r in self.records if r["corrupted"]}),
                           verification="recorded-only (hashes taken inside the pipeline step)",
                           records=self.records), f)


def load_audit(path):
    """Rebuild an InputAudit from save() output (records, schedule, manifest, sample metadata)."""
    with open(path) as f:
        d = json.load(f)
    sched = {k: SceneOutage(**v) for k, v in d["schedule"].items()}
    a = InputAudit(d["model"], d["condition"], d["duration_s"], sched, run_id=d["run_id"],
                   causal_tolerance_us=d["causal_tolerance_us"])
    a.records = d["records"]
    if d.get("expected_manifest") is not None:
        a.set_manifest({tuple(k) for k in d["expected_manifest"]}, d["manifest_id"])
    if d.get("sample_metadata") is not None:
        a.set_sample_metadata({(k[0], k[1]): k[2] for k in d["sample_metadata"]})
    return a


def compare_runs(clean: InputAudit, outage: InputAudit):
    """Paired check. Key sets must match exactly; every image outside the corruption window
    must be hash-identical (all cameras, prefix AND restored inputs); corrupted images must have
    the same raw source as the clean run."""
    errs = []
    if clean.model != outage.model:
        errs.append("model mismatch")
    if clean.schedule != outage.schedule:
        errs.append("schedule mismatch between paired audits")
    if clean.condition != "clean":
        errs.append("first audit is not a clean run")
    errs += [f"clean audit: {e}" for e in clean.verify()]
    errs += [f"outage audit: {e}" for e in outage.verify()]
    c = {key(r): r for r in clean.records}
    o = {key(r): r for r in outage.records}
    if not c or not o:
        return ["empty audit in pair"]
    errs += [f"in clean only {k}" for k in sorted(set(c) - set(o), key=str)]
    errs += [f"in outage only {k}" for k in sorted(set(o) - set(c), key=str)]
    for k in set(c) & set(o):
        a, b = c[k], o[k]
        if a["sha1_src"] != b["sha1_src"]:
            errs.append(f"source pixels differ {k}")
        if not b["corrupted"] and a["sha1"] != b["sha1"]:
            errs.append(f"uncorrupted input changed {k}")
        if a["corrupted"]:
            errs.append(f"clean run has corrupted input {k}")
    return errs
