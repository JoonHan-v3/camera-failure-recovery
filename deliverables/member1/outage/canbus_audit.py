"""CAN-bus causal audit for BEVFormer info files (supplements the per-image input audit; tied to the info pkl sha256).

Upstream converter (_get_can_bus_info): `last_pose` = last pose message with utime <= sample ts (or the first message if all are
later); the loop variable `pose` at break = FIRST message with utime > sample ts (or the last message if none is later).
pos (3) + orientation (4) come from last_pose; accel, rotation_rate, vel (3 each = 9 auxiliary values, can_bus[7:16]) come from
`pose`. At inference CustomNuScenesDataset.get_data_info OVERWRITES can_bus[0:7] with the ego pose and [16:18] with yaw, so the
values that reach the model from CAN are exactly the 9 auxiliary ones (-> can_bus_mlp). For every info this script recomputes
both selections from the raw pose JSON, checks the 9 stored values equal the selected message exactly, and records
offset = message utime - sample ts. Policy: the frozen sample+50 ms alignment allowance (audit.CAUSAL_TOL_US); any offset above
it is a violation. Usage: python -m outage.canbus_audit <infos.pkl> <out.json>   (exit 1 on violation / mismatch)"""
import hashlib, json, os, pickle, sys
import numpy as np
from .audit import CAUSAL_TOL_US

CAN = os.path.expanduser("~/storage/nuscenes_can_bus/can_bus")
META = os.path.expanduser("~/storage/nuscenes/v1.0-trainval")


def audit(infos_pkl):
    infos = pickle.load(open(infos_pkl, "rb"))["infos"]
    scene_name = {s["token"]: s["name"] for s in json.load(open(os.path.join(META, "scene.json")))}
    cache, rows, errs = {}, [], []
    for inf in infos:
        name = scene_name[inf["scene_token"]]
        if name not in cache:
            cache[name] = json.load(open(os.path.join(CAN, f"{name}_pose.json")))
        pl, ts = cache[name], inf["timestamp"]
        last, sel = pl[0], pl[-1]
        for p in pl:
            if p["utime"] > ts:
                sel = p
                break
            last = p
        aux = np.array(list(sel["accel"]) + list(sel["rotation_rate"]) + list(sel["vel"]))
        stored = np.asarray(inf["can_bus"])[7:16]
        match = bool(np.array_equal(aux, stored))
        off_aux, off_pos = sel["utime"] - ts, last["utime"] - ts
        if not match:
            errs.append(f"aux mismatch {inf['token']}")
        if off_aux > CAUSAL_TOL_US:
            errs.append(f"aux CAN message {off_aux} us after sample {inf['token']} > {CAUSAL_TOL_US}")
        rows.append(dict(sample=inf["token"], scene=name, sample_ts_us=ts, aux_msg_utime=sel["utime"], aux_offset_us=off_aux,
                         pos_msg_utime=last["utime"], pos_offset_us=off_pos, aux_match=match))
    sha = hashlib.sha256(open(infos_pkl, "rb").read()).hexdigest()
    a = np.array([r["aux_offset_us"] for r in rows]); p = np.array([r["pos_offset_us"] for r in rows])
    return dict(infos=os.path.abspath(infos_pkl), infos_sha256=sha, n=len(rows), scenes=len(cache), policy_us=CAUSAL_TOL_US,
                aux_offset_us=dict(min=int(a.min()), max=int(a.max()), median=float(np.median(a)), n_positive=int((a > 0).sum()),
                                   n_over_policy=int((a > CAUSAL_TOL_US).sum())),
                pos_offset_us_unused_overwritten=dict(min=int(p.min()), max=int(p.max()), n_positive=int((p > 0).sum())),
                all_aux_match=all(r["aux_match"] for r in rows), violations=errs, PASS=not errs, records=rows)


def main(infos_pkl, out):
    rep = audit(infos_pkl)
    json.dump(rep, open(out, "w"), indent=1)
    print({k: v for k, v in rep.items() if k != "records"})
    sys.exit(0 if rep["PASS"] else 1)


if __name__ == "__main__":
    main(*sys.argv[1:3])
