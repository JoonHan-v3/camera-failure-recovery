"""Online vs offline execution path (source review item MODELS.md §3(iii)): single-GPU test mode uses load_online + simple_test_online
(six current images decoded, history from the filename-keyed feature cache); multi-GPU test mode uses load_offline + simple_test_offline
(all 8 frames decoded and passed through the backbone in one batch). This script runs BOTH on the same clean scenes with the study
numerics and compares predictions. The study uses the online path (one GPU); this check says whether that choice changes any prediction.
Note: the offline loader reads history frames directly from disk (after OutageFrontCamera), so a corrupted history frame would NOT stay
corrupted there; only the clean comparison is meaningful, which is all that is claimed.
Usage (from sparsebev_src): python -m outage.online_offline_check <infos.pkl> <online_clean_preds.pkl> <out.json> <scene,...>"""
import json, pickle, sys
import numpy as np, torch, mmcv
from .run_sparsebev import Runner
from .adapter import OutageFrontCamera, CorruptFrontCamera
from .state import SceneStateGuard
from .verify_runs import pred_diff


def main(infos, online_pkl, out, scenes):
    from .numerics import set_study_numerics
    set_study_numerics()
    scenes = scenes.split(",")
    r = Runner(infos, scenes)
    from loaders.pipelines.loading import LoadMultiViewImageFromMultiSweeps as L
    L.__call__ = lambda self, res: self.load_offline(res)               # force the multi-GPU loader
    det = r.det
    seen = dict(offline_calls=0, backbone_batches=[], online_calls=0)
    orig_off, orig_on = det.simple_test_offline, det.simple_test_online
    det.simple_test_online = lambda *a, **k: (seen.__setitem__("online_calls", seen["online_calls"] + 1), orig_on(*a, **k))[1]
    det.simple_test = lambda img_metas, img=None, rescale=False: (seen.__setitem__("offline_calls", seen["offline_calls"] + 1), orig_off(img_metas, img, rescale))[1]
    hook = det.img_backbone.register_forward_pre_hook(lambda m, inp: seen["backbone_batches"].append(int(inp[0].shape[0])))
    OutageFrontCamera.STEP = CorruptFrontCamera({}, 0.0, None, guard=SceneStateGuard(), ts_mode="abs_s")
    off = {}
    with torch.no_grad():
        for i in range(len(r.dataset)):
            data = mmcv.parallel.collate([r.dataset[i]], 1)
            b = r.dp(return_loss=False, rescale=True, **data)[0]["pts_bbox"]
            off[r.dataset.data_infos[i]["token"]] = torch.cat([b["boxes_3d"].tensor.float().cpu(), b["labels_3d"].float().cpu()[:, None],
                                                                 b["scores_3d"].float().cpu()[:, None]], 1).numpy()
    on = {k: v for k, v in pickle.load(open(online_pkl, "rb")).items() if k in off}
    per = [pred_diff({k: on[k]}, {k: off[k]}) for k in off]
    same_shape = sum(on[k].shape == off[k].shape for k in off)
    finite = [d for d in per if np.isfinite(d)]
    hook.remove()
    rep = dict(offline_calls=seen["offline_calls"], online_calls_during_offline=seen["online_calls"],
               backbone_batch_sizes=sorted(set(seen["backbone_batches"])), n_samples=len(off), same_row_count=same_shape, max_abs_diff_finite=float(max(finite)) if finite else None,
               n_exact_equal=int(sum(d == 0.0 for d in per)), n_diff_above_1e_3=int(sum(d > 1e-3 for d in per)),
               n_shape_mismatch=int(sum(not np.isfinite(d) for d in per)))
    json.dump(rep, open(out, "w"), indent=1); print(rep)


if __name__ == "__main__":
    main(*sys.argv[1:5])
