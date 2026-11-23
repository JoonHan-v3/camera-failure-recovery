"""Control: shim-backed FlashMHA (6 decoder layers, released weights) vs torch.nn.functional.multi_head_attention_forward.
Run: PYTHONPATH=outage/shims:streampetr_src:. python -m outage.controls.shim_fidelity artifacts/controls/shim_fidelity.json
Random fp32 inputs (seed 0, 300 queries x 1100 keys), fp16 casting inside FlashAttention. Pass criterion (declared before
running): per-layer max abs error <= 0.05 AND max relative-to-output-scale error <= 0.01. Limited implementation check only:
it does NOT establish equivalence to the released FlashAttention kernel or explain the model-level baseline gap."""
import json, sys, importlib, hashlib, torch, torch.nn.functional as F
from mmcv import Config
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint, wrap_fp16_model

CFG = "streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py"
CKPT = "ckpts/stream_petr_r50_flash_704_bs2_seq_90e.pth"


def main(out):
    cfg = Config.fromfile(CFG)
    importlib.import_module("projects.mmdet3d_plugin")
    cfg.model.pretrained = None; cfg.model.train_cfg = None
    m = build_model(cfg.model, test_cfg=cfg.get("test_cfg")); wrap_fp16_model(m)
    load_checkpoint(m, CKPT, map_location="cpu"); m = m.cuda().eval()
    from projects.mmdet3d_plugin.models.utils.attention import FlashMHA
    mods = [x for x in m.modules() if isinstance(x, FlashMHA)]
    torch.manual_seed(0)
    rows = []
    for i, a in enumerate(mods):
        E, H = a.embed_dim, a.num_heads
        q, k, v = (torch.randn(1, n, E, device="cuda") for n in (300, 1100, 1100))
        with torch.no_grad():
            o, _ = a(q, k, v)
            ref, _ = F.multi_head_attention_forward(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), E, H,
                a.in_proj_weight, a.in_proj_bias, None, None, False, 0.0, a.out_proj.weight, a.out_proj.bias,
                training=False, need_weights=False)
        ref = ref.transpose(0, 1).float(); err = (o.float() - ref).abs()
        rows.append(dict(layer=i, max_abs_err=err.max().item(), mean_abs_err=err.mean().item(),
                         out_scale=ref.abs().max().item(), max_rel_to_scale=err.max().item() / ref.abs().max().item()))
    ok = all(r["max_abs_err"] <= 0.05 and r["max_rel_to_scale"] <= 0.01 for r in rows)
    shim = "outage/shims/flash_attn/flash_attn_interface.py"
    rep = dict(seed=0, n_layers=len(rows), layers=rows, criterion="max_abs<=0.05 and max_rel_to_scale<=0.01", passed=ok,
               shim_sha256=hashlib.sha256(open(shim, "rb").read()).hexdigest(), torch=torch.__version__,
               gpu=torch.cuda.get_device_name(0), checkpoint=CKPT)
    json.dump(rep, open(out, "w"), indent=1)
    print("passed", ok); [print(r) for r in rows]


if __name__ == "__main__":
    main(sys.argv[1])
