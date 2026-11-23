import os
import torch

_FP16 = os.environ.get('FLASH_SHIM_MODE') == 'fp16'   # sensitivity experiment: attention math in the input dtype


def flash_attn_unpadded_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                      dropout_p=0.0, softmax_scale=None, causal=False, return_attn_probs=False):
    """q: (total_q, H, D); kv: (total_k, 2, H, D); cu_seqlens: int32 (B+1,). Returns (total_q, H, D)."""
    assert dropout_p == 0.0, "shim supports inference only"
    scale = softmax_scale if softmax_scale is not None else q.shape[-1] ** -0.5
    out = torch.empty_like(q)
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    for b in range(len(cq) - 1):
        f = (lambda t: t) if _FP16 else (lambda t: t.float())
        qb = f(q[cq[b]:cq[b + 1]]).transpose(0, 1)                 # H, Tq, D
        kb = f(kv[ck[b]:ck[b + 1], 0]).transpose(0, 1)             # H, Tk, D
        vb = f(kv[ck[b]:ck[b + 1], 1]).transpose(0, 1)
        att = (qb @ kb.transpose(1, 2)) * scale
        if causal:
            m = torch.ones(att.shape[-2:], dtype=torch.bool, device=att.device).triu(1)
            att = att.masked_fill(m, float("-inf"))
        out[cq[b]:cq[b + 1]] = (att.softmax(-1) @ vb).transpose(0, 1).to(q.dtype)
    return out
