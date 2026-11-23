import torch
import torch.nn.functional as F
from einops import rearrange


def index_first_axis(x, idx):
    return x[idx]


def unpad_input(hidden_states, attention_mask):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item())
    cu = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    return rearrange(hidden_states, "b s ... -> (b s) ...")[indices], indices, cu, max_seqlen


def pad_input(hidden_states, indices, batch, seqlen):
    out = torch.zeros((batch * seqlen, *hidden_states.shape[1:]), device=hidden_states.device, dtype=hidden_states.dtype)
    out[indices] = hidden_states
    return rearrange(out, "(b s) ... -> b s ...", b=batch)
