"""torch 1.9 compatibility for SparseBEV d341052 (written for torch >= 1.10): the only incompatibility is
`torch.meshgrid(x, y, indexing='ij')` in models/sparsebev_head.py:59. In torch 1.9 `meshgrid` has no `indexing` argument and
behaves as indexing='ij', so dropping the keyword is exactly equivalent. Any other value is rejected. Upstream source stays unmodified."""
import inspect
import torch

_orig = torch.meshgrid


def install():
    if "indexing" in str(inspect.signature(_orig)) or getattr(torch.meshgrid, "_outage_compat", False):
        return

    def meshgrid(*tensors, indexing=None):
        if indexing not in (None, "ij"):
            raise ValueError("only indexing='ij' is equivalent to torch 1.9 meshgrid")
        return _orig(*tensors)
    meshgrid._outage_compat = True
    torch.meshgrid = meshgrid
