"""Pure-PyTorch stand-in for flash-attn 0.2.2 (no CUDA 11.1 toolchain here). Exact softmax attention in
fp32 -> same function as FlashAttention up to rounding; documented deviation from the reference env."""
