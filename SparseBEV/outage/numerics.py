"""The exact numerics of the study runs (run_sparsebev.main) for the diagnostic scripts that build their own Runner:
TF32 off, mmdet set_random_seed(0, deterministic=True) (=> cudnn.deterministic), cudnn.benchmark False. Without this, a diagnostic can differ
from a study run by tiny conv-algorithm differences that flip near-tied top-k rows (observed: isolation diff ~50-90 on re-ordered rows)."""
import torch


def set_study_numerics():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from mmdet.apis import set_random_seed
    set_random_seed(0, deterministic=True)
    torch.backends.cudnn.benchmark = False
