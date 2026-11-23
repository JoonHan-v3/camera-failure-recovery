Runs val_a, val_d05, pilot_*, mini_* were produced with TF32 matmul/cuDNN ENABLED (torch 1.9 default on Ampere RTX 3060)
and the attention shim; val_a failed the baseline gate (NDS 0.5086 / mAP 0.3894). They are UNVALIDATED for the study.
The old matrix was stopped after val_d05; the partial val_d10 was deleted. val_notf32 tests TF32 as the cause.
