#!/bin/bash
# official nuScenes detection eval (via the dataset's own evaluate) of saved rf_val_* predictions; run from streampetr_src (data root)
# usage: ./artifacts/eval_runs_official.sh rf_val_b rf_val_d05 ...
set -uo pipefail
R=$(cd "$(dirname "$0")/.." && pwd); export PATH=$HOME/anaconda3/envs/streampetr_fa/bin:$PATH
cd $R/streampetr_src
for run in "$@"; do
  PYTHONPATH=$R/streampetr_src:$R python -m outage.eval_official \
    $R/streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py \
    $R/artifacts/trainval_infos/val_infos.pkl $R/artifacts/runs/$run/preds.pkl $R/artifacts/eval/$run \
    > $R/artifacts/eval/$run.log 2>&1 && echo "EVAL_OK $run" || echo "EVAL_FAIL $run"
done
