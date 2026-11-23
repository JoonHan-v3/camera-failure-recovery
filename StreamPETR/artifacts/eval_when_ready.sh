#!/bin/bash
# waits for a finished val run, then scores it with the official nuScenes devkit (via the dataset's own evaluate)
set -uo pipefail
R=/home/rockhan/Documents/UM/rob535/projects/StreamPETR; run=$1
until [ -s $R/artifacts/runs/$run/run.json ]; do sleep 20; done
export PATH=$HOME/anaconda3/envs/streampetr/bin:$PATH
cd $R/streampetr_src
PYTHONPATH=$R/outage/shims:$R/streampetr_src:$R python -m outage.eval_official $R/streampetr_src/projects/configs/StreamPETR/stream_petr_r50_flash_704_bs2_seq_90e.py $R/artifacts/trainval_infos/val_infos.pkl $R/artifacts/runs/$run/preds.pkl $R/artifacts/eval/$run 2>&1 | grep -v Warn | tail -25
echo EVAL_DONE
