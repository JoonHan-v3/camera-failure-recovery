#!/bin/bash
# Real StreamPETR runs on nuScenes v1.0-mini (debug set): clean twice + three outage durations.
cd "$(dirname "$0")/.."
export PATH=$HOME/anaconda3/envs/streampetr/bin:$PATH PYTHONPATH=outage/shims:streampetr_src:.
for c in "clean a" "clean b" "0.5 d05" "1 d10" "2 d20"; do set -- $c
  python -m outage.run_streampetr --split mini --condition $1 --out artifacts/runs/mini_$2 2>&1 | grep -v Warning | tail -3
done
echo MATRIX_DONE
