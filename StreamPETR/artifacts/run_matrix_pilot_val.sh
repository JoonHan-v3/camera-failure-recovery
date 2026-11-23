#!/bin/bash
# Real StreamPETR runs. pilot = 10 frozen training-split scenes (clean twice + outages); val = all 150 val scenes.
cd "$(dirname "$0")/.."
export PATH=$HOME/anaconda3/envs/streampetr/bin:$PATH PYTHONPATH=outage/shims:streampetr_src:.
run() { python -m outage.run_streampetr --split $1 --condition $2 --out artifacts/runs/$1_$3 2>&1 | grep -v Warning | tail -2; }
for c in "clean a" "clean b" "0.5 d05" "1 d10" "2 d20"; do set -- $c; run pilot $1 $2; done
echo PILOT_DONE
for c in "clean a" "0.5 d05" "1 d10" "2 d20" "clean b"; do set -- $c; run val $1 $2; done
echo VAL_DONE
