#!/bin/bash
# Independent reproduction of ONE outage result per model (pilot, 2 s outage) from the saved configuration, compared bit-for-bit with the saved run.
cd "$(dirname "$0")/.."
export PATH=$HOME/anaconda3/envs/sparsebev/bin:$PATH
for m in sparsebev bevformer streampetr; do python -m recovery.orchestrate reproduce $m pilot 2; done
echo REPRO_DONE >> artifacts/logs/status.txt
