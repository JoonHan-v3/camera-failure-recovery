#!/bin/bash
# THE SparseBEV runner script (fail-fast). Usage: artifacts/run_matrix.sh <pilot|val> <clean|0.5|1|2> <name>
set -euo pipefail
P=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/anaconda3/envs/sparsebev/bin:$PATH
[ -f "$P/artifacts/runs/${1}_$3/preds.pkl" ] && { echo "$1 $3 already done"; exit 0; }
cd "$P/sparsebev_src"; export PYTHONPATH=$PWD:$PWD/..
python -m outage.run_sparsebev --split "$1" --condition "$2" --out "../artifacts/runs/${1}_$3" --no-tf32 > "../artifacts/logs/${1}_$3.log" 2>&1
echo "$1 $3 exit $?" >> ../artifacts/logs/status.txt
