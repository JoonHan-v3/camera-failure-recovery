#!/bin/bash
# THE BEVFormer runner script (fail-fast). Usage: artifacts/run_matrix.sh <pilot|val> <clean|0.5|1|2> <name>
set -euo pipefail
P=$(cd "$(dirname "$0")/.." && pwd)
export PATH=$HOME/anaconda3/envs/bevformer/bin:$PATH
cd "$P/bevformer_src"; export PYTHONPATH=$PWD:$PWD/..
python -m outage.run_bevformer --split "$1" --condition "$2" --out "../artifacts/runs/${1}_$3" --no-tf32 > "../artifacts/logs/${1}_$3.log" 2>&1
echo "$1 $3 exit $?" >> ../artifacts/logs/status.txt
