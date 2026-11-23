#!/bin/bash
# at most 3 concurrent GPU processes (each ~3.5 GB; 5 concurrent crashed with CUDA OOM): run these three sequentially
set -uo pipefail
cd "$(dirname "$0")/.."
for c in "1 d10" "2 d20" "clean b"; do set -- $c
  ./artifacts/run_matrix_realflash.sh val $1 $2 > artifacts/rf_val_$2.log 2>&1 || echo "FAILED val $2" >> artifacts/rf_val_chain.log
done
echo CHAIN_DONE >> artifacts/rf_val_chain.log
