#!/bin/bash
# FROZEN configuration (baseline gate PASSED 2026-11-21, artifacts/runs/val_realflash): REAL flash-attn 0.2.2 built in conda env
# streampetr_fa (nvcc 11.8 + system gcc-11; see artifacts/flashbuild/PROVENANCE.txt), NO shim on PYTHONPATH, --no-tf32.
# usage: ./artifacts/run_matrix_realflash.sh pilot            -> rf_pilot_{a,b,d05,d10,d20} sequentially
#        ./artifacts/run_matrix_realflash.sh val <cond> <name> -> one val run (rf_val_<name>), e.g. val 0.5 d05
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/anaconda3/envs/streampetr_fa/bin:$PATH PYTHONPATH=streampetr_src:.
run() {  # split cond name
  out=artifacts/runs/rf_$1_$3
  python -m outage.run_streampetr --split $1 --condition $2 --out $out --no-tf32 2>&1 | grep -v Warning
  for f in preds.pkl audit.json run.json; do [ -s $out/$f ] || { echo "MISSING $out/$f" >&2; exit 1; }; done
  python - "$out" <<'P'
import json, sys
d = json.load(open(sys.argv[1] + "/run.json"))
assert d["audit_status"] == "CONSISTENT", d["violations"]
assert d["flash_attn"]["kind"] == "REAL flash-attn", d["flash_attn"]
assert d["tf32_matmul"] is False and d["tf32_cudnn"] is False
print("OK", sys.argv[1], d["audit_status"], d["readiness"])
P
}
if [ "$1" = pilot ]; then
  for c in "clean a" "clean b" "0.5 d05" "1 d10" "2 d20"; do set -- pilot $c; run $1 $2 $3; done
  echo PILOT_RF_DONE
else
  run val $2 $3; echo VAL_RF_$3_DONE
fi
