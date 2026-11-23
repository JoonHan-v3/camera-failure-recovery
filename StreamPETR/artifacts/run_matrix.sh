#!/bin/bash
# Fail-fast reproduction of the StreamPETR run matrix: ./artifacts/run_matrix.sh <mini|pilot|val>   (writes artifacts/runs/<split>_{a,b,d05,d10,d20})
# Verified invocation: PYTHONPATH=outage/shims:streampetr_src:. python -m outage.run_streampetr   (never `python -I`).
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=$HOME/anaconda3/envs/streampetr/bin:$PATH PYTHONPATH=outage/shims:streampetr_src:.
split=$1
for c in "clean a" "clean b" "0.5 d05" "1 d10" "2 d20"; do set -- $c
  out=artifacts/runs/${split}_$2
  python -m outage.run_streampetr --split $split --condition $1 --out $out 2>&1 | grep -v Warning
  for f in preds.pkl audit.json run.json; do [ -s $out/$f ] || { echo "MISSING $out/$f" >&2; exit 1; }; done
  python - "$out" <<'P'
import json, sys
d = json.load(open(sys.argv[1] + "/run.json"))
assert d["audit_status"] == "CONSISTENT", d["violations"]
print("OK", sys.argv[1], d["audit_status"], d["readiness"])
P
done
echo MATRIX_DONE
