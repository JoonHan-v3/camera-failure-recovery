#!/bin/bash
# Sequential worker with retries: artifacts/run_worker.sh "<split> <cond> <name>" ... ; run_matrix.sh is idempotent (skips finished runs).
# Two workers run concurrently (a 12 GB GPU shared with the desktop fits two ~4.4 GB processes; a start-up race can still hit CUDA OOM,
# which leaves no preds.pkl, so the run is simply retried).
cd "$(dirname "$0")/.."
for item in "$@"; do
  for attempt in 1 2 3 4 5; do
    artifacts/run_matrix.sh $item || true
    [ -f artifacts/runs/$(echo $item | awk '{print $1"_"$3}')/preds.pkl ] && break
    echo "retry $item attempt $attempt" >> artifacts/logs/status.txt; sleep 40
  done
done
