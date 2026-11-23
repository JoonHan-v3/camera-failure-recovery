#!/bin/bash
# Run the remaining matrix, at most 4 GPU processes at once (8 at once -> CUDA OOM on the 12 GB RTX 3060).
cd "$(dirname "$0")/.."
printf '%s\n' "pilot clean b" "pilot 0.5 d05" "pilot 1 d10" "pilot 2 d20" "val clean a" "val 0.5 d05" "val 1 d10" "val 2 d20" "val clean b" \
 | xargs -P 3 -L 1 artifacts/run_matrix.sh
echo QUEUE_DONE >> artifacts/logs/status.txt
