#!/bin/bash
# Run the study matrix, at most 2 GPU processes at once (each SparseBEV process holds ~4.4 GB incl. CUDA context; 3 -> CUDA OOM on the 12 GB RTX 3060 shared with the desktop).
cd "$(dirname "$0")/.."
printf '%s\n' "pilot clean a" "pilot clean b" "pilot 0.5 d05" "pilot 1 d10" "pilot 2 d20" "val clean a" "val 0.5 d05" "val 1 d10" "val 2 d20" "val clean b" \
 | xargs -P 2 -L 1 artifacts/run_matrix.sh
echo QUEUE_DONE >> artifacts/logs/status.txt
