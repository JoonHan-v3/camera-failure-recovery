#!/bin/bash
# start the second clean val run as soon as the 0.5 s val run frees its GPU slot (max 3 concurrent processes)
cd "$(dirname "$0")/.."
until grep -q VAL_RF_d05_DONE artifacts/rf_val_d05.log 2>/dev/null; do sleep 20; done
./artifacts/run_matrix_realflash.sh val clean b > artifacts/rf_val_b.log 2>&1
