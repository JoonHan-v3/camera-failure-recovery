#!/bin/bash
# Backup-model verification (tiny_fp16): full val clean (gate) + pilot clean + pilot 2 s with the same adapter/audit.
set -uo pipefail
P=$(cd "$(dirname "$0")/.." && pwd); export PATH=$HOME/anaconda3/envs/bevformer/bin:$PATH; cd "$P/bevformer_src"; export PYTHONPATH=$PWD:$PWD/..
run() { python -m outage.run_bevformer --model tiny_fp16 --split $1 --condition $2 --out ../artifacts/runs_backup/$3 --no-tf32 > ../artifacts/logs/backup_$3.log 2>&1; echo "backup $3 exit $?" >> ../artifacts/logs/status.txt; }
run val clean val_a & run pilot clean pilot_a & run pilot 2 pilot_d20 & wait
echo BACKUP_DONE >> ../artifacts/logs/status.txt
