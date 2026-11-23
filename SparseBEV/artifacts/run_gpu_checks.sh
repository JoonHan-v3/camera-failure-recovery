#!/bin/bash
# GPU checks (isolation, backbone hook probe, online/offline check). A 12 GB GPU shared with the desktop fits only two ~4.4 GB SparseBEV processes,
# so this starts as soon as fewer than two study processes are running.
cd "$(dirname "$0")/.."
export PATH=$HOME/anaconda3/envs/sparsebev/bin:$PATH
while pgrep -f "outage[.]run_sparsebev" > /dev/null && [ $(pgrep -fc "outage[.]run_sparsebev") -ge 2 ]; do sleep 30; done   # waits only while two study processes are running
cd sparsebev_src; export PYTHONPATH=$PWD:$PWD/..
SC=$(cat ../artifacts/probe_scenes.txt)
python -m outage.isolation ../artifacts/infos/pilot_infos.pkl ../artifacts/runs/pilot_a/preds.pkl ../artifacts/schedule_trainval_train.json ../artifacts/isolation_pilot.json > ../artifacts/logs/isolation_pilot.log 2>&1
python -m outage.backbone_hook_probe ../artifacts/infos/pilot_infos.pkl ../artifacts/schedule_trainval_train.json 2 ../artifacts/backbone_hook_probe.json $SC > ../artifacts/logs/backbone_hook_probe.log 2>&1
python -m outage.online_offline_check ../artifacts/infos/pilot_infos.pkl ../artifacts/runs/pilot_a/preds.pkl ../artifacts/online_offline_check.json $SC > ../artifacts/logs/online_offline_check.log 2>&1
echo GPU_CHECKS_DONE >> ../artifacts/logs/status.txt
