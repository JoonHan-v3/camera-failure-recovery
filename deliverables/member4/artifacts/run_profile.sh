#!/bin/bash
# Resource profile of the three released detectors, one at a time on an otherwise idle GPU (each in its own conda env / project, like the study runs).
# Usage: artifacts/run_profile.sh   -> artifacts/profile/<model>.json
cd "$(dirname "$0")/.."
R=$PWD; mkdir -p artifacts/profile artifacts/logs; P=$(dirname "$R")
if pgrep -f "[o]utage[.]run_(sparsebev|bevformer|streampetr)" > /dev/null; then echo "a study run is using the GPU; not profiling" >&2; exit 1; fi
( export PATH=$HOME/anaconda3/envs/sparsebev/bin:$PATH; cd $P/SparseBEV/sparsebev_src; PYTHONPATH=$PWD:$PWD/.. python $R/recovery/profile_model.py sparsebev $R/artifacts/profile/sparsebev.json 3 > $R/artifacts/logs/profile_sparsebev.log 2>&1 )
( export PATH=$HOME/anaconda3/envs/bevformer/bin:$PATH; cd $P/BEVFormer/bevformer_src; PYTHONPATH=$PWD:$PWD/.. python $R/recovery/profile_model.py bevformer $R/artifacts/profile/bevformer.json 3 > $R/artifacts/logs/profile_bevformer.log 2>&1 )
( export PATH=$HOME/anaconda3/envs/streampetr_fa/bin:$PATH; cd $P/StreamPETR; PYTHONPATH=streampetr_src:. python $R/recovery/profile_model.py streampetr $R/artifacts/profile/streampetr.json 3 > $R/artifacts/logs/profile_streampetr.log 2>&1 )
echo PROFILE_DONE >> artifacts/logs/status.txt
