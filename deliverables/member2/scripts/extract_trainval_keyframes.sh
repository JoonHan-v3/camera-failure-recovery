#!/bin/bash
# Extract ONLY keyframe camera + lidar images from the 10 downloaded trainval blob archives into a NEW empty directory
# (downloaded archives are untrusted data: never extract into an existing tree). ~52 GB output, ~20 min with 10 parallel readers.
# usage: ./artifacts/extract_trainval_keyframes.sh [archive_dir=~/storage/nuscenes_dl] [out_dir=~/storage/nuscenes_trainval_keyframes]
set -u
SRC=${1:-$HOME/storage/nuscenes_dl}; D=${2:-$HOME/storage/nuscenes_trainval_keyframes}
[ -e "$D" ] && { echo "refusing: $D already exists (use a new, empty directory)"; exit 1; }
mkdir -p "$D"; LOG="$D/../nuscenes_extract.log"; cd "$SRC"
for i in 01 02 03 04 05 06 07 08 09 10; do
  ( tar -xzf v1.0-trainval${i}_blobs.tgz -C "$D" --no-same-owner --wildcards 'samples/CAM_*' 'samples/LIDAR_TOP/*' 2>"$D/../nuscenes_extract_$i.err"
    echo "done $i rc=$?" >> "$LOG" ) &
done
wait; echo ALL_DONE >> "$LOG"
# then: mkdir ~/storage/nuscenes_trainval_root && cd it && ln -s ../nuscenes_trainval_keyframes/samples samples \
#         && ln -s ../nuscenes/v1.0-trainval v1.0-trainval && ln -s ../nuscenes/maps maps
