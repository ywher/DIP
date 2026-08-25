#!/usr/bin/env bash
set -euo pipefail

profile="${1:-vfm}"
gpu="${2:-0}"
if [[ "${profile}" != "vfm" && "${profile}" != "vfm_rdkc" && "${profile}" != "native" && "${profile}" != "native_rdkc" ]]; then
  echo "Profile must be native, native_rdkc, vfm, or vfm_rdkc" >&2
  exit 2
fi

tasks=(
  gta2cityscapes_1_64
  synthia2cityscapes_1_64
  cityscapes2acdc_1_64
  cityscapes2muses_1_64
  cityscapes2mapillary_1_128
)
for task in "${tasks[@]}"; do
  bash scripts/run_road_experiment.sh "configs/road/${profile}/${task}.yaml" "${gpu}"
done

