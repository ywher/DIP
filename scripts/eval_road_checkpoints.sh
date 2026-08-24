#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/eval_road_checkpoints.sh CONFIG GPU CHECKPOINT [CHECKPOINT ...]" >&2
  exit 2
fi
config="$1"
gpu="$2"
shift 2
for checkpoint in "$@"; do
  CUDA_VISIBLE_DEVICES="${gpu}" python tools/road/run.py eval \
    --config "${config}" --checkpoint "${checkpoint}"
done

