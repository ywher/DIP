#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/run_road_experiment.sh CONFIG GPU [RESUME]" >&2
  exit 2
fi

config="$1"
gpu="$2"
resume="${3:-}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

args=(train --config "${config}")
if [[ -n "${resume}" ]]; then
  args+=(--resume "${resume}")
fi
CUDA_VISIBLE_DEVICES="${gpu}" python tools/road/run.py "${args[@]}"

