#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${1:-${DIP_DATA_ROOT:-}}"

if [[ -z "${data_root}" ]]; then
  echo "Usage: bash scripts/setup_road_data.sh /path/to/datasets" >&2
  exit 2
fi

declare -A candidates=(
  [gtav]="GTAV"
  [synthia]="synthia"
  [cityscapes]="cityscapes"
  [acdc]="acdc"
  [muses]="muses"
  [mapillary]="mapillary"
)

mkdir -p "${repo_root}/data"
for name in "${!candidates[@]}"; do
  source_path="${data_root}/${candidates[$name]}"
  if [[ ! -d "${source_path}" && "${name}" == "gtav" ]]; then
    source_path="${data_root}/gtav"
  fi
  if [[ ! -d "${source_path}" ]]; then
    echo "Missing ${name}: ${source_path}" >&2
    exit 1
  fi
  ln -sfn "${source_path}" "${repo_root}/data/${name}"
  printf '%-12s -> %s\n' "${name}" "${source_path}"
done

echo "Dataset links are ready under ${repo_root}/data"

