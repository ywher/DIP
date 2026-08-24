#!/usr/bin/env python3
"""Validate configs, manifests, data links, and pretrained checkpoints."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from road_dip.config import load_config, validate_config
from road_dip.data import read_manifest, verify_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument("--all-files", action="store_true")
    args = parser.parse_args()

    failed = False
    for config_path in args.configs:
        try:
            config = load_config(config_path)
            validate_config(config)
            data = config["data"]
            for prefix in ("source", "support", "val"):
                records = read_manifest(
                    data[f"{prefix}_root"], data[f"{prefix}_list"], data[f"{prefix}_kind"]
                )
                verify_records(records, None if args.all_files else 20)
            pretrained = Path(config["model"]["pretrained"])
            if not pretrained.is_file():
                raise FileNotFoundError(pretrained)
            print(f"OK  {config_path}")
        except Exception as error:
            failed = True
            print(f"ERR {config_path}: {error}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()

