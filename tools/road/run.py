#!/usr/bin/env python3
"""Unified entry point for road-scene DIP training and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from road_dip.config import apply_overrides, load_config, validate_config
from road_dip.engine import evaluate_checkpoint, train


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "eval", "show-config"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--resume")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    validate_config(config)
    if args.command == "show-config":
        print(json.dumps(config, indent=2))
    elif args.command == "train":
        train(config, resume=args.resume)
    else:
        if not args.checkpoint:
            raise SystemExit("eval requires --checkpoint")
        evaluate_checkpoint(config, args.checkpoint)


if __name__ == "__main__":
    main()

