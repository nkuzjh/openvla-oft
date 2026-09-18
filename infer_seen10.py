#!/usr/bin/env python
"""Run native OpenVLA-OFT Seen-10 localization inference."""

from __future__ import annotations

import os

os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

from csgo_seen10.runner import build_arg_parser, inference, load_config


def main() -> None:
    parser = build_arg_parser("Native OpenVLA-OFT Seen-10 localization inference")
    parser.add_argument("--checkpoint", default=None, help="best/late checkpoint or run directory")
    parser.add_argument("--resume", action="store_true", help="fill only missing sample IDs in an existing JSONL")
    args = parser.parse_args()
    config = load_config(args.config)
    inference(config, seed=args.seed, checkpoint=args.checkpoint, smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    main()
