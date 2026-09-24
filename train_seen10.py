#!/usr/bin/env python
"""Train native OpenVLA-OFT on CSGO Benchmark v2 Seen-10."""

from __future__ import annotations

import os

os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

from csgo_seen10.cli import build_arg_parser, load_cli_config, print_paths


def main() -> None:
    parser = build_arg_parser("Native OpenVLA-OFT Seen-10 localization training")
    parser.add_argument("--resume-checkpoint", default=None, help="step directory, best/late link, or run directory")
    args = parser.parse_args()
    config = load_cli_config(args)
    if args.print_paths:
        print_paths(config, args)
        return
    from csgo_seen10.runner import train

    train(config, seed=args.seed, resume_checkpoint=args.resume_checkpoint, smoke=args.smoke)


if __name__ == "__main__":
    main()
