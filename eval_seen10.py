#!/usr/bin/env python
"""Run the shared, model-independent CSGO evaluator."""

from __future__ import annotations

import os

os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

from csgo_seen10.runner import build_arg_parser, eval_command, load_config


def main() -> None:
    parser = build_arg_parser("Shared CSGO Seen-10 evaluator")
    args = parser.parse_args()
    config = load_config(args.config)
    raise SystemExit(eval_command(config, seed=args.seed, smoke=args.smoke))


if __name__ == "__main__":
    main()
