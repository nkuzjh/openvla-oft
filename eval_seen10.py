#!/usr/bin/env python
"""Run the shared, model-independent CSGO evaluator."""

from __future__ import annotations

import os

os.environ.setdefault("OPENVLA_ROBOT_PLATFORM", "CSGO")

from csgo_seen10.cli import build_arg_parser, load_cli_config, print_paths


def main() -> None:
    parser = build_arg_parser("Shared CSGO Seen-10 evaluator")
    args = parser.parse_args()
    config = load_cli_config(args)
    if args.print_paths:
        print_paths(config, args)
        return
    from csgo_seen10.runner import eval_command

    raise SystemExit(eval_command(config, seed=args.seed, smoke=args.smoke))


if __name__ == "__main__":
    main()
