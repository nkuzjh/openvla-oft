"""Lightweight CLI and config handling, including non-executing path inspection."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import paths


def load_config(path: str | os.PathLike[str]) -> dict[str, Any]:
    config_path = paths.project_path(path).resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def build_arg_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/csgo_seen10.yaml")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", help="Benchmark data directory; overrides environment/YAML")
    parser.add_argument("--eval-root", help="Shared evaluator directory")
    parser.add_argument("--eval-python", "--unilip-python", dest="unilip_python",
                        help="Evaluator Python (default: <shared_eval_dir>/.venv/bin/python)")
    parser.add_argument("--model-path", help="Original OpenVLA base directory")
    parser.add_argument("--print-paths", action="store_true",
                        help="Print resolved paths without loading models or running any phase")
    return parser


def load_cli_config(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    fields = {"data_root": "data_root", "eval_root": "shared_eval_dir",
              "unilip_python": "unilip_python", "model_path": "model_path"}
    config["_path_overrides"] = {
        key: getattr(args, flag) for flag, key in fields.items() if getattr(args, flag) is not None
    }
    return config


def print_paths(config: Mapping[str, Any], args: argparse.Namespace) -> None:
    output_key = "smoke_output_root" if args.smoke else "output_root"
    default_output = paths.DEFAULT_OUTPUT_ROOT + ("_smoke" if args.smoke else "")
    run_dir = paths.project_path(config.get(output_key, default_output)) / "OpenVLA-OFT" / f"seed_{args.seed}"
    print(json.dumps({
        "project_root": str(paths.PROJECT_ROOT), "config": config["_config_path"],
        "data_root": str(paths.data_root(config).resolve()),
        "shared_eval_dir": str(paths.evaluator_root(config).resolve()),
        "evaluator_python": str(paths.evaluator_python(config)),
        "model_path": paths.model_path(config), "run_dir": str(run_dir.resolve()),
        "checkpoint": getattr(args, "checkpoint", None),
        "resume_checkpoint": getattr(args, "resume_checkpoint", None),
        "seed": args.seed, "smoke": args.smoke,
        "execution": "paths_only",
    }, indent=2))
