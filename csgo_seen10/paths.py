"""Host-independent Seen-10 paths; no model imports or filesystem writes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = "../UniLIP/data/csgo_benchmark_v2"
DEFAULT_SHARED_EVAL_DIR = "../csgo_benchmark_v2_eval_general"
DEFAULT_MODEL_PATH = "checkpoints/openvla-7b"
DEFAULT_OUTPUT_ROOT = "outputs/csgo_benchmark_v2_seen10"

# Only these original host defaults may fall back when absent. An explicit
# CLI/environment override or a custom YAML path is never silently replaced.
LEGACY_DEFAULTS = {
    "data_root": "/home/jiahao/task/UniLIP/data/csgo_benchmark_v2",
    "shared_eval_dir": "/home/jiahao/task/csgo_benchmark_v2_eval_general",
    "unilip_python": "/home/jiahao/miniconda3/envs/UniLIP/bin/python",
}


def project_path(value: Any, *, root: Path = PROJECT_ROOT) -> Path:
    candidate = Path(os.fspath(value)).expanduser()
    # Keep Python executable symlinks: resolving .venv/bin/python to a base
    # interpreter would bypass the project's installed dependencies.
    return candidate if candidate.is_absolute() else root / candidate


def _resolve(config: Mapping[str, Any], key: str, default: Any, env_names: tuple[str, ...],
             *, root: Path, env: Mapping[str, str]) -> Path:
    explicit = config.get("_path_overrides", {}).get(key)
    if explicit is not None:
        return project_path(explicit, root=root)
    for name in env_names:
        if env.get(name):
            return project_path(env[name], root=root)
    value = config.get(key)
    if value:
        candidate = project_path(value, root=root)
        if str(value) != LEGACY_DEFAULTS.get(key) or candidate.exists():
            return candidate
    return project_path(default, root=root)


def data_root(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT,
              env: Mapping[str, str] | None = None) -> Path:
    values = dict(config)
    values.setdefault("data_root", config.get("data", {}).get("root"))
    return _resolve(values, "data_root", DEFAULT_DATA_ROOT, ("CSGO_DATA_ROOT", "DATA_ROOT"),
                    root=root, env=os.environ if env is None else env)


def evaluator_root(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT,
                   env: Mapping[str, str] | None = None) -> Path:
    return _resolve(config, "shared_eval_dir", DEFAULT_SHARED_EVAL_DIR,
                    ("SHARED_EVAL_DIR", "CSGO_EVAL_ROOT"), root=root,
                    env=os.environ if env is None else env)


def evaluator_python(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT,
                     env: Mapping[str, str] | None = None) -> Path:
    return _resolve(config, "unilip_python", ".venv/bin/python", ("UNILIP_PYTHON",),
                    root=root, env=os.environ if env is None else env)


def model_path(config: Mapping[str, Any], *, root: Path = PROJECT_ROOT,
               env: Mapping[str, str] | None = None) -> str:
    environment = os.environ if env is None else env
    value = config.get("_path_overrides", {}).get("model_path")
    if value is None:
        value = environment.get("OPENVLA_MODEL_PATH") or config.get(
            "model_path", config.get("model", {}).get("path", DEFAULT_MODEL_PATH))
    # Retain the legacy public Hub ID; aligned still requires local base shards.
    if str(value) == "openvla/openvla-7b":
        return str(value)
    return str(project_path(value, root=root))
