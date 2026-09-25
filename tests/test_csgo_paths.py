"""Portable path/CLI checks; no model, GPU, dataset, or evaluator execution."""

import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from csgo_seen10 import paths


class PortablePathsTest(unittest.TestCase):
    def test_other_server_and_overrides(self):
        root = Path("/home/user/yc57963/task/openvla-oft")
        self.assertEqual(paths.data_root({}, root=root, env={}).resolve(),
                         root.parent / "UniLIP/data/csgo_benchmark_v2")
        self.assertEqual(paths.evaluator_root({}, root=root, env={}).resolve(),
                         root.parent / "csgo_benchmark_v2_eval_general")
        self.assertEqual(paths.evaluator_python({}, root=root, env={}),
                         root / "../csgo_benchmark_v2_eval_general/.venv/bin/python")
        self.assertEqual(paths.model_path({}, root=root, env={}), str(root / "checkpoints/openvla-7b"))
        config = {"data_root": "yaml-data", "_path_overrides": {"data_root": "cli-data"}}
        env = {"CSGO_DATA_ROOT": "env-data", "DATA_ROOT": "old-alias"}
        self.assertEqual(paths.data_root(config, root=root, env=env), root / "cli-data")
        del config["_path_overrides"]
        self.assertEqual(paths.data_root(config, root=root, env=env), root / "env-data")
        self.assertEqual(paths.data_root(config, root=root, env={}), root / "yaml-data")
        self.assertEqual(paths.evaluator_root({}, root=root, env={"CSGO_EVAL_ROOT": "eval"}), root / "eval")

    def test_legacy_defaults_only_fall_back_when_missing(self):
        root = Path("/new/task/openvla-oft")
        config = dict(paths.LEGACY_DEFAULTS)
        with patch.object(Path, "exists", return_value=False):
            self.assertEqual(paths.data_root(config, root=root, env={}).resolve(),
                             root.parent / "UniLIP/data/csgo_benchmark_v2")
        with patch.object(Path, "exists", return_value=True):
            self.assertEqual(paths.data_root(config, root=root, env={}),
                             Path(paths.LEGACY_DEFAULTS["data_root"]))
        config["data_root"] = "/custom/missing-data"
        self.assertEqual(paths.data_root(config, root=root, env={}), Path("/custom/missing-data"))
        self.assertEqual(paths.data_root({}, root=root, env={"DATA_ROOT": paths.LEGACY_DEFAULTS["data_root"]}),
                         Path(paths.LEGACY_DEFAULTS["data_root"]))

    def test_python_symlink_keeps_venv_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = root / ".venv/bin/python"
            executable.parent.mkdir(parents=True)
            executable.symlink_to(sys.executable)
            self.assertEqual(paths.evaluator_python({"shared_eval_dir": str(root)}, root=root, env={}), executable)
            self.assertNotEqual(executable, executable.resolve())

    def test_evaluator_python_selection_has_no_validation_or_fallback(self):
        root = Path("/new/task/openvla-oft")
        with patch.object(Path, "exists", side_effect=AssertionError("No environment probe allowed")):
            self.assertEqual(paths.evaluator_python({"shared_eval_dir": "/shared/eval"}, root=root, env={}),
                             Path("/shared/eval/.venv/bin/python"))
            self.assertEqual(paths.evaluator_python({"unilip_python": "/missing/python"}, root=root, env={}),
                             Path("/missing/python"))
            env = {"CSGO_EVAL_PYTHON": "/unified/python", "UNILIP_PYTHON": "/legacy/python"}
            self.assertEqual(paths.evaluator_python({}, root=root, env=env), Path("/unified/python"))
            config = {"_path_overrides": {"unilip_python": "/cli/python"}}
            self.assertEqual(paths.evaluator_python(config, root=root, env=env), Path("/cli/python"))

    def test_all_entrypoints_print_paths_without_model_imports(self):
        import builtins
        original_import = builtins.__import__

        def guarded_import(name, *args, **kwargs):
            if name.split(".")[0] in {"torch", "prismatic"} or name == "csgo_seen10.runner":
                raise AssertionError(f"Path-only command imported model/runtime: {name}")
            return original_import(name, *args, **kwargs)

        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
                os.chdir(directory)
                for phase in ("train", "infer", "eval"):
                    entry = paths.PROJECT_ROOT / f"{phase}_seen10.py"
                    argv = [str(entry), "--config", "configs/csgo_seen10_aligned_v2.yaml",
                            "--seed", "42", "--print-paths", "--data-root", "/custom/data"]
                    output = io.StringIO()
                    with patch.object(sys, "argv", argv), patch("builtins.__import__", guarded_import), \
                            contextlib.redirect_stdout(output):
                        runpy.run_path(str(entry), run_name="__main__")
                    report = json.loads(output.getvalue())
                    self.assertEqual(report["execution"], "paths_only")
                    self.assertEqual(report["data_root"], "/custom/data")
                    self.assertEqual(report["evaluator_python"],
                                     str(paths.PROJECT_ROOT / "../csgo_benchmark_v2_eval_general/.venv/bin/python"))
                    self.assertTrue(report["run_dir"].endswith("OpenVLA-OFT/seed_42"))
                self.assertEqual(list(Path(directory).iterdir()), [])
        finally:
            os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
