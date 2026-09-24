#!/usr/bin/env python3
"""Download the official OpenVLA base checkpoint into this checkout."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
configured_target = Path(os.environ.get("OPENVLA_MODEL_PATH", "checkpoints/openvla-7b")).expanduser()
TARGET = (configured_target if configured_target.is_absolute() else ROOT / configured_target).resolve()
REPO_ID = "openvla/openvla-7b"
REVISION = "main"


def inspect_local() -> None:
    if not TARGET.is_dir():
        raise SystemExit(f"[model] missing checkpoint directory: {TARGET}")
    config = TARGET / "config.json"
    if not config.is_file():
        raise SystemExit(f"[model] missing {config}")
    with config.open() as stream:
        json.load(stream)
    index = TARGET / "model.safetensors.index.json"
    if index.is_file():
        with index.open() as stream:
            weight_map = json.load(stream).get("weight_map", {})
        names = set(weight_map.values())
        if not names or not all(isinstance(name, str) for name in names):
            raise SystemExit(f"[model] invalid weight map in {index}")
        weights = [TARGET / name for name in sorted(names)]
    else:
        weights = [TARGET / "model.safetensors"]
    for weight in weights:
        if not weight.is_file():
            raise SystemExit(f"[model] missing weight file: {weight}")
        if not weight.stat().st_size:
            raise SystemExit(f"[model] empty weight file: {weight}")
    print(f"[model] local checkpoint present: {TARGET} ({len(weights)} weight files)")
    print("[model] --check verifies local presence only; downloads verify Hub sizes and LFS SHA-256.")


def verify_sha256(path: Path, oid: object) -> bool:
    if not isinstance(oid, str) or len(oid) != 64:
        return True
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == oid


def download(entry: dict, output: Path) -> None:
    size = entry.get("size")
    lfs_oid = (entry.get("lfs") or {}).get("oid")
    if (
        isinstance(size, int)
        and output.is_file()
        and output.stat().st_size == size
        and verify_sha256(output, lfs_oid)
    ):
        print(f"[model] already complete {entry['path']} ({size} bytes)", flush=True)
        return
    if output.is_file() and output.stat().st_size:
        print(
            f"[model] replacing incomplete materialized file {entry['path']} "
            f"({output.stat().st_size} bytes; expected {size})",
            flush=True,
        )
    partial = output.with_name(output.name + ".partial")
    url = (
        f"https://huggingface.co/{REPO_ID}/resolve/{REVISION}/"
        f"{urllib.parse.quote(entry['path'], safe='/')}?download=true"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    if (
        isinstance(size, int)
        and partial.is_file()
        and partial.stat().st_size == size
        and verify_sha256(partial, lfs_oid)
    ):
        os.replace(partial, output)
        partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
        print(f"[model] materialized complete partial {entry['path']} ({size} bytes)", flush=True)
        return
    if shutil.which("aria2c") and (not isinstance(size, int) or size >= 8 * 1024 * 1024):
        command = [
            "aria2c", "--continue=true", "--allow-overwrite=true",
            "--auto-file-renaming=false", "--file-allocation=none",
            "--max-connection-per-server=16", "--split=16",
            "--min-split-size=4M", "--summary-interval=10",
            "--console-log-level=notice", "--retry-wait=5", "--max-tries=0",
            "--connect-timeout=30", "--timeout=60", f"--out={partial.name}",
            f"--dir={partial.parent}", url,
        ]
    else:
        command = [
            "curl", "--fail", "--location", "--continue-at", "-",
            "--retry", "10", "--retry-delay", "5", "--retry-all-errors",
            "--connect-timeout", "30", "--output", str(partial), url,
        ]
    result = subprocess.run(command)
    if result.returncode and command[0] == "aria2c":
        print(
            f"[model] aria2c failed for {entry['path']} (rc={result.returncode}); "
            "retrying with curl resume",
            flush=True,
        )
        subprocess.run(
            [
                "curl", "--fail", "--location", "--continue-at", "-",
                "--retry", "10", "--retry-delay", "5", "--retry-all-errors",
                "--connect-timeout", "30", "--output", str(partial), url,
            ],
            check=True,
        )
    elif result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    if isinstance(size, int) and partial.stat().st_size != size:
        raise RuntimeError(
            f"incomplete download for {entry['path']}: "
            f"got {partial.stat().st_size}, expected {size}"
        )
    if not verify_sha256(partial, lfs_oid):
        raise RuntimeError(f"SHA-256 mismatch for {entry['path']}")
    os.replace(partial, output)
    partial.with_name(partial.name + ".aria2").unlink(missing_ok=True)
    print(f"[model] materialized {entry['path']} ({output.stat().st_size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true", help="show paths; no writes or network")
    modes.add_argument("--check", action="store_true", help="inspect local files; no writes or network")
    args = parser.parse_args()
    print(f"[model] {REPO_ID}@{REVISION} -> {TARGET}")
    if args.dry_run:
        print("[model] would query the official Hub manifest and resume/verify each file")
        return
    if args.check:
        inspect_local()
        return

    hf_home = ROOT / "checkpoints" / "huggingface"
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_home / "hub")
    os.environ["HF_XET_CACHE"] = str(hf_home / "xet")
    os.environ["TRANSFORMERS_CACHE"] = str(hf_home / "transformers")
    os.environ.setdefault("HF_XET_DISABLE", "1")
    for upper, lower in (("HTTPS_PROXY", "https_proxy"), ("HTTP_PROXY", "http_proxy")):
        if os.environ.get(upper) and not os.environ.get(lower):
            os.environ[lower] = os.environ[upper]
    TARGET.mkdir(parents=True, exist_ok=True)
    api_url = f"https://huggingface.co/api/models/{REPO_ID}/tree/{REVISION}?recursive=true"
    with urllib.request.urlopen(api_url, timeout=60) as response:
        entries = json.load(response)
    files = [entry for entry in entries if entry.get("type") == "file"]
    if not files:
        raise RuntimeError(f"Hugging Face returned no files for {REPO_ID}")
    for entry in files:
        output = (TARGET / entry["path"]).resolve()
        if not output.is_relative_to(TARGET):
            raise RuntimeError(f"unsafe path in Hub manifest: {entry['path']}")
        download(entry, output)
    print(f"[model] checkpoint ready: {TARGET}")


if __name__ == "__main__":
    main()
