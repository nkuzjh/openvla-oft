#!/usr/bin/env bash
set -euo pipefail

# Project-local setup for OpenVLA-OFT CSGO Benchmark v2 Seen-10.  This keeps
# Python packages and Hugging Face artifacts out of the UniLIP environment.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
HF_HOME_DIR="${CHECKPOINT_ROOT}/huggingface"
MODEL_DIR="${CHECKPOINT_ROOT}/openvla-7b"
REQ_FILE="${PROJECT_ROOT}/requirements-csgo-seen10.txt"
SETUP_LOG_DIR="${PROJECT_ROOT}/outputs/setup_logs"
PYPI_INDEX_URL="${CSGO_PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${CSGO_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
# This setup entry point is specific to the CSGO Seen-10 native path.  Keep
# the opt-in platform explicit even when it is invoked directly, so the
# validation import exercises the same five-dimensional action contract used
# by train_seen10.py/infer_seen10.py.
export OPENVLA_ROBOT_PLATFORM="${OPENVLA_ROBOT_PLATFORM:-CSGO}"
export HF_HOME="${HF_HOME_DIR}"
export HF_HUB_CACHE="${HF_HOME_DIR}/hub"
export HF_XET_CACHE="${HF_HOME_DIR}/xet"
export TRANSFORMERS_CACHE="${HF_HOME_DIR}/transformers"
export PIP_CACHE_DIR="${PROJECT_ROOT}/.cache/pip"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"
# Keep Xet's resumable chunk cache with this project and allow enough range
# requests to use the available network bandwidth behind the benchmark proxy.
export HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY="${HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY:-32}"
export HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY="${HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY:-64}"
export HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS="${HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS:-3}"
# The Rust Xet client reads lowercase proxy variables on some versions, while
# pip/curl commonly receive only the uppercase forms from the host.
if [[ -n "${HTTPS_PROXY:-}" && -z "${https_proxy:-}" ]]; then
    export https_proxy="${HTTPS_PROXY}"
fi
if [[ -n "${HTTP_PROXY:-}" && -z "${http_proxy:-}" ]]; then
    export http_proxy="${HTTP_PROXY}"
fi

mkdir -p "${CHECKPOINT_ROOT}" "${HF_HOME_DIR}" "${PIP_CACHE_DIR}" "${SETUP_LOG_DIR}"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    PYTHON311="$(command -v python3.11 || true)"
    # The benchmark host exposes the supported 3.11 interpreter through the
    # reference conda environment.  venv isolates site-packages, so nothing is
    # installed into that environment; this is only a Python runtime fallback.
    if [[ -z "${PYTHON311}" && -n "${UNILIP_PYTHON:-}" && -x "${UNILIP_PYTHON}" ]]; then
        PYTHON311="${UNILIP_PYTHON}"
    fi
    if [[ -z "${PYTHON311}" && -x "/home/jiahao/miniconda3/envs/UniLIP/bin/python" ]]; then
        PYTHON311="/home/jiahao/miniconda3/envs/UniLIP/bin/python"
    fi

    if [[ -n "${PYTHON311}" ]]; then
        echo "[setup] creating project-local venv ${VENV_DIR} with $(${PYTHON311} -c 'import sys; print(sys.version.split()[0])')"
        "${PYTHON311}" -m venv --without-pip "${VENV_DIR}"
        curl --fail --silent --show-error https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python" -
    else
        if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
            CONDA_BIN="${CONDA_EXE}"
        elif command -v conda >/dev/null 2>&1; then
            CONDA_BIN="$(command -v conda)"
        else
            CONDA_BIN=""
        fi
        if [[ -z "${CONDA_BIN}" ]]; then
            echo "[setup] Python 3.11 or conda is required to create ${VENV_DIR}" >&2
            exit 1
        fi
        echo "[setup] creating project-local conda prefix ${VENV_DIR} (Python 3.11)"
        "${CONDA_BIN}" create --prefix "${VENV_DIR}" python=3.11 pip -y
    fi
fi

PYTHON="${VENV_DIR}/bin/python"
echo "[setup] interpreter: $(${PYTHON} -c 'import sys; print(sys.executable)')"
echo "[setup] upgrading packaging tools from ${PYPI_INDEX_URL}"
"${PYTHON}" -m pip install --index-url "${PYPI_INDEX_URL}" --upgrade pip setuptools wheel
echo "[setup] installing CUDA 12.8 torch wheels from ${PYTORCH_INDEX_URL}"
# Keep the CUDA packages in a separate pip transaction.  The requirements file
# is intentionally PyPI-only; passing the CUDA index globally makes pip query
# it for every unrelated dependency and can leave a half-installed venv.
"${PYTHON}" -m pip install \
    --index-url "${PYTORCH_INDEX_URL}" \
    --extra-index-url "${PYPI_INDEX_URL}" \
    "torch==2.7.1+cu128" "torchvision==0.22.1+cu128" "torchaudio==2.7.1+cu128"
echo "[setup] installing runtime dependencies from ${PYPI_INDEX_URL}"
"${PYTHON}" -m pip install --index-url "${PYPI_INDEX_URL}" -r "${REQ_FILE}"
# pyproject.toml includes optional RLDS/TensorFlow dependencies for the
# original robot workloads.  Install this checkout editable without pulling
# those unrelated dependencies; the requirements file above is the CSGO set.
"${PYTHON}" -m pip install --no-deps --editable "${PROJECT_ROOT}"

if [[ "${1:-}" != "--skip-model" ]]; then
    echo "[setup] downloading openvla/openvla-7b into ${MODEL_DIR}"
    # Prefer resumable range downloads through the official Hub resolver.  The
    # direct HTTP path also works when the optional Xet client is unavailable.
    HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}" \
        "${PYTHON}" - "${MODEL_DIR}" <<'PY'
import json
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

target = Path(sys.argv[1]).resolve()
target.mkdir(parents=True, exist_ok=True)
repo_id = "openvla/openvla-7b"
revision = "main"
api_url = f"https://huggingface.co/api/models/{repo_id}/tree/{revision}?recursive=true"
with urllib.request.urlopen(api_url, timeout=60) as response:
    entries = json.load(response)
files = [entry for entry in entries if entry.get("type") == "file"]
if not files:
    raise RuntimeError("Hugging Face returned no files for openvla/openvla-7b")

def download(entry: dict, output: Path) -> None:
    size = entry.get("size")
    lfs_oid = (entry.get("lfs") or {}).get("oid")

    def verify_sha256(path: Path) -> bool:
        if not isinstance(lfs_oid, str) or len(lfs_oid) != 64:
            return True
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == lfs_oid

    if (
        isinstance(size, int)
        and output.is_file()
        and output.stat().st_size == size
        and verify_sha256(output)
    ):
        print(f"[setup] already complete {entry['path']} ({size} bytes)", flush=True)
        return
    if output.is_file() and output.stat().st_size:
        print(
            f"[setup] replacing incomplete materialized file {entry['path']} "
            f"({output.stat().st_size} bytes; expected {size})",
            flush=True,
        )
    partial = output.with_name(output.name + ".partial")
    url = (
        f"https://huggingface.co/{repo_id}/resolve/{revision}/"
        f"{urllib.parse.quote(entry['path'], safe='/')}?download=true"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # A previous process may have finished the materialized bytes but exited
    # before the atomic rename.  Verify and promote that file directly instead
    # of asking the Hub to serve the already-complete shard again.
    if (
        isinstance(size, int)
        and partial.is_file()
        and partial.stat().st_size == size
        and verify_sha256(partial)
    ):
        os.replace(partial, output)
        aria2_control = partial.with_name(partial.name + ".aria2")
        aria2_control.unlink(missing_ok=True)
        print(f"[setup] materialized complete partial {entry['path']} ({size} bytes)", flush=True)
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
            f"[setup] aria2c failed for {entry['path']} (rc={result.returncode}); "
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
    if not verify_sha256(partial):
        raise RuntimeError(f"SHA-256 mismatch for {entry['path']}")
    os.replace(partial, output)
    aria2_control = partial.with_name(partial.name + ".aria2")
    aria2_control.unlink(missing_ok=True)
    print(f"[setup] materialized {entry['path']} ({output.stat().st_size} bytes)", flush=True)

for entry in files:
    download(entry, target / entry["path"])
print(f"[setup] checkpoint ready: {target}")
PY
fi

echo "[setup] validating project-local imports and CUDA"
"${PYTHON}" - <<'PY'
import os
import sys

import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__} cuda={torch.version.cuda}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable in the project environment")
device = torch.device("cuda:0")
print(f"gpu={torch.cuda.get_device_name(device)} capability={torch.cuda.get_device_capability(device)}")
x = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
y = torch.randn((256, 256), device=device, dtype=torch.bfloat16)
z = x @ y
torch.cuda.synchronize()
print(f"bf16_matmul={tuple(z.shape)}")

import prismatic
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from transformers import AutoProcessor

print(f"prismatic={prismatic.__file__}")
print(f"openvla_config={OpenVLAConfig.__name__} model={OpenVLAForActionPrediction.__name__}")
print(f"transformers={__import__('transformers').__version__}")
print(f"hf_home={os.environ.get('HF_HOME')}")
PY

echo "[setup] complete"
