#!/usr/bin/env bash
set -euo pipefail

# Project-local setup for OpenVLA-OFT CSGO Benchmark v2 Seen-10.  This keeps
# Python packages and Hugging Face artifacts out of the UniLIP environment.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
CHECKPOINT_ROOT="${PROJECT_ROOT}/checkpoints"
HF_HOME_DIR="${CHECKPOINT_ROOT}/huggingface"
MODEL_DIR="${CHECKPOINT_ROOT}/openvla-7b"
if [[ -n "${OPENVLA_MODEL_PATH:-}" ]]; then
    case "${OPENVLA_MODEL_PATH}" in
        /*) MODEL_DIR="${OPENVLA_MODEL_PATH}" ;;
        "~") MODEL_DIR="${HOME}" ;;
        "~/"*) MODEL_DIR="${HOME}/${OPENVLA_MODEL_PATH#\~/}" ;;
        *) MODEL_DIR="${PROJECT_ROOT}/${OPENVLA_MODEL_PATH}" ;;
    esac
fi
REQ_FILE="${PROJECT_ROOT}/requirements-csgo-seen10.txt"
SETUP_LOG_DIR="${PROJECT_ROOT}/outputs/setup_logs"
PYPI_INDEX_URL="${CSGO_PYPI_INDEX_URL:-https://pypi.org/simple}"
PYTORCH_INDEX_URL="${CSGO_PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"

usage() {
    cat <<'EOF'
Usage: scripts/setup_csgo_seen10.sh [--skip-model] [--dry-run | --check] [--help]

Create a project-local Python 3.11 environment, install pinned CUDA 12.8
dependencies, download openvla/openvla-7b, and validate imports/CUDA.
Set OPENVLA_SETUP_PYTHON to an installed Python 3.11 interpreter.
--skip-model leaves the download to scripts/download_csgo_model.py.
--dry-run shows planned actions without writes or network access.
--check inspects the existing environment and local model without writes,
network access, or CUDA initialization.
EOF
}

SKIP_MODEL=0
DRY_RUN=0
CHECK_ONLY=0
for arg in "$@"; do
    case "${arg}" in
        --skip-model) SKIP_MODEL=1 ;;
        --dry-run) DRY_RUN=1 ;;
        --check) CHECK_ONLY=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown argument: ${arg}" >&2; usage >&2; exit 2 ;;
    esac
done
if (( DRY_RUN && CHECK_ONLY )); then
    echo "Choose either --dry-run or --check." >&2
    exit 2
fi

python311() {
    "$1" -c 'import sys; print(sys.version.split()[0] if sys.version_info[:2] == (3, 11) else "")' 2>/dev/null
}

validate_python() {
    local candidate="$1" version
    if ! command -v "${candidate}" >/dev/null 2>&1; then
        echo "Python interpreter not found: ${candidate}. Set OPENVLA_SETUP_PYTHON to Python 3.11." >&2
        return 1
    fi
    version="$(python311 "${candidate}")" || true
    if [[ -z "${version}" ]]; then
        echo "${candidate} is not a working Python 3.11 interpreter." >&2
        return 1
    fi
    echo "${version}"
}

check_venv() {
    if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
        echo "Existing ${VENV_DIR} has no executable bin/python. It was preserved; repair or move it before retrying." >&2
        return 1
    fi
    "${VENV_DIR}/bin/python" - "${VENV_DIR}" <<'PY'
import sys
from pathlib import Path

expected = Path(sys.argv[1]).resolve()
if Path(sys.prefix).resolve() != expected:
    raise SystemExit(f"Existing .venv points to {sys.prefix}, expected {expected}; repair or move it before retrying.")
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Existing .venv uses Python {sys.version.split()[0]}, expected 3.11.")
cfg = expected / "pyvenv.cfg"
conda_meta = expected / "conda-meta"
if cfg.is_file():
    if sys.prefix == sys.base_prefix:
        raise SystemExit("Existing .venv is not isolated; repair or move it before retrying.")
    if "include-system-site-packages = false" not in cfg.read_text().lower():
        raise SystemExit("Existing .venv exposes system site-packages; repair or move it before retrying.")
    for script in (expected / "bin").glob("pip*"):
        if script.is_file():
            with script.open("rb") as stream:
                first_line = stream.readline(4096).decode("utf-8", "replace").strip()
            if first_line.startswith("#!") and str(expected / "bin") not in first_line:
                raise SystemExit(f"{script} points outside this .venv ({first_line}); repair or recreate the moved environment.")
    activate = expected / "bin" / "activate"
    if activate.is_file():
        for line in activate.read_text().splitlines():
            if line.startswith("VIRTUAL_ENV=") and str(expected) not in line:
                raise SystemExit(f"{activate} points to an old location; repair or recreate the moved environment.")
elif not conda_meta.is_dir():
    raise SystemExit("Existing .venv has neither pyvenv.cfg nor conda-meta; repair or move it before retrying.")
print(f"[setup] isolated Python {sys.version.split()[0]} at {sys.prefix}")
PY
}

SELECTED_PYTHON=""
if [[ -n "${OPENVLA_SETUP_PYTHON:-}" ]]; then
    validate_python "${OPENVLA_SETUP_PYTHON}" >/dev/null
    SELECTED_PYTHON="${OPENVLA_SETUP_PYTHON}"
fi
if [[ -e "${VENV_DIR}" || -L "${VENV_DIR}" ]]; then
    check_venv
else
    if (( CHECK_ONLY )); then
        echo "[setup] missing environment: ${VENV_DIR}" >&2
        exit 1
    fi
    if [[ -z "${SELECTED_PYTHON}" ]]; then
        for candidate in python3.11 python3 python; do
            if command -v "${candidate}" >/dev/null 2>&1 && [[ -n "$(python311 "${candidate}" || true)" ]]; then
                SELECTED_PYTHON="$(command -v "${candidate}")"
                break
            fi
        done
    fi
    if [[ -z "${SELECTED_PYTHON}" ]]; then
        if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
            CONDA_BIN="${CONDA_EXE}"
        else
            CONDA_BIN="$(command -v conda || true)"
        fi
        if [[ -z "${CONDA_BIN}" ]]; then
            echo "Python 3.11 or Conda is required. Set OPENVLA_SETUP_PYTHON to a Python 3.11 interpreter." >&2
            exit 1
        fi
    fi
fi

echo "[setup] root: ${PROJECT_ROOT}"
echo "[setup] environment: ${VENV_DIR}"
echo "[setup] Hugging Face cache: ${HF_HOME_DIR}"
echo "[setup] model: ${MODEL_DIR}"
if (( DRY_RUN )); then
    if [[ ! -e "${VENV_DIR}" && ! -L "${VENV_DIR}" ]]; then
        if [[ -n "${SELECTED_PYTHON}" ]]; then
            echo "[setup] would create Python 3.11 venv with ${SELECTED_PYTHON}"
        else
            echo "[setup] would create Python 3.11 Conda prefix with ${CONDA_BIN}"
        fi
    fi
    echo "[setup] would install pinned torch==2.7.1+cu128, torchvision==0.22.1+cu128, torchaudio==2.7.1+cu128 and ${REQ_FILE}"
    if (( ! SKIP_MODEL )); then
        if [[ -e "${VENV_DIR}" || -L "${VENV_DIR}" ]]; then
            "${VENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/download_csgo_model.py" --dry-run
        elif [[ -n "${SELECTED_PYTHON}" ]]; then
            "${SELECTED_PYTHON}" "${PROJECT_ROOT}/scripts/download_csgo_model.py" --dry-run
        else
            echo "[model] would query the official Hub manifest and resume/verify files into ${MODEL_DIR}"
        fi
    fi
    echo "[setup] would validate imports and CUDA"
    exit 0
fi

if (( CHECK_ONLY )); then
    "${VENV_DIR}/bin/python" - "${PROJECT_ROOT}" <<'PY'
from importlib import metadata
import json
import site
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

pins = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "torchaudio": "2.7.1+cu128"}
for name, expected in pins.items():
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        raise SystemExit(f"[setup] missing package: {name}")
    if actual != expected:
        raise SystemExit(f"[setup] {name} is {actual}, expected {expected}")
    print(f"[setup] {name}=={actual}")
for name in ("openvla-oft", "transformers", "peft", "accelerate", "diffusers", "PyYAML", "numpy", "Pillow"):
    try:
        print(f"[setup] {name}=={metadata.version(name)}")
    except metadata.PackageNotFoundError:
        raise SystemExit(f"[setup] missing package: {name}")
distribution = next(
    (
        dist for dist in metadata.distributions(path=site.getsitepackages())
        if dist.metadata["Name"].lower().replace("_", "-") == "openvla-oft"
    ),
    None,
)
if distribution is None:
    raise SystemExit("[setup] openvla-oft is missing from this environment.")
direct_url = distribution.read_text("direct_url.json")
if not direct_url:
    raise SystemExit("[setup] openvla-oft lacks editable install metadata; reinstall this checkout.")
source = json.loads(direct_url)
url = urlparse(source.get("url", ""))
if not source.get("dir_info", {}).get("editable") or url.scheme != "file":
    raise SystemExit("[setup] openvla-oft is not installed editable from this checkout.")
installed_path = Path(unquote(url.path)).resolve()
if installed_path != Path(sys.argv[1]).resolve():
    raise SystemExit(f"[setup] openvla-oft points to {installed_path}; reinstall the moved checkout.")
PY
    if (( ! SKIP_MODEL )); then
        "${VENV_DIR}/bin/python" "${PROJECT_ROOT}/scripts/download_csgo_model.py" --check
    fi
    echo "[setup] checks passed"
    exit 0
fi

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

if [[ ! -e "${VENV_DIR}" && ! -L "${VENV_DIR}" ]]; then
    if [[ -n "${SELECTED_PYTHON}" ]]; then
        echo "[setup] creating project-local venv ${VENV_DIR} with ${SELECTED_PYTHON}"
        "${SELECTED_PYTHON}" -m venv --without-pip "${VENV_DIR}"
        check_venv
        curl --fail --silent --show-error https://bootstrap.pypa.io/get-pip.py | "${VENV_DIR}/bin/python" -
    else
        echo "[setup] creating project-local Conda prefix ${VENV_DIR} (Python 3.11)"
        "${CONDA_BIN}" create --prefix "${VENV_DIR}" python=3.11 pip -y
        check_venv
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

if (( ! SKIP_MODEL )); then
    "${PYTHON}" "${PROJECT_ROOT}/scripts/download_csgo_model.py"
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
