#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
NPROC="${NPROC_PER_NODE:-1}"
TORCHRUN=("${PYTHON_BIN}" -m torch.distributed.run)

usage() {
    echo "Usage: $0 {smoke|train|infer|eval} [--config PATH] [--seed N] [options]" >&2
    exit 2
}

[[ $# -ge 1 ]] || usage
COMMAND="$1"
shift
export OPENVLA_ROBOT_PLATFORM="${OPENVLA_ROBOT_PLATFORM:-CSGO}"

case "${COMMAND}" in
    smoke)
        # Smoke is intentionally a separate output tree and uses one process;
        # it still loads the real pretrained snapshot and performs fwd/bwd,
        # checkpoint save/reload, prediction and shared-evaluator validation.
        "${PYTHON_BIN}" train_seen10.py --smoke "$@"
        "${PYTHON_BIN}" infer_seen10.py --smoke "$@"
        "${PYTHON_BIN}" eval_seen10.py --smoke "$@"
        ;;
    train)
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" train_seen10.py "$@"
        ;;
    infer)
        "${TORCHRUN[@]}" --standalone --nproc_per_node="${NPROC}" infer_seen10.py "$@"
        ;;
    eval)
        "${PYTHON_BIN}" eval_seen10.py "$@"
        ;;
    *)
        usage
        ;;
esac
