#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
DEVICE=cu128
VENV="$ROOT/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3.12}"
DRY_RUN=0

usage() {
    cat <<'EOF'
Usage: bash scripts/setup_env.sh [--device cpu|cu128] [--venv PATH] [--python PATH] [--dry-run]
Requires Linux x86_64, glibc >= 2.28, and CPython 3.12.x (reference patch: 3.12.2).
Installs fixed training/preparation dependencies and the editable checkout.
Defaults: --device cu128 --venv .venv --python python3.12 (or PYTHON_BIN).
No model, tokenizer, or dataset downloads. --dry-run only prints the install plan.
EOF
}
while (($#)); do
    case "$1" in
        --device|--venv|--python)
            (($# >= 2)) || { usage >&2; exit 2; }
            case "$1" in
                --device) DEVICE="$2" ;;
                --venv) VENV="$2" ;;
                --python) PYTHON_BIN="$2" ;;
            esac
            shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
case "$DEVICE" in cpu|cu128) ;; *) usage >&2; exit 2 ;; esac
[[ -n "$VENV" && -n "$PYTHON_BIN" ]] || { usage >&2; exit 2; }
REQ="$ROOT/requirements/reproduce-linux.txt"
TORCH_VERSION="2.11.0+$DEVICE"
TORCH_INDEX="https://download.pytorch.org/whl/$DEVICE"
PY="$VENV/bin/python"

plan() {
    printf 'Python: %s (required 3.12.x)\nEnvironment: %s\n' "$PYTHON_BIN" "$VENV"
    printf '%q ' "$PYTHON_BIN" -m venv "$VENV"; printf '\n'
    printf '%q ' "$PY" -m pip install pip==24.0; printf '\n'
    printf '%q ' "$PY" -m pip install --only-binary=:all: --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -c "$REQ" "torch==$TORCH_VERSION"; printf '\n'
    printf '%q ' "$PY" -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r "$REQ"; printf '\n'
    printf '%q ' "$PY" -m pip install --no-deps --no-build-isolation -e "$ROOT"; printf '\n'
}
if ((DRY_RUN)); then plan; exit 0; fi

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
    echo 'This fixed environment supports Linux x86_64 only.' >&2; exit 1;
}
check_python() {
    "$1" - <<'PY'
import platform, sys
if sys.version_info[:2] != (3, 12) or platform.python_implementation() != 'CPython':
    raise SystemExit('Use CPython 3.12.x; see docs/reproduction_setup.md.')
libc, version = platform.libc_ver()
if libc != 'glibc' or tuple(map(int, version.split('.')[:2])) < (2, 28):
    raise SystemExit('Linux glibc >= 2.28 is required by the fixed wheels.')
PY
}
if [[ -e "$VENV" ]]; then
    [[ -f "$VENV/pyvenv.cfg" && -x "$PY" ]] || {
        echo 'Choose a fresh --venv directory or an existing virtual environment.' >&2; exit 1;
    }
    check_python "$PY"
    "$PY" - "$TORCH_VERSION" <<'PY'
import importlib.metadata as m, sys
try:
    installed = m.version('torch')
except m.PackageNotFoundError:
    installed = sys.argv[1]
if installed != sys.argv[1]:
    raise SystemExit('Existing torch differs from this profile; use a separate --venv directory.')
PY
else
    check_python "$PYTHON_BIN"
    "$PYTHON_BIN" -m venv "$VENV"
fi
"$PY" -m pip install --index-url https://pypi.org/simple pip==24.0
"$PY" -m pip install --only-binary=:all: --index-url "$TORCH_INDEX" --extra-index-url https://pypi.org/simple -c "$REQ" "torch==$TORCH_VERSION"
"$PY" -m pip install --only-binary=:all: --index-url https://pypi.org/simple -r "$REQ"
"$PY" -m pip install --no-deps --no-build-isolation -e "$ROOT"
"$PY" -m pip check
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$PY" - "$REQ" "$TORCH_VERSION" <<'PY'
import importlib.metadata as m
from pathlib import Path
import sys
import av, cv2, h5py, numpy, pyarrow, torch
from diffusers import AutoencoderKLWan
from transformers import AutoTokenizer, UMT5Config, UMT5EncoderModel
import open_wam, gradientwam
for line in Path(sys.argv[1]).read_text().splitlines():
    if line and not line.startswith('#'):
        name, expected = line.split('==')
        if m.version(name) != expected:
            raise SystemExit(f'Version mismatch: {name}')
if m.version('torch') != sys.argv[2] or torch.cuda.is_initialized():
    raise SystemExit('Unexpected torch profile or CUDA initialization during import check.')
print(f'Import/version check passed: torch={torch.__version__}, compiled CUDA={torch.version.cuda}; no model constructed.')
PY
cat <<EOF
Environment ready: $VENV
Activate: source "$VENV/bin/activate"
Next, explicitly prepare assets as described in docs/reproduction_setup.md:
  1. OpenWAM video-only model_state.pt (not a pretrained action policy).
  2. Wan2.2 VAE, UMT5 encoder, and matching tokenizer; convert frontends on CPU.
  3. The pinned public LeRobot v2.1 dataset and an explicit train/heldout split.
  4. Direct asset/output paths, offline latents and prompt cache, then config/data checks.
CPU engineering checks (no asset download):
  CUDA_VISIBLE_DEVICES='' "$PY" -m pytest tests/test_gradientwam_delivery.py tests/test_gradientwam_distributed.py
Installation does not validate GPU capacity, eight-GPU training, or rollout quality.
EOF
