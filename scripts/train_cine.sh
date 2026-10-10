#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

if [[ "${1:-}" == --help || "$#" != 1 ]]; then
  cat <<'EOF'
Usage: bash scripts/train_cine.sh CINE_CONFIG.yaml
Use gradientwam-cine prepare first; this launcher does not encode or download assets.
Environment: NPROC_PER_NODE=8, CHECK_ONLY=0, RESUME_CHECKPOINT=.
Select free devices with CUDA_VISIBLE_DEVICES. Set run.steps in the config explicitly.
EOF
  [[ "${1:-}" == --help ]] && exit 0
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
config="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1")"
cd "$repo_root"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m gradientwam.cine_runner check-config --config "$config"
python -m gradientwam.cine_runner check-data --config "$config" --sample-limit 1
[[ "${CHECK_ONLY:-0}" == 1 ]] && exit 0

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  resume_args=(--resume "$RESUME_CHECKPOINT")
fi
torchrun --standalone --nnodes=1 --nproc-per-node="${NPROC_PER_NODE:-8}" \
  -m gradientwam.cine_runner train --config "$config" "${resume_args[@]}"
