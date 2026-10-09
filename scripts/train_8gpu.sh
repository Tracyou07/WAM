#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_config="${RUN_CONFIG:-${repo_root}/configs/gradientwam_distributed.example.yaml}"
nproc_per_node="${NPROC_PER_NODE:-8}"

if [[ ! "${nproc_per_node}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer" >&2
  exit 2
fi
if [[ ! -f "${run_config}" ]]; then
  echo "Run config not found: ${run_config}" >&2
  exit 2
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
args=(--run-config "${run_config}")
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  args+=(--resume "${RESUME_CHECKPOINT}")
fi

exec torchrun --standalone --nnodes=1 --nproc-per-node="${nproc_per_node}" \
  -m gradientwam.distributed_train "${args[@]}"
