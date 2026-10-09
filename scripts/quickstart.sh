#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

if [[ "${1:-}" == --help || "$#" != 2 ]]; then
  cat <<'EOF'
Usage: bash scripts/quickstart.sh ARM_CONFIG EPISODE_SPLIT_JSON
Requires the installed environment and GW_* asset/output paths from README.
Checks config, optionally prepares the explicit split, checks data, then trains
with the native synchronized entrypoint and reports heldout denoising proxies.

Environment: NPROC_PER_NODE=8, TRAIN_STEPS=2,
PREPARE=0 (set 1 to explicitly encode into a fresh GW_PREPARATION_ROOT),
PREPARE_DEVICE=cpu, CHECK_ONLY=0 (set 1 to stop before GPU/model construction),
RESUME_CHECKPOINT= (matching trusted full-state checkpoint directory or file).
No models or datasets are downloaded by this script.
EOF
  [[ "${1:-}" == --help ]] && exit 0
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
config="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$1")"
split="$(python -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).resolve())' "$2")"
cd "$repo_root"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -m gradientwam.runner check-config --config "$config"
if [[ "${PREPARE:-0}" == 1 ]]; then
  python -m gradientwam.runner prepare --config "$config" --episodes-file "$split" \
    --device "${PREPARE_DEVICE:-cpu}" --execute
fi
if [[ -z "${GW_PROMPT_FINGERPRINT:-}" ]]; then
  : "${GW_PREPARATION_ROOT:?Set GW_PREPARATION_ROOT to the shared prepared cache}"
  if [[ ! -f "$GW_PREPARATION_ROOT/metadata/preparation.json" ]]; then
    echo 'Missing preparation metadata: prepare explicitly with PREPARE=1, or supply a completed cache and GW_PROMPT_FINGERPRINT.' >&2
    exit 2
  fi
  GW_PROMPT_FINGERPRINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt_encoder_fingerprint"])' \
    "$GW_PREPARATION_ROOT/metadata/preparation.json")"
  export GW_PROMPT_FINGERPRINT
fi
python -m gradientwam.runner check-data --config "$config" --episodes-file "$split"
[[ "${CHECK_ONLY:-0}" == 1 ]] && exit 0

run_config="$(mktemp "${TMPDIR:-/tmp}/gradientwam-run.XXXXXX.yaml")"
trap 'rm -f -- "$run_config"' EXIT
python - "$run_config" "$config" "$split" <<'PY'
import os
from pathlib import Path
import sys
import yaml

steps = int(os.environ.get('TRAIN_STEPS', '2'))
if steps <= 0:
    raise SystemExit('TRAIN_STEPS must be a positive integer.')
Path(sys.argv[1]).write_text(yaml.safe_dump({
    'settings_config': sys.argv[2],
    'episode_split_json': sys.argv[3],
    'steps': steps,
    'eval_seed': 20261009,
}), encoding='utf-8')
PY
RUN_CONFIG="$run_config" NPROC_PER_NODE="${NPROC_PER_NODE:-8}" \
  bash scripts/train_8gpu.sh
