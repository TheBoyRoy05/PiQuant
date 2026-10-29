#!/usr/bin/env bash
# Evaluate pi0.5 on LIBERO. Linux with an NVIDIA GPU only: LIBERO does not install on macOS.
# Usage: scripts/eval_libero.sh [suite] [episodes per task]
set -euo pipefail

suite="${1:-libero_spatial}"
episodes="${2:-10}"

# Headless rendering for clusters (DSMLP, NRP) with no display.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

uv run lerobot-eval \
  --policy.path="${POLICY:-lerobot/pi05_libero_finetuned}" \
  --env.type=libero \
  --env.task="$suite" \
  --eval.batch_size=1 \
  --eval.n_episodes="$episodes"
