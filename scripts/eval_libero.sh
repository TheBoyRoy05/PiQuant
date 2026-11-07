#!/usr/bin/env bash
# Evaluate pi0.5 on LIBERO. Linux with an NVIDIA GPU only: LIBERO does not install on macOS.
# Usage: scripts/eval_libero.sh [suite] [episodes per task]
set -euo pipefail

suite="${1:-libero_spatial}"
episodes="${2:-10}"

# Headless rendering for clusters (DSMLP, NRP) with no display.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# `<<< n` answers LIBERO's first-import prompt for a custom dataset path.
# The checkpoint enables torch.compile (max-autotune), which hit a CUDA illegal memory access on an A10; run eager.
# The checkpoint stores n_action_steps=50, but LeRobot's published LIBERO results replan every 10 steps, as OpenPI does.
uv run python scripts/record_token_counts.py \
  --policy.path="${POLICY:-lerobot/pi05_libero_finetuned}" \
  --policy.compile_model=false \
  --policy.n_action_steps=10 \
  --env.type=libero \
  --env.task="$suite" \
  --eval.batch_size=1 \
  --eval.n_episodes="$episodes" <<< n
