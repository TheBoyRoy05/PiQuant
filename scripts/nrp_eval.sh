#!/usr/bin/env bash
# Run one LIBERO eval inside the NRP job (k8s/eval-libero.yaml) and copy outputs to the data volume.
# Usage: scripts/nrp_eval.sh <suite> <episodes per task>
set -euo pipefail

suite="${1:?suite required}"
episodes="${2:?episodes per task required}"

# cmake builds egl_probe from source; the libs back headless MuJoCo EGL rendering and OpenCV.
apt-get update -qq
apt-get install -y -qq --no-install-recommends cmake libegl1 libgl1 libglib2.0-0 > /dev/null

uv sync --frozen

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Copy on any exit, so a failed eval still keeps its partial token counts and eval outputs.
results="/data/results/$suite-$(date -u +%Y%m%dT%H%M%SZ)"
trap 'mkdir -p "$results" outputs; cp -r outputs/. "$results/"; echo "results copied to $results"' EXIT

scripts/eval_libero.sh "$suite" "$episodes"
