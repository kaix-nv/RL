#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run a command inside the nvcr.io/nvidia/nemo-rl:v0.5.0 image with our local
# recipes mounted in. Designed for LoRA + RL smoke runs on the local A6000 host.
#
# Usage:
#   bash tools/run_in_nemo_rl_container.sh <gpu-spec> <cmd...>
# Example:
#   bash tools/run_in_nemo_rl_container.sh 'device=0,1' \
#     uv run python examples/run_grpo.py \
#     --config examples/configs/recipes/llm/grpo-qwen3-8b-base-1n2g-megatron-lora-smoke.yaml

set -euo pipefail

IMAGE="${IMAGE:-nvcr.io/nvidia/nemo-rl:v0.5.0}"
WORKTREE="${WORKTREE:-/home/scratch.kaix_coreai/workspace/rl/RL/.worktrees/feat-minimax-m2.7-pipeclean}"
# Local (non-NFS) paths so the container's root user can write without
# tripping NFS root_squash on /home/scratch.kaix_coreai.
HF_CACHE="${HF_CACHE:-/home/kaix/nemo-rl-runs/hf-cache}"
RESULTS="${RESULTS:-/home/kaix/nemo-rl-runs/results}"
LOGS="${LOGS:-/home/kaix/nemo-rl-runs/logs}"

GPUS="${1:?gpu spec required, e.g. 'device=0' or 'all'}"
shift

mkdir -p "$HF_CACHE" "$RESULTS" "$LOGS"

exec docker run --rm \
  --gpus "\"$GPUS\"" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_USE_V1=0 \
  -v "$WORKTREE/examples/configs/recipes/llm":/opt/nemo-rl/examples/configs/recipes/llm \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$RESULTS":/opt/nemo-rl/results \
  -v "$LOGS":/opt/nemo-rl/logs \
  -w /opt/nemo-rl \
  "$IMAGE" \
  bash -c "$*"
