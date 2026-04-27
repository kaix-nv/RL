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

# Pre-flight: verify vLLM can serve MiniMax-M2 with TP=4.
# Usage: bash tools/preflight_vllm_minimax_m2.sh [PORT]
# Requires 4x H100/H200 on one node.

set -euo pipefail
PORT="${1:-8765}"
MODEL="MiniMaxAI/MiniMax-M2.7"

echo "[preflight] starting vLLM on port $PORT ..."
uv run vllm serve "$MODEL" \
  --tensor-parallel-size 4 \
  --trust-remote-code \
  --port "$PORT" &
SERVER_PID=$!

cleanup() {
  echo "[preflight] stopping server (pid=$SERVER_PID)"
  kill "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[preflight] waiting for ready (up to 30 min)..."
for i in $(seq 1 360); do
  if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
    echo "[preflight] ready after ~$((i*5))s"; break
  fi
  sleep 5
done

echo "[preflight] test completion..."
curl -sf "http://localhost:$PORT/v1/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"2 + 2 =\", \"max_tokens\": 4}" \
  | tee /tmp/vllm_preflight.json
echo
echo "[preflight] OK"
