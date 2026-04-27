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

"""Verify MiniMax-M2 HF config dispatches to MiniMaxM2Bridge.

Inspects only HF config and the bridge registry — does not download
or load model weights. Runs in seconds on CPU.
"""

from transformers import AutoConfig
from megatron.bridge.models.conversion.model_bridge import get_model_bridge


MODEL_ID = "MiniMaxAI/MiniMax-M2.7"


def main() -> None:
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    assert cfg.architectures, f"HF config missing architectures: {cfg}"
    arch = cfg.architectures[0]
    print(f"HF architectures[0] = {arch}")
    print(f"HF model_type      = {cfg.model_type}")

    bridge = get_model_bridge(arch, hf_config=cfg)
    bridge_name = type(bridge).__name__
    print(f"Resolved bridge    = {bridge_name}")
    assert bridge_name == "MiniMaxM2Bridge", (
        f"Expected MiniMaxM2Bridge, got {bridge_name}. "
        "megatron-bridge may be out of date."
    )
    print("OK: MiniMax-M2 dispatches to MiniMaxM2Bridge.")


if __name__ == "__main__":
    main()
