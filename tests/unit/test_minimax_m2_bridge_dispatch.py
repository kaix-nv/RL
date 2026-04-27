"""CI-friendly version of tools/preflight_minimax_m2_dispatch.py."""

import pytest
from transformers import AutoConfig
from megatron.bridge.models.conversion.model_bridge import get_model_bridge


MODEL_ID = "MiniMaxAI/MiniMax-M2.7"


@pytest.mark.mcore
def test_minimax_m2_dispatches_to_minimax_m2_bridge() -> None:
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    assert cfg.architectures
    arch = cfg.architectures[0]
    assert arch == "MiniMaxM2ForCausalLM"

    bridge = get_model_bridge(arch, hf_config=cfg)
    assert type(bridge).__name__ == "MiniMaxM2Bridge"
