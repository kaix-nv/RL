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

"""Unit tests for dequantizing quantized HF checkpoints in the bridge loader."""

import pytest
import torch

from nemo_rl.models.megatron.modelopt_hf_dequant import (
    dequantize_fp8_blockwise,
    dequantize_loaded_weight,
    dequantize_nvfp4,
    install_dequantizing_loader,
)

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _nvfp4_case(rows=3, k=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 16, (rows, k), generator=g, dtype=torch.uint8)
    packed = (codes[:, 1::2] << 4) | codes[:, 0::2]
    block_scale = (torch.rand(rows, k // 16, generator=g) * 4 + 0.25).to(torch.float8_e4m3fn)
    global_scale = torch.tensor([0.37])
    return codes, packed, block_scale, global_scale


def test_nvfp4_matches_hand_decoding():
    codes, packed, block_scale, global_scale = _nvfp4_case()
    out = dequantize_nvfp4(packed, block_scale, global_scale)
    assert out.dtype == torch.bfloat16 and out.shape == codes.shape
    expected = torch.empty(codes.shape)
    for r in range(codes.shape[0]):
        for c in range(codes.shape[1]):
            scale = block_scale[r, c // 16].float() * global_scale[0]
            expected[r, c] = E2M1[int(codes[r, c])] * scale
    torch.testing.assert_close(out.float(), expected.to(torch.bfloat16).float())


def test_nvfp4_matches_modelopt():
    nvfp4_tensor = pytest.importorskip("modelopt.torch.quantization.qtensor.nvfp4_tensor")
    _, packed, block_scale, global_scale = _nvfp4_case(rows=4, k=128, seed=1)
    reference = nvfp4_tensor.NVFP4QTensor((4, 128), torch.float32, packed).dequantize(
        torch.float32, scale=block_scale, double_scale=global_scale, block_sizes={-1: 16}
    )
    out = dequantize_nvfp4(packed, block_scale, global_scale, torch.float32)
    torch.testing.assert_close(out, reference)


def test_fp8_per_tensor_and_blockwise():
    w = (torch.randn(130, 260) * 3).to(torch.float8_e4m3fn)
    state = {"a.weight_scale": torch.tensor([0.5])}
    torch.testing.assert_close(
        dequantize_loaded_weight("a.weight", w, state).float(), (w.float() * 0.5).to(torch.bfloat16).float()
    )
    scale_inv = torch.rand(2, 3) + 0.5
    out = dequantize_fp8_blockwise(w, scale_inv)
    assert out.shape == w.shape
    torch.testing.assert_close(out[129, 259].float(), (w[129, 259].float() * scale_inv[1, 2]).to(torch.bfloat16).float())
    torch.testing.assert_close(out[0, 0].float(), (w[0, 0].float() * scale_inv[0, 0]).to(torch.bfloat16).float())


def test_unquantized_pass_through_and_missing_scales_raise():
    bf16 = torch.randn(4, 4, dtype=torch.bfloat16)
    assert dequantize_loaded_weight("b.weight", bf16, {}) is bf16
    raw = torch.zeros(4, 4, dtype=torch.uint8)
    assert dequantize_loaded_weight("c.weight", raw, {}) is raw
    with pytest.raises(ValueError, match="only one of"):
        dequantize_loaded_weight("c.weight", raw, {"c.weight_scale": torch.ones(4, 1)})
    with pytest.raises(ValueError, match="no recognized scale"):
        dequantize_loaded_weight("d.weight", torch.zeros(4, 4).to(torch.float8_e4m3fn), {})


def test_install_dequantizes_str_and_dict_params_once():
    class Bridge:
        def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
            if isinstance(hf_param, str):
                return hf_state_dict[hf_param]
            return {k: hf_state_dict[v] for k, v in hf_param.items()}

    w = torch.full((2, 2), 4.0).to(torch.float8_e4m3fn)
    state = {"x.weight": w, "x.weight_scale": torch.tensor([0.5]), "y.weight": torch.ones(2, 2)}
    install_dequantizing_loader(Bridge)
    wrapped = Bridge.maybe_modify_loaded_hf_weight
    install_dequantizing_loader(Bridge)
    assert Bridge.maybe_modify_loaded_hf_weight is wrapped
    bridge = Bridge()
    assert torch.equal(bridge.maybe_modify_loaded_hf_weight("x.weight", state), torch.full((2, 2), 2.0).bfloat16())
    out = bridge.maybe_modify_loaded_hf_weight({"q": "x.weight", "k": "y.weight"}, state)
    assert out["q"].dtype == torch.bfloat16 and torch.equal(out["k"], state["y.weight"])
