# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Dequantize quantized Hugging Face checkpoints while the Megatron bridge loads them.

Megatron-Bridge's Qwen3.5 MoE bridge casts every checkpoint tensor to the model dtype and ignores
the scale tensors stored beside it, so a quantized checkpoint loads with silently wrong weights
(an FP8 checkpoint loaded this way gave a top-300 overlap of 0.004 against the reference). This
module undoes three layouts, each recognized by the scale tensors that sit next to ``<name>.weight``:

* NVFP4 (ModelOpt): ``uint8`` weight packing two E2M1 values per byte (low nibble first), an FP8
  E4M3 ``weight_scale`` per 16 input values and an FP32 ``weight_scale_2``.
* FP8 per tensor (ModelOpt): ``float8_e4m3fn`` weight and a one-element ``weight_scale``.
* FP8 per 128 x 128 block (``*_scale_inv``): ``float8_e4m3fn`` weight and ``weight_scale_inv``.

Only weights are dequantized. Activation scales (``input_scale``) are ignored, so a model loaded
through this path runs BF16 activations: it carries the checkpoint's weight quantization error
but not its activation quantization error. A tensor in a quantized dtype with missing or
unrecognized scales raises instead of loading wrongly; other tensors pass through unchanged.
"""

from collections.abc import Mapping

import torch

NVFP4_BLOCK = 16
FP8_BLOCK = 128
# E2M1 code -> value; the sign is bit 3 and ModelOpt decodes negative zero as 0.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def dequantize_nvfp4(
    weight: torch.Tensor,
    block_scale: torch.Tensor,
    global_scale: torch.Tensor,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """NVFP4 [..., K // 2] uint8 -> [..., K] ``dtype``, as ModelOpt's ``NVFP4QTensor.dequantize``."""
    if weight.dtype != torch.uint8:
        raise ValueError(f"NVFP4 weight must be uint8, got {weight.dtype}")
    k = weight.shape[-1] * 2
    if block_scale.shape != (*weight.shape[:-1], k // NVFP4_BLOCK):
        raise ValueError(f"NVFP4 weight {tuple(weight.shape)} does not match block scale {tuple(block_scale.shape)}")
    table = torch.tensor(_E2M1, dtype=torch.float32, device=weight.device)
    codes = torch.stack((weight & 0x0F, weight >> 4), dim=-1).reshape(*weight.shape[:-1], k)
    values = table[codes.long()].view(*weight.shape[:-1], k // NVFP4_BLOCK, NVFP4_BLOCK)
    scale = block_scale.float() * global_scale.float().reshape(())
    return (values * scale.unsqueeze(-1)).reshape(*weight.shape[:-1], k).to(dtype)


def dequantize_fp8_blockwise(
    weight: torch.Tensor, scale_inv: torch.Tensor, dtype: torch.dtype = torch.bfloat16
) -> torch.Tensor:
    """FP8 [rows, cols] with a [ceil(rows / 128), ceil(cols / 128)] ``scale_inv`` -> ``dtype``."""
    rows, cols = weight.shape
    expected = (-(-rows // FP8_BLOCK), -(-cols // FP8_BLOCK))
    if tuple(scale_inv.shape) != expected:
        raise ValueError(f"FP8 weight {tuple(weight.shape)} needs a {expected} scale, got {tuple(scale_inv.shape)}")
    scale = scale_inv.float().repeat_interleave(FP8_BLOCK, 0).repeat_interleave(FP8_BLOCK, 1)
    return (weight.float() * scale[:rows, :cols]).to(dtype)


def dequantize_loaded_weight(
    name: str,
    weight: torch.Tensor,
    hf_state_dict: Mapping[str, torch.Tensor],
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Return ``weight`` (the checkpoint tensor ``name``) dequantized if it is stored quantized."""
    if weight.dtype == torch.uint8:
        block_scale = hf_state_dict.get(f"{name}_scale")
        global_scale = hf_state_dict.get(f"{name}_scale_2")
        if block_scale is None and global_scale is None:
            return weight
        if block_scale is None or global_scale is None:
            raise ValueError(f"{name} is uint8 with only one of {name}_scale and {name}_scale_2")
        return dequantize_nvfp4(weight, block_scale, global_scale, dtype)
    if weight.dtype != torch.float8_e4m3fn:
        return weight
    scale_inv = hf_state_dict.get(f"{name}_scale_inv")
    if scale_inv is not None:
        return dequantize_fp8_blockwise(weight, scale_inv, dtype)
    scale = hf_state_dict.get(f"{name}_scale")
    if scale is not None and scale.numel() == 1:
        return (weight.float() * scale.float().reshape(())).to(dtype)
    raise ValueError(f"{name} is FP8 but has no recognized scale tensor ({name}_scale_inv or one-element {name}_scale)")


def install_dequantizing_loader(bridge_cls: type) -> None:
    """Make ``bridge_cls.maybe_modify_loaded_hf_weight`` dequantize quantized checkpoint tensors.

    Idempotent. Checkpoints with no quantized tensors load exactly as before.
    """
    if getattr(bridge_cls, "_dequantizes_hf_weights", False):
        return
    original = bridge_cls.maybe_modify_loaded_hf_weight

    def maybe_modify_loaded_hf_weight(self, hf_param, hf_state_dict):
        loaded = original(self, hf_param, hf_state_dict)
        if isinstance(hf_param, str):
            return dequantize_loaded_weight(hf_param, loaded, hf_state_dict)
        return {key: dequantize_loaded_weight(hf_param[key], tensor, hf_state_dict) for key, tensor in loaded.items()}

    bridge_cls.maybe_modify_loaded_hf_weight = maybe_modify_loaded_hf_weight
    bridge_cls._dequantizes_hf_weights = True


def install_for_qwen35_moe() -> None:
    """Install the dequantizing loader on Megatron-Bridge's Qwen3.5 MoE bridge."""
    from megatron.bridge.models.qwen.qwen35_bridge import Qwen35MoEBridge

    install_dequantizing_loader(Qwen35MoEBridge)
