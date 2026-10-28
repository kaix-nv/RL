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

"""Re-encode GGUF IQ1/IQ2 blocks with their super-block scale d held fixed.

Code training for a GGUF student: the latent weight is mapped to the nearest code of its format
at the file's own d, so trained weights re-pack into the same file layout. ModelOpt's GGML
encoders do the search: its CUDA packers take d as an input for IQ2_XXS, IQ2_XS, IQ2_S and IQ1_M;
for IQ1_S, whose CUDA packer predicts d itself, ModelOpt's torch grid search runs at the given d.
Sub-block scales, grid entries, signs (and IQ1 deltas) are all re-searched.

With an importance (``llama_cpp_weights``: llama.cpp's quantizer weights from an imatrix) the search
minimizes weighted squared error. The image's ModelOpt 0.48 packers have no importance, so the
weighted search runs ModelOpt's newer packers from ggml_kernels/, built as their own extension.
"""

import os

import torch
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml.iq1_s import _search_shifted_grid, iq1_s_grid
from modelopt.torch.quantization.ggml.iq2_s import iq2_s_grid
from modelopt.torch.quantization.ggml.iq2_xs import iq2_xs_grid
from modelopt.torch.quantization.ggml.iq2_xxs import iq2_xxs_grid

_GRIDS = {
    "IQ1_S": iq1_s_grid,
    "IQ1_M": iq1_s_grid,
    "IQ2_XXS": iq2_xxs_grid,
    "IQ2_XS": iq2_xs_grid,
    "IQ2_S": iq2_s_grid,
}
CODE_TYPES = tuple(_GRIDS)
_IQ1_S_BYTES = 50
# Blocks per IQ1_S torch search. Its [blocks, 32, 128] float32 temporaries are ~0.5 GB each at
# this size; smaller chunks make the search launch-bound (~8x slower at 4096).
_IQ1_S_CHUNK = 32768


def _iq1_s_pack(
    blocks: torch.Tensor, d: torch.Tensor, grid: torch.Tensor, importance: torch.Tensor | None = None
) -> torch.Tensor:
    """ModelOpt's IQ1_S encoder (ggml/iq1_s.py ``_encode_blocks``) at a given d."""
    n = blocks.shape[0]
    if importance is None:
        best_error, best_entry = _search_shifted_grid(blocks.reshape(n, 32, 8), d.float(), grid)
    else:
        best_error, best_entry = _weighted_shifted_grid(
            blocks.reshape(n, 32, 8), d.float(), grid, importance.reshape(n, 32, 8)
        )
    group_error = best_error.reshape(n, 8, 4, 16).sum(dim=2)
    choice = group_error.argmin(dim=-1)
    entry = best_entry.gather(2, choice.repeat_interleave(4, dim=1).unsqueeze(-1)).squeeze(-1)
    high = (entry >> 8).reshape(n, 8, 4)
    qh = (
        high[:, :, 0]
        | (high[:, :, 1] << 3)
        | (high[:, :, 2] << 6)
        | (high[:, :, 3] << 9)
        | ((choice & 0x7) << 12)
        | ((choice >> 3) << 15)
    )
    packed = torch.empty((n, _IQ1_S_BYTES), dtype=torch.uint8, device=blocks.device)
    packed[:, :2] = d.contiguous().view(torch.uint8).reshape(n, 2)
    packed[:, 2:34] = (entry & 0xFF).to(torch.uint8)
    packed[:, 34:50:2] = (qh & 0xFF).to(torch.uint8)
    packed[:, 35:50:2] = (qh >> 8).to(torch.uint8)
    return torch.where((d == 0).unsqueeze(1), 0, packed)


_KERNELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ggml_kernels")
_WEIGHTED_EXT = None
_IQ1_S_DELTA = 0.125
# llama.cpp quantize_row_<type>_impl: sigma2 = c * mean(x^2) over the 256-value super-block.
_SIGMA2 = {"IQ2_XXS": 1.0, "IQ2_XS": 1.0, "IQ2_S": 2.0, "IQ1_S": 2.0, "IQ1_M": 2.0}


def weighted_ext():
    """ModelOpt's importance-weighted GGML packers (ggml_kernels/), built on first use."""
    global _WEIGHTED_EXT
    if _WEIGHTED_EXT is None:
        from torch.utils.cpp_extension import load

        sources = ["ggml.cpp", "iq1_s.cu", "iq2_xs.cu", "iq2_xxs.cu", "iq2_s.cu", "iq1_m.cu", "q8_0.cu"]
        _WEIGHTED_EXT = load(
            name="nrl_ggml_weighted_ext",
            sources=[os.path.join(_KERNELS, s) for s in sources],
            extra_cuda_cflags=["-O3"],
        )
    return _WEIGHTED_EXT


def llama_cpp_weights(qtype: str, blocks: torch.Tensor, second_moment: torch.Tensor) -> torch.Tensor:
    """Per-value search weights as llama.cpp's quantizer builds them from an imatrix:
    E[x_col^2] * sqrt(sigma2 + w^2). blocks [n, 256]; second_moment [k * 256] per column of a row of
    k blocks, with n a multiple of k."""
    qw = second_moment.to(device=blocks.device, dtype=torch.float32).reshape(-1, 256)
    qw = qw.repeat(blocks.shape[0] // qw.shape[0], 1)
    sigma2 = _SIGMA2[qtype] * blocks.square().mean(dim=1, keepdim=True)
    return qw * torch.sqrt(sigma2 + blocks.square())


def _weighted_shifted_grid(vectors, d, grid, weights):
    """ModelOpt's IQ1_S grid search (ggml/iq1_s.py ``_search_shifted_grid``, the revision in
    ggml_kernels/) with each value's squared error scaled by ``weights`` [blocks, 32, 8]: lowest
    error and entry per choice ``shift * 8 + local``, both [blocks, 32, 16]."""
    n = vectors.shape[0]
    xnorm = (weights * vectors.square()).sum(dim=-1)
    xsum = (weights * vectors).sum(dim=-1)
    weight_sum = weights.sum(dim=-1, keepdim=True)
    best_error = torch.full((n, 32, 16), torch.inf, dtype=torch.float32, device=vectors.device)
    best_entry = torch.zeros((n, 32, 16), dtype=torch.int64, device=vectors.device)
    for start in range(0, 2048, 128):
        tile = grid[start : start + 128]
        dot = torch.matmul(weights * vectors, tile.T)
        tile_norm = torch.matmul(weights, tile.square().T)
        tile_sum = torch.matmul(weights, tile.T)
        for shift in range(2):
            delta = -_IQ1_S_DELTA if shift else _IQ1_S_DELTA
            shifted_dot = dot + delta * xsum.unsqueeze(-1)
            shifted_norm = tile_norm + 2 * delta * tile_sum + weight_sum * delta * delta
            for local in range(8):
                choice = shift * 8 + local
                scale = d.reshape(-1, 1, 1) * (2 * local + 1)
                error = (xnorm.unsqueeze(-1) - 2 * scale * shifted_dot + scale.square() * shifted_norm).clamp_min_(0)
                tile_error, tile_index = error.min(dim=-1)
                replace = tile_error < best_error[:, :, choice]
                best_error[:, :, choice] = torch.where(replace, tile_error, best_error[:, :, choice])
                best_entry[:, :, choice] = torch.where(replace, tile_index + start, best_entry[:, :, choice])
    return best_error, best_entry


@torch.no_grad()
def encode(
    qtype: str, blocks: torch.Tensor, d: torch.Tensor, importance: torch.Tensor | None = None
) -> torch.Tensor:
    """Pack float32 CUDA blocks [n, 256] at FP16 scales d [n]; GGUF block bytes [n, block bytes].

    ``importance`` [n, 256] weights each value's squared error in the search."""
    grid = _GRIDS[qtype](blocks.device)
    d = d.to(device=blocks.device, dtype=torch.float16).contiguous()
    blocks = blocks.float().contiguous()
    if importance is not None:
        importance = importance.to(device=blocks.device, dtype=torch.float32).contiguous()
    if qtype == "IQ1_S":
        return torch.cat(
            [
                _iq1_s_pack(
                    blocks[i : i + _IQ1_S_CHUNK],
                    d[i : i + _IQ1_S_CHUNK],
                    grid,
                    None if importance is None else importance[i : i + _IQ1_S_CHUNK],
                )
                for i in range(0, blocks.shape[0], _IQ1_S_CHUNK)
            ]
        )
    if importance is not None:
        return getattr(weighted_ext(), qtype.lower() + "_pack")(blocks, grid, d, importance)
    ext = get_cuda_ext_ggml(raise_if_failed=True)
    # ModelOpt 0.48 packers take (input, grid, scales); later releases add an optional importance.
    return getattr(ext, qtype.lower() + "_pack")(blocks, grid, d)


@torch.no_grad()
def decode(qtype: str, packed: torch.Tensor) -> torch.Tensor:
    """GGUF block bytes [n, block bytes] -> float32 [n, 256]."""
    ext = get_cuda_ext_ggml(raise_if_failed=True)
    return getattr(ext, qtype.lower() + "_unpack")(packed.contiguous(), _GRIDS[qtype](packed.device), torch.float32)
