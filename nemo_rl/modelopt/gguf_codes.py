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
"""

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


def _iq1_s_pack(blocks: torch.Tensor, d: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """ModelOpt's IQ1_S encoder (ggml/iq1_s.py ``_encode_blocks``) at a given d."""
    n = blocks.shape[0]
    best_error, best_entry = _search_shifted_grid(blocks.reshape(n, 32, 8), d.float(), grid)
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


@torch.no_grad()
def encode(qtype: str, blocks: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    """Pack float32 CUDA blocks [n, 256] at FP16 scales d [n]; GGUF block bytes [n, block bytes]."""
    grid = _GRIDS[qtype](blocks.device)
    d = d.to(device=blocks.device, dtype=torch.float16).contiguous()
    blocks = blocks.float().contiguous()
    if qtype == "IQ1_S":
        return torch.cat(
            [
                _iq1_s_pack(blocks[i : i + _IQ1_S_CHUNK], d[i : i + _IQ1_S_CHUNK], grid)
                for i in range(0, blocks.shape[0], _IQ1_S_CHUNK)
            ]
        )
    ext = get_cuda_ext_ggml(raise_if_failed=True)
    # ModelOpt 0.48 packers take (input, grid, scales); later releases add an optional importance.
    return getattr(ext, qtype.lower() + "_pack")(blocks, grid, d)


@torch.no_grad()
def decode(qtype: str, packed: torch.Tensor) -> torch.Tensor:
    """GGUF block bytes [n, block bytes] -> float32 [n, 256]."""
    ext = get_cuda_ext_ggml(raise_if_failed=True)
    return getattr(ext, qtype.lower() + "_unpack")(packed.contiguous(), _GRIDS[qtype](packed.device), torch.float32)
