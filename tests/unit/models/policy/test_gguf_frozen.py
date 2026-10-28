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
"""CPU tests for the frozen-code GGUF projection (nemo_rl/modelopt/gguf_frozen.py)."""

import pytest
import torch

gguf_frozen = pytest.importorskip("nemo_rl.modelopt.gguf_frozen")


def _codes(rows, cols, block, with_min, seed=0):
    g = torch.Generator().manual_seed(seed)
    u = torch.randint(-300, 300, (rows, cols), generator=g, dtype=torch.int16)
    v = -torch.randint(0, 64, (rows, cols // 16), generator=g, dtype=torch.int8) if with_min else None
    return u, v


def _gguf_weight(u, v, block, seed=1):
    """A weight the GGUF can hold exactly: d * u + dmin * v per block."""
    g = torch.Generator().manual_seed(seed)
    r, c = u.shape
    d = torch.randn(r, c // block, 1, generator=g) * 1e-3
    w = (d * u.float().reshape(r, -1, block)).reshape(r, c)
    if v is not None:
        dmin = torch.rand(r, c // block, 1, generator=g) * 1e-3
        w = w + (dmin * v.float().repeat_interleave(16, dim=1).reshape(r, -1, block)).reshape(r, c)
    return w


@pytest.mark.parametrize("block,with_min", [(256, False), (256, True), (32, False)])
def test_projection_is_exact_symmetric_and_idempotent(block, with_min):
    u, v = _codes(6, 4 * block, block, with_min)
    proj = lambda x: gguf_frozen._project_blocks(x, u, v, block)  # noqa: E731

    w = _gguf_weight(u, v, block)
    # Float32 2x2 solves: far below BF16 rounding (2^-9).
    torch.testing.assert_close(proj(w), w, rtol=1e-4, atol=1e-7)

    a, b = torch.randn(2, *u.shape, dtype=torch.float32)
    pa = proj(a)
    torch.testing.assert_close(proj(pa), pa, rtol=1e-4, atol=1e-6)
    # Symmetric: <Pa, b> = <a, Pb>, so backward(grad) = P grad is the exact gradient.
    torch.testing.assert_close((pa * b).sum(), (a * proj(b)).sum(), rtol=1e-4, atol=1e-4)
    # Least squares: the residual is orthogonal to the codes in every block.
    res = (a - pa).reshape(6, -1, block)
    assert (res * u.float().reshape(6, -1, block)).sum(-1).abs().max() < 1e-2


def test_degenerate_blocks():
    block = 256
    u, v = _codes(2, 2 * block, block, with_min=True)
    u[0, :block] = 0  # no codes: only the min term remains
    v[1, : block // 16] = 0  # no min term
    x = torch.randn(*u.shape)
    q = gguf_frozen._project_blocks(x, u, v, block)
    assert torch.isfinite(q).all()
    torch.testing.assert_close(gguf_frozen._project_blocks(q, u, v, block), q, rtol=1e-4, atol=1e-6)


def test_frozen_codes_segments_rows_and_column_permutation():
    """Two sources interleaved by rows, one with GGUF columns permuted, plus a padding row."""
    block = 32
    ua, _ = _codes(3, 4 * block, block, False, seed=2)
    ub, vb = _codes(2, 4 * block, block, True, seed=3)
    # Local layout: rows [a0, b0, a1, b1, a2, pad]; source b's local columns are GGUF block order
    # [2, 0, 3, 1] (each block of 32 contiguous).
    perm_blocks = torch.tensor([2, 0, 3, 1])
    cols_b = (perm_blocks[:, None] * block + torch.arange(block)).flatten()  # local col -> GGUF col
    order = torch.argsort(cols_b)  # local columns in GGUF order
    segs = [
        gguf_frozen._Segment("a", torch.tensor([0, 2, 4]), None, ua, None, block),
        gguf_frozen._Segment("b", torch.tensor([1, 3]), order, ub, vb, block),
    ]
    frozen = gguf_frozen.FrozenCodes(segs)
    x = torch.randn(6, 4 * block)
    q = frozen.quantize(x)
    torch.testing.assert_close(q[[0, 2, 4]], gguf_frozen._project_blocks(x[[0, 2, 4]], ua, None, block))
    ref_b = gguf_frozen._project_blocks(x[[1, 3]][:, order], ub, vb, block)
    torch.testing.assert_close(q[[1, 3]][:, order], ref_b)
    torch.testing.assert_close(q[5], x[5])  # padding passes through

    w = x.clone().requires_grad_()
    out = gguf_frozen._Project.apply(w, frozen)
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(w.grad, frozen.grad(grad))


def test_reencoded_segment_straight_through_and_cached(monkeypatch):
    """Re-encoded rows use the encoder with a straight-through gradient; projected rows keep the
    projection; the result is cached per weight version."""
    import sys
    import types

    calls = []
    fake = types.ModuleType("nemo_rl.modelopt.gguf_codes")
    fake.encode = lambda qtype, blocks, d, importance=None: (
        calls.append(qtype),
        torch.round(blocks / d[:, None].float()),
    )[1]
    fake.decode = lambda qtype, packed: packed * 0.5
    fake.llama_cpp_weights = None
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.gguf_codes", fake)

    block = 256
    ua, _ = _codes(2, block, block, False, seed=4)
    segs = [
        gguf_frozen._Segment("a", slice(0, 2), None, ua, None, block),
        gguf_frozen._Segment("b", slice(2, 4), None, None, None, block, "IQ2_XXS", torch.full((2, 1), 0.5).half()),
    ]
    frozen = gguf_frozen.FrozenCodes(segs)
    w = torch.randn(4, block).requires_grad_()
    out = gguf_frozen._Project.apply(w, frozen)
    torch.testing.assert_close(out[2:], torch.round(w[2:].detach() / 0.5) * 0.5)
    torch.testing.assert_close(out[:2], gguf_frozen._project_blocks(w[:2].detach(), ua, None, block))
    grad = torch.randn_like(out)
    out.backward(grad)
    torch.testing.assert_close(w.grad[2:], grad[2:])
    torch.testing.assert_close(w.grad[:2], gguf_frozen._project_blocks(grad[:2], ua, None, block))

    gguf_frozen._Project.apply(w, frozen)
    assert calls == ["IQ2_XXS"]  # same version: cached
    with torch.no_grad():
        w.add_(1.0)
    gguf_frozen._Project.apply(w, frozen)
    assert calls == ["IQ2_XXS", "IQ2_XXS"]  # optimizer update: re-encoded


def test_reencoded_segment_passes_imatrix_weights(monkeypatch):
    """A segment with imatrix second moments searches with llama.cpp's weights of its blocks."""
    import sys
    import types

    seen = {}

    def encode(qtype, blocks, d, importance=None):
        seen["blocks"], seen["importance"] = blocks.clone(), importance
        return blocks

    fake = types.ModuleType("nemo_rl.modelopt.gguf_codes")
    fake.encode = encode
    fake.decode = lambda qtype, packed: packed
    fake.llama_cpp_weights = lambda qtype, blocks, e2: (
        e2.reshape(-1, 256).repeat(blocks.shape[0] // (e2.numel() // 256), 1) + blocks
    )
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.gguf_codes", fake)

    e2 = torch.rand(512)
    seg = gguf_frozen._Segment("b", slice(0, 3), None, None, None, 256, "IQ2_XXS", torch.ones(3, 2).half(), e2)
    w = torch.randn(3, 512)
    gguf_frozen.FrozenCodes([seg]).quantize(w)
    torch.testing.assert_close(seen["blocks"], w.reshape(-1, 256))
    torch.testing.assert_close(seen["importance"], e2.reshape(-1, 256).repeat(3, 1) + w.reshape(-1, 256))


def test_llama_cpp_weights():
    """imatrix[col] * sqrt(sigma2 + x^2), sigma2 = c * mean(x^2) per 256-value block."""
    gguf_codes = pytest.importorskip("nemo_rl.modelopt.gguf_codes")
    x = torch.randn(6, 256)
    e2 = torch.rand(512)
    for qtype, c in (("IQ2_XXS", 1.0), ("IQ1_S", 2.0)):
        want = e2.reshape(2, 256).repeat(3, 1) * torch.sqrt(c * x.square().mean(1, keepdim=True) + x.square())
        torch.testing.assert_close(gguf_codes.llama_cpp_weights(qtype, x, e2), want)


def test_freeze_untrained_keeps_downstream_gradients():
    """A frozen embedding must still hand the decoder an input that requires grad, or
    reentrant activation recompute skips every layer's backward."""

    class Embedding(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.word_embeddings = torch.nn.Embedding(10, 4)

        def forward(self, ids):
            return self.word_embeddings(ids)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = Embedding()
            self.layer = torch.nn.Linear(4, 4)

        def forward(self, ids):
            h = self.embedding(ids)
            return torch.utils.checkpoint.checkpoint(self.layer, h, use_reentrant=True)

    model = Model()
    gguf_frozen.freeze_untrained([model])
    assert not model.embedding.word_embeddings.weight.requires_grad
    model(torch.tensor([1, 2, 3])).sum().backward()
    assert model.layer.weight.grad is not None
    assert model.embedding.word_embeddings.weight.grad is None


def test_digits():
    assert gguf_frozen._digits(255) == 1
    assert gguf_frozen._digits(256) == 2
    assert gguf_frozen._digits(248320) == 3


class _Slice:
    def __init__(self, t):
        self.t = t

    def __getitem__(self, s):
        return self.t[s]


class _File:
    def __init__(self, tensors):
        self.tensors = tensors
        self.reads = []

    def get_slice(self, key):
        t = self.tensors[key]
        file = self

        class _Logged(_Slice):
            def __getitem__(self, s):
                file.reads.append((s.start, s.stop))
                return super().__getitem__(s)

        return _Logged(t)


def test_read_rows_reads_runs_in_local_order():
    t = torch.arange(40).reshape(20, 2)
    f = _File({"u": t})
    rows = torch.tensor([7, 3, 4, 5, 15, 16, 3])
    torch.testing.assert_close(gguf_frozen._read_rows(f, "u", rows), t[rows])
    assert f.reads == [(3, 6), (7, 8), (15, 17)]
