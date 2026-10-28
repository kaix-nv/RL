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

"""Frozen-code fake quantization for a llama.cpp GGUF student (scale-only QAOPD).

Each block of a quantized GGUF tensor dequantizes as d * U + dmin * V, with U and V fixed by the
block's codes; only Q2_K, Q4_K and Q5_K have a dmin. Training changes only d and dmin. Megatron
keeps a latent weight W, and the ``gguf_frozen`` weight quantizer replaces W with its per-block
least-squares projection onto span{U, V}: the closest weight the GGUF holds with its codes fixed.
That projection is linear and symmetric, so its exact gradient is the same projection applied to
the incoming gradient. A student loaded from the dequantized GGUF already lies in the span, so at
step 0 it is the GGUF.

The projection does not depend on how U and V are scaled, so the frozen codes are used as stored by
``cluster/scripts/gguf_frozen.py`` (integers, HF row order, GGUF column order). Each rank finds where
its weight shard's rows and columns come from by pushing BF16 index probes through the Bridge's own
HF -> Megatron mappings, so fused weights (QKV, gate/up, GatedDeltaNet in_proj), vocabulary padding
and TP sharding need no layout code here.

Rollouts need nothing special: weights are quantized, activations are not, so the fake-quant refit
(``weight_quantizer(W)`` folded into BF16) gives vLLM exactly the GGUF's weights.
"""

from __future__ import annotations

import json
import os
import weakref
from dataclasses import dataclass

import torch
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    TensorQuantizer,
    register_quant_backend,
)
from safetensors import safe_open

BACKEND = "gguf_frozen"
# Stage (b): IQ1/IQ2 blocks re-encode at their frozen d (gguf_codes.py, straight-through
# gradient); blocks of other types keep the gguf_frozen projection.
CODES_BACKEND = "gguf_codes"
# Rows of a segment projected at a time, bounding the float32 temporaries.
_CHUNK_ELEMENTS = 1 << 26
# Quantizer -> FrozenCodes. Kept off the quantizer itself: ModelOpt serializes quantizer attributes
# into Megatron checkpoints' extra state, which would pickle the code tables into every checkpoint.
_FROZEN: "weakref.WeakKeyDictionary[TensorQuantizer, FrozenCodes]" = weakref.WeakKeyDictionary()


@dataclass
class _Segment:
    """The rows of a local weight shard that come from one quantized GGUF tensor."""

    name: str
    rows: torch.Tensor | slice  # local rows
    cols: torch.Tensor | None  # local column order with whole GGUF blocks contiguous; None: as is
    u: torch.Tensor | None  # int16 [rows, cols], block order; None for re-encoded segments
    v: torch.Tensor | None  # int8 [rows, cols // 16], min term (Q2_K, Q4_K, Q5_K)
    block: int  # 256, or 32 for Q8_0
    qtype: str | None = None  # re-encoded GGUF type; None: frozen codes, projected
    d: torch.Tensor | None = None  # FP16 [rows, blocks] scales of a re-encoded segment


def _project_blocks(x: torch.Tensor, u: torch.Tensor, v: torch.Tensor | None, block: int) -> torch.Tensor:
    """Least-squares projection of x [r, c] onto span{u, v} per block of `block` columns."""
    r, c = x.shape
    nb = c // block
    x = x.reshape(r, nb, block)
    u = u.reshape(r, nb, block).float()
    suu = (u * u).sum(-1)
    sxu = (x * u).sum(-1)
    has_u = suu > 0
    if v is None:
        a = torch.where(has_u, sxu / torch.where(has_u, suu, 1), 0)
        return (a.unsqueeze(-1) * u).reshape(r, c)
    # v is constant over 16 values: sums over v reduce to sums over 16-value groups.
    vg = v.reshape(r, nb, block // 16).float()
    svv = 16 * (vg * vg).sum(-1)
    suv = (u.reshape(r, nb, block // 16, 16).sum(-1) * vg).sum(-1)
    sxv = (x.reshape(r, nb, block // 16, 16).sum(-1) * vg).sum(-1)
    det = suu * svv - suv * suv
    both = det > 1e-6 * suu * svv
    safe = torch.where(both, det, 1)
    # Degenerate blocks (codes or min term all zero, or u parallel to v) keep the 1-D projection.
    a1 = torch.where(has_u, sxu / torch.where(has_u, suu, 1), 0)
    b1 = torch.where(~has_u & (svv > 0), sxv / torch.where(svv > 0, svv, 1), 0)
    a = torch.where(both, (sxu * svv - sxv * suv) / safe, a1)
    b = torch.where(both, (suu * sxv - suv * sxu) / safe, b1)
    out = a.unsqueeze(-1) * u + (b.unsqueeze(-1) * vg).repeat_interleave(16, dim=-1)
    return out.reshape(r, c)


@torch.no_grad()
def _checksum(x: torch.Tensor):
    """Exact, order-independent fingerprint of a weight's BF16 values.

    Integer sums of the BF16 bit patterns, so the BF16 parameter and the float32 copy the refit
    folds give the same key, and the key survives Megatron offloading and reloading the parameter.
    An optimizer step changes it.
    """
    b = x.detach().to(torch.bfloat16).view(torch.int16)
    return tuple(x.shape), int(b.sum(dtype=torch.int64)), int((b.to(torch.int32) * b).sum(dtype=torch.int64))


class FrozenCodes:
    """Quantizer of one local weight shard onto its GGUF blocks with the scales d held fixed.

    Frozen-code segments project onto the file's codes; re-encoded segments (stage (b)) map to the
    nearest code at d. Rows outside every segment (Megatron vocabulary padding) pass through.
    """

    def __init__(self, segments: list[_Segment]):
        self.segments = segments
        self.reencodes = any(seg.qtype is not None for seg in segments)
        # Re-encoding is a codebook search: run it once per weight value (optimizer step), shared
        # by the refit, every forward and activation recompute. (checksum, BF16 weight) or None.
        self._cache = None

    def to(self, device: torch.device | str) -> None:
        # Host copies are pageable, not pinned: PyTorch's pinned allocator can round each block up
        # to a power of two and keeps freed blocks, and with a BF16 student (vLLM backs its 50 GB
        # of weights up to host memory while asleep) that overhead ran nodes out of host memory.
        device = torch.device(device)
        if self._cache is not None and self._cache[1].device != device and device.type == "cpu":
            self._cache = (self._cache[0], self._cache[1].to(device))
        for seg in self.segments:
            for field in ("rows", "cols", "u", "v", "d"):
                t = getattr(seg, field)
                if isinstance(t, torch.Tensor) and t.device != device:
                    setattr(seg, field, t.to(device))

    def _segment_input(self, x, seg):
        xs = x[seg.rows]
        return xs[:, seg.cols] if seg.cols is not None else xs

    def _write(self, out, seg, q):
        if seg.cols is not None:
            q = torch.empty_like(q).index_copy_(1, seg.cols, q)
        out[seg.rows] = q.to(out.dtype)

    def _project(self, xs, seg):
        q = torch.empty(xs.shape, dtype=torch.float32, device=xs.device)
        step = max(1, _CHUNK_ELEMENTS // xs.shape[1])
        for i in range(0, xs.shape[0], step):
            q[i : i + step] = _project_blocks(
                xs[i : i + step].float(),
                seg.u[i : i + step],
                None if seg.v is None else seg.v[i : i + step],
                seg.block,
            )
        return q

    @torch.no_grad()
    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        key = _checksum(x) if self.reencodes else None
        if key is not None and self._cache is not None and self._cache[0] == key:
            if self._cache[1].device != x.device:
                self._cache = (key, self._cache[1].to(x.device))
            return self._cache[1].to(dtype=x.dtype).detach()
        self.to(x.device)
        out = x.clone()
        for seg in self.segments:
            xs = self._segment_input(x, seg)
            if seg.qtype is None:
                q = self._project(xs, seg)
            else:
                from nemo_rl.modelopt.gguf_codes import decode, encode

                packed = encode(seg.qtype, xs.float().reshape(-1, seg.block), seg.d.reshape(-1))
                q = decode(seg.qtype, packed).reshape(xs.shape)
            self._write(out, seg, q)
        if key is not None:
            self._cache = (key, out.to(torch.bfloat16))
            return out.detach()
        return out

    @torch.no_grad()
    def grad(self, g: torch.Tensor) -> torch.Tensor:
        """Backward: projected segments' projection (symmetric, so exact); identity (straight
        through) on re-encoded segments and padding."""
        out = g.clone()
        for seg in self.segments:
            if seg.qtype is None:
                self._write(out, seg, self._project(self._segment_input(g, seg), seg))
        return out


class _Project(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight: torch.Tensor, frozen: FrozenCodes) -> torch.Tensor:
        ctx.frozen = frozen
        return frozen.quantize(weight)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return ctx.frozen.grad(grad.contiguous()), None


def _fake_quant(inputs: torch.Tensor, quantizer: TensorQuantizer) -> torch.Tensor:
    frozen = _FROZEN.get(quantizer)
    if frozen is None:
        raise RuntimeError(
            f"{BACKEND} weight quantizer has no frozen GGUF codes; "
            "attach_gguf_frozen() must run before the first forward"
        )
    return _Project.apply(inputs, frozen)


register_quant_backend(BACKEND, _fake_quant)
register_quant_backend(CODES_BACKEND, _fake_quant)


# --------------------------------------------------------------------------- #
# Layout probe
# --------------------------------------------------------------------------- #
def _digits(n: int) -> int:
    """Base-256 digits needed for 1..n (BF16 holds the integers 0..256 exactly)."""
    k = 1
    while 256**k <= n:
        k += 1
    return k


def _push(task, names: list[str], shapes: dict[str, tuple[int, int]], make) -> torch.Tensor:
    """Run the task's HF -> Megatron mapping on probe tensors make(source index, shape)."""
    hf_param = task.mapping.hf_param
    probes = [make(i, shapes[n]) for i, n in enumerate(names)]
    arg = probes[0] if isinstance(hf_param, str) else dict(zip(hf_param.keys(), probes))
    return task.mapping.hf_to_megatron(arg, task.megatron_module)


def probe_layout(task, shapes: dict[str, tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For each element of the task's local Megatron shard, where it comes from in HF.

    Returns (source, row, col): local row -> index into the task's HF names (-1: padding), local
    row -> HF row, local column -> HF column. Collective over the TP group, like a weight load.
    """
    hf_param = task.mapping.hf_param
    names = [hf_param] if isinstance(hf_param, str) else list(hf_param.values())
    device = torch.cuda.current_device()

    def rows_of(local):
        if not bool((local == local[:, :1]).all()):
            raise RuntimeError(f"{task.global_param_name}: a probe row is not constant")
        return local[:, 0].long()

    def cols_of(local):
        if not bool((local == local[:1]).all()):
            raise RuntimeError(f"{task.global_param_name}: a probe column is not constant")
        return local[0].long()

    def digit_probe(k, along_rows):
        def make(i, shape):
            n = shape[0] if along_rows else shape[1]
            d = ((torch.arange(n, device=device) + 1) >> (8 * k)) & 255
            d = d.to(torch.bfloat16)
            return (d[:, None] if along_rows else d[None, :]).expand(*shape).contiguous()

        return make

    max_rows = max(shapes[n][0] for n in names)
    max_cols = max(shapes[n][1] for n in names)
    row = sum(
        rows_of(_push(task, names, shapes, digit_probe(k, True))) << (8 * k) for k in range(_digits(max_rows))
    )
    col = sum(
        cols_of(_push(task, names, shapes, digit_probe(k, False))) << (8 * k) for k in range(_digits(max_cols))
    )
    if len(names) > 1:
        src = rows_of(
            _push(task, names, shapes, lambda i, shape: torch.full(shape, i + 1, dtype=torch.bfloat16, device=device))
        )
    else:
        src = (row > 0).long()
    return src - 1, row - 1, col - 1


# --------------------------------------------------------------------------- #
# Attach
# --------------------------------------------------------------------------- #
def _read_rows(f, key: str, rows: torch.Tensor) -> torch.Tensor:
    """Rows of a stored tensor, reading only the contiguous runs that cover them."""
    uniq = torch.unique(rows)
    breaks = torch.nonzero(uniq[1:] != uniq[:-1] + 1).flatten() + 1
    runs = list(zip([0, *breaks.tolist()], [*breaks.tolist(), len(uniq)]))
    sl = f.get_slice(key)
    parts = [sl[int(uniq[a]) : int(uniq[b - 1]) + 1] for a, b in runs]
    block = torch.cat(parts) if len(parts) > 1 else parts[0]
    return block[torch.searchsorted(uniq, rows)]


def _as_slice(idx: torch.Tensor) -> torch.Tensor | slice:
    if idx.numel() and bool((idx == torch.arange(idx[0], idx[0] + idx.numel())).all()):
        return slice(int(idx[0]), int(idx[0]) + idx.numel())
    return idx


def _segment(frozen_dir: str, name: str, meta: dict, local_rows, hf_rows, hf_cols, reencode: bool) -> _Segment:
    block = meta["block"]
    with safe_open(os.path.join(frozen_dir, name + ".safetensors"), framework="pt") as f:
        u = None if reencode else _read_rows(f, "u", hf_rows)
        v = _read_rows(f, "v", hf_rows) if meta["has_min"] and not reencode else None
        d = _read_rows(f, "d", hf_rows) if reencode else None
        gcol = f.get_tensor("gcol").long() if meta["has_gcol"] else None
    gc = hf_cols if gcol is None else gcol[hf_cols]
    order = torch.argsort(gc)
    gs = gc[order]
    starts = gs.reshape(-1, block)[:, 0]
    if gs.numel() % block or not bool(
        (gs.reshape(-1, block) == starts[:, None] + torch.arange(block)).all() and (starts % block == 0).all()
    ):
        raise RuntimeError(f"{name}: local columns do not cover whole GGUF blocks")
    if u is not None:
        u = u[:, gs].contiguous()
    if v is not None:
        v = v[:, gs[::16] // 16].contiguous()
    if d is not None:
        d = d[:, starts // block].contiguous()
    cols = None if bool((order == torch.arange(order.numel())).all()) else order
    return _Segment(name, _as_slice(local_rows), cols, u, v, block, meta["type"] if reencode else None, d)


def attach_gguf_frozen(worker, frozen_dir: str) -> dict:
    """Give every enabled GGUF weight quantizer of the worker's model its frozen codes or scales.

    ``gguf_frozen`` quantizers project onto the file's codes. ``gguf_codes`` quantizers re-encode
    IQ1/IQ2 blocks at the file's d and project the rest. Checks that every quantized GGUF tensor
    except the token embedding is covered on this rank, and reports the relative distance of each
    weight to its quantized value: for a scale-only student loaded from the dequantized GGUF it is
    BF16 rounding (~2e-3), and O(1) means a wrong layout.
    """
    from nemo_rl.modelopt.gguf_codes import CODE_TYPES

    index = json.load(open(os.path.join(frozen_dir, "index.json")))
    quantized = index["quantized"]
    tasks = worker.megatron_bridge.get_conversion_tasks([worker.model], exclude_param_patterns=("*_quantizer*",))
    covered, disabled, dist, scale_only = set(), [], {}, True
    for task in tasks:
        if task is None or task.megatron_module is None or task.param_weight is None:
            continue
        wq = worker._find_weight_quantizer(task.megatron_module, task.param_weight)
        backend = getattr(wq, "backend", None)
        if backend not in (BACKEND, CODES_BACKEND):
            continue
        scale_only &= backend == BACKEND
        hf_param = task.mapping.hf_param
        names = [hf_param] if isinstance(hf_param, str) else list(hf_param.values())
        if not any(n in quantized for n in names):
            wq.disable()
            disabled.append(task.global_param_name)
            continue
        missing = [n for n in names if n not in quantized]
        if missing:
            raise RuntimeError(f"{task.global_param_name}: GGUF does not quantize {missing}")
        shapes = {n: tuple(quantized[n]["shape"]) for n in names}
        src, hrow, hcol = probe_layout(task, shapes)
        src, hrow, hcol = src.cpu(), hrow.cpu(), hcol.cpu()
        segments = []
        for i, n in enumerate(names):
            local = torch.nonzero(src == i).flatten()
            if local.numel():
                reencode = backend == CODES_BACKEND and quantized[n]["type"] in CODE_TYPES
                segments.append(_segment(frozen_dir, n, quantized[n], local, hrow[local], hcol, reencode))
                covered.add(n)
        frozen = FrozenCodes(segments)
        frozen.to(task.param_weight.device)
        w = task.param_weight.detach()
        err = (frozen.quantize(w).float() - w.float()).norm() / w.float().norm().clamp_min(1e-30)
        dist[task.global_param_name] = float(err)
        _FROZEN[wq] = frozen

    expected = {n for n in quantized if not n.endswith("embed_tokens.weight")}
    if covered != expected:
        raise RuntimeError(
            f"gguf_frozen covers {len(covered)} of {len(expected)} quantized tensors; "
            f"uncovered: {sorted(expected - covered)[:8]}, unexpected: {sorted(covered - expected)[:8]}"
        )
    worst = max(dist.values())
    # A re-encoding student may start away from the GGUF (e.g. from BF16, at its format's own error
    # of up to ~0.5), so only scale-only students are checked.
    if scale_only and worst > 0.5:
        bad = sorted(dist, key=dist.get)[-5:]
        raise RuntimeError(f"gguf_frozen projection moves weights by up to {worst:.3f} (wrong layout?): {bad}")
    return {"weights": len(dist), "disabled": disabled, "max_dist": worst,
            "mean_dist": sum(dist.values()) / len(dist), "dist": dist}


def offload_gguf_frozen(model: torch.nn.Module) -> None:
    """Move every attached code table and encode cache to host memory; the next forward reloads it."""
    for module in model.modules():
        frozen = _FROZEN.get(module) if isinstance(module, TensorQuantizer) else None
        if frozen is not None:
            frozen.to("cpu")


def _output_requires_grad(module, args, output):
    # A frozen embedding's output needs no gradient, but Megatron's (reentrant) activation
    # recompute only backpropagates through a recomputed block whose input requires grad:
    # without this, no decoder layer would get gradients.
    if isinstance(output, torch.Tensor) and not output.requires_grad:
        return output.detach().requires_grad_()
    return output


def freeze_untrained(megatron_model):
    """Pre-wrap hook: freeze what a GGUF student does not train.

    The input embedding stays at its GGUF values (GGUF quantizes it, but it has no weight
    quantizer), and the vision tower is not in the GGUF at all. Freezing the vision tower also
    matters for Megatron's DDP: with the 1.27B-parameter embedding gone from the buckets, vision
    parameters (no gradient on text) would share a bucket with decoder parameters, which fails
    its all-or-none gradient check.
    """
    models = megatron_model if isinstance(megatron_model, list) else [megatron_model]
    for model in models:
        for name, module in model.named_modules():
            weight = getattr(getattr(module, "word_embeddings", None), "weight", None)
            if weight is not None and name.endswith("embedding") and "vision" not in name:
                weight.requires_grad = False
                module.register_forward_hook(_output_requires_grad)
        for name, param in model.named_parameters():
            if "vision_model." in name:
                param.requires_grad = False
    return megatron_model


def gradient_coverage(model) -> tuple[int, int, list[str]]:
    """(params that got a gradient in the last step, trainable params, names of params missing
    from partially filled buckets), from Megatron DDP's per-bucket bookkeeping."""
    groups = [*getattr(model, "bucket_groups", []), *getattr(model, "expert_parallel_bucket_groups", [])]
    names = {p: n for n, p in model.named_parameters()}
    seen = sum(len(g.per_param_grad_ready_counts) for g in groups)
    total = sum(len(g.params) for g in groups)
    missing = [
        names.get(p, "?")
        for g in groups
        if g.per_param_grad_ready_counts
        for p in g.params
        if p not in g.per_param_grad_ready_counts
    ]
    return seen, total, missing
