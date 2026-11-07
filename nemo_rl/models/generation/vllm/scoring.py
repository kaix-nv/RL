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

"""Exact-token teacher scoring using vLLM's native prompt logprobs."""

import asyncio
import uuid
from typing import Any

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def prompt_logprobs_to_tensor(output: Any, token_ids: list[int]) -> torch.Tensor:
    """Convert a completed request, preserving the policy's causal alignment."""
    if output is None or output.prompt_token_ids != token_ids:
        raise ValueError("Teacher prompt token IDs do not match the scoring request")
    logprobs = output.prompt_logprobs
    if logprobs is None or len(logprobs) != len(token_ids):
        raise ValueError("Teacher returned incomplete prompt logprobs")
    result = torch.zeros(len(token_ids), dtype=torch.float32, device="cpu")
    for position, token_id in enumerate(token_ids[1:], start=1):
        row = logprobs[position]
        if row is None or token_id not in row:
            raise ValueError(f"Teacher omitted target token at position {position}")
        result[position] = row[token_id].logprob
    if not torch.isfinite(result).all() or (result > 1e-5).any():
        raise ValueError("Teacher returned invalid normalized logprobs")
    return result


async def score_token_batch(
    engine: Any,
    data: BatchedDataDict,
    *,
    sampling_params: Any,
    micro_batch_size: int,
    max_model_len: int,
    vocab_size: int,
) -> BatchedDataDict:
    """Score rows with bounded concurrency and drain requests before returning.

    Args:
        engine: AsyncLLM on its owning event loop.
        data: Right-padded input_ids and explicit input_lengths. Text only.
        sampling_params: Private prompt-scoring parameters, independent of rollout.
        micro_batch_size: Maximum live scoring requests per engine.
        max_model_len: Engine context, including the one ignored generated token.
        vocab_size: Teacher output vocabulary bound.

    Returns:
        CPU float32 reference_logprobs with the input batch's order and width.
    """
    if data.get_multimodal_dict(as_tensors=False):
        raise ValueError(
            "vLLM teacher scoring currently supports text-only trajectories"
        )
    input_ids = data["input_ids"]
    lengths = data["input_lengths"]
    if input_ids.ndim != 2 or lengths.shape != (input_ids.shape[0],):
        raise ValueError("Teacher requires input_ids [B, S] and input_lengths [B]")
    if input_ids.dtype not in (torch.int32, torch.int64) or lengths.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Teacher token IDs and lengths must be integer tensors")
    if micro_batch_size < 1:
        raise ValueError("Teacher micro_batch_size must be positive")
    if (
        (lengths < 1) | (lengths > input_ids.shape[1]) | (lengths >= max_model_len)
    ).any():
        raise ValueError(
            "Teacher input length is invalid or leaves no context for scoring"
        )
    result = torch.zeros(input_ids.shape, dtype=torch.float32, device="cpu")

    async def score_row(index: int) -> None:
        token_ids = input_ids[index, : int(lengths[index])].tolist()
        if min(token_ids) < 0 or max(token_ids) >= vocab_size:
            raise ValueError("Teacher input token ID is outside its output vocabulary")
        if len(token_ids) == 1:
            return
        request_id = f"teacher-{uuid.uuid4().hex}"
        completed = False
        try:
            final_output = None
            async for output in engine.generate(
                prompt={"prompt_token_ids": token_ids},
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                final_output = output
            completed = True
            result[index, : len(token_ids)] = prompt_logprobs_to_tensor(
                final_output, token_ids
            )
        finally:
            if not completed:
                await engine.abort(request_id)

    for start in range(0, input_ids.shape[0], micro_batch_size):
        tasks = [
            asyncio.create_task(score_row(index))
            for index in range(start, min(start + micro_batch_size, input_ids.shape[0]))
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
    return BatchedDataDict(reference_logprobs=result)
