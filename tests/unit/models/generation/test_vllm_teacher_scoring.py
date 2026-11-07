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

import asyncio
from types import SimpleNamespace

import pytest
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.scoring import (
    prompt_logprobs_to_tensor,
    score_token_batch,
)


def output_for(ids):
    return SimpleNamespace(
        prompt_token_ids=ids,
        prompt_logprobs=[None]
        + [
            {token: SimpleNamespace(logprob=-float(position))}
            for position, token in enumerate(ids[1:], start=1)
        ],
    )


def test_alignment_and_actual_target_token():
    output = output_for([7, 8, 9])
    output.prompt_logprobs[1][42] = SimpleNamespace(logprob=-0.1)
    assert prompt_logprobs_to_tensor(output, [7, 8, 9]).tolist() == [0, -1, -2]


@pytest.mark.parametrize("defect", ["ids", "length", "missing", "nan", "positive"])
def test_rejects_invalid_teacher_scores(defect):
    output = output_for([7, 8, 9])
    if defect == "ids":
        output.prompt_token_ids = [7, 9, 8]
    elif defect == "length":
        output.prompt_logprobs.pop()
    elif defect == "missing":
        output.prompt_logprobs[1] = {42: SimpleNamespace(logprob=-1)}
    else:
        output.prompt_logprobs[1][8].logprob = float("nan") if defect == "nan" else 1.0
    with pytest.raises(ValueError):
        prompt_logprobs_to_tensor(output, [7, 8, 9])


class FakeEngine:
    def __init__(self):
        self.active = 0
        self.peak = 0
        self.prompts = []
        self.aborted = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.block = False
        self.fail = False

    async def generate(self, prompt, sampling_params, request_id):
        del sampling_params, request_id
        self.active += 1
        self.peak = max(self.active, self.peak)
        self.prompts.append(prompt["prompt_token_ids"])
        self.started.set()
        try:
            if self.block:
                await self.release.wait()
            else:
                await asyncio.sleep(0)
            if self.fail:
                raise RuntimeError("engine failed")
            yield output_for(prompt["prompt_token_ids"])
        finally:
            self.active -= 1

    async def abort(self, request_id):
        self.aborted.append(request_id)


def score(engine, data, *, micro_batch_size=2):
    return score_token_batch(
        engine,
        data,
        sampling_params=object(),
        micro_batch_size=micro_batch_size,
        max_model_len=8,
        vocab_size=100,
    )


def test_lengths_padding_eos_and_bounded_requests():
    async def run():
        engine = FakeEngine()
        data = BatchedDataDict(
            input_ids=torch.tensor(
                [[7, 8, 99, -1], [7, 99, -1, -1], [7, -1, -1, -1], [8, 9, 10, 99]]
            ),
            input_lengths=torch.tensor([3, 2, 1, 4]),
        )
        scores = (await score(engine, data))["reference_logprobs"]
        assert scores.dtype == torch.float32 and scores.device.type == "cpu"
        assert scores.tolist() == [
            [0, -1, -2, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 0],
            [0, -1, -2, -3],
        ]
        assert engine.prompts == [[7, 8, 99], [7, 99], [8, 9, 10, 99]]
        assert engine.peak <= 2
        assert engine.active == 0 and engine.aborted == []
        empty = BatchedDataDict(
            input_ids=torch.empty((0, 4), dtype=torch.long),
            input_lengths=torch.empty(0, dtype=torch.long),
        )
        assert (await score(engine, empty))["reference_logprobs"].shape == (0, 4)

    asyncio.run(run())


@pytest.mark.parametrize(
    "ids,length", [([1, 2], 0), ([1, 2], 3), ([1] * 8, 8), ([1, 100], 2)]
)
def test_rejects_invalid_lengths_and_token_ids(ids, length):
    data = BatchedDataDict(
        input_ids=torch.tensor([ids]), input_lengths=torch.tensor([length])
    )
    with pytest.raises(ValueError):
        asyncio.run(score(FakeEngine(), data))


def test_request_failure_aborts_and_drains_siblings():
    async def run():
        engine = FakeEngine()
        engine.fail = True
        data = BatchedDataDict(
            input_ids=torch.tensor([[1, 2], [3, 4]]), input_lengths=torch.tensor([2, 2])
        )
        with pytest.raises(RuntimeError, match="engine failed"):
            await score(engine, data)
        assert engine.active == 0
        assert len(engine.aborted) == 2

    asyncio.run(run())


def test_actor_cancellation_aborts_requests():
    async def run():
        engine = FakeEngine()
        engine.block = True
        data = BatchedDataDict(
            input_ids=torch.tensor([[1, 2], [3, 4]]), input_lengths=torch.tensor([2, 2])
        )
        task = asyncio.create_task(score(engine, data))
        await engine.started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert engine.active == 0
        assert len(engine.aborted) == 2

    asyncio.run(run())
