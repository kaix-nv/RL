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

"""Frozen vLLM teachers implementing the legacy OPD logprob contract."""

from typing import Any, Optional

import ray
import torch
from transformers import PreTrainedTokenizerBase

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.huggingface.common import check_vocab_equality
from nemo_rl.models.policy.teacher_worker_group import TeacherConfig


def build_vllm_teacher_runtime_config(teacher_cfg: TeacherConfig) -> VllmConfig:
    """Build an isolated runtime config without inheriting student features."""
    limits = teacher_cfg.vllm_cfg
    if teacher_cfg.backend != "vllm" or limits is None:
        raise ValueError("Expected a resolved vLLM teacher configuration")
    precision = {"bf16": "bfloat16", "fp16": "float16"}.get(
        teacher_cfg.precision, teacher_cfg.precision
    )
    return VllmConfig(
        backend="vllm",
        model_name=teacher_cfg.model_name,
        max_new_tokens=1,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        val_temperature=1.0,
        val_top_p=1.0,
        val_top_k=None,
        stop_token_ids=None,
        stop_strings=None,
        quant_cfg=None,
        colocated={
            "enabled": False,
            "resources": {"num_nodes": None, "gpus_per_node": None},
        },
        vllm_cfg={
            "tensor_parallel_size": teacher_cfg.tensor_model_parallel_size,
            "pipeline_parallel_size": teacher_cfg.pipeline_model_parallel_size,
            "expert_parallel_size": 1,
            "precision": precision,
            "async_engine": True,
            "load_format": "auto",
            "skip_tokenizer_init": True,
            "expose_http_server": False,
            "logprobs_mode": "raw_logprobs",
            "kv_cache_dtype": "auto",
            "enforce_eager": True,
            "gpu_memory_utilization": limits.gpu_memory_utilization,
            "max_model_len": limits.max_model_len,
        },
        vllm_kwargs={
            "max_num_batched_tokens": limits.max_num_batched_tokens,
            "max_num_seqs": limits.max_num_seqs,
            "enable_chunked_prefill": True,
        },
    )


class VllmTeacherWorkerGroup:
    """A permanently resident teacher backed by the existing vLLM runtime."""

    def __init__(
        self,
        teacher_cfg: TeacherConfig,
        cluster: RayVirtualCluster,
        policy_config: dict[str, Any],
        tokenizer: PreTrainedTokenizerBase,
        teacher_index: int,
    ) -> None:
        self.alias = teacher_cfg.alias
        self.model_name = teacher_cfg.model_name
        self.teacher_cfg = teacher_cfg
        self.teacher_index = teacher_index
        # None distinguishes native ragged scoring from Megatron packing modes.
        self.use_sequence_packing = None
        self.sequence_length_pad_multiple = 1
        self._micro_batch_size = teacher_cfg.micro_batch_size
        check_vocab_equality(
            tokenizer,
            policy_config["model_name"],
            self.model_name,
            check_special_tokens=True,
        )
        cfg = build_vllm_teacher_runtime_config(teacher_cfg)
        cfg["_pad_token_id"] = tokenizer.pad_token_id
        self.generation = VllmGeneration(
            cluster, cfg, name_prefix=f"teacher_{self.alias}"
        )
        self.worker_group = self.generation.worker_group
        self.sharding_annotations = self.generation.sharding_annotations
        try:
            # Exercise actual model execution, not merely Ray actor construction.
            token_id = next(iter(tokenizer.get_vocab().values()))
            self.get_logprobs(
                BatchedDataDict(
                    input_ids=torch.tensor([[token_id, token_id]]),
                    input_lengths=torch.tensor([2]),
                )
            )
        except BaseException:
            self.shutdown()
            raise

    def get_logprobs(
        self, data: BatchedDataDict, micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict:
        """Score exact tokens and retain the caller's batch order and padded width."""
        if data["input_ids"].shape[0] == 0:
            return BatchedDataDict(
                reference_logprobs=torch.zeros(
                    data["input_ids"].shape, dtype=torch.float32, device="cpu"
                )
            )
        shards = data.shard_by_batch_size(
            self.sharding_annotations.get_axis_size("data_parallel"),
            allow_uneven_shards=True,
        )
        futures = self.worker_group.run_all_workers_sharded_data(
            "score_tokens",
            data=shards,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,
            output_is_replicated=None,
            common_kwargs={
                "micro_batch_size": (
                    self._micro_batch_size
                    if micro_batch_size is None
                    else micro_batch_size
                )
            },
        )
        try:
            results = self.worker_group.get_all_worker_results(futures)
        except BaseException:
            # ray.get(list) raises immediately on the first failed replica.
            # Keep the caller's teacher lock until every dispatched RPC finishes.
            ray.wait(
                futures.futures, num_returns=len(futures.futures), fetch_local=False
            )
            raise
        return BatchedDataDict.from_batches(results)

    def shutdown(self) -> bool:
        """Stop the teacher's engines using the existing generation lifecycle."""
        return self.generation.shutdown()
