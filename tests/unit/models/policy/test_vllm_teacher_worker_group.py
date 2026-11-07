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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.opd import (
    OnPolicyDistillationConfig,
    TeacherResourceConfig,
    _opd_cfg,
    teacher_seq_pad_multiple,
    validate_teacher_backend_configs,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.teacher_worker_group import (
    create_teacher_configs_from_opd_config,
)
from nemo_rl.models.policy.vllm_teacher_worker_group import (
    build_vllm_teacher_runtime_config,
    VllmTeacherWorkerGroup,
)


def config():
    return {
        "enabled": True,
        "teacher_model_by_agent_name": {"teacher": "teacher-model"},
        "non_colocated_teachers": {
            "enabled": True,
            "default_teacher_cfg": {
                "backend": "vllm",
                "gpus_per_node": 1,
                "vllm_cfg": {"max_model_len": 33},
            },
        },
    }


def test_clean_runtime_does_not_requantize_or_start_dummy_weights():
    teacher = create_teacher_configs_from_opd_config(config())[0]
    runtime = build_vllm_teacher_runtime_config(teacher)
    assert runtime["model_name"] == "teacher-model"
    assert runtime["quant_cfg"] is None
    assert runtime["colocated"]["enabled"] is False
    assert runtime["vllm_cfg"]["precision"] == "bfloat16"
    assert runtime["vllm_cfg"]["load_format"] == "auto"
    assert runtime["vllm_cfg"]["logprobs_mode"] == "raw_logprobs"
    assert runtime["vllm_cfg"]["async_engine"] is True
    assert runtime["vllm_cfg"]["expose_http_server"] is False
    assert "quantization" not in runtime["vllm_kwargs"]
    assert "speculative_config" not in runtime["vllm_kwargs"]


@pytest.mark.parametrize(
    "extra",
    [
        {"context_parallel_size": 2},
        {"expert_model_parallel_size": 2},
        {"tensor_model_parallel_size": 3},
        {"micro_batch_size": 0},
        {"vllm_kwargs": {"load_format": "dummy"}},
        {"moe_shared_expert_overlap": False},
        {"megatron_cfg_overrides": {"enabled": True}},
        {"vllm_cfg": {"max_model_len": 33, "quantization": "fp8"}},
    ],
)
def test_rejects_ignored_or_conflicting_settings(extra):
    params = config()["non_colocated_teachers"]["default_teacher_cfg"]
    with pytest.raises(ValueError):
        TeacherResourceConfig(**(params | extra))


def test_deduplication_checks_resolved_alias_settings():
    cfg = config()
    cfg["teacher_model_by_agent_name"]["other"] = "teacher-model"
    cfg["non_colocated_teachers"]["teacher_overrides"] = {
        "other": {"micro_batch_size": 2}
    }
    with pytest.raises(ValueError, match="conflicting"):
        create_teacher_configs_from_opd_config(cfg)


def test_typed_alias_override_inherits_backend_and_resources():
    cfg = config()
    cfg["non_colocated_teachers"]["teacher_overrides"] = {
        "teacher": {"micro_batch_size": 2}
    }
    typed = OnPolicyDistillationConfig(**cfg)
    resolved = create_teacher_configs_from_opd_config(
        _opd_cfg({"on_policy_distillation": typed})
    )[0]
    assert resolved.backend == "vllm"
    assert resolved.gpus_per_node == 1
    assert resolved.micro_batch_size == 2
    assert resolved.vllm_cfg.max_model_len == 33


def master_config():
    return {
        "on_policy_distillation": config(),
        "policy": {
            "max_total_sequence_length": 32,
            "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": None},
        },
    }


@pytest.mark.parametrize(
    "field,value", [("temperature", 0.7), ("top_p", 0.9), ("top_k", 20)]
)
def test_preflight_rejects_different_recomputation_distribution(field, value):
    master = master_config()
    master["policy"]["generation"][field] = value
    with pytest.raises(ValueError, match="raw logprobs"):
        validate_teacher_backend_configs(master)


def test_preflight_rejects_insufficient_context_and_single_controller():
    master = master_config()
    assert len(validate_teacher_backend_configs(master)) == 1
    with pytest.raises(ValueError, match="legacy"):
        validate_teacher_backend_configs(master, single_controller=True)
    master["policy"]["max_total_sequence_length"] = 33
    with pytest.raises(ValueError, match="exceed"):
        validate_teacher_backend_configs(master)


def test_native_ragged_teacher_does_not_conflict_with_megatron_packing():
    native = SimpleNamespace(use_sequence_packing=None, sequence_length_pad_multiple=1)
    packed = SimpleNamespace(use_sequence_packing=True, sequence_length_pad_multiple=4)
    assert teacher_seq_pad_multiple({"native": native}, 8) == 1
    assert teacher_seq_pad_multiple({"native": native, "packed": packed}, 8) == 1
    packed.use_sequence_packing = False
    assert teacher_seq_pad_multiple({"native": native, "packed": packed}, 8) == 8


def test_group_dispatches_uneven_shards_and_preserves_shape():
    teacher = object.__new__(VllmTeacherWorkerGroup)
    teacher._micro_batch_size = 1
    teacher.sharding_annotations = MagicMock()
    teacher.sharding_annotations.get_axis_size.return_value = 2
    teacher.worker_group = MagicMock()
    teacher.worker_group.get_all_worker_results.return_value = [
        BatchedDataDict(reference_logprobs=torch.tensor([[0.0, -1.0, 0.0]])),
        BatchedDataDict(
            reference_logprobs=torch.tensor([[0.0, -2.0, -3.0], [0.0, -4.0, 0.0]])
        ),
    ]
    data = BatchedDataDict(
        input_ids=torch.tensor([[1, 2, 0], [3, 4, 5], [6, 7, 0]]),
        input_lengths=torch.tensor([2, 3, 2]),
    )
    result = teacher.get_logprobs(data)
    assert result["reference_logprobs"].tolist() == [
        [0.0, -1.0, 0.0],
        [0.0, -2.0, -3.0],
        [0.0, -4.0, 0.0],
    ]
    call = teacher.worker_group.run_all_workers_sharded_data.call_args
    assert call.args == ("score_tokens",)
    assert call.kwargs["replicate_on_axes"] is None
    assert call.kwargs["common_kwargs"] == {"micro_batch_size": 1}


def test_vocab_validation_allows_padding_alias_but_checks_semantic_tokens(monkeypatch):
    from nemo_rl.models.huggingface import common

    class Tokenizer:
        bos_token_id = 1
        eos_token_id = 2

        def __init__(self, pad, specials):
            self.pad_token_id = pad
            self.all_special_ids = specials

        def get_vocab(self):
            return {"pad": 0, "bos": 1, "eos": 2, "other": 3}

        def __len__(self):
            return 4

    student = Tokenizer(2, [1, 2])
    teacher = Tokenizer(0, [0, 1, 2])
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda name: teacher)
    monkeypatch.setattr(
        common.AutoConfig,
        "from_pretrained",
        lambda name: SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(vocab_size=4)
        ),
    )
    common.check_vocab_equality(
        student, "student", "teacher", check_special_tokens=True
    )
    teacher.eos_token_id = 3
    with pytest.raises(ValueError, match="eos_token_id"):
        common.check_vocab_equality(
            student, "student", "teacher", check_special_tokens=True
        )


@pytest.mark.parametrize(
    "patch", [{"max_model_len": 65}, {"gpu_memory_utilization": 0.5}]
)
def test_nested_alias_override_merges_before_validation(patch):
    cfg = config()
    cfg["non_colocated_teachers"]["teacher_overrides"] = {
        "teacher": {"vllm_cfg": patch}
    }
    typed = OnPolicyDistillationConfig(**cfg)
    for source in (cfg, _opd_cfg({"on_policy_distillation": typed})):
        resolved = create_teacher_configs_from_opd_config(source)[0]
        assert resolved.backend == "vllm"
        assert resolved.vllm_cfg.max_model_len == patch.get("max_model_len", 33)
        assert resolved.vllm_cfg.gpu_memory_utilization == patch.get(
            "gpu_memory_utilization", 0.85
        )


def test_failed_dp_replica_drains_all_dispatched_rpcs(monkeypatch):
    from nemo_rl.models.policy import vllm_teacher_worker_group as module

    teacher = object.__new__(VllmTeacherWorkerGroup)
    teacher._micro_batch_size = 1
    teacher.sharding_annotations = MagicMock()
    teacher.sharding_annotations.get_axis_size.return_value = 2
    teacher.worker_group = MagicMock()
    refs = [object(), object()]
    teacher.worker_group.run_all_workers_sharded_data.return_value = SimpleNamespace(
        futures=refs
    )
    teacher.worker_group.get_all_worker_results.side_effect = RuntimeError(
        "replica failed"
    )
    waited = []
    monkeypatch.setattr(
        module.ray, "wait", lambda values, **kwargs: waited.append((values, kwargs))
    )
    data = BatchedDataDict(
        input_ids=torch.tensor([[1, 2], [3, 4]]), input_lengths=torch.tensor([2, 2])
    )
    with pytest.raises(RuntimeError, match="replica failed"):
        teacher.get_logprobs(data)
    assert waited == [(refs, {"num_returns": 2, "fetch_local": False})]


def test_cross_node_teacher_reserves_unified_placement_before_loading(monkeypatch):
    from nemo_rl.algorithms import opd

    master = master_config()
    resources = master["on_policy_distillation"]["non_colocated_teachers"][
        "default_teacher_cfg"
    ]
    resources.update(num_nodes=4, gpus_per_node=4, tensor_model_parallel_size=16)
    cluster = MagicMock()
    cluster.num_gpus_per_node = 4
    monkeypatch.setattr(opd, "RayVirtualCluster", lambda **kwargs: cluster)
    assert opd.reserve_teacher_clusters(master) == {"teacher": cluster}
    cluster._init_placement_groups.assert_called_once_with(
        strategy="PACK", use_unified_pg=True
    )
    cluster.get_placement_groups.assert_not_called()
