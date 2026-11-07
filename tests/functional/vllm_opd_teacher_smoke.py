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

"""Local GPU smoke: real vLLM teacher, HF parity, and one sampled-token OPD update.

Run with the vLLM extra installed and NEMO_RL_PY_EXECUTABLES_SYSTEM=1 if all Ray
actors should use the driver's prepared environment. Requires two GPUs. This
exercises the teacher factory and legacy collector scoring boundary; it does not
launch the full NeMo-Gym/Megatron training recipe. Use --teacher-gpus 2 to
exercise DP2, or --teacher-gpus 2 --teacher-tp 2 for TP2; these need three GPUs.
"""

import argparse
import json
import threading

import ray
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from nemo_rl.algorithms.advantage_estimator import (
    AdvEstimatorConfig,
    OPDAdvantageEstimator,
)
from nemo_rl.algorithms.async_utils import AsyncTrajectoryCollector
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.opd import (
    reserve_teacher_clusters,
    create_teacher_worker_groups,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def token_logprobs(model, ids, lengths):
    mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    scores = logits.log_softmax(-1).gather(-1, ids[:, 1:, None]).squeeze(-1)
    return torch.nn.functional.pad(scores, (1, 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--teacher-gpus", type=int, default=1)
    parser.add_argument("--teacher-tp", type=int, default=1)
    args = parser.parse_args()
    student_device = torch.device("cuda", args.teacher_gpus)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    policy = {
        "model_name": args.model,
        "max_total_sequence_length": 256,
        "make_sequence_length_divisible_by": 1,
        "generation": {"temperature": 1.0, "top_p": 1.0, "top_k": None},
    }
    master = {
        "policy": policy,
        "on_policy_distillation": {
            "enabled": True,
            "teacher_model_by_agent_name": {"teacher": args.model},
            "default_teacher_alias": "teacher",
            "non_colocated_teachers": {
                "enabled": True,
                "default_teacher_cfg": {
                    "backend": "vllm",
                    "gpus_per_node": args.teacher_gpus,
                    "tensor_model_parallel_size": args.teacher_tp,
                    "micro_batch_size": 1,
                    "vllm_cfg": {
                        "max_model_len": 257,
                        "gpu_memory_utilization": 0.4,
                        "max_num_batched_tokens": 64,
                        "max_num_seqs": 1,
                    },
                },
            },
        },
    }
    ray.init(num_gpus=args.teacher_gpus, num_cpus=8, include_dashboard=False)
    clusters = reserve_teacher_clusters(master)
    teachers = {}
    try:
        teachers, aliases = create_teacher_worker_groups(
            master, policy, tokenizer, teacher_clusters=clusters
        )
        teacher = teachers["teacher"]
        model = (
            AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
            .to(student_device)
            .eval()
        )
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        # Hit the exact maximum admitted length and cross four prefill chunks.
        ids = torch.full((2, 256), tokenizer.eos_token_id, dtype=torch.long)
        repeated = tokenizer.encode(
            "The capital of France is Paris. ", add_special_tokens=False
        )
        ids[0] = torch.tensor((repeated * 256)[:256])
        ids[1, :5] = torch.tensor(repeated[:4] + [tokenizer.eos_token_id])
        lengths = torch.tensor([256, 5])
        data = BatchedDataDict(input_ids=ids, input_lengths=lengths)
        reference = teacher.get_logprobs(data)["reference_logprobs"]
        repeated_reference = teacher.get_logprobs(data)["reference_logprobs"]
        assert torch.equal(reference, repeated_reference), (
            "Frozen teacher changed between requests"
        )
        with torch.no_grad():
            expected = token_logprobs(
                model, ids.to(student_device), lengths.to(student_device)
            ).cpu()
        valid = torch.arange(256)[None, :] < lengths[:, None]
        valid[:, 0] = False
        errors = (expected - reference).abs()[valid]
        # BF16 attention and GEMM implementations differ across the two engines.
        # Bound both aggregate error and outliers; a causal shift fails by many nats.
        print(
            f"BF16 parity: mean={errors.mean().item():.6f}, max={errors.max().item():.6f}",
            flush=True,
        )
        assert errors.mean() < 0.05
        assert errors.max() < 0.5
        assert reference[1, 5:].eq(0).all()
        assert reference[:, 0].eq(0).all()

        # Make student/teacher deliberately different, then use actual on-policy tokens.
        model.lm_head.weight.requires_grad_(True)
        with torch.no_grad():
            model.lm_head.weight.add_(0.002 * torch.randn_like(model.lm_head.weight))
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is 2 + 2? Answer briefly."}],
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )["input_ids"].to(student_device)
        with torch.no_grad():
            rollout = model.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=12,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
            )
        rollout_lengths = torch.tensor([rollout.shape[1]], device=student_device)
        prev = token_logprobs(model, rollout, rollout_lengths).detach()

        collector_cls = AsyncTrajectoryCollector.__ray_metadata__.modified_class
        collector = object.__new__(collector_cls)
        collector.teacher_worker_groups = teachers
        collector.alias_to_group_alias = aliases
        collector.on_policy_distillation_cfg = master["on_policy_distillation"]
        collector._teacher_locks = {"teacher": threading.Lock()}
        teacher_scores, _ = collector._compute_teacher_logprobs(
            rollout.cpu(),
            [{"name": "teacher"}],
            input_lengths=rollout_lengths.cpu(),
        )
        mask = torch.zeros_like(prev)
        mask[:, prompt.shape[1] :] = 1
        loss_cfg = ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            disable_ppo_ratio=True,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="icepop",
            truncated_importance_sampling_ratio_min=0.2,
            truncated_importance_sampling_ratio=5.0,
        )
        estimator = OPDAdvantageEstimator(AdvEstimatorConfig(name="opd"), loss_cfg)
        advantages = estimator.compute_advantage(
            prompt_ids=None,
            rewards=None,
            mask=mask,
            teacher_logprobs=teacher_scores.to(prev.device),
            prev_logprobs=prev,
        )
        assert torch.isfinite(advantages).all() and advantages.abs().sum() > 0
        current = token_logprobs(model, rollout, rollout_lengths)
        loss_data = BatchedDataDict(
            input_ids=rollout,
            advantages=advantages,
            prev_logprobs=prev,
            generation_logprobs=prev,
            token_mask=mask,
            sample_mask=torch.ones(1, device=prev.device),
        )
        loss, _ = ClippedPGLossFn(loss_cfg)(
            next_token_logprobs=current[:, 1:],
            data=loss_data,
            global_valid_seqs=torch.tensor(1, device=prev.device),
            global_valid_toks=mask.sum(),
        )
        assert torch.isfinite(loss)
        loss.backward()
        grad_norm = model.lm_head.weight.grad.float().norm().item()
        assert grad_norm > 0 and torch.isfinite(model.lm_head.weight.grad).all()
        # SGD is enough to prove an actual parameter update without optimizer-state allocation.
        torch.optim.SGD([model.lm_head.weight], lr=1e-4).step()
        with torch.no_grad():
            after = token_logprobs(model, rollout, rollout_lengths)
        update = (after - prev).abs()[mask.bool()].max().item()
        assert update > 0, "OPD optimizer step did not change student logprobs"
        assert torch.equal(reference, teacher.get_logprobs(data)["reference_logprobs"])
        print(
            json.dumps(
                {
                    "model": args.model,
                    "teacher_backend": "vllm",
                    "teacher_gpus": args.teacher_gpus,
                    "teacher_tp": args.teacher_tp,
                    "parity_mean_abs_error": errors.mean().item(),
                    "parity_max_abs_error": errors.max().item(),
                    "max_scored_length": 256,
                    "prefill_chunk_tokens": 64,
                    "rollout_tokens": int(mask.sum()),
                    "opd_loss": loss.item(),
                    "gradient_norm": grad_norm,
                    "student_max_logprob_change": update,
                    "teacher_unchanged": True,
                },
                indent=2,
            )
        )
    finally:
        for teacher in teachers.values():
            teacher.shutdown()
        for cluster in clusters.values():
            cluster.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
