# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Lightweight ModelOpt helpers shared by Megatron and vLLM workers."""

from __future__ import annotations

import copy
from typing import Any

MODELOPT_REAL_QUANT_REFIT_TIMEOUT_MS = 600_000


def validate_real_quant_policy_config(policy_config: dict[str, Any]) -> None:
    """Validate static policy requirements for real-quant rollout."""
    if not policy_config.get("megatron_cfg", {}).get("enabled", False):
        raise ValueError(
            "policy.generation.real_quant=true requires "
            "policy.megatron_cfg.enabled=true; DTensor real-quant export is unsupported"
        )
    if not policy_config.get("quant_cfg"):
        raise ValueError(
            "policy.quant_cfg must be set when policy.generation.real_quant=true"
        )


def prepare_real_quant_generation_config(policy: Any, generation_config: dict) -> None:
    """Inject the policy-produced canonical ModelOpt config into vLLM config."""
    policy_config = policy.cfg
    validate_real_quant_policy_config(policy_config)

    quantization_config = policy.get_real_quantization_config()
    hf_overrides = generation_config.setdefault("vllm_kwargs", {}).setdefault(
        "hf_overrides", {}
    )
    existing = hf_overrides.get("quantization_config")
    if existing is not None and existing != quantization_config:
        raise ValueError(
            "generation.vllm_kwargs.hf_overrides.quantization_config conflicts "
            "with the policy-produced ModelOpt config"
        )
    hf_overrides["quantization_config"] = copy.deepcopy(quantization_config)


def keep_quantized_output_layer_extra_state() -> None:
    """Let a quantized output_layer (lm_head) be saved and loaded as a Megatron checkpoint.

    ``GPTModel.sharded_state_dict`` drops ``output_layer._extra_state`` and asserts it is empty,
    but a quantized output_layer keeps its ModelOpt quantizer state there. ModelOpt patches the
    method (``keep_gpt_output_layer_extra_state``) only when its body matches a statement
    fingerprint of older megatron-core; megatron-core main binds the payload to ``data`` before
    the assert (one more assignment), so ModelOpt 0.48.0rc1 skips the patch and saving fails.
    Accept that body too, then let ModelOpt apply its own patch.

    TODO: drop once the image's ModelOpt accepts the newer body.
    """
    import ast
    import inspect
    import textwrap

    from megatron.core.models.gpt.gpt_model import GPTModel
    from modelopt.torch.quantization.plugins import megatron as mtq_megatron

    patch = getattr(mtq_megatron, "keep_gpt_output_layer_extra_state", None)
    expected = getattr(mtq_megatron, "_GPT_SSD_STATEMENTS", None)
    if patch is None or not isinstance(expected, list):
        return  # ModelOpt without the patch, or one that already accepts several bodies
    func = ast.parse(textwrap.dedent(inspect.getsource(GPTModel.sharded_state_dict))).body[0]
    body = func.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    kinds = [type(stmt).__name__ for stmt in body]
    if kinds == ["Assign"] + expected:
        mtq_megatron._GPT_SSD_STATEMENTS = kinds
        patch.cache_clear()
        if not patch():
            raise RuntimeError(
                "ModelOpt did not patch GPTModel.sharded_state_dict; a quantized "
                "output_layer cannot be checkpointed"
            )


def resolve_quant_cfg(quant_cfg: str) -> dict[str, Any]:
    """Resolve a quantization config string into a dict consumable by ``mtq.quantize``.

    Resolution order:

    1. Built-in ModelOpt config constant exposed on ``modelopt.torch.quantization``
       (e.g. ``"NVFP4_DEFAULT_CFG"``, ``"FP8_DEFAULT_CFG"``).
    2. A ModelOpt PTQ recipe — either the name of a built-in recipe shipped under
       ``modelopt_recipes/`` (e.g. ``"general/ptq/nvfp4_default-fp8_kv"``; the
       ``.yml`` / ``.yaml`` suffix is optional) or the path to a user-authored
       YAML recipe. Resolution is performed by ``modelopt.recipe.load_config``,
       which searches the filesystem first and then the built-in recipe library.
       For Ray/container workers, use an absolute path for user-authored recipe
       files; NeMo-RL repo-relative recipe paths are not resolved here.

    YAML recipes are expected to follow the standard ModelOpt PTQ recipe layout
    with a top-level ``quantize:`` section in the ``{"quant_cfg": [...],
    "algorithm": ...}`` shape that ``mtq.quantize`` expects. A bare
    ``{"quant_cfg": [...], "algorithm": ...}`` document (without a wrapping
    ``quantize:`` key) is also accepted for convenience. If ``algorithm`` is
    omitted, it defaults to ``"max"`` so ModelOpt's calibration helpers see the
    same normalized config as ``mtq.quantize``. The extracted dict — not the full
    recipe — is returned.

    See ``modelopt_recipes/general/ptq/`` in the NVIDIA/Model-Optimizer repo
    (https://github.com/NVIDIA/Model-Optimizer) for the canonical format and
    ``examples/modelopt/quant_configs/`` for a user-authored example.
    """
    import modelopt.torch.quantization as mtq
    from modelopt.recipe import load_config

    def _normalize_mtq_cfg(config: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(config, dict):
            raise ValueError(
                f"Quantization recipe '{quant_cfg}' must resolve to a dict."
            )
        mtq_cfg = config.get("quantize", config)
        if not isinstance(mtq_cfg, dict) or "quant_cfg" not in mtq_cfg:
            raise ValueError(
                f"Quantization recipe '{quant_cfg}' must contain a 'quant_cfg' "
                f"entry (optionally nested under a top-level 'quantize:' section)."
            )
        if "algorithm" not in mtq_cfg:
            mtq_cfg = {**mtq_cfg, "algorithm": "max"}
        return mtq_cfg

    builtin = getattr(mtq, quant_cfg, None)
    if builtin is not None:
        return _normalize_mtq_cfg(builtin)

    try:
        loaded = load_config(quant_cfg)
    except (ValueError, FileNotFoundError) as e:
        raise ValueError(
            f"Unknown quant_cfg '{quant_cfg}'. Must be either a built-in "
            f"ModelOpt config name (e.g. 'NVFP4_DEFAULT_CFG'), a built-in "
            f"ModelOpt PTQ recipe name (e.g. "
            f"'general/ptq/nvfp4_default-fp8_kv'), or an absolute path to a "
            f"YAML quantization recipe."
        ) from e

    return _normalize_mtq_cfg(loaded)
