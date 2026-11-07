# vLLM teachers for sampled-token OPD

Status: implemented for legacy async GRPO sampled-token OPD. Local GPU validation
uses Qwen3-0.6B; the 2.4T NVFP4 checkpoint still needs validation on Blackwell.

## Decision

Add a `vllm` implementation of the existing teacher logprob interface. It owns a
frozen, separately placed `VllmGeneration` instance and scores student trajectories
with vLLM's native prompt logprobs. The GRPO algorithm continues to consume the
same tensor it consumes from a Megatron teacher.

The first target is Qwen3.8-27B QAOPD with
`nvidia/Qwen3.8-2.4T-A95B-NVFP4` on the existing async GRPO path in
`kaix-nv/RL:kaix/qaopd-gguf` at `1181179c5`. This is the deck's sampled-token
path (b). The teacher's quantization belongs entirely to vLLM's checkpoint loader;
student fake quantization belongs entirely to the training policy.

```text
student rollout → exact prompt + response token IDs
                → teacher.get_logprobs(input_ids, input_lengths)
                → float32 teacher_reference_logprobs [B, S]
                → existing OPD advantages → existing policy update
```

This initial change covers the legacy async GRPO entrypoint used by the existing
recipe. SingleController needs a separate metadata/data-plane adapter and is
rejected explicitly until that adapter is implemented. Teacher top-300 KL,
full-vocabulary OPD, and image/video scoring are separate work.

## One scoring contract

Keep `get_logprobs(data, micro_batch_size=None)` returning the existing
`BatchedDataDict` with `reference_logprobs`. For a right-padded input `[B, S]`
with valid length `L[b]`:

```text
result[b, 0] = 0
result[b, t] = log p_teacher(input_ids[b, t] | input_ids[b, :t])  for 1 ≤ t < L[b]
result[b, t] = 0                                              for t ≥ L[b]
```

The result has the input's batch order and padded width, on CPU in float32.
The existing response mask selects assistant tokens for advantages and loss.
Score the complete valid trajectory, including EOS and any intermediate tool
messages; do not score isolated response chunks without their preceding context.

Send token IDs directly. Never decode and re-tokenize, insert a BOS token, apply
a chat template, or infer valid lengths by searching for a padding token. The
deck reports identical vocabulary mappings but different pre-tokenizers; this
contract preserves the student trajectory despite that difference. At startup,
reuse or extract `distillation.check_vocab_equality` for token-to-ID mapping,
effective tokenizer size, and model output vocabulary size; also verify
semantic special-token IDs, including added tokens. The transport's padding ID
need not be the teacher's padding choice because padding is removed. A size-only
comparison is insufficient. Require token IDs within the teacher's output
vocabulary and a nonempty sequence; a one-token sequence returns a zero without
inference.

The first version scores the raw, normalized model distribution at temperature
one. vLLM 0.29 prompt scoring does not apply generation sampling processors.
Require matching student recomputation semantics: temperature one and no top-k
or top-p filtering on the current policy configuration, which shares its sampling
settings with recomputation. Reject incompatible settings before allocating GPUs.
Do not silently reinterpret an existing non-unit-temperature recipe.

## Reuse the existing runtime

Add `VllmTeacherWorkerGroup` alongside the existing Megatron teacher group.
Select the implementation once in `create_teacher_worker_groups`, with
`backend: megatron` remaining the default. Keep the Megatron implementation's
checkpoint loading and scoring behavior intact.

The new group composes `VllmGeneration`; it reuses its worker environment,
TP/PP placement, model loading, initialization, and shutdown. Add a token-scoring
RPC to the existing `VllmAsyncGenerationWorker`. Only each engine's model owner
receives scoring requests; its existing vLLM engine coordinates the internal GPU
workers. Keep scoring as a vLLM-specific operation rather than widening the
generation interface for every backend.

There are three responsibilities:

| Component | Responsibility |
|---|---|
| Existing OPD collector | Teacher routing, per-teacher lock, existing DP batch padding, advantages |
| `VllmTeacherWorkerGroup` | Build teacher-only runtime config, shard rows across independent engines, restore row order and output width |
| Existing vLLM async worker | Slice valid token IDs, submit bounded scoring requests, normalize and validate the output |

Build the teacher runtime config from clean teacher defaults. Do not deep-copy
the student's generation configuration: student quantizers, dummy weight loading,
speculation, refit settings, token capture, and rollout stop strings must not enter
the teacher. Explicitly resolve `quant_cfg=None`, `async_engine=True`,
`colocated.enabled=False`, and HTTP serving off. This selects the ordinary vLLM
worker, bypassing NeMo-RL's student ModelOpt quantization-worker dispatch. Resolve
existing resource fields into the existing vLLM runtime config rather than
creating another engine launcher.

The teacher loads real weights once with `load_format=auto`, retains the
checkpoint's `quantization_config`, and uses `logprobs_mode=raw_logprobs`.
`precision=bf16` describes the runtime's unquantized tensors; it does not request
expansion of the checkpoint to BF16. Do not force a blanket `modelopt_fp4` mode:
the checkpoint uses per-layer mixed precision. Do not attach a weight synchronizer
or call sleep/refit on this frozen, non-colocated runtime. Start with the default
KV-cache precision; NVFP4 KV is an independent numerical/performance experiment.

## Native prompt scoring

Use the pinned vLLM 0.29.0 API, with a private scoring parameter object:

```python
prompt = {"prompt_token_ids": input_ids[:valid_length].tolist()}
params = SamplingParams(
    prompt_logprobs=0,
    max_tokens=1,
    temperature=1.0,
    top_p=1.0,
    top_k=-1,
    detokenize=False,
    skip_reading_prefix_cache=True,
)
```

Submit through `AsyncLLM.generate` on the loop that owns the engine. Consume the
final output, then extract `prompt_logprobs[t][input_ids[t]].logprob` for each
`t > 0`. With zero top candidates requested, vLLM still gathers the actual prompt
token. Its first position has no probability. Ignore the one newly generated
token: the entire student response was supplied as part of the prompt.

vLLM requires at least one generated token. Therefore reserve one position:
`teacher.max_model_len >= maximum_student_trajectory_length + 1`, within the
teacher architecture's supported context. For a 131,072-token trajectory, request
131,073 positions. Reject overflow; never truncate the student trajectory. Do not
use `max_tokens=0`, which the pinned API rejects.

Disable prefix-cache reads for scoring so every required position has a logprob.
Let vLLM chunk the prefill while preserving its causal state. Validate that the
returned prompt IDs and length match the request, that each target token exists
at its position, and that valid scores are finite normalized logprobs. Missing
scores are errors, not zeros. Only the first token and actual right padding are
zero-filled.

Use the existing teacher `micro_batch_size` as the maximum number of live scoring
requests per independent engine. Begin with one. Consume, convert to a contiguous
float32 row, and release each vLLM output promptly. Do not enqueue an entire
512 × 128K batch or retain its Python logprob dictionaries. Scheduler limits
`max_num_seqs` and `max_num_batched_tokens` bound engine work separately; chunked
prefill bounds the vocabulary-sized temporary logits working set.

The externally visible payload is one scalar per token: 512 × 131,072 × 4 bytes
is 256 MiB. A top-300 target is a different contract and must not be introduced
as an incidental extension of this implementation.

## Configuration and placement

Add `backend` and one optional typed teacher `vllm_cfg` containing four controls:
`max_model_len`, `gpu_memory_utilization`, `max_num_batched_tokens`, and
`max_num_seqs`. Keep existing resource fields as the single source of truth for
TP, PP, dtype, nodes, and GPUs. Internally map these controls to the existing
runtime's `vllm_cfg` and `vllm_kwargs`; expose no teacher-level passthrough. This
avoids overlapping configuration sources and duplicate engine arguments. Add a
new control only when a concrete workload needs it. Defaults live on the config
schema, with an exemplar YAML documenting them. Reject unknown teacher vLLM
settings; arbitrary legacy Megatron overrides remain Megatron-only and are
rejected for a vLLM teacher.

Illustrative teacher block, to merge into the existing OPD recipe:

```yaml
on_policy_distillation:
  enabled: true
  teacher_model_by_agent_name:
    default_teacher: nvidia/Qwen3.8-2.4T-A95B-NVFP4
  default_teacher_alias: default_teacher
  non_colocated_teachers:
    enabled: true
    default_teacher_cfg:
      backend: vllm
      num_nodes: 4
      gpus_per_node: 4
      tensor_model_parallel_size: 16
      pipeline_model_parallel_size: 1
      micro_batch_size: 1
      vllm_cfg:
        max_model_len: 131073
        gpu_memory_utilization: 0.85
        max_num_batched_tokens: 1024
        max_num_seqs: 1
```

The 16-GPU TP16 layout is a bring-up candidate, not a validated capacity result.
Check per-rank weight residency, load-time peaks, long-context prefill workspace,
KV/recurrent state, and throughput on GB300 before fixing the allocation. The
initial supported vLLM configuration uses CP1 and no expert parallelism; ordinary
TP/PP and independent DP replicas use the existing runtime's constraints.

Reserve the teacher GPUs before the training/rollout workers, as today. For vLLM,
call `VllmGeneration.init_cluster_placement_groups` during reservation using the
resolved teacher config. A cross-node engine requires a unified placement group.
Calling plain `get_placement_groups()` first creates the default groups and makes
the later vLLM initializer reuse the wrong layout. Keep existing topology-domain
constraints and resource accounting; do not build a second scheduler.

One engine uses TP × PP GPUs; any additional engines are independent DP replicas,
each loading a complete sharded teacher. Validate divisibility against the
reserved GPUs. At the input boundary vLLM requires no SP/CP padding: its divisor
is one and it strips right padding. Adjust the shared padding helper so a
Megatron packing restriction is not incorrectly applied across backends; preserve
Megatron's own padding requirements.

Existing alias deduplication may share one checkpoint only when the resolved
backend and runtime settings agree. Reject conflicting alias overrides rather
than letting the first alias choose the backend silently.

## Errors and lifecycle

Before resource allocation, reject unsupported backends, incompatible sampling,
nondefault Megatron-only options, full/top-k OPD requests, and SingleController
with this initial backend. For this text-only checkpoint, reject media-bearing
rows before rollout admission and keep the existing runtime media validation.
Do not silently turn image/video examples into text-only examples. The choice
of text data versus another teacher for the deck's vision blend is separate.

For reproducible runs, resolve the teacher checkpoint revision to an immutable
local directory and use that path in `teacher_model_by_agent_name`. Log the
checkpoint identity, vLLM version, and quantization method with the run. Resume
must use the same teacher identity and scoring semantics.

Initialization includes actual engine readiness and a short scoring call. Actor
construction alone is insufficient. If any teacher fails to initialize, shut down
already-created teachers and release their reservations. Reuse ordinary job
checkpoint/restart behavior: teacher weights are immutable and are reloaded on
restart; there is no teacher optimizer state or refit state to checkpoint.

A scoring failure must fail the enclosing collection operation. Keep the existing
per-teacher serialization and error propagation. Handle cancellation at both
boundaries:

- The legacy collector awaits teacher work through `asyncio.to_thread`.
  Cancelling that await does not stop the thread or its remote RPC. Mirror
  `TQTeacherLogprobCoordinator.enrich`: retain the thread task, await it under
  `asyncio.shield`, and drain it before propagating cancellation. Do not enqueue
  the cancelled group. Cancellation can therefore wait for active scoring to
  finish.
- Inside the scoring RPC, abort and drain outstanding request IDs in `finally`
  on request failure or actor-local cancellation. This handles failures within
  the RPC; it does not replace collector-side draining.

The worker group also drains every dispatched DP RPC on failure: `ray.get` can
raise on one failed engine while other engines are still scoring. Only after all
RPCs finish may the collector release its teacher lock.

Use these existing cancellation patterns without a new cancellation service.
Do not add automatic fallback to another teacher or per-token retries. The first
version does not promise recovery from a lost teacher engine without job restart.

## Implementation and validation

1. Extend `TeacherResourceConfig` and resolved `TeacherConfig`, validate backend
   combinations, and dispatch the teacher factory. Preserve default Megatron
   resolution and reject conflicting deduplicated aliases.
2. Add `VllmTeacherWorkerGroup`, composing the existing runtime. Select vLLM's
   placement strategy during teacher reservation and make input padding requirements
   backend-specific. Reuse the worker group's existing DP dispatch/collection.
3. Add the async worker scoring RPC and a small output conversion function. Use
   bounded request submission, exact token IDs, strict alignment, and cleanup.
   Shield and drain the legacy collector's teacher thread on cancellation.
4. Add a text-only sampled-token OPD example. Keep the existing loss, ICE-POP,
   proximal-teacher alpha, student quantization, and rollout implementation intact.

Validate the actual failure modes:

- CPU tests: causal alignment, first-token convention, EOS, variable lengths,
  padding, batch reorder, empty batches, length-one rows, missing/nonfinite
  logprobs, conflicting configs, exact token ID preservation, bounded requests,
  and request abort/drain after failure. A collector cancellation test must show
  that the teacher thread is drained and the cancelled group is never enqueued.
- Small-model GPU test: compare vLLM teacher values to an explicit shifted
  Hugging Face log-softmax/gather for the same checkpoint and exact token IDs.
  Include a trajectory crossing a prefill-chunk boundary, repeated scoring, and
  the maximum admitted context length. Set numerical tolerances from that BF16
  comparison, not from an assumed bitwise match across engines.
- Integration smoke: one existing GRPO OPD update with a small vLLM teacher;
  confirm nonzero finite advantages and unchanged batch/mask semantics. Run the
  existing Megatron teacher tests as the backward-compatibility check.
- Target smoke: load the real mixed-precision NVFP4 teacher across nodes; inspect
  resolved quantization, score short and 128K trajectories, and measure peak GPU
  and host memory plus scoring time. Compare the adapter with direct vLLM scoring
  of that same quantized checkpoint. A BF16 2.4T reference is not required to prove
  adapter correctness; teacher quantization quality is a separate experiment.

SingleController can later reuse this exact scorer behind
`get_logprobs_from_meta`: fetch and write rows on the engine owner, preserve the
existing readiness/cancellation contract, and participate in the data-plane's
checkpoint RPCs. It should not inherit the Megatron policy worker's collective
fetch mixin or introduce a second scoring implementation.

## Local GPU smoke

With the vLLM and NeMo-Gym dependencies installed, run:

```bash
NEMO_RL_PY_EXECUTABLES_SYSTEM=1 uv run --extra vllm --extra nemo_gym \
  tests/functional/vllm_opd_teacher_smoke.py --model Qwen/Qwen3-0.6B
```

The default needs two GPUs: one teacher engine and one HF student/reference.
Use `--teacher-gpus 2` for two DP engines, or add `--teacher-tp 2` for one TP2
engine; either configuration needs a third GPU for the student. A downloaded,
immutable model directory can replace the Hub ID.

Local validation passed with both one teacher engine and two independent DP
engines. The focused teacher/OPD and shared-vocabulary suites passed 102 tests;
lint and the new modules' type checks passed.

The smoke exercises actual Ray placement, teacher factory initialization, native
scoring over a 256-token trajectory with 64-token prefill chunks, repeated
scoring, and the legacy collector's routing. It compares against HF BF16 scores,
then generates an on-policy response and runs the existing OPD advantage and
ICE-POP loss with one HF student optimizer update. It checks that student scores
change and teacher scores remain unchanged. This is a feature integration test;
the complete NeMo-Gym/Megatron training entrypoint and 128K target run remain
separate checks.

On RTX A5000, the Qwen3-0.6B BF16 reference comparison measured mean absolute
logprob error 0.0192 and maximum 0.3119 across 259 valid token positions. The smoke
bounds both mean error (0.05) and outliers (0.5), allowing differences between
attention/GEMM implementations while detecting a causal alignment error.

`examples/configs/recipes/llm/mopd-qwen3-0.6b-3n8g-megatron-vllm-teacher.yaml`
shows the teacher backend in the existing full MOPD recipe; it inherits that
recipe's text data paths and NeMo-Gym setup.

## Sources

- [QAOPD training plan](https://docs.google.com/presentation/d/19JXlkjIxptnNtsJ3NhqmUHgUC54OXlm_nWsDutx4sCE/edit?slide=id.slide_code#slide=id.slide_code).
- [Existing teacher implementation at the target commit](https://github.com/kaix-nv/RL/blob/1181179c5/nemo_rl/models/policy/teacher_worker_group.py).
- [Existing OPD setup and routing](https://github.com/kaix-nv/RL/blob/1181179c5/nemo_rl/algorithms/opd.py).
- [Existing vLLM runtime and placement](https://github.com/kaix-nv/RL/blob/1181179c5/nemo_rl/models/generation/vllm/vllm_generation.py).
- [vLLM 0.29 sampling parameters and prefix-cache behavior](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/sampling_params.py).
- [vLLM 0.29 prompt scoring and target-token gather](https://github.com/vllm-project/vllm/blob/v0.29.0/vllm/v1/worker/gpu_model_runner.py).
- [Teacher model card](https://huggingface.co/nvidia/Qwen3.8-2.4T-A95B-NVFP4) and [checkpoint configuration](https://huggingface.co/nvidia/Qwen3.8-2.4T-A95B-NVFP4/blob/main/config.json).
