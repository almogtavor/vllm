# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402  (VLLM_BATCH_INVARIANT must be set before vLLM is imported)
import os

# SPANS: some tests compare K/V across different batch shapes (e.g. an in-prompt
# span vs the same chunk computed standalone). Those are only bit-identical when
# matmuls/attention are batch-shape-invariant, so enable vLLM's batch-invariant
# mode for the spans suite. Must be set before vLLM is imported (read at import).
os.environ.setdefault("VLLM_BATCH_INVARIANT", "1")

import gc
import hashlib

import pytest
import torch

import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import BlockHash, get_request_block_hasher
from vllm.v1.request import Request

BLOCK_SIZE = 16
SPAN_TOKEN_PLUS = 10
SPAN_TOKEN_CROSS = 31
HUGE_GAP_LENGTH = 1_000_000

SEED = 42
MAX_TOKENS = 16
LOGPROBS_TOPK = 10


def greedy_sp(
    extra_args: dict | None = None,
    logprobs: int | None = None,
    max_tokens: int = MAX_TOKENS,
) -> SamplingParams:
    """Deterministic greedy SamplingParams shared by the spans e2e runs.

    `max_tokens` defaults to MAX_TOKENS; pass 1 for cache-warming runs that
    only need the prefill, not a full decode.
    """
    return SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=logprobs,
        extra_args=extra_args,
    )


def generate_single_output(llm, prompt_token_ids, sampling_params):
    """Generate from one token-id prompt; return the single RequestOutput."""
    outputs = llm.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    assert len(outputs) == 1
    return outputs[0]


MODELS = ["Qwen/Qwen3-0.6B", "NousResearch/Meta-Llama-3.1-8B-Instruct"]
LARGE_MODELS = {"NousResearch/Meta-Llama-3.1-8B-Instruct"}
LARGE_MODEL_MIN_GIB = 24


def _has_enough_gpu_for(model: str) -> bool:
    if not torch.cuda.is_available():
        return False
    if model not in LARGE_MODELS:
        return True
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return total_gib >= LARGE_MODEL_MIN_GIB


@pytest.fixture(params=MODELS, ids=["qwen3_0_6b", "llama3_1_8b"])
def model(request) -> str:
    name = request.param
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for spans e2e tests")
    if not _has_enough_gpu_for(name):
        pytest.skip(
            f"{name} needs >= {LARGE_MODEL_MIN_GIB} GiB GPU memory; "
            f"have {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GiB"
        )
    return name


def _set_spans_env(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", enabled)
    monkeypatch.setenv("VLLM_V1_SPANS_ENABLED", str(enabled))


def build_llm(
    model: str,
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    max_num_batched_tokens: int | None = None,
) -> LLM:
    """Construct an LLM configured for one of FR / SPANS / SPANS-PC / LL-16 / LL-FULL.

    SPANS-PC: spans on, prefix caching on, NO gap policy. This is the mode
    that surfaces "K/V reuse across requests": chunk + tail block hashes
    collide across requests that share chunk + tail tokens, the cache hits,
    and no recompute fires - so the K/V bytes stored under those hashes are
    whichever request wrote them first.
    """
    if mode == "FR":
        spans_enabled = False
        gap_policy_name = None
        gap_policy_config = None
        enable_prefix_caching = False
    elif mode == "SPANS":
        spans_enabled = True
        gap_policy_name = None
        gap_policy_config = None
        enable_prefix_caching = False
    elif mode == "SPANS-PC":
        spans_enabled = True
        gap_policy_name = None
        gap_policy_config = None
        enable_prefix_caching = True
    elif mode == "LL-16":
        spans_enabled = True
        gap_policy_name = "span_aware"
        gap_policy_config = {
            "gap_length": BLOCK_SIZE,
            "block_size": BLOCK_SIZE,
        }
        enable_prefix_caching = True
    elif mode == "LL-32":
        spans_enabled = True
        gap_policy_name = "span_aware"
        gap_policy_config = {
            "gap_length": 2 * BLOCK_SIZE,
            "block_size": BLOCK_SIZE,
        }
        enable_prefix_caching = True
    elif mode == "LL-FULL":
        spans_enabled = True
        gap_policy_name = "span_aware"
        gap_policy_config = {
            "gap_length": HUGE_GAP_LENGTH,
            "block_size": BLOCK_SIZE,
        }
        enable_prefix_caching = True
    else:
        raise ValueError(f"unknown mode: {mode}")

    _set_spans_env(monkeypatch, spans_enabled)

    # KV-cache snapshot helpers send a function to the workers via
    # collective_rpc; on a multiprocessing engine that payload must be
    # serializable. Set both ways: the env var is inherited by the spawned
    # engine-core/workers, the attr covers the in-process path.
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setattr(envs, "VLLM_ALLOW_INSECURE_SERIALIZATION", True)

    extra: dict = {}
    if gap_policy_name is not None:
        extra["gap_policy_name"] = gap_policy_name
        extra["gap_policy_config"] = gap_policy_config
    if max_num_batched_tokens is not None:
        extra["max_num_batched_tokens"] = max_num_batched_tokens

    return LLM(
        model=model,
        tensor_parallel_size=1,
        kv_transfer_config=None,
        gpu_memory_utilization=0.3,
        # Spans tests use <=~200-token prompts. Capping max_model_len keeps
        # the KV-cache pool small enough to fit without the engine needing
        # to be sized for the model's full (e.g. 128K) context window.
        max_model_len=2048,
        enforce_eager=True,
        block_size=BLOCK_SIZE,
        enable_prefix_caching=enable_prefix_caching,
        async_scheduling=False,
        attention_backend="TRITON_ATTN",
        **extra,
    )


def cleanup(llm: LLM | None) -> None:
    if llm is not None:
        del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def extract_step0_topk(out, topk: int = 10) -> list[tuple[int, float]]:
    """Step-0 top-K logprobs as a stably-sorted list. Bit-exact equality across
    runs ⇔ bit-exact match of the top-K distribution.
    """
    if out.logprobs is None or len(out.logprobs) == 0:
        return []
    items = [(tid, float(lp.logprob)) for tid, lp in out.logprobs[0].items()]
    items.sort(key=lambda x: (-x[1], x[0]))
    return items[:topk]


LAYER_IDX = 0  # Layer to snapshot. 0 always exists; some example models lack 19.
LAYER_IDX_KV = 1  # Layer for raw K/V comparison; >=1 so K/V is prefix-dependent.


def _rpc_first(llm, fn):
    """Run `fn` on the workers via collective_rpc; return rank 0's result."""
    results = llm.llm_engine.engine_core.collective_rpc(fn)
    assert results, "collective_rpc returned no worker results"
    return results[0]


def _kv_cache_block_hashes(llm, layer_idx: int) -> list[str]:
    """Per-block SHA-256 of layer `layer_idx` in the worker's KV cache.

    Mirrors examples/offline_inference/spans/spans_time_and_kv.py:
    _get_kv_cache_info_from_worker.
    """

    def _grab(worker_self):
        import torch

        kv = worker_self.model_runner.kv_caches[layer_idx]
        cpu = kv.detach().cpu()
        if cpu.dtype == torch.bfloat16:
            cpu = cpu.to(torch.float32)
        num_blocks = cpu.shape[0] if cpu.ndim > 0 else 1
        return [
            hashlib.sha256(cpu[i].numpy().tobytes()).hexdigest()
            for i in range(num_blocks)
        ]

    return _rpc_first(llm, _grab)


def _physical_block_tensor(llm, block_id: int, layer_idx: int = LAYER_IDX_KV):
    """Raw K/V tensor of one physical KV-cache block, on CPU as float32."""

    def _grab(worker_self):
        import torch

        kv = worker_self.model_runner.kv_caches[layer_idx][block_id]
        cpu = kv.detach().cpu()
        if cpu.dtype == torch.bfloat16:
            cpu = cpu.to(torch.float32)
        return cpu

    return _rpc_first(llm, _grab)


def _block_id_for_hash(llm, block_hash: BlockHash) -> int:
    """Physical block id currently backing `block_hash` in the prefix cache."""
    block_pool = (
        llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
    )
    cached = block_pool.get_cached_block(block_hash, [0])
    assert cached is not None, f"block hash {block_hash!r} not in prefix cache"
    return cached[0].block_id


def _block_kv(llm, block_hash: BlockHash):
    """Raw K/V tensor of the physical block currently backing `block_hash`."""
    return _physical_block_tensor(llm, _block_id_for_hash(llm, block_hash))


def _capture_request_block_ids(monkeypatch, llm) -> dict[str, list[int]]:
    """Snapshot each request's per-position physical block ids, keyed by id.

    PIC occurrences collide in the block *hash*, so per-occurrence K/V can
    only be read by logical position from the request's block table. That
    table is popped on free, so snapshot it just before the free runs.
    """
    kcm = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager
    captured: dict[str, list[int]] = {}

    def _free(request):
        captured[request.request_id] = kcm.get_block_ids(request.request_id)[0]
        return type(kcm).free(kcm, request)  # original method off the class

    monkeypatch.setattr(kcm, "free", _free)
    return captured


def _force_in_process_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run EngineCore in-process.

    `_block_id_for_hash` reads the scheduler-side block pool, which lives
    in the EngineCore process; running in-process makes it reachable, and
    makes test-computed block_hashes share the engine's NONE_HASH.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


def _warmup_prompt(
    llm,
    prompt_token_ids: list[int],
    extra_args: dict | None = None,
) -> None:
    """Populate prefix cache entries for the given full-block prompt.

    When the prompt is exactly the PIC chunk, this creates the same
    NONE_HASH-rooted block entry that a later span-start block should hit.
    """
    generate_single_output(llm, prompt_token_ids, greedy_sp(extra_args, max_tokens=1))


def _generate_num_cached_tokens(
    llm,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> int:
    out = generate_single_output(llm, prompt_token_ids, sampling_params)
    num_cached_tokens = out.num_cached_tokens
    assert num_cached_tokens is not None
    return num_cached_tokens


def _request_block_hashes(
    prompt_token_ids: list[int],
    span_starts: list[int] | None,
    cross_span_starts: list[int] | None = None,
) -> list[BlockHash]:
    hash_fn = get_hash_fn_by_name("sha256")
    if not hasattr(kv_cache_utils, "NONE_HASH"):
        kv_cache_utils.init_none_hash(hash_fn)

    extra_args = {}
    if span_starts is not None:
        extra_args["span_starts"] = span_starts
    if cross_span_starts is not None:
        extra_args["cross_span_starts"] = cross_span_starts
    extra_args = extra_args or None
    sp = greedy_sp(extra_args)
    sp.update_from_generation_config({}, eos_token_id=100)
    req = Request(
        request_id="hash_probe",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, hash_fn),
    )
    return req.block_hashes
