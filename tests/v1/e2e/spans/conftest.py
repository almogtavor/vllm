# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared fixtures and helpers for spans / Legolink e2e tests.

Mode names:
    FR      - VLLM_V1_SPANS_ENABLED=False, no gap policy (full recompute baseline).
    SPANS   - VLLM_V1_SPANS_ENABLED=True, no gap policy.
    LL-16   - Legolink: span-aware gap policy with gap_length == block_size,
              prefix caching enabled.
    LL-FULL - Legolink: span-aware gap policy with gap_length >> prompt length,
              prefix caching enabled. On a cache hit this forces a full
              recompute of the cached prefix, which means run #2 ≡ FR.
"""
import gc

import pytest
import torch

import vllm.envs as envs
from vllm import LLM, SamplingParams

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
