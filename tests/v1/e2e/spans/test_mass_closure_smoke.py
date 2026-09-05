"""MassClosure e2e smoke: the production env path (arg_utils picks the policy
from VLLM_V1_SPANS_MASS_CLOSURE_ENABLE) must build, reuse a warmed span, and
return - including when gap overhead exceeds the step budget, the shape that
stalled every span arm on the grid image."""

import threading

import pytest

from .conftest import (
    BLOCK_SIZE,
    _force_in_process_engine,
    _generate_num_cached_tokens,
    _warmup_prompt,
    build_llm,
    cleanup,
    greedy_sp,
)

pytestmark = pytest.mark.spans


def _mc_env(monkeypatch, k_per_span):
    monkeypatch.setenv("VLLM_V1_SPANS_PREROTATE", "True")
    monkeypatch.setenv("VLLM_V1_SPANS_MASS_CLOSURE_ENABLE", "True")
    monkeypatch.setenv("VLLM_V1_SPANS_QCFUSE_K_PER_SPAN", str(k_per_span))
    monkeypatch.setenv("VLLM_V1_SPANS_QCFUSE_RHO", "0.15")
    monkeypatch.setenv("VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "20,24,26")


def _run_within(fn, seconds, what):
    done = threading.Event()
    out = {}
    threading.Thread(
        target=lambda: (out.setdefault("v", fn()), done.set()), daemon=True
    ).start()
    assert done.wait(timeout=seconds), what
    return out["v"]


def test_mass_closure_policy_is_selected_from_env(model, monkeypatch):
    _mc_env(monkeypatch, k_per_span=2 * BLOCK_SIZE)
    _force_in_process_engine(monkeypatch)
    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        sched = llm.llm_engine.engine_core.engine_core.scheduler
        assert type(sched.gap_policy).__name__ == "MassClosurePolicy", type(
            sched.gap_policy
        )
    finally:
        cleanup(llm)


def test_mass_closure_reuse_returns(model, monkeypatch):
    _mc_env(monkeypatch, k_per_span=2 * BLOCK_SIZE)
    prompt = list(range(8 * BLOCK_SIZE))
    xa = {"span_starts": [2 * BLOCK_SIZE], "cross_span_starts": [5 * BLOCK_SIZE]}
    sp = greedy_sp(xa)
    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        _warmup_prompt(llm, prompt, xa)
        for i in range(3):  # 2nd/3rd reuse exercise the memo + probe-seeded path
            n = _run_within(
                lambda: _generate_num_cached_tokens(llm, prompt, sp),
                90,
                f"reuse #{i} did not return within 90s",
            )
            assert n > 0, "no prefix reuse on a warmed prompt"
    finally:
        cleanup(llm)


def test_mass_closure_overhead_above_budget_does_not_stall(model, monkeypatch):
    # 64 blocks, 16 spans of 4 blocks, K=4 blocks -> gap overhead 1024 > budget 512.
    # On the grid image this shape hit scheduler.py:851 `break` every step.
    _mc_env(monkeypatch, k_per_span=4 * BLOCK_SIZE)
    prompt = list(range(64 * BLOCK_SIZE))
    starts = [i * 4 * BLOCK_SIZE for i in range(16)]
    xa = {
        "span_starts": starts,
        "cross_span_starts": [s + 4 * BLOCK_SIZE for s in starts],
    }
    sp = greedy_sp(xa)
    llm = build_llm(model, "SPANS-PC", monkeypatch, max_num_batched_tokens=512)
    try:
        _warmup_prompt(llm, prompt, xa)
        _run_within(
            lambda: _generate_num_cached_tokens(llm, prompt, sp),
            120,
            "engine stalled: gap overhead >= budget and the request never ran",
        )
    finally:
        cleanup(llm)
