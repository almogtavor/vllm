# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-mode output equivalence for spans / Legolink.

Migrated from examples/offline_inference/spans/basic_spans_determinism.py:
the example's diagnostics (top-K logprob equivalence, multi-seed drift,
gap-policy replay) become the assertions of these tests.

The contract:
  * On a prompt with no PIC chunk, FR == SPANS == LL-16 == LL-FULL
    (text + top-K logprobs bit-identical at temp=0).
  * On a prompt whose first 16 tokens are a PIC chunk and no preload
    (cache empty), the four modes still agree because no cache hit means
    the gap policy never fires.
  * LL-FULL with prefix-caching ON, run twice: run #2 hits the cache,
    gap policy with gap_length >> prompt forces a full recompute, output
    must equal a clean FR reference.
"""
import pytest

from vllm import SamplingParams

from .conftest import (
    BLOCK_SIZE,
    build_llm,
    cleanup,
    extract_step0_topk,
)

pytestmark = pytest.mark.spans

SEED = 42
MAX_TOKENS = 16
LOGPROBS_TOPK = 10
PLAIN_PROMPT = "Hello world! Please write a short greeting in one sentence."

ALL_MODES = ("FR", "SPANS", "LL-16", "LL-FULL")


def _greedy_params(extra_args: dict | None = None) -> SamplingParams:
    return SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=LOGPROBS_TOPK,
        extra_args=extra_args,
    )


def _run(llm, prompt: str, extra_args: dict | None = None):
    res = llm.generate(prompt, sampling_params=_greedy_params(extra_args), use_tqdm=False)
    out = res[0].outputs[0]
    return out.text, extract_step0_topk(out, LOGPROBS_TOPK)


def _reference_results(model: str, prompt: str, monkeypatch, extra_args=None):
    """Run the prompt under each mode and return {mode: (text, top10)}.

    Each mode gets its own fresh LLM (modes change global env + LLM kwargs)."""
    out: dict[str, tuple[str, list]] = {}
    for mode in ALL_MODES:
        llm = build_llm(model, mode, monkeypatch)
        try:
            out[mode] = _run(llm, prompt, extra_args)
        finally:
            cleanup(llm)
    return out


def test_no_pic_all_modes_match(model, monkeypatch):
    """Plain prompt, no PIC: FR == SPANS == LL-16 == LL-FULL."""
    results = _reference_results(model, PLAIN_PROMPT, monkeypatch, extra_args=None)
    fr_text, fr_top = results["FR"]
    for mode in ALL_MODES:
        text, top = results[mode]
        assert text == fr_text, f"{mode} text drifted vs FR"
        assert top == fr_top, f"{mode} top-{LOGPROBS_TOPK} logprobs drifted vs FR"


def test_pic_at_start_all_modes_match(model, monkeypatch):
    """PIC chunk at position 0 (<= 16 tokens), no preload → no cache hit, so
    even Legolink modes match FR.
    """
    extra_args = {"span_starts": [0]}
    results = _reference_results(model, PLAIN_PROMPT, monkeypatch, extra_args=extra_args)
    fr_text, fr_top = results["FR"]
    for mode in ALL_MODES:
        text, top = results[mode]
        assert text == fr_text, f"{mode} text drifted vs FR with PIC at start"
        assert top == fr_top, f"{mode} top-{LOGPROBS_TOPK} drifted vs FR with PIC at start"


def test_legolink_gap_huge_equals_full_recompute(model, monkeypatch):
    """Verify that 5 ostensibly-equivalent configurations all produce the
    same next-token top-K logprobs on the same prompt.

    Modes compared (all on a 4-block, 64-token prompt with no padding):

      1. FR              regular vLLM, no spans, no gap policy.
      2. SPANS + no_sp   VLLM_V1_SPANS_ENABLED=True but no span_starts on
                         the request - the PIC code path never fires.
      3. SPANS + sp[0]   VLLM_V1_SPANS_ENABLED=True with span_starts=[0]
                         (entire prompt is one span). The PIC reset at
                         block 0 is a no-op (block 0's parent was already
                         None), so the hash chain is unchanged.
      4. LL-FULL+no_sp   gap_policy=span_aware, gap_length=huge, prefix
                         caching ON, but no span_starts -> gap policy
                         early-returns []; cold + replay both produce
                         FR-equivalent K/V.
      5. LL-FULL+sp[0]   same as 4 but with span_starts=[0]. Cold prefill
                         is FR-equivalent (PIC at 0 is a no-op). On
                         replay the gap policy fires gap=(0, num_computed),
                         re-prefills the entire prompt against the actual
                         prefix; output must still match FR.

    None of these configurations should actually change the K/V the model
    sees. Strict top-K logprob equality is the assertion.

    This test does NOT use a chunk warmup, because warmup would only
    matter for cross-request PIC fan-in (which is covered by tests 1/3).
    Here we're just asserting that all "no-real-PIC-fan-in" code paths
    produce identical numerical output.
    """
    prompt_tokens = list(range(0, BLOCK_SIZE * 4))

    def _greedy(extra_args=None):
        return SamplingParams.from_optional(
            seed=SEED,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            logprobs=LOGPROBS_TOPK,
            extra_args=extra_args,
        )

    def _run_tokens(llm, sp):
        res = llm.generate(
            {"prompt_token_ids": prompt_tokens},
            sampling_params=sp,
            use_tqdm=False,
        )
        out = res[0].outputs[0]
        return out.text, extract_step0_topk(out, LOGPROBS_TOPK)

    # Mode 1: FR baseline.
    fr_llm = build_llm(model, "FR", monkeypatch)
    try:
        ref_text, ref_top = _run_tokens(fr_llm, _greedy())
    finally:
        cleanup(fr_llm)

    results: dict[str, tuple[str, list]] = {"FR": (ref_text, ref_top)}

    # Mode 2: SPANS, no span_starts.
    spans_llm = build_llm(model, "SPANS", monkeypatch)
    try:
        results["SPANS+no_sp"] = _run_tokens(spans_llm, _greedy())
    finally:
        cleanup(spans_llm)

    # Mode 3: SPANS, span_starts=[0] (entire prompt is one span).
    spans_llm = build_llm(model, "SPANS", monkeypatch)
    try:
        results["SPANS+sp[0]"] = _run_tokens(
            spans_llm, _greedy(extra_args={"span_starts": [0]})
        )
    finally:
        cleanup(spans_llm)

    # Mode 4: LL-FULL, no span_starts. Cold then replay (gap won't fire
    # either way, but we run twice to mirror modes 5 + the cache state).
    ll_llm = build_llm(model, "LL-FULL", monkeypatch)
    try:
        _run_tokens(ll_llm, _greedy())  # cold
        results["LL-FULL+no_sp"] = _run_tokens(ll_llm, _greedy())  # replay
    finally:
        cleanup(ll_llm)

    # Mode 5: LL-FULL, span_starts=[0]. Cold + replay; on replay the gap
    # policy fires gap=(0, num_computed) and re-prefills the entire
    # prompt against the current prefix.
    ll_llm = build_llm(model, "LL-FULL", monkeypatch)
    try:
        _run_tokens(ll_llm, _greedy(extra_args={"span_starts": [0]}))  # cold
        results["LL-FULL+sp[0]"] = _run_tokens(
            ll_llm, _greedy(extra_args={"span_starts": [0]})
        )  # replay
    finally:
        cleanup(ll_llm)

    # All modes must produce bit-identical text + top-K to FR.
    drifts: list[str] = []
    for mode, (text, top) in results.items():
        if mode == "FR":
            continue
        if text != ref_text:
            drifts.append(
                f"{mode} text drift:\n  ref:    {ref_text!r}\n  got:    {text!r}"
            )
        if top != ref_top:
            ref_ids = [tid for tid, _ in ref_top]
            got_ids = [tid for tid, _ in top]
            drifts.append(
                f"{mode} top-{LOGPROBS_TOPK} drift:\n"
                f"  ref ids: {ref_ids}\n"
                f"  got ids: {got_ids}"
            )

    assert not drifts, "Mode equivalence failed:\n\n" + "\n\n".join(drifts)


def test_block_size_constant_matches_conftest():
    """Defensive: tests assume block_size == 16. If conftest changes, tests
    that rely on PIC alignment must be revisited."""
    assert BLOCK_SIZE == 16
