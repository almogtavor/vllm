# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading

import pytest
import torch

from .conftest import (
    BLOCK_SIZE,
    LOGPROBS_TOPK,
    _block_kv,
    _force_in_process_engine,
    _generate_num_cached_tokens,
    _request_block_hashes,
    _warmup_prompt,
    build_llm,
    cleanup,
    extract_step0_topk,
    generate_single_output,
    greedy_sp,
)

pytestmark = pytest.mark.spans


def test_legolink_partial_recompute_within_gap_interval(model, monkeypatch):
    """LL-32 gap_length = 2 blocks bounds the recompute at the block level.

    Span is 4 blocks, gap_length covers only the first 2. After the run:
      - the first 2 span blocks were recomputed against the real prefix
        (K/V differs from the pre-warmed stale K/V);
      - the last 2 span blocks were NOT touched (K/V byte-identical to
        the stale warmup K/V) - gap_length is a hard upper bound.
    """
    _force_in_process_engine(monkeypatch)

    prefix = list(range(0, BLOCK_SIZE * 2))
    span = list(range(500, 500 + BLOCK_SIZE * 4))
    prompt_tokens = prefix + span
    sp = greedy_sp({"span_starts": [BLOCK_SIZE * 2]})

    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        # Warm the span standalone (stale: NONE-rooted at positions 0-63 vs
        # the marked prompt's positions 32-95) and the prefix. Capture the
        # stale span K/V via the standalone span's hashes (which the marked
        # prompt's span blocks 2-5 also hash to).
        _warmup_prompt(llm, span)
        _warmup_prompt(llm, prefix)
        span_hashes = _request_block_hashes(span, span_starts=None)
        stale_span_kv = [_block_kv(llm, h) for h in span_hashes]

        # Run the marked prompt - gap policy fires on the span at block 2
        # with gap (32, 64) and recomputes only the first 2 span blocks.
        _generate_num_cached_tokens(llm, prompt_tokens, sp)
        after_span_kv = [_block_kv(llm, h) for h in span_hashes]

        # Span blocks 0,1 (= prompt blocks 2,3) ARE the gap interval -
        # their K/V must differ from the stale warmup K/V.
        for i in (0, 1):
            assert not torch.allclose(
                stale_span_kv[i], after_span_kv[i], atol=2e-2, rtol=2e-2
            ), (
                f"span block {i} (prompt block {i + 2}, inside gap) was NOT "
                f"recomputed - still matches the stale warmup K/V"
            )

        # Span blocks 2,3 (= prompt blocks 4,5) are OUTSIDE the gap - the
        # gap policy must not touch them; their K/V must be byte-identical
        # to the warmup.
        for i in (2, 3):
            assert torch.equal(stale_span_kv[i], after_span_kv[i]), (
                f"span block {i} (prompt block {i + 2}, outside gap) was "
                f"wrongly touched - gap_length = 2 blocks should not reach "
                f"this far"
            )
    finally:
        cleanup(llm)


def test_legolink_recompute_precedes_cross_tail_and_decode_e2e(model, monkeypatch):
    """LL gap recompute runs before the cross-tail prefill and the decode.
    The 2-block PIC span is pre-warmed standalone (stale, context-free K/V).
    The marked LL-32 run hits prefix+span, gap-recomputes the span against the
    real prefix, then prefills the cross-tail and decodes. The cross-tail K/V
    and decoded top-K match a no-marker FR reference only if they consumed the
    recomputed span - proving the ordering by data dependency.
    """
    _force_in_process_engine(monkeypatch)
    prefix = list(range(0, BLOCK_SIZE * 2))
    span = list(range(500, 500 + BLOCK_SIZE * 2))
    cross_tail = list(range(900, 900 + BLOCK_SIZE * 2))
    cold_suffix = list(range(2000, 2000 + BLOCK_SIZE))
    prompt = prefix + span + cross_tail + cold_suffix
    span_starts = [BLOCK_SIZE * 2]
    cross_span_starts = [BLOCK_SIZE * 4]

    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        # Phase 1: no markers -> gap policy no-op -> FR-equivalent reference.
        ref_hashes = _request_block_hashes(prompt, span_starts=None)
        ref_out = generate_single_output(llm, prompt, greedy_sp(logprobs=LOGPROBS_TOPK))
        ref_top = extract_step0_topk(ref_out.outputs[0], LOGPROBS_TOPK)
        ref_span_kv = [_block_kv(llm, h) for h in ref_hashes[2:4]]
        ref_cross_tail_kv = [_block_kv(llm, h) for h in ref_hashes[4:6]]

        # Phase 2: warm span (standalone, stale) then prefix, so Phase 3 hits both.
        standalone_span_hashes = _request_block_hashes(span, span_starts=None)
        _warmup_prompt(llm, span)
        _warmup_prompt(llm, prefix)
        stale_span_kv = [_block_kv(llm, h) for h in standalone_span_hashes]

        # Premise: the span K/V to be hit is stale, so the recompute isn't a no-op.
        assert not any(
            torch.allclose(s, r, atol=2e-2, rtol=2e-2)
            for s, r in zip(stale_span_kv, ref_span_kv)
        ), "warmed span K/V already matches the reference; recompute is a no-op"

        # Phase 3: marked LL request hits prefix+span, recomputes, prefills, decodes.
        marked_hashes = _request_block_hashes(
            prompt, span_starts=span_starts, cross_span_starts=cross_span_starts
        )
        marked_out = generate_single_output(
            llm,
            prompt,
            greedy_sp(
                extra_args={
                    "span_starts": span_starts,
                    "cross_span_starts": cross_span_starts,
                },
                logprobs=LOGPROBS_TOPK,
            ),
        )
        cached_marked = marked_out.num_cached_tokens
        actual_top = extract_step0_topk(marked_out.outputs[0], LOGPROBS_TOPK)
        actual_span_kv = [_block_kv(llm, h) for h in marked_hashes[2:4]]
        actual_cross_tail_kv = [_block_kv(llm, h) for h in marked_hashes[4:6]]

        assert cached_marked == BLOCK_SIZE * 4, (
            f"marked request should hit prefix + span (4 blocks); "
            f"got cached={cached_marked}"
        )
        for a, r in zip(actual_span_kv, ref_span_kv):
            assert torch.allclose(a, r, atol=2e-2, rtol=2e-2), (
                "gap recompute did not restore the context-aware span K/V"
            )
        # Ordering proof: cross-tail K/V matches the reference only if the
        # post-cross prefill ran after the recompute (else it sees stale span).
        for a, r in zip(actual_cross_tail_kv, ref_cross_tail_kv):
            assert torch.allclose(a, r, atol=2e-2, rtol=2e-2), (
                "cross-tail K/V differs from reference - prefill ran before recompute"
            )
        # ...and the decoded top-K matches only if decode ran after the recompute.
        assert actual_top[0][0] == ref_top[0][0], (
            f"decoded top-1 token drift: ref={ref_top[0]}, got={actual_top[0]}"
        )
        assert {t for t, _ in actual_top} == {t for t, _ in ref_top}, (
            "decoded top-K candidate set drifted - decode read stale K/V"
        )
    finally:
        cleanup(llm)


def test_large_gap_length_does_not_livelock_e2e(model, monkeypatch):
    """When gap_overhead >= max_num_batched_tokens the scheduler run num_new_tokens <= 0
    every step (req never schedules). Red if generate does not return within a timeout.
    """
    prompt = list(range(1024))  # 64 blocks, > the 512-token batch budget
    sp = greedy_sp({"span_starts": [0]})  # LL-FULL gap spans the whole prefix
    llm = build_llm(model, "LL-FULL", monkeypatch, max_num_batched_tokens=512)
    try:
        _warmup_prompt(llm, prompt)  # cold prefill caches all blocks
        done = threading.Event()
        threading.Thread(
            target=lambda: (_generate_num_cached_tokens(llm, prompt, sp), done.set()),
            daemon=True,
        ).start()
        assert done.wait(timeout=60), (
            "engine livelocked: large gap_length tiled the cached prefix so "
            "gap_overhead >= max_num_batched_tokens and the request never ran"
        )
    finally:
        cleanup(llm)
