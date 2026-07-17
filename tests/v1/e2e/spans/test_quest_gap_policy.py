# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QUEST (write-side): the gap budget recomputes the span blocks the
following query attends most, wherever they sit in the span, instead of the
span's first ``gap_length`` tokens.

One QUEST engine (gap_length = 2 blocks) serves every phase:
  1. Ground truth: a no-span-marker run of prefixB+span+tail caches each
     span block's prefix-aware K/V under plain chain hashes.
  2. First occurrence: prefixA+span+tail computes the span prefix-free; the
     worker scores its 6 blocks against the first post-span query (summed
     over layers) and the scheduler stores the top-2 block offsets.
  3. Reuse: prefixB+span+tail pic-hits the span. The gap policy must emit
     one single-block gap per SELECTED offset - each recomputed against its
     full causal prefix (prefixB + all earlier span tokens) into a fresh
     pd-keyed block - while unselected blocks stay on the shared warmed
     copy, untouched.
"""

import os

import pytest
import torch

from .conftest import (
    BLOCK_SIZE,
    _block_kv,
    _capture_request_block_ids,
    _force_in_process_engine,
    _generate_num_cached_tokens,
    _physical_block_tensor,
    _request_block_hashes,
    _warmup_prompt,
    build_llm,
    cleanup,
    greedy_sp,
)

pytestmark = pytest.mark.spans

N_SPAN_BLOCKS = 6
TOP_K = 2  # gap_length (2 blocks) / block_size


@pytest.mark.timeout(600)
def test_quest_recomputes_query_selected_span_blocks_e2e(model, monkeypatch):
    _force_in_process_engine(monkeypatch)
    monkeypatch.setenv("VLLM_QUEST_CAPTURE", "1")

    prefix_a = list(range(0, BLOCK_SIZE * 2))
    prefix_b = list(range(2000, 2000 + BLOCK_SIZE * 2))
    span = list(range(500, 500 + BLOCK_SIZE * N_SPAN_BLOCKS))
    tail = list(range(1200, 1200 + BLOCK_SIZE))
    span_start = BLOCK_SIZE * 2
    cross = span_start + BLOCK_SIZE * N_SPAN_BLOCKS
    extra_args = {"span_starts": [span_start], "cross_span_starts": [cross]}
    span_blk0 = span_start // BLOCK_SIZE

    llm = build_llm(model, "QUEST", monkeypatch)
    try:
        # Phase 1: prefix-aware ground truth (no span markers -> plain chain
        # hashes; the QUEST gap policy is a no-op without span_starts).
        prompt_b = prefix_b + span + tail
        _warmup_prompt(llm, prefix_b)
        _warmup_prompt(llm, prompt_b)
        ref_hashes = _request_block_hashes(prompt_b, span_starts=None)
        correct_kv = [
            _block_kv(llm, ref_hashes[span_blk0 + o]) for o in range(N_SPAN_BLOCKS)
        ]

        captured = _capture_request_block_ids(monkeypatch, llm)

        # Phase 2: first occurrence scores the span for its following query.
        prompt_a = prefix_a + span + tail
        _warmup_prompt(llm, prompt_a, extra_args=extra_args)
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        selections = scheduler.gap_policy.selections
        assert len(selections) == 1, (
            f"first occurrence must store exactly one span selection, "
            f"got {len(selections)}"
        )
        sel = sorted(next(iter(selections.values())))
        assert len(sel) == TOP_K and all(0 <= o < N_SPAN_BLOCKS for o in sel), (
            f"selection must be {TOP_K} span-block offsets, got {sel}"
        )
        # Plumbing check: the stored offsets are the layer-summed capture top-K.
        from vllm.v1.attention.backends.triton_attn import QUEST_CAPTURE

        assert QUEST_CAPTURE, "VLLM_QUEST_CAPTURE=1 captured no span scores"
        summed = torch.stack([s for s, _ in QUEST_CAPTURE]).sum(0)
        assert sorted(summed.topk(TOP_K).indices.tolist()) == sel
        if os.environ.get("QUEST_CAPTURE_OUT"):
            torch.save(QUEST_CAPTURE, os.environ["QUEST_CAPTURE_OUT"])

        a_id = max(captured, key=lambda k: len(captured[k]))
        a_ids = captured[a_id]
        warmed_kv = [
            _physical_block_tensor(llm, a_ids[span_blk0 + o])
            for o in range(N_SPAN_BLOCKS)
        ]

        # Phase 3: reuse behind a different (warmed) prefix. Spy on the gap
        # policy to record the exact recompute ranges it emits (virtual gap
        # requests never reach kv_cache_manager.free, so block-table capture
        # cannot see them).
        emitted: set[tuple[tuple[int, int], ...]] = set()
        orig_get_gaps = scheduler.gap_policy.get_gaps

        def _spy(request, num_computed_tokens, num_external_tokens):
            gaps = orig_get_gaps(request, num_computed_tokens, num_external_tokens)
            if gaps:
                emitted.add(tuple(gaps))
            return gaps

        monkeypatch.setattr(scheduler.gap_policy, "get_gaps", _spy)

        pre_keys = set(captured)
        cached = _generate_num_cached_tokens(llm, prompt_b, greedy_sp(extra_args))
        assert cached == cross, (
            f"reuse run should hit prefixB + the whole span ({cross} tokens), "
            f"got {cached}"
        )
        expected: list[tuple[int, int]] = []
        for o in sel:  # adjacent selected blocks coalesce into one gap
            s, e = span_start + o * BLOCK_SIZE, span_start + (o + 1) * BLOCK_SIZE
            if expected and expected[-1][1] == s:
                expected[-1] = (expected[-1][0], e)
            else:
                expected.append((s, e))
        assert emitted == {tuple(expected)}, (
            f"gap policy must emit exactly the Quest-selected blocks {sel} "
            f"(adjacent ones coalesced), got {emitted}"
        )
        new_keys = set(captured) - pre_keys
        b_id = max(new_keys, key=lambda k: len(captured[k]))

        b_ids = captured[b_id]
        for o in range(N_SPAN_BLOCKS):
            a_blk, b_blk = a_ids[span_blk0 + o], b_ids[span_blk0 + o]
            b_kv = _physical_block_tensor(llm, b_blk)
            if o in sel:
                assert b_blk != a_blk, (
                    f"selected span block {o} must be swapped to a fresh block"
                )
                assert torch.allclose(b_kv, correct_kv[o], atol=2e-2, rtol=2e-2), (
                    f"selected span block {o} was recomputed but does not match "
                    f"its prefix-aware ground truth"
                )
            else:
                assert b_blk == a_blk, (
                    f"unselected span block {o} must stay on the shared warmed "
                    f"block"
                )
                assert torch.equal(b_kv, warmed_kv[o]), (
                    f"unselected span block {o} was modified by the recompute"
                )
    finally:
        cleanup(llm)
