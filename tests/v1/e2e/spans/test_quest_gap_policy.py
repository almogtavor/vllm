# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
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


def test_quest_gap_recompute_selects_critical_prefix_blocks_e2e(model, monkeypatch):
    """QUEST mode: the span's gap recompute reads only the top-K prefix key
    blocks ranked by the Quest upper bound against the parent's post-span
    query. Checks, per layer: the score ranks 4 prefix blocks, the recompute
    still lands (span K/V diverges from the stale warmup), and the Quest
    approximation tracks the true attention mass (its top-1 block is inside
    the Quest top-2 for most layers). Captured pairs feed the approx-vs-real
    plot via QUEST_CAPTURE_OUT."""
    _force_in_process_engine(monkeypatch)
    monkeypatch.setenv("VLLM_QUEST_CAPTURE", "1")
    from vllm.v1.attention.backends.triton_attn import QUEST_CAPTURE

    QUEST_CAPTURE.clear()
    prefix = list(range(0, BLOCK_SIZE * 4))
    span = list(range(500, 500 + BLOCK_SIZE * 2))
    tail = list(range(900, 900 + BLOCK_SIZE))
    prompt = prefix + span + tail
    sp = greedy_sp(
        {"span_starts": [BLOCK_SIZE * 4], "cross_span_starts": [BLOCK_SIZE * 6]}
    )

    llm = build_llm(model, "QUEST", monkeypatch)
    try:
        _warmup_prompt(llm, span)
        _warmup_prompt(llm, prefix)
        stale_span_kv = [
            _block_kv(llm, h) for h in _request_block_hashes(span, span_starts=None)
        ]
        captured = _capture_request_block_ids(monkeypatch, llm)
        cached = _generate_num_cached_tokens(llm, prompt, sp)
        assert cached == BLOCK_SIZE * 6, f"expected prefix+span hit, got {cached}"
        block_ids = max(captured.values(), key=len)
        after_span_kv = [
            _physical_block_tensor(llm, block_ids[i + 4]) for i in range(2)
        ]
    finally:
        cleanup(llm)

    # The gap recompute ran under the Quest restriction and rewrote the span.
    for i in range(2):
        assert not torch.allclose(
            stale_span_kv[i], after_span_kv[i], atol=2e-2, rtol=2e-2
        ), f"span block {i} was not recomputed under QUEST"

    # Quest fired once per layer, scoring all 4 prefix blocks with budget 2.
    assert QUEST_CAPTURE, "Quest never scored a gap recompute"
    assert all(len(s) == 4 and len(m) == 4 for s, m in QUEST_CAPTURE)
    hits = sum(
        int(mass.argmax().item() in score.topk(2).indices.tolist())
        for score, mass in QUEST_CAPTURE
    )
    recall = hits / len(QUEST_CAPTURE)
    assert recall >= 0.7, (
        f"Quest top-2 contains the true top-1 block in only {recall:.0%} of "
        f"{len(QUEST_CAPTURE)} layer scorings - the upper bound is not tracking "
        f"real attention"
    )
    out = os.environ.get("QUEST_CAPTURE_OUT")
    if out:
        torch.save(QUEST_CAPTURE, out)
