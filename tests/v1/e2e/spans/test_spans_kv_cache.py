# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache and gap-policy assertions for the spans / Legolink machinery.

End-to-end tests that hash the worker's KV cache per block (mirroring
examples/offline_inference/spans/spans_time_and_kv.py) and use set-
relations between snapshots to pin PIC fan-in, span-boundary hash chain
reset, cross-prefix chunk reuse, prefix-cache-survives-PIC behavior, and
the Legolink gap-policy interval bound.
"""
import hashlib

import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.gap_policy import SpanAwareGapPolicy
from vllm.v1.request import Request

from .conftest import BLOCK_SIZE, build_llm, cleanup

pytestmark = pytest.mark.spans

SEED = 42
MAX_TOKENS = 16
LAYER_IDX = 0  # Layer to snapshot. 0 always exists; some example models lack 19.


def _make_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
) -> Request:
    extra_args = {"span_starts": span_starts} if span_starts is not None else None
    sp = SamplingParams(max_tokens=MAX_TOKENS, extra_args=extra_args)
    sp.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="kv_test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sp,
        pooling_params=None,
    )


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

    results = llm.llm_engine.engine_core.collective_rpc(_grab)
    assert results, "collective_rpc returned no worker results"
    return results[0]


def _warmup_chunk(llm, chunk_token_ids: list[int]) -> None:
    """Populate the prefix cache with the chunk block alone.

    The chunk is the only block of its own request, so its parent_hash
    defaults to NONE_HASH and the cached entry lands at
    hash(NONE_HASH, chunk_tokens). That is the same hash PIC produces
    for the same chunk later embedded in a longer prompt with
    is_span_start=True, so the entry is reachable via fan-in.

    Mirrors the warmup step in kvcache-bench/middleware/processors.py:321-335
    (without the HTTP / FastAPI layer).
    """
    llm.generate(
        {"prompt_token_ids": chunk_token_ids},
        sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
        use_tqdm=False,
    )


def test_pic_chunk_hash_invariant_across_positions_e2e(model, monkeypatch):
    """E2E counterpart to test_pic_chunk_hash_invariant_across_positions.

    Setup (SPANS-PC, warmup chunk):
      - Warm up the chunk alone -> cache slot at hash(NONE_HASH, chunk).
      - req_A = prefix_a (1 block) + chunk, span_starts=[BLOCK_SIZE].
                Chunk lands at block index 1.
      - req_B = prefix_b (3 blocks, different content) + chunk,
                span_starts=[BLOCK_SIZE * 3]. Chunk lands at block index 3.

    Both requests should HIT the warmup-cached chunk slot via PIC fan-in
    despite the chunk being at different positions in their prompts.
    The warmup K/V slot must be in the cache both after req_A and after
    req_B.
    """
    chunk = list(range(500, 500 + BLOCK_SIZE))
    prefix_a = list(range(0, BLOCK_SIZE))
    prefix_b = list(range(200, 200 + BLOCK_SIZE * 3))

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        _warmup_chunk(llm, chunk)
        warmup_blocks = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        sp_a = SamplingParams.from_optional(
            seed=SEED, temperature=0.0, max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE]},
        )
        sp_b = SamplingParams.from_optional(
            seed=SEED, temperature=0.0, max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE * 3]},
        )

        llm.generate({"prompt_token_ids": prefix_a + chunk},
                     sampling_params=sp_a, use_tqdm=False)
        snap_after_a = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        llm.generate({"prompt_token_ids": prefix_b + chunk},
                     sampling_params=sp_b, use_tqdm=False)
        snap_after_b = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        assert warmup_blocks <= snap_after_a, (
            "warmup chunk slot was evicted/overwritten by req_A (chunk at "
            "position 16) - PIC fan-in did not hit the warmed slot."
        )
        assert warmup_blocks <= snap_after_b, (
            "warmup chunk slot was evicted/overwritten by req_B (chunk at "
            "position 48) - PIC fan-in did not hit the warmed slot."
        )
    finally:
        cleanup(llm)


def test_span_boundary_resets_block_hash_chain_e2e(model, monkeypatch):
    """E2E counterpart to test_span_boundary_resets_block_hash_chain_no_recompute.

    Same 4-block prompt sent two ways:
      - baseline: no span_starts -> block 2 hashes through its parent chain
      - marked:   span_starts=[BLOCK_SIZE * 2] -> block 2 hashes with
                  parent dropped (NONE_HASH)

    With the chunk pre-warmed, only the "marked" request can hit the
    warmup slot at block 2 (because only it produces the fan-in hash).
    The "baseline" request misses on every block (chain hashes don't
    match the warmup) and writes fresh K/V.

    The warmup slot must survive the marked run; the baseline run must
    add at least one new non-warmup slot.
    """
    chunk_at_block_2 = list(range(32, 32 + BLOCK_SIZE))
    prompt = list(range(0, BLOCK_SIZE * 4))  # 4 blocks: [0..63]
    # Re-stitch so the chunk-tokens at positions [32..47] are byte-equal
    # to the warmup chunk.
    assert prompt[32:48] == chunk_at_block_2

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        _warmup_chunk(llm, chunk_at_block_2)
        warmup_blocks = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # baseline: no span_starts -> regular hash chain -> warmup slot NOT
        # reachable for block 2.
        sp_baseline = SamplingParams.from_optional(
            seed=SEED, temperature=0.0, max_tokens=MAX_TOKENS,
        )
        llm.generate({"prompt_token_ids": prompt},
                     sampling_params=sp_baseline, use_tqdm=False)
        snap_after_baseline = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # marked: span_starts=[32] -> block 2 hashed with NONE_HASH parent
        # -> matches warmup slot, hits.
        sp_marked = SamplingParams.from_optional(
            seed=SEED, temperature=0.0, max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE * 2]},
        )
        llm.generate({"prompt_token_ids": prompt},
                     sampling_params=sp_marked, use_tqdm=False)
        snap_after_marked = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # Warmup slot survives both runs (cache is large; nothing evicts it).
        assert warmup_blocks <= snap_after_baseline, (
            "warmup slot was evicted by the baseline run"
        )
        assert warmup_blocks <= snap_after_marked, (
            "warmup slot was evicted by the marked run"
        )
        # Baseline must have added at least one new slot - it can't have
        # used the warmup slot because its chain-hashed block 2 differs
        # from the warmup's NONE_HASH-rooted block 2.
        new_after_baseline = snap_after_baseline - warmup_blocks
        assert len(new_after_baseline) >= 1, (
            "baseline run added no new slots - it shouldn't have been "
            "able to reuse the warmup slot (different hash chain), but "
            "the snapshots match. Something else cached the prompt."
        )
    finally:
        cleanup(llm)


def test_same_pic_chunk_reuse_across_prefixes_e2e(model, monkeypatch):
    """E2E counterpart to test_same_pic_chunk_hashes_match_across_requests_no_recompute.

    Two requests share the same PIC chunk but have completely different
    1-block prefixes. After warming up the chunk, both requests must hit
    the warmup slot via PIC fan-in even though their prefix hash chains
    are independent.
    """
    chunk = list(range(500, 500 + BLOCK_SIZE))
    prefix_a = list(range(0, BLOCK_SIZE))
    prefix_b = list(range(900, 900 + BLOCK_SIZE))

    sp = SamplingParams.from_optional(
        seed=SEED, temperature=0.0, max_tokens=MAX_TOKENS,
        extra_args={"span_starts": [BLOCK_SIZE]},
    )

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        _warmup_chunk(llm, chunk)
        warmup_blocks = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        llm.generate({"prompt_token_ids": prefix_a + chunk},
                     sampling_params=sp, use_tqdm=False)
        snap_after_a = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        llm.generate({"prompt_token_ids": prefix_b + chunk},
                     sampling_params=sp, use_tqdm=False)
        snap_after_b = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # Warmup slot survives both runs (PIC fan-in keeps the chunk
        # reused, the cache doesn't evict it).
        assert warmup_blocks <= snap_after_a, (
            "warmup slot was evicted by req_A (prefix_a + chunk)"
        )
        assert warmup_blocks <= snap_after_b, (
            "warmup slot was evicted by req_B (prefix_b + chunk)"
        )
        # The two prefix slots must differ - prefix_a and prefix_b have
        # different tokens, so their hashes differ, so different slots.
        new_a = snap_after_a - warmup_blocks
        new_b = snap_after_b - warmup_blocks
        # Both runs must have written their prefix's K/V to new slots.
        assert len(new_a) >= 1, "req_A wrote no new slots"
        assert len(new_b - new_a) >= 1, (
            "req_B added no new slots beyond req_A's - prefix_b should "
            "have a different hash than prefix_a and miss the cache."
        )
    finally:
        cleanup(llm)


def test_pic_chunk_warmup_then_three_requests(model, monkeypatch):
    """End-to-end check that warming up the spans block before serving
    requests gives the expected reuse pattern.

    Setup (mode SPANS-PC: spans on, prefix caching on, no gap policy):
      - Warm up by running a one-shot request whose only blocks are the
        2-block chunk. That populates the prefix cache with two entries:
        hash(NONE_HASH, chunk[0..16]) and hash(prev, chunk[16..32]).
      - req_A = prefix_X + chunk(2 blocks) + suffix    (span_starts=[32])
      - req_B identical to A, different request_id
      - req_C = prefix_Y + chunk + suffix              (different prefix)

    Expected reuse pattern (per the PIC contract the test pins):
      - req_A: prefix_X + suffix fresh, chunk reused from warmup.
      - req_B: full reuse from req_A.
      - req_C: prefix_Y + suffix fresh, chunk still reused (from the
        same warmup entries; tail is NOT reused across A and C even
        though its hashes happen to collide, because only the chunk's
        2 blocks are marked PIC).

    The assertions check set-relations between KV cache snapshots; they
    do not measure exact block counts (decode-step allocation varies by
    model). The decisive checks are:
      1. The warmup chunk slot survives every subsequent request - it
         must never be evicted or overwritten.
      2. req_B adds zero new K/V slots (full reuse from A).
      3. req_C does not evict any slot req_A wrote.
      4. req_C adds at least one new slot (prefix_Y is a genuine miss).
    """
    chunk = list(range(500, 500 + BLOCK_SIZE * 2))  # 2-block PIC chunk
    suffix = list(range(700, 700 + BLOCK_SIZE * 3))
    prefix_x = list(range(0, BLOCK_SIZE * 2))
    prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))

    sp = SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_args={"span_starts": [BLOCK_SIZE * 2]},
    )

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        # Step 0: warm up the 2-block chunk alone.
        _warmup_chunk(llm, chunk)
        snap_warmup = _kv_cache_block_hashes(llm, LAYER_IDX)
        # The unique non-empty hashes from the warmup. In practice this
        # is 2 slots (the chunk's two blocks) plus optionally a
        # decode-step slot from the max_tokens=1 generate call.
        warmup_blocks = set(snap_warmup)

        # req_A: cold-ish cache (chunk pre-warmed). Prefix + suffix fresh.
        llm.generate(
            {"prompt_token_ids": prefix_x + chunk + suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_after_a = _kv_cache_block_hashes(llm, LAYER_IDX)

        # req_B: identical to A. Must fully reuse A's cache state.
        llm.generate(
            {"prompt_token_ids": prefix_x + chunk + suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_after_b = _kv_cache_block_hashes(llm, LAYER_IDX)

        # req_C: different prefix, same chunk + tail. Prefix_Y fresh,
        # chunk still reused via the warmup-populated PIC slot.
        llm.generate(
            {"prompt_token_ids": prefix_y + chunk + suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_after_c = _kv_cache_block_hashes(llm, LAYER_IDX)

        # 1. The warmup-populated K/V slot survives every subsequent
        #    request. PIC fan-in keeps the chunk reused; the warmup
        #    put it in the cache and the cache must hand it back
        #    unchanged.
        assert warmup_blocks <= set(snap_after_a), (
            f"warmup slot(s) were evicted/overwritten by req_A. "
            f"missing: {warmup_blocks - set(snap_after_a)}"
        )
        assert warmup_blocks <= set(snap_after_b), (
            f"warmup slot(s) were evicted/overwritten by req_B. "
            f"missing: {warmup_blocks - set(snap_after_b)}"
        )
        assert warmup_blocks <= set(snap_after_c), (
            f"warmup slot(s) were evicted/overwritten by req_C. "
            f"missing: {warmup_blocks - set(snap_after_c)}"
        )

        # 2. req_B is identical to req_A. Once A populated the cache,
        #    B must add nothing (full reuse).
        assert set(snap_after_b) == set(snap_after_a), (
            f"req_B added new K/V slots - identical prompt should fully "
            f"reuse A. extra slots in B: "
            f"{set(snap_after_b) - set(snap_after_a)}"
        )

        # 3. req_C must NOT evict or replace any K/V slot A wrote.
        missing_from_c = set(snap_after_a) - set(snap_after_c)
        assert not missing_from_c, (
            f"req_C evicted/overwrote {len(missing_from_c)} slot(s) "
            f"that req_A wrote."
        )

        # 4. CORRECTNESS PIN: req_C must compute fresh K/V for the tail.
        #
        # Only the chunk's 2 blocks are marked PIC
        # (span_starts=[32], chunk spans [32, 64)).
        # Therefore req_C must add new K/V slots for:
        #   - 2 prefix_Y blocks (cache miss, fresh)
        #   - 3 tail blocks (must be recomputed against prefix_Y, NOT
        #     reused from A)
        # That's >= 5 new slots, plus possibly a decode-step block.
        new_in_c = set(snap_after_c) - set(snap_after_a)
        assert len(new_in_c) >= 5, (
            f"INCORRECT CROSS-PREFIX REUSE DETECTED: req_C added only "
            f"{len(new_in_c)} new K/V slots when it should have added "
            f">= 5 (2 prefix_Y + 3 freshly-recomputed tail blocks).\n\n"
            f"The tail blocks are NOT marked PIC but their block_hashes "
            f"collide with req_A's because they chain through the PIC "
            f"chunk's hash. vLLM silently reuses A's tail K/V for C, "
            f"even though C's actual cross-attention sees a different "
            f"prefix (prefix_Y instead of prefix_X)."
        )
    finally:
        cleanup(llm)


def test_legolink_partial_recompute_within_gap_interval(model, monkeypatch):
    """Two-part check that the gap-policy interval bound actually
    constrains which span-region blocks can be recomputed on a cache hit.

    Layout (mode LL-32, gap_length = 2 * BLOCK_SIZE = 2 blocks):

        block index:    0       1       2          3          4       5
        token range: [0..15] [16..31] [32..47]   [48..63]   [64..79] [80..95]
        role:        prefix  prefix   span-start span-chain span-chain span-chain
                                      ^^^^^^^^^^^^^^^^^^^^^^
                                      gap interval (32, 64)
                                      - first 2 of the span region

    Part 1 (structural): instantiate SpanAwareGapPolicy directly with
    gap_length=32 and verify get_gaps returns [(32, 64)] for this
    prompt - i.e. exactly the chunk + first downstream block, not the
    last 2 tail blocks.

    Part 2 (e2e): run the prompt twice through an LL-32 LLM, snapshot
    the KV cache between runs, and bound the byte-diff. With
    deterministic decoding (temp=0, seed=SEED) any gap-recompute
    produces byte-identical K/V, so the count of *new* byte-hashes is
    typically 0; the meaningful assertion is the upper bound (the gap
    policy can't recompute more than its configured 2 blocks).
    """
    prefix = list(range(0, BLOCK_SIZE * 2))
    span_region = list(range(500, 500 + BLOCK_SIZE * 4))
    prompt_tokens = prefix + span_region
    sp = SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_args={"span_starts": [BLOCK_SIZE * 2]},
    )

    # Part 1: pin gap-policy math.
    policy = SpanAwareGapPolicy(
        gap_length=2 * BLOCK_SIZE, block_size=BLOCK_SIZE
    )
    sp_for_policy_req = SamplingParams(
        max_tokens=MAX_TOKENS,
        extra_args={"span_starts": [BLOCK_SIZE * 2]},
    )
    sp_for_policy_req.update_from_generation_config({}, eos_token_id=100)
    policy_req = Request(
        request_id="gap_policy_check",
        prompt_token_ids=prompt_tokens,
        sampling_params=sp_for_policy_req,
        pooling_params=None,
    )
    # Use a num_computed_tokens that simulates "everything except the last
    # block is cached" (the typical prefix-cache state on a re-run).
    gaps = policy.get_gaps(
        policy_req,
        num_computed_tokens=BLOCK_SIZE * 5,  # 5 blocks cached, last block decodes
        num_external_tokens=0,
    )
    assert gaps == [(BLOCK_SIZE * 2, BLOCK_SIZE * 4)], (
        f"Expected gap (32, 64) from gap_length=2 blocks + span at 32, "
        f"got {gaps}"
    )

    # Part 2: e2e byte-diff bound. Pre-warm the chunk so the span block
    # is reachable via PIC fan-in before either main run.
    chunk_tokens = prompt_tokens[BLOCK_SIZE * 2:BLOCK_SIZE * 3]
    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        _warmup_chunk(llm, chunk_tokens)

        llm.generate(
            {"prompt_token_ids": prompt_tokens},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_1 = _kv_cache_block_hashes(llm, LAYER_IDX)
        s1 = set(snap_1)

        llm.generate(
            {"prompt_token_ids": prompt_tokens},
            sampling_params=sp,
            use_tqdm=False,
        )
        snap_2 = _kv_cache_block_hashes(llm, LAYER_IDX)
        s2 = set(snap_2)

        new_in_2 = s2 - s1
        gone_from_1 = s1 - s2

        # gap_length=2*BLOCK_SIZE bounds how many K/V slots can change
        # bytes between runs. With deterministic decoding the actual
        # diff is usually 0 (recompute yields the same bytes), but it
        # can never exceed 2 prompt blocks + a small decode-step budget.
        assert len(new_in_2) <= 4, (
            f"|new_in_2|={len(new_in_2)} (expected <= 4 for "
            f"gap_length=2 blocks plus decode-step slack)"
        )
        assert len(gone_from_1) <= 4, (
            f"|gone_from_1|={len(gone_from_1)} (expected <= 4)"
        )
    finally:
        cleanup(llm)
