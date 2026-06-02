# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.v1.core.kv_cache_utils import BlockHash

from .conftest import (
    BLOCK_SIZE,
    LAYER_IDX,
    _block_kv,
    _capture_request_block_ids,
    _force_in_process_engine,
    _generate_num_cached_tokens,
    _kv_cache_block_hashes,
    _physical_block_tensor,
    _request_block_hashes,
    _warmup_prompt,
    build_llm,
    cleanup,
    generate_single_output,
    greedy_sp,
)

pytestmark = pytest.mark.spans


def test_span_boundary_resets_block_hash_chain_e2e(model, monkeypatch):
    """E2E counterpart to test_span_boundary_resets_block_hash_chain_no_recompute.

    The same 5-block prompt is sent two ways:
      - baseline: no span_starts -> block index 2 hashes through its
                  parent chain
      - marked:   span_starts=[BLOCK_SIZE * 2] -> block index 2 hashes
                  with parent dropped (NONE_HASH)

    Even though the prompt tokens are identical, the hash of the next
    block, index 3, must also differ because its parent hash differs.
    Block index 4 is a cold suffix, so num_cached_tokens can report the
    exact 3-block marked hit.
    """
    span_chunk_at_block_2 = list(range(32, 32 + BLOCK_SIZE))
    prompt = list(range(0, BLOCK_SIZE * 5))  # 5 blocks: [0..79]
    # The chunk-tokens at positions [32..47] are byte-equal to the warmup chunk.
    assert prompt[32:48] == span_chunk_at_block_2

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        baseline_hashes: list[BlockHash] = _request_block_hashes(
            prompt,
            span_starts=None,
        )
        marked_hashes: list[BlockHash] = _request_block_hashes(
            prompt,
            span_starts=[BLOCK_SIZE * 2],
            cross_span_starts=[BLOCK_SIZE * 3],
        )
        warmup_chunk_hash: BlockHash = _request_block_hashes(
            span_chunk_at_block_2,
            span_starts=None,
        )[0]

        assert baseline_hashes[2] != marked_hashes[2], (
            "span_starts should reset the block index 2 parent hash"
        )
        assert marked_hashes[2] == warmup_chunk_hash, (
            "marked block index 2 should match the standalone warmup chunk"
        )
        assert baseline_hashes[3] != marked_hashes[3], (
            "block index 3 should diverge because its parent block hash differs"
        )

        _warmup_prompt(llm, prompt[: BLOCK_SIZE * 2])
        _warmup_prompt(llm, span_chunk_at_block_2)  # [32, 33, 34, ..., 47]
        warmup_kv_blocks = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # marked: span_starts=[32] -> block 2 hashed with NONE_HASH parent
        # -> matches warmup slot, hits.
        sp_marked = greedy_sp(
            {
                "span_starts": [BLOCK_SIZE * 2],
                "cross_span_starts": [BLOCK_SIZE * 3],
            }
        )
        cached_marked = _generate_num_cached_tokens(llm, prompt, sp_marked)
        assert cached_marked == BLOCK_SIZE * 3
        kv_hashes_after_marked = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # baseline: no span_starts -> regular hash chain -> warmup slot NOT reachable.
        sp_baseline = greedy_sp()
        cached_baseline = _generate_num_cached_tokens(llm, prompt, sp_baseline)
        kv_hashes_after_baseline_req_a = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # <= shows every item in warmup_kv_blocks is also present in
        # kv_hashes_after_baseline_req_a.
        assert warmup_kv_blocks <= kv_hashes_after_baseline_req_a, (
            "warmup slot was evicted by the baseline run"
        )
        assert warmup_kv_blocks <= kv_hashes_after_marked, (
            "warmup slot was evicted by the marked run"
        )
        # Baseline hits only the 2 prefix blocks: its chain-hashed block 2
        # differs from the warmup's NONE_HASH-rooted block 2, so the PIC
        # chunk slot is unreachable. marked hit 3 blocks; the missing block
        # is exactly the PIC chunk.
        assert cached_baseline == BLOCK_SIZE * 2, (
            f"baseline should hit only the 2 prefix blocks, not the PIC "
            f"chunk; got {cached_baseline}"
        )
    finally:
        cleanup(llm)


def test_repeated_pic_span_reuse_and_gap_recompute_e2e(model, monkeypatch):
    """A repeated identical chunk in one prompt is served wrong K/V.

    The same 2-block chunk appears twice, at different positions and behind
    different prefixes. With span_starts the two copies share one
    block_hash, hence one physical KV block. LL-32's gap policy recomputes
    both blocks of each copy (gap_length = 2 blocks = the chunk exactly),
    but both virtual gap requests share the parent block table and write
    those two physical blocks in place.

    The test reads the actual physical K/V bytes:
      * correct_copy1 / correct_copy2 - each occurrence's first block under a
        plain prefix-cached run with no span markers, where the two copies
        hash distinctly and are each computed correctly: the per-occurrence
        ground truth.
      * occ1_kv / occ2_kv - the physical block each occurrence actually lands on.
        occ1_first == occ2_first would mean both were wrongly deduped onto one slot.

    Comparisons use torch.allclose, not byte hashes: the reference prefill
    and the gap recompute take different batch shapes and may differ by
    harmless bf16 drift, while the clobber signal is gross (a whole block of
    wrong-prefix K/V).
    """
    _force_in_process_engine(monkeypatch)

    chunk = list(range(500, 500 + BLOCK_SIZE * 2))
    prefix = list(range(0, BLOCK_SIZE))
    mid_text = list(range(900, 900 + BLOCK_SIZE * 2))
    tail = list(range(1200, 1200 + BLOCK_SIZE * 2))
    prompt = prefix + chunk + mid_text + chunk + tail
    span_starts = [BLOCK_SIZE, BLOCK_SIZE * 5]
    cross_span_starts = [BLOCK_SIZE * 3, BLOCK_SIZE * 7]
    extra_args = {
        "span_starts": span_starts,
        "cross_span_starts": cross_span_starts,
    }
    sp = greedy_sp(extra_args)

    # A single LL-32 engine serves both runs: in-process vLLM does not fully
    # release GPU memory between engines, so the test must not build a second
    # one. An LL-32 engine handling a request with no span_starts makes the
    # gap policy a no-op, so the no-span run below is a plain prefix-cached,
    # FR-equivalent reference. All block hashes are computed against this
    # engine, so they share its NONE_HASH (re-randomized per engine).
    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        # Phase 1: per-occurrence ground truth. With no span markers the two
        # chunk copies hash distinctly (copy 1 chains through prefix, copy 2
        # through prefix+chunk+separator), so each first block is computed and
        # cached correctly.
        _warmup_prompt(llm, prompt)
        ref_hashes = _request_block_hashes(prompt, span_starts=None)
        assert ref_hashes[1] != ref_hashes[5], (
            "without span markers the two chunk copies must hash to distinct blocks"
        )
        correct_copy1 = _block_kv(llm, ref_hashes[1])
        correct_copy2 = _block_kv(llm, ref_hashes[5])

        # Phase 2: spans run on the same engine. The two copies share two
        # physical blocks (chunk[0], chunk[1]); LL-32 recomputes both
        # blocks of each copy in place.
        chunk_hashes = _request_block_hashes(chunk, span_starts=None)
        prompt_hashes = _request_block_hashes(
            prompt,
            span_starts=span_starts,
            cross_span_starts=cross_span_starts,
        )
        assert prompt_hashes[1] == prompt_hashes[5] == chunk_hashes[0]
        assert prompt_hashes[2] == prompt_hashes[6] == chunk_hashes[1]
        assert prompt_hashes[3] != prompt_hashes[7], (
            "cross-rooted regular blocks after each PIC span should stay "
            "prefix-dependent"
        )

        # Sequential warmup, each request consumed by the next: chunk,
        # prefix, prefix+chunk+mid. The measured request then reuses
        # blocks 0-6 and prefills the tail itself.
        chunk_ea = {"span_starts": [BLOCK_SIZE], "cross_span_starts": [BLOCK_SIZE * 3]}
        _warmup_prompt(llm, chunk)
        _warmup_prompt(llm, prefix)
        _warmup_prompt(llm, prompt[: BLOCK_SIZE * 5], extra_args=chunk_ea)
        # Read each occurrence's physical block by logical position (block 1 =
        # occurrence 1, block 5 = occurrence 2).
        captured = _capture_request_block_ids(monkeypatch, llm)
        cached_b = _generate_num_cached_tokens(llm, prompt, sp)
        assert cached_b == BLOCK_SIZE * 7, (
            f"req_B should reuse prefix + both chunks + mid (7 blocks), got {cached_b}"
        )
        block_ids = max(captured.values(), key=len)  # request with longest block table
        occ1_first, occ2_first = block_ids[1], block_ids[5]
        occ1_kv = _physical_block_tensor(llm, occ1_first)
        occ2_kv = _physical_block_tensor(llm, occ2_first)
    finally:
        cleanup(llm)

    # Phase 3: per-occurrence correctness. Each occurrence's gap recompute
    # must produce its own prefix-aware K/V, and that needs its own physical block.
    assert occ1_first != occ2_first, (
        "the two PIC occurrences were deduped onto one physical block; "
        "distinct per-occurrence K/V is impossible while they share a slot"
    )
    match1 = torch.allclose(occ1_kv, correct_copy1, atol=2e-2, rtol=2e-2)
    match2 = torch.allclose(occ2_kv, correct_copy2, atol=2e-2, rtol=2e-2)
    assert match1 and match2, (
        "repeated PIC span served wrong K/V: each occurrence's recomputed "
        "K/V must match its own prefix-aware ground truth "
        f"(copy1={match1}, copy2={match2})"
    )


@pytest.mark.parametrize(
    "prefix_blocks,block_size,force_tile_size,alignment_label",
    [
        # span_start = BLOCK_SIZE * 2 = 32 → multiple of TILE_SIZE_PREFILL (32).
        # The KV boundary lands ON a tile boundary; a coarse tile_start skip is
        # sufficient.
        (2, 16, None, "tile-aligned"),
        # span_start = BLOCK_SIZE = 16 → block-aligned but NOT a multiple of
        # TILE_SIZE_PREFILL (32). A KV tile straddles the span boundary; even
        # with the per-key K-RoPE shift, online-softmax tile-merge ordering
        # differs from the standalone reference, so bit-exact fails by ULP.
        (1, 16, None, "tile-unaligned"),
        # block_size=32, TILE_SIZE_PREFILL forced to 16. Now blocks are *bigger*
        # than tiles, so every block-aligned span_start is automatically
        # tile-aligned (32 is a multiple of 16). Standalone and in-prompt should
        # iterate the same tile count, and bit-exact should hold.
        (1, 32, 16, "block-bigger-than-tile"),
    ],
    ids=["aligned", "unaligned", "block-bigger-than-tile"],
)
def test_unwarmed_pic_chunk_halts_prefix_cache_reuse_e2e(
    model, monkeypatch, prefix_blocks, block_size, force_tile_size, alignment_label
):
    """Unwarmed PIC chunk, span in the middle (SPANS-PC, 3 requests A/B/C).
    A: cold prompt prefills fully and stores the chunk NONE_HASH-rooted; the
       span's K/V must be bit-identical to the chunk computed standalone.
    B: a new-prefix request run as-is reuses nothing (block 0 misses, run halts).
    C: a third-prefix request with its prefix warmed reuses both prefix+chunk.

    Three alignment regimes (see parametrize comments above): tile-aligned,
    tile-unaligned (within a single block), and block-bigger-than-tile. The
    bit-exact span K/V check hinges on standalone and in-prompt iterating the
    same number of KV tiles in the kernel."""
    _force_in_process_engine(monkeypatch)
    if force_tile_size is not None:
        # Pin TILE_SIZE_PREFILL for this run (e.g. force a smaller tile than
        # the kernel would default to, so block boundaries land on tile
        # boundaries even when BLOCK_SIZE > default TILE_SIZE).
        import vllm.v1.attention.ops.triton_unified_attention as _kernel_mod
        monkeypatch.setattr(
            _kernel_mod, "_get_tile_size",
            lambda head_size, sliding_window, element_size, is_prefill: (
                force_tile_size if is_prefill else
                _kernel_mod._is_gemma3_attention(head_size, sliding_window)
                and 32
                or (16 if element_size >= 2 else 32)
            ),
        )
    prefix_len = block_size * prefix_blocks
    chunk = list(range(500, 500 + block_size * 2))
    tail = list(range(1200, 1200 + block_size))
    prefix_a = list(range(0, prefix_len))
    prefix_b = list(range(3000, 3000 + prefix_len))
    prefix_c = list(range(6000, 6000 + prefix_len))
    span_start = prefix_len
    cross_start = prefix_len + block_size * 2
    sp = greedy_sp(
        {"span_starts": [span_start], "cross_span_starts": [cross_start]}
    )
    # Standalone reference (separate engine): the in-prompt span and a standalone chunk
    # collide on one NONE-rooted slot, so a second engine is the only way to compare.
    llm = build_llm(model, "SPANS-PC", monkeypatch, block_size=block_size)
    try:
        _warmup_prompt(llm, chunk)
        standalone_hashes = _request_block_hashes(
            chunk, span_starts=None, block_size=block_size
        )
        standalone_chunk_kv = [_block_kv(llm, h) for h in standalone_hashes]
    finally:
        cleanup(llm)
    llm = build_llm(model, "SPANS-PC", monkeypatch, block_size=block_size)
    try:
        # A: cold run - the whole prompt prefills, chunk stored NONE_HASH-rooted.
        prompt_a = prefix_a + chunk + tail
        cached_a = _generate_num_cached_tokens(llm, prompt_a, sp)
        scheduler = llm.llm_engine.engine_core.engine_core.scheduler
        pool = scheduler.kv_cache_manager.block_pool

        def cached(h):
            return pool.get_cached_block(h, [0]) is not None

        stored = _request_block_hashes(
            prompt_a, [span_start], [cross_start], block_size=block_size
        )
        chained_chunk = _request_block_hashes(
            prompt_a, span_starts=None, block_size=block_size
        )[prefix_blocks]
        full_prefill = all(cached(h) for h in stored)
        # Span occupies blocks [prefix_blocks, prefix_blocks + 2).
        span_block_slice = slice(prefix_blocks, prefix_blocks + 2)
        chunk_none_rooted = all(
            cached(h) for h in stored[span_block_slice]
        ) and not cached(chained_chunk)
        inprompt_chunk_kv = [_block_kv(llm, h) for h in stored[span_block_slice]]

        # B: different prefix, run as-is (unwarmed) - block 0 misses, no reuse.
        cached_b = _generate_num_cached_tokens(llm, prefix_b + chunk + tail, sp)

        # C: third prefix, warmed first -> prefix + chunk both hit.
        _warmup_prompt(llm, prefix_c)
        cached_c = _generate_num_cached_tokens(llm, prefix_c + chunk + tail, sp)
    finally:
        cleanup(llm)
    assert cached_a == 0, f"cold run should reuse nothing, got {cached_a}"
    assert full_prefill, "cold run did not fully prefill the prompt"
    assert chunk_none_rooted, "chunk not stored under its NONE_HASH-rooted hash"
    assert cached_b == 0, f"unwarmed new prefix must halt at block 0, got {cached_b}"
    assert cached_c == prefix_len + block_size * 2, (
        f"warmed prefix should reuse prefix + chunk, got {cached_c}"
    )
    # 4th check: A's in-prompt span K/V must be bit-identical to standalone.
    for i, (ip, st) in enumerate(zip(inprompt_chunk_kv, standalone_chunk_kv)):
        assert torch.equal(ip, st), (
            f"[{alignment_label}] span block {i} K/V from A's in-prompt prefill "
            f"is not bit-identical to the standalone chunk - the PIC span is "
            f"not position-invariant"
        )


def test_pic_tail_not_reused_across_prefixes_e2e(model, monkeypatch):
    """Setup (mode SPANS-PC: spans on, prefix caching on, no gap policy):
      - Warm up by running a one-shot request whose only blocks are the
        2-block chunk. That populates the prefix cache with two entries:
        hash(NONE_HASH, chunk[0..16]) and hash(prev, chunk[16..32]).
      - req_A = prefix_X + chunk(2 blocks) + suffix
        (span_starts=[32], cross_span_starts=[64])
      - req_B identical to A, different request_id
      - req_C = prefix_Y + chunk + suffix              (different prefix)

    The decisive checks are:
      1. Chunk re-use: req_A's span-marked chunk blocks hash to the same
         block_hashes as the standalone warmup chunk (so the lookup hits
         the warmup slots), and those slots still hold byte-identical K/V
         after all three requests (the cache hit is read-only).
      2. req_B reports BLOCK_SIZE * 7 cached tokens - ordinary prefix-cache
         reuse of A's prompt on a byte-identical re-run.
      3. req_C does not evict any slot req_A wrote.
      4. req_C reuses only prefix_Y + chunk (4 blocks): the cross-anchored
         suffix hashes against prefix_Y, misses req_A's prefix_X-rooted
         tail, and is recomputed - so no cross-prefix tail reuse.
    """
    _force_in_process_engine(monkeypatch)

    chunk = list(range(500, 500 + BLOCK_SIZE * 2))  # 2-block PIC chunk
    suffix = list(range(700, 700 + BLOCK_SIZE * 3))
    prefix_x = list(range(0, BLOCK_SIZE * 2))
    prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))
    cold_suffix = list(range(2000, 2000 + BLOCK_SIZE))  # 8th block, the last
    prompt_a = prefix_x + chunk + suffix + cold_suffix
    prompt_c = prefix_y + chunk + suffix + cold_suffix
    sp = greedy_sp(
        {"span_starts": [BLOCK_SIZE * 2], "cross_span_starts": [BLOCK_SIZE * 4]}
    )

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        # Step 0: sequential warmup - chunk, prefix_X, prefix_Y - so req_A
        # reuses chunk + prefix_X and req_C reuses chunk + prefix_Y.
        _warmup_prompt(llm, chunk)
        _warmup_prompt(llm, prefix_x)
        _warmup_prompt(llm, prefix_y)
        chunk_hashes = _request_block_hashes(chunk, span_starts=None)

        # 1a. Chunk re-use (structural): req_A's span-marked chunk blocks
        #     (index 2, 3) must carry the same block_hashes as the
        #     standalone chunk. The prefix cache is keyed by block_hash,
        #     so this equality is exactly what makes req_A's chunk lookup
        #     hit the warmup slots.
        req_a_hashes = _request_block_hashes(
            prompt_a,
            span_starts=[BLOCK_SIZE * 2],
            cross_span_starts=[BLOCK_SIZE * 4],
        )
        assert req_a_hashes[2:4] == chunk_hashes, (
            "req_A's PIC chunk blocks must hash to the standalone warmup "
            "chunk's block_hashes; without that equality req_A's chunk "
            "cannot hit the warmup slots"
        )

        # req_A: prefix_X + suffix fresh, chunk hits the warmup slots.
        generate_single_output(llm, prompt_a, sp)
        kv_hashes_after_a = _kv_cache_block_hashes(llm, LAYER_IDX)

        # 2. req_B: identical to A; all 8 blocks are cached, but the lookup caps
        # the hit at num_tokens - 1 (block-aligned), dropping the last block -> 7.
        cached_b = _generate_num_cached_tokens(llm, prompt_a, sp)
        assert cached_b == BLOCK_SIZE * 7, (
            f"req_B should reuse all of A's prompt via prefix cache; "
            f"got cached_b={cached_b}, expected {BLOCK_SIZE * 7}"
        )

        # req_C: different prefix, same chunk + tail.
        cached_c = _generate_num_cached_tokens(llm, prompt_c, sp)
        kv_hashes_after_c = _kv_cache_block_hashes(llm, LAYER_IDX)

        # 3. req_C must NOT replace any K/V slot A wrote.
        missing_from_c = set(kv_hashes_after_a) - set(kv_hashes_after_c)
        assert not missing_from_c, (
            f"req_C overwrote {len(missing_from_c)} slot(s) that req_A wrote."
        )

        # 4. CORRECTNESS PIN: req_C uses prefix_Y (warmed), so its prefix
        # cache hit reaches prefix_Y + chunk = 4 blocks. The tail must NOT
        # hit: cross_span_starts re-anchors the first suffix block to the
        # full prefix, so req_C's suffix hashes against prefix_Y and misses
        # req_A's prefix_X-rooted tail. Without that cross boundary the tail
        # would chain through the NONE-rooted chunk hash, collide across
        # A/C, and cached_c would be BLOCK_SIZE * 7 instead of BLOCK_SIZE * 4.
        assert cached_c == BLOCK_SIZE * 4, (
            f"INCORRECT CROSS-PREFIX REUSE DETECTED: req_C reported "
            f"cached_c={cached_c}, expected {BLOCK_SIZE * 4} "
            f"(prefix_Y + chunk). {BLOCK_SIZE * 7} would mean vLLM "
            f"silently reused req_A's tail K/V across a different prefix."
        )
    finally:
        cleanup(llm)
