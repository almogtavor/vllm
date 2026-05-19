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
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import BlockHash, get_request_block_hasher
from vllm.v1.core.sched.gap_policy import SpanAwareGapPolicy
from vllm.v1.request import Request

from .conftest import BLOCK_SIZE, build_llm, cleanup

pytestmark = pytest.mark.spans

SEED = 42
MAX_TOKENS = 16
LAYER_IDX = 0  # Layer to snapshot. 0 always exists; some example models lack 19.
LAYER_IDX_KV = 1  # Layer for raw K/V comparison; >=1 so K/V is prefix-dependent.


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


def _physical_block_tensor(llm, block_id: int, layer_idx: int):
    """Raw K/V tensor of one physical KV-cache block, on CPU as float32."""

    def _grab(worker_self):
        import torch

        kv = worker_self.model_runner.kv_caches[layer_idx][block_id]
        cpu = kv.detach().cpu()
        if cpu.dtype == torch.bfloat16:
            cpu = cpu.to(torch.float32)
        return cpu

    results = llm.llm_engine.engine_core.collective_rpc(_grab)
    assert results, "collective_rpc returned no worker results"
    return results[0]


def _block_id_for_hash(llm, block_hash: BlockHash) -> int:
    """Physical block id currently backing `block_hash` in the prefix cache."""
    block_pool = (
        llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.block_pool
    )
    cached = block_pool.get_cached_block(block_hash, [0])
    assert cached is not None, f"block hash {block_hash!r} not in prefix cache"
    return cached[0].block_id


def _warmup_prompt(
    llm,
    prompt_token_ids: list[int],
    extra_args: dict | None = None,
) -> None:
    """Populate prefix cache entries for the given full-block prompt.

    When the prompt is exactly the PIC chunk, this creates the same
    NONE_HASH-rooted block entry that a later span-start block should hit.
    """
    llm.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params=SamplingParams(
            max_tokens=1, temperature=0.0, extra_args=extra_args
        ),
        use_tqdm=False,
    )


def _generate_num_cached_tokens(
    llm,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> int:
    outputs = llm.generate(
        {"prompt_token_ids": prompt_token_ids},
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    assert len(outputs) == 1
    num_cached_tokens = outputs[0].num_cached_tokens
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
    sp = SamplingParams(max_tokens=MAX_TOKENS, extra_args=extra_args)
    sp.update_from_generation_config({}, eos_token_id=100)
    req = Request(
        request_id="hash_probe",
        prompt_token_ids=prompt_token_ids,
        sampling_params=sp,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, hash_fn),
    )
    return req.block_hashes


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
        sp_marked = SamplingParams.from_optional(
            seed=SEED,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            extra_args={
                "span_starts": [BLOCK_SIZE * 2],
                "cross_span_starts": [BLOCK_SIZE * 3],
            },
        )
        cached_marked = _generate_num_cached_tokens(llm, prompt, sp_marked)
        assert cached_marked == BLOCK_SIZE * 3
        kv_hashes_after_marked = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # baseline: no span_starts -> regular hash chain -> warmup slot NOT reachable.
        sp_baseline = SamplingParams.from_optional(
            seed=SEED,
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
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
    block_hash, hence one physical KV block. LL-16's gap policy recomputes
    the first block of each copy, but both virtual gap requests share the
    parent block table and write that single physical block in place.

    The test reads the actual physical K/V bytes:
      * correct_copy1 / correct_copy2 - each occurrence's first block under a
        plain prefix-cached run with no span markers, where the two copies
        hash distinctly and are each computed correctly: the per-occurrence
        ground truth.
      * shared - the single block both copies share after the LL-16 gap
        recompute.

    Comparisons use torch.allclose, not byte hashes: the reference prefill
    and the gap recompute take different batch shapes and may differ by
    harmless bf16 drift, while the clobber signal is gross (a whole block of
    wrong-prefix K/V).
    """
    import torch

    # This test reads scheduler-side state (the KV-cache block pool), which
    # lives in the EngineCore process. Force the engine core in-process so
    # llm.llm_engine.engine_core.engine_core.scheduler is reachable.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    chunk = list(range(500, 500 + BLOCK_SIZE * 2))
    prefix = list(range(0, BLOCK_SIZE))
    bridge = list(range(900, 900 + BLOCK_SIZE * 2))
    tail = list(range(1200, 1200 + BLOCK_SIZE * 2))
    cold_suffix = list(range(1500, 1500 + BLOCK_SIZE))
    prompt = prefix + chunk + bridge + chunk + tail + cold_suffix
    span_starts = [BLOCK_SIZE, BLOCK_SIZE * 5]
    cross_span_starts = [BLOCK_SIZE * 3, BLOCK_SIZE * 7]
    extra_args = {
        "span_starts": span_starts,
        "cross_span_starts": cross_span_starts,
    }
    sp = SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_args=extra_args,
    )

    # A single LL-16 engine serves both runs: in-process vLLM does not fully
    # release GPU memory between engines, so the test must not build a second
    # one. An LL-16 engine handling a request with no span_starts makes the
    # gap policy a no-op, so the no-span run below is a plain prefix-cached,
    # FR-equivalent reference. All block hashes are computed against this
    # engine, so they share its NONE_HASH (re-randomized per engine).
    llm = build_llm(model, "LL-16", monkeypatch)
    try:
        # Phase 1: per-occurrence ground truth. With no span markers the two
        # chunk copies hash distinctly (copy 1 chains through prefix, copy 2
        # through prefix+chunk+bridge), so each first block is computed and
        # cached correctly.
        _warmup_prompt(llm, prompt)
        ref_hashes = _request_block_hashes(prompt, span_starts=None)
        assert ref_hashes[1] != ref_hashes[5], (
            "without span markers the two chunk copies must hash to distinct blocks"
        )
        correct_copy1 = _physical_block_tensor(
            llm, _block_id_for_hash(llm, ref_hashes[1]), LAYER_IDX_KV
        )
        correct_copy2 = _physical_block_tensor(
            llm, _block_id_for_hash(llm, ref_hashes[5]), LAYER_IDX_KV
        )

        # Phase 2: sharing verdict. Different prefixes -> different attention
        # context -> the two occurrences genuinely need different K/V, so one
        # shared physical block cannot serve both.
        assert not torch.allclose(correct_copy1, correct_copy2, atol=2e-2, rtol=2e-2), (
            "the two chunk occurrences produced byte-equal K/V; sharing one "
            "physical block would then be safe and this test's premise is moot"
        )

        # Phase 3: spans run on the same engine. The two copies share one
        # physical block; LL-16 recomputes the first block of each, both
        # writing that one block in place.
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

        _warmup_prompt(llm, chunk)
        _warmup_prompt(llm, prompt, extra_args=extra_args)
        cached_b = _generate_num_cached_tokens(llm, prompt, sp)
        assert cached_b == BLOCK_SIZE * 9, (
            f"req_B should report all non-suffix prompt blocks cached "
            f"({BLOCK_SIZE * 9} tokens), got {cached_b}"
        )
        shared = _physical_block_tensor(
            llm, _block_id_for_hash(llm, prompt_hashes[1]), LAYER_IDX_KV
        )
    finally:
        cleanup(llm)

    # Phase 4: per-occurrence correctness. The single fanned-in block must
    # hold correct K/V for BOTH occurrences - impossible, since phase 2
    # proved they differ. The two gap recomputes race on the same physical
    # slots, so `shared` may match copy 1, copy 2, or neither; "matches
    # both" is the only correct outcome and the only one this accepts.
    match1 = torch.allclose(shared, correct_copy1, atol=2e-2, rtol=2e-2)
    match2 = torch.allclose(shared, correct_copy2, atol=2e-2, rtol=2e-2)
    assert match1 and match2, (
        "repeated PIC span served wrong K/V: the two occurrences need "
        "distinct K/V but share one physical block; the gap recompute "
        f"clobbers it (matches copy1={match1}, copy2={match2})"
    )


def test_pic_chunk_warmup_then_three_requests(model, monkeypatch):
    """Setup (mode SPANS-PC: spans on, prefix caching on, no gap policy):
      - Warm up by running a one-shot request whose only blocks are the
        2-block chunk. That populates the prefix cache with two entries:
        hash(NONE_HASH, chunk[0..16]) and hash(prev, chunk[16..32]).
      - req_A = prefix_X + chunk(2 blocks) + suffix    (span_starts=[32])
      - req_B identical to A, different request_id
      - req_C = prefix_Y + chunk + suffix              (different prefix)

    The decisive checks are:
      1. The warmup chunk slot survives every subsequent request - it
         must never be evicted or overwritten.
      2. req_B reports BLOCK_SIZE * 7 cached tokens (engine itself
         confirms it reused all of A's prompt via PIC fan-in).
      3. req_C does not evict any slot req_A wrote.
      4. req_C adds >= 5 new slots (2 prefix_Y + 3 fresh tail);
         this is the correctness pin and currently FAILS by design.
    """
    chunk = list(range(500, 500 + BLOCK_SIZE * 2))  # 2-block PIC chunk
    suffix = list(range(700, 700 + BLOCK_SIZE * 3))
    prefix_x = list(range(0, BLOCK_SIZE * 2))
    prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))
    cold_suffix = list(range(2000, 2000 + BLOCK_SIZE))  # bounds cached count

    sp = SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        extra_args={"span_starts": [BLOCK_SIZE * 2]},
    )

    llm = build_llm(model, "SPANS-PC", monkeypatch)
    try:
        # Step 0: warm up the 2-block chunk alone.
        _warmup_prompt(llm, chunk)
        kv_hashes_warmup = _kv_cache_block_hashes(llm, LAYER_IDX)
        # The unique non-empty hashes from the warmup. In practice this
        # is 2 slots (the chunk's two blocks) plus optionally a
        # decode-step slot from the max_tokens=1 generate call.
        warmup_blocks = set(kv_hashes_warmup)

        # req_A: cold-ish cache (chunk pre-warmed). Prefix + suffix fresh.
        llm.generate(
            {"prompt_token_ids": prefix_x + chunk + suffix + cold_suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        kv_hashes_after_a = _kv_cache_block_hashes(llm, LAYER_IDX)

        # req_B: identical to A. Engine should charge all of A's prompt
        # (prefix_X + chunk + suffix = 7 blocks) as cached; cold_suffix is
        # the 8th block, bounding cached_b at 7 * BLOCK_SIZE.
        cached_b = _generate_num_cached_tokens(
            llm, prefix_x + chunk + suffix + cold_suffix, sp
        )
        kv_hashes_after_b = _kv_cache_block_hashes(llm, LAYER_IDX)
        assert cached_b == BLOCK_SIZE * 7, (
            f"req_B should reuse all of A's prompt via prefix cache "
            f"(chunk included); got cached_b={cached_b}, "
            f"expected {BLOCK_SIZE * 7}"
        )

        # req_C: different prefix, same chunk + tail. Prefix_Y fresh,
        # chunk still reused via the warmup-populated PIC slot.
        llm.generate(
            {"prompt_token_ids": prefix_y + chunk + suffix + cold_suffix},
            sampling_params=sp,
            use_tqdm=False,
        )
        kv_hashes_after_c = _kv_cache_block_hashes(llm, LAYER_IDX)

        # 1. The warmup-populated K/V slot survives every subsequent
        #    request. PIC fan-in keeps the chunk reused; the warmup
        #    put it in the cache and the cache must hand it back
        #    unchanged.
        assert warmup_blocks <= set(kv_hashes_after_a), (
            f"warmup slot(s) were evicted/overwritten by req_A. "
            f"missing: {warmup_blocks - set(kv_hashes_after_a)}"
        )
        assert warmup_blocks <= set(kv_hashes_after_b), (
            f"warmup slot(s) were evicted/overwritten by req_B. "
            f"missing: {warmup_blocks - set(kv_hashes_after_b)}"
        )
        assert warmup_blocks <= set(kv_hashes_after_c), (
            f"warmup slot(s) were evicted/overwritten by req_C. "
            f"missing: {warmup_blocks - set(kv_hashes_after_c)}"
        )

        # 2. req_C must NOT evict or replace any K/V slot A wrote.
        missing_from_c = set(kv_hashes_after_a) - set(kv_hashes_after_c)
        assert not missing_from_c, (
            f"req_C evicted/overwrote {len(missing_from_c)} slot(s) that req_A wrote."
        )

        # 3. CORRECTNESS PIN: req_C must compute fresh K/V for the tail.
        #
        # Only the chunk's 2 blocks are marked PIC
        # (span_starts=[32], chunk spans [32, 64)).
        # Therefore req_C must add new K/V slots for:
        #   - 2 prefix_Y blocks (cache miss, fresh)
        #   - 3 tail blocks (must be recomputed against prefix_Y, NOT
        #     reused from A)
        # That's >= 5 new slots, plus possibly a decode-step block.
        new_in_c = set(kv_hashes_after_c) - set(kv_hashes_after_a)
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

    Part 2 (e2e): run the prompt twice through an LL-32 LLM, assert the
    replay sees the expected prefix-cache hit, snapshot the KV cache
    between runs, and bound the byte-diff. With deterministic decoding
    (temp=0, seed=SEED) any gap-recompute produces byte-identical K/V,
    so the count of *new* byte-hashes is typically 0; the meaningful
    assertion is the upper bound (the gap policy can't recompute more
    than its configured 2 blocks).
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
    policy = SpanAwareGapPolicy(gap_length=2 * BLOCK_SIZE, block_size=BLOCK_SIZE)
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
    policy_req.span_starts = [BLOCK_SIZE * 2]
    # Use a num_computed_tokens that simulates "everything except the last
    # block is cached" (the typical prefix-cache state on a re-run).
    gaps = policy.get_gaps(
        policy_req,
        num_computed_tokens=BLOCK_SIZE * 5,  # 5 blocks cached, last block decodes
        num_external_tokens=0,
    )
    assert gaps == [(BLOCK_SIZE * 2, BLOCK_SIZE * 4)], (
        f"Expected gap (32, 64) from gap_length=2 blocks + span at 32, got {gaps}"
    )

    # Part 2: e2e byte-diff bound. Pre-warm the chunk so the span block
    # is reachable via PIC fan-in before either main run.
    chunk_tokens = prompt_tokens[BLOCK_SIZE * 2 : BLOCK_SIZE * 3]
    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        _warmup_prompt(llm, chunk_tokens)

        cached_1 = _generate_num_cached_tokens(llm, prompt_tokens, sp)
        snap_1 = _kv_cache_block_hashes(llm, LAYER_IDX)
        s1 = set(snap_1)

        cached_2 = _generate_num_cached_tokens(llm, prompt_tokens, sp)
        assert cached_2 == BLOCK_SIZE * 5, (
            f"LL-32 replay should prefix-cache all non-final prompt blocks "
            f"({BLOCK_SIZE * 5} tokens), got {cached_2}; cold run got {cached_1}"
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
