# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache and gap-policy assertions for the spans / Legolink machinery.

End-to-end tests that hash the worker's KV cache per block (mirroring
examples/offline_inference/spans/spans_time_and_kv.py) and use set-
relations between snapshots to pin shared-chunk cache hits, span-boundary
hash chain reset, cross-prefix chunk reuse, prefix-cache-survives-PIC
behavior, and the Legolink gap-policy interval bound.
"""

import hashlib

import pytest
import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import BlockHash, get_request_block_hasher
from vllm.v1.request import Request

from .conftest import BLOCK_SIZE, build_llm, cleanup, extract_step0_topk

pytestmark = pytest.mark.spans

SEED = 42
MAX_TOKENS = 16
LOGPROBS_TOPK = 10
LAYER_IDX = 0  # Layer to snapshot. 0 always exists; some example models lack 19.
LAYER_IDX_KV = 1  # Layer for raw K/V comparison; >=1 so K/V is prefix-dependent.


def _greedy_sp(
    extra_args: dict | None = None, logprobs: int | None = None
) -> SamplingParams:
    """Deterministic greedy SamplingParams shared by the e2e runs."""
    return SamplingParams.from_optional(
        seed=SEED,
        temperature=0.0,
        max_tokens=MAX_TOKENS,
        logprobs=logprobs,
        extra_args=extra_args,
    )


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
        sp_marked = _greedy_sp(
            {
                "span_starts": [BLOCK_SIZE * 2],
                "cross_span_starts": [BLOCK_SIZE * 3],
            }
        )
        cached_marked = _generate_num_cached_tokens(llm, prompt, sp_marked)
        assert cached_marked == BLOCK_SIZE * 3
        kv_hashes_after_marked = set(_kv_cache_block_hashes(llm, LAYER_IDX))

        # baseline: no span_starts -> regular hash chain -> warmup slot NOT reachable.
        sp_baseline = _greedy_sp()
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
      * shared - the single block both copies share after the LL-32 gap
        recompute.

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
    sp = _greedy_sp(extra_args)

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
        correct_copy1 = _physical_block_tensor(
            llm, _block_id_for_hash(llm, ref_hashes[1])
        )
        correct_copy2 = _physical_block_tensor(
            llm, _block_id_for_hash(llm, ref_hashes[5])
        )

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
        cached_b = _generate_num_cached_tokens(llm, prompt, sp)
        assert cached_b == BLOCK_SIZE * 7, (
            f"req_B should reuse prefix + both chunks + mid (7 blocks), got {cached_b}"
        )
        shared = _physical_block_tensor(llm, _block_id_for_hash(llm, prompt_hashes[1]))
    finally:
        cleanup(llm)

    # Phase 3: per-occurrence correctness. The single shared block must
    # hold correct K/V for BOTH occurrences - impossible, since RoPE
    # position and prefix-dependent attention make their correct K/V
    # differ. The two gap recomputes race on the same physical slots, so
    # `shared` may match copy 1, copy 2, or neither; "matches both" is
    # the only correct outcome and the only one this accepts.
    match1 = torch.allclose(shared, correct_copy1, atol=2e-2, rtol=2e-2)
    match2 = torch.allclose(shared, correct_copy2, atol=2e-2, rtol=2e-2)
    assert match1 and match2, (
        "repeated PIC span served wrong K/V: the two occurrences need "
        "distinct K/V but share one physical block; the gap recompute "
        f"clobbers it (matches copy1={match1}, copy2={match2})"
    )


def test_pic_tail_not_reused_across_prefixes_e2e(model, monkeypatch):
    """Setup (mode SPANS-PC: spans on, prefix caching on, no gap policy):
      - Warm up by running a one-shot request whose only blocks are the
        2-block chunk. That populates the prefix cache with two entries:
        hash(NONE_HASH, chunk[0..16]) and hash(prev, chunk[16..32]).
      - req_A = prefix_X + chunk(2 blocks) + suffix    (span_starts=[32])
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
      4. req_C adds >= 5 new slots (2 prefix_Y + 3 fresh tail);
         this is the correctness pin and currently FAILS by design.
    """
    _force_in_process_engine(monkeypatch)

    chunk = list(range(500, 500 + BLOCK_SIZE * 2))  # 2-block PIC chunk
    suffix = list(range(700, 700 + BLOCK_SIZE * 3))
    prefix_x = list(range(0, BLOCK_SIZE * 2))
    prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))
    cold_suffix = list(range(2000, 2000 + BLOCK_SIZE))  # bounds cached count
    prompt_a = prefix_x + chunk + suffix + cold_suffix
    prompt_c = prefix_y + chunk + suffix + cold_suffix

    sp = _greedy_sp({"span_starts": [BLOCK_SIZE * 2]})

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
        req_a_hashes = _request_block_hashes(prompt_a, span_starts=[BLOCK_SIZE * 2])
        assert req_a_hashes[2:4] == chunk_hashes, (
            "req_A's PIC chunk blocks must hash to the standalone warmup "
            "chunk's block_hashes; without that equality req_A's chunk "
            "cannot hit the warmup slots"
        )

        # req_A: prefix_X + suffix fresh, chunk hits the warmup slots.
        llm.generate({"prompt_token_ids": prompt_a}, sampling_params=sp, use_tqdm=False)
        kv_hashes_after_a = _kv_cache_block_hashes(llm, LAYER_IDX)

        # 2. req_B: identical to A. The engine charges all of A's prompt
        # (prefix_X + chunk + suffix = 7 blocks) as cached; cold_suffix is
        # the 8th block, bounding cached_b at 7 * BLOCK_SIZE.
        cached_b = _generate_num_cached_tokens(llm, prompt_a, sp)
        assert cached_b == BLOCK_SIZE * 7, (
            f"req_B should reuse all of A's prompt via prefix cache; "
            f"got cached_b={cached_b}, expected {BLOCK_SIZE * 7}"
        )

        # req_C: different prefix, same chunk + tail.
        cached_c = _generate_num_cached_tokens(llm, prompt_c, sp)
        kv_hashes_after_c = _kv_cache_block_hashes(llm, LAYER_IDX)

        # 3. req_C must NOT evict or replace any K/V slot A wrote.
        missing_from_c = set(kv_hashes_after_a) - set(kv_hashes_after_c)
        assert not missing_from_c, (
            f"req_C evicted/overwrote {len(missing_from_c)} slot(s) that req_A wrote."
        )

        # 4. CORRECTNESS PIN: req_C uses prefix_Y (warmed), so its prefix
        # cache hit can only reach the warmed blocks: prefix_Y + chunk = 4
        # blocks. The tail must NOT hit, because its true context is
        # prefix_Y while req_A's cached tail was computed against prefix_X.
        # If vLLM wrongly reuses req_A's tail (tail blocks chain through
        # the NONE-rooted chunk hash, so their hashes collide across A/C),
        # cached_c would be BLOCK_SIZE * 7 instead of BLOCK_SIZE * 4.
        assert cached_c == BLOCK_SIZE * 4, (
            f"INCORRECT CROSS-PREFIX REUSE DETECTED: req_C reported "
            f"cached_c={cached_c}, expected {BLOCK_SIZE * 4} "
            f"(prefix_Y + chunk). {BLOCK_SIZE * 7} would mean vLLM "
            f"silently reused req_A's tail K/V across a different prefix."
        )
    finally:
        cleanup(llm)


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
    sp = _greedy_sp({"span_starts": [BLOCK_SIZE * 2]})

    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        # Warm the span standalone (stale: NONE-rooted at positions 0-63 vs
        # the marked prompt's positions 32-95) and the prefix. Capture the
        # stale span K/V via the standalone span's hashes (which the marked
        # prompt's span blocks 2-5 also hash to).
        _warmup_prompt(llm, span)
        _warmup_prompt(llm, prefix)
        span_hashes = _request_block_hashes(span, span_starts=None)
        stale_span_kv = [
            _physical_block_tensor(llm, _block_id_for_hash(llm, h)) for h in span_hashes
        ]

        # Run the marked prompt - gap policy fires on the span at block 2
        # with gap (32, 64) and recomputes only the first 2 span blocks.
        _generate_num_cached_tokens(llm, prompt_tokens, sp)
        after_span_kv = [
            _physical_block_tensor(llm, _block_id_for_hash(llm, h)) for h in span_hashes
        ]

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

    def _block_kv(block_hash: BlockHash):
        return _physical_block_tensor(llm, _block_id_for_hash(llm, block_hash))

    llm = build_llm(model, "LL-32", monkeypatch)
    try:
        # Phase 1: no markers -> gap policy no-op -> FR-equivalent reference.
        ref_hashes = _request_block_hashes(prompt, span_starts=None)
        ref_out = llm.generate(
            {"prompt_token_ids": prompt},
            sampling_params=_greedy_sp(logprobs=LOGPROBS_TOPK),
            use_tqdm=False,
        )
        ref_top = extract_step0_topk(ref_out[0].outputs[0], LOGPROBS_TOPK)
        ref_span_kv = [_block_kv(h) for h in ref_hashes[2:4]]
        ref_cross_tail_kv = [_block_kv(h) for h in ref_hashes[4:6]]

        # Phase 2: warm the span standalone, then the prefix - so Phase 3's
        # marked request hits prefix+span and the span warmup is consumed.
        # (The span K/V is context-free/stale, cached at the NONE_HASH-rooted
        # hash the PIC-marked span block also hits.)
        standalone_span_hashes = _request_block_hashes(span, span_starts=None)
        _warmup_prompt(llm, span)
        _warmup_prompt(llm, prefix)
        stale_span_kv = [_block_kv(h) for h in standalone_span_hashes]

        # Premise: the span K/V the marked request will hit is genuinely stale -
        # it differs from the context-aware reference, so the gap recompute has
        # real work and the assertions below are not vacuous.
        assert not any(
            torch.allclose(s, r, atol=2e-2, rtol=2e-2)
            for s, r in zip(stale_span_kv, ref_span_kv)
        ), (
            "warmed span K/V already matches the context-aware span; the gap "
            "recompute would be a no-op and this test's premise is moot"
        )

        # Phase 3: the LL marked request. Hits prefix+span; the gap recomputes
        # the span; cross-tail prefills fresh; decode runs.
        marked_hashes = _request_block_hashes(
            prompt, span_starts=span_starts, cross_span_starts=cross_span_starts
        )
        marked_out = llm.generate(
            {"prompt_token_ids": prompt},
            sampling_params=_greedy_sp(
                extra_args={
                    "span_starts": span_starts,
                    "cross_span_starts": cross_span_starts,
                },
                logprobs=LOGPROBS_TOPK,
            ),
            use_tqdm=False,
        )
        cached_marked = marked_out[0].num_cached_tokens
        actual_top = extract_step0_topk(marked_out[0].outputs[0], LOGPROBS_TOPK)
        actual_span_kv = [_block_kv(h) for h in marked_hashes[2:4]]
        actual_cross_tail_kv = [_block_kv(h) for h in marked_hashes[4:6]]

        # The marked request hit prefix + span (4 blocks): confirms there was a
        # cached (stale) span entry for the gap to recompute.
        assert cached_marked == BLOCK_SIZE * 4, (
            f"marked request should hit prefix + span (4 blocks); "
            f"got cached={cached_marked}"
        )

        # The gap recompute produced context-aware span K/V.
        for a, r in zip(actual_span_kv, ref_span_kv):
            assert torch.allclose(a, r, atol=2e-2, rtol=2e-2), (
                "gap recompute did not restore the context-aware span K/V"
            )

        # (a) post-cross prefill happened AFTER the recompute: the cross-tail
        # K/V attends over the span, so it matches the context-aware reference
        # only if the cross-tail prefill observed the recomputed span. If it
        # had run before the recompute it would reflect the stale span.
        for a, r in zip(actual_cross_tail_kv, ref_cross_tail_kv):
            assert torch.allclose(a, r, atol=2e-2, rtol=2e-2), (
                "cross-tail K/V differs from the context-aware reference - the "
                "post-cross prefill read the stale span K/V, i.e. it ran "
                "before the gap recompute"
            )

        # (b) decode happened AFTER the recompute: the first-token top-K
        # matches the reference only if decode observed the recomputed K/V.
        assert actual_top[0][0] == ref_top[0][0], (
            f"decoded top-1 token drift: ref={ref_top[0]}, got={actual_top[0]}"
        )
        assert {t for t, _ in actual_top} == {t for t, _ in ref_top}, (
            "decoded top-K candidate set drifted from the reference - decode "
            "read stale (pre-recompute) K/V"
        )
    finally:
        cleanup(llm)
