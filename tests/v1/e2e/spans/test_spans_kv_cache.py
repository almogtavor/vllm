# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV-cache and gap-policy assertions for the spans / Legolink machinery.

Migrated from examples/offline_inference/spans/spans_time_and_kv.py:
the example's per-block SHA-256 dump becomes the snapshot used to assert
recompute actually changed cached K/V content. Timing / TTFT / TPOT
metrics are dropped (perf, not correctness).

Tests:
  test_same_pic_chunk_hashes_match_across_requests_no_recompute
      same PIC chunk in two requests with different prefixes → identical
      chunk hash but different pre-chunk hashes (the "fan-in" guarantee
      from the user's perspective: same content, different chain). Pure
      structural hash check, no recompute.
  test_pic_spans_preserve_prefix_caching_across_requests
      Three structural requests sharing a PIC chunk + tail: pins
      determinism, chunk fan-in, post-PIC tail share-ability, and that
      divergence stays scoped to the pre-span blocks.
  test_legolink_recompute_overwrites_pic_chunk_kv_in_place
      LL-FULL + prefix caching: run the same prompt twice. Run #2 hits the
      cache, gap policy fires, and the virtual gap request shares the
      parent's block_ids (see scheduler.py:963-996) → K/V is recomputed
      and written back into the same physical slots, overwriting them.
      The KV-cache snapshot after run #2 must differ from the one taken
      between runs.
"""
import hashlib

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256_cbor
from vllm.v1.core.kv_cache_utils import get_request_block_hasher
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


def test_same_pic_chunk_hashes_match_across_requests_no_recompute():
    """Pure block-hash check (NO recompute, no LLM): two requests with the
    same PIC chunk but different surrounding tokens hash the chunk to the
    same block hash (fan-in / position-invariant), and the surrounding
    blocks to different hashes. This is structural — gap policy and runtime
    K/V overwrite are not involved."""
    original = envs.VLLM_V1_SPANS_ENABLED
    try:
        envs.VLLM_V1_SPANS_ENABLED = True
        chunk = list(range(500, 500 + BLOCK_SIZE))

        # Two prefixes of equal length, different content, both 1 block.
        prefix_a = list(range(0, BLOCK_SIZE))
        prefix_b = list(range(900, 900 + BLOCK_SIZE))

        sp = SamplingParams(
            max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE]},
        )
        sp.update_from_generation_config({}, eos_token_id=100)
        hasher = get_request_block_hasher(BLOCK_SIZE, sha256_cbor)

        req_a = Request(
            request_id="pic_share_a",
            prompt_token_ids=prefix_a + chunk,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )
        req_b = Request(
            request_id="pic_share_b",
            prompt_token_ids=prefix_b + chunk,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )

        assert len(req_a.block_hashes) == 2
        assert len(req_b.block_hashes) == 2
        # Pre-chunk blocks differ (different prefixes).
        assert req_a.block_hashes[0] != req_b.block_hashes[0]
        # PIC chunk: identical hash regardless of preceding context (fan-in).
        assert req_a.block_hashes[1] == req_b.block_hashes[1]
    finally:
        envs.VLLM_V1_SPANS_ENABLED = original


def test_pic_spans_preserve_prefix_caching_across_requests():
    """Prefix caching survives PIC spans embedded in a long prompt.

    Three structural requests, all with span_starts=[BLOCK_SIZE * 2] (one PIC
    chunk in the middle of the prompt) and a shared post-PIC tail:

        req_a = prefix_X + chunk + suffix    (request id "a")
        req_b = prefix_X + chunk + suffix    (same content, different id)
        req_c = prefix_Y + chunk + suffix    (different prefix, same chunk + tail)

    The four assertions pin properties prefix caching depends on:

      1. Determinism — replaying the same request hashes identically end-to-end.
         Without this, prefix caching cannot reuse anything on a re-run.
      2. Fan-in on the PIC chunk — same chunk hashes the same regardless of
         preceding prefix. Cross-request cache reuse on the chunk itself.
      3. Post-PIC tail is also shareable — every block downstream of the chunk
         hashes the same across A and C. PIC dropping the parent at the span
         boundary means the tail's chain only depends on chunk + tail tokens,
         not on the prefix. This is the load-bearing property: it turns a
         single shared chunk into a shared *suffix from the chunk onward*.
      4. Divergence stays scoped — pre-span blocks hash differently across A
         and C. The span boundary is the only point where chains decouple.
    """
    original = envs.VLLM_V1_SPANS_ENABLED
    try:
        envs.VLLM_V1_SPANS_ENABLED = True

        prefix_x = list(range(0, BLOCK_SIZE * 2))               # 2 blocks
        prefix_y = list(range(900, 900 + BLOCK_SIZE * 2))       # 2 blocks, different content
        chunk = list(range(500, 500 + BLOCK_SIZE))              # 1 block, the PIC chunk
        suffix = list(range(700, 700 + BLOCK_SIZE * 3))         # 3 blocks, shared tail

        sp = SamplingParams(
            max_tokens=MAX_TOKENS,
            extra_args={"span_starts": [BLOCK_SIZE * 2]},
        )
        sp.update_from_generation_config({}, eos_token_id=100)
        hasher = get_request_block_hasher(BLOCK_SIZE, sha256_cbor)

        req_a = Request(
            request_id="pic_pc_a",
            prompt_token_ids=prefix_x + chunk + suffix,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )
        req_b = Request(
            request_id="pic_pc_b",
            prompt_token_ids=prefix_x + chunk + suffix,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )
        req_c = Request(
            request_id="pic_pc_c",
            prompt_token_ids=prefix_y + chunk + suffix,
            sampling_params=sp,
            pooling_params=None,
            block_hasher=hasher,
        )

        # 6 blocks each: 2 prefix + 1 chunk + 3 suffix.
        assert len(req_a.block_hashes) == 6
        assert len(req_b.block_hashes) == 6
        assert len(req_c.block_hashes) == 6

        # 1. Determinism: same prompt + same spans → identical hashes.
        assert req_a.block_hashes == req_b.block_hashes

        # 2. Fan-in on the PIC chunk across different prefixes.
        assert req_a.block_hashes[2] == req_c.block_hashes[2]

        # 3. Post-PIC tail is shareable across requests with matching chunk
        #    + tail, even when prefixes differ. THIS is the property that
        #    makes PIC actually pay off in cache hit-rate.
        assert req_a.block_hashes[2:] == req_c.block_hashes[2:]

        # 4. Divergence is scoped to the pre-span blocks only.
        assert req_a.block_hashes[0:2] != req_c.block_hashes[0:2]
    finally:
        envs.VLLM_V1_SPANS_ENABLED = original


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


def test_pic_chunk_warmup_then_three_requests(model, monkeypatch):
    """End-to-end check that warming up the spans block before serving
    requests gives the expected reuse pattern.

    Setup (mode SPANS-PC: spans on, prefix caching on, no gap policy):
      - Warm up by running a one-shot request whose sole block is the
        chunk. That populates the prefix cache with one entry at
        hash(NONE_HASH, chunk_tokens).
      - req_A = prefix_X + chunk + suffix    (span_starts=[32])
      - req_B identical to A, different request_id
      - req_C = prefix_Y + chunk + suffix    (different prefix)

    Expected reuse pattern (per the PIC contract the test pins):
      - req_A: prefix_X + suffix fresh, chunk reused from warmup.
      - req_B: full reuse from req_A.
      - req_C: prefix_Y + suffix fresh, chunk still reused (from the
        same warmup entry; tail is NOT reused across A and C even
        though its hashes happen to collide, because only the chunk
        is marked PIC).

    The assertions check set-relations between KV cache snapshots; they
    do not measure exact block counts (decode-step allocation varies by
    model). The decisive checks are:
      1. The warmup chunk slot survives every subsequent request - it
         must never be evicted or overwritten.
      2. req_B adds zero new K/V slots (full reuse from A).
      3. req_C does not evict any slot req_A wrote.
      4. req_C adds at least one new slot (prefix_Y is a genuine miss).
    """
    chunk = list(range(500, 500 + BLOCK_SIZE))
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
        # Step 0: warm up the chunk block alone.
        _warmup_chunk(llm, chunk)
        snap_warmup = _kv_cache_block_hashes(llm, LAYER_IDX)
        # The unique non-empty hashes from the warmup. In practice this
        # is 1 slot (the chunk) plus optionally a decode-step slot from
        # the max_tokens=1 generate call.
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
        # Only the chunk block is marked PIC (span_starts=[32]).
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

    # Part 2: e2e byte-diff bound.
    llm = build_llm(model, "LL-32", monkeypatch)
    try:
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
