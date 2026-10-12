# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import PrefixHitSource
from vllm.v1.core.sched.gap_policy import (
    GapPolicyFactory,
    NoGapPolicy,
    SpanAwareGapPolicy,
    miss_gaps,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.spans


def make_span_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
    cross_span_starts: list[int] | None = None,
) -> Request:
    extra_args = {}
    if span_starts is not None:
        extra_args["span_starts"] = span_starts
    if cross_span_starts is not None:
        extra_args["cross_span_starts"] = cross_span_starts

    sampling_params = SamplingParams(
        max_tokens=17,
        extra_args=extra_args if extra_args else None,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="gap_test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
    )


class TestSpanAwareGapPolicy:
    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_gaps_at_span_starts(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 96), (128, 160)]

    def test_no_gaps_when_no_span_starts(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=None)
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_gap_clamped_to_next_span(self):
        policy = SpanAwareGapPolicy(gap_length=100)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 128), (128, 228)]

    def test_gap_clamped_to_computed_tokens(self):
        policy = SpanAwareGapPolicy(gap_length=100)
        req = make_span_request(256, span_starts=[200])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(200, 256)]

    def test_zero_gap_length_disables(self):
        policy = SpanAwareGapPolicy(gap_length=0)
        req = make_span_request(256, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_span_starts_beyond_computed_tokens_ignored(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 200])
        gaps = policy.get_gaps(req, num_computed_tokens=100, num_external_tokens=0)
        assert gaps == [(64, 96)]

    def test_no_gap_policy_returns_empty(self):
        policy = NoGapPolicy()
        req = make_span_request(256, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_factory_creates_span_aware(self):
        policy = GapPolicyFactory.create_policy("span_aware", {"gap_length": 64})
        assert isinstance(policy, SpanAwareGapPolicy)
        assert policy.gap_length == 64


def test_miss_gaps_cover_only_miss_blocks():
    req = make_span_request(256, span_starts=[64])
    D, M, S = PrefixHitSource.PD, PrefixHitSource.MISS, PrefixHitSource.PIC
    req.prefix_hit_sources = [M, M, D, S, M, S]
    assert miss_gaps(req, 16) == [(0, 16), (16, 32), (64, 80)]
    req.prefix_hit_sources = None
    assert miss_gaps(req, 16) == []


def test_qcfuse_selects_inside_spans_only(monkeypatch):
    from vllm.v1.core.sched.gap_policy import QCFusePolicy

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_PREROTATE", True)
    req = make_span_request(160, span_starts=[64])
    imp = [0.0] * 160
    imp[0] = 100.0  # the sink: exact prefix, never worth recomputing
    imp[85] = 1.0
    assert QCFusePolicy(k_per_span=16)._select(req, imp, 160) == [(80, 96)]
    assert QCFusePolicy(rho=0.2)._select(req, imp, 160) == [(80, 96)]


def test_drop_pd_gaps_cuts_pd_blocks_out():
    from vllm.v1.core.sched.gap_policy import _drop_pd_gaps

    req = make_span_request(64, span_starts=[0])
    D, M, S = PrefixHitSource.PD, PrefixHitSource.MISS, PrefixHitSource.PIC
    req.prefix_hit_sources = [D, S, D, S]
    assert _drop_pd_gaps(req, [(0, 64)], 16) == [(16, 32), (48, 64)]
    assert _drop_pd_gaps(req, [(20, 40)], 16) == [(20, 32)]
    req.prefix_hit_sources = [D, D, D, D]
    assert _drop_pd_gaps(req, [(0, 64)], 16) == []
    req.prefix_hit_sources = [S, M, S, M]
    assert _drop_pd_gaps(req, [(0, 64)], 16) == [(0, 64)]


def test_tail_after_span_hits_pd_copy_never_the_blind_plain_one(monkeypatch):
    """A tail block caches blind KV under its plain key; a repaired copy under
    its pd key must win, and the plain one alone is labeled PIC, not PD."""
    from types import SimpleNamespace

    from vllm.utils.hashing import sha256
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import UnitaryKVCacheCoordinator
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    init_none_hash(sha256)
    pool = BlockPool(num_gpu_blocks=8, enable_caching=True, hash_block_size=16)
    sp = SamplingParams(
        max_tokens=1, extra_args={"span_starts": [16], "cross_span_starts": [48]}
    )
    sp.update_from_generation_config({}, eos_token_id=100)
    req = Request("r", list(range(64)), sp, None,
                  block_hasher=get_request_block_hasher(16, sha256))
    coord = SimpleNamespace(block_size=16, block_pool=pool)
    lookup = UnitaryKVCacheCoordinator.find_longest_cache_hit_dual
    D, S = PrefixHitSource.PD, PrefixHitSource.PIC

    blind = pool.get_new_blocks(4)
    pool.cache_full_blocks(req, blind, 0, 4, 16, 0)
    (got,), _, src = lookup(coord, req, 64)
    assert got == blind and src == [D, S, S, S]

    fresh = pool.get_new_blocks(1)
    pool.cache_blocks_under_hashes(fresh, [req.pd_block_hashes[3]], 0)
    (got,), _, src = lookup(coord, req, 64)
    assert got[3] is fresh[0] and blind[3].block_hash is not None
    assert src == [D, S, S, D]
