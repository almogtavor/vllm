# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.gap_policy import (
    GapPolicyFactory,
    MassClosurePolicy,
    NoGapPolicy,
    QCFusePolicy,
    SpanAwareGapPolicy,
)
from vllm.v1.core.sched.qcfuse_store import QCFuseImportanceStore
from vllm.v1.request import Request

pytestmark = pytest.mark.spans


def make_span_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
    cross_span_starts: list[int] | None = None,
    qcfuse_importance: list[float] | None = None,
) -> Request:
    extra_args: dict[str, object] = {}
    if span_starts is not None:
        extra_args["span_starts"] = span_starts
    if cross_span_starts is not None:
        extra_args["cross_span_starts"] = cross_span_starts
    if qcfuse_importance is not None:
        extra_args["qcfuse_importance"] = qcfuse_importance

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


def _importance_with_hot_blocks(
    prompt_len: int, hot_blocks: list[int], block_size: int = 16
) -> list[float]:
    """Flat 1.0 everywhere, 100.0 on the tokens of the named blocks."""
    imp = [1.0] * prompt_len
    for b in hot_blocks:
        for t in range(b * block_size, (b + 1) * block_size):
            imp[t] = 100.0
    return imp


class TestQCFusePolicy:
    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_extra_args_importance_reaches_request(self):
        imp = _importance_with_hot_blocks(256, [3])
        req = make_span_request(256, qcfuse_importance=imp)
        assert req.qcfuse_importance == imp

    def test_importance_not_read_when_spans_disabled(self):
        envs.VLLM_V1_SPANS_ENABLED = False
        req = make_span_request(256, qcfuse_importance=[1.0] * 256)
        assert req.qcfuse_importance is None

    def test_no_gaps_before_probe_runs(self):
        policy = QCFusePolicy(rho=0.25, critical_layers="2,5", block_size=16)
        req = make_span_request(256)
        assert req.qcfuse_importance is None
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_block_aligned_gaps_honor_rho(self):
        # rho=0.25 of 256 computed tokens = 64 tokens = 4 blocks of 16.
        hot = [2, 5, 9, 13]
        policy = QCFusePolicy(rho=0.25, critical_layers="2,5", block_size=16)
        req = make_span_request(
            256, qcfuse_importance=_importance_with_hot_blocks(256, hot)
        )
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(b * 16, (b + 1) * 16) for b in hot]
        assert sum(e - s for s, e in gaps) == int(0.25 * 256)
        assert all(s % 16 == 0 and e % 16 == 0 for s, e in gaps)

    def test_rho_shrinks_the_budget(self):
        hot = [2, 5, 9, 13]
        policy = QCFusePolicy(rho=0.125, critical_layers="2", block_size=16)
        req = make_span_request(
            256, qcfuse_importance=_importance_with_hot_blocks(256, hot)
        )
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert len(gaps) == 2
        assert sum(e - s for s, e in gaps) == 32
        assert {s // 16 for s, _ in gaps} <= set(hot)

    def test_gaps_confined_to_computed_prefix(self):
        policy = QCFusePolicy(rho=0.25, critical_layers="2", block_size=16)
        req = make_span_request(
            256, qcfuse_importance=_importance_with_hot_blocks(256, [2, 12])
        )
        gaps = policy.get_gaps(req, num_computed_tokens=128, num_external_tokens=0)
        assert gaps
        assert all(0 <= s < e <= 128 for s, e in gaps)

    def test_token_granularity_selects_individual_tokens(self):
        imp = [0.0] * 64
        for t in (5, 17, 33, 60):
            imp[t] = 9.0
        policy = QCFusePolicy(
            rho=0.0625, critical_layers="2", block_size=16, granularity="token"
        )
        req = make_span_request(64, qcfuse_importance=imp)
        gaps = policy.get_gaps(req, num_computed_tokens=64, num_external_tokens=0)
        assert len(gaps) == 4
        assert [s for s, _ in gaps] == [5, 17, 33, 60]

    def test_factory_creates_qcfuse(self):
        policy = GapPolicyFactory.create_policy(
            "qcfuse", {"rho": 0.2, "critical_layers": "1,4,7", "granularity": "block"}
        )
        assert isinstance(policy, QCFusePolicy)
        assert policy.rho == 0.2
        assert policy.critical_layers == (1, 4, 7)

    def test_empty_critical_layers_rejected(self):
        with pytest.raises(ValueError):
            QCFusePolicy(rho=0.1, critical_layers="")


class TestQCFuseImportanceStore:
    """The probe cannot serve the request that produced it: get_gaps runs from
    the waiting queue before that request's first forward. The store is what
    carries importance to the next request over the same blocks."""

    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    @staticmethod
    def _hashes(n: int, tag: bytes = b"h") -> list[BlockHash]:
        return [BlockHash(tag + bytes([i])) for i in range(n)]

    def test_lookup_is_none_before_anything_measured(self):
        store = QCFuseImportanceStore(block_size=16)
        assert store.lookup(self._hashes(16), 256) is None

    def test_roundtrip_reproduces_the_measured_vector(self):
        store = QCFuseImportanceStore(block_size=16)
        hashes = self._hashes(16)
        imp = _importance_with_hot_blocks(256, [2, 5, 9, 13])
        store.store(hashes, imp)
        assert store.lookup(hashes, 256) == imp

    def test_measured_importance_drives_the_next_requests_gaps(self):
        store = QCFuseImportanceStore(block_size=16)
        hashes = self._hashes(16)
        hot = [2, 5, 9, 13]
        store.store(hashes, _importance_with_hot_blocks(256, hot))

        policy = QCFusePolicy(rho=0.25, critical_layers="2,5", block_size=16)
        nxt = make_span_request(256)
        assert policy.get_gaps(nxt, 256, 0) == []
        nxt.qcfuse_importance = store.lookup(hashes, 256)
        assert policy.get_gaps(nxt, 256, 0) == [(b * 16, (b + 1) * 16) for b in hot]

    def test_unmeasured_blocks_score_zero_and_are_never_picked(self):
        store = QCFuseImportanceStore(block_size=16)
        known = self._hashes(16)
        store.store(known[:8], _importance_with_hot_blocks(128, [2, 5]))
        mixed = known[:8] + self._hashes(8, tag=b"z")
        out = store.lookup(mixed, 256)
        assert out is not None and len(out) == 256
        assert out[128:] == [0.0] * 128
        policy = QCFusePolicy(rho=0.125, critical_layers="2", block_size=16)
        req = make_span_request(256)
        req.qcfuse_importance = out
        assert policy.get_gaps(req, 256, 0) == [(32, 48), (80, 96)]

    def test_lru_evicts_oldest_blocks(self):
        store = QCFuseImportanceStore(block_size=16, max_blocks=4)
        store.store(self._hashes(8), [1.0] * 128)
        assert len(store._by_hash) == 4

    def test_later_measurement_overwrites(self):
        store = QCFuseImportanceStore(block_size=16)
        hashes = self._hashes(2)
        store.store(hashes, [1.0] * 32)
        store.store(hashes, [7.0] * 32)
        assert store.lookup(hashes, 32) == [7.0] * 32


class TestMassClosurePolicy:
    """Selection by attention x staleness x closure, at legolink's budget.

    Every test fixes the budget at 4 blocks per span so the comparisons are
    about which blocks get picked, never about how many.
    """

    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    @staticmethod
    def _policy(**kw) -> MassClosurePolicy:
        cfg = dict(rho=0.1, critical_layers="2,5", block_size=16, k_per_span=64)
        cfg.update(kw)
        return MassClosurePolicy(**cfg)

    @staticmethod
    def _blocks(gaps: list[tuple[int, int]], block_size: int = 16) -> list[int]:
        return [b for s, e in gaps for b in range(s // block_size, e // block_size)]

    def test_no_gaps_before_probe_runs(self):
        policy = self._policy()
        req = make_span_request(512, span_starts=[0])
        assert req.qcfuse_importance is None
        assert policy.get_gaps(req, 512, 0) == []

    def test_no_gaps_without_spans(self):
        # This policy selects inside spans; a request with none has nothing to
        # repair, unlike QCFuse which ranks the whole context.
        policy = self._policy()
        req = make_span_request(512, qcfuse_importance=[1.0] * 512)
        assert policy.get_gaps(req, 512, 0) == []

    def test_budget_is_k_per_span_and_block_aligned(self):
        policy = self._policy()
        req = make_span_request(
            512,
            span_starts=[0],
            qcfuse_importance=_importance_with_hot_blocks(512, [7, 19, 26]),
        )
        gaps = policy.get_gaps(req, 512, 0)
        assert sum(e - s for s, e in gaps) == 64
        assert all(s % 16 == 0 and e % 16 == 0 for s, e in gaps)

    def test_budget_is_per_span_not_global(self):
        policy = self._policy()
        req = make_span_request(
            512,
            span_starts=[0, 256],
            qcfuse_importance=_importance_with_hot_blocks(512, [7, 26]),
        )
        gaps = policy.get_gaps(req, 512, 0)
        # Two spans at 64 tokens each: same total as legolink-64 on two spans.
        assert sum(e - s for s, e in gaps) == 128
        picked = self._blocks(gaps)
        assert any(b < 16 for b in picked) and any(b >= 16 for b in picked)

    def test_anchor_block_makes_it_contain_legolink_16(self):
        policy = self._policy(anchor_blocks=1)
        req = make_span_request(
            512,
            span_starts=[0, 256],
            # Hot blocks deliberately far from either span head.
            qcfuse_importance=_importance_with_hot_blocks(512, [12, 28]),
        )
        picked = self._blocks(policy.get_gaps(req, 512, 0))
        assert 0 in picked and 16 in picked

    def test_closure_prefers_a_block_whose_predecessors_are_repaired(self):
        # Two equally hot blocks, one right after the repaired head and one far
        # away. Attention alone cannot separate them; closure can, and prefers
        # the near one because its context is already correct.
        policy = self._policy(anchor_blocks=1, k_per_span=32)
        req = make_span_request(
            512,
            span_starts=[0],
            qcfuse_importance=_importance_with_hot_blocks(512, [1, 25]),
        )
        picked = self._blocks(policy.get_gaps(req, 512, 0))
        assert picked == [0, 1]

    def test_attention_still_dominates_a_large_enough_gap(self):
        # Closure is a multiplier, not a veto: a far block that is hot enough
        # still wins. Otherwise this would just be legolink with extra steps.
        policy = self._policy(anchor_blocks=1, k_per_span=32)
        imp = [1.0] * 512
        for t in range(25 * 16, 26 * 16):
            imp[t] = 10_000.0
        req = make_span_request(512, span_starts=[0], qcfuse_importance=imp)
        picked = self._blocks(policy.get_gaps(req, 512, 0))
        assert picked == [0, 25]

    def test_large_c_degenerates_to_plain_attention(self):
        # The closure term is only meaningful while c is small; this pins that
        # claim so a future default change cannot silently disable the method.
        hot = [3, 11, 20, 29]
        policy = self._policy(c=1000.0, anchor_blocks=0)
        req = make_span_request(
            512,
            span_starts=[0],
            qcfuse_importance=_importance_with_hot_blocks(512, hot),
        )
        assert self._blocks(policy.get_gaps(req, 512, 0)) == hot

    def test_gaps_confined_to_computed_prefix(self):
        policy = self._policy()
        req = make_span_request(
            512,
            span_starts=[0],
            qcfuse_importance=_importance_with_hot_blocks(512, [3, 20]),
        )
        gaps = policy.get_gaps(req, 256, 0)
        assert gaps and all(0 <= s < e <= 256 for s, e in gaps)

    def test_adjacent_blocks_coalesce_into_one_gap(self):
        policy = self._policy(anchor_blocks=4, k_per_span=64)
        req = make_span_request(512, span_starts=[0], qcfuse_importance=[1.0] * 512)
        gaps = policy.get_gaps(req, 512, 0)
        assert gaps[0] == (0, 64) and len(gaps) == 1

    def test_span_head_always_has_full_closure(self):
        # Block 0 has no in-span predecessors, so its closure is 1 whatever the
        # kernel says. Greedy therefore opens on it unless something else is
        # overwhelmingly hot, which is what makes the anchor nearly free.
        imp = [1.0] * 512
        for t in range(20 * 16, 21 * 16):
            imp[t] = 50.0
        policy = self._policy(anchor_blocks=0, k_per_span=32)
        picked = self._blocks(
            policy.get_gaps(
                make_span_request(512, span_starts=[0], qcfuse_importance=imp),
                512,
                0,
            )
        )
        assert picked[0] == 0

    def test_closure_kernel_matches_the_measured_shape(self):
        # amp*d^-beta + floor: fast decay onto a floor, not a bare power law,
        # plus a sink every row spends on the span head. Pinned because the
        # sink is the constant the method depends on -- an order of magnitude
        # too small and this degenerates to plain attention ranking.
        policy = self._policy()
        w, rowtot = policy._closure_weights(64)
        assert w[0] == 0.0
        assert w[1] > w[2] > w[8] > w[32] > policy.floor
        # flat tail: the last third of a span contributes near-equally
        assert w[32] / w[63] < 1.2
        # every row total carries the sink, and the sink dominates a short row
        assert rowtot[0] == 0.0
        assert rowtot[1] == pytest.approx(w[1] + policy.sink)
        assert policy.sink / rowtot[1] > 0.85

    def test_factory_creates_mass_closure(self):
        policy = GapPolicyFactory.create_policy(
            "mass_closure",
            {
                "rho": 0.1,
                "critical_layers": "1,4,7",
                "k_per_span": 256,
                "c": 0.05,
                "alpha": 0.4,
            },
        )
        assert isinstance(policy, MassClosurePolicy)
        assert policy.k_per_span == 256 and policy.c == 0.05 and policy.alpha == 0.4

    def test_missing_per_span_budget_rejected(self):
        # rho is a whole-context ratio and means nothing per span, so a config
        # without k_per_span would silently select nothing.
        with pytest.raises(ValueError):
            self._policy(k_per_span=0)

    def test_empty_critical_layers_rejected(self):
        with pytest.raises(ValueError):
            self._policy(critical_layers="")
