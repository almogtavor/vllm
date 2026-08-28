# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.gap_policy import (
    GapPolicyFactory,
    NoGapPolicy,
    QCFusePolicy,
    SpanAwareGapPolicy,
)
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
