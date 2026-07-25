# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.gap_policy import (
    GapPolicyFactory,
    LegoQuestGapPolicy,
    NoGapPolicy,
    QuestGapPolicy,
    SpanAwareGapPolicy,
)
from vllm.v1.request import Request

pytestmark = pytest.mark.spans


def make_span_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
    cross_span_starts: list[int] | None = None,
    prompt_token_ids: list[int] | None = None,
) -> Request:
    if prompt_token_ids is None:
        prompt_token_ids = list(range(prompt_len))
    else:
        prompt_len = len(prompt_token_ids)

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
        prompt_token_ids=prompt_token_ids,
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


class TestQuestGapPolicy:
    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_no_selection_matches_span_aware(self):
        policy = QuestGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 96), (128, 160)]

    def test_selection_emits_selected_quest_blocks(self):
        policy = QuestGapPolicy(
            gap_length=32, block_size=16, anchor_blocks=0
        )  # budget: 2 blocks
        req = make_span_request(256, span_starts=[64], cross_span_starts=[224])
        req.block_hashes = [bytes([b]) for b in range(256 // 16)]
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [7, 2])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(96, 112), (176, 192)]

    def test_selection_clamped_and_per_span(self):
        policy = QuestGapPolicy(gap_length=32, block_size=16, anchor_blocks=0)
        req = make_span_request(
            256, span_starts=[64, 128], cross_span_starts=[224, 240]
        )
        req.block_hashes = [bytes([b]) for b in range(256 // 16)]
        # offset 5 -> 144, past the next span at 128: dropped. Span 2 has no
        # selection stored -> contiguous fallback.
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [1, 5])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(80, 96), (128, 160)]

    def test_factory_creates_quest(self):
        policy = GapPolicyFactory.create_policy("span_quest", {"gap_length": 64})
        assert isinstance(policy, QuestGapPolicy)
        assert policy.gap_length == 64

    def test_default_anchor_uses_first_128_tokens_within_budget(self):
        policy = QuestGapPolicy(gap_length=256, block_size=16)
        req = make_span_request(512, span_starts=[64], cross_span_starts=[448])
        req.block_hashes = [bytes([b]) for b in range(512 // 16)]
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [18])

        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)

        assert gaps == [(64, 192), (352, 368)]

    def test_no_selection_defaults_to_anchor_not_full_budget(self):
        policy = QuestGapPolicy(gap_length=256, block_size=16)
        req = make_span_request(512, span_starts=[64], cross_span_starts=[448])

        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)

        assert gaps == [(64, 192)]

    def test_adjacent_selected_blocks_coalesce(self):
        policy = QuestGapPolicy(
            gap_length=48, block_size=16, anchor_blocks=0
        )  # budget: 3
        req = make_span_request(256, span_starts=[64], cross_span_starts=[224])
        req.block_hashes = [bytes([b]) for b in range(256 // 16)]
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [3, 4, 0])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 80), (112, 144)]

    def test_selection_requires_matching_following_query_tokens(self):
        policy = QuestGapPolicy(gap_length=32, block_size=16)
        req = make_span_request(256, span_starts=[64], cross_span_starts=[224])
        req.block_hashes = [bytes([b]) for b in range(256 // 16)]
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [7, 2])

        prompt_token_ids = list(range(256))
        prompt_token_ids[224] = 9999
        other = make_span_request(
            256,
            span_starts=[64],
            cross_span_starts=[224],
            prompt_token_ids=prompt_token_ids,
        )
        other.block_hashes = req.block_hashes

        gaps = policy.get_gaps(other, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 96)]

    def test_selection_uses_anchor_within_same_budget(self):
        policy = QuestGapPolicy(gap_length=48, block_size=16, anchor_blocks=1)
        req = make_span_request(256, span_starts=[64], cross_span_starts=[224])
        req.block_hashes = [bytes([b]) for b in range(256 // 16)]
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [7, 2])

        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)

        assert gaps == [(64, 80), (96, 112), (176, 192)]

    def test_anchor_capped_at_half_budget_when_selection_exists(self):
        # gap-128: budget 8 blocks; the default 8-block anchor must leave
        # half the budget for query-selected blocks.
        policy = QuestGapPolicy(gap_length=128, block_size=16, anchor_blocks=8)
        req = make_span_request(512, span_starts=[64], cross_span_starts=[480])
        req.block_hashes = [bytes([b]) for b in range(512 // 16)]
        key = policy.get_selection_key(req, 64)
        assert key is not None
        policy.store_selection(key, [20, 10, 15, 12, 6])

        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)

        # anchor blocks 0-3 (one coalesced gap) + selected offsets 20,10,15,12
        assert gaps == [
            (64, 128),
            (224, 240),
            (256, 272),
            (304, 320),
            (384, 400),
        ]


class TestLegoQuestGapPolicy:
    """legoquest: query-aware budget ALLOCATION,
    contiguous prefix within a span."""

    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_no_selection_matches_span_aware(self):
        # With no scores stored the budget is uniform, i.e. exactly span_aware.
        policy = LegoQuestGapPolicy(gap_length=32, block_size=16)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 96), (128, 160)]

    def test_gaps_are_always_contiguous_prefixes(self):
        # The whole point: even for a late-attended block, never scatter.
        policy = LegoQuestGapPolicy(gap_length=32, block_size=16)
        req = make_span_request(256, span_starts=[64], cross_span_starts=[224])
        req.block_hashes = [bytes([b]) for b in range(256 // 16)]
        key = policy.get_selection_key(req, 64)
        policy.store_selection(key, [5, 1])  # attention reaches offset 5
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        # one range starting at the span start (not scattered blocks 1 and 5)
        assert len(gaps) == 1
        assert gaps[0][0] == 64

    def test_single_span_equals_span_aware(self):
        # With one span there is nothing to reallocate from: total budget is
        # per_span * 1, so devia must degrade exactly to span_aware.
        policy = LegoQuestGapPolicy(gap_length=32, block_size=16)
        req = make_span_request(512, span_starts=[64], cross_span_starts=[480])
        req.block_hashes = [bytes([b]) for b in range(512 // 16)]
        key = policy.get_selection_key(req, 64)
        policy.store_selection(key, [5, 0])
        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)
        assert gaps == [(64, 96)]  # 2 blocks == gap_length

    def test_budget_moves_from_shallow_span_to_deep_span(self):
        # THE point of devia: span A's query attention reaches offset 3 (needs 4
        # blocks), span B's only block 0 (needs 1). span_aware would spend 2+2;
        # devia spends 4+1, same total, allocated where the query looks.
        policy = LegoQuestGapPolicy(gap_length=32, block_size=16)
        req = make_span_request(
            512, span_starts=[64, 256], cross_span_starts=[240, 500]
        )
        req.block_hashes = [bytes([b]) for b in range(512 // 16)]
        policy.store_selection(policy.get_selection_key(req, 64), [3, 0])
        policy.store_selection(policy.get_selection_key(req, 256), [0])
        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)
        # budget is 4 blocks total; wants 4+1 -> scaled to 3+1 (deep span still
        # gains over the uniform 2, shallow span gives one up). Both contiguous.
        assert gaps == [(64, 112), (256, 272)]
        assert sum((e - s) // 16 for s, e in gaps) <= 2 * (32 // 16)

    def test_total_budget_not_exceeded_when_oversubscribed(self):
        # Two spans both wanting deep repair must still fit 2*gap_length total.
        policy = LegoQuestGapPolicy(gap_length=32, block_size=16)
        req = make_span_request(
            512, span_starts=[64, 256], cross_span_starts=[240, 500]
        )
        req.block_hashes = [bytes([b]) for b in range(512 // 16)]
        for s in (64, 256):
            key = policy.get_selection_key(req, s)
            policy.store_selection(key, [10, 0])
        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)
        blocks = sum((e - s) // 16 for s, e in gaps)
        assert blocks <= 2 * (32 // 16)  # total budget preserved
        assert all(s in (64, 256) for s, _ in gaps)  # still prefixes

    def test_factory_creates_legoquest(self):
        policy = GapPolicyFactory.create_policy("span_legoquest", {"gap_length": 64})
        assert isinstance(policy, LegoQuestGapPolicy)
        assert policy.gap_length == 64
