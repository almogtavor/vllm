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
    """legoquest: whole spans, or only repairs the query can actually see.

    A partly repaired span mixes prefix-aware and prefix-free KV. On
    gemma-4-31b (sliding_window=1024, span 2048) that measured worse than
    leaving the span warm - 0.1146 vs 0.1292 generated-token match - because
    the query sits past the window and never attends the repaired prefix. On
    Qwen3-32B (no window) the same repair helped, 0.399 -> 0.805.
    """

    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_span_within_budget_is_repaired_whole(self):
        policy = LegoQuestGapPolicy(gap_length=128, block_size=16)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=192, num_external_tokens=0)
        assert gaps == [(64, 128), (128, 192)], gaps

    def test_no_sliding_window_keeps_partial_repair(self):
        # Full-attention models: partial repair helps, so match span_aware.
        lego = LegoQuestGapPolicy(gap_length=128, block_size=16)
        base = SpanAwareGapPolicy(gap_length=128, block_size=16)
        req = make_span_request(2048, span_starts=[64])
        kw = dict(num_computed_tokens=2048, num_external_tokens=0)
        assert lego.get_gaps(req, **kw) == base.get_gaps(req, **kw)

    def test_repair_behind_the_window_is_skipped(self):
        # Span at 64, budget 128 -> repair ends at 192. Query at 2048 with a
        # 1024 window attends back to 1024, so the repair is invisible.
        policy = LegoQuestGapPolicy(gap_length=128, block_size=16, sliding_window=1024)
        req = make_span_request(2048, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=2048, num_external_tokens=0)
        assert gaps == [], gaps

    def test_position_within_the_window_does_not_re_enable_partial_repair(self):
        # Measured: on gemma-4 a 1024-token span sitting entirely inside the
        # query's window still gained nothing from repairing its first 256
        # (0.0911 vs 0.0911 warm, 4 offsets). The gate is the architecture, not
        # where the repair lands.
        policy = LegoQuestGapPolicy(gap_length=128, block_size=16, sliding_window=1024)
        req = make_span_request(2048, span_starts=[1600])
        gaps = policy.get_gaps(req, num_computed_tokens=2048, num_external_tokens=0)
        assert gaps == [], gaps

    def test_never_recomputes_more_than_span_aware(self):
        for window in (0, 1024):
            for spans in ([64], [0, 128], [0, 128, 256], [64, 700]):
                lego = LegoQuestGapPolicy(
                    gap_length=128, block_size=16, sliding_window=window
                )
                base = SpanAwareGapPolicy(gap_length=128, block_size=16)
                req = make_span_request(1024, span_starts=spans)
                kw = dict(num_computed_tokens=1024, num_external_tokens=0)
                n_lego = sum(e - s for s, e in lego.get_gaps(req, **kw))
                n_base = sum(e - s for s, e in base.get_gaps(req, **kw))
                assert n_lego <= n_base, (window, spans, n_lego, n_base)

    def test_factory_creates_legoquest(self):
        policy = GapPolicyFactory.create_policy("span_legoquest", {"gap_length": 64})
        assert isinstance(policy, LegoQuestGapPolicy)
        assert policy.gap_length == 64


class TestPdTrim:
    """Already prefix-aware blocks must not be recomputed again."""

    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_leading_pd_blocks_are_trimmed_not_redone(self):
        from vllm.v1.core.kv_cache_utils import PrefixHitSource

        policy = SpanAwareGapPolicy(gap_length=128, block_size=16)
        req = make_span_request(512, span_starts=[64])
        # blocks 4..7 (tokens 64..127) already prefix-aware from an earlier turn
        req.prefix_hit_sources = [PrefixHitSource.PIC] * 32
        for b in range(4, 8):
            req.prefix_hit_sources[b] = PrefixHitSource.PD
        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)
        # gap would be (64,192); the first 4 blocks are pd -> start at 128
        assert gaps == [(128, 192)], gaps

    def test_all_pd_gap_still_dropped(self):
        from vllm.v1.core.kv_cache_utils import PrefixHitSource

        policy = SpanAwareGapPolicy(gap_length=128, block_size=16)
        req = make_span_request(512, span_starts=[64])
        req.prefix_hit_sources = [PrefixHitSource.PD] * 32
        gaps = policy.get_gaps(req, num_computed_tokens=512, num_external_tokens=0)
        assert gaps == []
