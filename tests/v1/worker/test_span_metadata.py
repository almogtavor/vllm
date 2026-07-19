# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.request import Request
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.span_metadata import (
    build_span_attention_metadata,
    compute_span_lb_regions,
)

pytestmark = pytest.mark.cpu_test


def test_adjacent_spans_stop_at_next_boundary():
    regions = compute_span_lb_regions([16, 48], [80], 128)
    assert regions == [(16, 48, 16), (48, 80, 48)]


def test_quest_scores_adjacent_spans_against_following_query():
    req = CachedRequestState(
        req_id="adjacent",
        prompt_token_ids=list(range(96)),
        mm_features=[],
        sampling_params=SamplingParams(
            extra_args={
                "span_starts": [16, 48],
                "cross_span_starts": [80],
            }
        ),
        generator=None,
        block_ids=([0, 1, 2, 3, 4, 5],),
        num_computed_tokens=0,
        output_token_ids=[],
    )

    attn_lb, _, quest_descs, quest_scores = build_span_attention_metadata(
        [req],
        np.array([0], dtype=np.int32),
        np.array([96], dtype=np.int32),
        np.array([0], dtype=np.int32),
        block_size=16,
        quest_top_k=1,
        device="cpu",
    )

    assert attn_lb[16:48].tolist() == [16] * 32
    assert attn_lb[48:80].tolist() == [48] * 32
    assert quest_descs == [(0, 1, 2, 80), (0, 3, 2, 80)]
    assert len(quest_scores) == 2


def test_request_pic_token_ranges_stop_at_next_span():
    import vllm.envs as envs

    original = envs.VLLM_V1_SPANS_ENABLED
    try:
        envs.VLLM_V1_SPANS_ENABLED = True
        sampling_params = SamplingParams(
            max_tokens=8,
            extra_args={
                "span_starts": [16, 48],
                "cross_span_starts": [80],
            },
        )
        sampling_params.update_from_generation_config({}, eos_token_id=100)
        req = Request(
            request_id="adjacent_spans",
            prompt_token_ids=list(range(128)),
            sampling_params=sampling_params,
            pooling_params=None,
        )
        assert req.pic_token_ranges == [(16, 48), (48, 80)]
    finally:
        envs.VLLM_V1_SPANS_ENABLED = original


def test_virtual_gap_recompute_attends_prefix_without_quest_scoring():
    gap_req = CachedRequestState(
        req_id="gap",
        prompt_token_ids=list(range(48)),
        mm_features=[],
        sampling_params=SamplingParams(
            extra_args={
                "span_starts": [16],
                "cross_span_starts": [64],
            }
        ),
        generator=None,
        block_ids=([0, 1, 2],),
        num_computed_tokens=32,
        output_token_ids=[],
        is_gap_recompute=True,
    )
    next_req = CachedRequestState(
        req_id="next",
        prompt_token_ids=[1, 2, 3, 4],
        mm_features=[],
        sampling_params=SamplingParams(),
        generator=None,
        block_ids=([3],),
        num_computed_tokens=0,
        output_token_ids=[],
    )

    attn_lb, req_starts, quest_descs, quest_scores = (
        build_span_attention_metadata(
            [gap_req, next_req],
            np.array([32, 0], dtype=np.int32),
            np.array([16, 4], dtype=np.int32),
            np.array([0, 16], dtype=np.int32),
            block_size=16,
            quest_top_k=1,
            device="cpu",
        )
    )

    assert req_starts.tolist() == [0, 48, 52]
    # A gap recompute must attend the real prefix: its lower bound stays 0
    # (NOT span_start), otherwise the recompute reproduces the prefix-free warm
    # KV and repairs nothing.
    assert attn_lb[16:48].tolist() == [0] * 32
    assert attn_lb[48:52].tolist() == [0] * 4
    assert quest_descs == []
    assert quest_scores == []
