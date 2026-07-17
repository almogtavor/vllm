# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.span_metadata import build_span_attention_metadata

pytestmark = pytest.mark.cpu_test


def test_virtual_gap_recompute_keeps_span_bounds_without_quest_scoring():
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
    assert attn_lb[16:48].tolist() == [16] * 32
    assert attn_lb[48:52].tolist() == [0] * 4
    assert quest_descs == []
    assert quest_scores == []
