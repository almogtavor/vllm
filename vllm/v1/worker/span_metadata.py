# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import numpy as np
import torch

from vllm.v1.worker.gpu_input_batch import CachedRequestState


def compute_span_lb_regions(
    span_starts: list[int],
    cross_span_starts: list[int] | None,
    req_len: int,
) -> list[tuple[int, int, int]]:
    """Return (start, end, lower_bound) regions for span attention."""
    crosses = sorted(cross_span_starts or [])
    spans_sorted = sorted(span_starts)
    regions = []
    for idx, span_start in enumerate(spans_sorted):
        next_span = spans_sorted[idx + 1] if idx + 1 < len(spans_sorted) else req_len
        cross = next((c for c in crosses if c > span_start), req_len)
        regions.append((span_start, min(next_span, cross), span_start))
    return regions


def _following_query_start(
    span_start: int,
    span_starts: list[int],
    cross_span_starts: list[int] | None,
) -> int | None:
    crosses = cross_span_starts or []
    for idx, start in enumerate(span_starts):
        if start == span_start and idx < len(crosses):
            cross = crosses[idx]
            return cross if cross > span_start else None
    return next((cross for cross in sorted(crosses) if cross > span_start), None)


def build_span_attention_metadata(
    req_states: Sequence[CachedRequestState],
    num_computed: np.ndarray,
    num_scheduled_tokens: np.ndarray,
    q_start: np.ndarray,
    block_size: int,
    quest_top_k: int,
    device: torch.device | str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[tuple[int, int, int, int]],
    list[torch.Tensor],
]:
    seq_lens_arr = num_computed + num_scheduled_tokens
    req_kv_starts = np.zeros(len(req_states) + 1, dtype=np.int32)
    np.cumsum(seq_lens_arr, out=req_kv_starts[1:])
    attn_lb = np.zeros(int(req_kv_starts[-1]), dtype=np.int32)
    quest_score_descs: list[tuple[int, int, int, int]] = []
    quest_span_scores: list[torch.Tensor] = []

    for i, req in enumerate(req_states):
        params = req.sampling_params
        if params is None:
            continue
        extra_args = params.extra_args
        spans = extra_args.get("span_starts") if extra_args else None
        if not spans:
            continue
        req_start = int(req_kv_starts[i])
        req_len = int(seq_lens_arr[i])
        nc = int(num_computed[i])
        ns = int(num_scheduled_tokens[i])
        crosses = extra_args.get("cross_span_starts")
        for span_start, region_end, _ in compute_span_lb_regions(
            spans, crosses, req_len
        ):
            if span_start >= req_len:
                continue
            region_end = min(region_end, req_len)
            if region_end <= span_start:
                continue
            # The lower bound both (1) skips prefix tiles for the span's own
            # warm query and (2) shifts span keys to span-relative RoPE. A gap
            # recompute's query MUST attend the real prefix (that is the whole
            # point of the repair), so it must NOT be clamped to span_start;
            # leaving attn_lb=0 makes the gap query attend [0, pos) as intended.
            if not req.is_gap_recompute:
                attn_lb[req_start + span_start:req_start + region_end] = span_start

            n_blk = (region_end - span_start) // block_size
            query_start = _following_query_start(span_start, spans, crosses)
            # Virtual gap rows repair cached KV and must not update Quest scores.
            if (
                not req.is_gap_recompute
                and query_start is not None
                and 0 < quest_top_k < n_blk
                and nc <= query_start < nc + ns
            ):
                quest_score_descs.append(
                    (
                        i,
                        span_start // block_size,
                        n_blk,
                        int(q_start[i]) + query_start - nc,
                    )
                )
                quest_span_scores.append(torch.zeros(n_blk, device=device))

    return attn_lb, req_kv_starts, quest_score_descs, quest_span_scores
