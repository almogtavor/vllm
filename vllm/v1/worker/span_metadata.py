# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

import numpy as np
import torch

from vllm.v1.worker.gpu_input_batch import CachedRequestState


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
        crosses = extra_args.get("cross_span_starts") or []
        req_start = int(req_kv_starts[i])
        req_len = int(seq_lens_arr[i])
        nc = int(num_computed[i])
        ns = int(num_scheduled_tokens[i])
        for j, span_start in enumerate(spans):
            if span_start >= req_len:
                continue
            cross = crosses[j] if j < len(crosses) else req_len
            cross = min(cross, req_len)
            if cross <= span_start:
                continue
            attn_lb[req_start + span_start:req_start + cross] = span_start

            n_blk = (cross - span_start) // block_size
            # Virtual gap rows repair cached KV and must not update Quest scores.
            if (
                not req.is_gap_recompute
                and 0 < quest_top_k < n_blk
                and nc <= cross < nc + ns
            ):
                quest_score_descs.append(
                    (
                        i,
                        span_start // block_size,
                        n_blk,
                        int(q_start[i]) + cross - nc,
                    )
                )
                quest_span_scores.append(torch.zeros(n_blk, device=device))

    return attn_lb, req_kv_starts, quest_score_descs, quest_span_scores
