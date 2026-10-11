# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QCFuse importance probe.

Recomputes ``softmax(Q_query @ K_ctx^T)`` on the critical layers against the
paged K (fused backends expose no per-key mass) and sums it per context token.
"""

from __future__ import annotations

import torch

from vllm import envs

# trailing rows treated as the query, bounding the probe's cost
QCFUSE_MAX_QUERY_TOKENS = 64


def parse_critical_layers() -> tuple[int, ...]:
    raw = envs.VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS
    return tuple(int(x) for x in raw.split(",") if x.strip())


def probe_enabled() -> bool:
    return envs.VLLM_V1_SPANS_QCFUSE_ENABLE or envs.VLLM_V1_SPANS_NEIGHBOR_AWARE_ENABLE


class QCFuseImportanceCapturer:
    """Per-worker importance buffer, filled once per critical layer.

    Descriptor ``(row, q_start, q_end, ctx_len)``: batch row, flat query rows,
    and cached prefix length; results go to ``buffer[row, :ctx_len]``.
    """

    def __init__(self, max_num_reqs: int, max_model_len: int, device) -> None:
        self.buffer = torch.zeros(
            (max_num_reqs, max_model_len), dtype=torch.float32, device=device
        )
        self.block_table: torch.Tensor | None = None
        self.descs: list[tuple[int, int, int, int]] = []

    def begin_step(self, block_table: torch.Tensor | None, descs: list) -> None:
        self.block_table = block_table
        self.descs = descs
        for row, _, _, ctx_len in descs:
            self.buffer[row, :ctx_len].zero_()

    def end_step(self) -> None:
        self.descs = []
        self.block_table = None

    def capture(self, query, kv_cache, num_queries_per_kv: int, scale: float) -> None:
        block_table = self.block_table
        if not self.descs or block_table is None:
            return
        # K/V-first (2, n_blocks, ...) or blocks-first (n_blocks, 2, ...)
        if (
            kv_cache.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or kv_cache.dim() != 5
            or 2 not in kv_cache.shape[:2]
        ):
            # fail loudly: without importance the arm is plain spans
            raise RuntimeError(
                "QCFuse probe cannot read this KV cache "
                f"({kv_cache.dtype}, {tuple(kv_cache.shape)})"
            )
        key_cache = kv_cache[0] if kv_cache.shape[0] == 2 else kv_cache[:, 0]
        bs = key_cache.shape[1]
        for row, q_start, q_end, ctx_len in self.descs:
            n_blk = min(ctx_len // bs, block_table.shape[1])
            if n_blk <= 0 or q_end <= q_start:
                continue
            blk_ids = block_table[row, :n_blk].long()
            # (n_blk, bs, n_kv_heads, head) -> (ctx, n_kv_heads, head)
            k_ctx = key_cache[blk_ids].flatten(0, 1).float()
            q = query[q_start:q_end].float()
            nq, n_heads, head_dim = q.shape
            n_kv = n_heads // num_queries_per_kv
            # mean-pool each GQA group onto its KV head
            q = q.view(nq, n_kv, num_queries_per_kv, head_dim).mean(2)
            scores = torch.einsum("qhd,chd->hqc", q, k_ctx) * scale
            mass = scores.softmax(dim=-1).sum(dim=(0, 1))
            self.buffer[row, : n_blk * bs] += mass


_CAPTURER: QCFuseImportanceCapturer | None = None


def bind_capturer(capturer: QCFuseImportanceCapturer | None) -> None:
    global _CAPTURER
    _CAPTURER = capturer


def get_capturer() -> QCFuseImportanceCapturer | None:
    return _CAPTURER
