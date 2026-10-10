# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QCFuse importance probe.

Recomputes ``softmax(Q_query @ K_ctx^T)`` on the critical layers against the
paged K (fused backends expose no per-key mass) and sums it per context token.
Runs only on prefill steps, which are eager; used only to rank tokens.
"""

from __future__ import annotations

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# trailing rows treated as the query, bounding the probe's cost
QCFUSE_MAX_QUERY_TOKENS = 64

_SUPPORTED_KV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def parse_critical_layers() -> tuple[int, ...]:
    """Layer indices the probe runs on, from the env knob."""
    raw = envs.VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS
    return tuple(int(x) for x in raw.split(",") if x.strip())


def probe_enabled() -> bool:
    """True when QCFuse or neighbor-aware needs the importance probe."""
    return envs.VLLM_V1_SPANS_QCFUSE_ENABLE or envs.VLLM_V1_SPANS_NEIGHBOR_AWARE_ENABLE


class QCFuseImportanceCapturer:
    """Per-worker importance buffer, filled once per critical layer.

    Descriptor ``(row, q_start, q_end, ctx_len)``: batch row, flat query rows,
    and cached prefix length; results go to ``buffer[row, :ctx_len]``.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        block_size: int,
        device: torch.device,
    ) -> None:
        self.block_size = block_size
        self.critical_layers = set(parse_critical_layers())
        self.buffer = torch.zeros(
            (max_num_reqs, max_model_len), dtype=torch.float32, device=device
        )
        self.block_table: torch.Tensor | None = None
        self.descs: list[tuple[int, int, int, int]] = []
        logger.info(
            "QCFuseImportanceCapturer: buffer %.1f MB (reqs=%d, len=%d), "
            "critical_layers=%s",
            self.buffer.numel() * 4 / 1e6,
            max_num_reqs,
            max_model_len,
            sorted(self.critical_layers),
        )

    def begin_step(
        self,
        block_table: torch.Tensor | None,
        descs: list[tuple[int, int, int, int]],
    ) -> None:
        """Install this step's descriptors and zero only the rows they touch."""
        self.block_table = block_table
        self.descs = descs
        for row, _, _, ctx_len in descs:
            self.buffer[row, :ctx_len].zero_()

    def end_step(self) -> None:
        self.descs = []
        self.block_table = None

    def capture(
        self,
        layer_idx: int,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        num_queries_per_kv: int,
        scale: float,
    ) -> None:
        """Accumulate one critical layer's query-to-context attention mass."""
        if layer_idx not in self.critical_layers or not self.descs:
            return
        block_table = self.block_table
        if block_table is None:
            return
        if kv_cache.dtype not in _SUPPORTED_KV_DTYPES:
            self._warn_layout(f"unsupported kv_cache dtype {kv_cache.dtype}")
            return

        # K/V-first (2, n_blocks, ...) or blocks-first (n_blocks, 2, ...)
        if kv_cache.dim() != 5:
            self._warn_layout(f"unexpected kv_cache rank {tuple(kv_cache.shape)}")
            return
        if kv_cache.shape[0] == 2:
            key_cache = kv_cache[0]
        elif kv_cache.shape[1] == 2:
            key_cache = kv_cache[:, 0]
        else:
            self._warn_layout(f"unexpected kv_cache shape {tuple(kv_cache.shape)}")
            return
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

    def read_importance(self, row: int, ctx_len: int) -> list[float]:
        """Copy one request's accumulated importance back to host."""
        return self.buffer[row, :ctx_len].tolist()

    def _warn_layout(self, why: str) -> None:
        # fail loudly: without importance the arm is plain spans
        raise RuntimeError(
            f"QCFuse probe cannot read this model's KV cache: {why}. "
            "Refusing to run: without the probe this arm silently degrades to "
            "plain spans with zero recompute."
        )


_CAPTURER: QCFuseImportanceCapturer | None = None


def bind_capturer(capturer: QCFuseImportanceCapturer | None) -> None:
    """Publish the worker's capturer to the attention custom op."""
    global _CAPTURER
    _CAPTURER = capturer


def get_capturer() -> QCFuseImportanceCapturer | None:
    return _CAPTURER
