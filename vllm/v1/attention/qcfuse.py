# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QCFuse importance probe.

Recomputes ``softmax(Q_query @ K_ctx^T)`` on the critical layers against the
paged K (fused backends expose no per-key mass) and sums it per context token.
"""

from __future__ import annotations

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# trailing rows treated as the query, bounding the probe's cost
QCFUSE_MAX_QUERY_TOKENS = 64


def parse_critical_layers() -> tuple[int, ...]:
    raw = envs.VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS
    return tuple(int(x) for x in raw.split(",") if x.strip())


def probe_enabled() -> bool:
    return (
        envs.VLLM_V1_SPANS_QCFUSE_ENABLE
        or envs.VLLM_V1_SPANS_NEIGHBOR_AWARE_ENABLE
        or envs.VLLM_V1_SPANS_SPANFUSE_ENABLE
    )


class QCFuseImportanceCapturer:
    """Per-worker importance buffer, filled once per critical layer.

    Descriptor ``(row, q_start, q_end, ctx_len, span_blocks, q_pos0)``: batch
    row, flat query rows, scored prefix length, (SpanFuse) the span block
    indices the softmax runs over or None for the whole prefix, and the
    absolute position of row ``q_start`` (keys after a row are masked).
    """

    def __init__(self, max_num_reqs: int, max_model_len: int, device) -> None:
        self.buffer = torch.zeros(
            (max_num_reqs, max_model_len), dtype=torch.float32, device=device
        )
        self.block_table: torch.Tensor | None = None
        self.descs: list[tuple] = []

    def begin_step(self, block_table: torch.Tensor | None, descs: list) -> None:
        self.block_table = block_table
        self.descs = descs
        for row, _, _, ctx_len, _, _ in descs:
            self.buffer[row, :ctx_len].zero_()

    def end_step(self) -> None:
        self.descs = []
        self.block_table = None

    def capture(
        self,
        query,
        kv_cache,
        num_queries_per_kv: int,
        scale: float,
        cos_sin_cache: torch.Tensor | None = None,
        rotary_dim: int = 0,
    ) -> None:
        block_table = self.block_table
        if not self.descs or block_table is None:
            return
        if envs.VLLM_V1_SPANS_ENABLED and cos_sin_cache is None:
            # SPANS caches K unrotated: scoring it against rotated Q is wrong
            raise RuntimeError("QCFuse probe under SPANS needs the layer's RoPE cache")
        rope = (cos_sin_cache, rotary_dim) if cos_sin_cache is not None else None
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
        for row, q_start, q_end, ctx_len, span_blocks, q_pos0 in self.descs:
            n_blk = min(ctx_len // bs, block_table.shape[1])
            if n_blk <= 0 or q_end <= q_start:
                continue
            q = query[q_start:q_end].float()
            nq, n_heads, head_dim = q.shape
            n_kv = n_heads // num_queries_per_kv
            # mean-pool each GQA group onto its KV head
            q = q.view(nq, n_kv, num_queries_per_kv, head_dim).mean(2)
            q_pos = q_pos0 + torch.arange(nq, device=q.device)
            if span_blocks is not None:
                # SPANFUSE: normalize over the span blocks only, not the prefix
                idx = torch.tensor(
                    [b for b in span_blocks if b < n_blk],
                    dtype=torch.long,
                    device=block_table.device,
                )
                if idx.numel() == 0:
                    continue
                pos = (
                    idx[:, None] * bs + torch.arange(bs, device=idx.device)
                ).flatten()
                k_ctx = key_cache[block_table[row, idx].long()].flatten(0, 1).float()
                mass = self._mass(q, self._rope(k_ctx, pos, rope), scale, q_pos, pos)
                self.buffer[row].index_add_(0, pos.to(self.buffer.device), mass)
                if envs.VLLM_V1_SPANS_SPANFUSE_NORM_CHECK:
                    self._log_norm_check(
                        q,
                        q_pos,
                        key_cache,
                        block_table[row, :n_blk],
                        pos,
                        mass,
                        scale,
                        rope,
                    )
                continue
            blk_ids = block_table[row, :n_blk].long()
            # (n_blk, bs, n_kv_heads, head) -> (ctx, n_kv_heads, head)
            k_ctx = key_cache[blk_ids].flatten(0, 1).float()
            pos = torch.arange(n_blk * bs, device=k_ctx.device)
            self.buffer[row, : n_blk * bs] += self._mass(
                q, self._rope(k_ctx, pos, rope), scale, q_pos, pos
            )

    @staticmethod
    def _rope(k, pos, rope):
        """Neox-rotate K to its absolute position, as the span kernel does for
        the tail (lb=0) rows the probe scores from (_load_rotate_k)."""
        if rope is None:
            return k
        cos_sin_cache, rotary_dim = rope
        d = k.shape[-1]
        rot = d if rotary_dim <= 0 or rotary_dim > d else rotary_dim
        half = rot // 2
        cs = cos_sin_cache[pos.to(cos_sin_cache.device)].float()[:, None, :]
        cos, sin = cs[..., :half], cs[..., half:rot]
        x1, x2 = k[..., :half], k[..., half:rot]
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, k[..., rot:]), -1)

    @staticmethod
    def _mass(q, k_ctx, scale: float, q_pos=None, k_pos=None) -> torch.Tensor:
        scores = torch.einsum("qhd,chd->hqc", q, k_ctx) * scale
        if q_pos is not None:
            causal = k_pos.to(q_pos.device)[None, :] > q_pos[:, None]
            scores = scores.masked_fill(causal[None], float("-inf"))
        return scores.softmax(dim=-1).nan_to_num(0.0).sum(dim=(0, 1))

    def _log_norm_check(
        self, q, q_pos, key_cache, blk_row, pos, mass, scale, rope
    ) -> None:
        k_all = key_cache[blk_row.long()].flatten(0, 1).float()
        ar = torch.arange(k_all.shape[0], device=k_all.device)
        full = self._mass(q, self._rope(k_all, ar, rope), scale, q_pos, ar)[pos]
        n = max(1, pos.numel() // 10)
        a = set(mass.topk(n).indices.tolist())
        b = set(full.topk(n).indices.tolist())
        logger.info("SpanFuse norm check: top-%d overlap %d/%d", n, len(a & b), n)


_CAPTURER: QCFuseImportanceCapturer | None = None


def bind_capturer(capturer: QCFuseImportanceCapturer | None) -> None:
    global _CAPTURER
    _CAPTURER = capturer


def get_capturer() -> QCFuseImportanceCapturer | None:
    return _CAPTURER
