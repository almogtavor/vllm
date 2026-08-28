# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QCFuse worker-side importance probe.

Measures I(t) = query-to-context attention mass at context position ``t``,
summed over the trailing user-query rows and over heads, on the critical
layers only. The result is a *selection* signal: nothing here reads or writes
a KV entry, so the model's numerics are untouched.

Fused attention backends return only ``softmax_lse`` (a per-row normalizer),
never per-key mass, so the scores are recomputed explicitly as ``Q_U @ K_C^T``
against the paged K. For ~64 query rows x 8k context x 3 layers that is
~1.6 GFLOP, negligible next to the prefill it rides along with.

CUDA graphs: the probe only ever fires on prefill-bearing steps (a descriptor
is emitted only when a request has more than one scheduled token), and this
branch already forces eager on those steps
(``gpu_model_runner._select_cudagraph_mode``: ``max_num_scheduled_tokens > 1``
=> ``force_eager``). The per-step device buffer is preallocated once, so no
allocation happens on any decode path. Everything below is additionally dead
unless ``VLLM_V1_SPANS_QCFUSE_ENABLE`` is set.

Fidelity caveat: the probe reads K *as stored*. Under SPANS the cache holds
pre-RoPE / span-rotated K, so I(t) is a monotone proxy for the true post-RoPE
attention mass, not the exact quantity. It is only ever used to rank context
positions.
"""

from __future__ import annotations

import torch

from vllm import envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Trailing rows of a request treated as "the user query" for I(t). Bounds the
# probe's cost to a constant regardless of prompt length.
QCFUSE_MAX_QUERY_TOKENS = 64

_SUPPORTED_KV_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def parse_critical_layers() -> tuple[int, ...]:
    """Layer indices the probe runs on, from the env knob."""
    raw = envs.VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS
    return tuple(int(x) for x in raw.split(",") if x.strip())


class QCFuseImportanceCapturer:
    """Accumulates per-context-token importance into a persistent device buffer.

    One instance per worker. ``begin_step`` is called by the model runner with
    this step's probe descriptors; ``capture`` is called once per critical
    attention layer from the ``qcfuse_capture_importance`` custom op.

    Descriptor: ``(row, q_start, q_end, ctx_len)`` where ``row`` is the
    input-batch / block-table row, ``[q_start, q_end)`` are flat query rows in
    this step's token batch, and ``ctx_len`` is the request's cached prefix
    length. Importance is written to ``buffer[row, :ctx_len]``.
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
        self._warned_layout = False
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
        """Accumulate one critical layer's query-to-context attention mass.

        ``query`` is the step's flat ``(num_tokens, num_heads, head_size)``
        tensor; ``kv_cache`` is the layer's paged cache.
        """
        if layer_idx not in self.critical_layers or not self.descs:
            return
        block_table = self.block_table
        if block_table is None:
            return
        if kv_cache.dtype not in _SUPPORTED_KV_DTYPES:
            self._warn_layout(f"unsupported kv_cache dtype {kv_cache.dtype}")
            return

        # Two paged layouts exist and rank alone does not separate them:
        #   (2, n_blocks, block, kv_heads, head)  K/V-first (vLLM default)
        #   (n_blocks, 2, block, kv_heads, head)  blocks-first (gemma-4)
        # Guessing wrong scores garbage silently, so key off whichever axis holds
        # the K/V pair. n_blocks == 2 is genuinely ambiguous; prefer K/V-first.
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
            # Mean-pool each GQA group onto its KV head so the score matrix is
            # (n_kv, nq, ctx) rather than (n_heads, nq, ctx).
            q = q.view(nq, n_kv, num_queries_per_kv, head_dim).mean(2)
            scores = torch.einsum("qhd,chd->hqc", q, k_ctx) * scale
            mass = scores.softmax(dim=-1).sum(dim=(0, 1))
            self.buffer[row, : n_blk * bs] += mass

    def read_importance(self, row: int, ctx_len: int) -> list[float]:
        """Copy one request's accumulated importance back to host."""
        return self.buffer[row, :ctx_len].tolist()

    def _warn_layout(self, why: str) -> None:
        # Do NOT degrade quietly. With no importance the policy returns no gaps,
        # so the arm silently becomes plain `spans` while still labelling itself
        # QCFuse -- it would publish a real-looking number for a method that
        # never ran. A hard failure is recoverable; a fake arm is not.
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
