# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QCFuse importance keyed by prefix-cache block hash.

A request's gaps are chosen before its own probe runs, so importance measured
on one request is reused by later requests over the same blocks.
"""

from collections import OrderedDict

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)

DEFAULT_MAX_BLOCKS = 1 << 20


class QCFuseImportanceStore:
    """LRU map from a prefix-cache block hash to that block's importance."""

    def __init__(self, block_size: int, max_blocks: int = DEFAULT_MAX_BLOCKS):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._by_hash: OrderedDict[BlockHash, list[float]] = OrderedDict()

    def store(self, block_hashes: list[BlockHash], importance: list[float]) -> None:
        """Store per-block importance; the newest measurement wins."""
        bs = self.block_size
        n = min(len(importance) // bs, len(block_hashes))
        for b in range(n):
            key = block_hashes[b]
            self._by_hash.pop(key, None)
            self._by_hash[key] = importance[b * bs : (b + 1) * bs]
        while len(self._by_hash) > self.max_blocks:
            self._by_hash.popitem(last=False)

    def lookup(
        self, block_hashes: list[BlockHash], num_computed_tokens: int
    ) -> list[float] | None:
        """Importance for the cached prefix; unmeasured blocks score 0, None if all are."""
        bs = self.block_size
        n_blocks = num_computed_tokens // bs
        if n_blocks <= 0:
            return None
        out: list[float] = []
        hits = 0
        for b in range(n_blocks):
            entry = (
                self._by_hash.get(block_hashes[b]) if b < len(block_hashes) else None
            )
            if entry is None:
                out.extend([0.0] * bs)
            else:
                self._by_hash.move_to_end(block_hashes[b])
                out.extend(entry)
                hits += 1
        if hits == 0:
            return None
        logger.debug(
            "QCFuse store: %d/%d prefix blocks have measured importance",
            hits,
            n_blocks,
        )
        return out
