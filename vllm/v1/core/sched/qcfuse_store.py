# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""QCFuse importance keyed by block hash: gaps are chosen before a request's
own probe runs, so later requests reuse importance measured on earlier ones."""

from collections import OrderedDict

from vllm.v1.core.kv_cache_utils import BlockHash


class QCFuseImportanceStore:
    """LRU map from a prefix-cache block hash to that block's importance."""

    def __init__(self, block_size: int, max_blocks: int = 1 << 20):
        self.block_size = block_size
        self.max_blocks = max_blocks
        self._by_hash: OrderedDict[BlockHash, list[float]] = OrderedDict()

    def store(self, block_hashes: list[BlockHash], importance: list[float]) -> None:
        bs = self.block_size
        for b in range(min(len(importance) // bs, len(block_hashes))):
            self._by_hash.pop(block_hashes[b], None)
            self._by_hash[block_hashes[b]] = importance[b * bs : (b + 1) * bs]
        while len(self._by_hash) > self.max_blocks:
            self._by_hash.popitem(last=False)

    def lookup(
        self, block_hashes: list[BlockHash], num_computed_tokens: int
    ) -> list[float] | None:
        """Importance for the cached prefix; unmeasured blocks score 0, None if all are."""
        bs = self.block_size
        out: list[float] = []
        hit = False
        for b in range(num_computed_tokens // bs):
            h = block_hashes[b] if b < len(block_hashes) else None
            entry = self._by_hash.get(h)
            if entry is None:
                out.extend([0.0] * bs)
            else:
                self._by_hash.move_to_end(h)
                out.extend(entry)
                hit = True
        return out if hit else None
