# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Content-keyed store for QCFuse importance.

``get_gaps`` is called once per request, from the waiting queue, *before* any
forward pass for that request has run (``did_prefix_lookup`` is
``request.num_computed_tokens == 0``). So a request can never consume its own
probe. The probe's output is only useful to a *later* request that reuses the
same cached prefix, which is exactly what this store keys on: per-block
importance under that block's prefix-cache hash, mirroring how LegoQuest keyed
its span-block selections.

Nothing here changes ``QCFusePolicy.get_gaps``' contract - the scheduler
populates ``request.qcfuse_importance`` from this store and the policy reads
that field as before.
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
        """Split one request's importance vector into per-block entries.

        Later measurements overwrite earlier ones: attention mass is measured
        against whatever query followed the block, and the freshest query is
        the better predictor of the next one.
        """
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
        """Reassemble an importance vector for a request's cached prefix.

        Returns ``None`` when no block of the prefix has ever been measured, so
        the policy falls back to "probe first, no gaps". Unmeasured blocks
        inside a partially-known prefix score zero and are simply never picked.
        """
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
