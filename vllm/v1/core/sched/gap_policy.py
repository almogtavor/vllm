# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gap Policy for KV Cache Recomputation

This module provides abstractions for deciding where to insert recomputation gaps
within prefix-cached tokens. Gap policies are independent of where cached tokens
came from (local prefix cache, external connector, or both).
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, PrefixHitSource

if TYPE_CHECKING:
    from vllm.v1.request import Request

logger = init_logger(__name__)


class GapPolicy(ABC):
    """
    Decides where to insert recomputation gaps within prefix-cached tokens.

    Gap policies are independent of where cached tokens came from (local prefix
    cache, external connector, or both). They operate on the unified view of
    all computed tokens.
    """

    @abstractmethod
    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        """
        Return gap intervals within [0, num_computed_tokens) to recompute.

        Args:
            request: The request object containing prompt tokens and metadata
            num_computed_tokens: Total cached tokens (local + external)
            num_external_tokens: Number of tokens from external connector

        Returns:
            List of (start, end) tuples representing half-open intervals [start, end)
            that should be recomputed. Intervals must be:
            - Within bounds: 0 <= start < end <= num_computed_tokens
            - Non-overlapping and strictly increasing
            - Empty list means no gaps (use all cached tokens)
        """
        pass


class NoGapPolicy(GapPolicy):
    """Default policy: no gaps, use all cached tokens."""

    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        """Return empty list - no gaps."""
        return []


class SpanAwareGapPolicy(GapPolicy):
    """
    Creates gaps at span boundaries specified via per-request metadata.

    Reads span start positions from request.span_starts (set via
    SamplingParams.extra_args) and creates gaps of configurable length.
    """

    DEFAULT_GAP_LENGTH = 32

    def __init__(
        self,
        gap_length: int = DEFAULT_GAP_LENGTH,
        block_size: int = 16,
    ):
        self.gap_length = gap_length
        self.block_size = block_size

        logger.info(
            "SpanAwareGapPolicy initialized: gap_length=%d",
            gap_length,
        )

    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        if self.gap_length <= 0 or num_computed_tokens == 0:
            return []

        span_starts = request.span_starts
        if not span_starts:
            return []

        span_starts = [s for s in span_starts if s < num_computed_tokens]
        if not span_starts:
            return []

        logger.debug(
            "Found %d span starts within computed range: %s",
            len(span_starts),
            span_starts,
        )

        gaps = []
        for idx, gap_start in enumerate(span_starts):
            next_start = (
                span_starts[idx + 1]
                if idx + 1 < len(span_starts)
                else num_computed_tokens
            )
            end_lim = min(next_start, num_computed_tokens)
            gaps.extend(self._span_gap_ranges(request, gap_start, end_lim))

        # SPANS: drop gaps whose blocks all hit a prefix-aware (pd) copy from an
        # earlier recompute of this prefix - recompute-once-per-unique-prefix.
        sources = request.prefix_hit_sources
        if sources is not None:
            bs = self.block_size
            kept = []
            for s, e in gaps:
                blocks = range(s // bs, min(e // bs, len(sources)))
                if blocks and all(sources[b] == PrefixHitSource.PD for b in blocks):
                    continue
                kept.append((s, e))
            gaps = kept

        logger.info(
            "Created %d gaps for request %s: %s", len(gaps), request.request_id, gaps
        )

        self._print_gaps_representation(gaps, num_external_tokens, num_computed_tokens)

        return gaps

    def _span_gap_ranges(
        self, request: "Request", span_start: int, end_lim: int
    ) -> list[tuple[int, int]]:
        gap_end = min(span_start + self.gap_length, end_lim)
        return [(span_start, gap_end)] if gap_end > span_start else []

    def _print_gaps_representation(
        self,
        gaps: list[tuple[int, int]],
        num_external_tokens: int,
        num_computed_tokens: int,
    ) -> None:
        total_tokens = num_computed_tokens
        block_size = self.block_size
        representation = []

        num_local_tokens = num_computed_tokens - num_external_tokens

        for block_start in range(0, total_tokens, block_size):
            block_end = min(block_start + block_size, total_tokens)
            block_chars = []

            for i in range(block_start, block_end):
                in_gap = any(start <= i < end for start, end in gaps)

                if in_gap:
                    block_chars.append("-")
                elif i < num_local_tokens:
                    block_chars.append("L")
                else:
                    block_chars.append("E")

            unique_chars = set(block_chars)
            char = unique_chars.pop() if len(unique_chars) == 1 else "X"
            representation.append(char)

        logger.debug("Cache status per block (L=local, E=external, -=gap, X=mixed):")
        logger.debug("".join(representation))
        logger.debug("Gaps: %s", gaps)
        logger.debug(
            "Total tokens: %d (local: %d, external: %d)",
            total_tokens,
            num_local_tokens,
            num_external_tokens,
        )


class QuestGapPolicy(SpanAwareGapPolicy):
    """Recompute span blocks selected by the following query's Quest score.

    Scores are measured worker-side at the span's first occurrence against its
    first post-span query and stored here keyed by both the span's first pic
    block hash and first following query tokens. A span with no stored
    selection falls back to the contiguous span_aware gap.
    """

    MAX_SELECTIONS = 4096
    DEFAULT_ANCHOR_BLOCKS = 8

    def __init__(
        self,
        *args,
        anchor_blocks: int = DEFAULT_ANCHOR_BLOCKS,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.anchor_blocks = max(anchor_blocks, 0)
        self.selections: dict[tuple[BlockHash, tuple[int, ...]], list[int]] = {}
        logger.info("QuestGapPolicy initialized: anchor_blocks=%d", anchor_blocks)

    def get_selection_key(
        self,
        request: "Request",
        span_start: int,
    ) -> tuple[BlockHash, tuple[int, ...]] | None:
        bs = self.block_size
        blk = span_start // bs
        if blk >= len(request.block_hashes):
            return None

        cross = self._following_query_start(request, span_start)
        if cross is None or cross >= request.num_tokens:
            return None

        query_tokens = tuple(request.all_token_ids[cross : cross + bs])
        if not query_tokens:
            return None
        return request.block_hashes[blk], query_tokens

    def _following_query_start(
        self,
        request: "Request",
        span_start: int,
    ) -> int | None:
        spans = request.span_starts or []
        crosses = request.cross_span_starts or []
        for idx, start in enumerate(spans):
            if start == span_start and idx < len(crosses):
                cross = crosses[idx]
                return cross if cross > span_start else None
        return next((cross for cross in sorted(crosses) if cross > span_start), None)

    def store_selection(
        self,
        key: tuple[BlockHash, tuple[int, ...]],
        block_offsets: list[int],
    ) -> None:
        # First writer wins: re-scores of a reused span read blocks that are
        # being gap-recomputed in the same step, so only the first-occurrence
        # scores (prefix-free warmed K) are trustworthy.
        if key in self.selections:
            return
        self.selections[key] = block_offsets
        if len(self.selections) > self.MAX_SELECTIONS:
            self.selections.pop(next(iter(self.selections)))

    def _span_gap_ranges(
        self, request: "Request", span_start: int, end_lim: int
    ) -> list[tuple[int, int]]:
        bs = self.block_size
        budget = self.gap_length // bs
        if budget <= 0:
            return []

        key = self.get_selection_key(request, span_start)
        offsets = self.selections.get(key) if key is not None else None
        if not offsets:
            return super()._span_gap_ranges(request, span_start, end_lim)

        n_blocks = (end_lim - span_start + bs - 1) // bs
        if n_blocks <= 0:
            return []

        selected_offsets = []
        anchor_count = min(self.anchor_blocks, budget, n_blocks)
        for o in range(anchor_count):
            selected_offsets.append(o)

        remaining = min(budget - len(selected_offsets), n_blocks - anchor_count)
        if remaining > 0:
            target = next(
                (o for o in offsets if anchor_count <= o < n_blocks),
                anchor_count,
            )
            window_start = max(anchor_count, target - remaining + 1)
            window_end = min(window_start + remaining, n_blocks)
            if window_end - window_start < remaining:
                window_start = max(anchor_count, n_blocks - remaining)
                window_end = n_blocks
            selected_offsets.extend(range(window_start, window_end))

        gaps = []
        for o in sorted(selected_offsets):
            s = span_start + o * bs
            e = min(s + bs, end_lim)
            if e <= s:
                continue
            if gaps and gaps[-1][1] == s:  # coalesce adjacent selected blocks
                gaps[-1] = (gaps[-1][0], e)
            else:
                gaps.append((s, e))
        return gaps


class GapPolicyFactory:
    """Factory for creating GapPolicy instances from configuration."""

    _POLICIES = {
        "none": NoGapPolicy,
        "span_quest": QuestGapPolicy,
        "span_aware": SpanAwareGapPolicy,
    }

    @classmethod
    def create_policy(
        cls,
        policy_name: str | None = None,
        policy_config: dict | None = None,
    ) -> GapPolicy | None:
        """
        Create a GapPolicy instance from configuration.

        Args:
            policy_name: Name of the policy ("none", "span_aware", or None)
            policy_config: Configuration dict for the policy

        Returns:
            GapPolicy instance or None if policy_name is None
        """
        if policy_name is None:
            return None

        policy_name_lower = policy_name.lower()
        if policy_name_lower not in cls._POLICIES:
            logger.warning(
                "Unknown gap policy '%s'. Available: %s. Using NoGapPolicy.",
                policy_name,
                list(cls._POLICIES.keys()),
            )
            policy_name_lower = "none"

        policy_class = cls._POLICIES[policy_name_lower]
        policy_config = policy_config or {}

        try:
            return policy_class(**policy_config)
        except TypeError as e:
            logger.error(
                "Failed to create %s policy with config %s: %s. Using NoGapPolicy.",
                policy_name,
                policy_config,
                e,
            )
            return NoGapPolicy()

    @classmethod
    def register_policy(cls, name: str, policy_class: type[GapPolicy]) -> None:
        """
        Register a custom gap policy.

        Args:
            name: Name to register the policy under
            policy_class: GapPolicy subclass to register
        """
        if not issubclass(policy_class, GapPolicy):
            raise ValueError(f"{policy_class} must be a subclass of GapPolicy")

        cls._POLICIES[name.lower()] = policy_class
        logger.info("Registered gap policy: %s -> %s", name, policy_class.__name__)
