# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gap Policy for KV Cache Recomputation

This module provides abstractions for deciding where to insert recomputation gaps
within prefix-cached tokens. Gap policies are independent of where cached tokens
came from (local prefix cache, external connector, or both).
"""

from abc import ABC, abstractmethod
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import PrefixHitSource
from vllm.v1.core.sched.output import NewRequestData

if TYPE_CHECKING:
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import Request

logger = init_logger(__name__)

# Sentinel: request has no span gaps, so the caller schedules it normally.
NO_SPAN_GAPS = object()


def split_gaps_for_budget(
    gaps: list[tuple[int, int]], token_budget: int, block_size: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Split gaps into a block-aligned chunk that fits token_budget and the rest."""
    block_budget = max(0, token_budget) // block_size * block_size
    if block_budget <= 0:
        return [], gaps
    scheduled: list[tuple[int, int]] = []
    remaining: list[tuple[int, int]] = []
    budget = block_budget
    for start, end in gaps:
        take = min(end - start, budget)
        if take < end - start:
            take = take // block_size * block_size
        if take <= 0:
            remaining.append((start, end))
            continue
        scheduled.append((start, start + take))
        budget -= take
        if start + take < end:
            remaining.append((start + take, end))
    return scheduled, remaining


def miss_gaps(request: "Request", block_size: int) -> list[tuple[int, int]]:
    """Forced gaps over the dual lookup's MISS placeholders, whatever the policy."""
    sources = request.prefix_hit_sources or []
    return [
        (i * block_size, (i + 1) * block_size)
        for i, src in enumerate(sources)
        if src == PrefixHitSource.MISS
    ]


def append_virtual_gap_reqs(
    parent_nrd: NewRequestData,
    gaps: list[tuple[int, int]],
    out: list[NewRequestData],
    virtual_gap_req_ids: set[str],
    num_scheduled_tokens: dict[str, int],
) -> None:
    """Emit one virtual gap-recompute request per gap, sharing the parent's blocks."""
    logger.info(
        "Processing computed_token_gaps for request %s: %s", parent_nrd.req_id, gaps
    )
    for start, end in gaps:
        nrd = replace(parent_nrd)
        nrd.req_id = parent_nrd.req_id + "." + str(start)
        # Virtual gap requests share the parent's blocks and write directly
        # to the gap slots in the parent's KV cache.
        nrd.num_computed_tokens = start
        nrd.is_gap_recompute = True
        nrd.parent_req_id = parent_nrd.req_id
        nrd.gap_start = start
        nrd.block_ids = parent_nrd.block_ids
        num_scheduled_tokens[nrd.req_id] = end - start
        out.append(nrd)
        virtual_gap_req_ids.add(nrd.req_id)


def schedule_span_gaps(
    sched: "Scheduler",
    request: "Request",
    did_prefix_lookup: bool,
    num_computed_tokens: int,
    num_external_computed_tokens: int,
    num_new_local_computed_tokens: int,
    new_computed_blocks,
    token_budget: int,
    num_scheduled_tokens: dict[str, int],
    virtual_reqs_out: list[NewRequestData],
    virtual_gap_req_ids: set[str],
    request_queue,
    step_skipped_waiting,
):
    """Reserve gap-recompute work for `request` and defer the parent one step.

    Gaps come from the request's pending remainder, the gap policy, and the
    connector. Returns NO_SPAN_GAPS when there is nothing to recompute (the
    caller schedules the request normally), None when there are gaps but no
    room this step (the caller breaks), or the updated token_budget when the
    parent was deferred (the caller sets token_budget and continues).
    """
    request_id = request.request_id
    if request.pending_span_gaps:
        span_gaps = request.pending_span_gaps
    elif did_prefix_lookup and sched.gap_policy is not None:
        # memoize per (request, prefix length): the probe rewrites importance between retries
        memo = request.span_gaps_selection
        if memo is not None and memo[0] == num_computed_tokens:
            span_gaps = list(memo[1])
        else:
            span_gaps = sched.gap_policy.get_gaps(
                request, num_computed_tokens, num_external_computed_tokens
            )
            request.span_gaps_selection = (num_computed_tokens, list(span_gaps))
    else:
        span_gaps = []
    if did_prefix_lookup and not request.pending_span_gaps:
        span_gaps = span_gaps + miss_gaps(request, sched.block_size)
    if did_prefix_lookup and sched.connector is not None:
        connector_gaps = sched.connector.get_computed_token_gaps(request)
        if connector_gaps:
            logger.info(
                "Connector %s returned gaps via get_computed_token_gaps(). "
                "Consider migrating to use GapPolicy at scheduler level.",
                type(sched.connector).__name__,
            )
            span_gaps.extend(connector_gaps)
    if not span_gaps:
        return NO_SPAN_GAPS

    span_gaps = sched._merge_gaps(span_gaps)
    gap_work, remaining_gaps = split_gaps_for_budget(
        span_gaps, token_budget, sched.block_size
    )
    gap_overhead = sum(end - start for start, end in gap_work)
    if gap_overhead <= 0:
        return None

    gap_blocks = sched.kv_cache_manager.allocate_slots(
        request,
        0,
        num_new_computed_tokens=num_new_local_computed_tokens,
        new_computed_blocks=new_computed_blocks,
        num_external_computed_tokens=num_external_computed_tokens,
        span_gaps=gap_work,
    )
    if gap_blocks is None:
        if request.has_encoder_inputs:
            sched.encoder_cache_manager.free(request)
        return None

    if sched.connector is not None:
        sched.connector.update_state_after_alloc(
            request,
            sched.kv_cache_manager.get_blocks(request_id),
            num_external_computed_tokens,
        )

    request.num_computed_tokens = num_computed_tokens
    request.pending_span_gaps = remaining_gaps
    parent_nrd = NewRequestData.from_request(
        request, sched.kv_cache_manager.get_blocks(request_id).get_block_ids()
    )
    append_virtual_gap_reqs(
        parent_nrd,
        gap_work,
        virtual_reqs_out,
        virtual_gap_req_ids,
        num_scheduled_tokens,
    )
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    return token_budget - gap_overhead


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
            gap_end = min(
                gap_start + self.gap_length,
                next_start,
                num_computed_tokens,
            )
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end))

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


def _drop_pd_gaps(
    request: "Request", gaps: list[tuple[int, int]], bs: int
) -> list[tuple[int, int]]:
    """Drop gaps whose blocks all hit a pd copy (recompute once per prefix)."""
    sources = request.prefix_hit_sources
    if sources is None:
        return gaps
    return [
        (s, e)
        for s, e in gaps
        if not (
            (blocks := range(s // bs, min(e // bs, len(sources))))
            and all(sources[b] == PrefixHitSource.PD for b in blocks)
        )
    ]


class QCFusePolicy(GapPolicy):
    """Recompute the cached tokens with the most query attention.

    Budget is ``k_per_span`` tokens per span, else ``rho`` of the cached tokens.
    """

    def __init__(
        self,
        rho: float = 0.1,
        block_size: int = 16,
        granularity: str = "block",
        k_per_span: int = 0,
    ):
        from vllm.v1.attention.qcfuse import parse_critical_layers

        # ValueError, since create_policy() swallows TypeError into NoGapPolicy
        if not parse_critical_layers():
            raise ValueError("QCFuse needs VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS")
        if not 0.0 < rho <= 1.0 or granularity not in ("block", "token"):
            raise ValueError(f"QCFuse: bad rho={rho} or granularity={granularity}")
        if k_per_span < 0:
            raise ValueError(f"QCFuse: k_per_span must be >= 0, got {k_per_span}")
        # without prerotate the span head carries a positional error QCFuse ignores
        if not envs.VLLM_V1_SPANS_PREROTATE:
            raise ValueError("QCFuse requires VLLM_V1_SPANS_PREROTATE=True")
        self.rho = rho
        self.block_size = block_size
        self.granularity = granularity
        self.k_per_span = k_per_span

    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        importance = getattr(request, "qcfuse_importance", None)
        if num_computed_tokens == 0 or importance is None:
            return []
        gaps = self._select(request, importance, num_computed_tokens)
        gaps = _drop_pd_gaps(request, gaps, self.block_size)
        logger.info(
            "%s: %d gaps for %s", type(self).__name__, len(gaps), request.request_id
        )
        return gaps

    def _select(
        self, request: "Request", importance: list[float], n: int
    ) -> list[tuple[int, int]]:
        # same total budget as legolink-K: K tokens per span
        if self.k_per_span > 0:
            n_spans = sum(1 for s in request.span_starts or [] if s < n) or 1
            budget = min(self.k_per_span * n_spans, n)
        else:
            budget = int(self.rho * n)
        if budget <= 0:
            return []

        bs = self.block_size
        if self.granularity == "token":
            ranked = sorted(
                range(min(len(importance), n)),
                key=lambda t: importance[t],
                reverse=True,
            )[:budget]
            return [(t, t + 1) for t in sorted(ranked)]
        scores = sorted(
            ((sum(importance[b * bs : (b + 1) * bs]), b) for b in range(n // bs)),
            reverse=True,
        )
        keep = sorted(b for _, b in scores[: max(1, budget // bs)])
        return [(b * bs, (b + 1) * bs) for b in keep]


class NeighborAwarePolicy(QCFusePolicy):
    """Greedy span-block selection by attention x staleness x closure."""

    # kernel constants measured on Qwen3-32B; only the sink matters
    c = 0.1
    alpha = 0.33
    beta = 1.25
    amp = 0.06
    floor = 0.0085
    sink = 0.554
    anchor_blocks = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.k_per_span <= 0:
            raise ValueError("NeighborAware needs VLLM_V1_SPANS_QCFUSE_K_PER_SPAN")

    def _closure_weights(self, n: int) -> tuple[list[float], list[float]]:
        """Kernel weight by block distance, and each row's total incl. the sink."""
        w = [0.0] + [self.amp * d**-self.beta + self.floor for d in range(1, n)]
        rowtot = [0.0] * n
        acc = 0.0
        for b in range(1, n):
            acc += w[b]
            rowtot[b] = acc + self.sink
        return w, rowtot

    def _select_blocks(self, attn: list[float], budget_blocks: int) -> list[int]:
        """Greedy gain(b|R) over one span's blocks; returns local indices."""
        n = len(attn)
        k = min(budget_blocks, n)
        if k <= 0:
            return []
        w, rowtot = self._closure_weights(n)
        val = [attn[b] * (1.0 + b) ** -self.alpha for b in range(n)]
        got = [0.0] * n
        chosen: set[int] = set()

        def add(j: int) -> None:
            chosen.add(j)
            for b in range(j + 1, n):
                got[b] += w[b - j] + (self.sink if j == 0 else 0.0)

        # seed the span head so this arm contains legolink-16
        for j in range(min(self.anchor_blocks, k)):
            add(j)
        while len(chosen) < k:
            add(
                max(
                    (b for b in range(n) if b not in chosen),
                    key=lambda b: val[b] * (self.c + got[b]) / (self.c + rowtot[b]),
                )
            )
        return sorted(chosen)

    @staticmethod
    def _span_ranges(
        request: "Request", num_computed_tokens: int
    ) -> list[tuple[int, int]]:
        """Computed [start, end) of each span, clipped to the computed prefix."""
        ranges = getattr(request, "pic_token_ranges", None)
        if not ranges:
            starts = sorted(request.span_starts or [])
            ranges = [
                (s, starts[i + 1] if i + 1 < len(starts) else None)
                for i, s in enumerate(starts)
            ]
        out = [
            (s, num_computed_tokens if e is None else min(e, num_computed_tokens))
            for s, e in ranges
            if s < num_computed_tokens
        ]
        return sorted(r for r in out if r[1] > r[0])

    def _select(
        self, request: "Request", importance: list[float], n: int
    ) -> list[tuple[int, int]]:
        bs = self.block_size
        selected: set[int] = set()
        # spans end at their cross boundary (pic_token_ranges), not the next start
        for start, end in self._span_ranges(request, n):
            blk0 = start // bs
            n_blk = min(end // bs, len(importance) // bs) - blk0
            attn = [
                sum(importance[(blk0 + b) * bs : (blk0 + b + 1) * bs])
                for b in range(max(0, n_blk))
            ]
            selected.update(
                blk0 + b for b in self._select_blocks(attn, self.k_per_span // bs)
            )

        # coalesce adjacent blocks into one gap
        gaps: list[tuple[int, int]] = []
        for blk in sorted(selected):
            if gaps and gaps[-1][1] == blk * bs:
                gaps[-1] = (gaps[-1][0], (blk + 1) * bs)
            else:
                gaps.append((blk * bs, (blk + 1) * bs))
        return gaps


class GapPolicyFactory:
    """Factory for creating GapPolicy instances from configuration."""

    _POLICIES = {
        "none": NoGapPolicy,
        "span_aware": SpanAwareGapPolicy,
        "qcfuse": QCFusePolicy,
        "neighbor_aware": NeighborAwarePolicy,
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
