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
        span_gaps = sched.gap_policy.get_gaps(
            request, num_computed_tokens, num_external_computed_tokens
        )
    else:
        span_gaps = []
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


class QCFusePolicy(GapPolicy):
    """QCFuse: recompute a query-selected subset of tokens across ALL layers.

    Unlike SpanAwareGapPolicy, which recomputes a fixed-length head at each span
    boundary, QCFuse recomputes ``floor(rho * N)`` of the cached tokens chosen by
    query-to-context attention mass. The critical layers are only the cheap
    selection lens that produces that importance signal -- they are not
    themselves what gets recomputed.

    The importance vector is produced worker-side and handed back through
    ``request.qcfuse_importance``. Until it arrives this returns no gaps, so the
    request is scheduled normally and the probe runs first.

    Selection is block-granular by default: ``_span_swap_indices`` and the PD
    dedup filter both truncate via ``end // block_size``, so a sub-block gap
    would skip the PIC->PD swap and clobber a shared warmed block. Block
    granularity preserves the rho budget exactly (in block quanta) while keeping
    the existing gap plumbing correct and the gap count far below max_num_seqs.
    """

    def __init__(
        self,
        rho: float = 0.1,
        critical_layers: str | tuple[int, ...] = (),
        block_size: int = 16,
        granularity: str = "block",
    ):
        if isinstance(critical_layers, str):
            critical_layers = tuple(
                int(x) for x in critical_layers.split(",") if x.strip()
            )
        else:
            critical_layers = tuple(critical_layers)
        # A silently-empty lens would make this arm a no-op that still reports as
        # QCFuse, so refuse it. ValueError is deliberate: create_policy() catches
        # only TypeError, so this propagates instead of degrading to NoGapPolicy.
        if not critical_layers:
            raise ValueError(
                "QCFusePolicy requires critical_layers (offline-profiled per "
                "model); set VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS."
            )
        if not 0.0 < rho <= 1.0:
            raise ValueError(f"QCFusePolicy rho must be in (0, 1], got {rho}")
        if granularity not in ("block", "token"):
            raise ValueError(
                f"QCFusePolicy granularity must be block|token, got {granularity}"
            )
        self.rho = rho
        self.critical_layers = critical_layers
        self.block_size = block_size
        self.granularity = granularity

        logger.info(
            "QCFusePolicy initialized: rho=%.3f critical_layers=%s granularity=%s",
            rho,
            list(critical_layers),
            granularity,
        )

    def get_gaps(
        self,
        request: "Request",
        num_computed_tokens: int,
        num_external_tokens: int,
    ) -> list[tuple[int, int]]:
        if num_computed_tokens == 0:
            return []

        importance = getattr(request, "qcfuse_importance", None)
        if importance is None:
            # Probe has not run yet; schedule normally and select next step.
            return []

        budget = int(self.rho * num_computed_tokens)
        if budget <= 0:
            return []

        bs = self.block_size
        if self.granularity == "block":
            num_blocks = num_computed_tokens // bs
            if num_blocks == 0:
                return []
            scores = [
                (sum(importance[b * bs : (b + 1) * bs]), b) for b in range(num_blocks)
            ]
            scores.sort(reverse=True)
            keep = sorted(b for _, b in scores[: max(1, budget // bs)])
            gaps = [(b * bs, (b + 1) * bs) for b in keep]
        else:
            ranked = sorted(
                range(min(len(importance), num_computed_tokens)),
                key=lambda t: importance[t],
                reverse=True,
            )[:budget]
            gaps = [(t, t + 1) for t in sorted(ranked)]

        # SPANS: same recompute-once-per-unique-prefix dedup as SpanAwareGapPolicy.
        sources = request.prefix_hit_sources
        if sources is not None:
            kept = []
            for s, e in gaps:
                blocks = range(s // bs, min(e // bs, len(sources)))
                if blocks and all(sources[b] == PrefixHitSource.PD for b in blocks):
                    continue
                kept.append((s, e))
            gaps = kept

        logger.info(
            "QCFuse selected %d gaps (rho=%.3f, budget=%d tok) for request %s",
            len(gaps),
            self.rho,
            budget,
            request.request_id,
        )
        return gaps


class GapPolicyFactory:
    """Factory for creating GapPolicy instances from configuration."""

    _POLICIES = {
        "none": NoGapPolicy,
        "span_aware": SpanAwareGapPolicy,
        "qcfuse": QCFusePolicy,
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
