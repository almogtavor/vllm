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
        # Select once per (request, prefix length). This path re-runs on every
        # step until the gap work fits, and the worker probe rewrites
        # qcfuse_importance between steps, so an unmemoized policy hands back a
        # different block set each time instead of converging.
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
    """Recompute a query-selected subset of tokens across ALL layers.

    Unlike SpanAwareGapPolicy's fixed-length span heads, QCFuse recomputes
    ``k_per_span`` tokens per span chosen by query-to-context attention mass;
    the critical layers are only the cheap selection lens. The importance
    vector is produced worker-side and handed back through
    ``request.qcfuse_importance``; until it arrives this returns no gaps.
    Selection is block-granular because ``_span_swap_indices`` and the PD
    dedup filter truncate via ``end // block_size``, so a sub-block gap would
    clobber a shared warmed block.
    """

    def __init__(
        self,
        critical_layers: str | tuple[int, ...] = (),
        block_size: int = 16,
        k_per_span: int = 0,
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
        if k_per_span <= 0:
            # The budget is per span, matched to legolink-K; without it the
            # policy has nothing to spend.
            raise ValueError(
                "QCFusePolicy requires k_per_span > 0 "
                "(set VLLM_V1_SPANS_QCFUSE_K_PER_SPAN)."
            )
        # The premise of selecting purely by attention mass is that the span
        # boundary carries no special positional error: prerotate remaps K from
        # span-local to request positions (QCFuse's Pi), leaving only contextual
        # staleness, which is spread across the span rather than concentrated at
        # its head. Without prerotate that premise is false and this arm would be
        # measuring something else, so refuse to run rather than mislead.
        if not envs.VLLM_V1_SPANS_PREROTATE:
            raise ValueError(
                "QCFusePolicy requires VLLM_V1_SPANS_PREROTATE=True: without the "
                "K position remap the span boundary carries a positional error "
                "that attention-mass selection does not address."
            )
        self.critical_layers = critical_layers
        self.block_size = block_size
        self.k_per_span = k_per_span

        logger.info(
            "QCFusePolicy initialized: k_per_span=%d critical_layers=%s",
            k_per_span,
            list(critical_layers),
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

        # k_per_span budget-matches this arm to legolink-K: legolink recomputes K
        # tokens at each span head, so the same total is K * (number of spans).
        # Matching the BUDGET is what makes the comparison a test of the
        # selection rule (positional vs query-relevant) rather than of compute.
        spans = request.span_starts or []
        n_spans = sum(1 for s in spans if s < num_computed_tokens) or 1
        budget = min(self.k_per_span * n_spans, num_computed_tokens)
        if budget <= 0:
            return []

        bs = self.block_size
        num_blocks = num_computed_tokens // bs
        if num_blocks == 0:
            return []
        scores = [
            (sum(importance[b * bs : (b + 1) * bs]), b) for b in range(num_blocks)
        ]
        scores.sort(reverse=True)
        keep = sorted(b for _, b in scores[: max(1, budget // bs)])
        gaps = [(b * bs, (b + 1) * bs) for b in keep]

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
            "QCFuse selected %d gaps (budget=%d tok) for request %s",
            len(gaps),
            budget,
            request.request_id,
        )
        return gaps


class MassClosurePolicy(QCFusePolicy):
    """Pick span blocks by attention x staleness x closure, greedily.

    QCFuse ranks blocks by attention mass alone, but a repaired block re-reads
    everything before it: recomputing b with its in-span predecessors still
    warm rewrites b from a context that is itself wrong. The gain of block b
    given the chosen set R is a(b) * (1+b)^-alpha * r(b|R), where a(b) is the
    probe's attention mass and r(b|R) is the fraction of the mass b re-reads
    that is already correct, under a measured decay kernel
    w(d) = amp * d^-beta + floor with an attention sink on the span's first
    block. Selection is greedy (argmax, add to R, re-score), so the method
    builds correct runs instead of scattering. Budget is k_per_span, matched
    to legolink-K.
    """

    # Measured on Qwen3-32B (see the docstring); the selection is insensitive
    # to everything here except the sink.
    c = 0.1
    alpha = 0.33
    beta = 1.25
    amp = 0.06
    floor = 0.0085
    sink = 0.554
    anchor_blocks = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.info(
            "MassClosurePolicy initialized: k_per_span=%d", self.k_per_span
        )

    def _closure_weights(self, n: int) -> tuple[list[float], list[float]]:
        """Decay kernel by distance, and the row totals it implies.

        ``w[d]`` is the weight a block puts on the predecessor ``d`` blocks
        back, so row ``b``'s total is ``sum(w[1..b])`` plus the sink that every
        row spends on the span's first block. The floor matters: without it the
        far two thirds of a long span contribute nothing to any row total, and
        the closure term saturates.
        """
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

        # Seed with a short contiguous head. Legolink's benefit saturates
        # around 16 tokens, so one block is nearly free and makes this arm
        # contain legolink-16 by construction rather than by luck.
        chosen = set(range(min(self.anchor_blocks, k, n)))
        got = [0.0] * n
        for j in chosen:
            for b in range(j + 1, n):
                got[b] += w[b - j] + (self.sink if j == 0 else 0.0)

        while len(chosen) < k:
            best, best_val = -1, -1.0
            for b in range(n):
                if b in chosen:
                    continue
                v = val[b] * (self.c + got[b]) / (self.c + rowtot[b])
                if v > best_val:
                    best, best_val = b, v
            if best < 0:
                break
            chosen.add(best)
            for b in range(best + 1, n):
                got[b] += w[b - best] + (self.sink if best == 0 else 0.0)
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

        # A span ends at its cross boundary, not at the next span's start:
        # between two tool reads sits ordinary conversation that was never
        # warmed prefix-free. pic_token_ranges is the extent the dual pd/pic
        # lookup itself uses, so selection and lookup agree on what a span is.
        # Taking the next span's start instead makes the whole intervening
        # conversation eligible, which is both wrong to repair and unbounded:
        # legolink survives the same approximation only because it never looks
        # past gap_length tokens.
        ranges = self._span_ranges(request, num_computed_tokens)
        if not ranges:
            return []

        bs = self.block_size
        budget_blocks = self.k_per_span // bs
        if budget_blocks <= 0:
            return []

        selected: list[int] = []
        for start, end in ranges:
            blk0 = start // bs
            n_blk = min(end // bs, len(importance) // bs) - blk0
            if n_blk <= 0:
                continue
            attn = [
                sum(importance[(blk0 + b) * bs : (blk0 + b + 1) * bs])
                for b in range(n_blk)
            ]
            selected.extend(blk0 + b for b in self._select_blocks(attn, budget_blocks))

        if not selected:
            return []

        # Coalesce runs so the scheduler sees as few gap requests as possible;
        # a run of adjacent blocks is one interval, not one interval per block.
        gaps: list[tuple[int, int]] = []
        for blk in sorted(set(selected)):
            if gaps and gaps[-1][1] == blk * bs:
                gaps[-1] = (gaps[-1][0], (blk + 1) * bs)
            else:
                gaps.append((blk * bs, (blk + 1) * bs))

        # SPANS: same recompute-once-per-unique-prefix dedup as the others.
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
            "MassClosure selected %d gaps (%d blocks over %d spans, %d tok/span) "
            "for request %s",
            len(gaps),
            len(set(selected)),
            len(ranges),
            self.k_per_span,
            request.request_id,
        )
        return gaps


class GapPolicyFactory:
    """Factory for creating GapPolicy instances from configuration."""

    _POLICIES = {
        "none": NoGapPolicy,
        "span_aware": SpanAwareGapPolicy,
        "qcfuse": QCFusePolicy,
        "mass_closure": MassClosurePolicy,
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
