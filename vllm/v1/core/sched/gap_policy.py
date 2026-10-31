# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Gap Policy for KV Cache Recomputation

This module provides abstractions for deciding where to insert recomputation gaps
within prefix-cached tokens. Gap policies are independent of where cached tokens
came from (local prefix cache, external connector, or both).
"""

from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import replace
from typing import TYPE_CHECKING

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.qcfuse import QCFUSE_MAX_QUERY_TOKENS, spanfuse_query_ids
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
    if (
        isinstance(sched.gap_policy, SpanFusePolicy)
        and request.spanfuse_state != "done"
    ):
        state = request.spanfuse_state
        if state == "inflight":
            # wait for the probe's importance, holding no budget
            request_queue.pop_request()
            step_skipped_waiting.prepend_request(request)
            return token_budget
        if state is None and did_prefix_lookup and not request.spanfuse_otf_warmed:
            warmed = _schedule_spanfuse_otf_warm(
                sched,
                request,
                num_computed_tokens,
                num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                token_budget,
                num_scheduled_tokens,
                virtual_reqs_out,
                virtual_gap_req_ids,
                request_queue,
                step_skipped_waiting,
            )
            if warmed is not NO_SPAN_GAPS:
                return warmed
        if state is None and did_prefix_lookup:
            probed = _schedule_spanfuse_probe(
                sched,
                request,
                num_computed_tokens,
                num_external_computed_tokens,
                num_new_local_computed_tokens,
                new_computed_blocks,
                token_budget,
                num_scheduled_tokens,
                virtual_reqs_out,
                virtual_gap_req_ids,
                request_queue,
                step_skipped_waiting,
            )
            if probed is not NO_SPAN_GAPS:
                return probed
        # measured: select exactly as QCFuse would at its prefix lookup
        did_prefix_lookup = did_prefix_lookup or state == "measured"
        request.spanfuse_state = "done"
    if request.pending_span_gaps:
        span_gaps = request.pending_span_gaps
    elif did_prefix_lookup and sched.gap_policy is not None:
        # memoize per (request, prefix length): the probe rewrites importance
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
        # SPANFUSE: misses that read the blind spans are rewritten in place after repair
        span_gaps = (
            span_gaps
            + miss_gaps(request, sched.block_size)
            + request.spanfuse_miss_gaps
        )
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
    request.spanfuse_miss_gaps = []
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


def _schedule_spanfuse_otf_warm(
    sched: "Scheduler",
    request: "Request",
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
    """Encode the cold spans of the uncached tail span-locally and cache them
    under their plain keys, then look the prefix up again ("otf" state), so the
    probe and repair see them as warmed hits exactly as a separate warm-up would."""
    request.spanfuse_otf_warmed = True
    bs = sched.block_size
    spans = [
        (s, e // bs * bs)
        for s, e in QCFusePolicy._span_ranges(request, request.num_tokens)
        if s >= num_computed_tokens and s % bs == 0 and e // bs * bs > s
    ]
    if not spans or num_external_computed_tokens:
        return NO_SPAN_GAPS
    sources = request.prefix_hit_sources or []
    end = spans[-1][1]
    pol = sched.gap_policy
    seed = request.spanfuse_seed
    # every earlier span was repaired under this prefix, so an in-context encode reads exact KV
    prefix_exact = all(
        sources[b] == PrefixHitSource.PD
        for s, e in QCFusePolicy._span_ranges(request, num_computed_tokens)
        for b in range(s // bs, min(cdiv(e, bs), len(sources)))
    )
    # whole span: encode every cold span in context now and cache it under the pd key,
    # or, as a seed, under the plain key so later contexts start from this copy, not the blind one
    if (
        (seed or all(pol._budget(request, s, e) >= e - s for s, e in spans))
        and request.pd_block_hashes is not None
        and len(sources) >= num_computed_tokens // bs
        and PrefixHitSource.MISS not in sources
        and (seed or prefix_exact)
        and len(request.pd_block_hashes) >= end // bs
        and end - num_computed_tokens <= sched.max_num_scheduled_tokens
    ):
        work = end - num_computed_tokens
        if work > token_budget:
            request.spanfuse_otf_warmed = False
            return None
        blocks = sched.kv_cache_manager.allocate_slots(
            request,
            work,
            num_new_computed_tokens=num_new_local_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            delay_cache_blocks=True,
        )
        if blocks is None:
            request.spanfuse_otf_warmed = False
            return None
        manager = sched.kv_cache_manager.coordinator.single_type_managers[0]
        req_blocks = manager.req_to_blocks[request.request_id]
        first = min(request.span_starts)
        idx = range(num_computed_tokens // bs, end // bs)
        sched.kv_cache_manager.block_pool.cache_blocks_under_hashes(
            [req_blocks[b] for b in idx],
            [
                request.pd_block_hashes[b]
                if b * bs >= first and not seed
                else request.block_hashes[b]
                for b in idx
            ],
            0,
        )
        if seed and prefix_exact:
            # a block hash shared by two spans here resolves to the first one's copy
            n = Counter(request.block_hashes[b] for b in idx)
            request.spanfuse_seeded = {
                s
                for s, e in spans
                if all(n[request.block_hashes[b]] == 1 for b in range(s // bs, e // bs))
            }
        logger.info(
            "SpanFuse: %d cold spans (%d tokens) encoded in context in %s%s",
            len(spans),
            work,
            request.request_id,
            " (seed)" if seed else "",
        )
        request.num_computed_tokens = num_computed_tokens
        parent_nrd = NewRequestData.from_request(
            request, sched.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
        )
        append_virtual_gap_reqs(
            parent_nrd,
            [(num_computed_tokens, end)],
            virtual_reqs_out,
            virtual_gap_req_ids,
            num_scheduled_tokens,
        )
        request.spanfuse_state = "otf"
        request_queue.pop_request()
        step_skipped_waiting.prepend_request(request)
        return token_budget - work
    work = sum(e - s for s, e in spans)
    if work > sched.max_num_scheduled_tokens:
        logger.warning(
            "SpanFuse: on-the-fly warm-up of %d tokens too large for %s",
            work,
            request.request_id,
        )
        return NO_SPAN_GAPS
    if work > token_budget:
        request.spanfuse_otf_warmed = False
        return None
    # nothing is cached here but the span blocks: the crosses would read blind spans
    blocks = sched.kv_cache_manager.allocate_slots(
        request,
        spans[-1][1] - num_computed_tokens,
        num_new_computed_tokens=num_new_local_computed_tokens,
        new_computed_blocks=new_computed_blocks,
        delay_cache_blocks=True,
    )
    if blocks is None:
        request.spanfuse_otf_warmed = False
        return None
    manager = sched.kv_cache_manager.coordinator.single_type_managers[0]
    req_blocks = manager.req_to_blocks[request.request_id]
    idx = [b for s, e in spans for b in range(s // bs, e // bs)]
    sched.kv_cache_manager.block_pool.cache_blocks_under_hashes(
        [req_blocks[b] for b in idx], [request.block_hashes[b] for b in idx], 0
    )
    logger.info(
        "SpanFuse: %d cold spans (%d tokens) warmed on the fly in %s%s",
        len(spans),
        work,
        request.request_id,
        " (seed fell back to blind)" if seed else "",
    )
    request.num_computed_tokens = num_computed_tokens
    parent_nrd = NewRequestData.from_request(
        request, sched.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    )
    n0 = len(virtual_reqs_out)
    append_virtual_gap_reqs(
        parent_nrd, spans, virtual_reqs_out, virtual_gap_req_ids, num_scheduled_tokens
    )
    for nrd in virtual_reqs_out[n0:]:
        nrd.is_gap_recompute = False  # span-local, as the plain key promises
    # the scheduler frees the blocks and runs a fresh lookup on the next step
    request.spanfuse_state = "otf"
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    return token_budget - work


def _schedule_spanfuse_probe(
    sched: "Scheduler",
    request: "Request",
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
    """Run the uncached tail once so its trailing query rows score the span
    blocks before any repair; MISS blocks are recomputed in the same step so
    the probe never reads the null placeholder, and again with the repair."""
    bs = sched.block_size
    sources = request.prefix_hit_sources or []
    ranges = QCFusePolicy._span_ranges(request, num_computed_tokens)
    # a pd hit means this exact prefix already repaired the span: reuse it as cached
    request.spanfuse_settled = {
        s
        for s, e in ranges
        if any(
            sources[b] == PrefixHitSource.PD
            for b in range(s // bs, min(e // bs, len(sources)))
        )
    }
    # the probe scores warmed spans only: a span with a MISS block is recomputed whole
    cold = [
        (s, e)
        for s, e in ranges
        if s not in request.spanfuse_settled
        and any(
            sources[b] == PrefixHitSource.MISS
            for b in range(s // bs, min(cdiv(e, bs), len(sources)))
        )
    ]
    if cold:
        logger.info(
            "SpanFuse: %d of %d spans unwarmed in %s, recomputed in context",
            len(cold),
            len(ranges),
            request.request_id,
        )
    # a seed is this request's own in-context copy: repairing it would recompute the same KV
    request.spanfuse_settled |= {s for s, _ in cold} | request.spanfuse_seeded
    request.spanfuse_miss_gaps = cold
    span_blocks = [
        b
        for s, e in ranges
        if s not in request.spanfuse_settled
        for b in range(s // bs, e // bs)
    ]
    if not any(
        b < len(sources) and sources[b] == PrefixHitSource.PIC for b in span_blocks
    ):
        return NO_SPAN_GAPS
    budgets = [
        (sched.gap_policy._budget(request, s, e), e - s)
        for s, e in ranges
        if s not in request.spanfuse_settled
    ]
    # whole span or zero budget: importance cannot change the selection, so skip the scoring pass
    if all(b >= n for b, n in budgets) or all(b <= 0 for b, _ in budgets):
        request.qcfuse_importance = [0.0] * num_computed_tokens
        return NO_SPAN_GAPS
    qids = [
        int(x)
        for x in envs.VLLM_V1_SPANS_QCFUSE_QUERY_TOKEN_IDS.replace("\\", "").split(",")
        if x.strip()
    ]
    if request.spanfuse_query_starts and qids:
        return _schedule_spanfuse_span_probes(
            sched,
            request,
            qids,
            num_computed_tokens,
            num_external_computed_tokens,
            num_new_local_computed_tokens,
            new_computed_blocks,
            token_budget,
            num_scheduled_tokens,
            virtual_reqs_out,
            virtual_gap_req_ids,
            request_queue,
            step_skipped_waiting,
        )
    tail = request.num_tokens - num_computed_tokens
    # the appended instruction rows are the query, so no cached window is re-run
    suffix = spanfuse_query_ids(sched.vllm_config.model_config)
    if request.num_tokens + len(suffix) > sched.max_model_len:
        suffix = []
    # cached rows re-run as query rows; the runner drops their KV writes
    w = max(0, request.num_tokens - QCFUSE_MAX_QUERY_TOKENS) // bs * bs
    w = num_computed_tokens if suffix else min(num_computed_tokens, w)
    window = num_computed_tokens - w
    miss = sched._merge_gaps(miss_gaps(request, bs) + request.spanfuse_miss_gaps)
    work = tail + window + len(suffix) + sum(e - s for s, e in miss)
    if work > sched.max_num_scheduled_tokens:
        # ponytail: oversized probe falls back to span-head repair; chunk if it fires
        logger.warning(
            "SpanFuse: probe of %d tokens too large for %s", work, request.request_id
        )
        request.qcfuse_importance = [0.0] * num_computed_tokens
        return NO_SPAN_GAPS
    if work > token_budget:
        return None
    # delayed: the probe's tail KV is recomputed (and cached) by the parent prefill
    blocks = sched.kv_cache_manager.allocate_slots(
        request,
        tail,
        num_new_computed_tokens=num_new_local_computed_tokens,
        new_computed_blocks=new_computed_blocks,
        num_external_computed_tokens=num_external_computed_tokens,
        num_lookahead_tokens=len(suffix),
        delay_cache_blocks=True,
        span_gaps=miss,
    )
    if blocks is None:
        return None
    # MISS blocks are now fresh copies of this prefix: never re-swapped, but
    # they read the blind spans, so the repair step rewrites them in place
    request.prefix_hit_sources = [
        PrefixHitSource.PD if s == PrefixHitSource.MISS else s for s in sources
    ]
    request.spanfuse_miss_gaps = miss
    request.num_computed_tokens = num_computed_tokens
    parent_nrd = NewRequestData.from_request(
        request, sched.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    )
    if miss:
        append_virtual_gap_reqs(
            parent_nrd,
            miss,
            virtual_reqs_out,
            virtual_gap_req_ids,
            num_scheduled_tokens,
        )
    probe = replace(
        parent_nrd,
        req_id=request.request_id + ".probe",
        parent_req_id=request.request_id,
        num_computed_tokens=w,
        is_spanfuse_probe=True,
        spanfuse_span_blocks=span_blocks,
        spanfuse_window=window,
        # the suffix KV sits in decode slots, overwritten before any read
        prompt_token_ids=list(parent_nrd.prompt_token_ids or ()) + suffix,
        spanfuse_query_rows=len(suffix),
    )
    num_scheduled_tokens[probe.req_id] = tail + window + len(suffix)
    virtual_reqs_out.append(probe)
    virtual_gap_req_ids.add(probe.req_id)
    request.spanfuse_state = "inflight"
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    return token_budget - work


def _schedule_spanfuse_span_probes(
    sched: "Scheduler",
    request: "Request",
    qids: list[int],
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
    """One probe row per re-read span: the mw sentence spliced at that span's
    tail reads the whole prefix and scores only that span's blocks."""
    bs, nq = sched.block_size, len(qids)
    sources = request.prefix_hit_sources or []
    toks = request.prompt_token_ids or []
    # stripped: the mw sent only the block-aligned start; the sentence lives in scratch
    stripped = request.spanfuse_query_stripped
    probes = []
    for s, e in QCFusePolicy._span_ranges(request, num_computed_tokens):
        if s in request.spanfuse_settled or not any(
            sources[b] == PrefixHitSource.PIC
            for b in range(s // bs, min(e // bs, len(sources)))
        ):
            continue
        qs = next(
            (
                q
                for q in request.spanfuse_query_starts or ()
                # stripped: the start is the span's block-aligned end
                if ((s < q <= e and not q % bs) if stripped else s <= q < e)
                and q + (0 if stripped else nq) <= num_computed_tokens
            ),
            None,
        )
        if qs is None:
            continue
        if not stripped and list(toks[qs : qs + nq]) != qids:
            logger.warning(
                "SpanFuse: sentence ids differ at %d in %s", qs, request.request_id
            )
            continue
        probes.append((qs, list(range(s // bs, qs // bs)), s))
    if not probes:
        return NO_SPAN_GAPS
    miss = sched._merge_gaps(miss_gaps(request, bs) + request.spanfuse_miss_gaps)
    # every probe is a batch row; the input batch holds max_num_running_reqs
    free = (
        sched.max_num_running_reqs
        - len(sched.running)
        - len(virtual_reqs_out)
        - len(miss)
    )
    if free < 1:
        return None
    f, nqb = cdiv(request.num_tokens, bs), cdiv(nq, bs)
    if stripped:
        # each probe writes its sentence KV to its own scratch blocks past the prompt
        free = min(free, (cdiv(sched.max_model_len, bs) - f) // nqb)
        if free < 1:
            return NO_SPAN_GAPS
    probes = probes[-free:]  # ponytail: newest spans win when rows run out
    work = nq * len(probes) + sum(e - s for s, e in miss)
    if work > token_budget:
        return None
    tail = request.num_tokens - num_computed_tokens
    scratch = (f + nqb * len(probes)) * bs - request.num_tokens if stripped else 0
    blocks = sched.kv_cache_manager.allocate_slots(
        request,
        tail,
        num_new_computed_tokens=num_new_local_computed_tokens,
        new_computed_blocks=new_computed_blocks,
        num_external_computed_tokens=num_external_computed_tokens,
        num_lookahead_tokens=scratch,
        delay_cache_blocks=True,
        span_gaps=miss,
    )
    if blocks is None:
        return None
    request.prefix_hit_sources = [
        PrefixHitSource.PD if s == PrefixHitSource.MISS else s for s in sources
    ]
    request.spanfuse_miss_gaps = miss
    request.num_computed_tokens = num_computed_tokens
    parent_nrd = NewRequestData.from_request(
        request, sched.kv_cache_manager.get_blocks(request.request_id).get_block_ids()
    )
    if miss:
        append_virtual_gap_reqs(
            parent_nrd,
            miss,
            virtual_reqs_out,
            virtual_gap_req_ids,
            num_scheduled_tokens,
        )
    for j, (qs, span_blocks, s) in enumerate(probes):
        # the sentence is cached: window = all its rows, so no KV is rewritten
        probe = replace(
            parent_nrd,
            req_id=f"{request.request_id}.probe{j}",
            parent_req_id=request.request_id,
            num_computed_tokens=qs,
            is_spanfuse_probe=True,
            spanfuse_span_blocks=span_blocks,
            spanfuse_window=nq,
            spanfuse_query_rows=nq,
        )
        if stripped:
            lo = f + j * nqb
            probe = replace(
                probe,
                block_ids=tuple(
                    g[: qs // bs] + g[lo : lo + nqb] for g in parent_nrd.block_ids
                ),
                prompt_token_ids=list(toks[:qs]) + qids,
                spanfuse_window=0,
                spanfuse_probe_lb=s,
            )
        num_scheduled_tokens[probe.req_id] = nq
        virtual_reqs_out.append(probe)
        virtual_gap_req_ids.add(probe.req_id)
    logger.info(
        "SpanFuse per-span probe %s: %d sentences", request.request_id, len(probes)
    )
    request.spanfuse_state = "inflight"
    request_queue.pop_request()
    step_skipped_waiting.prepend_request(request)
    return token_budget - work


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

        gaps = _drop_pd_gaps(request, gaps, self.block_size)

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
    """Cut pd-hit blocks out of gaps: a pd copy already holds this prefix's
    KV and is shared, so it is never rewritten (recompute once per prefix)."""
    sources = request.prefix_hit_sources
    if sources is None:
        return gaps
    out = []
    for s, e in gaps:
        cur = s
        for b in range(s // bs, min(-(-e // bs), len(sources))):
            if sources[b] != PrefixHitSource.PD:
                continue
            if b * bs > cur:
                out.append((cur, b * bs))
            cur = max(cur, min(e, (b + 1) * bs))
        if e > cur:
            out.append((cur, e))
    return out


class QCFusePolicy(GapPolicy):
    """Recompute the cached tokens with the most query attention.

    Budget is ``k_per_span`` tokens per span, else ``rho`` of each span's tokens.
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

    def _budget(self, request: "Request", start: int, end: int) -> int:
        """Tokens to repair in span [start, end): the request's share, else the server's."""
        rho = getattr(request, "spanfuse_rho", None)
        if rho is not None:
            return int(rho * (end - start))
        return self.k_per_span or int(self.rho * (end - start))

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
        # TopK inside each span: prefix blocks are already exact
        bs, gaps = self.block_size, []
        for start, end in self._span_ranges(request, n):
            budget = self._budget(request, start, end)
            if budget <= 0:
                continue
            if self.granularity == "token":
                end = min(end, len(importance))
                ranked = sorted(range(start, end), key=lambda t: -importance[t])
                gaps += [(t, t + 1) for t in sorted(ranked[:budget])]
                continue
            blocks = range(start // bs, min(end // bs, len(importance) // bs))
            mass = {b: sum(importance[b * bs : (b + 1) * bs]) for b in blocks}
            top = sorted(blocks, key=lambda b: -mass[b])[: max(1, budget // bs)]
            gaps += [(b * bs, (b + 1) * bs) for b in sorted(top)]
        return gaps


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


class SpanFusePolicy(QCFusePolicy):
    """QCFuse whose importance comes from the current request's probe."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # every span keeps its first N blocks repaired; the scoring pass ranks the rest
        self.anchor_blocks = envs.VLLM_V1_SPANS_SPANFUSE_ANCHOR_BLOCKS
        if self.anchor_blocks < 0:
            raise ValueError(f"SpanFuse: anchor_blocks must be >= 0, got {self.anchor_blocks}")
        logger.info(
            "SpanFusePolicy initialized: k_per_span=%d rho=%s anchor_blocks=%d",
            self.k_per_span, self.rho, self.anchor_blocks,
        )

    def _select(
        self, request: "Request", importance: list[float], n: int
    ) -> list[tuple[int, int]]:
        settled = request.spanfuse_settled
        if self.granularity == "token":
            gaps = super()._select(request, importance, n)
            return [
                g
                for g in gaps
                if not any(s <= g[0] < e for s, e in self._span_ranges(request, n) if s in settled)
            ]
        bs, gaps = self.block_size, []
        toks, pad = request.prompt_token_ids, envs.VLLM_V1_SPANS_PAD_TOKEN

        # pad rows carry no content, so they never vote for a block
        def mass(b: int) -> float:
            r = range(b * bs, (b + 1) * bs)
            return sum(importance[i] for i in r if toks[i] != pad)

        for start, end in self._span_ranges(request, n):
            budget = self._budget(request, start, end)
            if budget <= 0 or start in settled:
                continue
            k = max(1, budget // bs)
            blocks = list(range(start // bs, min(end // bs, len(importance) // bs)))
            head = blocks[: min(self.anchor_blocks, k)]
            rest = sorted(blocks[len(head) :], key=lambda b: -mass(b))
            top = head + rest[: k - len(head)]
            gaps += [(b * bs, (b + 1) * bs) for b in sorted(top)]
        return gaps


class GapPolicyFactory:
    """Factory for creating GapPolicy instances from configuration."""

    _POLICIES = {
        "none": NoGapPolicy,
        "span_aware": SpanAwareGapPolicy,
        "qcfuse": QCFusePolicy,
        "neighbor_aware": NeighborAwarePolicy,
        "spanfuse": SpanFusePolicy,
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
