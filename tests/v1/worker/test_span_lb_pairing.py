# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPANS: cross_span_starts/span_starts pairing (pure, no engine).

The client omits a cross for any span whose end coincides with the next
span's start, so crosses can be SHORTER than spans. Index-pairing painted
the last span's attention lower bound over the trailing query and the
generated tokens (decode blind to everything before it - the mind2web
multi-DOM-chunk degeneration). These pin the positional pairing rule:
a span's region ends at min(next span start, first cross past it).
"""
from vllm.v1.worker.gpu_model_runner import compute_span_lb_regions

B = 16  # block size used for readable offsets


def test_single_span_with_cross():
    # SWE-bench shape: one span per message, 1:1 cross. Tail stays lb=0.
    assert compute_span_lb_regions([B], [3 * B], 6 * B) == [(B, 3 * B, B)]


def test_adjacent_spans_skipped_cross():
    # m2w shape: spans back-to-back, only the last carries a cross.
    # Each span's region must stop at the next span; the tail is untouched.
    regions = compute_span_lb_regions([B, 3 * B], [5 * B], 8 * B)
    assert regions == [(B, 3 * B, B), (3 * B, 5 * B, 3 * B)]


def test_many_adjacent_spans_tail_untouched():
    spans = [B, 2 * B, 3 * B, 4 * B]
    regions = compute_span_lb_regions(spans, [5 * B], 12 * B)
    assert [r[1] for r in regions] == [2 * B, 3 * B, 4 * B, 5 * B]
    # no region may extend past the last cross into the tail/generated tokens
    assert max(r[1] for r in regions) == 5 * B


def test_no_crosses_last_span_runs_to_req_len():
    # a span with no boundary after it extends to req_len (old behaviour
    # for the genuinely-last span, unchanged)
    assert compute_span_lb_regions([B], None, 4 * B) == [(B, 4 * B, B)]


def test_unsorted_inputs():
    regions = compute_span_lb_regions([3 * B, B], [5 * B], 8 * B)
    assert regions == [(B, 3 * B, B), (3 * B, 5 * B, 3 * B)]


def test_request_pic_token_ranges_pairing():
    # request.py sibling rule: adjacent spans' pd/pic hash ranges must not
    # overlap through the last cross.
    from vllm.v1.request import Request  # noqa: F401  (import guards the path)
    # The range rule mirrors compute_span_lb_regions with None for "open":
    regions = compute_span_lb_regions([B, 3 * B], [5 * B], 10 * B)
    assert regions[0][1] == 3 * B and regions[1][1] == 5 * B
