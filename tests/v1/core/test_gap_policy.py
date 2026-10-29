# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

import vllm.envs as envs
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import PrefixHitSource
from vllm.v1.core.sched.gap_policy import (
    GapPolicyFactory,
    NoGapPolicy,
    SpanAwareGapPolicy,
    miss_gaps,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request

pytestmark = pytest.mark.spans


def make_span_request(
    prompt_len: int,
    span_starts: list[int] | None = None,
    cross_span_starts: list[int] | None = None,
) -> Request:
    extra_args = {}
    if span_starts is not None:
        extra_args["span_starts"] = span_starts
    if cross_span_starts is not None:
        extra_args["cross_span_starts"] = cross_span_starts

    sampling_params = SamplingParams(
        max_tokens=17,
        extra_args=extra_args if extra_args else None,
    )
    sampling_params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id="gap_test",
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=sampling_params,
        pooling_params=None,
    )


class TestSpanAwareGapPolicy:
    def setup_method(self):
        self._original = envs.VLLM_V1_SPANS_ENABLED
        envs.VLLM_V1_SPANS_ENABLED = True

    def teardown_method(self):
        envs.VLLM_V1_SPANS_ENABLED = self._original

    def test_gaps_at_span_starts(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 96), (128, 160)]

    def test_no_gaps_when_no_span_starts(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=None)
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_gap_clamped_to_next_span(self):
        policy = SpanAwareGapPolicy(gap_length=100)
        req = make_span_request(256, span_starts=[64, 128])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(64, 128), (128, 228)]

    def test_gap_clamped_to_computed_tokens(self):
        policy = SpanAwareGapPolicy(gap_length=100)
        req = make_span_request(256, span_starts=[200])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == [(200, 256)]

    def test_zero_gap_length_disables(self):
        policy = SpanAwareGapPolicy(gap_length=0)
        req = make_span_request(256, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_span_starts_beyond_computed_tokens_ignored(self):
        policy = SpanAwareGapPolicy(gap_length=32)
        req = make_span_request(256, span_starts=[64, 200])
        gaps = policy.get_gaps(req, num_computed_tokens=100, num_external_tokens=0)
        assert gaps == [(64, 96)]

    def test_no_gap_policy_returns_empty(self):
        policy = NoGapPolicy()
        req = make_span_request(256, span_starts=[64])
        gaps = policy.get_gaps(req, num_computed_tokens=256, num_external_tokens=0)
        assert gaps == []

    def test_factory_creates_span_aware(self):
        policy = GapPolicyFactory.create_policy("span_aware", {"gap_length": 64})
        assert isinstance(policy, SpanAwareGapPolicy)
        assert policy.gap_length == 64


def test_miss_gaps_cover_only_miss_blocks():
    req = make_span_request(256, span_starts=[64])
    D, M, S = PrefixHitSource.PD, PrefixHitSource.MISS, PrefixHitSource.PIC
    req.prefix_hit_sources = [M, M, D, S, M, S]
    assert miss_gaps(req, 16) == [(0, 16), (16, 32), (64, 80)]
    req.prefix_hit_sources = None
    assert miss_gaps(req, 16) == []


def test_qcfuse_selects_inside_spans_only(monkeypatch):
    from vllm.v1.core.sched.gap_policy import QCFusePolicy

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    req = make_span_request(160, span_starts=[64])
    imp = [0.0] * 160
    imp[0] = 100.0  # the sink: exact prefix, never worth recomputing
    imp[85] = 1.0
    assert QCFusePolicy(k_per_span=16)._select(req, imp, 160) == [(80, 96)]
    assert QCFusePolicy(rho=0.2)._select(req, imp, 160) == [(80, 96)]


def test_qcfuse_unmeasured_store_repairs_span_head(monkeypatch):
    from vllm.v1.core.sched.gap_policy import QCFusePolicy
    from vllm.v1.core.sched.qcfuse_store import QCFuseImportanceStore

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    req = make_span_request(160, span_starts=[64])
    imp = QCFuseImportanceStore(16).lookup([object() for _ in range(10)], 160)
    assert imp == [0.0] * 160
    req.qcfuse_importance = imp
    assert QCFusePolicy(k_per_span=16).get_gaps(req, 160, 0) == [(64, 80)]


def test_spanfuse_span_only_scoring_and_query_probe(monkeypatch):
    """(1) the probe softmax covers span blocks only; (2) a current-query probe
    runs before repair; the span head block and the probe's top block are what
    get repaired; (5) the appended instruction's rows are the probe's query."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import torch

    from vllm.v1.attention import qcfuse
    from vllm.v1.attention.qcfuse import QCFuseImportanceCapturer
    from vllm.v1.core.sched.gap_policy import SpanFusePolicy, schedule_span_gaps
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    monkeypatch.setattr(qcfuse, "_SPANFUSE_QUERY_IDS", [])  # (2)-(4): prompt tail
    bs, n_blk, span = 16, 10, [4, 5, 6, 7, 8]

    # (1) mass lands only on span blocks, sums to 1 per (head, query row),
    # ignores non-span keys, and keeps the full-softmax ranking inside the spans
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", False)
    torch.manual_seed(0)
    kv = torch.randn(2, n_blk, bs, 2, 8)
    q = torch.randn(4, 4, 8)
    table = torch.arange(n_blk)[None]
    cap = QCFuseImportanceCapturer(1, n_blk * bs, "cpu")
    cap.begin_step(table, [(0, 0, 4, n_blk * bs, span, n_blk * bs, None)])
    cap.capture(q, kv, 2, 0.3)
    got = cap.buffer[0].clone()
    in_span = torch.zeros(n_blk * bs, dtype=torch.bool)
    for b in span:
        in_span[b * bs : (b + 1) * bs] = True
    assert got[~in_span].eq(0).all()
    assert torch.isclose(got.sum(), torch.tensor(4.0 * 2))
    kv2 = kv.clone()
    kv2[0, :4] = torch.randn_like(kv2[0, :4])
    cap.begin_step(table, [(0, 0, 4, n_blk * bs, span, n_blk * bs, None)])
    cap.capture(q, kv2, 2, 0.3)
    assert torch.allclose(cap.buffer[0], got)
    cap.begin_step(table, [(0, 0, 4, n_blk * bs, None, n_blk * bs, None)])
    cap.capture(q, kv, 2, 0.3)
    full = cap.buffer[0][in_span]
    # same rows, one shared normalizer per row: per-row rankings are identical
    top = lambda m: m.topk(8).indices.sort().values
    assert torch.equal(top(full), top(got[in_span]))

    # SPANS caches K at p - L: the probe rotates it back by L (here L = p, K
    # cached at 0) before scoring Q; the query rows sit at L = 0
    hd, rot = 8, 6
    inv = 1.0 / (10000 ** (torch.arange(0, rot, 2).float() / rot))
    ang = torch.arange(n_blk * bs).float()[:, None] * inv
    cos_sin = torch.cat((ang.cos(), ang.sin()), -1)
    k = kv[0].flatten(0, 1)
    z = (
        torch.complex(k[..., : rot // 2], k[..., rot // 2 : rot])
        * torch.polar(torch.ones_like(ang), ang)[:, None]
    )
    k_rot = torch.cat((z.real, z.imag, k[..., rot:hd]), -1)
    pos = torch.cat([torch.arange(b * bs, (b + 1) * bs) for b in span])
    lk = torch.cat((torch.arange(n_blk * bs), torch.zeros(4, dtype=torch.long)))
    want = QCFuseImportanceCapturer._mass(q.view(4, 2, 2, hd).mean(2), k_rot[pos], 0.3)
    cap.begin_step(table, [(0, 0, 4, n_blk * bs, span, n_blk * bs, lk)])
    cap.capture(q, kv, 2, 0.3, cos_sin, rot)
    assert torch.allclose(cap.buffer[0][pos], want, atol=1e-5)
    assert not torch.allclose(got[pos], want, atol=1e-3)
    cap.begin_step(table, [(0, 0, 4, n_blk * bs, None, n_blk * bs, lk)])
    cap.capture(q, kv, 2, 0.3, cos_sin, rot)
    want_full = QCFuseImportanceCapturer._mass(q.view(4, 2, 2, hd).mean(2), k_rot, 0.3)
    assert torch.allclose(cap.buffer[0], want_full, atol=1e-5)
    # query rows inside the last span block never score keys after themselves
    cap.begin_step(table, [(0, 0, 4, n_blk * bs, span, 8 * bs + 2, None)])
    cap.capture(q, kv, 2, 0.3)
    assert cap.buffer[0][8 * bs + 6 :].eq(0).all()
    assert cap.buffer[0][8 * bs + 2 : 8 * bs + 6].gt(0).all()
    assert torch.isclose(cap.buffer[0].sum(), torch.tensor(4.0 * 2))
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    with pytest.raises(RuntimeError, match="RoPE"):
        cap.capture(q, kv, 2, 0.3)

    # (2) first pass schedules only the probe over the tail; second pass repairs
    # the block the probe ranked highest
    req = make_span_request(176, span_starts=[64])
    D, S = PrefixHitSource.PD, PrefixHitSource.PIC
    req.prefix_hit_sources = [D] * 4 + [S] * 6
    req.status = None
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([0] * 11,)
    sched = SimpleNamespace(
        gap_policy=SpanFusePolicy(k_per_span=32),
        block_size=bs,
        max_num_scheduled_tokens=8192,
        max_model_len=4096,
        vllm_config=SimpleNamespace(model_config=None),
        connector=None,
        _merge_gaps=Scheduler._merge_gaps,
        kv_cache_manager=SimpleNamespace(
            allocate_slots=MagicMock(return_value=blocks),
            get_blocks=MagicMock(return_value=blocks),
        ),
    )
    queue = create_request_queue(SchedulingPolicy.FCFS)
    queue.add_request(req)
    skipped = create_request_queue(SchedulingPolicy.FCFS)
    out: list = []
    ids: set[str] = set()
    nsched: dict[str, int] = {}
    call = lambda: schedule_span_gaps(
        sched,
        req,
        True,
        160,
        0,
        160,
        MagicMock(),
        8192,
        nsched,
        out,
        ids,
        queue,
        skipped,
    )
    call()
    assert [r.req_id for r in out] == ["gap_test.probe"]
    assert out[0].is_spanfuse_probe and out[0].spanfuse_span_blocks == [
        4,
        5,
        6,
        7,
        8,
        9,
    ]
    # always QCFUSE_MAX_QUERY_TOKENS query rows: 16 uncached + 48 cached rows
    assert nsched == {"gap_test.probe": 64} and req.spanfuse_state == "inflight"
    assert (out[0].num_computed_tokens, out[0].spanfuse_window) == (112, 48)
    imp = [0.0] * 160
    imp[120] = 1.0
    req.qcfuse_importance, req.spanfuse_state = imp, "measured"
    queue.add_request(req)
    out.clear()
    ids.clear()
    nsched.clear()
    call()
    assert req.spanfuse_state == "done"
    got = [(g.gap_start, g.gap_start + nsched[g.req_id]) for g in out]
    assert (64, 80) in got and (112, 128) in got

    # (3) a span with MISS blocks was never warmed: recomputed whole, never probed
    M = PrefixHitSource.MISS
    req.prefix_hit_sources = [D] * 4 + [M] * 3 + [S] * 3
    req.spanfuse_state = req.span_gaps_selection = None
    req.pending_span_gaps = []
    queue.add_request(req)
    out.clear()
    ids.clear()
    nsched.clear()
    call()
    assert req.spanfuse_state == "done" and req.spanfuse_settled == {64}
    assert [(g.gap_start, g.gap_start + nsched[g.req_id]) for g in out] == [(64, 160)]

    # (4) MISS blocks outside a warmed span ride the probe as ONE virtual
    # request, and the repair step recomputes them again
    req0, req = req, make_span_request(176, span_starts=[80])
    req.prefix_hit_sources = [D] + [M] * 3 + [D] + [S] * 5
    req.status = None
    queue.add_request(req)
    out.clear()
    ids.clear()
    nsched.clear()
    call()
    assert [r.req_id for r in out] == ["gap_test.16", "gap_test.probe"]
    assert nsched == {"gap_test.16": 48, "gap_test.probe": 64}
    req.qcfuse_importance, req.spanfuse_state = imp, "measured"
    queue.add_request(req)
    out.clear()
    ids.clear()
    nsched.clear()
    call()
    got = [(g.gap_start, g.gap_start + nsched[g.req_id]) for g in out]
    assert (16, 64) in got and (80, 96) in got
    assert req.spanfuse_miss_gaps == []
    req = req0

    # (5) an appended instruction replaces the cached window as the query rows
    monkeypatch.setattr(qcfuse, "_SPANFUSE_QUERY_IDS", [7] * 12)
    req.spanfuse_state = req.span_gaps_selection = None
    req.prefix_hit_sources = [D] * 4 + [S] * 6
    queue.add_request(req)
    out.clear()
    ids.clear()
    nsched.clear()
    call()
    p = out[0]
    assert nsched == {"gap_test.probe": 16 + 12} and p.spanfuse_window == 0
    assert p.spanfuse_query_rows == 12 and p.prompt_token_ids[-12:] == [7] * 12


def test_spanfuse_per_span_probe_reads_each_spliced_sentence(monkeypatch):
    """With mw query starts, each re-read span gets its own probe row: the
    sentence spliced at that span's tail scores only that span's blocks."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from vllm.v1.attention import qcfuse
    from vllm.v1.core.sched.gap_policy import SpanFusePolicy, schedule_span_gaps
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_QUERY_TOKEN_IDS", "7,8,9,10")
    D, S = PrefixHitSource.PD, PrefixHitSource.PIC
    blocks = MagicMock()
    blocks.get_block_ids.return_value = (list(range(13)),)

    def run(starts, rows=8, bad=False, stripped=False, settled=False):
        req = make_span_request(176, span_starts=[32, 96])
        req.spanfuse_query_stripped = stripped
        req.prompt_token_ids[64:68] = [7, 8, 9, 10]
        req.prompt_token_ids[128:132] = [7, 8, 9, 99] if bad else [7, 8, 9, 10]
        req.spanfuse_query_starts = starts
        req.prefix_hit_sources = [D] * (3 if settled else 2) + [S] * (7 if settled else 8)
        req.status = None
        sched = SimpleNamespace(
            gap_policy=SpanFusePolicy(k_per_span=32),
            block_size=16,
            max_num_scheduled_tokens=8192,
            max_model_len=4096,
            max_num_running_reqs=rows,
            running=[],
            vllm_config=SimpleNamespace(model_config=None),
            connector=None,
            _merge_gaps=Scheduler._merge_gaps,
            kv_cache_manager=SimpleNamespace(
                allocate_slots=MagicMock(return_value=blocks),
                get_blocks=MagicMock(return_value=blocks),
            ),
        )
        queue = create_request_queue(SchedulingPolicy.FCFS)
        queue.add_request(req)
        out, nsched = [], {}
        schedule_span_gaps(
            sched, req, True, 160, 0, 160, MagicMock(), 8192, nsched, out, set(),
            queue, create_request_queue(SchedulingPolicy.FCFS),
        )
        return req, out, nsched

    req, out, nsched = run([64, 128])
    got = [(p.req_id, p.num_computed_tokens, p.spanfuse_span_blocks) for p in out]
    assert got == [("gap_test.probe0", 64, [2, 3]), ("gap_test.probe1", 128, [6, 7])]
    assert all(p.spanfuse_window == p.spanfuse_query_rows == 4 for p in out)
    assert nsched == {"gap_test.probe0": 4, "gap_test.probe1": 4}
    assert req.spanfuse_state == "inflight"
    # a sentence whose ids differ is not a probe
    _, out, _ = run([64, 128], bad=True)
    assert [p.num_computed_tokens for p in out] == [64]
    # one free batch row: the newest span keeps it
    _, out, _ = run([64, 128], rows=1)
    assert [(p.req_id, p.num_computed_tokens) for p in out] == [("gap_test.probe0", 128)]
    # stripped: the sentence is not in the prompt; each probe writes it to scratch
    req, out, _ = run([64, 128, 136], stripped=True)
    assert [p.block_ids for p in out] == [(list(range(4)) + [11],), (list(range(8)) + [12],)]
    assert [p.prompt_token_ids[-4:] for p in out] == [[7, 8, 9, 10]] * 2
    assert [p.spanfuse_probe_lb for p in out] == [32, 96]
    assert all(p.spanfuse_window == 0 for p in out)
    # span 0 hit its pd repair from an earlier turn: reused as cached, never re-probed
    req, out, _ = run([64, 128], settled=True)
    assert req.spanfuse_settled == {32}
    assert [p.spanfuse_span_blocks for p in out] == [[6, 7]]
    gaps = SpanFusePolicy(k_per_span=32)._select(req, [1.0] * 160, 160)
    assert gaps and all(s >= 96 for s, _ in gaps)
    # no mw starts: the single tail probe
    monkeypatch.setattr(qcfuse, "_SPANFUSE_QUERY_IDS", [])
    _, out, _ = run(None)
    assert [p.req_id for p in out] == ["gap_test.probe"]


def test_spanfuse_pad_rows_do_not_score(monkeypatch):
    """A block of pads with high probe mass loses to a content block."""
    from vllm.v1.core.sched.gap_policy import SpanFusePolicy

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_PAD_TOKEN", 27)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    req = make_span_request(96, span_starts=[16])
    req.prompt_token_ids[64:80] = [27] * 16
    imp = [0.0] * 96
    imp[64:80] = [1.0] * 16
    imp[40] = 0.5
    pol = SpanFusePolicy(k_per_span=32)
    assert pol._select(req, imp, 96) == [(16, 32), (32, 48)]
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_PAD_TOKEN", -1)
    assert pol._select(req, imp, 96) == [(16, 32), (64, 80)]


def test_spanfuse_anchor_blocks_env(monkeypatch):
    """ANCHOR_BLOCKS=2 repairs span blocks 0-1, the probe ranks only the rest."""
    from vllm.v1.core.sched.gap_policy import SpanFusePolicy

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    req = make_span_request(128, span_starts=[16])
    imp = [0.0] * 128
    imp[96:112] = [1.0] * 16
    imp[80] = 0.5
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_SPANFUSE_ANCHOR_BLOCKS", 2)
    assert SpanFusePolicy(k_per_span=48)._select(req, imp, 128) == [
        (16, 32), (32, 48), (96, 112)
    ]
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_SPANFUSE_ANCHOR_BLOCKS", 1)
    assert SpanFusePolicy(k_per_span=48)._select(req, imp, 128) == [
        (16, 32), (80, 96), (96, 112)
    ]


def test_spanfuse_otf_warm_then_probe(monkeypatch):
    """A cold tail span is encoded span-locally and cached under its plain key,
    then the re-lookup sees it as a PIC hit and the probe repairs it."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from vllm.v1.attention import qcfuse
    from vllm.v1.core.sched.gap_policy import SpanFusePolicy, schedule_span_gaps
    from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue

    monkeypatch.setattr(envs, "VLLM_V1_SPANS_ENABLED", True)
    monkeypatch.setattr(qcfuse, "_SPANFUSE_QUERY_IDS", [])
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "0")
    bs = 16
    req = make_span_request(176, span_starts=[64], cross_span_starts=[128])
    req.status = None
    req.block_hashes = [f"h{i}" for i in range(11)]
    blocks = MagicMock()
    blocks.get_block_ids.return_value = ([0] * 11,)
    pool = MagicMock()
    sched = SimpleNamespace(
        gap_policy=SpanFusePolicy(k_per_span=32),
        block_size=bs,
        max_num_scheduled_tokens=8192,
        max_model_len=4096,
        vllm_config=SimpleNamespace(model_config=None),
        connector=None,
        _merge_gaps=Scheduler._merge_gaps,
        kv_cache_manager=SimpleNamespace(
            allocate_slots=MagicMock(return_value=blocks),
            get_blocks=MagicMock(return_value=blocks),
            block_pool=pool,
            coordinator=SimpleNamespace(
                single_type_managers=[
                    SimpleNamespace(req_to_blocks={"gap_test": [f"b{i}" for i in range(11)]})
                ]
            ),
        ),
    )
    queue = create_request_queue(SchedulingPolicy.FCFS)
    skipped = create_request_queue(SchedulingPolicy.FCFS)
    out: list = []
    nsched: dict[str, int] = {}
    call = lambda n: schedule_span_gaps(
        sched, req, True, n, 0, n, MagicMock(), 8192, nsched, out, set(), queue, skipped
    )

    # only the head is cached: the span is warmed span-locally, never probed
    queue.add_request(req)
    assert call(64) == 8192 - 64
    assert [(r.req_id, r.is_gap_recompute) for r in out] == [("gap_test.64", False)]
    assert nsched == {"gap_test.64": 64} and req.spanfuse_state == "otf"
    pool.cache_blocks_under_hashes.assert_called_once_with(
        ["b4", "b5", "b6", "b7"], ["h4", "h5", "h6", "h7"], 0
    )

    # the re-lookup returns the warmed span as PIC: the probe runs, then repair
    D, S = PrefixHitSource.PD, PrefixHitSource.PIC
    req.spanfuse_state, req.prefix_hit_sources = None, [D] * 4 + [S] * 4
    queue.add_request(req)
    out.clear()
    nsched.clear()
    call(128)
    assert [r.req_id for r in out] == ["gap_test.probe"]
    assert out[0].spanfuse_span_blocks == [4, 5, 6, 7]
    imp = [0.0] * 128
    imp[100] = 1.0
    req.qcfuse_importance, req.spanfuse_state = imp, "measured"
    queue.add_request(req)
    out.clear()
    nsched.clear()
    call(128)
    assert req.spanfuse_state == "done"
    got = [(g.gap_start, g.gap_start + nsched[g.req_id]) for g in out]
    assert (64, 80) in got and (96, 112) in got and pool.cache_blocks_under_hashes.call_count == 1
