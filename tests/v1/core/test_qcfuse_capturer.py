# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.envs as envs
from vllm.v1.attention.qcfuse import (
    QCFuseImportanceCapturer,
    parse_critical_layers,
)

pytestmark = pytest.mark.spans

BLOCK_SIZE = 16
N_BLOCKS = 8
N_KV_HEADS = 2
QUERIES_PER_KV = 2
HEAD_DIM = 8
CTX = N_BLOCKS * BLOCK_SIZE


@pytest.fixture
def critical_layers(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", "1,3")


def make_capturer() -> QCFuseImportanceCapturer:
    return QCFuseImportanceCapturer(
        max_num_reqs=2,
        max_model_len=CTX,
        block_size=BLOCK_SIZE,
        device=torch.device("cpu"),
    )


def make_kv_cache(num_blocks: int = N_BLOCKS + 2) -> torch.Tensor:
    return torch.zeros(
        (2, num_blocks, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM), dtype=torch.float32
    )


def test_probe_is_a_noop_on_non_critical_layers(critical_layers):
    cap = make_capturer()
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, 4, CTX)])
    cap.capture(
        0,
        torch.randn(4, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM),
        make_kv_cache(),
        QUERIES_PER_KV,
        1.0,
    )
    assert cap.buffer.abs().sum().item() == 0.0


def test_probe_is_a_noop_without_descriptors(critical_layers):
    cap = make_capturer()
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [])
    cap.capture(
        1,
        torch.randn(4, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM),
        make_kv_cache(),
        QUERIES_PER_KV,
        1.0,
    )
    assert cap.buffer.abs().sum().item() == 0.0


def test_unexpected_kv_layout_is_a_silent_noop(critical_layers):
    cap = make_capturer()
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, 4, CTX)])
    cap.capture(
        1,
        torch.randn(4, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM),
        torch.zeros(N_BLOCKS, BLOCK_SIZE, N_KV_HEADS, HEAD_DIM),
        QUERIES_PER_KV,
        1.0,
    )
    assert cap.buffer.abs().sum().item() == 0.0


def test_mass_is_a_softmax_over_context(critical_layers):
    """Total captured mass == num_query_rows * num_kv_heads per critical layer:
    each (kv head, query row) contributes exactly one softmax distribution."""
    cap = make_capturer()
    kv = make_kv_cache()
    kv[0].normal_()
    nq = 5
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, nq, CTX)])
    q = torch.randn(nq, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM)
    cap.capture(1, q, kv, QUERIES_PER_KV, 1.0)
    assert cap.buffer[0, :CTX].sum().item() == pytest.approx(nq * N_KV_HEADS, rel=1e-5)
    assert cap.buffer[1].abs().sum().item() == 0.0


def test_layers_accumulate(critical_layers):
    cap = make_capturer()
    kv = make_kv_cache()
    kv[0].normal_()
    nq = 3
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, nq, CTX)])
    q = torch.randn(nq, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM)
    cap.capture(1, q, kv, QUERIES_PER_KV, 1.0)
    cap.capture(3, q, kv, QUERIES_PER_KV, 1.0)
    total = cap.buffer[0, :CTX].sum().item()
    assert total == pytest.approx(2 * nq * N_KV_HEADS, rel=1e-5)


def test_the_aligned_context_block_wins(critical_layers):
    """A context block whose keys point along the query direction takes almost
    all the mass - the signal QCFusePolicy ranks on."""
    cap = make_capturer()
    kv = make_kv_cache()
    direction = torch.zeros(HEAD_DIM)
    direction[0] = 10.0
    hot_block = 5
    kv[0, hot_block, :, :, :] = direction
    q = torch.zeros(1, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM)
    q[:, :, 0] = 1.0
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, 1, CTX)])
    cap.capture(1, q, kv, QUERIES_PER_KV, 1.0)

    per_block = cap.buffer[0, :CTX].view(N_BLOCKS, BLOCK_SIZE).sum(-1)
    assert int(per_block.argmax()) == hot_block
    assert per_block[hot_block].item() > 0.99 * per_block.sum().item()


def test_begin_step_clears_the_previous_requests_rows(critical_layers):
    cap = make_capturer()
    kv = make_kv_cache()
    kv[0].normal_()
    q = torch.randn(2, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM)
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, 2, CTX)])
    cap.capture(1, q, kv, QUERIES_PER_KV, 1.0)
    assert cap.buffer[0].abs().sum().item() > 0
    cap.begin_step(torch.arange(N_BLOCKS).view(1, N_BLOCKS), [(0, 0, 2, CTX)])
    assert cap.buffer[0].abs().sum().item() == 0.0


def test_block_table_maps_context_positions_to_physical_blocks(critical_layers):
    """Importance must land at logical context positions, not physical slots."""
    cap = make_capturer()
    kv = make_kv_cache()
    physical = torch.tensor([[7, 3, 5, 1, 9, 0, 2, 4]], dtype=torch.int32)
    direction = torch.zeros(HEAD_DIM)
    direction[0] = 10.0
    kv[0, 9, :, :, :] = direction  # physical 9 == logical block 4
    q = torch.zeros(1, N_KV_HEADS * QUERIES_PER_KV, HEAD_DIM)
    q[:, :, 0] = 1.0
    cap.begin_step(physical, [(0, 0, 1, CTX)])
    cap.capture(1, q, kv, QUERIES_PER_KV, 1.0)
    per_block = cap.buffer[0, :CTX].view(N_BLOCKS, BLOCK_SIZE).sum(-1)
    assert int(per_block.argmax()) == 4


def test_knob_defaults_to_off():
    """With the knob unset the layer gate is False everywhere, so the custom op
    is never emitted into the traced graph and the probe cannot allocate."""
    assert envs.VLLM_V1_SPANS_QCFUSE_ENABLE is False
    assert parse_critical_layers() == ()


def test_critical_layers_parse(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_V1_SPANS_QCFUSE_CRITICAL_LAYERS", " 2, 7 ,11 ")
    assert parse_critical_layers() == (2, 7, 11)


def test_custom_op_is_registered_and_inert_without_a_capturer():
    import vllm.model_executor.layers.attention.attention as attn
    from vllm.v1.attention.qcfuse import bind_capturer, get_capturer

    assert hasattr(torch.ops.vllm, "qcfuse_capture_importance")
    assert attn.get_qcfuse_capturer is get_capturer
    bind_capturer(None)
    assert get_capturer() is None
