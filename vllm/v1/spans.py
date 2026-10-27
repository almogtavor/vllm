# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPANS: only span tokens keep K un-rotated; every other token is stored
rotated at its absolute position, as with spans off. Bit 30 of an attn_lb
word, and of a scheduled token's position, marks a span (raw-K) token."""

import torch

SPANS_RAW_BIT = 1 << 30
SPANS_LB_MASK = SPANS_RAW_BIT - 1


def split_span_positions(
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Q positions, K positions): raw-K tokens rotate K at 0 (identity)."""
    q_pos = positions & SPANS_LB_MASK
    k_pos = torch.where((positions & SPANS_RAW_BIT) != 0, 0, q_pos)
    return q_pos, k_pos
