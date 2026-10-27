# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPANS raw-K bit: span tokens keep K un-rotated, every other token is
rotated at its absolute position on write (vllm/v1/spans.py)."""

import torch

from vllm.v1.spans import SPANS_RAW_BIT, split_span_positions


def test_split_span_positions():
    pos = torch.tensor([0, 5, 3 | SPANS_RAW_BIT, 7 | SPANS_RAW_BIT, 9])
    q, k = split_span_positions(pos)
    assert q.tolist() == [0, 5, 3, 7, 9]
    assert k.tolist() == [0, 5, 0, 0, 9]
