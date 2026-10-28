# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SPANS: every token's K and Q are rotated at p - L, L = attn_lb word & LB_MASK
(the span start inside a span, 0 elsewhere), so a span's K is position
independent. FULLQ marks a query row that reads the whole prefix (gap recompute,
SpanFuse probe) while keeping its own span-local L."""

SPANS_FULLQ = 1 << 29
SPANS_LB_MASK = SPANS_FULLQ - 1
