# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fp32 router logits must not depend on how many tokens share the batch."""

import torch
from utils import skip_unsupported

from vllm.model_executor.layers.batch_invariant import init_batch_invariance
from vllm.model_executor.layers.fused_moe.router.gate_linear import (
    fp32_router_gemm_dispatch_impl,
)


@skip_unsupported
def test_fp32_router_gemm_batch_invariant():
    init_batch_invariance()
    torch.manual_seed(0)
    w = torch.randn(256, 3072, device="cuda", dtype=torch.float32) * 0.02
    x = torch.randn(4096, 3072, device="cuda", dtype=torch.bfloat16)
    big = fp32_router_gemm_dispatch_impl(x, w)
    for m in [*range(1, 65), 100, 257, 1000]:
        assert torch.equal(fp32_router_gemm_dispatch_impl(x[:m], w), big[:m]), m
        assert torch.equal(
            fp32_router_gemm_dispatch_impl(x[7 : 7 + m], w), big[7 : 7 + m]
        ), m
