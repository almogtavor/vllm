"""PR #44584 window-aligned KV-tile iteration: correctness + microbenchmark.

Run with WINDOW_ALIGN_MODE in {0,1,2}:
  0 = floor-rounded baseline (upstream)
  1 = 2D pointer path base-shift (PR #44584 / V1)
  2 = 2D + 3D-segmented base-shift (V2)

Loads the patched kernel/helper via sys.modules injection so the rest of the
installed vLLM is untouched.
"""
import os
import sys
import json
import importlib.util

import torch

MODE = int(os.environ.get("WINDOW_ALIGN_MODE", "0"))
PATCH_DIR = "/results/pr44584/patched"
OUT_DIR = "/results/pr44584/out"
os.makedirs(OUT_DIR, exist_ok=True)


def _load(name, fname):
    path = os.path.join(PATCH_DIR, fname)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


import vllm  # noqa: E402
import vllm.v1.attention.ops  # noqa: E402

_load("vllm.v1.attention.ops.triton_attention_helpers", "triton_attention_helpers.py")
kmod = _load("vllm.v1.attention.ops.triton_unified_attention", "triton_unified_attention.py")
unified_attention = kmod.unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

assert kmod._WINDOW_ALIGN_MODE == MODE, (kmod._WINDOW_ALIGN_MODE, MODE)
DEV = "cuda"


def next_pow2(x):
    return 1 << (x - 1).bit_length()


def make_decode_inputs(num_seqs, seq_len, num_q_heads, num_kv_heads, head_size,
                       block_size, seed=0, dtype=torch.bfloat16):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nqpkv = num_q_heads // num_kv_heads
    max_blocks = (seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks + 1
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                            dtype=dtype, device=DEV, generator=g)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                              dtype=dtype, device=DEV, generator=g)
    block_table = torch.zeros(num_seqs, max_blocks, dtype=torch.int32, device=DEV)
    for s in range(num_seqs):
        block_table[s] = torch.arange(1 + s * max_blocks, 1 + (s + 1) * max_blocks,
                                      dtype=torch.int32, device=DEV)
    query = torch.randn(num_seqs, num_q_heads, head_size, dtype=dtype, device=DEV, generator=g)
    cu = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=DEV)
    kv_lens = torch.full((num_seqs,), seq_len, dtype=torch.int32, device=DEV)
    return dict(query=query, key_cache=key_cache, value_cache=value_cache,
                block_table=block_table, cu=cu, kv_lens=kv_lens, nqpkv=nqpkv,
                head_size=head_size, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads,
                block_size=block_size, seq_len=seq_len, num_seqs=num_seqs)


def run_kernel(inp, sliding_window, force_3d, num_seg=16):
    query = inp["query"]
    num_seqs, num_q_heads, head_size = query.shape
    scale = head_size ** -0.5
    out = torch.empty_like(query)
    hsp = next_pow2(head_size)
    thr = num_seqs if force_3d else 0
    so = torch.empty((thr, num_q_heads, num_seg, hsp), dtype=torch.float32, device=DEV)
    sm = torch.empty((thr, num_q_heads, num_seg), dtype=torch.float32, device=DEV)
    se = torch.empty((thr, num_q_heads, num_seg), dtype=torch.float32, device=DEV)
    unified_attention(
        q=query, k=inp["key_cache"], v=inp["value_cache"], out=out,
        cu_seqlens_q=inp["cu"], max_seqlen_q=1, seqused_k=inp["kv_lens"],
        max_seqlen_k=int(inp["kv_lens"].max()), softmax_scale=scale, causal=True,
        window_size=(sliding_window - 1, 0), block_table=inp["block_table"], softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=thr, num_par_softmax_segments=num_seg,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
        kv_quant_mode=KVQuantMode.NONE)
    return out


def ref_decode(inp, sliding_window):
    query = inp["query"]
    num_seqs, num_q_heads, head_size = query.shape
    bs = inp["block_size"]
    bt = inp["block_table"]
    kc, vc = inp["key_cache"], inp["value_cache"]
    nqpkv = inp["nqpkv"]
    scale = head_size ** -0.5
    out = torch.empty(num_seqs, num_q_heads, head_size, dtype=torch.float32, device=DEV)
    for s in range(num_seqs):
        L = int(inp["kv_lens"][s]); p = L - 1
        lo = max(0, p - sliding_window + 1)
        idxs = torch.arange(lo, p + 1, device=DEV)
        blk = bt[s, idxs // bs].long(); slot = (idxs % bs).long()
        K = kc[blk, slot].float()  # (w, num_kv_heads, hs)
        V = vc[blk, slot].float()
        for h in range(num_q_heads):
            kvh = h // nqpkv
            q = query[s, h].float()
            scores = (K[:, kvh, :] @ q) * scale
            w = torch.softmax(scores, dim=0)
            out[s, h] = w @ V[:, kvh, :]
    return out


def bench(inp, sliding_window, force_3d, iters=200, warmup=40, num_seg=16):
    for _ in range(warmup):
        run_kernel(inp, sliding_window, force_3d, num_seg)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        run_kernel(inp, sliding_window, force_3d, num_seg)
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) for s, e in zip(starts, ends))
    return times[len(times) // 2]  # median ms


def main():
    results = {"mode": MODE, "correctness": [], "bench2d": [], "bench3d": []}

    # ---- Correctness (small configs, vs fp32 reference) + save outputs ----
    corr_cfgs = [
        dict(num_seqs=2, seq_len=600, num_q_heads=8, num_kv_heads=4, head_size=128, block_size=16, sw=256),
        dict(num_seqs=2, seq_len=1100, num_q_heads=8, num_kv_heads=4, head_size=128, block_size=16, sw=512),
        dict(num_seqs=2, seq_len=2200, num_q_heads=8, num_kv_heads=4, head_size=128, block_size=16, sw=1024),
        dict(num_seqs=3, seq_len=900, num_q_heads=4, num_kv_heads=2, head_size=128, block_size=32, sw=300),
    ]
    for ci, c in enumerate(corr_cfgs):
        inp = make_decode_inputs(c["num_seqs"], c["seq_len"], c["num_q_heads"],
                                 c["num_kv_heads"], c["head_size"], c["block_size"], seed=100 + ci)
        ref = ref_decode(inp, c["sw"])
        for f3 in (False, True):
            out = run_kernel(inp, c["sw"], f3).float()
            md = (out - ref).abs().max().item()
            tag = f"cfg{ci}_{'3D' if f3 else '2D'}"
            torch.save(out.cpu(), os.path.join(OUT_DIR, f"out_mode{MODE}_{tag}.pt"))
            results["correctness"].append(dict(cfg=ci, path="3D" if f3 else "2D",
                                               sw=c["sw"], max_abs_diff_vs_ref=md))
            print(f"[mode{MODE}] {tag} sw={c['sw']} max|kernel-ref|={md:.4e}", flush=True)

    # ---- Benchmark 2D path (decode batch) ----
    b2_heads = dict(num_q_heads=16, num_kv_heads=8, head_size=128, block_size=16)
    for sw in [256, 512, 1024, 2048, 4096]:
        inp = make_decode_inputs(num_seqs=256, seq_len=2 * sw, seed=7, **b2_heads)
        ms = bench(inp, sw, force_3d=False)
        results["bench2d"].append(dict(sw=sw, seq_len=2 * sw, batch=256, ms=ms))
        print(f"[mode{MODE}] 2D sw={sw} batch=256 median={ms:.4f}ms", flush=True)

    # ---- Benchmark 3D path (few seqs, long context) ----
    for sw in [512, 1024, 2048, 4096]:
        for ns in [1, 4]:
            inp = make_decode_inputs(num_seqs=ns, seq_len=max(8192, 2 * sw), seed=9, **b2_heads)
            ms = bench(inp, sw, force_3d=True)
            results["bench3d"].append(dict(sw=sw, num_seqs=ns, seq_len=max(8192, 2 * sw), ms=ms))
            print(f"[mode{MODE}] 3D sw={sw} ns={ns} median={ms:.4f}ms", flush=True)

    with open(os.path.join(OUT_DIR, f"results_mode{MODE}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[mode{MODE}] DONE", flush=True)


if __name__ == "__main__":
    main()
