"""Cleaner microbenchmark: measure the per-step tile saving where it actually
occurs - at decode positions whose SW window straddles a tile boundary under
floor-rounding (which is ~ (T-1)/T of all decode steps when the window is a
multiple of TILE_SIZE, i.e. all standard windows 512/1024/2048/4096).

Two regimes:
  straddle  : every seq at residue T-1  -> floor does ceil(W/T)+1 tiles, aligned
              does ceil(W/T). This is the work on a tile-saving step.
  averaged  : seq_lens span all residues mod T -> realistic decode average.

Run per WINDOW_ALIGN_MODE; compares are done across mode JSON files.
"""
import os
import json
import bench_kernel as bk
import torch

DEV = "cuda"
MODE = bk.MODE
HEADS = dict(num_q_heads=16, num_kv_heads=8, head_size=128, block_size=16)


def make_batch(num_seqs, seq_lens, **heads):
    """Decode batch; per-seq seq_len given by seq_lens (list)."""
    nq, nkv, hs, bs = heads["num_q_heads"], heads["num_kv_heads"], heads["head_size"], heads["block_size"]
    nqpkv = nq // nkv
    maxlen = max(seq_lens)
    blocks_per = (maxlen + bs - 1) // bs
    num_blocks = num_seqs * blocks_per + 1
    g = torch.Generator(device=DEV).manual_seed(3)
    key_cache = torch.randn(num_blocks, bs, nkv, hs, dtype=torch.bfloat16, device=DEV, generator=g)
    value_cache = torch.randn(num_blocks, bs, nkv, hs, dtype=torch.bfloat16, device=DEV, generator=g)
    block_table = torch.zeros(num_seqs, blocks_per, dtype=torch.int32, device=DEV)
    for s in range(num_seqs):
        block_table[s] = torch.arange(1 + s * blocks_per, 1 + (s + 1) * blocks_per, dtype=torch.int32, device=DEV)
    query = torch.randn(num_seqs, nq, hs, dtype=torch.bfloat16, device=DEV, generator=g)
    cu = torch.arange(0, num_seqs + 1, dtype=torch.int32, device=DEV)
    kv_lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    return dict(query=query, key_cache=key_cache, value_cache=value_cache, block_table=block_table,
                cu=cu, kv_lens=kv_lens, nqpkv=nqpkv, head_size=hs, block_size=bs, num_seqs=num_seqs)


def call(inp, sw, force_3d, num_seg=16):
    q = inp["query"]; ns, nq, hs = q.shape
    out = torch.empty_like(q); hsp = bk.next_pow2(hs)
    thr = ns if force_3d else 0
    so = torch.empty((thr, nq, num_seg, hsp), dtype=torch.float32, device=DEV)
    sm = torch.empty((thr, nq, num_seg), dtype=torch.float32, device=DEV)
    se = torch.empty((thr, nq, num_seg), dtype=torch.float32, device=DEV)
    from vllm.v1.kv_cache_interface import KVQuantMode
    bk.unified_attention(q=q, k=inp["key_cache"], v=inp["value_cache"], out=out, cu_seqlens_q=inp["cu"],
        max_seqlen_q=1, seqused_k=inp["kv_lens"], max_seqlen_k=int(inp["kv_lens"].max()),
        softmax_scale=hs ** -0.5, causal=True, window_size=(sw - 1, 0), block_table=inp["block_table"],
        softcap=0, q_descale=None, k_descale=None, v_descale=None, seq_threshold_3D=thr,
        num_par_softmax_segments=num_seg, softmax_segm_output=so, softmax_segm_max=sm,
        softmax_segm_expsum=se, kv_quant_mode=KVQuantMode.NONE)
    return out


def timeit(inp, sw, force_3d, iters=400, warmup=80):
    for _ in range(warmup):
        call(inp, sw, force_3d)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); call(inp, sw, force_3d); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return dict(min=ts[0], med=ts[len(ts) // 2], p25=ts[len(ts) // 4])


def main():
    T2D, T3D = 32, 16
    res = {"mode": MODE, "straddle2d": [], "avg2d": [], "straddle3d": [], "avg3d": []}
    SWS = [128, 256, 512, 1024, 2048, 4096]

    # ---- 2D straddle: residue T-1 (every seq saves exactly one tile under align)
    for sw in SWS:
        seq_len = sw + (T2D - 1)  # first_allowed_key = seq_len-sw = T-1 residue
        inp = make_batch(256, [seq_len] * 256, **HEADS)
        t = timeit(inp, sw, False)
        res["straddle2d"].append(dict(sw=sw, seq_len=seq_len, **t))
        print("[m%d] 2D straddle sw=%d  min=%.4f med=%.4f" % (MODE, sw, t["min"], t["med"]), flush=True)

    # ---- 2D residue-averaged: seq_lens span all residues (realistic decode avg)
    for sw in SWS:
        seq_lens = [sw + 16 + i for i in range(256)]  # spans many residues mod 32
        inp = make_batch(256, seq_lens, **HEADS)
        t = timeit(inp, sw, False)
        res["avg2d"].append(dict(sw=sw, **t))
        print("[m%d] 2D avg     sw=%d  min=%.4f med=%.4f" % (MODE, sw, t["min"], t["med"]), flush=True)

    # ---- 3D straddle: residue T-1 mod 16, small batch (the 3D regime)
    for sw in SWS:
        seq_len = sw + (T3D - 1)
        for ns in [2, 8]:
            inp = make_batch(ns, [seq_len] * ns, **HEADS)
            t = timeit(inp, sw, True)
            res["straddle3d"].append(dict(sw=sw, num_seqs=ns, seq_len=seq_len, **t))
            print("[m%d] 3D straddle sw=%d ns=%d  min=%.4f med=%.4f" % (MODE, sw, ns, t["min"], t["med"]), flush=True)

    json.dump(res, open("/results/pr44584/out/bench2_mode%d.json" % MODE, "w"), indent=2)
    print("[m%d] BENCH2 DONE" % MODE, flush=True)


if __name__ == "__main__":
    main()
