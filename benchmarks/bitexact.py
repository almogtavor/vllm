"""Bit-exactness across batch shapes (mirrors PR #44584's regression test),
extended to both the 2D (TILE=32) and 3D (TILE=16) paths.

Same query + same logical (K,V) window content placed at two kv_lens whose
windows land at different residues mod TILE_SIZE (one fits a single tile, the
other straddles a tile boundary under floor-rounding). A residue-invariant
kernel returns byte-identical output for both. Run per WINDOW_ALIGN_MODE.
"""
import os
import json
import bench_kernel as bk
import torch

DEV = "cuda"


def run_shared(kv_len, sw, force_3d, q_token, shared_k, shared_v,
               num_q_heads=8, num_kv_heads=4, head_size=128, block_size=16, num_seg=16):
    nqpkv = num_q_heads // num_kv_heads
    max_blocks = (kv_len + block_size - 1) // block_size
    num_blocks = max_blocks + 2
    g = torch.Generator(device=DEV).manual_seed(12345)
    key_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                            dtype=torch.bfloat16, device=DEV, generator=g)
    value_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_size,
                              dtype=torch.bfloat16, device=DEV, generator=g)
    block_table = torch.arange(1, 1 + max_blocks, dtype=torch.int32, device=DEV).view(1, max_blocks)
    p = kv_len - 1
    for i, pos in enumerate(range(kv_len - sw, kv_len)):
        blk = int(block_table[0, pos // block_size].item())
        slot = pos % block_size
        key_cache[blk, slot] = shared_k[i]
        value_cache[blk, slot] = shared_v[i]
    query = q_token.clone()
    cu = torch.tensor([0, 1], dtype=torch.int32, device=DEV)
    kv_lens = torch.tensor([kv_len], dtype=torch.int32, device=DEV)
    out = torch.empty_like(query)
    hsp = bk.next_pow2(head_size)
    thr = 1 if force_3d else 0
    so = torch.empty((thr, num_q_heads, num_seg, hsp), dtype=torch.float32, device=DEV)
    sm = torch.empty((thr, num_q_heads, num_seg), dtype=torch.float32, device=DEV)
    se = torch.empty((thr, num_q_heads, num_seg), dtype=torch.float32, device=DEV)
    from vllm.v1.kv_cache_interface import KVQuantMode
    bk.unified_attention(
        q=query, k=key_cache, v=value_cache, out=out, cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=kv_lens, max_seqlen_k=kv_len, softmax_scale=head_size ** -0.5, causal=True,
        window_size=(sw - 1, 0), block_table=block_table, softcap=0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=thr, num_par_softmax_segments=num_seg,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
        kv_quant_mode=KVQuantMode.NONE)
    return out


def main():
    sw = 10
    num_q_heads, num_kv_heads, head_size = 8, 4, 128
    g = torch.Generator(device=DEV).manual_seed(777)
    q_token = torch.randn(1, num_q_heads, head_size, dtype=torch.bfloat16, device=DEV, generator=g)
    shared_k = torch.randn(sw, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEV, generator=g)
    shared_v = torch.randn(sw, num_kv_heads, head_size, dtype=torch.bfloat16, device=DEV, generator=g)

    # 2D path: TILE=32. A: residue 0 (single tile); B: residue 30 (straddle).
    a2 = run_shared(42, sw, False, q_token, shared_k, shared_v)
    b2 = run_shared(40, sw, False, q_token, shared_k, shared_v)
    # 3D path: TILE=16. A: residue 0 (single tile); B: residue 10 (straddle).
    a3 = run_shared(26, sw, True, q_token, shared_k, shared_v)
    b3 = run_shared(20, sw, True, q_token, shared_k, shared_v)

    r = dict(mode=bk.MODE,
             d2=(a2 - b2).abs().max().item(), eq2=bool(torch.equal(a2, b2)),
             d3=(a3 - b3).abs().max().item(), eq3=bool(torch.equal(a3, b3)))
    print("mode %d | 2D: max|A-B|=%.4e equal=%s | 3D: max|A-B|=%.4e equal=%s"
          % (r["mode"], r["d2"], r["eq2"], r["d3"], r["eq3"]))
    with open(os.path.join(bk.OUT_DIR, "bitexact_m%d.json" % bk.MODE), "w") as f:
        json.dump(r, f)


if __name__ == "__main__":
    main()
