"""PR #44584 window-aligned KV-tile iteration - FLOPs / KV-load savings model
and figures. Combines the exact analytical tile model with the measured H100
kernel latencies.
"""
import json
import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/pr44584/out"
ASSET = "/tmp/pr44584/assets"
os.makedirs(ASSET, exist_ok=True)

# Tile sizes actually used by the kernel (vllm _get_tile_size):
#   2D pointer path (prefill + large-batch decode): TILE_SIZE_PREFILL = 32
#   3D-segmented path (small-batch decode):          TILE_SIZE_DECODE  = 16
T2D, T3D = 32, 16


def tile_stats(W, T):
    """Average over residues r = first_allowed_key mod T (uniform as the window
    slides one position per decode step)."""
    floor = [math.ceil((r + W) / T) for r in range(T)]
    aligned = math.ceil(W / T)
    avg_floor = sum(floor) / T
    avg_saved = avg_floor - aligned
    frac_saving = sum(1 for f in floor if f > aligned) / T
    return dict(aligned=aligned, avg_floor=avg_floor, avg_saved=avg_saved,
                frac_saving=frac_saving,
                pct_tiles_saved=100.0 * avg_saved / avg_floor)


# Real sliding-window / chunked-attention configs (decode window W in keys).
MODELS = [
    ("gpt-oss (SWA layers)", 128),
    ("Gemma-3 local", 1024),
    ("Phi-3-mini", 2048),
    ("Mistral-7B / Gemma-2 / Llama-4 chunk", 4096),
]

# ---------------------------------------------------------------- analytics
print("=== Per-decode-step tile model (window = multiple of TILE_SIZE) ===")
rows = []
for name, W in MODELS:
    s2 = tile_stats(W, T2D)
    s3 = tile_stats(W, T3D)
    rows.append((name, W, s2, s3))
    print(f"{name:38s} W={W:5d} | 2D(T=32): {s2['avg_floor']:.3f}->{s2['aligned']} "
          f"({s2['pct_tiles_saved']:.1f}% tiles, {100*s2['frac_saving']:.0f}% of steps) | "
          f"3D(T=16): {s3['avg_floor']:.3f}->{s3['aligned']} ({s3['pct_tiles_saved']:.1f}%)")

# absolute FLOPs / KV-bytes per eliminated tile (per Q-block, bf16)
BLOCK_M, HEAD = 16, 128
flops_per_tile_2d = 4 * BLOCK_M * T2D * HEAD          # QK^T + PV MMAs
bytes_per_tile_2d = 2 * T2D * HEAD * 2                # K+V tile, bf16
print(f"\nPer eliminated 2D tile (BLOCK_M=16, head=128, bf16): "
      f"{flops_per_tile_2d/1e3:.0f} kFLOP MMA + {bytes_per_tile_2d/1024:.0f} KiB KV-load per Q-block")

# ---------------------------------------------------------------- load bench
b2 = json.load(open(f"{OUT}/bench2_summary.json"))
be = {m: json.load(open(f"{OUT}/bitexact_m{m}.json")) for m in (0, 1, 2)}

# =====================================================================
# Figure 1: theoretical tiles eliminated + measured 2D latency reduction
# =====================================================================
fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 6))

Ws = np.array([t for t in range(T2D, 8192 + 1, T2D)])  # multiples of 32
axA.plot(Ws, [tile_stats(int(w), T2D)["pct_tiles_saved"] for w in Ws],
         color="#1f77b4", lw=2.2, label="2D path (TILE=32): prefill + large-batch decode")
Ws16 = np.array([t for t in range(T3D, 8192 + 1, T3D)])
axA.plot(Ws16, [tile_stats(int(w), T3D)["pct_tiles_saved"] for w in Ws16],
         color="#ff7f0e", lw=2.2, label="3D path (TILE=16): small-batch decode")
_OFFS = {128: (8, 6), 1024: (10, 22), 2048: (10, 34), 4096: (-20, 30)}
for name, W in MODELS:
    s = tile_stats(W, T2D)
    axA.scatter([W], [s["pct_tiles_saved"]], color="#1f77b4", zorder=5, s=45)
    axA.annotate(f"{name}\nW={W} → {s['pct_tiles_saved']:.1f}%", (W, s["pct_tiles_saved"]),
                 textcoords="offset points", xytext=_OFFS.get(W, (6, 8)), fontsize=8.5,
                 arrowprops=dict(arrowstyle="->", lw=0.6, color="grey"))
axA.set_xscale("log", base=2)
axA.set_xlabel("sliding-window / chunk size  W  (keys)", fontsize=11)
axA.set_ylabel("% of SW/chunked-attention KV-tiles eliminated\nper decode step (avg over window slide)", fontsize=11)
axA.set_title("(A) FLOPs + KV-load saved: ~one KV-tile removed every step\n"
              "savings ≈ (TILE-1)/TILE per step, on " r"$\sim$" "97% (2D) / 94% (3D) of steps", fontsize=11)
axA.grid(True, which="both", alpha=0.3)
axA.legend(fontsize=9, loc="upper right")
axA.set_ylim(0, None)

# Panel B: measured H100 2D kernel latency reduction vs theory
sw_s = [r["sw"] for r in b2["straddle2d"]]
meas_strad = [r["saved_pct"] for r in b2["straddle2d"]]
meas_avg = [r["saved_pct"] for r in b2["avg2d"]]
theory_strad = [100.0 / (math.ceil(s / T2D) + 1) for s in sw_s]  # 1 of (n+1) tiles on a straddle step
axB.plot(sw_s, theory_strad, "k--", lw=1.6, label="theory: 1/(⌈W/T⌉+1) tiles (straddle step)")
axB.plot(sw_s, meas_strad, "o-", color="#1f77b4", lw=2, label="measured: tile-saving step (residue T-1)")
axB.plot(sw_s, meas_avg, "s-", color="#2ca02c", lw=2, label="measured: residue-averaged decode")
axB.axhline(0, color="grey", lw=0.8)
axB.set_xscale("log", base=2)
axB.set_xlabel("sliding-window size  W  (keys)", fontsize=11)
axB.set_ylabel("kernel latency reduction  (%)", fontsize=11)
axB.set_title("(B) Measured 2D-path latency reduction on H100\n"
              "batch=256 decode, bf16, head=128 (16 q / 8 kv heads)", fontsize=11)
axB.grid(True, which="both", alpha=0.3)
axB.legend(fontsize=9)

plt.tight_layout()
fig.savefig(f"{ASSET}/pr44584_flops_savings.png", dpi=130, bbox_inches="tight")
print(f"\nwrote {ASSET}/pr44584_flops_savings.png")

# =====================================================================
# Figure 2: bit-exactness across batch shapes (OLD / 2D / 3D stages x 2D / 3D paths)
# =====================================================================
fig2, ax = plt.subplots(figsize=(7.2, 3.6))
modes = ["OLD\n(baseline)", "2D\n(this PR)", "3D\n(follow-up)"]
paths = ["2D pointer path", "3D-segmented path"]
# 1 = bit-exact (good), 0 = drifts
grid = np.array([
    [1 if be[0]["eq2"] else 0, 1 if be[0]["eq3"] else 0],
    [1 if be[1]["eq2"] else 0, 1 if be[1]["eq3"] else 0],
    [1 if be[2]["eq2"] else 0, 1 if be[2]["eq3"] else 0],
]).T  # paths x modes
ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
for i in range(2):
    for j in range(3):
        d = be[j]["d2"] if i == 0 else be[j]["d3"]
        txt = "bit-exact" if grid[i, j] else f"drifts\nmax|Δ|={d:.1e}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                color="black", fontweight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(modes, fontsize=9)
ax.set_yticks(range(2)); ax.set_yticklabels(paths, fontsize=9)
ax.set_title("Output invariance across batch shapes (same logical Q,K,V window)\n"
             "two decode positions: window fits 1 tile vs straddles a tile boundary", fontsize=10)
plt.tight_layout()
fig2.savefig(f"{ASSET}/pr44584_bitexact.png", dpi=130, bbox_inches="tight")
print(f"wrote {ASSET}/pr44584_bitexact.png")
