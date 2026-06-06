"""Realistic FLOPs / KV-load + measured-latency accounting for 2 real SWA models.
No gpt-oss outlier. Exact FLOPs (deterministic) + measured H100 kernel latency.
"""
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/pr44584/out"
ASSET = "/tmp/pr44584/assets"

# 2D pointer path tile (prefill + large-batch decode). Gemma-3 forces 32 anyway.
T = 32


def avg_tiles_saved(W, T):
    floor = [math.ceil((r + W) / T) for r in range(T)]
    return sum(floor) / T - math.ceil(W / T)   # = (T-1)/T for W % T == 0


CONFIGS = [
    # name, window, q_heads, kv_heads, head_dim, n_sw_layers, params, label
    ("Gemma-3-27B (local layers)", 1024, 32, 16, 128, 52, 27e9, "Gemma-3-27B\nW=1024"),
    ("Mistral-7B-v0.1 (all layers)", 4096, 32, 8, 128, 32, 7.24e9, "Mistral-7B\nW=4096"),
]
GEN = 4096  # tokens generated, for the cumulative column

# measured kernel latency (2D path, straddle = a tile-saving step) from bench2
b2 = json.load(open(f"{OUT}/bench2_summary.json"))
meas = {r["sw"]: r["saved_pct"] for r in b2["straddle2d"]}
meas_avg = {r["sw"]: r["saved_pct"] for r in b2["avg2d"]}

rows = []
print(f"{'config':32s} {'%attn':>6} {'MFLOP/tok':>10} {'MiB-KV/tok':>11} "
      f"{'%model':>7} {'GFLOP/'+str(GEN):>10} {'meas lat':>9}")
for name, W, nq, nkv, hd, L, P, lab in CONFIGS:
    s = avg_tiles_saved(W, T)                       # ~0.969 tiles
    flops_tile = 4 * nq * T * hd                    # QK^T + PV over T keys, per token
    kv_tile = 2 * T * hd * nkv * 2                  # K+V tile, bf16, all kv heads
    fl_layer = s * flops_tile                       # per step per SW layer
    kv_layer = s * kv_tile
    fl_tok = fl_layer * L                           # per generated token (all SW layers)
    kv_tok = kv_layer * L
    pct_attn = 100 * s * T / W                      # of that layer's attention FLOPs/KV
    pct_model = 100 * fl_tok / (2 * P)              # of total model FLOPs/token (2*params)
    gflops_gen = fl_tok * GEN / 1e9
    rows.append(dict(name=name, W=W, lab=lab, pct_attn=pct_attn, mflop_tok=fl_tok / 1e6,
                     mib_kv_tok=kv_tok / 2**20, pct_model=pct_model, gflops_gen=gflops_gen,
                     meas=meas.get(W), meas_avg=meas_avg.get(W), tiles=math.ceil(W / T)))
    print(f"{name:32s} {pct_attn:5.1f}% {fl_tok/1e6:9.1f} {kv_tok/2**20:10.1f} "
          f"{pct_model:6.2f}% {gflops_gen:9.1f} {meas.get(W):+7.1f}%")

# ---------------------------------------------------------------- figure
fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.4))

# Panel A: % of SW-attention FLOPs + KV-load removed per step vs window (TILE=32)
Ws = np.array([t for t in range(T, 8192 + 1, T)])
axA.plot(Ws, [100 * avg_tiles_saved(int(w), T) * T / w for w in Ws], color="#1f77b4", lw=2.3)
for r in rows:
    axA.scatter([r["W"]], [r["pct_attn"]], color="#d62728", zorder=5, s=55)
    axA.annotate(f"{r['lab']} → {r['pct_attn']:.1f}%", (r["W"], r["pct_attn"]),
                 textcoords="offset points", xytext=(8, 10), fontsize=10,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="grey"))
axA.set_xscale("log", base=2)
axA.set_xlabel("sliding-window size  W  (keys)", fontsize=11)
axA.set_ylabel("% of sliding-window-attention FLOPs + KV-load\nremoved per decode step", fontsize=11)
axA.set_title("(A) Work removed per step on a SW layer = TILE/W\n"
              "(one KV tile, on ~97% of steps; TILE=32)", fontsize=11)
axA.grid(True, which="both", alpha=0.3)
axA.set_ylim(0, None)

# Panel B: per-config FLOPs-saved (%) vs measured latency-saved (%)
labels = [r["lab"] for r in rows]
x = np.arange(len(rows)); w = 0.36
flops_pct = [r["pct_attn"] for r in rows]
lat_pct = [r["meas"] for r in rows]
NOISE = 1.5  # H100 measurement noise floor (% latency), from the window sweep spread
b1 = axB.bar(x - w / 2, flops_pct, w, label="attention FLOPs + KV-load removed (exact)", color="#1f77b4")
b2b = axB.bar(x + w / 2, lat_pct, w, yerr=NOISE, capsize=5, ecolor="#444",
              label="kernel latency removed (measured ±noise, H100)", color="#2ca02c")
axB.axhline(0, color="grey", lw=0.8)
axB.axhspan(-NOISE, NOISE, color="grey", alpha=0.12, zorder=0)
for r, xi in zip(rows, x):
    axB.annotate(f"{r['mflop_tok']:.0f} MFLOP + {r['mib_kv_tok']:.1f} MiB KV / token",
                 (xi - w / 2, r["pct_attn"]), textcoords="offset points", xytext=(0, 5),
                 ha="center", fontsize=8.5)
axB.set_xticks(x); axB.set_xticklabels(labels, fontsize=10)
axB.set_ylim(-1.9, 4.4)
axB.set_ylabel("removed per decode step (%)", fontsize=11)
axB.set_title("(B) Exact FLOPs/KV-load saved vs measured latency\n"
              "realistic windows: latency change sits inside the ±1.5% noise band", fontsize=11)
axB.legend(fontsize=8.5, loc="upper right")
axB.grid(True, axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(f"{ASSET}/pr44584_flops_savings.png", dpi=130, bbox_inches="tight")
print(f"\nwrote {ASSET}/pr44584_flops_savings.png")
print("measured latency (avg-residue):", {r["W"]: r["meas_avg"] for r in rows})
