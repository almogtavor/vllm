# PR #44584 - window-aligned KV-tile iteration: benchmark assets

Reproducible artifacts for the FLOPs / KV-load savings and bit-exactness
analysis of [vllm-project/vllm#44584](https://github.com/vllm-project/vllm/pull/44584)
(window-align the SW / chunked KV-tile iteration) and its follow-up extension to the
3D-segmented decode path.

Measured on an **NVIDIA H100 80GB** (OpenShift, `llm-d-pic`), torch 2.11+cu130,
triton 3.6.0, bf16.

## Method

The kernel/helper were instrumented with an `ALIGN_MODE` constexpr toggling the
SW/chunked KV-tile iteration start:

* `0` - floor-rounded baseline (upstream `main`)
* `1` - 2D pointer path base-shift (PR #44584)
* `2` - 2D **and** 3D-segmented base-shift (follow-up)

All three modes run identical code differing only by the compile-time constant,
on the same GPU, so the comparison is fully controlled.

## Files

* `benchmarks/bench_kernel.py` - correctness vs fp32 reference + first benchmark
* `benchmarks/bench2.py` - cleaner straddle / residue-averaged latency benchmark
* `benchmarks/bitexact.py` - batch-shape invariance test (2D and 3D paths)
* `benchmarks/correctness.py`, `crosscompare.py` - cross-mode equivalence
* `benchmarks/flops_model.py` - analytical tile/FLOPs model + figure generation
* `benchmarks/analyze*.py` - speedup tables
* `data/*.json` - raw measured numbers
* `pr44584_flops_savings.png`, `pr44584_bitexact.png` - figures

## Headline numbers

* Per SW/chunked decode step, the window-aligned start removes **one KV tile**
  (`TILE_SIZE` keys of KV-cache load + the corresponding QK^T/PV MMA) on
  ~`(TILE-1)/TILE` of steps (~97% for the 2D `TILE=32` path), for every standard
  window (a multiple of `TILE_SIZE`). As a share of the SW layer's attention
  it's `TILE/W`.
* Exact FLOPs/KV-load removed at realistic windows (per generated token, summed
  over the model's sliding-window layers, `TILE=32`):
  * **Gemma-3-27B, W=1024**: 3.0% of SW-attention - 26 MFLOP + 12.6 MiB KV/token.
  * **Mistral-7B, W=4096**: 0.8% of SW-attention - 16 MFLOP + 3.9 MiB KV/token.
* Measured kernel latency at those realistic windows is **within the ±1.5% noise
  band**; it only becomes a mover for sub-1k windows (6.8% / 4.9% / 2.0% at
  W=128 / 256 / 512, the gpt-oss-style exception).
* The unconditional win is **bit-exactness**: the 2D base-shift (this PR) makes
  the 2D path byte-identical across batch shapes, and the 3D base-shift extends
  that to the split-KV decode path (both previously bf16-ULP drift).
