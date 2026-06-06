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
  ~`(TILE-1)/TILE` of steps (~97% for the 2D `TILE=32` path, ~94% for the 3D
  `TILE=16` path), for every standard window (a multiple of `TILE_SIZE`).
* Measured 2D-path kernel latency reduction on a tile-saving step:
  **6.8% (W=128), 4.9% (W=256), 2.0% (W=512)**, tracking `1/⌈W/T⌉`.
* The 3D base-shift makes the **3D-segmented** path bit-exact across batch shapes (previously
  bf16-ULP drift) and removes the same leading masked tile.
