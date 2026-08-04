# PIC: why span reuse inflated token counts, and where it actually pays

## The symptom

Position-independent cache (PIC) warms a span prefix-free, then repairs part of it on reuse via a
gap recompute. On SWE-bench the span arms consistently *cost* more than doing nothing: legolink at
gap-128 spent 96K vs 78K recompute tokens, took 14% more LLM calls, 15% more agent steps and 37%
more total tokens. Reuse was buying nothing and charging for it.

## How we measured

We stopped inferring KV quality from SWE-bench resolution rates and measured it directly: an
offline harness assembles the KV cache by hand, so `truth`, `warm` and any repair policy are
compared on identical inputs. The metric is greedy-generation agreement against the true cache;
single-token KL proved far too insensitive on code text. Two controls run before every result --
`truth vs truth` and `repair-all-from-truth` -- and they caught two harness defects that would
otherwise have produced confident nonsense. Under bf16 the repair path has its own error floor
(0.021 Qwen3-32B, 0.102 gemma-4), large enough on gemma-4 to swamp the effect; all headline
numbers are fp32, where the floor is exactly 0.0000.

## Finding 1: the selection signal points the wrong way

Deviation from truth concentrates at the span's **start** -- those blocks lost the prefix they
would have attended. Attention concentrates at the **end**, by recency. The two are anti-correlated
(Spearman -0.68), so Quest attention scoring reliably picks the blocks that are least wrong, and
rewrites them against still-warm predecessors into a state matching neither warm nor truth.

## Finding 2: coverage is a step, not a curve -- on sliding-window models

On gemma-4-31b, half a span is worth almost nothing; only full coverage restores fidelity, and full
coverage costs exactly what recomputing the span costs. There is no operating point: partial repair
at the production budget was indistinguishable from doing nothing while spending the full budget.

## Finding 3: on full-attention models it pays, and cheaply

Qwen3-32B behaves completely differently -- 12% coverage captures essentially the whole achievable
gain, and quadrupling the budget to 50% adds 0.003. That is a correct-enough cache for an eighth of
the prefill. The failure was never the policy; it was the model family every campaign ran on.

## What we shipped, and what is closed

`LegoQuestGapPolicy` now repairs a span end-to-end when the budget covers it and skips an oversized
span on sliding-window models rather than partly rewriting it; it never recomputes more than
`span_aware`. Position within the attention window is *not* the discriminator -- tested and dropped.
Four directions are measured-dead: warming behind sink/generic tokens is strictly worse; a light
transform cannot work (an *oracle* linear map removes only 12% of the residual); per-layer repair
saves nothing since damage grows with depth; and gating on warm-time KV error is hopeless (+0.02).

## The blocker, and where SOTA is

The fork silently corrupts non-gemma-4 models: with spans enabled and zero spans marked, Qwen3-32B
answers "The capital of France is" with " of the of the of the", while the same image with spans off
is coherent. It emits fluent garbage rather than erroring, which put upstream at 40/40 and every
spans arm at 0/40 on SWE-bench. The cause is the deferred key rotation -- `rotary_embedding/base.py`
stops rotating the key under spans, expecting the backend to re-apply it, and only `triton_attn`
does; forcing Triton did not fix it, so the re-application is wrong for Qwen too and that part is
not yet isolated. So every result this project holds comes from the one model where the mechanism
provably cannot pay. The path to SOTA is to make spans model-agnostic, then run PIC on a
full-attention model at ~12% coverage: offline that is an 8x prefill saving at near-perfect
fidelity. Until the fork is fixed that is a strong offline result, not a validated system result,
and should not be reported as one.

---

## Tables

**Coverage is a step on gemma-4-31b** (span 2048, prefix 1024, generation agreement vs truth):

| budget | 0 | 256 | 512 | 1024 | 1536 | 2048 |
|---|---|---|---|---|---|---|
| agreement | .125 | .021 | .135 | .042 | .125 | **1.000** |

**Coverage saturates early on Qwen3-32B** (span 2048, matched offsets):

| offset | warm | 12% | 25% | 50% | 100% |
|---|---|---|---|---|---|
| 0 | 0.354 | 0.990 | 1.000 | 1.000 | 1.000 |
| 1 | 1.000 | 1.000 | 1.000 | 1.000 | - |
| 2 | 0.052 | 0.542 | 0.542 | 0.542 | - |
| 3 | 0.188 | 0.688 | 0.688 | 0.750 | - |

**Partial repair by architecture** (first 256 tokens of the span):

| model | span | n | no repair | repaired | record |
|---|---|---|---|---|---|
| Qwen3-32B (full attention) | 2048 | 8 | 0.4349 | **0.6068** | wins 4, loses 1 |
| gemma-4-31b (SWA 1024) | 2048 | 5 | 0.1292 | 0.1146 | wins 2, loses 2 |
| gemma-4-31b | 1024, inside window | 4 | 0.0911 | 0.0911 | wins 1, loses 1 |

**Full coverage works at several span sizes** (gemma-4, n=3):

| span / budget | coverage | warm | repaired |
|---|---|---|---|
| 256 / 256 | 100% | 0.3889 | **1.0000** |
| 512 / 512 | 100% | 0.3854 | **1.0000** |
| 512 / 256 | 50% | 0.3854 | 0.3923 |
| 1024 / 512 | 50% | 0.0764 | 0.0903 |

**Can a cheap transform replace repair?** Oracle fits, share of residual removed (gemma-4, 60 layers):

| shift | affine | rank-1 | rank-4 | rank-16 | full linear |
|---|---|---|---|---|---|
| 0.3% | -8.7% | 1.4% | 4.7% | 13.0% | 12.1% |

**Harness noise floors** (must be checked before trusting any cell):

| model / dtype | truth vs truth | repair-all-from-truth |
|---|---|---|
| Qwen3-32B fp32 | 0.00000000 | KV err **0.0000** |
| Qwen3-32B bf16 | 0.00000000 | KV err 0.0209 |
| gemma-4 bf16 | 0.00000000 | KV err 0.1017 |
| MiniMax-M2 bf16 | **0.216** (non-deterministic fp8 MoE) | unusable |
