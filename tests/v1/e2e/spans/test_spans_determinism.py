# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from .conftest import (
    BLOCK_SIZE,
    LOGPROBS_TOPK,
    build_llm,
    cleanup,
    extract_step0_topk,
    generate_single_output,
    greedy_sp,
)

pytestmark = pytest.mark.spans

MODES = ("FR", "SPANS", "LL-16", "LL-FULL")
SPAN_STARTS_VARIANTS = (None, [0])


def test_all_configs_match_full_recompute(model, monkeypatch):
    """Every (mode x span_starts) configuration produces next-token output
    bit-identical to plain full-recompute (FR).

    Matrix: modes {FR, SPANS, LL-16, LL-FULL} x span_starts {none, [0]}, on a
    tokenized 4-block prompt. One engine is built per mode (span_starts is a
    per-request parameter, not an engine one); each (mode, span_starts) runs
    twice - a cold run and a replay. For the LL modes the replay is a
    prefix-cache hit, so the gap policy actually fires (LL-FULL re-prefills
    the whole prompt, LL-16 the first block); the `cached` check pins that
    the hit happened. FR/SPANS have prefix caching off, so cache nothing.

    None here should change the outputs. Comparison is exact (no drift tolerance):
    generated text, and the full top-K (token order and logprob floats).
    """
    prompt_tokens = list(range(0, BLOCK_SIZE * 4))

    ref_text: str | None = None
    ref_top: list | None = None
    drifts: list[str] = []

    for mode in MODES:
        # One engine per mode: the 4 modes are distinct LLM() configs
        # (gap_policy, enable_prefix_caching - both construction-time).
        # span_starts, by contrast, is a per-request parameter, so both
        # variants run on the same engine.
        llm = build_llm(model, mode, monkeypatch)
        try:
            for span_starts in SPAN_STARTS_VARIANTS:
                extra = (
                    {"span_starts": span_starts}
                    if span_starts is not None else None
                )
                # cold run, then a replay that hits the prefix cache (for LL
                # modes); each yields (text, top-K, num_cached_tokens).
                runs = {}
                for run_label in ("cold", "replay"):
                    out = generate_single_output(
                        llm, prompt_tokens, greedy_sp(extra, logprobs=LOGPROBS_TOPK)
                    )
                    o = out.outputs[0]
                    runs[run_label] = (
                        o.text,
                        extract_step0_topk(o, LOGPROBS_TOPK),
                        out.num_cached_tokens,
                    )
                cold, replay = runs["cold"], runs["replay"]

                if ref_text is None:
                    ref_text, ref_top, _ = cold

                label = (
                    f"{mode}+{'sp[0]' if span_starts is not None else 'no_sp'}"
                )

                # LL modes have prefix caching on: the replay must hit 3 of
                # the 4 blocks (the last is held back), which is what makes
                # the gap policy actually fire. FR/SPANS have caching off.
                replay_cached = replay[2]
                expected_cached = BLOCK_SIZE * 3 if mode.startswith("LL") else 0
                if replay_cached != expected_cached:
                    drifts.append(
                        f"{label}: replay cached={replay_cached}, "
                        f"expected {expected_cached}"
                    )

                for run_label, (text, top, _) in (
                    ("cold", cold), ("replay", replay),
                ):
                    if text != ref_text:
                        drifts.append(
                            f"{label} {run_label}: text drift\n"
                            f"  ref: {ref_text!r}\n  got: {text!r}"
                        )
                    if top != ref_top:
                        drifts.append(
                            f"{label} {run_label}: top-{LOGPROBS_TOPK} drift\n"
                            f"  ref: {ref_top}\n  got: {top}"
                        )
        finally:
            cleanup(llm)

    assert not drifts, "mode equivalence failed:\n\n" + "\n\n".join(drifts)


def test_block_size_constant_matches_conftest():
    """Defensive: tests assume block_size == 16. If conftest changes, tests
    that rely on PIC alignment must be revisited."""
    assert BLOCK_SIZE == 16
