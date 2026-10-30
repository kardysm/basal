# strix phase 2 on the 8060S: findings (2026-10-28)

Verification of `basal-strix-phase2-padding.patch` (phase 2 of `basal-strix-verified-plan.md`) on the box it was
written for: AMD Radeon 8060S, torch 2.12.0+rocm7.2, Triton 3.7.0, basal-2.0-4.5B, code `strix-warm-shapes` @ 7f4628f
plus the patch.

## Result

| | before (padding between units) | after (every unit in whole key blocks) |
|---|---|---|
| G2: strix 8 in flight vs one at a time, 2,000 decisions | 0 answers differ, **21 probabilities differ** (max \|Δp\| 0.0026): FAIL | 0 answers differ, **0 probabilities differ**: PASS (`--deterministic`, 8 in flight: PASS too) |
| second 8-in-flight run vs one at a time | 0 answers differ, 19 probabilities differ | |
| one at a time, this code vs the patch as written | | 0 answers differ, 28 probabilities differ (trailing padding) |
| captured real batches replayed group by group, alone vs batched | 19 of 351 groups differ | **0 of 351** |
| `tests/test_strix.py` | 41 passed | 73 passed |
| full suite | 834 / 7 / 25 (plan's baseline) | 876 passed / 7 failed / 25 skipped: the 42 new tests; the same 7 CUDA cascade and gemma4 tests fail on 7f4628f |

## What was wrong

The patch pads every unit *after the first* to a multiple of `ATTN["BN"] = 32`, so a unit's keys fall in the same
blocks as alone. It does not cover the unit's **end**. Alone, a unit ends the sequence: its last 8-row query block is
cut at `L` (`q_last = L - 1`), and the rest of its last key block lies past the end (loaded as 0). Batched, the same
query block continues into padding rows (`q_last = q0 + 7`), and the rest of the key block holds real pad tokens.

So two things differ for the unit's last rows:

1. **The mask decision.** The kernel decides per (query block, key block) from `q_last` whether to apply the mask.
   Alone, the state's key blocks run unmasked. Batched, they run masked, because pad rows cannot see the state.
2. **The values of keys masked for those rows** (probability 0) in the unit's last key block.

In exact arithmetic neither can change the result. On this GPU and compiler, (2) does whenever the state blocks run
unmasked. Below is layer 32 of one real request under variant C, where alone and batched both run the state blocks
unmasked, so (1) is equal and only (2) differs (`isolate.py`):

| batched inputs, changed | last row vs alone |
|---|---|
| as they are | 1 value differs |
| cut to the unit (`L = U`) | same |
| pad rows' q zeroed | 1 value differs |
| pad rows' k zeroed | 1 value differs |
| pad rows' v zeroed (only in the unit's last key block) | same |

With a padding row in the last query block (variant B, and the committed code), the state blocks run masked in both
cases. Then the same request's last row stayed identical even with the pad rows' v ×1000. Why masked values leak only
in the unmasked configuration is not established; turning off FMA fusion (variant A) removes 4 of the 19 cases.

Symptom: only the readout at a unit's last token can change, i.e. one question per case (the 4th or 5th in the
typed-decisions layouts), by one bf16 step in one layer, Δp up to 0.003. Decisions never changed in 2,000.

Variants measured on the 351 captured groups:

| variant | groups differing |
|---|---|
| the patch as written | 19 |
| A: attention kernel compiled with `enable_fp_fusion=False` | 15 |
| C: no trailing pad, but the unit's root `sub_end` extended over the padding (keeps today's single-request answers) | 17 |
| B: pad the sequence end to a multiple of `BT = 8` | 0 |
| **every unit padded to a multiple of `BN = 32`, the last one too (committed)** | **0** |

B passes too, but a unit whose length is a multiple of 8 and not of 32 still ends the sequence unpadded alone, the
configuration above. Padding every unit to whole key blocks makes its region (tokens + trailing padding) identical
alone and batched, so the full-length layers compute its rows from the same data in the same blocks by construction.

## Not by construction, and other caveats

- **Last layer.** Readout rows are gathered into 8-row query tiles, and a tile can mix units. A mixed tile only adds or
  drops key blocks that are fully masked for the row. It adds another unit's keys, and it drops blocks whose keys all
  end before the tile's first row. GATHER always takes the masked path. So the last layer rests on one assumption:
  in the masked path, entries masked for a row contribute exactly zero. That held in every measurement (351 real
  batches, the v ×1000 test above, the tiny-model split tests). The one place masked values leaked was the unmasked
  configuration, which the gather path never runs. It is measured, not by construction. Aligning each unit's readout
  rows to whole gather tiles would make it so, as hardening against a compiler change.
- **Single requests change.** Every request now carries up to 31 trailing pad tokens, so one-at-a-time answers are no
  longer byte-identical to 7f4628f (only where the effect above hit; it does not move decisions). `strix.jsonl` from
  phase 1 must be (and was) regenerated with this code.
- **Speed** (measured 2026-10-29, below): single requests 0–2% slower, **32 clients 7.5% slower**, over the plan's
  3% budget. A single request carries up to 31 pad rows: +3% rows on a 715-token case, +60% on a 20-token one.
- The new GPU test `test_first_request_equals_alone` does **not** reproduce the effect on the tiny test model (it
  passes on the old patch too). The guards are `test_a_unit_packs_the_same_alone_and_next_to_others` (first, middle
  and last unit, causal and branched packs) and the exact-size assert in the budget test: 31 of their 34 cases fail on
  the old patch.
- `basal-strix-gates.py` counted per line (400 cases) instead of per decision (2,000), and margins and G2 probability
  checks looked only at the first question; fixed. `basal-strix-verify.sh gates` passed missing files to
  `pairwise.py`, which then crashed the script under `set -e`; fixed.

## Review

Hermes (agentbox, `deepseek` profile, no GPU) reviewed the first version of the two commits on this branch. It
modelled the kernel's per-tile control flow, ran the packing code on 400 random chunks, and checked the gates tool on
fixtures. Its verdict: packing correct, no defect in `basal/strix.py`, six LOW findings:

| # | finding | done |
|---|---|---|
| 1 | the module docstring claims identical blocks for the last layer too | taken: "every full-length layer", plus a clause on gathered readout tiles |
| 2 | the last-layer caveat is too pessimistic: mixed tiles only add or drop fully masked blocks | taken in part: wording above. The full-length layers do not rest on masked-block exactness (identical by construction), so the caveat stays |
| 3 | masked keys "cannot contribute"; only the mask decision flips | the mechanism claim is not taken: the isolate table shows the masked values move the row with the mask decision equal. The paragraph now names both differences |
| 4 | +3% understates the cost (+60% at 20 tokens) | taken |
| 5 | budget test asserts only `<=`; region test covers only a causal last unit | taken: exact size; first / middle / last, causal and branched |
| 6 | `pad_token_id` may be None | not taken: `__init__` sets the pad token from eos, which every basal model has |

## G1 (2026-10-29)

Phase 1 on the test split (2,000 decisions), engine 2.4.1; eager runs one prompt at a time, strix with this branch:

| against | eager bf16 packed | strix | eager bf16 alone | eager fp32 |
|---|---|---|---|---|
| eager bf16 alone | 0.9910 (18 flips) | 0.9900 (20) | reference | 0.9905 (19) |
| eager fp32 alone | 0.9935 (13) | 0.9935 (13) | 0.9905 (19) | reference |

- **G1: S − F = +0.10 pt, F = 0.90%**: the bar is at or below the bf16 floor on this set; phase 5, not phase 3. Every
  flip of strix and eager-packed has a reference margin under 0.05.
- Every bf16 forward is as far from every other as from fp32 (largest |Δp| per decision: median 0.004–0.005, p99
  0.034–0.039; eager-packed vs eager-alone included), so making strix more exact cannot raise its agreement with eager
  bf16 above fp32's 0.9905. Eager-alone's 3 flips against fp32 at margins ≥ 0.05 are tail noise: 7 decisions move by
  more than 0.05 for it, 7 for eager-packed.
- The eager reference runs the math SDPA kernel (`_scaled_dot_product_attention_math`, softmax in fp32, all 60 layers):
  flash and memory-efficient attention are off on gfx1151 unless `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
- PR #51's code (7f4628f, the superseded answer file) scores the same 20 / 13 flips.
- Machine time: eager bf16 16.4 min, eager-packed 15.0, eager fp32 80 (5× bf16). Runs stay one server at a time: the
  GPU is ~99% busy with one, and a second eager server next to fp32 ran at 0.25× its solo speed (answers unchanged).
- A bf16 GGUF (made from the safetensors with the official Q8_0 file's metadata; F32 tensors bit-equal to it) runs in
  `--mode gguf`: 49 / 50 decisions equal to eager bf16 in a smoke test, about eager's speed. Not run on the full split.

**Speed of this branch** (`bench_ab.py`, two servers interleaved, median of 5, old = 7f4628f): S-q1 165 → 166 ms, M-q1
278 → 283, L-q1 753 → 763, XL-q20 3056 → 3123 (0–2% slower); 32 clients 18.3 → 16.9 decisions/s (**−7.5%**). The
padding adds 9.5% rows to the 32-client burst (128 units, median 151 tokens); BN = 16 would add 4.7%, 8-row padding with
unit-relative key blocks in the kernel 2.4%.

## `--near-ties eager` (branch `strix-near-ties`, 2026-10-30)

A question whose calibrated top-two margin under strix is below 0.05 is computed again by the transformers forward on
strix's weights, alone as `--mode eager --soam off --max-batch 1` runs it (`docs/benchmarks/STRIX_HALO.md`):

| check (test split) | result |
|---|---|
| strix `--near-ties eager` vs eager bf16 | **0.9990** (2 flips), 95% 0.9964–0.9997, P(pass on 1,000) = 1.00 |
| EXACT: `--near-tie-margin 1.01` (every decision computed again) vs `eager.jsonl` | 2,000 / 2,000 equal, every probability |
| G2: 8 in flight vs one at a time | 0 probabilities differ |
| vs eager fp32 | 0.9915 (17 flips; strix alone 0.9935) |
| computed again | 114 decisions (5.7%) in 104 of 400 requests |
| server time | ×1.42 one at a time, ×1.39 with 8 in flight; +0.48 s median on a touched request (max 1.2 s) |

The 2 flips left have strix margins of 0.059 and 0.090; one of them (`customer_service_000056`) is a decision where
eager bf16 alone disagrees with fp32 at a margin of 0.107 and strix agrees with fp32.

## Files

| file | what |
|---|---|
| `basal-strix-verify.sh` | phases 0-1 + gates harness (with the pairwise fix); `ties` runs the three near-ties configs |
| `basal-strix-gates.py` | agreement tables, G1 / G2 and EXACT identity checks, per decision |
| `basal-strix-verified-plan.md` | the plan, as written before this verification |
| `results/gates-bf16.json`, `results/gates-fp32.json` | the gate numbers of every run above (answer files: `~/strix-verify/*.jsonl` on the box; the patch-as-written ones are `*.prev-patch-7f4628f-padding-v1`) |
| `results/bench-ab.json` | `bench_ab.py`, 7f4628f vs this branch |
| `experiments/` | `serve_capture.py` (server that records every `run_shared` call), `replay.py` (alone vs batched per group, fix variants), `trace.py` (first differing op), `isolate.py` (which input moves the last row), `rowdep.py`, `exp_split.py`, `exp_real.py` |

This directory is verification tooling, not part of the upstream change: drop the commit that adds it before a PR
to rkinas/basal.
