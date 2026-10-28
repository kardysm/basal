# Plan: `--mode strix` verified on the current version

*Written 2026-10-28.* Goal: get the AMD Strix Halo section of basal's `HARDWARE.md` from "community-maintained, not yet
verified on our hardware" to verified. The maintainers decide that by re-running their test, and the bar is **≥ 99.5%
of answers equal to `--mode eager` bf16** (their 2.5 test: 99.2% of 1,000 prompts). This plan makes strix as likely as
possible to pass, and produces the evidence they need when it cannot.

## Current version

| what | where |
|---|---|
| upstream main | `v2.4.1` (`b20f0f9`); 2.5 is not tagged yet |
| strix as it goes into 2.5 | branch `strix-warm-shapes` @ `7f4628f` = 2.4.1 + the start-up warm-up; worktree `~/projects/basal-strix-warm` |
| 2.5 additions we don't have yet | the C-compiler note in `HARDWARE.md`; the GPU wait spinning for the first 0.3 s (`_wait`). Neither changes the numbers strix computes |
| live server on this box | `:8022`, basal **2.2.0** from `~/basal-env`, `--mode strix` (basal-ask uses it): leave it alone |

Run the current version from the worktree with the existing env (checked: imports as 2.4.1):

```bash
export PYTHONPATH=$HOME/projects/basal-strix-warm
PY=$HOME/basal-env/bin/python
$PY -c "import basal; print(basal.__file__, basal.__version__)"   # must print the worktree path, 2.4.1
```

## What we know

**1. On public data, strix is as close to eager as eager is to itself.** Engine 2.2, basal-2.0-4.5B, 2,000 decisions
of `LocalLLaMA/typed-decisions` (test split). Reference: `--mode eager` bf16 with every prompt alone. Raw files on
branch `strix-parity-data`; recomputed with `tools/strix/pairwise.py`:

| against the reference | agreement | 95% interval | flips |
|---|---|---|---|
| eager bf16, default packing | 0.9910 | 0.9858–0.9943 | 18 |
| strix, one request at a time | 0.9900 | 0.9846–0.9935 | 20 (9 shared with eager-packed) |
| strix, 8 requests in flight | 0.9885 | 0.9828–0.9923 | 23 |
| gguf Q8_0 | 0.9770 | 0.9695–0.9827 | 46 |
| *strix one at a time vs strix 8 in flight* | *0.9935* | | *13 differ* |

Every flip of every bf16 path is a near-tie: at a reference top-2 margin ≥ 0.05 (1,888 of 2,000 decisions), eager
packed and strix agree 100%.

**2. strix is not batch-invariant, and the cause is attention alone.** `tools/strix/probe_batch_invariance.py`, run
today on this 8060S with the 2.4.1 + warm-up code:

| kernel | same rows alone vs inside a bigger forward |
|---|---|
| qkv, o, down projections; FFN input (gate / up / SwiGLU) | bit-identical at every row count (600–5,000) and offset, across all tile-config ranges |
| attention, request starting at a multiple of 32 in the packed sequence | bit-identical |
| attention, any other start (8, 37, 100, 1,000 tokens in) | 11–16% of outputs differ by one bf16 step |

The attention kernel walks keys in blocks of `ATTN["BN"] = 32` aligned to the whole packed sequence
(`basal/strix.py:65`). A request packed after others (`run_shared`, `off += len(t)` at `strix.py:482`) gets its keys
split into different blocks. That changes the running max that `p` is scaled by before it is rounded to bf16 for the
`p @ v` product. One request alone always sits at offset 0, so `--mode strix` with one request at a time (or
`--deterministic`) is already invariant. Any batching is not: server concurrency, and `basal-bench`, which runs 32
decisions per forward.

**3. On 1,000 prompts, 99.5% leaves no room.** 99.5% means at most 5 flips. 992/1,000 has a 95% interval of
98.4–99.6%.

| true flip rate | P(pass) on 1,000 | on 2,000 | on 5,000 |
|---|---|---|---|
| 1.0% (strix, public set) | 0.07 | 0.01 | 0.00 |
| 0.8% (strix, maintainers' run) | 0.19 | 0.08 | 0.01 |
| 0.5% | 0.62 | 0.58 | 0.55 |
| 0.3% | 0.92 | 0.96 | 0.99 |

To pass reliably, strix needs a true flip rate of about 0.3% against their reference. On public data, eager's own
packing scores 0.9%.

**4. Making strix more exact has a ceiling.** If strix's rounding errors and eager's are independent, their difference
grows as √(σ_strix² + σ_eager²). A strix with zero error would still disagree with eager bf16 as often as fp32 does. It
would remove at most ~30% of the flips: 0.8% → ~0.57%, a pass chance of ~0.4 on 1,000. So kernel work can close the
gap only if, on the maintainers' set, eager bf16 itself sits close to fp32.

Two separate problems, then:

- **A. Batch dependence.** A strix defect with a known cause and a small fix (phase 2). Fix it whatever happens to the
  bar.
- **B. The bf16 floor.** Whether any packed bf16 engine can reach 99.5% against eager-alone on the maintainers' set.
  This has to be measured (phase 1). It decides whether phases 3–4 can work, or whether the result is an evidence
  package (phase 5).

## Inputs needed from the maintainers

The plan runs without these but gets sharper with them. Phase 1 covers both likely harnesses meanwhile.

- [ ] Prompt set and harness: `basal-bench --questions <file> --n 1000` (32 decisions per forward), or HTTP requests
  (concurrency?).
- [ ] Reference command: `--mode eager` default packing, or `--soam off --max-batch 1`; GPU, torch and ROCm versions.
- [ ] The 8 flips: id, reference top-2 margin, strix top-2 margin.
- [ ] How stock paths score on the same set: eager bf16 packed vs the reference, `--mode fast` bf16 (CUDA) vs eager
  bf16.
- [ ] Whether the 99.5% bar is new: `HARDWARE.md` marks vLLM (0.994 vs fp32), llama.cpp / Ollama (0.994) and MLX 8-bit
  (0.992 vs bf16) as verified.

## Phase 0: setup (½ day)

1. `git -C /run/media/kardysm/4TB/projects/basal fetch upstream --tags`. If `v2.5.0` exists, rebase the work below onto
   it (see phase 6).
2. Cases from the public set (test 400 cases / 2,000 decisions; train 1,200 / 6,000), with `parity.py` from the fork:
   ```bash
   mkdir -p ~/strix-verify && cd ~/strix-verify && git -C /run/media/kardysm/4TB/projects/basal show strix-parity-data:docs/benchmarks/parity.py > parity.py
   SNAP=~/.cache/huggingface/hub/datasets--LocalLLaMA--typed-decisions/snapshots/7189f46b02248bd2569df7325c9df8e3b24e90a7/all
   uv run --no-project --with pyarrow python parity.py cases $SNAP/test-00000-of-00001.parquet cases-test.jsonl
   uv run --no-project --with pyarrow python parity.py cases $SNAP/train-00000-of-00001.parquet cases-train.jsonl
   ```
3. Check basal-2.0's model card for whether `typed-decisions` train was in its training or calibration data. If so,
   its decisions are more confident, there are fewer near-ties, and agreement looks better. Report train numbers
   separately and never as the headline.

## Phase 1: baseline and noise floor on the current version (1 day, mostly machine time)

One server at a time on `127.0.0.1:8031` (2.2-era times on this box: eager bf16 alone ~15 min per 2,000 decisions,
strix ~2 min, fp32 eager ~2–3× bf16):

```bash
M=~/.cache/huggingface/hub/models--Remek--basal-2.0-4.5B/snapshots/e5ba9149760d99b73b1963d8e0ccf948b58afacb
bserve() { $PY -c 'from basal.server import main; main()' --model $M --name basal-2.0-4.5B --host 127.0.0.1 --port 8031 "$@"; }
brun() { $PY -c 'from basal.run import main; main()' --url http://127.0.0.1:8031/v1/systemone "$@"; }
# e.g.  bserve --mode eager --soam off --max-batch 1   then   brun --input cases-test.jsonl --output eager.jsonl --concurrency 1
```

| answer file | server flags | `basal-run --concurrency` |
|---|---|---|
| `eager.jsonl` (reference) | `--mode eager --soam off --max-batch 1` | 1 |
| `eager-fp32.jsonl` (second reference) | `--mode eager --dtype float32 --soam off --max-batch 1` | 1 |
| `eager-packed.jsonl` | `--mode eager` | 1 |
| `strix.jsonl` | `--mode strix` | 1 |
| `strix-c8.jsonl` | `--mode strix` | 8 |
| `strix-det-c8.jsonl` | `--mode strix --deterministic` | 8 |

Run all six on the test split. Add the train split for the reference, eager-packed, strix and strix-c8 only, for a
tighter interval. Also log which SDPA kernel the eager reference actually runs on gfx1151 (flash, efficient or math;
`torch.profiler` kernel names over one request). Then:

```bash
PW=/run/media/kardysm/4TB/projects/dangerous/basal-systems/tools/strix/pairwise.py
python3 $PW eager.jsonl eager-packed.jsonl strix.jsonl strix-c8.jsonl strix-det-c8.jsonl --out pw-bf16.json
python3 $PW eager-fp32.jsonl eager.jsonl eager-packed.jsonl strix.jsonl --out pw-fp32.json
```

If the maintainers confirm `basal-bench`, also run `basal-bench --model $M --modes eager strix --questions <their file>
--n 1000` (and with `--modes eager-fp32 eager strix`).

**Gate G1.** F = eager-packed's flip rate, S = strix's (one at a time), both against `eager.jsonl`:

| outcome | meaning | next |
|---|---|---|
| S ≥ F + 0.3 pt | strix has error of its own beyond bf16 noise | phase 2, then phase 3 is the main path |
| S ≈ F, F ≥ 0.4% | the bar is at or below the bf16 floor on this set: no packed engine passes reliably | phase 2, then phase 5 (evidence); phase 4 only if the maintainers' floor turns out lower |
| S ≈ F, F < 0.3% | strix already passes most of the time | phase 2, then phase 5 |

Also check: `strix-det-c8` must equal `strix` decision for decision (one request per forward = offset 0). If it doesn't,
something other than attention offsets is batch-dependent; find it before phase 2.

## Phase 2: batch-invariant strix (1–2 days)

**Change** (`basal/strix.py`, `StrixHaloFastBackend.run_shared` and `_chunks_tok`):

1. Before appending each unit after the first in a chunk, pad `off` to a multiple of `ATTN["BN"]`. Pad tokens: the
   tokenizer's pad id, position 0, `sub_end = own index + 1` (visible only to itself, so no empty softmax), never in
   `reads`. At most `BN − 1` = 31 tokens per unit.
2. Count the padding in `_chunks_tok`'s `TOKEN_BUDGET` check (`n + t + BN − 1 > budget`).
3. A unit alone stays at offset 0, so one-request-at-a-time answers are bit-identical to today. Only batched answers
   change, and they then equal the answers the request gets alone. A request over 16,384 tokens is bisected into
   several units even when alone; its later units move to aligned offsets, so its answers change slightly too.

If padding costs too much at 32 clients, the fallback is in the kernel: pass each query block's unit start, and start
the key loop there instead of at 0. That is more invasive. Try padding first.

**Tests** (`tests/test_strix.py`):

- [ ] Kernel: attention of a tree alone vs after a tree of P ∈ {1, 8, 31, 33, 100} tokens, `torch.equal`, full rows and
  readout-only rows (the probe, made a test).
- [ ] Backend, tiny random Llama: a request's probabilities alone vs in one `run_shared` call after 1–3 other requests
  of lengths not divisible by 32, `torch.equal`.
- [ ] Every existing test still passes: 31 in `test_strix.py`, including the warm-up test (no kernel compile for 1–5,000
  rows; padding changes row counts, but the warm-up covers row-count classes, not values). Then the full suite: same
  pass/fail set as before (834 / 7 / 25 at 2.4.1 + warm-up).

**Speed:** `bench_ab.py` from the fork (S / M / L / XL, 1–20 questions, 32 clients), new code vs the current code,
interleaved, median of 5. Expect 0% on single requests (no padding). Accept ≤ 3% on 32-client throughput. If it costs
more, try `BN = 16` for attention (half the padding; re-tune) before the kernel fallback.

**Gate G2:** strix with 8 in flight agrees with strix one at a time on 2,000 / 2,000 decisions, with identical
probabilities (`pairwise.py strix.jsonl strix-c8.jsonl`).

Expected effect on the bar: batched runs get the one-at-a-time result. On public data that is 0.9885 → 0.9900, inside
the noise. It matters most if the maintainers measured with `basal-bench` or with concurrency.

## Phase 3: where strix and eager part (2 days; the main path only if G1 says strix has its own error)

1. A per-layer diff tool. For every flip from phase 1 plus 50 random non-flips, run the prompt alone through eager bf16,
   eager fp32 and strix, and hook the residual stream after every layer. Plot the relative error against fp32 per layer
   for both bf16 engines, next to eager-packed vs eager-alone for scale.
2. Op by op, on real activations from layers 0, 30 and 59: each strix kernel, the stock op (`F.linear` + add; the SDPA
   kernel eager really uses on gfx1151) and an fp32 op on the same bf16 inputs. Error of each against fp32.
3. **Gate G3:** a strix op whose error against fp32 is ≥ 1.5× the stock op's is a bug: fix it and rerun phase 1 for
   strix. If there is none, strix is at the bf16 floor: stop kernel work.

## Phase 4: more exact strix (optional, 2–3 days)

Only if G1 shows the floor on the maintainers' set is close to the bar (F ≤ 0.4%) and G3 found no bug. Ceiling: about
30% fewer flips (see "What we know" §4).

- **Attention:** split `p` into a bf16 high part and a bf16 low part, and compute `p @ v` as two dots. That gives
  near-fp32 attention output for about 2× the `p @ v` cost. Attention was 2.5–28.5% of a `gguf` forward by length;
  measure strix's share first.
- **Residual stream in fp32** between layers. This departs from "rounding follows the Hugging Face bf16 forward", which
  the docs promise, so it goes behind a flag unless it is better against both eager bf16 and fp32.

Measure each on the test split against `eager.jsonl` and `eager-fp32.jsonl`, plus `bench_ab`. Keep it only if it cuts
flips against both references and costs ≤ 5%.

## Phase 5: evidence and handoff (1 day)

1. Results in the fork, like the 2.2 run: `docs/benchmarks/results/2.5/parity/` with every answer file, `pw-*.json`,
   the probe output, bench_ab before / after, and a README with machine, versions (torch 2.12 + ROCm 7.2, Triton 3.7)
   and commands.
2. PR to `rkinas/basal`: the phase 2 change and its tests; `STRIX_HALO.md` "Parity" section
   with the floor table, strix one-at-a-time = batched, flips only at margins < 0.05, accuracy and ECE unchanged. Don't
   touch the "not yet verified" wording in `HARDWARE.md`: the maintainers change that.
3. The evidence states how strix compares with eager's own packing on the same set. Whether the bar is measured against
   that floor, against fp32 (as for vLLM and SGLang), or on more decisions with an interval is the maintainers' call.

## Phase 6: when 2.5 is tagged (½ day)

Rebase onto `v2.5.0` (their `_wait` change touches the same file), rerun `tests/test_strix.py` and the full suite,
rerun G2 and phase 1 for `strix` and `strix-c8` on the test split, and update the numbers in the PR.

## Done when

- **Ours:** G2 passes; on ≥ 2,000 test decisions, strix one at a time has a flip rate against the reference ≤ 0.3%
  (upper end of the 95% interval ≤ 0.5%). Or, if G1 put the floor above the bar, the evidence package shows strix
  within it.
- **Theirs:** the maintainers re-run their test on their Strix Halo and mark the path verified.

## Risks

| risk | effect | handling |
|---|---|---|
| floor above the bar on their set | no kernel work passes reliably | phase 1 measures it first; phase 5 hands them the numbers |
| their reference differs (packing, SDPA kernel, torch version) | our floor is not theirs | inputs list; run both eager variants; pin versions in the report |
| train split seen in training | agreement looks better than it is | headline on the test split only |
| padding slows 32-client throughput | slower under load | measure; `BN = 16`; kernel fallback |
| Triton / ROCm update changes the kernels' instruction choice | bit-identity claims are per version | the tests check invariance on every run, not stored values |
| live `:8022` server on 2.2.0 shares the GPU | timing noise in bench runs | numbers aren't affected; stop nothing; bench when basal-ask is idle |

## Files

| file | what |
|---|---|
| `tools/strix/probe_batch_invariance.py` | kernel-level batch-invariance probe (today's result in its docstring) |
| `tools/strix/pairwise.py` | pairwise agreement, Wilson intervals, by-margin bands, shared flips, P(pass) |
| fork branch `strix-parity-data` | 2.2 answer files and `parity.py` |
| `~/projects/basal-strix-warm` | worktree of `strix-warm-shapes`, the code under test |

Time: 6–9 working days to phase 5 (phase 4 optional), most of it machine time in phase 1.
