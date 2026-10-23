# SOAM: state once, ask many

Most real requests ask several questions about the same state: the category of a ticket, its urgency, whether it
needs a photo, whether it mentions personal data. In basal-1.0 every question (and each of its two option orders) was
a separate prompt, so the state, usually most of the tokens, was computed again for every question. Five questions
cost about five times one question.

Since basal-1.5 the engine answers **all questions of a request, in both option orders, over one shared state**. The
state is computed once; each question adds only its own text and options.

## How it works

A decision prompt has three parts: a fixed system prompt with the **state**, then the **question**, then the
**lettered options** followed by the answer prefix. The answer is read from the next-token probabilities of the option
letters.

```
[system + state] ── [question 1] ── [options 1, original order] → letters
                 │               └─ [options 1, reversed order] → letters
                 ├─ [question 2] ── [options 2, original order] → letters
                 │               └─ [options 2, reversed order] → letters
                 └─ ...
```

1. **Prefix tree.** All prompts of a request (questions × option orders, and one yes/no branch per label for
   `multi`) form a tree of shared prefixes: the state once at the root, each question's text once for both of its
   orders, the options as leaves. The tree is flattened into **one packed row** of tokens.
2. **Mask and positions.** An attention mask lets every token see only its own ancestors in the tree, never a
   sibling branch, and the position ids of a branch continue from the end of its prefix. Each answer therefore sees
   exactly the tokens it would see as a separate prompt, at the same positions.
3. **Readout.** The engine reads the hidden state at the answer position of every leaf and projects it onto the
   option-letter rows of the output layer only (the full vocabulary is never computed). Both option orders are mapped
   back to the canonical order, averaged, and calibrated per question type.
4. **Two passes.** The fast engine runs the packed row in two passes. Pass 1 computes the root of the tree (the
   system prompt and the state, whatever all prompts of the request share) with plain causal attention, in the same
   CUDA graphs as single prompts, and keeps every layer's keys and values. Pass 2 computes everything below the root,
   all branches of the request as one short row, attending to those keys and values and, through the tree mask, to
   their own ancestors only. The state is computed once however long it is, and the long part runs with the fast
   causal attention kernel; only the short branch row needs a mask.
5. **Batching across requests.** The rows of all requests that wait at the same time are sorted by length and run
   together, at most 12,288 padded tokens per forward pass, and every request is answered as soon as its passes are
   read out.
6. **Sliding-window layers (basal-omni, Gemma 4).** A sliding-window layer lets a token see only the last W tokens of
   its own prompt (W = 512 or 1,024). Positions in the packed row are each token's position in its own prompt, so the
   window is one more condition of the mask: a branch token at position p sees an ancestor at position j only when
   p - j < W, in pass 2 over the root's keys as well. Every answer therefore still sees exactly what it sees as a
   separate prompt. Since every branch comes after the root, pass 2 of a sliding layer reads only the root's last W
   tokens: for a state beyond its largest graph the engine keeps only those.

### Which kernel runs

| unit | pass 1 | pass 2 |
|---|---|---|
| a single prompt (one question asked in one order) | plain CUDA graph up to 8,192 tokens (a forward without a graph beyond), causal attention without a mask | – |
| several branches over one root (option orders, questions, `multi` labels) | the root in the same plain graphs, keys and values kept | the branches, one row per request, over them with the tree mask: a compiled CUDA graph for up to 8 requests of up to 256 branch tokens over roots up to 3,072 tokens, a normal forward otherwise |

A single prompt needs no mask: the padding sits after its last token, so causal attention cannot see it. On an H100 a
1,792-token forward of the 4.5B model takes 35.8 ms in a plain graph and 44.8 ms with a materialised mask (the masked
graphs of earlier releases), so pass 1 is where the time goes and pass 2 adds only the branch tokens.

## Identity guarantee

SOAM changes the arithmetic layout, not the mathematics. Measured on 1,500 answers (development states, 5 questions per
request, 4.5B model, one H100):

| comparison | agreement of the top answer | p99 \|Δp\| | max \|Δp\| |
|---|---|---|---|
| fp32: SOAM vs separate prompts | **1.000** | 7 × 10⁻⁶ | 1.4 × 10⁻⁵ |
| bf16 separate prompts vs fp32 (rounding floor) | 0.991 | 0.030 | 0.054 |
| bf16 SOAM vs fp32 | 0.991 | 0.026 | 0.057 |

- In **fp32** the answers of SOAM and of separate prompts are the same (differences at the level of fp32 rounding).
- In **bf16** (the default `fast` mode) any change of tensor shapes changes the rounding, so near-tie answers can flip
  between any two layouts. SOAM is exactly as close to the fp32 reference as separate prompts are (0.9907 vs 0.9913 in
  a second run on the same weights).
- The basal test suite checks the prefix tree (nested sharing, duplicate prompts, splitting of long rows), the
  row-only readout and the equality of answers with SOAM on and off.

### The paths of this engine

`basal-bench --identity` runs single prompts, both option orders of a question and five-question requests over one
state, alone and batched across requests, and long states, through the engine, and compares every readout with an
eager forward of the same prompt (1,200 readouts per path, synthetic states plus development items):

| comparison | agreement of the top answer | max \|Δp\| |
|---|---|---|
| fp32, every path (single prompts, batched, both orders, SOAM, long states), basal-1.5 | **1.000** | 6.4 × 10⁻⁵ |
| fp32, the same, basal-1.5-mini | **1.000** | 3.3 × 10⁻⁵ |
| bf16, every path against the same prompts computed alone in the engine, basal-1.5 | 0.982–1.000 (decisions 0.995–0.998) | 0.10 |
| bf16, the same, basal-1.5-mini | 0.983–1.000 (decisions 0.992–0.998) | 0.12 |

In fp32 every path gives the answers of separate prompts. In bf16 the batched single prompts already differ from the
same prompts run alone by as much as the shared-state paths do: a change of tensor shapes changes the rounding, and
the disagreements are close calls (the top two options within a few points). The same holds for the `-FP8`
checkpoints. `--deterministic` removes the dependence on other requests: every request then runs in its own forward
passes.

## Measured speed

Latency of one request on an idle server, 4.5B model, one H100, the same requests for both rows (Polish decision
states, median 336 tokens), p50 / p95:

| engine | 1 question | 5 questions | 12 questions |
|---|---|---|---|
| basal-1.0 engine (one prompt per question and order) | 20.1 / 27.1 ms | 111.7 / 182.6 ms | 222.0 / 331.1 ms |
| **SOAM** | 19.0 / 24.3 ms | **37.8 / 53.6 ms** | **80.3 / 103.9 ms** |
| speed-up | – | **3.0×** | **2.8×** |

- SOAM brings no gain for a single question (the 1-question difference above is within run-to-run variation). On a
  second machine (a cloud H100, another 4.5B checkpoint) the same comparison gave 22 / 50 / 93 ms with SOAM against
  22 / 119 / 236 ms without (p50; 2.4× at 5 and 2.5× at 12 questions).
- In the same-node comparison, the server handled 73.4 instead of 57.4 decisions per second with 32 concurrent
  clients, with the same accuracy on the served benchmarks.
- Release check with the final basal-1.5 (4.5B) weights (1,500 decisions): SOAM and separate prompts agree on 1.000
  of answers in fp32 (largest probability difference 3.3e-5), so SOAM is exact in fp32 on these weights too. In bf16
  they agree on 0.993, at the level of bf16 noise (bf16 against fp32 with separate prompts: 0.995); the default `fast`
  mode (bf16 with SOAM) agrees with the fp32 reference on 0.997 of decisions.
- Engine-level latency of the final basal-1.5 weights on the release machine (one H100, one request at a time): 18.3 /
  37.0 / 76.9 ms p50 for 1 / 5 / 12 questions per state (p95 23.7 / 51.6 / 99.4 ms).
  In the HTTP load test on one H100 (a different setup from the table above), one question took 13.7 ms p50 / 23.8 ms
  p95 and the server handled 64.4 decisions/s with 32 concurrent clients. basal-1.0 was not re-measured in that test,
  so it is not a per-question comparison; the gain of SOAM is for requests with several questions.

**Why not more?** At 12 questions the 24 branches (12 questions × 2 option orders) are most of the packed tokens, and
they cannot be shared because each one holds its own options. Asking with one option order halves them: the default
for prompt-format-2 models, which are trained with random option orders; the basal-1.5 models keep two orders by
default (`--orders 1` and `--orders auto` are available, see [FORMATS.md](FORMATS.md#option-orders)). The exact cost of
a request is roughly one state plus, per question, its text once and its options once per order.

## How to use it

Nothing to change: SOAM is on by default (`--soam on`). The two passes are used by the compiled CUDA modes (`fast`,
`fast-nocompile`, `fp8`, `nvfp4`), `fast-exit`, `mps`, `mlx` and `gguf` share the state their own way, and the `eager`
reference path computes every prompt separately (same answers, no speed-up). Put all questions about one state into
one request:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": {"ticket": "Dzień dobry, 12.05 zamówiłam czajnik (nr 88412), przyszedł z pękniętą obudową. Proszę o wymianę albo zwrot pieniędzy. Anna Nowak",
            "customer": {"segment": "VIP", "orders_last_year": 14}},
  "questions": {
    "category":    {"type": "choice", "instructions": "Jaki jest rodzaj zgłoszenia?",
                    "criteria": {"complaint": "Reklamacja towaru", "delivery": "Opóźniona dostawa", "invoice": "Faktura", "other": "Inne"}},
    "needs_photo": {"type": "noul", "instructions": "Czy do rozpatrzenia potrzebne jest zdjęcie uszkodzenia?"},
    "tone":        {"type": "score", "instructions": "Jak zdenerwowana jest klientka?",
                    "criteria": ["spokojna", "zirytowana", "bardzo zdenerwowana"]},
    "issues":      {"type": "multi", "instructions": "Czego dotyczy zgłoszenie?",
                    "criteria": {"damage": "Uszkodzony towar", "exchange": "Wymiana", "refund": "Zwrot pieniędzy", "invoice": "Faktura"}}}}'
```

`usage.branches` in the response counts the question branches of the request (here 1 + 1 + 1 + 4 = 7; each is asked
in both option orders, so 14 prompts), all over one state.

- `--soam off` restores the per-question path of basal-1.0 (one queue entry per question; the two option orders of a
  question still share the state). Use it only to compare.
- The `vllm` and `sglang` modes send one prompt per branch; their own prefix caching (vLLM's automatic prefix caching,
  SGLang's RadixAttention) shares the state between them.
- Early exits (`fast-exit`, models with exit heads): a batch stops early only when all its branches are confident.
