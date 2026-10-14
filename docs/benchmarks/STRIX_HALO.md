# basal-1.5 on AMD Strix Halo: benchmarks and tests

**PRE** is what the machine served before this work: `llama-server` in a Vulkan container (Mesa 25.3) behind
`basal-serve --mode llamacpp`, and the graph modes with upstream defaults. **POST** is what it serves now:
`basal-serve --mode gguf-strix`: llama.cpp in process (Vulkan, Mesa 26.2) with the cached prompt start of this branch.
`gguf-strix` is a mode of its own, chosen explicitly; `--mode gguf` and every other mode behave as on `main`.

Machine: AMD Ryzen AI Max+ PRO 395, Radeon 8060S (gfx1151, 40 CUs), 128 GB unified memory; Fedora 44, kernel 7.2.8.
Model: basal-1.5 (4.5B); `Q8_0` GGUF for the llama.cpp engines, bf16 for PyTorch. Software: PyTorch 2.12.0 + ROCm 7.2,
transformers 5.17, llama-cpp-python 0.3.36 (Vulkan build), llama.cpp ROCm 7.2.4 and Vulkan containers
(kyuz0/amd-strix-halo-toolboxes). Every engine was measured after the GPU cooled below 55 °C: back to back, the second
run came out 4–20% slower.

**Re-run on 2026-10-14 (`gguf` and `gguf-strix`, commit bd8887f).** The first run (a378c25) came before the
code-review fixes of the prefix cache and before `gguf-strix` became a mode of its own; its `gguf` numbers were
`gguf-strix` with the cache off. Both engines were measured again on the final code, as `--mode gguf` and
`--mode gguf-strix` (request matrix and edge cases; start-up: median of three warm starts). Every answer is
bit-identical to the first run (all 152 matrix questions and all 12 edge requests, max |Δp| 0.0000). Latencies came out
0–4% lower and throughput 5–9% higher for both engines, so `gguf-strix` keeps the same lead over `gguf`. GTT is now
the memory the server adds (`mem_info_gtt_used` before the start and at `/health`; it does not grow under load):
5.8 GiB for both. The first run reported 6.5 GiB, measured in a way that wasn't recorded. The other engines and the
two summary tables below (bundled examples) were not re-run.

## Summary: PRE → POST

The bundled examples (44 single questions, the five `complex.jsonl` requests, their 14 questions over one state, a
~2,400-token state with 1 and with 5 questions); p50 per request, unique states, HTTP.

| | **PRE** llama-server Vulkan | **POST** `gguf-strix` | change |
|---|---|---|---|
| one question | 256 ms | **119 ms** | 2.2× faster |
| five complex requests | 710 ms | **314 ms** | 2.3× faster |
| 14 questions over one state | 3.61 s | **1.27 s** | 2.8× faster |
| long state, 1 / 5 questions | 1.90 / 3.15 s | **1.36 / 1.82 s** | 1.4× / 1.7× faster |
| 32 concurrent clients | 4.3 decisions/s | **10.1 decisions/s** | 2.3× more |
| start-up / GPU memory | ~5 s / ~6.5 GiB | 1–4 s / 6.3 GiB | – |

### Step by step (each row adds to the one above it)

| stage | engine | one question | 5 complex | 14 questions | long 1 q / 5 q | 32 clients | start-up |
|---|---|---|---|---|---|---|---|
| **PRE** | llama-server, Vulkan (container, Mesa 25.3) | 256 ms | 710 ms | 3.61 s | 1.90 / 3.15 s | 4.3/s | ~5 s |
| | llama-server on ROCm 7.2.4, `-fa on -ub 2048` | 166 ms | 475 ms | 2.15 s | 1.49 / 2.30 s | 6.1/s | ~4 s |
| | `--mode gguf`: llama.cpp in process (Vulkan, host Mesa 26.2) | 143 ms | 357 ms | 1.27 s | 1.38 / 1.86 s | 7.8/s | 1–4 s |
| **POST** | `--mode gguf-strix`: + cached prompt start | **119 ms** | **314 ms** | **1.27 s** | **1.36 / 1.82 s** | **10.1/s** | 1–4 s |

The PyTorch graph path, PRE (`--mode fast-nocompile`, upstream defaults) → POST (`--mode fast-nocompile-strix` of
this branch with AOTriton attention and TunableOp, see below):

| stage | graph mode | one question | 5 complex | 14 questions | long 1 q / 5 q | 32 clients | start-up / GTT |
|---|---|---|---|---|---|---|---|
| **PRE** | `fast-nocompile`, rocBLAS, math attention ¹ | 250 ms | 496 ms | 3.69 s | 3.69 / 10.1 s | 4.7/s | 888–942 s / 31–32 GiB |
| | + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | 217 ms | 368 ms | 1.57 s | 1.57 / 3.94 s | 6.3/s | 60 s / 12.7 GiB |
| | + TunableOp (rocBLAS solutions) | 154 ms | 338 ms | 1.48 s | 1.49 / 3.74 s | 7.0/s | 55 s / 12.7 GiB |
| **POST** | + captured lengths up to 8,192 (= `fast-nocompile-strix`) | 155 ms | 342 ms | 1.48 s | 1.47 / 2.22 s | 6.9/s | 76 s / 14.5 GiB |

¹ latencies measured with 4,096 tokens per forward and one warm-up forward (same kernels as the defaults, shorter
start-up).

## Method

**Request matrix** (`bench_matrix.py`): four states built from the bundled examples, S / M / L / XL ≈ 100 / 570 / 2,000 /
6,060 tokens, each with 1, 2, 5, 10 and 20 questions from a fixed mixed pool (choice, yes/no, score, an `act` with costs
and a 4-label `multi`, so 20 questions are 23 branches and 46 prompts). Pass 0 keeps the answers; two timed passes put a
marker in front of the state so no engine can answer from a prompt cache (the fixed system-prompt start stays shared,
as in real traffic). Concurrency: single questions over short states, 48–256 requests per level.

**Reference**: `basal-serve --mode eager --dtype float32 --soam off --max-batch 1`, every prompt computed on its own in
fp32 (S, M and L with all question counts, XL with 1 and 2; XL with more questions takes minutes per request in fp32).

**Edge cases** (`edge_check.py`): 12 requests (mixed PL / EN questions over one state, 10 options, the same question
three times, a JSON state, `facts: auto`, long option texts, a 5-level score, an 8-label `multi` with `min` / `max`, an
`act` with a cost matrix, English only, a 2-character state, `option_keys` show vs hide), sent one after another and
then all at once (the server batches them).

**Engines** (all `basal-serve` on one port; GGUF engines read tokenizer and calibration from the `-GGUF` folder):

| engine | server |
|---|---|
| **PRE** llama-server Vulkan | `llama-server -m basal-1.5-4.5B-Q8_0.gguf -ngl 99 -fa on --no-mmap -c 8192 -np 1` (Vulkan build, Mesa 25.3) + `--mode llamacpp` |
| llama-server ROCm | the same with `-dev ROCm0 -ub 2048` (ROCm 7.2.4 build) |
| **PRE** `fast-nocompile` | `--mode fast-nocompile` (upstream defaults) |
| `fast-nocompile-strix` | `--mode fast-nocompile-strix` (or `fast-strix`; or `--soam off`) with the environment below ³ |
| **POST** `gguf-strix` | `--mode gguf-strix --gguf basal-1.5-4.5B-Q8_0.gguf`, llama-cpp-python built with `-DGGML_VULKAN=on` |
| `gguf` | `--mode gguf --gguf basal-1.5-4.5B-Q8_0.gguf` (same build) ² |

² Measured as `--mode gguf` in the re-run (see above). The first run used the cache of `gguf-strix` turned off
(`prefixes=[]`), which decodes the same tokens in the same `llama_decode` calls; the answers of the two are identical.

Graph environment: `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 PYTORCH_TUNABLEOP_ENABLED=1
PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 PYTORCH_TUNABLEOP_TUNING=0 PYTORCH_TUNABLEOP_FILENAME=<file>`, after one tuning run
with `PYTORCH_TUNABLEOP_TUNING=1` in a process that exits normally (the file is written at exit; about 4 minutes).

³ Measured as `--mode fast-nocompile` / `fast` with `BASAL_LENS=128,…,3072,3584,4096,5120,6144,8192
BASAL_TOKEN_BUDGET=4096 BASAL_WARMUP=1`, the shapes that are now the defaults of the `-strix` graph modes (the
`BASAL_*` variables apply to those modes only).

Reproduce: start an engine, then

```bash
python docs/benchmarks/bench_matrix.py --url http://127.0.0.1:8000/v1/systemone --label gguf-strix
python docs/benchmarks/edge_check.py --url http://127.0.0.1:8000/v1/systemone --label gguf-strix
python docs/benchmarks/report.py results/matrix results/edge > report.md
```

## Tests

- `tests/test_graphs.py` (new) runs the real start-up of `StrixHaloGraphBackend` (the `-strix` graph modes) with a
  tiny random Llama on the GPU: the shapes captured under the `BASAL_*` overrides, 1–5 questions over one state, several requests in one replay of a batch
  graph (and the same answer as alone), a request bisected because its packed row is longer than the largest captured
  length, a prompt longer than every shape run eagerly, deterministic replay. Every readout equals a separate fp32
  forward of its prompt. Passes with the default attention and with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1`.
  The tests fail when the attention mask lets sibling branches see each other (6 of 9) or when positions do not
  restart per branch (5 of 9).
- `tests/test_gguf_strix.py` (new; `tests/test_gguf.py` is unchanged): rows of `--mode gguf-strix` that begin with
  either cached prefix, and rows that do not, equal a full decode, also on a second round after the first chunk's
  readout sequences were removed; with the cache off it equals `gguf`. Plain `gguf` keeps its decode: a context of
  N_CTX cells and N_SEQ sequences, the whole memory cleared per chunk, no sequence copied or removed. Regression tests
  from the code review (each fails on the first version of the cache): a row may add N_CTX tokens after a cached
  prefix (the prefixes have cells of their own); readouts never reach the prefix sequences when `N_SEQ` is raised
  after start-up (the first version aborted in llama.cpp); a failure half-way through sharing the prefix cells leaves
  no stale cells behind. `basal-serve` / `basal-bench` select `GGUFBackend` for `gguf` and `StrixHaloGGUFBackend`
  only for `gguf-strix`.
- `tests/test_graphs.py` also checks the `BASAL_*` validation (empty means default, `BASAL_BATCHES` must include 1,
  positive values only), that `fast` / `fast-nocompile` (all 81 default shapes) and `mps` ignore those variables, and
  that `basal-serve` / `basal-bench` pick the Strix Halo classes for the `-strix` modes only.
- Full suite on this machine: 91 passed, 3 skipped (MLX / Apple only).

## Results

Generated by `report.py` from `results/` (raw answers and latencies of every run).

- **POST is the fastest engine on 12 of the 20 request shapes.** The compiled graph mode (`--mode fast-strix`:
  torch.compile plus HIP graphs, works on ROCm, 216 s start-up) is 1–14% ahead on 5 of them (M with 2–10 questions, L with 1 and 5).
  `gguf` is level with POST on large requests (0.4–1.6% ahead on S-q20, L-q10 and L-q20), where the cached start
  saves the same ~60 tokens out of thousands.
- **The cache pays off on small requests and under load**: one question over a short state 178 → 162 ms;
  7.7 → 10.0 decisions/s with 32 clients.
- **Every decision that differs between engines is one of two near-ties**, in the fp32 reference as well:
  S with 20 questions, `s02` (fp32: 0.372 vs 0.374), and L with 10 questions, `click_target` (fp32: 0.489 vs 0.468).
  Every engine agrees with fp32 on all other decisions (143–144 of 144) and on all 26 edge-case decisions.
- **Batching does not change answers**: the edge requests sent all at once give the same decisions as one at a time
  in every engine (differences ≤ 0.018, bf16 / Q8_0 rounding of other batch shapes); the same question asked three
  times in one request gets identical probabilities.
- **SOAM is what makes many questions cheap** in graph mode: without it the state is computed again for every branch
  (XL with 20 questions: 156 s instead of 13.6 s).
- **Graph mode on long states**: a 6,269-token row is padded to the 8,192 shape (about 30% extra compute) and XL with
  20 questions (more than 8,192 packed tokens) is split into two rows, each with the state again; finer `BASAL_LENS`
  steps (`BASAL_LENS`) would reduce the padding. `gguf` and `gguf-strix` have no shapes and compute exactly the packed tokens. With the upstream defaults
  rows over 3,072 tokens run eagerly with the math attention path (XL: 34–67 s per request).
- **Throughput**: PRE stays at 3.8 decisions/s for any number of clients (one llama-server slot, one HTTP request per
  prompt); POST batches concurrent requests into one `llama_decode` and reaches 10.0–10.1.

### PRE → POST per request shape

PRE: llama-server (Vulkan, Mesa 25.3 container) + `basal-serve --mode llamacpp`, what `basal.service` ran before. POST: `basal-serve --mode gguf-strix` (llama.cpp in process, Vulkan, Mesa 26.2, cached prompt start), what it runs now. p50 of 2 timed passes, unique states.

| state | questions | prompt tokens | PRE | POST | speed-up |
|---|---|---|---|---|---|
| S | 1 | 480 | 342 ms | **162 ms** | 2.11× |
| S | 2 | 928 | 488 ms | **221 ms** | 2.21× |
| S | 5 | 3,640 | 1.52 s | **530 ms** | 2.87× |
| S | 10 | 6,496 | 2.96 s | **1.09 s** | 2.71× |
| S | 20 | 11,178 | 5.25 s | **1.80 s** | 2.92× |
| M | 1 | 1,416 | 729 ms | **408 ms** | 1.78× |
| M | 2 | 2,800 | 882 ms | **475 ms** | 1.86× |
| M | 5 | 11,128 | 1.93 s | **812 ms** | 2.38× |
| M | 10 | 18,664 | 3.44 s | **1.39 s** | 2.48× |
| M | 20 | 32,706 | 5.68 s | **2.12 s** | 2.69× |
| L | 1 | 4,296 | 1.87 s | **1.30 s** | 1.43× |
| L | 2 | 8,560 | 2.03 s | **1.33 s** | 1.53× |
| L | 5 | 34,168 | 3.18 s | **1.74 s** | 1.83× |
| L | 10 | 56,104 | 4.79 s | **2.40 s** | 2.00× |
| L | 20 | 98,946 | 7.22 s | **3.27 s** | 2.21× |
| XL | 1 | 12,404 | 5.67 s | **4.15 s** | 1.37× |
| XL | 2 | 24,776 | 5.85 s | **4.24 s** | 1.38× |
| XL | 5 | 99,032 | 7.26 s | **4.75 s** | 1.53× |
| XL | 10 | 161,508 | 9.08 s | **5.61 s** | 1.62× |
| XL | 20 | 285,430 | 11.92 s | **6.73 s** | 1.77× |

| clients (single questions) | PRE dec/s · p50 | POST dec/s · p50 | throughput |
|---|---|---|---|
| 1 | 3.8/s · 257 ms | **8.2/s** · 119 ms | 2.18× |
| 4 | 3.8/s · 1.01 s | **9.1/s** · 442 ms | 2.38× |
| 8 | 3.8/s · 2.08 s | **9.8/s** · 821 ms | 2.55× |
| 16 | 3.8/s · 4.16 s | **10.0/s** · 1.59 s | 2.60× |
| 32 | 3.8/s · 8.39 s | **10.0/s** · 3.20 s | 2.64× |
| 64 | 3.8/s · 16.69 s | **10.1/s** · 6.33 s | 2.64× |

### Start-up and memory

| engine | start-up | GTT after start |
|---|---|---|
| **PRE** llama-server Vulkan | 3.6 s | 6.0 GiB |
| llama-server ROCm | 3.6 s | 6.2 GiB |
| **PRE** fast-nocompile | 888.4 s | 32.1 GiB |
| fast-nocompile-strix, SOAM off | 78.1 s | 14.5 GiB |
| fast-nocompile-strix | 78.5 s | 14.5 GiB |
| fast-strix | 216.2 s | 14.3 GiB |
| gguf | 3.2 s | 5.8 GiB |
| **POST** gguf-strix | 3.7 s | 5.8 GiB |
| eager fp32 (reference) | 5.6 s | 18.4 GiB |

### Latency per request, all engines (p50; bold: fastest)

| state | q | branches | **PRE** llama-server Vulkan | llama-server ROCm | **PRE** fast-nocompile | fast-nocompile-strix, SOAM off | fast-nocompile-strix | fast-strix | gguf | **POST** gguf-strix |
|---|---|---|---|---|---|---|---|---|---|---|
| S | 1 | 1 | 342 ms | 208 ms | 309 ms | 227 ms | 224 ms | 202 ms | 178 ms | **162 ms** |
| S | 2 | 2 | 488 ms | 330 ms | 362 ms | 336 ms | 281 ms | 248 ms | 242 ms | **221 ms** |
| S | 5 | 8 | 1.52 s | 1.04 s | 1.30 s | 1.26 s | 749 ms | 679 ms | 573 ms | **530 ms** |
| S | 10 | 13 | 2.96 s | 1.94 s | 3.08 s | 3.26 s | 1.30 s | 1.20 s | 1.13 s | **1.09 s** |
| S | 20 | 23 | 5.25 s | 3.27 s | 5.28 s | 4.52 s | 2.20 s | 1.98 s | **1.79 s** | 1.80 s |
| M | 1 | 1 | 729 ms | 496 ms | 786 ms | 489 ms | 491 ms | 416 ms | 439 ms | **408 ms** |
| M | 2 | 2 | 882 ms | 622 ms | 792 ms | 921 ms | 488 ms | **417 ms** | 501 ms | 475 ms |
| M | 5 | 8 | 1.93 s | 1.39 s | 1.66 s | 3.40 s | 866 ms | **797 ms** | 817 ms | 812 ms |
| M | 10 | 13 | 3.44 s | 2.31 s | 3.62 s | 5.77 s | 1.49 s | **1.37 s** | 1.39 s | 1.39 s |
| M | 20 | 23 | 5.68 s | 3.71 s | 7.33 s | 10.40 s | 3.01 s | 2.84 s | 2.12 s | **2.12 s** |
| L | 1 | 1 | 1.87 s | 1.38 s | 3.09 s | 1.30 s | 1.30 s | **1.21 s** | 1.28 s | 1.30 s |
| L | 2 | 2 | 2.03 s | 1.53 s | 3.68 s | 2.61 s | 1.47 s | 1.36 s | 1.33 s | **1.33 s** |
| L | 5 | 8 | 3.18 s | 2.38 s | 4.96 s | 10.48 s | 1.86 s | **1.69 s** | 1.73 s | 1.74 s |
| L | 10 | 13 | 4.79 s | 3.44 s | 19.44 s | 17.35 s | 3.04 s | 2.75 s | **2.38 s** | 2.40 s |
| L | 20 | 23 | 7.22 s | 5.17 s | 24.44 s | 30.43 s | 4.76 s | 4.42 s | **3.22 s** | 3.27 s |
| XL | 1 | 1 | 5.67 s | 4.90 s | 34.02 s | 6.76 s | 6.70 s | 6.62 s | 4.16 s | **4.15 s** |
| XL | 2 | 2 | 5.85 s | 4.99 s | 67.15 s | 13.51 s | 6.73 s | 6.37 s | 4.25 s | **4.24 s** |
| XL | 5 | 8 | 7.26 s | 6.19 s | – | 54.05 s | 6.75 s | 6.17 s | 4.78 s | **4.75 s** |
| XL | 10 | 13 | 9.08 s | 7.66 s | – | 87.71 s | 6.78 s | 6.20 s | 5.65 s | **5.61 s** |
| XL | 20 | 23 | 11.92 s | 9.81 s | – | 155.72 s | 13.57 s | 12.36 s | 6.77 s | **6.73 s** |

### Cost of one more question (1 → 20 questions, per added question)

| state | **PRE** llama-server Vulkan | llama-server ROCm | **PRE** fast-nocompile | fast-nocompile-strix, SOAM off | fast-nocompile-strix | fast-strix | gguf | **POST** gguf-strix |
|---|---|---|---|---|---|---|---|---|
| S | 258 ms | 161 ms | 262 ms | 226 ms | 104 ms | 94 ms | 85 ms | 86 ms |
| M | 261 ms | 169 ms | 344 ms | 522 ms | 133 ms | 128 ms | 88 ms | 90 ms |
| L | 282 ms | 199 ms | 1.12 s | 1.53 s | 182 ms | 169 ms | 102 ms | 103 ms |
| XL | 329 ms | 258 ms | – | 7.84 s | 362 ms | 302 ms | 137 ms | 136 ms |

### Concurrency (single questions over short states, unique per request)

| clients | **PRE** llama-server Vulkan | llama-server ROCm | **PRE** fast-nocompile | fast-nocompile-strix, SOAM off | fast-nocompile-strix | fast-strix | gguf | **POST** gguf-strix |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.8/s · 257 ms | 5.7/s · 175 ms | 3.8/s · 246 ms | 5.7/s · 153 ms | 5.7/s · 155 ms | 6.4/s · 139 ms | 6.9/s · 141 ms | 8.2/s · 119 ms |
| 4 | 3.8/s · 1.01 s | 5.8/s · 697 ms | 3.8/s · 1.10 s | – | 4.8/s · 858 ms | 5.5/s · 759 ms | 7.4/s · 540 ms | 9.1/s · 442 ms |
| 8 | 3.8/s · 2.08 s | 5.8/s · 1.39 s | 4.1/s · 1.95 s | 5.5/s · 1.46 s | 5.5/s · 1.45 s | 6.2/s · 1.29 s | 7.6/s · 1.06 s | 9.8/s · 821 ms |
| 16 | 3.8/s · 4.16 s | 5.8/s · 2.75 s | 4.3/s · 3.70 s | – | 5.3/s · 2.89 s | 6.0/s · 2.53 s | 7.7/s · 2.07 s | 10.0/s · 1.59 s |
| 32 | 3.8/s · 8.39 s | 5.7/s · 5.57 s | 4.6/s · 6.93 s | 6.8/s · 4.70 s | 6.8/s · 4.67 s | 7.7/s · 4.15 s | 7.7/s · 4.12 s | 10.0/s · 3.20 s |
| 64 | 3.8/s · 16.69 s | 5.7/s · 11.28 s | 5.3/s · 12.06 s | – | 6.9/s · 8.99 s | 7.8/s · 7.96 s | 7.7/s · 8.33 s | 10.1/s · 6.33 s |

### Agreement with the fp32 reference

Reference: `--mode eager --dtype float32 --soam off --max-batch 1` (every prompt computed separately in fp32), on 17 requests (S-q1, S-q2, S-q5, S-q10, S-q20, M-q1, M-q2, M-q5, M-q10, M-q20, L-q1, L-q2, L-q5, L-q10, L-q20, XL-q1, XL-q2). A decision is the top option of a question or the yes/no of one label of a multi-label question.

| engine | same decision | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| **PRE** llama-server Vulkan | 144/144 | 0.0065 | 0.081 |
| llama-server ROCm | 143/144 | 0.0062 | 0.057 |
| **PRE** fast-nocompile | 143/144 | 0.0045 | 0.037 |
| fast-nocompile-strix, SOAM off | 144/144 | 0.0036 | 0.029 |
| fast-nocompile-strix | 143/144 | 0.0045 | 0.028 |
| fast-strix | 143/144 | 0.0037 | 0.024 |
| gguf | 143/144 | 0.0067 | 0.078 |
| **POST** gguf-strix | 143/144 | 0.0066 | 0.066 |

### Engine against engine (whole matrix)

| comparison | same decision | mean \|Δp\| | max \|Δp\| |
|---|---|---|---|
| fast-nocompile-strix: SOAM on vs off | 187/188 | 0.0046 | 0.042 |
| gguf vs gguf-strix (cached prompt start) | 188/188 | 0.0052 | 0.065 |
| fast-nocompile-strix vs fast-strix | 188/188 | 0.0047 | 0.036 |
| fast-nocompile vs fast-nocompile-strix | 142/144 | 0.0044 | 0.035 |
| PRE vs POST | 187/188 | 0.0061 | 0.067 |
| fast-nocompile-strix (bf16) vs gguf-strix (Q8_0) | 186/188 | 0.0082 | 0.064 |

### Edge cases

12 requests: mixed PL / EN questions over one state, 10 options, the same question three times, a JSON state, `facts: auto`, long option texts, a 5-level score, an 8-label multi with min / max, an `act` with a cost matrix, an English-only request, a 2-character state, `option_keys` show vs hide. Sent one after another, then all at once (the server batches them).

| engine | errors | concurrent = sequential decisions | concurrent vs sequential, max \|Δp\| | duplicate questions, max \|Δp\| | multi within min/max | act action | vs fp32: same decision | vs fp32: max \|Δp\| |
|---|---|---|---|---|---|---|---|---|
| llama-server ROCm | none | 26/26 | 0.0068 | 5.6e-05 | True | human | 26/26 | 0.026 |
| fast-nocompile-strix | none | 26/26 | 0.0146 | 0.0e+00 | True | human | 26/26 | 0.014 |
| gguf | none | 26/26 | 0.0183 | 0.0e+00 | True | human | 26/26 | 0.016 |
| **POST** gguf-strix | none | 26/26 | 0.0158 | 0.0e+00 | True | human | 26/26 | 0.018 |
| eager fp32 (reference) | none | 26/26 | 0.0000 | 0.0e+00 | True | human | – | – |
