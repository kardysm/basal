# basal 2.2

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--2.0%20models-yellow)](https://huggingface.co/collections/Remek/basal-20-6acd1712a8e443c86cd576fc)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-basal--omni--1.0%20models-yellow)](https://huggingface.co/collections/Remek/basal-omni-10-6acea28e8b5baed2db475c1f)
[![Website](https://img.shields.io/badge/web-basal.si5.pl-0a7d5a)](https://basal.si5.pl/)
[![basal-1.0 technical report](https://img.shields.io/badge/basal--1.0%20technical%20report-PDF-b31b1b.svg)](docs/basal-1.pdf)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23022986.svg)](https://doi.org/10.5281/zenodo.23022986)

Inference engine and models for **basal-2.0** and **basal-omni-1.0**: small, fast, calibrated *typed-decision* models
for Polish and English. The basal-omni models also take **images and audio** in a request (experimental).

**What it is.** basal is inspired by *System 1* (fast, intuitive) decision models such as Jev: instead of a chatbot
that writes an answer, the model reads a **state** (a message, a document, a case file, retrieved passages, an agent
trace, a web page as JSON) and answers a **typed question** about it by returning a **probability for each allowed
answer**, in one forward pass, without generating text. The answer can never fall outside the options you gave, and
the probability says how sure the model is.

**What it is for: a dynamic classifier.** You describe the classes *in the request*, in plain language, per call,
instead of training a classifier for them: routing tickets and documents (up to 255 options), rule and policy
checks, questions about long documents, RAG decisions, scoring on ordered scales, the next step of an agent, judging
answers, and triage with a confidence threshold that sends uncertain cases to a person.

> The name comes from the *basal ganglia*, the part of the brain that selects one action among competing options.

## Contents

- [What is new in basal 2.2](#what-is-new-in-basal-22) · [What is new in basal 2.1](#what-is-new-in-basal-21) ·
  [What is new in basal-2.0](#what-is-new-in-basal-20)
- [Models](#models) · [Results](#results)
- [Quick start](#quick-start) · [API](#api)
- [Images and audio: basal-omni](#images-and-audio-basal-omni-experimental) · [MCP for agents](#mcp-basal-as-a-tool-for-agents)
- [Documentation](#documentation): install paths, serving, API reference, results, formats, hardware
- [Limitations](#limitations) · [Previous releases](#previous-releases) · [Citation](#citation) ·
  [Contributors](#contributors) · [License](#license)

## What is new in basal 2.2

- **Calibrate the model on your own data**, for every basal and basal-omni model: label a few hundred of your own
  requests, run `basal-calibrate`, and serve the file it writes. The probabilities, and the `max_error` thresholds that
  decide what may be automated, are then certified on your data, not ours. A hosted decision API cannot be recalibrated
  on your data; a basal model can. Measured on a domain the model never saw, the shipped file's "1% error" setting would
  have been wrong 8% of the time; after calibration the calibration error fell about 3.2 times for choice questions,
  2 times for scores and 30% for yes / no, and nothing was automated above its target
  ([the example](docs/CALIBRATION.md#example-a-new-domain-measured)).
  ```bash
  basal-calibrate my_labelled.jsonl --model Remek/basal-omni-1.0-max --mode eager    # writes CALIBRATION.custom.json
  basal-serve --model Remek/basal-omni-1.0-max --mode eager --calibration CALIBRATION.custom.json
  ```
- New in 2.2: the task types' `automated` flag (`classify`, `verify`, and now `extract`) is certified from your
  labelled task questions; a few dozen labels already move the temperatures; the report shows how the shipped file
  does on your data, as served; `--report-only`; a calibration file knows the model it was fitted on.
  [docs/CALIBRATION.md](docs/CALIBRATION.md)
- **basal-omni on llama.cpp, Ollama and Apple Silicon** (text only): `--mode gguf` runs a GGUF conversion of
  basal-omni-1.0 through llama.cpp (Metal, CUDA or CPU) and `--mode mlx` / `mlx-q8` the bf16 weights through mlx-lm.
  New model repositories for basal-omni-1.0-mini and -max: **`-GGUF`** (Q8_0 and Q6_K, with a calibration fitted on
  Q8_0) and **`-FP8`** (`--mode vllm`); 8-bit **`-MLX`** ports (`--mode mlx`) of 1.0-mini, 1.0-max and 1.1-max.
  [docs/OMNI.md](docs/OMNI.md#gguf-llamacpp-ollama-and-mlx-text)
- **From the community:** `basal-export-gguf` writes or repairs GGUF files so that llama.cpp and Ollama tokenize like
  the Hugging Face tokenizer (Piotr Durlej, #8); two experimental, community-maintained engines for basal-1.5 /
  basal-2.0: `--mode tpu` on Google TPU (Kuba Sołtys, #15) and `--mode strix` on AMD Strix Halo (Michał Kardyś, #14);
  `basal-viz`, a local page that shows every step of a decision (@lukaspoloki, #2,
  [SERVING.md](docs/SERVING.md#see-how-a-decision-is-made)). [Contributors](#contributors)

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the
[release notes](https://github.com/rkinas/basal/releases/tag/v2.2.0).

## What is new in basal 2.1

- **Task types** (*experimental*), on the same API and the models you already serve: **`classify`** (labels with a
  rubric each), **`verify`** (is each statement supported, contradicted or not stated), **`compress`** (the parts of a
  document a question needs, as shorter text for another model) and **`extract`** (a JSON schema filled with values
  copied exactly from the document, with character offsets). No text is generated: each task is expanded into
  decisions with calibrated probabilities. [docs/TASKS.md](docs/TASKS.md); every question type on one page:
  [docs/QUESTION_TYPES.md](docs/QUESTION_TYPES.md).
- Every other request takes the same code path as in v2.0.0: on basal-omni-1.0-max, 23,276 of 23,278 served-check
  answers were the same (the two others were near-ties).
- **New models on Sat 17 Oct, 09:00 CEST** (the links work from then): **basal-omni-1.0-ultra** (31B, Werdykt 0.879)
  with an FP8 port, and **basal-omni-1.1-max**, an experimental task model (basal-omni-1.0-max with extra training on
  the task formats; 1.0-max stays the default). See [Models](#models).

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the
[release notes](https://github.com/rkinas/basal/releases/tag/v2.1.0).

## What is new in basal-2.0

- **Three models**: **basal-2.0-mini** (1.5B), **basal-2.0** (4.5B) and **basal-2.0-max** (11B), in bf16, FP8, GGUF and
  MLX 8-bit.
- **Up to 255 options per question** (basal-1.5: 10), read in one forward pass with one option order.
- **Werdykt**, the hidden Polish–English decision benchmark: every tier is above its basal-1.5 counterpart, with a
  lower calibration error, and faster. Not everything improved: on a panel of 10 public English decision tasks all
  three are below basal-1.5 ([results](#results)).
- **basal-omni-1.0** (*experimental*): **basal-omni-1.0-mini** (4B) and **basal-omni-1.0-max** (12B), basal models built
  on Google Gemma 4 that answer the same questions and also read images and audio.
- **Engine v2.0.0**: compact batching and FlashAttention 3 on Hopper, native FP8 checkpoints, the `tool` question
  type for agents, an MCP server, `basal-calibrate` and `--deterministic` serving.
- **Thinking mode** (4.5B, *experimental*, off by default): it lowered accuracy on Werdykt; we recommend leaving it
  off ([docs/THINKING.md](docs/THINKING.md)).

Full list: [docs/WHATS_NEW.md](docs/WHATS_NEW.md) and the [release notes](https://github.com/rkinas/basal/releases/tag/v2.0.0).

## Models

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-2.0-mini** | 1.5B | fastest; high-volume routing, laptops and small GPUs; context 8,192 tokens | [`Remek/basal-2.0-mini`](https://huggingface.co/Remek/basal-2.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-mini-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-mini-MLX-8bit) |
| **basal-2.0** | 4.5B | main model: best balance of accuracy and speed; experimental thinking mode | [`Remek/basal-2.0-4.5B`](https://huggingface.co/Remek/basal-2.0-4.5B) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-4.5B-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-4.5B-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-4.5B-MLX-8bit) |
| **basal-2.0-max** | 11B | most accurate; long documents, judging, hard English decisions | [`Remek/basal-2.0-max`](https://huggingface.co/Remek/basal-2.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-2.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-2.0-max-GGUF) · [`-MLX-8bit`](https://huggingface.co/Remek/basal-2.0-max-MLX-8bit) |

- Each repository ships its prompt format (`basal.json`) and calibration files. `-FP8`: NVIDIA ModelOpt FP8
  checkpoints for the basal engine (sm89+); `-GGUF` for Ollama and llama.cpp; `-MLX-8bit` for Apple Silicon. Every
  port is checked against its bf16 model; its card has the numbers. Files in detail:
  [docs/INSTALL.md](docs/INSTALL.md#what-the-model-repositories-contain).

**basal-omni** (experimental; text, images and audio; built on Google Gemma 4):

| model | params | use | Hugging Face |
|---|---|---|---|
| **basal-omni-1.0-mini** | 4B (Gemma 4 E4B) | decisions about text, images and audio; about 16 GB of GPU memory | [`Remek/basal-omni-1.0-mini`](https://huggingface.co/Remek/basal-omni-1.0-mini) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-mini-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-mini-GGUF) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.0-mini-MLX) |
| **basal-omni-1.0-max** | 12B | the default 12B model (Werdykt 0.823); text, images and audio; about 24 GB of GPU memory | [`Remek/basal-omni-1.0-max`](https://huggingface.co/Remek/basal-omni-1.0-max) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-max-FP8) · [`-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-max-GGUF) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX) |
| **basal-omni-1.1-max** | 12B | *experimental task model*: better on `classify`, `verify`, `compress` and `extract`, slightly behind 1.0-max on general decisions (Werdykt 0.821) | [`Remek/basal-omni-1.1-max`](https://huggingface.co/Remek/basal-omni-1.1-max) · [`-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX) |
| **basal-omni-1.0-ultra** | 31B (Gemma 4 31B) | highest Werdykt score of the basal models (0.879); text and images (no audio); 59 GB of bf16 weights | [`Remek/basal-omni-1.0-ultra`](https://huggingface.co/Remek/basal-omni-1.0-ultra) · [`-FP8`](https://huggingface.co/Remek/basal-omni-1.0-ultra-FP8) |

- bf16 weights, served by `basal-serve --mode eager` on an NVIDIA GPU (`--media on` for images and audio).
  The `-FP8` checkpoints (mini 12 GB, max 14 GB, ultra 33 GB) are served by `--mode vllm`, text only, each with its own
  calibration (`CALIBRATION.vllm.json`). The `-GGUF` files (Q8_0 and Q6_K, mini and max) are served by `--mode gguf`
  (llama.cpp, with `CALIBRATION.gguf.json`) or in front of Ollama, text only. The `-MLX` ports (8-bit; 1.0-mini, 1.0-max
  and 1.1-max) are served by `--mode mlx` on Apple Silicon, text only, with `CALIBRATION.mlx.json`.
- All models are fine-tuned from Apache-2.0 base models (Bielik v3 for basal-2.0, Gemma 4 for basal-omni; see
  [NOTICE](NOTICE)) and released under Apache-2.0.

Which one to choose: start with **basal-2.0 (4.5B)**. Use **max** when accuracy on long or hard inputs matters more
than latency, and **mini** for high volumes, laptops and small GPUs (its context is 8,192 tokens). All three use the
same API, prompt format and calibration procedure. Use **basal-omni** (experimental) when a request carries images or
audio, or when the highest accuracy matters more than speed and an eager-mode server is enough.

## Results

[Werdykt](https://basal.si5.pl/benchmark/): 10 categories in Polish and English, 5,000 decisions whose items are
not published, one protocol for every model. Each basal-2.0 model against the basal-1.5 model of the same size:

| | mini: 1.5 → 2.0 | 4.5B: 1.5 → 2.0 | max: 1.5 → 2.0 |
|---|---|---|---|
| Werdykt macro accuracy | 0.607 → **0.618** | 0.721 → **0.729** | 0.773 → **0.784** |
| calibration error (ECE, lower is better) | 0.086 → **0.035** | 0.068 → **0.060** | 0.074 → **0.044** |
| decisions/s at 16 clients, bf16, one H100 | 20.5 → **106.4** | 23.0 → **32.2** | 10.2 → **16.2** |
| English panel (10 public tasks), points | −2.1 | −3.6 | −1.8 |

basal-1.5 (4.5B) and basal-1.5-max are measured on SGLang, their fastest configuration. basal-omni-1.0 scores 0.769
(mini) and 0.823 (max) on Werdykt. Hosted API models and larger open decision systems score above basal-2.0: the
[board](https://basal.si5.pl/benchmark/) lists every tested system with its accuracy, latency and cost. Intervals,
calibration, other benchmarks and speed: [docs/RESULTS.md](docs/RESULTS.md).

## Quick start

NVIDIA GPU, with [uv](https://docs.astral.sh/uv/) in a fresh environment:

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[fp8] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --port 8000
```

- **`--template-prefix off` for the 4.5B model** (bf16 and `-FP8`); mini and max run with the default.
- **Install torch first, from the CUDA 12.8 index**: a plain `pip install basal` may take a torch built for a newer
  CUDA than your driver. DGX Spark and B300: the cu130 build ([docs/HARDWARE.md](docs/HARDWARE.md)).
- The first start in `--mode fast` compiles the model (several minutes); `--mode fast-nocompile` starts in seconds.

Apple Silicon:

```bash
uv venv --python 3.12 --allow-existing ~/basal-mlx && source ~/basal-mlx/bin/activate
uv pip install "basal[mlx] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000
```

Ask questions with curl. Several questions about one state go into one request and share the state:

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Klient: od wczoraj nie mogę zalogować się do bankowości internetowej, system pokazuje błąd hasła.",
  "questions": {
    "dept":   {"type": "choice", "instructions": "Do którego działu skierować zgłoszenie?",
               "criteria": {"cards": "Reklamacje kart", "online": "Wsparcie bankowości elektronicznej", "loans": "Kredyty"}},
    "urgent": {"type": "noul", "instructions": "Czy klient nie może korzystać z usługi?"}}}'
```

Each answer has a probability for every option, the chosen key and the confidence:

```jsonc
{
 "model": "basal-2.0-4.5B",
 "answers": {
  "dept":   {"type": "choice", "choice": "<key>", "probabilities": {"cards": p, "online": p, "loans": p}, "confidence": p},
  "urgent": {"type": "noul", "noul": p_yes, "probabilities": {"true": p, "false": p}, "confidence": p}
 },
 "usage": {"input_tokens": n, "output_tokens": 0, "questions": 2, "branches": 2, "latency_ms": t}
}
```

| other ways to run | |
|---|---|
| FP8 checkpoints (Hopper, Blackwell) | [docs/INSTALL.md](docs/INSTALL.md#fp8-checkpoints-basal-engine) |
| vLLM or SGLang with bf16 weights | [docs/INSTALL.md](docs/INSTALL.md#vllm-or-sglang-with-bf16-weights) |
| Ollama and llama.cpp (GGUF) | [docs/INSTALL.md](docs/INSTALL.md#ollama-and-llamacpp-gguf) |
| Apple Silicon in detail, CPU | [docs/INSTALL.md](docs/INSTALL.md#apple-silicon-mlx) |
| images and audio (basal-omni) | [below](#images-and-audio-basal-omni-experimental) |
| all seven engines compared | [docs/INSTALL.md](docs/INSTALL.md#engines) |
| a JSONL file, serving flags, benchmarking your GPU | [docs/SERVING.md](docs/SERVING.md) |
| Python client | [docs/INSTALL.md](docs/INSTALL.md#nvidia-gpu-the-basal-engine-recommended) |

## API

`POST /v1/systemone` with a `state` (text or JSON) and named, typed `questions`; every answer has a calibrated
probability for each option and a `confidence`. Up to 255 options per question (basal-1.x models: 10); several
questions over one state are answered in one pass.

| type | answer | example |
|---|---|---|
| `choice` | distribution over named options (up to 255) | route a ticket, pick the applicable rule, the intent among 150 |
| `noul` | probability of *yes* (with optional descriptions of both sides) | "was the appeal filed on time?" |
| `score` | distribution over ordered levels + expected level | urgency 0–3, severity, grade of an answer |
| `multi` | independent probability per label + the selected set | "which of these issues does the message raise?" |
| `act` | the action with the lowest expected cost under your cost matrix | approve / reject / send to a person |
| `tool` | the next tool from OpenAI / MCP tool definitions, or answer directly / ask the user / no suitable tool | the next step of an agent |

To send uncertain decisions to a person, use `act` or `tool` with `"max_error": 0.05`: the answer says
`"deferred": true` below the certified confidence threshold. Request and response fields, every question type,
using the confidence, calibrating on your own data and how to ask: [docs/API.md](docs/API.md).

## Images and audio: basal-omni (experimental)

The basal-omni models answer the same questions, and with `--media on` a request may also carry images and audio
(`"media"`):

```bash
uv venv --python 3.12 --allow-existing ~/basal-env && source ~/basal-env/bin/activate
uv pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
basal-serve --model Remek/basal-omni-1.0-mini --mode eager --media on --port 8000   # or Remek/basal-omni-1.0-max
```

They were trained on text decisions only: Gemma 4's encoders read the media zero-shot and the answers use the text
calibration, so validate them on your own media. Request format and limits:
[docs/INSTALL.md](docs/INSTALL.md#images-and-audio-basal-omni-experimental); memory and engine details:
[docs/OMNI.md](docs/OMNI.md).

## MCP: basal as a tool for agents

*Experimental.* With the `basal[mcp]` extra, `basal-serve --mcp on` also serves the decisions over the
[Model Context Protocol](https://modelcontextprotocol.io) at `/mcp`, with the read-only tools `decide` and
`decide_many`; `basal-mcp` is a stdio bridge for hosts that start local processes.

```bash
uv pip install "basal[fp8,mcp] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
basal-serve --model Remek/basal-2.0-4.5B --mode fast --template-prefix off --host 127.0.0.1 --port 8000 --mcp on
claude mcp add --transport http basal http://127.0.0.1:8000/mcp     # Claude Code; other hosts take the same URL
```

No authentication, like the HTTP API: bind `--host 127.0.0.1` or put a gateway in front. Details:
[docs/MCP.md](docs/MCP.md).

## Documentation

| | |
|---|---|
| [INSTALL](docs/INSTALL.md) | engines, the model files and every install path: NVIDIA (bf16, FP8), vLLM, SGLang, Apple Silicon, Ollama / llama.cpp, CPU, basal-omni |
| [API](docs/API.md) | request, response, question types, using the confidence, how to ask |
| [SERVING](docs/SERVING.md) | serving modes and flags, inference on a JSONL file, benchmarking your GPU |
| [RESULTS](docs/RESULTS.md) | results against basal-1.5: Werdykt, other benchmarks, speed |
| [MCP](docs/MCP.md) · [THINKING](docs/THINKING.md) · [OMNI](docs/OMNI.md) | MCP server and bridge; the experimental thinking mode; basal-omni on the engine |
| [FEATURES](docs/FEATURES.md) · [CALIBRATION](docs/CALIBRATION.md) | `multi`, `act`, `tool` and evidence in detail; calibrating on your own data |
| [QUESTION_TYPES](docs/QUESTION_TYPES.md) | all question types on one page: what each asks and answers, with an example |
| [TASKS](docs/TASKS.md) | *experimental*: `classify`, `verify`, `compress`, `extract` (JSON schema) on the same API |
| [FORMATS](docs/FORMATS.md) · [FORMAT2](docs/FORMAT2.md) · [SOAM](docs/SOAM.md) | prompt formats, option orders and calibration files; the runtime contract; shared-state packing |
| [HARDWARE](docs/HARDWARE.md) · [PORTS](docs/PORTS.md) · [GGUF](docs/GGUF.md) · [QUANTIZATION](docs/QUANTIZATION.md) | which mode on which GPU; MLX and GGUF ports; FP8 and other low precision |
| [WHATS_NEW](docs/WHATS_NEW.md) · [basal-1.0 report](docs/basal-1.pdf) | the full list of changes in 2.0; the basal-1.0 technical report |

## Limitations

- Evaluated on held-out and hidden test sets and on public benchmarks; claims about a specific document collection
  need validation on your own data.
- On a panel of 10 public English decision tasks all three basal-2.0 models are below basal-1.5 (−2.1 / −3.6 / −1.8
  points for mini / 4.5B / max); basal-2.0-mini is also below basal-1.5-mini on Polish decisions (−1.4 points).
- The calibration is cautious on unseen tasks: `max_error: 0.01` defers every decision with the shipped files (except
  `choice` with `basal-2.0-max-GGUF`'s `CALIBRATION.ollama.json`, confidence 0.989); use 0.05 or fit thresholds on your
  data. At the 1% target on Werdykt, basal-2.0 (4.5B) and max automate fewer decisions than basal-1.5.
- The thinking mode is experimental: on Werdykt it is less accurate, less calibrated and slower than the default.
- Polish world knowledge of a small model is limited; supply the relevant facts and rules in the state. Legal rules
  change; the models do not know rules introduced after their training.
- Exact arithmetic and thresholds remain a weak spot of one-pass decisions; compute them in code or use `"facts":
  "auto"` for Polish dates and amounts.
- basal-2.0-mini has an 8,192-token context; longer states are cut in the middle (or refused with
  `--context-overflow refuse`).
- `act`, `tool`, `multi` over many labels and evidence spans are experimental; `multi` asks each label as an
  independent yes/no question.
- basal-omni is experimental: trained on text decisions only, with images and audio read zero-shot by Gemma 4's
  encoders and answered with the text calibration (`"calibration": "text"`); validate it on your own media. It runs
  in `eager` mode (ultra's FP8 port: `vllm` mode, text only), without the thinking mode of basal-2.0.
- The task types of 2.1 are experimental: their API may change in a minor release, and `automated` needs thresholds
  certified for the task (only basal-omni-1.1-max ships one, `classify` at 5%).
- Decisions with serious consequences for people should be reviewed by a person.

## Previous releases

**basal-1.5** (October 2026; mini 1.5B, 4.5B, max 11B, with FP8 / NVFP4, MLX 4-bit / 8-bit and GGUF ports) remains
available on Hugging Face ([collection](https://huggingface.co/collections/Remek/basal-15-6ac0de0e9199d121c6adac20))
and is served unchanged by this engine (`--model Remek/basal-1.5-4.5B`; two option orders by default). Its release
README, with the full Werdykt board of that release: [v1.5.0](https://github.com/rkinas/basal/tree/v1.5.0).

**basal-1.0** (September 2026; 4.5B and 1.5B, plus FP8 / NVFP4 ModelOpt checkpoints) remains available on Hugging Face
(`Remek/basal-1.0-4.5B`, `Remek/basal-1.0-1.5B`). Its full tables, methods and limitations are in the
[basal-1.0 technical report](docs/basal-1.pdf) and the [v1.0.1 README](https://github.com/rkinas/basal/tree/v1.0.1).
The engine serves basal-1.0 models unchanged; `fast-exit` and the `vllm` mode with the ModelOpt checkpoints work as
before. The Apple Silicon engine modes (`mps`, `gguf`, `mlx-q8`) and the Apple Silicon engine study were contributed
by Paweł Kiszczak ([#4](https://github.com/rkinas/basal/pull/4)).

## Citation

```bibtex
@software{kinas2026basal20,
  title   = {basal-2.0: typed-decision models and inference engine for Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {2.0.0},
  url     = {https://github.com/rkinas/basal}
}

@techreport{kinas2026basal,
  title       = {basal-1.0: Reliable, Highly Optimized Typed Decisions for Polish},
  author      = {Kinas, Remigiusz},
  institution = {ai5},
  year        = {2026},
  type        = {Technical report},
  doi         = {10.5281/zenodo.23022986},
  url         = {https://doi.org/10.5281/zenodo.23022986}
}
```

basal-omni-1.0 (text, images and audio):

```bibtex
@software{kinas2026basalomni10,
  title   = {basal-omni-1.0: typed-decision models for text, images and audio in Polish and English},
  author  = {Kinas, Remigiusz},
  year    = {2026},
  version = {1.0},
  url     = {https://huggingface.co/collections/Remek/basal-omni-10-6acea28e8b5baed2db475c1f}
}
```

## Contributors

Thank you to the people whose work is in this repository, by pull request:

- Paweł Kiszczak ([@pawelkiszczak](https://github.com/pawelkiszczak)): the Apple Silicon engines `--mode mlx` /
  `mlx-q8` and `--mode mps`, the in-process GGUF engine `--mode gguf` (llama-cpp-python) and `--mode ollama`, and the
  comparison of Apple Silicon engines and quantised formats (MLX, oMLX oQ, GGUF) with its figures in
  [HARDWARE.md](docs/HARDWARE.md#apple-silicon) ([#4](https://github.com/rkinas/basal/pull/4)).
- Piotr Durlej ([@pdurlej](https://github.com/pdurlej)): `basal-export-gguf`, which converts a basal model to GGUF, or
  repairs an existing GGUF file, so that its tokenizer matches the Hugging Face one, as in the official `-GGUF` files
  ([#8](https://github.com/rkinas/basal/pull/8)).
- Kuba Sołtys ([@qooba](https://github.com/qooba)): the experimental Google TPU engine `--mode tpu` in JAX / XLA, with
  its Colab notebook ([#15](https://github.com/rkinas/basal/pull/15)).
- Michał Kardyś ([@kardysm](https://github.com/kardysm)): the experimental `--mode strix`, a basal-1.5 / basal-2.0
  forward with Triton kernels for AMD Strix Halo, and its benchmarks
  ([#14](https://github.com/rkinas/basal/pull/14)).
- [@lukaspoloki](https://github.com/lukaspoloki): `basal-viz`, a local page that shows every step of a decision, from
  the prompt to the per-layer readout, the calibration and the decision
  ([#2](https://github.com/rkinas/basal/pull/2)).

`--mode tpu` and `--mode strix` are community-maintained: their numbers in [HARDWARE.md](docs/HARDWARE.md) are the
contributors' own measurements.

## License

Apache-2.0. The models are derivatives of Apache-2.0 base models (Bielik v3, SpeakLeash and ACK Cyfronet AGH, for
basal-2.0; Gemma 4, Google, for basal-omni); see [NOTICE](NOTICE).
