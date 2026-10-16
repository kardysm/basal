# What is new in basal 2.1

**Engine v2.1.0: task types (experimental)**
- Four task types on the same API, for the basal models you already serve (no new model): **`classify`** (labels with
  a rubric each), **`verify`** (is each statement supported, contradicted or not stated by the state), **`compress`**
  (the sentences, lines or paragraphs needed for a question, as shorter text for another model) and **`extract`** (a
  JSON schema filled with values taken exactly from the document, with character offsets).
- Each task is expanded into the decision types over the shared state, so it costs about as much as the decisions it
  asks: no text is generated, and `extract` values are always exact text of the document.
- Measured on public sets the models were not trained on, with basal-omni-1.0-max: see
  [TASKS.md](TASKS.md#results). Experimental: the API may change, and the `automated` flag of a task answer needs
  thresholds fitted for that task.
- Every other request takes the same code path as in v2.0.0 (on basal-omni-1.0-max, 23,276 of 23,278 served-check
  answers were the same; the two others were near-ties).

**New models (Hugging Face, Sat 17 Oct, 09:00 CEST)**
- **basal-omni-1.0-ultra** (31B, Gemma 4 31B; text and images): Werdykt 0.879, the highest of the basal models; and
  **basal-omni-1.0-ultra-FP8** for `--mode vllm` (text only).
- **basal-omni-1.1-max** (*experimental*): basal-omni-1.0-max with extra training on the task formats, better on the
  tasks and slightly behind 1.0-max on general decisions; 1.0-max stays the default ([TASKS.md](TASKS.md#results)).

# What is new in basal-2.0


**Models**
- A family of three: **basal-2.0-mini** (1.5B, fastest), **basal-2.0** (4.5B, main model, with an experimental
  thinking mode) and **basal-2.0-max** (11B, most accurate). Weights in bf16, plus **FP8** (NVIDIA Hopper and
  Blackwell), **GGUF** (Ollama, llama.cpp) and **MLX 8-bit** (Apple Silicon).
- **Prompt format 2**: up to **255 options per question** (basal-1.5: 10), read in one forward pass with **one option
  order** by default (basal-1.5 read two orders). See [FORMATS.md](FORMATS.md).
- On **Werdykt**, the hidden Polish–English decision benchmark, every tier is above its basal-1.5 counterpart (+1.2 /
  +0.7 / +1.1 points for mini / 4.5B / max) with a lower calibration error (0.035 / 0.060 / 0.044 against
  0.086 / 0.068 / 0.074), and faster. Not everything improved: on a panel of 10 public English decision tasks all three
  are below basal-1.5 (−2.1 / −3.6 / −1.8 points). All numbers: [results against basal-1.5](RESULTS.md#results-against-basal-15).
- **Calibration** with a temperature per question type and option-count bucket, fitted on question families the model
  was not trained on. It is cautious on new tasks: the shipped files certify a 5% error target but no 1% target (one
  port file excepted), so `max_error: 0.01` defers every decision ([using the confidence](API.md#using-the-confidence)).
- **Thinking mode** (4.5B, *experimental, off by default*): `--think auto` writes a short reasoning note for uncertain
  questions before deciding. On Werdykt it lowered accuracy and calibration and cut throughput; we recommend leaving it
  off ([thinking mode](THINKING.md#thinking-mode-experimental)).

**basal-omni-1.0 (experimental): images and audio**
- Two models on Google Gemma 4: **basal-omni-1.0-mini** (4B, Gemma 4 E4B) and **basal-omni-1.0-max** (12B). They answer
  the same typed questions over the same API, and with `--media on` a request may also carry **images and audio**
  (`"media"`).
- On Werdykt: 0.769 (mini) and 0.823 (max), against 0.721 for basal-1.5 (4.5B) and 0.773 for basal-1.5-max.
- Trained on text decisions only: Gemma 4's own encoders read the images and audio zero-shot, and media answers use
  the text calibration. `--mode eager` only for now. See [images and audio](INSTALL.md#images-and-audio-basal-omni-experimental)
  and [OMNI.md](OMNI.md).

**Engine v2.0.0**
- **Faster serving on one GPU**: compact batching (single prompts packed into one row with FlashAttention 3 varlen),
  the prompt template computed once at start-up, a scheduler that keeps long requests from delaying short ones,
  merged projections and FlashAttention 3 on Hopper, native ModelOpt FP8 checkpoints in `--mode fp8`.
- **One engine for text and media**: the eager backend serves the basal-omni models (Gemma 4 architecture) and, with `--media on` and
  the `basal[media]` extra, requests with images and audio (`"media"`, `"detail"`, `"lang"`). In every model it now
  reads option codes in fp32, reads left-padded prompts at their own positions and prefills long prompts in
  2,048-token chunks.
- **New question type `tool`**: the next tool (or answer directly / ask the user / no suitable tool) from OpenAI, MCP
  or Anthropic tool definitions, also as a set of tools (*experimental*).
- **MCP for agents** (*experimental*): `basal-serve --mcp on` also serves the decisions as an MCP server at `/mcp`
  (tools `decide` and `decide_many`), and `basal-mcp` connects a running server to agent hosts over stdio
  ([MCP](MCP.md#mcp-basal-as-a-tool-for-agents)).
- **`multi`** gains set objectives (`iou`, `f1`, `exact`) and `exclusive` / `fallback` labels.
- **`basal-calibrate`**: fit the temperatures and certified confidence thresholds on your own labelled requests.
- **`--deterministic`** serving (answers independent of load), **`--orders 1 | 2 | auto`**, long states truncated in
  the middle instead of refused (`--context-overflow`), option-code logits in fp32 in every backend.
- The engine still serves basal-1.5 and basal-1.0 models (prompt format 1) with the same API.

Full list: [release notes](https://github.com/rkinas/basal/releases/tag/v2.0.0). Details of the experimental features:
[FEATURES.md](FEATURES.md).
