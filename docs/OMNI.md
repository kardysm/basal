# basal-omni on the basal engine (Gemma 4)

basal-omni-1.0 are experimental typed-decision models on Google Gemma 4: basal-omni-1.0-mini (4B, Gemma 4 E4B) and
basal-omni-1.0-max (12B, Gemma 4 12B), with prompt format 2 ([FORMAT2.md](FORMAT2.md)): up to 255 options, one forward pass per
question. They are trained on text; with `--media on` the server also takes images and audio in a request (below).

## Serve

```bash
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.2.0.tar.gz"
basal-serve --model Remek/basal-omni-1.0-max --mode eager --media on --port 8000    # or Remek/basal-omni-1.0-mini
```

- **Modes:** `eager`, the PyTorch forward in bf16 on CUDA (text, images and audio), and, since v2.1.0, `vllm`
  (text only; faster, with prefix caching and FlashAttention). The `-FP8` checkpoints (mini, max, ultra) run on
  `--mode vllm` only.
  With `--mode vllm` a model that ships `CALIBRATION.vllm.json` (basal-omni-1.1-max and the `-FP8` checkpoints) uses
  it; the others use their `CALIBRATION.json`.
- **Repeatable answers with vLLM:** vLLM's numbers depend on what the server computed before (prefix cache,
  batching), so a decision near a tie can change between runs. Start the server with `VLLM_BATCH_INVARIANT=1` for
  repeatable answers: on basal-omni-1.0-max-FP8 (RTX PRO 6000, 2,000 decisions at 32 in flight) two runs then gave the
  same top answer on all of them (without it, 94.7%), at about 23% fewer decisions per second. `--mode eager
  --deterministic` is the other way.
- **GGUF and MLX (text only):** `--mode gguf --gguf <file>` runs a GGUF conversion in process through llama.cpp
  (Metal, CUDA or CPU) and `--mode mlx` / `mlx-q8` runs the bf16 weights through mlx-lm on Apple Silicon (below).
- **Option orders:** like every prompt-format-2 model, it asks one option order by default and loads
  `CALIBRATION.json`. `--orders 2` averages both orders and loads `CALIBRATION.o2.json`.
- **Memory:** about 16 GB of GPU memory for mini and 24 GB for max (bf16); long states need more for attention.
- **API:** the requests and responses are the same as for every basal model ([API.md](API.md)).

## What the engine does differently for Gemma 4

- **Attention kernel:** SDPA runs on the math kernel only. The memory-efficient kernel returned wrong attention on
  real Gemma 4 activations in some shapes, and FlashAttention does not take Gemma's 512-wide global heads.
- **Padding:** left-padded prompts in a batch read at their own positions, so a padded prompt gives the same answer as
  the prompt alone.
- **Code logits:** option codes are read in fp32 from the output head's input, with Gemma's final logit softcap
  applied in fp32. The model body runs in bf16.
- **Not yet for Gemma 4:** `fast` (shared prefix, CUDA graphs), the engine's own `fp8` and `nvfp4` modes and SGLang.
  FP8 weights are served through vLLM (the `-FP8` checkpoints). The rest come with a later basal-omni release.

## GGUF (llama.cpp, Ollama) and MLX (text)

llama.cpp (b11312, the build in llama-cpp-python 0.3.36) and mlx-lm 0.32.2 implement Gemma 4's text decoder: sliding
and global attention, per-layer inputs (E4B), KV sharing, the final logit softcap. Convert with llama.cpp's converter
(the text decoder only; the image and audio towers are left out), as for basal-1.0 ([GGUF.md](GGUF.md#convert-yourself)):

```bash
python llama.cpp/convert_hf_to_gguf.py basal-omni-1.0-mini --outtype q8_0 --outfile basal-omni-1.0-mini-Q8_0.gguf
basal-serve --model ./basal-omni-1.0-mini --mode gguf --gguf basal-omni-1.0-mini-Q8_0.gguf --port 8000
basal-serve --model ./basal-omni-1.0-mini --mode mlx --port 8000        # Apple Silicon, bf16 weights (mlx-q8: 8-bit)
```

Ready files: [`Remek/basal-omni-1.0-mini-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-mini-GGUF) and
[`Remek/basal-omni-1.0-max-GGUF`](https://huggingface.co/Remek/basal-omni-1.0-max-GGUF) (Q8_0 and Q6_K, Modelfiles for
Ollama, `CALIBRATION.gguf.json` fitted on Q8_0); their cards have the commands.

8-bit MLX ports for `--mode mlx` (Apple Silicon, `CALIBRATION.mlx.json`):
[`Remek/basal-omni-1.0-mini-MLX`](https://huggingface.co/Remek/basal-omni-1.0-mini-MLX),
[`Remek/basal-omni-1.0-max-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX),
[`Remek/basal-omni-1.1-max-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX).

llama.cpp tokenizes the rendered basal-omni prompts exactly like the Hugging Face tokenizer (2,000 of 2,000 development
prompts), so text front ends (Ollama, `llama-server` with text prompts) read the same tokens. For Gemma 4 the engine
reads the decoder inside mlx-lm's multimodal wrapper, applies the softcap to the fp32 code logits and keeps the whole
shared prefix in the KV cache (mlx-lm's rotating cache would keep only the last window); `--mode gguf` loads
`CALIBRATION.gguf.json` and `--mode mlx` `CALIBRATION.mlx.json` when the model ships one, else `CALIBRATION.json`.

Measured 2026-10-17 against the eager bf16 engine, one option order, the shipped `CALIBRATION.json`, with the gate of
the FP8 ports (agreement of the top answer on 2-10 options and at bf16 margin >= 0.2 above the tier's bars, accuracy
within 1 point, ECE at most 0.01 above). CUDA: one RTX PRO 6000 (Blackwell), all 10,606 development decisions, 16
requests in flight. Apple: M4 Pro (24 GB), mini on 2,000 of them (1,500 / 200 / 300 with 2-10 / 11-26 / 27-255
options), max on 600 (400 / 100 / 100), 8 in flight. Quantisations below Q8_0 are made with `llama-quantize` from the
F16 file, without an importance matrix.

| model | port | file | where | agreement 2-10 | at margin >= 0.2 | accuracy | ECE | decisions/s | gate |
|---|---|---|---|---|---|---|---|---|---|
| mini | GGUF F16 | 14.9 GB | CUDA | 0.9949 | 0.9999 | -0.1 pt | +0.001 | 17.2 | pass |
| mini | GGUF Q8_0 | 7.9 GB | CUDA | 0.9871 | 0.9977 | -0.1 pt | -0.000 | 16.4 | pass |
| mini | GGUF Q6_K | 6.2 GB | CUDA | 0.9709 | 0.9932 | +0.1 pt | -0.005 | 14.4 | pass |
| mini | GGUF Q5_K_M | 5.7 GB | CUDA | 0.9476 | 0.9789 | -0.3 pt | +0.002 | 15.9 | fail (margin) |
| mini | GGUF Q4_K_M | 5.3 GB | CUDA | 0.9020 | 0.9479 | -0.2 pt | -0.003 | 16.3 | fail |
| mini | GGUF Q8_0 | 7.9 GB | Metal | 0.9887 | 0.9992 | +0.1 pt | -0.003 | 1.2 | pass |
| mini | MLX bf16 | 15 GB | Metal | 0.9933 | 0.9992 | -0.1 pt | +0.001 | 1.0 | pass (1) |
| mini | MLX `mlx-q8` | 8 GB | Metal | 0.9893 | 0.9992 | +0.1 pt | +0.001 | 0.8 | pass |
| mini | Ollama 0.35.0, Q8_0 | 7.9 GB | Metal | 0.9800 | 0.9961 | +0.3 pt | +0.003 | 2.1 | pass (2) |
| max | GGUF F16 | 23.8 GB | CUDA | 0.9930 | 0.9995 | +0.0 pt | +0.001 | 9.2 | pass |
| max | GGUF Q8_0 | 12.7 GB | CUDA | 0.9868 | 0.9983 | -0.1 pt | -0.001 | 8.9 | pass |
| max | GGUF Q6_K | 9.8 GB | CUDA | 0.9761 | 0.9947 | -0.1 pt | -0.005 | 7.5 | pass |
| max | GGUF Q5_K_M | 8.5 GB | CUDA | 0.9579 | 0.9825 | -0.4 pt | +0.001 | 8.5 | fail (margin) |
| max | GGUF Q4_K_M | 7.4 GB | CUDA | 0.9346 | 0.9607 | -1.0 pt | +0.016 | 8.7 | fail |
| max | GGUF Q8_0 | 12.7 GB | Metal | 0.9950 | 1.0000 | +0.0 pt | +0.013 | 0.4 | (3) |
| max | MLX `mlx-q8` | 14 GB | Metal | 0.9875 | 0.9973 | +0.8 pt | +0.008 | 0.3 | pass (4) |
| max | MLX 8-bit port (`-MLX`) | 13.6 GB | Metal | 0.989 | 1.000 | -0.1 pt | +0.002 | 0.25 | pass (5) |
| 1.1-max | MLX 8-bit port (`-MLX`) | 13.6 GB | Metal | 0.984 | 0.999 | +0.6 pt | -0.003 | 0.27 | pass (6) |

(1) On the 24 GB Mac, 191 of the 300 longest prompts (27-255 options) ran out of GPU memory with bf16 weights; the
2-10-option items the gate reads all ran. (2) 300 items with 2-10 options, one request at a time (Ollama returns at
most 20 log-probabilities, so up to 20 options). (3) max Q8_0 with the default 16,384-cell context does not fit in the
24 GB Mac's GPU memory (llama_decode fails); measured with 8,192 cells. Its ECE on 400 items is above the bar while
the CUDA run of the same file on the same 400 items is +0.001 and the relative temperature is 1.00: binned ECE at this
size moves by about 0.01; a Metal run on more items (or a larger Mac) is needed before claiming it. (4) 24 of the 100
longest prompts ran out of GPU memory on the 24 GB Mac; accuracy and ECE are close to the bars on 400 items.
(5) [`Remek/basal-omni-1.0-max-MLX`](https://huggingface.co/Remek/basal-omni-1.0-max-MLX) on the same M4 Pro (24 GB),
8 in flight: one question takes 453 ms; long prompts can run out of GPU memory at 24 GB. Its own
`CALIBRATION.mlx.json` automates 36% of decisions at `max_error` 1% and 85% at 5%.
(6) [`Remek/basal-omni-1.1-max-MLX`](https://huggingface.co/Remek/basal-omni-1.1-max-MLX), against basal-omni-1.1-max
bf16, on the same M4 Pro (24 GB), 8 in flight, 18-19 Oct: every 8th development decision (1,261 of 1,325 ran; 1,039
with 2-10 options). One question takes 525 ms; long prompts can run out of GPU memory at 24 GB. Its own
`CALIBRATION.mlx.json` (calibration run on every 2nd item, 4,280 test decisions) automates 35% of decisions at
`max_error` 1% and 80% at 5%.
For comparison on the same GPU: eager bf16 985 s (mini) and 2,157 s (max) for the 10,606 decisions at 8 in flight, vLLM
bf16 415 s and 840 s; GGUF Q8_0 646 s and 1,186 s at 16. Use Q8_0 (or Q6_K where memory is short); Q5_K_M and Q4_K_M
move too many close decisions.

## Images and audio (`--media on`)

`basal-serve --model <basal-omni model> --mode eager --media on` loads the model with Gemma 4's image and audio encoders
and its processor (install `basal[media]`: pillow, soundfile, scipy, torchvision matching torch). A request may then
carry `"media": [{"type": "image" | "audio", "data": "<base64>"}]`, `"detail": "default" | "high"` (280 or 1,120
image tokens) and `"lang": "pl" | "en"`; see `basal/media.py` for the formats and limits. The media go where the state
text goes, before it (`prompt.media_messages`), the prompt the basal-omni evaluation reads media with. Each prompt
runs alone (no batching); text requests on the same server take the unchanged text path. Answers carry
`"calibration": "text"`, and `usage.media` counts the items and their tokens.

Checked 2026-10-10 on Gemma 4 E4B and 12B: served answers against the fp32 readout of the same model
on 1,800 text items, 120 prompts over 2,048 tokens, 310 images, 285 audio clips and their text twins; every split is
within the bf16 readout's own deviation. On E4B the text answers with
`--media on` equal those of the earlier text-only engine on the same items.
