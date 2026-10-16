# basal-omni on the basal engine (Gemma 4)

basal-omni-1.0 are experimental typed-decision models on Google Gemma 4: basal-omni-1.0-mini (4B, Gemma 4 E4B) and
basal-omni-1.0-max (12B, Gemma 4 12B), with prompt format 2 ([FORMAT2.md](FORMAT2.md)): up to 255 options, one forward pass per
question. They are trained on text; with `--media on` the server also takes images and audio in a request (below).

## Serve

```bash
uv pip install "basal[media] @ https://github.com/rkinas/basal/archive/refs/tags/v2.1.0.tar.gz"
basal-serve --model Remek/basal-omni-1.0-max --mode eager --media on --port 8000    # or Remek/basal-omni-1.0-mini
```

- **Mode:** `eager` is the only mode for Gemma 4 in this release: the PyTorch forward in bf16 on CUDA.
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
- **Not yet for Gemma 4:** `fast` (shared prefix, CUDA graphs), `fp8`, `nvfp4`, vLLM, SGLang, MLX and GGUF. They come
  with a later basal-omni release.

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
