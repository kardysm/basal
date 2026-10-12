# AMD NPU: Strix Halo and other Ryzen AI processors (`--mode npu`)

`basal-serve --mode npu` runs basal on the NPU (XDNA 2) of AMD Ryzen AI processors: **Strix Halo** (Ryzen AI Max /
Max+ 300, e.g. the Max+ 395), Strix Point and Krackan Point (Ryzen AI 300). It goes through
[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) (OGA), the LLM runtime of AMD's Ryzen AI Software,
with a model folder that AMD's tools prepare for the NPU. Only the weights come from that folder. The tokenizer, chat
template and `CALIBRATION.json` come from `--model`, so the token ids are those of every other mode. The HTTP API, both
option orders, SOAM, `multi`, `act` and `facts` work as in `--mode fast`. Evidence spans are not available: they need
the PyTorch hidden states.

A basal decision is one prefill and no decoding. Ryzen AI's *hybrid* models run prefill on the NPU and decoding on the
iGPU, so with a hybrid model a basal decision runs entirely on the NPU, as with an NPU-only model.

> [!IMPORTANT]
> **Status: not yet measured on an NPU.** No Ryzen AI checkpoint of basal is published; you prepare the folder yourself
> (below). The backend's readout and prefix sharing are tested against PyTorch with the CPU build of onnxruntime-genai
> on a tiny model with basal's layout (`tests/test_npu.py`: same probabilities within 1e-4). The accuracy of the 4-bit
> NPU port, its speed and whether AMD's NPU recipes accept basal's architecture have not been measured: check your
> folder with `basal-bench` before you rely on it, and refit the confidence thresholds (the bf16 calibration is not
> validated for this port).

## How it works

NPU models take one sequence at a time. The backend keeps one KV cache and evaluates the prompts of a batch in
lexicographic token order, the depth-first order of their prefix trie. Each prompt rewinds the cache to its common
prefix with the previous prompt (`rewind_to`) and appends only the rest. So a shared prefix is computed once: the system
prompt, the state of a request (with SOAM, for every question and option order) and a question's text for its two
option orders. Every readout still sees exactly its own prompt. The answer is the softmax over the option letters'
logits at the last position (`get_logits`), as in the other modes.

At start-up the backend checks that `rewind_to` gives the same next-token distribution as a fresh prefill (execution
providers differ). If it fails, the server warns and computes every prompt in full: same answers, more work.

The context limit applies per prompt, not per request: the cache holds one prompt at a time. Prompts longer than the
folder's `context_length` (in `genai_config.json`) are refused with HTTP 422.

## Requirements

| | Windows 11 | Linux |
|---|---|---|
| Ryzen AI Software | 1.8 (lists Strix, Strix Halo and Krackan Point) | 1.8 for Linux (Ubuntu 24.04 packages; XRT and the `amdxdna` driver) |
| Ryzen AI model types | hybrid (NPU prefill, iGPU decode) and NPU-only | NPU-only; hybrid is Windows-only in 1.8 |
| preparing a model (`model_generate`) | yes | no (1.8); folders prepared on Windows run on Linux |
| Python OGA | `onnxruntime-genai-directml-ryzenai==0.14.0` (AMD's package index, part of the Ryzen AI install) | not verified: AMD documents LLMs on Linux through its C++ runtime (`deployment/`, `model_benchmark`) |

`--mode npu` needs the Python module `onnxruntime_genai` *with AMD's RyzenAI execution provider*. Install basal into
the Ryzen AI environment and do **not** `pip install onnxruntime-genai` there: the PyPI package has the same module
name, replaces AMD's build and has no NPU provider. basal does not depend on it.

## 1. Prepare the model (once)

AMD's recipe for fine-tuned models: 4-bit AWQ weights with [AMD Quark](https://quark.docs.amd.com/) on a Linux machine
with an AMD Instinct or NVIDIA GPU, then `model_generate` on a Windows PC with Ryzen AI. The exact flags and versions are
on the model preparation page of the Ryzen AI documentation (`oga_model_prepare`); the steps below are that recipe
applied to basal.

**a. Download the bf16 model and make its config readable for transformers 4.** AMD's Quark recipe pins
transformers 4.57.6. transformers 5 writes the rotary base as `rope_parameters`, which 4.57.6 does not read: it would
load the model with `rope_theta` 10000 instead of 1000000 and quantise a broken model without an error (the MLX ports
needed the same fix, [PORTS.md](PORTS.md#how-the-ports-are-made)). A tokenizer saved by transformers 5 may also name a
tokenizer class (`TokenizersBackend`) that 4.57.6 does not know. Both fixes keep the folder loadable with
transformers 5:

```bash
hf download Remek/basal-1.5-4.5B --local-dir basal-1.5-4.5B
python - basal-1.5-4.5B <<'EOF'   # any environment with basal installed
import json, sys
from pathlib import Path
from basal.engine import mlx_config_overrides  # rope settings for readers older than transformers 5
d = Path(sys.argv[1])
cfg = json.loads((d / "config.json").read_text())
(d / "config.json").write_text(json.dumps({**cfg, **mlx_config_overrides(cfg)}, indent=2))
tc = json.loads((d / "tokenizer_config.json").read_text())
if tc.get("tokenizer_class") == "TokenizersBackend":  # transformers 5 name, unknown to 4.57
    tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    (d / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
EOF
```

**b. Quantise with Quark** (Linux, GPU; Python 3.12, Quark 0.11, `transformers==4.57.6`, as on AMD's page), from
Quark's `examples/torch/language_modeling/llm_ptq/`:

```bash
python quantize_quark.py --no_trust_remote_code --model_dir basal-1.5-4.5B --output_dir basal-1.5-4.5B-awq \
  --quant_scheme uint4_wo_128 --quant_algo awq --dataset pileval_for_awq_benchmark --num_calib_data 128 \
  --seq_len 512 --model_export hf_format --data_type bfloat16
```

AMD's example adds `--exclude_layers []`, which quantises every layer including the output head. basal reads its
decision from the output head (the MLX ports keep it in bf16), so leave the flag out: Quark then applies its
model-specific exclusions, which can keep the head unquantised. If `model_generate` refuses that, quantise the head
with a smaller group instead, as AMD does for Phi-4: `--exclude_layers [] --layer_quant_scheme lm_head uint4_wo_32`.

**c. Prepare for the NPU** (Windows PC with Ryzen AI 1.8, `conda activate ryzen-ai-1.8.0`):

```bash
# hybrid (Windows only): prefill, i.e. the whole basal decision, on the NPU
model_generate --hybrid --input basal-1.5-4.5B-awq --output basal-1.5-4.5B-rai-hybrid
# NPU-only (also runs on Linux): AMD's safe default for new or untested architectures
model_generate --npu --basic --input basal-1.5-4.5B-awq --output basal-1.5-4.5B-rai-npu
```

basal is a Llama model with attention **and MLP** biases (Bielik v3). AMD lists Llama 3.x (no biases) and Qwen2.5
(attention biases) among the supported models; MLP biases in a Llama model are not on that list. The OGA model builder
inside `model_generate` converts them (checked in `tests/test_npu.py`); whether the fused NPU recipes
(`--npu --full_fusion`, best performance, 4,096 tokens; `--npu --token_fusion --extra_options max_seq_len=16384`, long
context) accept them is not verified. Start with `--hybrid` or `--npu --basic`. If `model_generate` stops on operator
shapes, the Ryzen AI documentation's page on new operator shapes (`oga_op_prepare`) covers that case. `--mem_optimize`
helps on 16 GB machines.

## 2. Serve

The prepared folder goes to `--onnx`, the original model (tokenizer, template, calibration) to `--model`; in this mode
`--model` is downloaded without its weights. `--onnx` can also be a Hugging Face repository id. The `npu` mode is newer
than the v1.5.0 release: install basal from a checkout that includes it.

**Windows** (in the Ryzen AI environment):

```bash
conda create --name basal-npu --clone ryzen-ai-1.8.0 && conda activate basal-npu
pip install -e .   # in a checkout of this repository; leaves AMD's onnxruntime-genai in place
basal-serve --model Remek/basal-1.5-4.5B --mode npu --onnx basal-1.5-4.5B-rai-hybrid --port 8000
```

The clone keeps the Ryzen AI environment itself unchanged (basal pins its own transformers version).

**Linux** (NPU-only folder; not verified, see Requirements). The environment follows AMD's Linux install and LLM pages:

```bash
source <TARGET-PATH>/venv/bin/activate && source /opt/xilinx/xrt/setup.sh
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:<TARGET-PATH>/venv/deployment/lib:${RYZEN_AI_INSTALLATION_PATH}/onnxruntime/lib/:$LD_LIBRARY_PATH
export RYZENAI_EP_PATH=<TARGET-PATH>/venv/deployment/lib/libonnxruntime_providers_ryzenai.so
pip install -e .   # in a checkout of this repository
basal-serve --model Remek/basal-1.5-4.5B --mode npu --onnx basal-1.5-4.5B-rai-npu --port 8000
```

The request and response are those of every other mode ([README](../README.md#api)). Requests are served one after
another (one sequence on the NPU); requests that arrive together go through one call, where prompts in the same
language share at least the system prompt.

## 3. Check the port

```bash
basal-bench --model Remek/basal-1.5-4.5B --modes eager-fp32 npu --onnx basal-1.5-4.5B-rai-hybrid --out npu.json
```

The first mode is the reference: `eager-fp32` runs the bf16 weights in fp32 on the CPU (about 18 GB for the 4.5B
model; slow, but it needs no GPU). `agree` is the share of decisions with the reference's top answer. For comparison, the
4-bit GGUF and MLX ports of basal-1.5 (4.5B) agree with bf16 on 0.970–0.980 of 500 development decisions
([QUANTIZATION.md](QUANTIZATION.md#ports-agreement-and-speed)); an AWQ uint4 port has not been measured. Use your own
items (`--questions my_items.jsonl`) for numbers that describe your workload. Several folders at once:
`--modes eager-fp32 npu@<folder-a> npu@<folder-b>`.

The server reads `CALIBRATION.npu.json` from `--model` when present, else `CALIBRATION.json` (`--calibration FILE`
overrides it). The thresholds in `CALIBRATION.json` were certified for the bf16 engine: refit them on a few hundred of
your labelled requests before you accept decisions automatically ([README](../README.md#using-the-confidence)).

## Notes

- **Context.** Full-fusion NPU models take 4,096 tokens (input and output), token-fusion models up to 16K; for hybrid
  models AMD documents `"hybrid_opt_chunk_context": "1"` (under `provider_options` / `RyzenAI`) and `"chunk_size": 2048`
  (under `search`) in `genai_config.json` for prompts over 4,096 tokens.
- **Validation.** With a full model directory as `--model`, the server checks that layers, hidden size and vocabulary in
  its `config.json` match `genai_config.json` before it loads the weights.
- **Other OGA folders.** The mode runs any ONNX Runtime GenAI folder with the execution provider its
  `genai_config.json` names: for example a CPU export from the OGA model builder
  (`python -m onnxruntime_genai.models.builder -i basal-1.5-4.5B -o basal-oga-cpu -p fp32 -e cpu`, with the
  PyPI `onnxruntime-genai` in a separate environment), which checks the pipeline without an NPU. The tests use such an
  export of a tiny model; they need `onnxruntime-genai`, `onnxruntime` and `onnx-ir` and are skipped without them.
- **Linux troubleshooting** (reported, not verified here): some models need an unlimited memlock limit (AMD:
  `* soft memlock unlimited` and `* hard memlock unlimited` in `/etc/security/limits.d/`); recent systems refuse to
  load libraries that request an executable stack, and `patchelf --clear-execstack` removes that request from the
  affected `.so` files.
- **The iGPU of the same chip** (Radeon 8060S on Strix Halo) can serve the GGUF ports with `--mode gguf` (a Vulkan or
  ROCm build of llama-cpp-python) or `--mode llamacpp` ([GGUF.md](GGUF.md)); not measured either.
