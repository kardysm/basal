# Parity against bf16 eager: raw answers

Measured on 19 Oct 2026, one engine at a time on one port. Summary and table:
[STRIX_HALO.md, "Parity against bf16 eager"](https://github.com/rkinas/basal/blob/main/docs/benchmarks/STRIX_HALO.md#parity-against-bf16-eager-2000-public-decisions).

- Machine: AMD Ryzen AI Max+ PRO 395, Radeon 8060S (gfx1151), 128 GB; Fedora 44, kernel 7.2.10.
- Software: basal engine 2.2.0 (v2.2.0, e49665c), PyTorch 2.12.0 + ROCm 7.2, Triton 3.7.0, transformers 5.17.0,
  llama-cpp-python 0.3.36 built with Vulkan.
- Model: `Remek/basal-2.0-4.5B` at e5ba914 (bf16); `Remek/basal-2.0-4.5B-GGUF` at 237294c (`Q8_0`) for `gguf`. Every
  engine used the model's `CALIBRATION.json`, one option order (the 2.0 default).
- Data: [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) at 7189f46
  (Apache-2.0), `all/test-00000-of-00001.parquet`: 400 cases, 2,000 decisions. The answer files repeat the score
  legends of its questions.

| file | server (`basal-serve --model <snapshot> --name basal-2.0-4.5B --host 127.0.0.1 --port 8031 ...`) | `basal-run` | time |
|---|---|---|---|
| `eager.jsonl` (reference) | `--mode eager --soam off --max-batch 1` | `--concurrency 1` | 933 s |
| `eager-packed.jsonl` | `--mode eager` | `--concurrency 1` | 910 s |
| `strix.jsonl` | `--mode strix` | `--concurrency 1` | 134 s |
| `strix-c8.jsonl` | `--mode strix` (same server as above) | `--concurrency 8` | 126 s |
| `gguf.jsonl` | `--mode gguf --model <-GGUF folder> --gguf basal-2.0-4.5B-Q8_0.gguf` | `--concurrency 1` | 223 s |

Every request was answered (0 errors). `parity.json` is the output of:

```
uv run --no-project --with pyarrow python ../../../parity.py cases all-test.parquet cases.jsonl
python ../../../parity.py compare cases.jsonl eager.jsonl eager-packed.jsonl strix.jsonl strix-c8.jsonl gguf.jsonl \
    --out parity.json
```
