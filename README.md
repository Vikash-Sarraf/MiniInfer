# MiniInfer

MiniInfer is a CPU-first Rust inference runtime built as a serious engineering and resume project. The V1 target is a complete GPT-2-style decoder-only inference path that owns model conversion, model loading, tokenization, generation, sampling, streaming, KV-cache decoding, tests, and benchmarks.

The goal is not to wrap an existing model runner. MiniInfer keeps the full inference pipeline visible and explainable, while allowing optimized math libraries behind an explicit backend boundary.

## Current Features

- GPT-2-style decoder-only model loading from MiniInfer artifacts.
- Hugging Face GPT-2 checkpoint conversion through `tools/convert_gpt2_pytorch.py`.
- Real GPT-2 byte-level BPE tokenizer files copied into the converted model artifact.
- FP32 tensor runtime with reference and ndarray-backed operation backends.
- Autoregressive generation with EOS stopping from model config.
- Greedy, temperature, top-k, and top-p sampling.
- CLI streaming output.
- Per-layer KV-cache decode path with optimized full-prompt prefill for GPT-2 generation.
- Benchmark commands for matmul, no-cache vs KV-cache generation, and KV-cache payload memory.
- Mixed FP32/int8 binary weight artifacts with per-channel symmetric int8 scales and dequantize-on-load support.

## Repository Layout

```text
crates/miniinfer-core   Runtime, tensor type, model loading, GPT-2 implementation, sampling, KV cache
crates/miniinfer-cli    Command-line interface for inspection, logits, generation, and benchmarks
tools/                  Model conversion utilities
tests/                  Integration and parity-oriented tests
docs/                   Design notes and benchmark reports
.github/docs/plan.md    Project roadmap and milestone plan
```

## Requirements

- Rust 1.89.0 or newer compatible with the workspace toolchain.
- Python with PyTorch installed when converting Hugging Face GPT-2 weights.
- Python with PyTorch and Transformers installed when running Hugging Face logits parity checks.
- Local GPT-2 source files for conversion. Large model artifacts should stay out of git.

## Build and Test

Build the workspace:

```powershell
cargo build
```

Run tests:

```powershell
cargo test
```

Build optimized binaries for benchmark runs:

```powershell
cargo build --release
```

## Correctness Checks

Compare MiniInfer final-token logits against a local Hugging Face GPT-2 checkpoint:

```powershell
.\.venv\Scripts\python.exe tools\check_gpt2_logits_parity.py --prompt "Hello world" --top-k 10
```

The parity tool loads Hugging Face GPT-2 with PyTorch, asks MiniInfer for the same selected token logits, and reports maximum and mean absolute difference. A passing run means the tokenizer IDs, converted weights, forward pass, and LM-head projection agree with the Hugging Face reference for the checked prompt.

## Convert GPT-2

MiniInfer expects converted model artifacts under a MiniInfer model directory. A typical local conversion command is:

```powershell
python tools/convert_gpt2_pytorch.py --source-dir models/gpt2 --output-dir models/gpt2-miniinfer --format binary --overwrite
```

The converter writes:

```text
config.json
weights.index.json
weights.bin
tokenizer.json or tokenizer files copied from the source directory
```

The binary weight format keeps large FP32 weights in `weights.bin` and stores tensor metadata in `weights.index.json`.

Per-channel int8 artifact conversion:

```powershell
.\.venv\Scripts\python.exe tools\convert_gpt2_pytorch.py --source-dir models/gpt2 --output-dir models/gpt2-miniinfer-int8-channel --format binary --quantization int8 --overwrite
```

The int8 converter stores selected GPT-2 block matrix weights as `i8_symmetric` tensors with `scales` and `scale_axis` metadata. MiniInfer currently dequantizes those weights back to FP32 during load.

## Run Generation

Greedy generation with the default ndarray backend:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer --prompt "Once upon a time" --max-new-tokens 60
```

Streaming generation:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer --prompt "Once upon a time" --max-new-tokens 60 --stream
```

KV-cache generation:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer --prompt "Once upon a time" --max-new-tokens 60 --kv-cache
```

Sampling example:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer --prompt "Once upon a time" --max-new-tokens 60 --temperature 0.8 --top-k 40 --top-p 0.9 --seed 42
```

GPT-2 is not instruction-tuned. Generated text can be repetitive, inconsistent, or factually wrong; the important runtime signal is that MiniInfer runs a real decoder-only model path end to end.

## Benchmarks

Compare no-cache and KV-cache generation:

```powershell
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hey I bet you're wondering how I got into this situation" --max-new-tokens 60 --compare-cache --runs 5
```

`bench-generate` defaults to one run when `--runs` is omitted. Use `--runs` above `1` to print min/median/max timing summaries.

Recent local result on Windows 11 Pro with a 13th Gen Intel Core i7-13800H:

| Prompt                                                     | Runs | Generated tokens | No-cache median tok/s | KV-cache median tok/s | Median speedup | Outputs match |
| ---------------------------------------------------------- | ---: | ---------------: | --------------------: | --------------------: | -------------: | ------------- |
| `Hey I bet you're wondering how I got into this situation` |    5 |               60 |                 6.081 |                13.054 |         2.147x | true          |
| `Hello world`                                              |    1 |               30 |                 8.866 |                12.186 |         1.374x | true          |

Per-channel int8 artifact result:

| Artifact         | `weights.bin` size | Logit max abs diff | Logit mean abs diff |
| ---------------- | -----------------: | -----------------: | ------------------: |
| FP32             |  497,759,232 bytes |                n/a |                 n/a |
| Per-channel int8 |  242,955,264 bytes |         0.46926117 |          0.35725098 |

For GPT-2 small, the current benchmark output also reports KV-cache payload memory. A full 1024-token cache reserves `75,497,472` bytes, or `72.000 MiB`, for FP32 keys and values.

See [docs/benchmarks.md](docs/benchmarks.md) for commands, environment details, and notes about benchmark limitations.

## Design Notes

- [docs/kv-cache.md](docs/kv-cache.md) explains the current KV-cache layout, decode flow, benchmark result, and limitations.
- [docs/benchmarks.md](docs/benchmarks.md) records reproducible benchmark commands and current local results.
- [docs/quantization.md](docs/quantization.md) explains the current int8 artifact format, dequantize-on-load path, and measured drift.
- [.github/docs/plan.md](.github/docs/plan.md) tracks the larger V1/V1.5 roadmap.

## Current Limitations

- GPT-2 is the only implemented model architecture.
- CPU is the only runtime target.
- Int8 weight artifacts currently dequantize to FP32 at load time; true int8 matmul is future work.
- Benchmark results are local measurements and can be summarized across repeated runs with `--runs`.
- The runtime is an inference engine, not a production server.

## Resume Signals

MiniInfer currently demonstrates:

- custom model artifact conversion and loading
- real tokenizer integration
- decoder-only transformer inference in Rust
- explicit backend boundary for reference and optimized CPU operations
- deterministic and stochastic generation controls
- streaming text output
- per-layer KV-cache decode with optimized prompt prefill, benchmarked speedup, and payload memory reporting
- per-channel int8 weight artifact compression with measured logit drift against a Hugging Face FP32 reference
- correctness-focused tests and reproducible benchmark commands
