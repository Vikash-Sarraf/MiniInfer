# Benchmarks

This document records local MiniInfer benchmark runs. The numbers are not meant to compete with production inference engines; they are evidence that the runtime can measure the effect of specific implementation choices, especially backend selection and KV-cache decoding.

## Environment

```text
Date: 2026-09-08
OS: Microsoft Windows 11 Pro
CPU: 13th Gen Intel(R) Core(TM) i7-13800H
Rust: rustc 1.89.0 (29483883e 2025-08-04)
Cargo: cargo 1.89.0 (c24e10642 2025-06-23)
Build profile: release
Backend: ndarray
Model: local GPT-2 MiniInfer artifact at models/gpt2-miniinfer
```

## Commands

Main cache comparison:

```powershell
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hey I bet you're wondering how I got into this situation" --max-new-tokens 60 --compare-cache --runs 5
```

Short smoke comparison:

```powershell
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hello world" --max-new-tokens 30 --compare-cache
```

If `--runs` is omitted, `bench-generate` defaults to one run and prints the same single-run timing fields. When `--runs` is greater than one, timing fields are reported as min/median/max summaries.

Matmul backend comparison:

```powershell
cargo run --release -p miniinfer-cli -- bench-matmul
```

## KV-Cache Generation Results

| Prompt                                                     | Prompt tokens | Generated tokens | Final tokens | No-cache generation | No-cache tok/s | KV-cache generation | KV-cache tok/s | Speedup | Outputs match |
| ---------------------------------------------------------- | ------------: | ---------------: | -----------: | ------------------: | -------------: | ------------------: | -------------: | ------: | ------------- |
| `Hey I bet you're wondering how I got into this situation` |            12 |               60 |           72 |              9.867s |          6.081 |              4.596s |         13.054 |  2.147x | true          |
| `Hello world`                                              |             2 |               30 |           32 |              3.384s |          8.866 |              2.462s |         12.186 |  1.374x | true          |

## Int8 Weight Artifact Results

Per-channel int8 conversion stores selected GPT-2 block matrix weights as `i8_symmetric` tensors with one scale per output column, then dequantizes them to FP32 during model load.

| Artifact | `weights.bin` size | Reduction |
| --- | ---: | ---: |
| FP32 | 497,759,232 bytes | 1.00x |
| Per-channel int8 | 242,955,264 bytes | 2.05x |

Logit drift for prompt `Hello world`, comparing MiniInfer per-channel int8 against Hugging Face FP32 for the top 10 Hugging Face logits:

| Metric | Value |
| --- | ---: |
| Max absolute difference | 0.46926117 |
| Mean absolute difference | 0.35725098 |
| Tolerance | 1.00000000 |
| Status | PASS |

| Prompt                                                     | Active cache payload | Allocated cache payload | Capacity cache payload |
| ---------------------------------------------------------- | -------------------: | ----------------------: | ---------------------: |
| `Hey I bet you're wondering how I got into this situation` |      5,308,416 bytes |        75,497,472 bytes |       75,497,472 bytes |
| `Hello world`                                              |      2,359,296 bytes |        75,497,472 bytes |       75,497,472 bytes |

The cache memory numbers are FP32 key/value payload bytes, not whole-process heap usage. `Active` reflects the tokens currently stored in the cache, while `Allocated` and `Capacity` reflect the preallocated per-layer, per-head buffers.

## Time to First Token

| Prompt                                                     | No-cache TTFT | KV-cache TTFT | Note                                                       |
| ---------------------------------------------------------- | ------------: | ------------: | ---------------------------------------------------------- |
| `Hey I bet you're wondering how I got into this situation` |        0.090s |        0.366s | 5-run median                                               |
| `Hello world`                                              |        0.085s |        0.100s | Short prompts have very little prefill work either way.    |

The current KV-cache implementation uses optimized full-prompt prefill, then one-token cached decode for generated tokens. KV-cache TTFT can still be higher than the no-cache first step because it fills the whole prompt cache up front, but the gap is much smaller than the earlier token-by-token prefill path.

## KV-Cache Phase Timing

| Prompt                                                     | Prompt prefill | Decode time | Decode tok/s |
| ---------------------------------------------------------- | -------------: | ----------: | -----------: |
| `Hey I bet you're wondering how I got into this situation` |         0.365s |      4.230s |       14.183 |
| `Hello world`                                              |         0.099s |      2.361s |       12.705 |

Decode time includes sampling generated tokens and updating the cache through the final generated token.

## Example Output

Main comparison output:

```text
Backend: ndarray
Runs: 5
Prompt tokens: 12
Requested tokens: 60
Load time: 0.298s
Encode time: 0.002s

No cache:
Generated tokens: 60
Final tokens: 72
Time to first token: min 0.086s / median 0.090s / max 0.095s
Generation time: min 9.581s / median 9.867s / max 10.176s
Tokens/sec: min 5.896 / median 6.081 / max 6.262

KV cache:
Generated tokens: 60
Final tokens: 72
Time to first token: min 0.355s / median 0.366s / max 0.411s
Prompt prefill time: min 0.355s / median 0.365s / max 0.410s
Decode time: min 3.994s / median 4.230s / max 4.732s
Decode tokens/sec: min 12.679 / median 14.183 / max 15.024
Generation time: min 4.350s / median 4.596s / max 5.144s
Tokens/sec: min 11.664 / median 13.054 / max 13.792
KV cache memory:
  Active: 5308416 bytes (5.062 MiB)
  Allocated: 75497472 bytes (72.000 MiB)
  Capacity: 75497472 bytes (72.000 MiB)

Speedup:
Generation time: 2.147x
Tokens/sec: 2.147x
Outputs match: true
Total time: 72.904s
```

Short comparison output:

```text
Backend: ndarray
Prompt tokens: 2
Requested tokens: 30
Load time: 0.289s
Encode time: 0.001s

No cache:
Generated tokens: 30
Final tokens: 32
Time to first token: 0.085s
Generation time: 3.384s
Tokens/sec: 8.866

KV cache:
Generated tokens: 30
Final tokens: 32
Time to first token: 0.100s
Prompt prefill time: 0.099s
Decode time: 2.361s
Decode tokens/sec: 12.705
Generation time: 2.462s
Tokens/sec: 12.186
KV cache memory:
  Active: 2359296 bytes (2.250 MiB)
  Allocated: 75497472 bytes (72.000 MiB)
  Capacity: 75497472 bytes (72.000 MiB)

Speedup:
Generation time: 1.374x
Tokens/sec: 1.374x
Outputs match: true
Total time: 6.136s
```

## Interpretation

The no-cache path recomputes the full sequence on each generated token. As the generated sequence grows, each decode step gets more expensive.

The KV-cache path stores keys and values per layer and per head. Prompt prefill fills the cache in one full-prompt pass, and each decode step computes attention for only the current token against cached keys and values. For these local runs, that produces a `1.374x` to `2.147x` generation throughput improvement.

The outputs match because the comparison uses greedy decoding. Matching output is a useful correctness signal: both paths sampled the same token sequence while using different attention execution strategies.

## Known Benchmark Gaps

- KV-cache reporting currently covers FP32 key/value payload bytes, not allocator overhead or total process memory.
- Cached decode attention reads borrowed key/value cache slices directly; owned tensor readback remains for tests and inspection.
- Current recorded results are local measurements; use `--runs` to reduce single-run timing noise.
- CPU frequency scaling, background load, and thermal state can affect these numbers.

## Next Measurements

Useful next benchmark improvements:

```text
1. Add process-level memory measurements for allocator overhead and temporary tensors.
2. Compare reference backend vs ndarray backend on generation, not only matmul.
3. Add true int8 weight-only matmul and compare it against dequantize-on-load artifacts.
4. Track prefill and decode timings across multiple prompt lengths.
```
