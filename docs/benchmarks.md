# Benchmarks

This document records local MiniInfer benchmark runs. The numbers are not meant to compete with production inference engines; they are evidence that the runtime can measure the effect of specific implementation choices, especially backend selection and KV-cache decoding.

## Environment

```text
Date: 2026-09-05
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
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hey I bet you're wondering how I got into this situation" --max-new-tokens 60 --compare-cache
```

Short smoke comparison:

```powershell
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hello world" --max-new-tokens 30 --compare-cache
```

Matmul backend comparison:

```powershell
cargo run --release -p miniinfer-cli -- bench-matmul
```

## KV-Cache Generation Results

| Prompt                                                     | Prompt tokens | Generated tokens | Final tokens | No-cache generation | No-cache tok/s | KV-cache generation | KV-cache tok/s | Speedup | Outputs match |
| ---------------------------------------------------------- | ------------: | ---------------: | -----------: | ------------------: | -------------: | ------------------: | -------------: | ------: | ------------- |
| `Hey I bet you're wondering how I got into this situation` |            12 |               60 |           72 |             21.400s |          2.804 |             12.195s |          4.920 |  1.755x | true          |
| `Hello world`                                              |             2 |               30 |           32 |              6.467s |          4.639 |              4.698s |          6.385 |  1.376x | true          |

| Prompt                                                     | Active cache payload | Allocated cache payload | Capacity cache payload |
| ---------------------------------------------------------- | -------------------: | ----------------------: | ---------------------: |
| `Hey I bet you're wondering how I got into this situation` |      5,308,416 bytes |        75,497,472 bytes |       75,497,472 bytes |
| `Hello world`                                              |      2,359,296 bytes |        75,497,472 bytes |       75,497,472 bytes |

The cache memory numbers are FP32 key/value payload bytes, not whole-process heap usage. `Active` reflects the tokens currently stored in the cache, while `Allocated` and `Capacity` reflect the preallocated per-layer, per-head buffers.

## Time to First Token

| Prompt                                                     | No-cache TTFT | KV-cache TTFT | Note                                                               |
| ---------------------------------------------------------- | ------------: | ------------: | ------------------------------------------------------------------ |
| `Hey I bet you're wondering how I got into this situation` |        0.226s |        1.928s | KV cache currently pre-fills the prompt token by token.            |
| `Hello world`                                              |        0.135s |        0.265s | Short prompts reduce the prefill penalty, but it is still visible. |

The current KV-cache implementation improves generation throughput after prefill, but it does not yet optimize prompt prefill. This is why total generation speed improves while time to first token is slower with `--kv-cache`.

## Example Output

Main comparison output:

```text
Backend: ndarray
Prompt tokens: 12
Requested tokens: 60
Load time: 0.639s
Encode time: 0.003s

No cache:
Generated tokens: 60
Final tokens: 72
Time to first token: 0.226s
Generation time: 21.400s
Tokens/sec: 2.804

KV cache:
Generated tokens: 60
Final tokens: 72
Time to first token: 1.928s
Generation time: 12.195s
Tokens/sec: 4.920
KV cache memory:
  Active: 5308416 bytes (5.062 MiB)
  Allocated: 75497472 bytes (72.000 MiB)
  Capacity: 75497472 bytes (72.000 MiB)

Speedup:
Generation time: 1.755x
Tokens/sec: 1.755x
Outputs match: true
Total time: 34.237s
```

Short comparison output:

```text
Backend: ndarray
Prompt tokens: 2
Requested tokens: 30
Load time: 0.424s
Encode time: 0.002s

No cache:
Generated tokens: 30
Final tokens: 32
Time to first token: 0.135s
Generation time: 6.467s
Tokens/sec: 4.639

KV cache:
Generated tokens: 30
Final tokens: 32
Time to first token: 0.265s
Generation time: 4.698s
Tokens/sec: 6.385
KV cache memory:
  Active: 2359296 bytes (2.250 MiB)
  Allocated: 75497472 bytes (72.000 MiB)
  Capacity: 75497472 bytes (72.000 MiB)

Speedup:
Generation time: 1.376x
Tokens/sec: 1.376x
Outputs match: true
Total time: 11.592s
```

## Interpretation

The no-cache path recomputes the full sequence on each generated token. As the generated sequence grows, each decode step gets more expensive.

The KV-cache path stores keys and values per layer and per head. Each decode step computes attention for only the current token against cached keys and values. For these local runs, that produces a `1.376x` to `1.755x` generation throughput improvement.

The outputs match because the comparison uses greedy decoding. Matching output is a useful correctness signal: both paths sampled the same token sequence while using different attention execution strategies.

## Known Benchmark Gaps

- Prompt prefill and decode timing are not reported separately yet.
- KV-cache reporting currently covers FP32 key/value payload bytes, not allocator overhead or total process memory.
- The cached attention path currently clones cached key/value buffers into tensors during readback.
- Results are local single-run measurements, not averaged benchmark suites.
- CPU frequency scaling, background load, and thermal state can affect these numbers.

## Next Measurements

Useful next benchmark improvements:

```text
1. Report prefill time separately from decode time.
2. Add process-level memory measurements for allocator overhead and temporary tensors.
3. Run averaged benchmark samples with min/median/max.
4. Compare reference backend vs ndarray backend on generation, not only matmul.
5. Add an int8 weight-only benchmark after quantization is implemented.
```
