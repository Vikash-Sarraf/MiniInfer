# KV Cache

MiniInfer uses a key-value cache to avoid recomputing attention keys and values for every previously processed token during autoregressive generation.

Without a KV cache, each generated token runs the model over the full sequence so far:

```text
prompt + generated tokens
  -> embed every token
  -> run every block over the full sequence
  -> recompute all attention keys and values
  -> sample one next token
```

With a KV cache, previously computed keys and values are stored per transformer layer. During decode, MiniInfer computes keys and values only for the new token, appends them to the cache, and lets the new token's query attend over the cached rows.

```text
new token
  -> compute q/k/v for current token
  -> append current k/v to per-layer cache
  -> query attends over cached keys/values
  -> sample one next token
```

## Cache Layout

The cache lives in `crates/miniinfer-core/src/runtime/kv_cache.rs`.

```text
KvCache
  layers: Vec<LayerKvCache>

LayerKvCache
  keys: Vec<Vec<f32>>
  values: Vec<Vec<f32>>
  num_heads: usize
  head_dim: usize
  max_seq_len: usize
  seq_len: usize
```

Each `LayerKvCache` stores one key buffer and one value buffer per attention head:

```text
keys[head]   = flat rows for shape [seq_len, head_dim]
values[head] = flat rows for shape [seq_len, head_dim]
```

For GPT-2 small, the cache shape is logically:

```text
num_layers = 12
num_heads = 12
head_dim = 64
max_seq_len = 1024
```

Each append adds one `[1, head_dim]` key row and one `[1, head_dim]` value row per head. Cached decode borrows the flat per-head slices directly. Owned readback methods still convert the flat storage back into tensors with shape `[seq_len, head_dim]` for tests and inspection.

Prompt prefill uses `append_many` to add all prompt key/value rows for each layer in one full-prompt pass. Decode still uses `append` because each generated token contributes one new row per head.

The cache reports three payload byte counts:

```text
active_bytes    = seq_len * num_heads * head_dim * 2 key/value buffers * sizeof(f32)
capacity_bytes  = max_seq_len * num_heads * head_dim * 2 key/value buffers * sizeof(f32)
allocated_bytes = reserved Vec<f32> capacity for all key/value buffers * sizeof(f32)
```

For GPT-2 small, `capacity_bytes` and initial `allocated_bytes` are both `75,497,472` bytes, or `72.000 MiB`, because MiniInfer preallocates all per-layer cache buffers up to the model context length.

## Decode Attention

Normal full-sequence attention computes square attention scores:

```text
query:  [seq_len, head_dim]
key:    [seq_len, head_dim]
scores: [seq_len, seq_len]
```

That path uses a causal softmax because earlier tokens must not attend to future tokens.

Cached decode computes attention only for the current token:

```text
query:        [1, head_dim]
cached key:   [cache_seq_len, head_dim]
scores:       [1, cache_seq_len]
cached value: [cache_seq_len, head_dim]
output:       [1, head_dim]
```

This path uses row-wise softmax, not causal softmax. The cache contains only past tokens plus the current token, so there are no future positions to mask.

Per-head cached attention is:

```text
scores = query @ cached_key.T / sqrt(head_dim)
probabilities = row_softmax(scores)
context = probabilities @ cached_value
```

The per-head outputs are merged back into `[1, hidden_size]`, projected through the attention output projection, and then passed through the usual residual and MLP block path.

## Current Generation Flow

The cached generation entry points are in `crates/miniinfer-core/src/runtime/generation.rs`:

```text
GenerationOptions::generate_with_kv_cache_and_backend
GenerationOptions::generate_streaming_with_kv_cache_and_backend
```

The model dispatch boundary is in `LoadedModel`:

```text
LoadedModel::forward_prefill_with_cache_and_backend
LoadedModel::forward_next_token_with_cache_and_backend
```

The GPT-2 implementation is in `Gpt2Weights`:

```text
Gpt2Weights::forward_prefill_with_cache_and_backend
Gpt2Weights::forward_next_token_with_cache_and_backend
```

The current cached generation flow is:

```text
1. Create KvCache from model config.
2. Run full-prompt prefill once and write prompt keys/values into the cache.
3. Use the final prompt-token logits to sample the first generated token.
4. Feed each generated token through cached one-token forward.
5. Stop on EOS or max_new_tokens.
6. Decode the final token sequence.
```

## CLI Usage

Run generation with the cache:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer --prompt "Once upon a time" --max-new-tokens 60 --kv-cache
```

Run streamed generation with the cache:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer --prompt "Once upon a time" --max-new-tokens 60 --stream --kv-cache
```

Compare cached and uncached benchmark runs:

```powershell
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hey I bet you're wondering how I got into this situation" --max-new-tokens 60 --compare-cache --runs 5
```

`--runs` defaults to `1` when omitted. Values above `1` report min/median/max timing summaries.

## Benchmark Result

Example local result using GPT-2 small with the ndarray backend:

```text
Prompt tokens: 12
Generated tokens: 60

No cache:
  Generation time: 9.867s median
  Tokens/sec: 6.081 median

KV cache:
  Time to first token: 0.366s median
  Prompt prefill time: 0.365s median
  Decode time: 4.230s median
  Decode tokens/sec: 14.183 median
  Generation time: 4.596s median
  Tokens/sec: 13.054 median
  Active cache payload: 5308416 bytes (5.062 MiB)
  Allocated cache payload: 75497472 bytes (72.000 MiB)
  Capacity cache payload: 75497472 bytes (72.000 MiB)

Speedup:
  Generation time: 2.147x
  Tokens/sec: 2.147x

Outputs match: true
```

The matching output confirms that cached greedy generation follows the same token path as uncached greedy generation for this 5-run benchmark.

See [benchmarks.md](benchmarks.md) for the full benchmark command, environment details, and a second shorter cache comparison.

## Current Limitations

The current implementation prioritizes correctness and explainability over peak performance.

- Prompt prefill already fills the cache in one full-prompt pass, but it is still optimized for readability over peak throughput.
- Cached decode attention reads borrowed key/value slices directly; owned tensor readback remains available for tests and inspection.
- KV-cache memory reporting covers key/value payload bytes, not allocator metadata, temporary tensors, model weights, tokenizer data, or total process memory.
- `bench-generate` uses greedy decoding only, which is useful for deterministic speed comparisons.

KV-cache time to first token can still be higher than the no-cache first step because the cache path fills the full prompt cache before sampling. Decode throughput improves because generated tokens reuse cached keys and values instead of recomputing attention over the full generated sequence.

## Next Improvements

Possible follow-up work:

```text
1. Add process-level memory measurements for allocator overhead and temporary tensors.
2. Track prefill and decode timings across multiple prompt lengths.
3. Add prefix-cache extension after the single-prompt prefill path is fully benchmarked.
```
