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

Each append adds one `[1, head_dim]` key row and one `[1, head_dim]` value row per head. Readback methods convert the flat storage back into tensors with shape `[seq_len, head_dim]`.

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
LoadedModel::forward_next_token_with_cache_and_backend
```

The GPT-2 implementation is in `Gpt2Weights`:

```text
Gpt2Weights::forward_next_token_with_cache_and_backend
```

The current cached generation flow is:

```text
1. Create KvCache from model config.
2. Feed prompt tokens one by one through cached one-token forward.
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
cargo run --release -p miniinfer-cli -- bench-generate --model models/gpt2-miniinfer --prompt "Hey I bet you're wondering how I got into this situation" --max-new-tokens 60 --compare-cache
```

## Benchmark Result

Example local result using GPT-2 small with the ndarray backend:

```text
Prompt tokens: 12
Generated tokens: 60

No cache:
  Generation time: 21.400s
  Tokens/sec: 2.804

KV cache:
  Generation time: 12.195s
  Tokens/sec: 4.920

Speedup:
  Generation time: 1.755x
  Tokens/sec: 1.755x

Outputs match: true
```

The matching output confirms that cached greedy generation follows the same token path as uncached greedy generation for this benchmark.

See [benchmarks.md](benchmarks.md) for the full benchmark command, environment details, and a second shorter cache comparison.

## Current Limitations

The current implementation prioritizes correctness and explainability over peak performance.

- Prompt prefill is token-by-token through the cached decode path.
- Cached key/value readback currently clones flat buffers into `Tensor` values.
- KV-cache memory reporting is not yet exposed in the benchmark output.
- `bench-generate` uses greedy decoding only, which is useful for deterministic speed comparisons.

The token-by-token prefill means time to first token can be slower with KV cache for longer prompts. Decode throughput still improves because generated tokens reuse cached keys and values instead of recomputing attention over the full generated sequence.

## Next Improvements

Possible follow-up work:

```text
1. Add KV-cache memory estimates to benchmark output.
2. Split prompt prefill time from decode time in benchmark output.
3. Optimize prompt prefill by computing full prompt keys/values in one pass.
4. Avoid cloning cached key/value buffers during attention readback.
5. Add a markdown benchmark report with hardware details and reproducible commands.
```
