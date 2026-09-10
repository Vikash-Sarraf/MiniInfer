# HTTP Server

MiniInfer includes a local Rust HTTP server for completion-style generation. The server loads a model once, keeps it in process memory, and serves requests through an OpenAI-style completions endpoint.

The server is intentionally small. It is a local inference surface for MiniInfer's existing runtime, not a production multi-user serving layer.

## Start Server

```powershell
cargo run --release -p miniinfer-cli -- serve --model models/gpt2-miniinfer-int8-channel --port 8080
```

Default server runtime settings match the CLI generation defaults:

```text
backend: ndarray
weight runtime: packed-int8
KV cache: enabled
host: 127.0.0.1
port: 8080
```

Use `--weight-runtime f32` for the FP32/dequantized runtime and `--no-kv-cache` for diagnostic full-context recompute.

## Endpoint

```http
POST /v1/completions
```

Request body:

```json
{
  "prompt": "Once upon a time",
  "max_tokens": 20,
  "temperature": 0.8,
  "top_k": 40,
  "top_p": 0.9,
  "seed": 42
}
```

`temperature` enables stochastic sampling. `seed`, `top_k`, and `top_p` require `temperature`, matching the CLI behavior. Unknown JSON fields are rejected, so use `top_k` and `top_p`, not `top-k` or `top-p`.

## Non-Streaming Example

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8080/v1/completions `
  -ContentType "application/json" `
  -Body '{"prompt":"Once upon a time","max_tokens":20,"temperature":0.8,"top_k":40,"top_p":0.9,"seed":42}'
```

Response shape:

```json
{
  "id": "cmpl-1788951002-774304300",
  "object": "text_completion",
  "created": 1788951002,
  "model": "models/gpt2-miniinfer-int8-channel",
  "choices": [
    {
      "text": "...",
      "index": 0,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 4,
    "completion_tokens": 20,
    "total_tokens": 24
  },
  "elapsed_ms": 1234,
  "backend": "ndarray",
  "weight_runtime": "packed-int8",
  "kv_cache": true
}
```

## Streaming Example

Set `"stream": true` to receive Server-Sent Events:

```powershell
curl.exe -N -X POST http://127.0.0.1:8080/v1/completions `
  -H "Content-Type: application/json" `
  -d '{"prompt":"Hello world","max_tokens":6,"temperature":0.9,"top_k":40,"top_p":0.9,"seed":42,"stream":true}'
```

Example stream:

```text
data: {"id":"cmpl-...","object":"text_completion.chunk","created":1789025501,"model":"models/gpt2-miniinfer-int8-channel","choices":[{"text":",","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion.chunk","created":1789025501,"model":"models/gpt2-miniinfer-int8-channel","choices":[{"text":" what","index":0,"finish_reason":null}]}

data: {"id":"cmpl-...","object":"text_completion.chunk","created":1789025501,"model":"models/gpt2-miniinfer-int8-channel","choices":[{"text":"","index":0,"finish_reason":"length"}]}

data: [DONE]
```

The stream emits completion text chunks only. The original prompt is not repeated as a streamed chunk.

## Current Limitations

- Only `/v1/completions` is implemented.
- `/v1/chat/completions` is not implemented because GPT-2 is a completion model, not an instruction/chat model.
- Tool calls, JSON mode, auth, batching, and request queues are not implemented.
- Prefix cache is not implemented yet; the server is the long-lived process that will make session-local prefix cache useful later.
- Streaming uses a blocking generation task per request, which is acceptable for the current local server but not a production concurrency model.

## Validation Commands

```powershell
cargo test -p miniinfer-cli
cargo clippy --workspace -- -D warnings
git diff --check
```
