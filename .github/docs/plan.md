# MiniInfer Project Plan

## 1. Project Summary

**MiniInfer** is a CPU-first LLM inference engine designed to be a credible resume project: small enough to finish, but real enough to demonstrate systems programming, ML infrastructure, numerical correctness, and measurable performance.

The project’s core goal is to build a real inference engine — not a fake runtime, hardcoded demo, or wrapper around an existing engine. It should load real model weights, tokenize real prompts, execute transformer decoding, maintain a KV cache, sample tokens, and stream generated output.

The V1 resume target is a complete, verified GPT-2-style inference path:

- convert real Hugging Face GPT-2-style checkpoints
- load a custom versioned model format
- tokenize real prompts
- run transformer prefill and autoregressive decode
- use a per-layer KV cache
- support greedy, temperature, top-k, and top-p sampling
- stream generated text from a CLI
- compare correctness against Python/PyTorch references
- publish reproducible benchmark results

The project will preserve future expansion paths for:

- modern Llama/Qwen-style model architectures
- quantized inference
- local HTTP serving
- OpenAI-compatible APIs
- agent tool-calling
- structured decoding
- prompt/prefix caching
- browser or web UI demos
- optional WebAssembly/WebGPU experiments

The first production-quality target is a **GPT-2-style decoder-only transformer** because it is the shortest honest path to a finished engine. Modern Llama/Qwen-style models remain future work, so model architecture differences should be isolated behind config and architecture modules.

---

## 2. Goals

### 2.1 Primary Goals

The primary goal is a finished V1 that can be shown on a resume, explained in interviews, and verified with tests and benchmarks.

1. Build a real CPU inference engine for GPT-2-style decoder-only transformer models.
2. Support conversion from real Hugging Face GPT-2-style checkpoints into a MiniInfer model format.
3. Implement a real tokenizer pipeline.
4. Implement the transformer forward pass:
   - token embeddings
   - positional embeddings
   - LayerNorm
   - causal self-attention
   - MLP
   - residual connections
   - final logits
5. Implement autoregressive generation.
6. Implement KV cache for efficient decoding.
7. Implement token sampling:
   - greedy
   - temperature
   - top-k
   - top-p
8. Stream generated text from a CLI.
9. Include correctness tests against Python/PyTorch or NumPy reference implementations.
10. Include benchmarks:
    - model load time
    - prompt prefill speed
    - decode speed
    - time to first token
    - memory usage
    - KV cache memory
    - no-cache vs KV-cache generation speed

### 2.2 Secondary Goals

Secondary goals are explicitly post-V1. They should not delay the core engine, correctness tests, tokenizer, KV cache, CLI generation, or benchmark report.

1. Add a local HTTP server.
2. Add an OpenAI-compatible API surface.
3. Add agent-oriented infrastructure:
   - tool-call schema validation
   - tool execution loop
   - structured JSON outputs
   - prompt/prefix cache abstraction
   - request tracing
4. Add a simple web UI/dashboard.
5. Add optional remote backend support later.
6. Add optional llama.cpp/Ollama adapter later for comparison, not as the core engine.
7. Add modern model architecture support in future phases:
   - Llama
   - Qwen
   - Mistral-like models

---

## 3. Non-Goals

The first version is **not** trying to be a full llama.cpp replacement.

V1 will not include:

- CUDA
- Vulkan
- Metal
- WebGPU kernels
- HTTP server
- OpenAI-compatible API
- agent runtime
- web dashboard
- GGUF loading
- MoE inference
- multi-GPU inference
- vLLM-style continuous batching
- paged attention
- speculative decoding
- advanced 4-bit quantization
- production-grade multi-user serving
- broad model-family compatibility

These may become future work, but they should not block the first complete engine. For resume value, a smaller finished inference engine with tests and benchmarks is more valuable than a broad unfinished platform.

---

## 4. Design Philosophy

The project should follow this principle:

> Small implementation, serious architecture.

This means:

- support one architecture first, but design for multiple architectures later
- support CPU first, but avoid making GPU/WebGPU impossible later
- support FP32 first, but design dtype/quantization abstractions now
- support a simple model format first, but version the format for future compatibility
- build a real engine, not a fake demo
- use test fixtures for development, but final runtime must use real model weights
- prioritize clear correctness evidence and benchmark artifacts over broad feature count

### 4.1 Learning-First AI Assistance

This project should rely on agentic coding as little as possible for the low-level inference internals. The project owner should handwrite and understand the core tensor, operation, tokenizer, model, KV-cache, and sampling code.

Copilot may be used for explanation, design review, scaffolding, test planning, debugging, documentation, and small non-core wiring. Core implementations should follow the workspace guardrails in [../instructions/miniinfer-learning-first.instructions.md](../instructions/miniinfer-learning-first.instructions.md).

---

## 5. High-Level Architecture

```text
User / CLI
        ↓
Generation API
        ↓
Runtime Engine
        ↓
Model Architecture Implementation
        ↓
Tensor Operations
        ↓
Model Weights + Tokenizer + KV Cache
```

Future extended architecture:

```text
CLI / HTTP API / Web UI / Agent Runtime
        ↓
Inference Interface
        ↓
Backend Layer
   ┌───────────────┬────────────────┬────────────────┐
   │ MiniInfer CPU │ Remote API      │ llama.cpp API   │
   │ Backend       │ Backend         │ Adapter         │
   └───────────────┴────────────────┴────────────────┘
        ↓
Tracing / Metrics / Prompt Cache / Tool Runtime
```

The actual inference engine remains the main project. The agent and web layers are built on top of it.

---

## 6. Repository Structure

Recommended V1 structure:

```text
miniinfer/
  crates/
    miniinfer-core/
      src/
        lib.rs
        dtype.rs
        tensor.rs
        error.rs

        ops/
          mod.rs
          matmul.rs
          softmax.rs
          layer_norm.rs
          gelu.rs
          sampling.rs

        tokenizer/
          mod.rs
          bpe.rs
          tokenizer_json.rs

        model/
          mod.rs
          config.rs
          format.rs
          loader.rs
          architecture.rs
          gpt2.rs
          llama.rs        # future stub only

        runtime/
          mod.rs
          engine.rs
          generation.rs
          kv_cache.rs
          sampler.rs
          metrics.rs

    miniinfer-cli/
      src/
        main.rs

  tools/
    convert_hf_gpt2.py
    compare_logits.py
    inspect_model.py

  tests/
    python_reference/
      reference_ops.py
      reference_gpt2.py

  benchmarks/
    prompts/
      simple_prompts.txt
    run_bench.py
    results/

  docs/
    architecture.md
    model-format.md
    kv-cache.md
    tokenizer.md
    benchmarks.md
    roadmap.md

  README.md
  PLAN.md
```

Post-V1 optional structure:

```text
miniinfer/
  crates/
    miniinfer-server/
      src/
        main.rs
        routes.rs
        streaming.rs
        openai_types.rs

    miniinfer-agent/
      src/
        mod.rs
        tools.rs
        schema.rs
        trace.rs
        prompt_cache.rs

  web/
    package.json
    src/
      App.tsx
      ChatPanel.tsx
      TraceView.tsx
      MetricsPanel.tsx
```

---

## 7. Language and Technology Choices

### 7.1 Core Engine

Recommended:

```text
Rust
```

Reasons:

- strong CV signal
- memory safety
- good performance
- good CLI/server ecosystem
- good WebAssembly future path
- suitable for systems-level ML infrastructure

Alternative:

```text
C++
```

C++ is also valid and closer to llama.cpp style, but Rust may be easier to make clean and safe for a portfolio project.

### 7.2 Converter and Reference Tests

Use Python for:

- Hugging Face model conversion
- PyTorch correctness comparison
- NumPy reference outputs
- benchmark report scripts

### 7.3 Server

Post-V1 only. Do not start server work until CLI generation, correctness checks, and benchmarks are working.

Use Rust:

- `axum` or `actix-web`
- Server-Sent Events for streaming
- `serde` for JSON
- optional SQLite for request traces

### 7.4 Web UI

Post-V1 only. The web UI is useful for demos, but it is not part of the resume-critical engine proof.

Use TypeScript:

- Vite
- React or Svelte
- simple dashboard for chat, metrics, and traces

---

## 8. Model Architecture Strategy

### 8.1 V1 Target: GPT-2-Style Models

V1 supports GPT-2-style decoder-only transformer models.

This includes:

- learned positional embeddings
- LayerNorm
- standard multi-head causal self-attention
- GELU MLP
- residual connections
- tied or untied LM head depending model config

GPT-2 is chosen because it is simpler than modern Llama/Qwen architectures and is excellent for validating the inference pipeline.

### 8.2 Concern: GPT-2 Is Old

GPT-2 is not the modern production architecture used in most current open-weight chat models. However, it is still a good v1 target because it teaches and validates the core inference pipeline:

- model loading
- tokenizer handling
- tensor ops
- attention
- KV caching
- sampling
- streaming generation
- benchmarking

The project must avoid baking GPT-2 assumptions into every layer.

### 8.3 Future Modern Architecture Support

Future Llama/Qwen support requires adding:

- RMSNorm instead of LayerNorm
- RoPE instead of learned positional embeddings
- SwiGLU instead of GELU MLP
- grouped-query attention or multi-query attention
- different tokenizer handling
- chat templates
- different tensor naming conventions
- possibly tied embedding/output weights
- longer-context scaling behavior

Therefore, the model code should be structured around architecture-specific implementations.

Example architecture abstraction:

```rust
pub trait ModelArchitecture {
    fn forward_prefill(&mut self, tokens: &[u32]) -> Result<Logits>;
    fn forward_decode(&mut self, token: u32) -> Result<Logits>;
    fn reset_cache(&mut self);
    fn config(&self) -> &ModelConfig;
}
```

Model types:

```rust
pub enum Architecture {
    Gpt2,
    Llama,
    Qwen,
}
```

Even if only `Gpt2` works in V1, the structure should support additional variants.

---

## 9. Model Format Requirements

### 9.1 V1 Format

Use a simple custom model directory format:

```text
models/gpt2-small/
  config.json
  tokenizer.json
  weights.mini
```

Alternative GPT-2 tokenizer files:

```text
vocab.json
merges.txt
```

The engine should load:

```bash
miniinfer run --model ./models/gpt2-small --prompt "The future of AI is"
```

### 9.2 Config

Example `config.json`:

```json
{
  "format_version": 1,
  "architecture": "gpt2",
  "dtype": "f32",
  "vocab_size": 50257,
  "max_position_embeddings": 1024,
  "hidden_size": 768,
  "num_layers": 12,
  "num_heads": 12,
  "intermediate_size": 3072,
  "layer_norm_epsilon": 1e-5,
  "activation": "gelu",
  "position_encoding": "learned",
  "tie_word_embeddings": true
}
```

### 9.3 Format Versioning

The model format must include:

- magic header
- format version
- architecture
- dtype
- tensor metadata
- tensor names
- tensor shapes
- byte offsets

This allows future support for:

- int8 weights
- block quantization
- memory mapping
- Llama-style models
- tokenizer metadata
- GGUF import

---

## 10. Tokenizer Requirements

### 10.1 V1

Support real GPT-2 BPE tokenization.

Inputs:

- `vocab.json`
- `merges.txt`

or:

- `tokenizer.json`

Required operations:

```text
encode(text) -> Vec<TokenId>
decode(tokens) -> String
```

### 10.2 Tokenizer Tests

Tokenizer should be verified against a Python reference or Hugging Face tokenizer output.

Example tests:

```text
"Hello world" -> expected token IDs
token IDs -> "Hello world"
roundtrip encode/decode for simple text
special token handling
```

### 10.3 Future

Add support for SentencePiece or tokenizer.json variants used by:

- Llama
- Qwen
- Mistral
- Gemma

---

## 11. Tensor and Operation Requirements

### 11.1 Tensor Type

Implement a minimal tensor type:

- dtype
- shape
- contiguous storage
- indexing helpers
- reshape/view where needed
- error handling for shape mismatch

V1 can assume contiguous tensors.

### 11.2 Required Ops

Implement:

- matrix multiplication
- vector addition
- elementwise operations
- softmax
- GELU
- LayerNorm
- embedding lookup
- transpose/reshape helpers
- causal masking
- attention score computation

### 11.3 Future Ops

Future architectures require:

- RMSNorm
- RoPE
- SiLU
- SwiGLU
- grouped-query attention
- quantized matmul
- dequantization kernels

---

## 12. GPT-2 Forward Pass Requirements

The GPT-2 model implementation must perform:

```text
input token IDs
  ↓
token embeddings + positional embeddings
  ↓
N transformer blocks
  ↓
final LayerNorm
  ↓
LM head
  ↓
logits
```

Each transformer block:

```text
x
  ↓
LayerNorm
  ↓
QKV projection
  ↓
causal self-attention
  ↓
attention output projection
  ↓
residual add
  ↓
LayerNorm
  ↓
MLP: Linear → GELU → Linear
  ↓
residual add
```

---

## 13. KV Cache Requirements

KV cache is mandatory.

### 13.1 Purpose

Without KV cache, each generated token recomputes attention over the full context. With KV cache, previous keys and values are reused.

### 13.2 V1 KV Cache Features

- per-layer cache
- stores keys and values
- supports prefill over prompt tokens
- supports decode-one-token loop
- tracks current sequence length
- supports reset
- reports memory usage
- enforces max context length

### 13.3 Benchmark Requirement

Benchmark generation with and without KV cache.

Example output:

```text
Model: gpt2-small
Prompt tokens: 32
Generated tokens: 64

No KV cache:
  decode tok/s: X

With KV cache:
  decode tok/s: Y

Speedup:
  Y / X
```

---

## 14. Sampling Requirements

V1 supports:

- greedy decoding
- temperature
- top-k
- top-p

Future:

- repetition penalty
- frequency penalty
- presence penalty
- min-p
- grammar-constrained decoding
- JSON-constrained decoding

Sampling should be isolated in a `sampler` module.

---

## 15. CLI Requirements

### 15.1 Run Command

```bash
miniinfer run \
  --model ./models/gpt2-small \
  --prompt "The future of AI is" \
  --max-tokens 80 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95
```

### 15.2 Benchmark Command

```bash
miniinfer bench \
  --model ./models/gpt2-small \
  --prompt-file ./benchmarks/prompts/simple_prompts.txt \
  --max-tokens 64
```

### 15.3 Inspect Command

```bash
miniinfer inspect --model ./models/gpt2-small
```

Should print:

```text
architecture
parameter count
dtype
vocab size
layers
heads
hidden size
max context
estimated weight memory
```

---

## 16. Correctness Requirements

Correctness is critical.

### 16.1 Operation-Level Tests

Compare Rust/C++ output against Python/NumPy/PyTorch for:

- matmul
- softmax
- LayerNorm
- GELU
- attention
- MLP

### 16.2 Block-Level Tests

Compare one transformer block output against PyTorch.

### 16.3 Full-Model Logit Test

For a real converted model, compare MiniInfer logits to PyTorch logits for the same prompt.

Acceptable tolerance depends on dtype:

```text
FP32: strict tolerance, e.g. 1e-4 or 1e-5 if possible
FP16/future quantized: relaxed tolerance
```

### 16.4 Generation Test

For deterministic greedy decoding, compare first N generated token IDs against a reference implementation if numerical parity is stable enough.

---

## 17. Benchmarking Requirements

Benchmark reports should include:

- hardware summary
- model name
- parameter count
- dtype
- prompt length
- generated tokens
- load time
- time to first token
- prefill tokens/sec
- decode tokens/sec
- peak memory usage
- KV cache memory usage

Example table:

```text
| Model | Params | DType | Prompt toks | Decode toks | TTFT ms | Decode tok/s | Peak RAM |
|---|---:|---|---:|---:|---:|---:|---:|
| GPT-2 small | 124M | FP32 | 32 | 64 | 180 | 8.5 | 620 MB |
```

Important comparisons:

1. no KV cache vs KV cache
2. single-thread vs future multithread
3. FP32 vs future int8
4. MiniInfer vs PyTorch CPU reference
5. optional: MiniInfer vs llama.cpp where applicable

---

## 18. Post-V1 Agent Infrastructure Requirements

Agent infrastructure should not replace the inference engine as the main project. It should sit above the engine and should begin only after V1 can run real converted weights and publish correctness and benchmark results.

### 18.1 Agent Concepts

Add after core inference is working:

- tool schema definitions
- JSON tool-call validation
- simple tool execution
- trace recording
- prompt/prefix cache abstraction

### 18.2 Tools

Start with simple safe tools:

- calculator
- note search over local test files
- todo list writer
- echo/debug tool

### 18.3 Tool Call Format

Example:

```json
{
  "tool": "calculator",
  "arguments": {
    "expression": "12 * 48"
  }
}
```

### 18.4 Structured Output

V1 may use validation and retry.

Future versions may support constrained decoding.

### 18.5 Prompt Cache

Implement a cache abstraction:

```text
stable prefix text
  ↓
hash
  ↓
cache entry
```

For MiniInfer, this can eventually map to real KV-cache prefix reuse.

For remote backends, it may only track estimated savings or deduplicate repeated requests.

---

## 19. Post-V1 HTTP Server Requirements

Add after CLI inference, correctness tests, KV-cache decode, and benchmark reporting work.

### 19.1 Endpoints

```text
GET  /health
POST /generate
POST /v1/completions
POST /v1/chat/completions
GET  /metrics
GET  /traces
```

### 19.2 Streaming

Support Server-Sent Events:

```text
data: {"token":"Hello"}

data: {"token":" world"}

data: [DONE]
```

### 19.3 OpenAI Compatibility

OpenAI-compatible support does not need to be perfect in V1, but should mimic:

```text
POST /v1/chat/completions
```

with:

- `messages`
- `model`
- `temperature`
- `top_p`
- `max_tokens`
- `stream`

---

## 20. Post-V1 Web UI Requirements

Add after server works.

### 20.1 Web Features

- chat input
- streamed output
- model/runtime status panel
- token/sec display
- latency display
- trace viewer
- tool-call inspector
- KV cache stats
- benchmark results view

### 20.2 Purpose

The web UI is for demonstration and developer experience. It should not be required for engine correctness.

---

## 21. Future Modern Architecture Support

After GPT-2 v1 works, add a modern architecture target.

Recommended v2/v3 architecture target:

```text
Llama-style model
```

Required additions:

- RMSNorm
- RoPE
- SwiGLU MLP
- Llama tokenizer support
- causal attention with RoPE
- optional grouped-query attention
- chat template support

Potential future model targets:

- TinyLlama
- Llama 3.2 1B
- Qwen 0.5B
- SmolLM variants

---

## 22. Quantization Roadmap

Do not implement advanced quantization first.

### 22.1 V1

- FP32

### 22.2 V2

- FP16 loading/conversion if feasible
- int8 weight-only quantization

### 22.3 V3

- block int8
- simple int4
- quantized matmul
- quantization benchmarks

### 22.4 V4

- GGUF import or compatibility layer
- Q4-style formats

Quantization code should be pluggable through dtype dispatch.

Example:

```rust
pub enum DType {
    F32,
    F16,
    I8,
    Q4Block,
}
```

---

## 23. Development Without Local Model Downloads

If the development machine cannot store public model weights, development can still proceed.

Allowed development activities:

- write engine code
- write tensor ops
- write model loader
- write converter code
- write tokenizer code
- write tests using tiny generated fixtures
- compare operation outputs against local NumPy/PyTorch on synthetic tensors
- build CLI
- write docs
- write benchmark harness

Real model integration can happen later on:

- personal machine
- cloud VM
- approved internal machine
- approved model endpoint
- CI environment if permitted

Synthetic tensors are acceptable for tests, but the final runtime must support real model weights. A tiny synthetic GPT-2 fixture should be used early so the loader, forward pass, and generation path can be tested without committing model weights.

---

## 24. Resume-Focused Milestones

The milestones are ordered to produce a finished resume artifact before expanding into server, agent, or web work.

### Milestone 0: Project Skeleton

Deliverables:

- Rust workspace
- core crate
- CLI crate
- basic README
- CI/test command
- formatting/linting

Completion criteria:

```bash
cargo test
cargo run -- --help
```

---

### Milestone 1: Tensor Core Foundation

Deliverables:

- Tensor type
- shape handling
- dtype enum with FP32 support
- contiguous row-major storage
- indexing helpers
- shape mismatch errors

Completion criteria:

```text
tensor shape, indexing, and error tests pass
```

---

### Milestone 2: Reference-Tested Ops

Deliverables:

- matmul
- softmax
- LayerNorm
- GELU
- embedding lookup
- tests against Python/NumPy reference

Completion criteria:

```text
all operation-level tests pass
```

---

### Milestone 3: GPT-2 Config and Tiny Fixture

Deliverables:

- config parser
- model structs
- architecture trait boundary
- tiny synthetic GPT-2 fixture with deterministic weights
- Python reference fixture output

Completion criteria:

```text
MiniInfer can load a tiny synthetic config and fixture metadata
```

---

### Milestone 4: GPT-2 Block Forward Pass

Deliverables:

- transformer block implementation
- forward pass over small test tensors

Completion criteria:

```text
one-block test output matches Python reference
```

---

### Milestone 5: Model Format and Loader

Deliverables:

- versioned MiniInfer model format
- tensor metadata
- weight loader
- model inspection command

Completion criteria:

```bash
miniinfer inspect --model ./models/example
```

prints model metadata.

---

### Milestone 6: Tokenizer

Deliverables:

- GPT-2 BPE tokenizer
- encode
- decode
- tokenizer tests

Completion criteria:

```text
MiniInfer tokenizer matches reference token IDs for selected examples.
```

---

### Milestone 7: Hugging Face Converter

Deliverables:

- `convert_hf_gpt2.py`
- maps GPT-2 Hugging Face tensor names to MiniInfer tensor names
- writes config/tokenizer/weights

Completion criteria:

```bash
python tools/convert_hf_gpt2.py --hf-model gpt2 --out ./models/gpt2-small
```

works in an approved environment.

---

### Milestone 8: Full Prefill Inference

Deliverables:

- full prompt forward pass
- logits output
- compare logits against PyTorch

Completion criteria:

```text
MiniInfer logits are within tolerance of PyTorch reference.
```

---

### Milestone 9: Autoregressive Generation

Deliverables:

- greedy decoding
- temperature
- top-k
- top-p
- streaming CLI

Completion criteria:

```bash
miniinfer run --model ./models/gpt2-small --prompt "The future of AI is"
```

generates real text.

---

### Milestone 10: KV Cache

Deliverables:

- per-layer KV cache
- prefill mode
- decode mode
- reset and stats
- no-cache vs cache benchmark

Completion criteria:

```text
KV-cache generation produces valid output and improves decode throughput.
```

---

### Milestone 11: Benchmark Suite

Deliverables:

- benchmark command
- prompt benchmark file
- memory reporting
- markdown benchmark report

Completion criteria:

```bash
miniinfer bench --model ./models/gpt2-small --prompt-file ./benchmarks/prompts/simple_prompts.txt
```

produces a report.

---

### Milestone 12: Resume-Ready Documentation

Deliverables:

- public README
- architecture overview
- correctness methodology
- benchmark results
- limitations and roadmap
- resume bullet examples

Completion criteria:

```text
README accurately explains what works, how to reproduce it, and what is future work.
```

---

### Milestone 13: Server (Post-V1)

Deliverables:

- HTTP server
- `/generate`
- `/v1/completions`
- streaming responses
- basic metrics

Completion criteria:

```bash
miniinfer serve --model ./models/gpt2-small --port 8080
```

and a curl request streams tokens.

---

### Milestone 14: Agent Runtime (Post-V1)

Deliverables:

- tool schema definition
- JSON validation
- calculator tool
- trace recording
- prompt cache abstraction

Completion criteria:

```text
A tool-call trace can be produced, validated, executed, and viewed.
```

---

### Milestone 15: Web UI (Post-V1)

Deliverables:

- chat UI
- streaming output
- metrics panel
- trace viewer
- tool-call inspector

Completion criteria:

```text
Browser UI can send prompt to local MiniInfer server and display streamed output plus metrics.
```

---

## 25. Public README Requirements

The README should include:

1. What MiniInfer is
2. What it is not
3. Architecture diagram
4. Supported model architecture
5. How to convert a model
6. How to run generation
7. How to run benchmarks
8. Correctness methodology
9. Benchmark results
10. Limitations
11. Roadmap
12. Resume-positioned project summary

The README should be honest:

```text
MiniInfer is not intended to beat llama.cpp. It is a learning-focused but real inference engine that demonstrates model loading, tokenization, transformer decoding, KV caching, sampling, correctness testing, and benchmarking.
```

---

## 26. Resume Positioning

Target CV bullet after V1:

> Built MiniInfer, a CPU-first LLM inference engine in Rust supporting real GPT-2 checkpoint conversion, BPE tokenization, transformer decoding, KV caching, top-k/top-p sampling, streaming generation, and PyTorch-verified numerical correctness.

Stronger CV bullet after agent/server additions:

> Extended MiniInfer with an HTTP inference server, OpenAI-compatible streaming API, request metrics, schema-validated tool calls, and prompt-cache abstractions for agent workloads.

Benchmark-oriented bullet:

> Improved autoregressive decode throughput by Xx using KV caching and published reproducible benchmarks for load time, TTFT, decode tokens/sec, and memory usage.

Interview explanation:

```text
MiniInfer is a deliberately small but real inference engine. I focused V1 on one complete, verifiable path: converting GPT-2 weights, loading them into a custom format, running FP32 CPU transformer inference, streaming generated text, validating outputs against PyTorch, and measuring the effect of KV caching.
```

---

## 27. Key Risks

### 27.1 Risk: GPT-2 Is Outdated

Mitigation:

- isolate GPT-2 behind architecture abstraction
- document future Llama/Qwen support
- avoid GPT-2-specific assumptions in runtime, sampler, CLI, server, and metrics

### 27.2 Risk: Tokenizer Complexity

Mitigation:

- start with GPT-2 BPE
- test against Hugging Face tokenizer
- later add tokenizer.json support

### 27.3 Risk: Numerical Mismatch

Mitigation:

- implement operation-level tests
- compare layer-by-layer against PyTorch
- use FP32 first
- avoid quantization until correctness is verified

### 27.4 Risk: Scope Creep

Mitigation:

- complete CPU GPT-2 inference first
- postpone GGUF, CUDA, WebGPU, MoE, batching, server, agent runtime, and web UI
- keep agent/web layers secondary until engine correctness and benchmarks are published
- measure progress by resume-visible artifacts, not feature count

### 27.5 Risk: Local Model Restrictions

Mitigation:

- develop with synthetic test tensors only
- run real model conversion/integration on approved environments
- do not commit model weights
- keep model files user-supplied

---

## 28. Definition of Done for V1

V1 is the resume-ready engine. It is done when:

1. A real GPT-2-style Hugging Face checkpoint can be converted.
2. MiniInfer can load the converted model.
3. MiniInfer can tokenize a prompt using the real tokenizer.
4. MiniInfer can run prefill.
5. MiniInfer can generate tokens autoregressively.
6. MiniInfer uses KV cache for decode.
7. CLI streams generated text.
8. Sampling supports greedy, temperature, top-k, and top-p.
9. Correctness tests compare against Python/PyTorch.
10. Benchmarks report load time, TTFT, decode speed, memory usage, and KV-cache speedup.
11. README documents usage, limitations, correctness methodology, benchmark results, and roadmap.
12. The project can be summarized honestly in one strong resume bullet.

---

## 29. Definition of Done for V2

V2 is done when:

1. MiniInfer includes an HTTP server.
2. Server supports streaming generation.
3. Server exposes a basic OpenAI-compatible endpoint.
4. Runtime records request metrics.
5. Agent runtime supports schema-validated tool calls.
6. Prompt cache abstraction exists.
7. A basic web dashboard shows chat, traces, and metrics.

---

## 30. Definition of Done for V3

V3 is done when one or more of the following is implemented:

1. Llama-style architecture support.
2. RMSNorm, RoPE, and SwiGLU.
3. int8 weight-only quantization.
4. memory-mapped model loading.
5. WebAssembly browser demo.
6. GGUF import prototype.
7. structured JSON constrained decoding.
8. improved CPU matmul/multithreading.

---

## 31. Guiding Principle

The engine should be honest, real, and measurable.

It does not need to be commercially competitive.

It does need to:

- load real weights
- run real inference
- expose clean architecture
- demonstrate meaningful optimizations
- include tests
- include benchmarks
- leave a clear path to modern architectures and agent infrastructure
