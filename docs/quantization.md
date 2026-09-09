# Quantization

MiniInfer supports mixed FP32 and symmetric int8 binary weight artifacts for GPT-2-style models. The current int8 path compresses selected weights on disk and dequantizes them back to FP32 tensors during model load.

This is not yet true int8 inference. Matmul and activations still run through the FP32 runtime path.

## Format

Quantized artifacts use `weights.index.json` format version 2 with per-tensor dtype metadata:

```json
{
  "format_version": 2,
  "endianness": "little",
  "lm_head": { "type": "tied" },
  "tensors": {
    "blocks.0.c_attn_weight": {
      "shape": [768, 2304],
      "offset_bytes": 0,
      "len": 1769472,
      "dtype": "i8_symmetric",
      "scales": [0.0012, 0.0028],
      "scale_axis": 1
    }
  }
}
```

Legacy FP32 artifacts remain supported through format version 1 with top-level `"dtype": "f32"`.

## Method

Per-channel symmetric quantization uses one scale per output column for rank-2 GPT-2 linear weights shaped `[in_features, out_features]`:

```text
scales[col] = max(abs(weight[:, col])) / 127
q[row, col] = round(weight[row, col] / scales[col]).clamp(-127, 127)
dequantized[row, col] = q[row, col] * scales[col]
```

MiniInfer records this as `scale_axis = 1`.

The converter currently quantizes these GPT-2 block matrix weights:

```text
blocks.*.c_attn_weight
blocks.*.attn_c_proj_weight
blocks.*.c_fc_weight
blocks.*.mlp_c_proj_weight
```

Embeddings, LayerNorm weights/biases, and linear biases remain FP32.

## Commands

Create a per-channel int8 artifact:

```powershell
.\.venv\Scripts\python.exe tools\convert_gpt2_pytorch.py --source-dir models/gpt2 --output-dir models/gpt2-miniinfer-int8-channel --format binary --quantization int8 --overwrite
```

Inspect it:

```powershell
cargo run --release -p miniinfer-cli -- inspect --model models/gpt2-miniinfer-int8-channel
```

Run it with the default cached packed-int8 runtime:

```powershell
cargo run --release -p miniinfer-cli -- run --model models/gpt2-miniinfer-int8-channel --prompt "Once upon a time" --max-new-tokens 60 --stream
```

Use `--weight-runtime f32` to dequantize block weights on load for comparison, or `--no-kv-cache --weight-runtime f32` to measure the older full-context path.

Compare logits against Hugging Face FP32:

```powershell
.\.venv\Scripts\python.exe tools\check_gpt2_logits_parity.py --model models/gpt2-miniinfer-int8-channel --prompt "Hello world" --top-k 10 --tolerance 1.0
```

## Results

Local GPT-2 small artifact size comparison:

| Artifact         | `weights.bin` size | Reduction |
| ---------------- | -----------------: | --------: |
| FP32             |  497,759,232 bytes |     1.00x |
| Per-channel int8 |  242,955,264 bytes |     2.05x |

Logit drift for prompt `Hello world`, comparing MiniInfer per-channel int8 against Hugging Face FP32 for the top 10 Hugging Face logits:

| Metric                   |      Value |
| ------------------------ | ---------: |
| Compared tokens          |         10 |
| Max absolute difference  | 0.46926117 |
| Mean absolute difference | 0.35725098 |
| Tolerance                | 1.00000000 |
| Status                   |       PASS |

Per-channel scales significantly improve over the earlier per-tensor int8 checkpoint, which measured approximately `4.8978` max absolute difference and `3.942` mean absolute difference on the same prompt.

## Limitations

- The current converter quantizes only selected GPT-2 block matrix weights.
- Packed-int8 runtime is optimized for cached one-token decode; multi-row prefill and no-cache paths still favor FP32 ndarray matmul.
- Per-channel metadata increases index size, but the large binary payload still shrinks substantially.
- Full int8 inference would require activation quantization and `i8 x i8 -> i32` kernels; the current runtime keeps activations in FP32 and uses weight-only int8 packing.
