"""Compare one-layer GPT-2 smoke logits between PyTorch and MiniInfer."""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare selected last-token logits for the one-layer GPT-2 smoke model."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("models/gpt2/raw/pytorch_model.bin"))
    parser.add_argument("--model", type=Path, default=Path("models/gpt2-miniinfer-smoke"))
    parser.add_argument("--prompt", default="Hello, world")
    parser.add_argument("--token-ids", default="15496,11,995")
    parser.add_argument("--ids", default="995,11,15496,0,50256")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        import torch.nn.functional as functional
    except ModuleNotFoundError as error:
        raise SystemExit("PyTorch is required for this reference check.") from error

    token_ids = parse_csv_usize(args.token_ids)
    selected_ids = parse_csv_usize(args.ids)
    state = load_torch_state(torch, args.checkpoint)
    reference_logits = pytorch_one_layer_logits(torch, functional, state, token_ids, selected_ids)
    miniinfer_values = load_miniinfer_logits(args.model, args.prompt, selected_ids)

    max_abs_diff = 0.0
    print("token_id\tminiinfer\tpytorch\tabs_diff")
    for token_id in selected_ids:
        miniinfer_value = miniinfer_values[token_id]
        reference_value = reference_logits[token_id]
        abs_diff = abs(miniinfer_value - reference_value)
        max_abs_diff = max(max_abs_diff, abs_diff)
        print(f"{token_id}\t{miniinfer_value:.8f}\t{reference_value:.8f}\t{abs_diff:.8f}")

    print(f"max_abs_diff\t{max_abs_diff:.8f}")
    if max_abs_diff > args.tolerance:
        raise SystemExit(
            f"logit parity failed: max_abs_diff {max_abs_diff:.8f} > tolerance {args.tolerance}"
        )


def parse_csv_usize(text: str) -> list[int]:
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def load_torch_state(torch: Any, checkpoint: Path) -> dict[str, Any]:
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise SystemExit("checkpoint did not contain a tensor dictionary")
    return state


def tensor_for(state: dict[str, Any], name: str) -> Any:
    for candidate in [name, f"transformer.{name}"]:
        tensor = state.get(candidate)
        if tensor is not None:
            return tensor
    raise SystemExit(f"missing tensor: {name}")


def pytorch_one_layer_logits(
    torch: Any,
    functional: Any,
    state: dict[str, Any],
    token_ids: list[int],
    selected_ids: list[int],
) -> dict[int, float]:
    with torch.no_grad():
        token_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        position_ids = torch.arange(len(token_ids), dtype=torch.long)
        hidden = tensor_for(state, "wte.weight")[token_ids_tensor]
        hidden = hidden + tensor_for(state, "wpe.weight")[position_ids]

        hidden = apply_block(torch, functional, state, hidden, layer_index=0)
        hidden = functional.layer_norm(
            hidden,
            (hidden.shape[-1],),
            weight=tensor_for(state, "ln_f.weight"),
            bias=tensor_for(state, "ln_f.bias"),
            eps=1e-5,
        )
        logits = hidden @ tensor_for(state, "wte.weight").transpose(0, 1)
        last_logits = logits[-1]
        return {token_id: float(last_logits[token_id].item()) for token_id in selected_ids}


def apply_block(torch: Any, functional: Any, state: dict[str, Any], hidden: Any, layer_index: int) -> Any:
    prefix = f"h.{layer_index}"
    residual = hidden
    normalized = functional.layer_norm(
        hidden,
        (hidden.shape[-1],),
        weight=tensor_for(state, f"{prefix}.ln_1.weight"),
        bias=tensor_for(state, f"{prefix}.ln_1.bias"),
        eps=1e-5,
    )

    qkv = normalized @ tensor_for(state, f"{prefix}.attn.c_attn.weight")
    qkv = qkv + tensor_for(state, f"{prefix}.attn.c_attn.bias")
    query, key, value = qkv.chunk(3, dim=-1)
    context = attention_output(torch, query, key, value, num_heads=12)
    attention_projected = context @ tensor_for(state, f"{prefix}.attn.c_proj.weight")
    attention_projected = attention_projected + tensor_for(state, f"{prefix}.attn.c_proj.bias")
    hidden = residual + attention_projected

    residual = hidden
    normalized = functional.layer_norm(
        hidden,
        (hidden.shape[-1],),
        weight=tensor_for(state, f"{prefix}.ln_2.weight"),
        bias=tensor_for(state, f"{prefix}.ln_2.bias"),
        eps=1e-5,
    )
    expanded = normalized @ tensor_for(state, f"{prefix}.mlp.c_fc.weight")
    expanded = expanded + tensor_for(state, f"{prefix}.mlp.c_fc.bias")
    activated = gelu_new(torch, expanded)
    projected = activated @ tensor_for(state, f"{prefix}.mlp.c_proj.weight")
    projected = projected + tensor_for(state, f"{prefix}.mlp.c_proj.bias")
    return residual + projected


def attention_output(torch: Any, query: Any, key: Any, value: Any, num_heads: int) -> Any:
    seq_len = query.shape[0]
    hidden_size = query.shape[1]
    head_dim = hidden_size // num_heads

    query_heads = query.reshape(seq_len, num_heads, head_dim).permute(1, 0, 2)
    key_heads = key.reshape(seq_len, num_heads, head_dim).permute(1, 0, 2)
    value_heads = value.reshape(seq_len, num_heads, head_dim).permute(1, 0, 2)

    scores = query_heads @ key_heads.transpose(-1, -2)
    scores = scores / math.sqrt(head_dim)
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores, dim=-1)
    context_heads = probabilities @ value_heads
    return context_heads.permute(1, 0, 2).contiguous().reshape(seq_len, hidden_size)


def gelu_new(torch: Any, values: Any) -> Any:
    return 0.5 * values * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (values + 0.044715 * values.pow(3))))


def load_miniinfer_logits(model: Path, prompt: str, selected_ids: list[int]) -> dict[int, float]:
    command = [
        "cargo",
        "run",
        "--quiet",
        "-p",
        "miniinfer-cli",
        "--",
        "logits",
        "--model",
        str(model),
        "--prompt",
        prompt,
        "--ids",
        ",".join(str(token_id) for token_id in selected_ids),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    logits: dict[int, float] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit():
            logits[int(parts[0])] = float(parts[1])
    missing = [token_id for token_id in selected_ids if token_id not in logits]
    if missing:
        raise SystemExit(f"MiniInfer logits output was missing token IDs: {missing}")
    return logits


if __name__ == "__main__":
    main()
