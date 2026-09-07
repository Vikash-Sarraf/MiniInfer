"""Compare MiniInfer GPT-2 final-token logits against Hugging Face Transformers."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MiniInfer final-token logits with a local Hugging Face GPT-2 checkpoint."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path("models/gpt2"),
        help="Directory containing Hugging Face GPT-2 config and tokenizer files.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to pytorch_model.bin. Defaults to <source-dir>/raw/pytorch_model.bin.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/gpt2-miniinfer"),
        help="Converted MiniInfer model directory.",
    )
    parser.add_argument(
        "--prompt",
        default="Hello world",
        help="Prompt to compare.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of Hugging Face top logits to compare.",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="Optional comma-separated token IDs to compare in addition to HF top-k IDs.",
    )
    parser.add_argument(
        "--backend",
        choices=["ndarray", "reference"],
        default="ndarray",
        help="MiniInfer backend to use for the logits command.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-3,
        help="Maximum allowed absolute logit difference.",
    )
    parser.add_argument(
        "--debug-build",
        action="store_true",
        help="Use cargo's debug profile instead of --release for MiniInfer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = args.weights or args.source_dir / "raw" / "pytorch_model.bin"

    try:
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
    except ModuleNotFoundError as error:
        raise SystemExit(
            "PyTorch and Transformers are required. Install them in the active Python environment first."
        ) from error

    tokenizer = GPT2TokenizerFast.from_pretrained(args.source_dir, local_files_only=True)
    model = load_hf_model(torch, GPT2Config, GPT2LMHeadModel, args.source_dir, weights_path)
    token_ids, hf_logits = hf_final_token_logits(torch, tokenizer, model, args.prompt)
    hf_top_ids = top_token_ids(torch, hf_logits, args.top_k)
    selected_ids = stable_unique(hf_top_ids + parse_optional_ids(args.ids))
    miniinfer_logits = load_miniinfer_logits(args, selected_ids)

    max_abs_diff = 0.0
    total_abs_diff = 0.0

    print(f"prompt\t{args.prompt!r}")
    print(f"hf_token_ids\t{token_ids}")
    print("token_id\ttoken\tminiinfer\thuggingface\tabs_diff")

    for token_id in selected_ids:
        miniinfer_value = miniinfer_logits[token_id]
        hf_value = float(hf_logits[token_id].item())
        abs_diff = abs(miniinfer_value - hf_value)
        max_abs_diff = max(max_abs_diff, abs_diff)
        total_abs_diff += abs_diff
        token = tokenizer.decode([token_id])
        print(f"{token_id}\t{token!r}\t{miniinfer_value:.8f}\t{hf_value:.8f}\t{abs_diff:.8f}")

    mean_abs_diff = total_abs_diff / len(selected_ids) if selected_ids else 0.0
    print(f"compared_tokens\t{len(selected_ids)}")
    print(f"max_abs_diff\t{max_abs_diff:.8f}")
    print(f"mean_abs_diff\t{mean_abs_diff:.8f}")
    print(f"tolerance\t{args.tolerance:.8f}")

    if max_abs_diff > args.tolerance:
        raise SystemExit(
            f"logit parity failed: max_abs_diff {max_abs_diff:.8f} > tolerance {args.tolerance:.8f}"
        )

    print("status\tPASS")


def load_hf_model(
    torch: Any,
    config_cls: Any,
    model_cls: Any,
    source_dir: Path,
    weights_path: Path,
) -> Any:
    if not weights_path.exists():
        raise SystemExit(f"missing PyTorch weights: {weights_path}")

    config = config_cls.from_pretrained(source_dir, local_files_only=True)
    model = model_cls(config)
    state = load_torch_state(torch, weights_path)

    if any(name.startswith("transformer.") for name in state):
        missing, unexpected = model.load_state_dict(state, strict=False)
    else:
        missing, unexpected = model.transformer.load_state_dict(state, strict=False)
        model.tie_weights()

    unexpected = [name for name in unexpected if not name.endswith("attn.bias")]
    if missing or unexpected:
        raise SystemExit(
            f"failed to load checkpoint cleanly: missing={missing}, unexpected={unexpected[:10]}"
        )

    model.eval()
    return model


def load_torch_state(torch: Any, weights_path: Path) -> dict[str, Any]:
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(weights_path, map_location="cpu")

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if not isinstance(state, dict):
        raise SystemExit("PyTorch checkpoint did not contain a tensor dictionary")
    return state


def hf_final_token_logits(torch: Any, tokenizer: Any, model: Any, prompt: str) -> tuple[list[int], Any]:
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    return inputs["input_ids"].tolist()[0], logits


def top_token_ids(torch: Any, logits: Any, top_k: int) -> list[int]:
    if top_k <= 0:
        raise SystemExit("--top-k must be greater than zero")
    _, token_ids = torch.topk(logits, top_k)
    return [int(token_id) for token_id in token_ids.tolist()]


def parse_optional_ids(ids: str | None) -> list[int]:
    if ids is None:
        return []
    return [int(part.strip()) for part in ids.split(",") if part.strip()]


def stable_unique(values: list[int]) -> list[int]:
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def load_miniinfer_logits(args: argparse.Namespace, selected_ids: list[int]) -> dict[int, float]:
    command = ["cargo", "run", "--quiet"]
    if not args.debug_build:
        command.append("--release")
    command.extend(
        [
            "-p",
            "miniinfer-cli",
            "--",
            "logits",
            "--model",
            str(args.model),
            "--prompt",
            args.prompt,
            "--ids",
            ",".join(str(token_id) for token_id in selected_ids),
            "--backend",
            args.backend,
        ]
    )

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