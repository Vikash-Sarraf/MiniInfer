"""Convert a Hugging Face/OpenAI GPT-2 PyTorch checkpoint to MiniInfer JSON."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Convert GPT-2 PyTorch weights into MiniInfer model artifacts."
	)
	parser.add_argument(
		"--source-dir",
		type=Path,
		default=Path("models/gpt2"),
		help="Directory containing Hugging Face GPT-2 config/tokenizer files.",
	)
	parser.add_argument(
		"--weights",
		type=Path,
		default=None,
		help="Path to pytorch_model.bin. Defaults to <source-dir>/raw/pytorch_model.bin.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("models/gpt2-miniinfer"),
		help="Directory where MiniInfer artifacts will be written.",
	)
	parser.add_argument(
		"--max-layers",
		type=int,
		default=None,
		help="Convert only the first N layers for a smaller smoke artifact.",
	)
	parser.add_argument(
		"--overwrite",
		action="store_true",
		help="Allow overwriting an existing output directory.",
	)
	parser.add_argument(
		"--dry-run",
		action="store_true",
		help="Validate config and tensor mapping without writing output files.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	source_dir = args.source_dir
	weights_path = args.weights or source_dir / "raw" / "pytorch_model.bin"
	output_dir = args.output_dir

	if source_dir.resolve() == output_dir.resolve():
		raise SystemExit("output directory must be different from source directory")

	if output_dir.exists() and not args.overwrite:
		raise SystemExit(
			f"output directory already exists: {output_dir}\n"
			"pass --overwrite to replace its contents"
		)

	try:
		import torch
	except ModuleNotFoundError as error:
		raise SystemExit(
			"PyTorch is required to read pytorch_model.bin. "
			"Install it in your Python environment first."
		) from error

	hf_config = load_json(source_dir / "config.json")
	state = load_torch_state(torch, weights_path)
	model_config = convert_config(hf_config, state, args.max_layers)

	if args.dry_run:
		validate_weight_shapes(state, model_config)
		print("Dry run OK")
		print(f"Layers: {model_config['num_layers']}")
		print(f"Vocab size: {model_config['vocab_size']}")
		print(f"Hidden size: {model_config['hidden_size']}")
		return

	weights = convert_weights(state, model_config)

	if output_dir.exists():
		shutil.rmtree(output_dir)
	output_dir.mkdir(parents=True)

	write_json(output_dir / "config.json", model_config)
	write_json(output_dir / "weights.json", weights)
	copy_tokenizer_files(source_dir, output_dir)

	print(f"Wrote MiniInfer model to {output_dir}")
	print(f"Layers: {model_config['num_layers']}")
	print(f"Vocab size: {model_config['vocab_size']}")
	print(f"Hidden size: {model_config['hidden_size']}")


def load_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as file:
		return json.load(file)


def write_json(path: Path, value: dict[str, Any]) -> None:
	with path.open("w", encoding="utf-8") as file:
		json.dump(value, file, separators=(",", ":"))
		file.write("\n")


def load_torch_state(torch: Any, weights_path: Path) -> dict[str, Any]:
	if not weights_path.exists():
		raise SystemExit(f"missing PyTorch weights: {weights_path}")

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


def convert_config(
	hf_config: dict[str, Any], state: dict[str, Any], max_layers: int | None
) -> dict[str, Any]:
	hidden_size = int(hf_config["n_embd"])
	num_layers = int(hf_config["n_layer"])
	intermediate_size = int(tensor_for(state, "h.0.mlp.c_fc.bias").shape[0])

	if max_layers is not None:
		if max_layers < 1 or max_layers > num_layers:
			raise SystemExit(f"--max-layers must be between 1 and {num_layers}")
		num_layers = max_layers

	return {
		"format_version": 1,
		"architecture": "gpt2",
		"dtype": "f32",
		"vocab_size": int(hf_config["vocab_size"]),
		"max_position_embeddings": int(hf_config["n_positions"]),
		"hidden_size": hidden_size,
		"num_layers": num_layers,
		"num_heads": int(hf_config["n_head"]),
		"intermediate_size": intermediate_size,
		"layer_norm_epsilon": float(hf_config["layer_norm_epsilon"]),
	}


def convert_weights(state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
	hidden_size = config["hidden_size"]
	vocab_size = config["vocab_size"]
	max_positions = config["max_position_embeddings"]
	intermediate_size = config["intermediate_size"]

	wte = tensor_for(state, "wte.weight", [vocab_size, hidden_size])
	blocks = []
	for layer_index in range(config["num_layers"]):
		prefix = f"h.{layer_index}"
		blocks.append(
			{
				"ln_1_weight": tensor_file(
					tensor_for(state, f"{prefix}.ln_1.weight", [hidden_size])
				),
				"ln_1_bias": tensor_file(
					tensor_for(state, f"{prefix}.ln_1.bias", [hidden_size])
				),
				"c_attn_weight": tensor_file(
					tensor_for(state, f"{prefix}.attn.c_attn.weight", [hidden_size, 3 * hidden_size])
				),
				"c_attn_bias": tensor_file(
					tensor_for(state, f"{prefix}.attn.c_attn.bias", [3 * hidden_size])
				),
				"attn_c_proj_weight": tensor_file(
					tensor_for(state, f"{prefix}.attn.c_proj.weight", [hidden_size, hidden_size])
				),
				"attn_c_proj_bias": tensor_file(
					tensor_for(state, f"{prefix}.attn.c_proj.bias", [hidden_size])
				),
				"ln_2_weight": tensor_file(
					tensor_for(state, f"{prefix}.ln_2.weight", [hidden_size])
				),
				"ln_2_bias": tensor_file(
					tensor_for(state, f"{prefix}.ln_2.bias", [hidden_size])
				),
				"c_fc_weight": tensor_file(
					tensor_for(state, f"{prefix}.mlp.c_fc.weight", [hidden_size, intermediate_size])
				),
				"c_fc_bias": tensor_file(
					tensor_for(state, f"{prefix}.mlp.c_fc.bias", [intermediate_size])
				),
				"mlp_c_proj_weight": tensor_file(
					tensor_for(state, f"{prefix}.mlp.c_proj.weight", [intermediate_size, hidden_size])
				),
				"mlp_c_proj_bias": tensor_file(
					tensor_for(state, f"{prefix}.mlp.c_proj.bias", [hidden_size])
				),
			}
		)

	return {
		"wte": tensor_file(wte),
		"wpe": tensor_file(tensor_for(state, "wpe.weight", [max_positions, hidden_size])),
		"blocks": blocks,
		"ln_f_weight": tensor_file(tensor_for(state, "ln_f.weight", [hidden_size])),
		"ln_f_bias": tensor_file(tensor_for(state, "ln_f.bias", [hidden_size])),
		"lm_head_weight": tensor_file(wte.transpose(0, 1).contiguous()),
	}


def validate_weight_shapes(state: dict[str, Any], config: dict[str, Any]) -> None:
	hidden_size = config["hidden_size"]
	vocab_size = config["vocab_size"]
	max_positions = config["max_position_embeddings"]
	intermediate_size = config["intermediate_size"]

	tensor_for(state, "wte.weight", [vocab_size, hidden_size])
	tensor_for(state, "wpe.weight", [max_positions, hidden_size])
	for layer_index in range(config["num_layers"]):
		prefix = f"h.{layer_index}"
		tensor_for(state, f"{prefix}.ln_1.weight", [hidden_size])
		tensor_for(state, f"{prefix}.ln_1.bias", [hidden_size])
		tensor_for(state, f"{prefix}.attn.c_attn.weight", [hidden_size, 3 * hidden_size])
		tensor_for(state, f"{prefix}.attn.c_attn.bias", [3 * hidden_size])
		tensor_for(state, f"{prefix}.attn.c_proj.weight", [hidden_size, hidden_size])
		tensor_for(state, f"{prefix}.attn.c_proj.bias", [hidden_size])
		tensor_for(state, f"{prefix}.ln_2.weight", [hidden_size])
		tensor_for(state, f"{prefix}.ln_2.bias", [hidden_size])
		tensor_for(state, f"{prefix}.mlp.c_fc.weight", [hidden_size, intermediate_size])
		tensor_for(state, f"{prefix}.mlp.c_fc.bias", [intermediate_size])
		tensor_for(state, f"{prefix}.mlp.c_proj.weight", [intermediate_size, hidden_size])
		tensor_for(state, f"{prefix}.mlp.c_proj.bias", [hidden_size])
	tensor_for(state, "ln_f.weight", [hidden_size])
	tensor_for(state, "ln_f.bias", [hidden_size])


def tensor_for(
	state: dict[str, Any], name: str, expected_shape: list[int] | None = None
) -> Any:
	candidates = [name, f"transformer.{name}"]
	for candidate in candidates:
		tensor = state.get(candidate)
		if tensor is not None:
			if expected_shape is not None and list(tensor.shape) != expected_shape:
				raise SystemExit(
					f"unexpected shape for {candidate}: "
					f"got {tuple(tensor.shape)}, expected {tuple(expected_shape)}"
				)
			return tensor

	raise SystemExit(f"missing tensor: {name}")


def tensor_file(tensor: Any) -> dict[str, Any]:
	tensor = tensor.detach().cpu().float().contiguous()
	return {
		"shape": list(tensor.shape),
		"data": tensor.reshape(-1).tolist(),
	}


def copy_tokenizer_files(source_dir: Path, output_dir: Path) -> None:
	tokenizer_dir = output_dir / "tokenizer"
	tokenizer_dir.mkdir()
	for filename in ["vocab.json", "merges.txt", "tokenizer.json"]:
		source_path = source_dir / filename
		if source_path.exists():
			shutil.copy2(source_path, tokenizer_dir / filename)


if __name__ == "__main__":
	main()
