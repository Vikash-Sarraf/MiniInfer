use std::{fs::File, path::Path};

use serde::Deserialize;

use crate::{
    error::{MiniInferError, Result},
    model::config::{Architecture, ModelConfig},
    model::gpt2::{Gpt2BlockWeights, Gpt2Weights},
    tensor::Tensor,
};

#[derive(Deserialize)]
struct ModelConfigFile {
    format_version: u32,
    architecture: String,
    dtype: String,
    vocab_size: usize,
    max_position_embeddings: usize,
    hidden_size: usize,
    num_layers: usize,
    num_heads: usize,
    intermediate_size: usize,
    layer_norm_epsilon: f32,
}

#[derive(Deserialize)]
struct TensorFile {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Deserialize)]
struct Gpt2BlockWeightsFile {
    ln_1_weight: TensorFile,
    ln_1_bias: TensorFile,
    c_attn_weight: TensorFile,
    c_attn_bias: TensorFile,
    attn_c_proj_weight: TensorFile,
    attn_c_proj_bias: TensorFile,
    ln_2_weight: TensorFile,
    ln_2_bias: TensorFile,
    c_fc_weight: TensorFile,
    c_fc_bias: TensorFile,
    mlp_c_proj_weight: TensorFile,
    mlp_c_proj_bias: TensorFile,
}

#[derive(Deserialize)]
struct Gpt2WeightsFile {
    wte: TensorFile,
    wpe: TensorFile,
    blocks: Vec<Gpt2BlockWeightsFile>,
    ln_f_weight: TensorFile,
    ln_f_bias: TensorFile,
    lm_head_weight: TensorFile,
}

pub fn load_config(path: impl AsRef<Path>) -> Result<ModelConfig> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open config {}: {error}", path.display()),
    })?;

    let file_config: ModelConfigFile =
        serde_json::from_reader(file).map_err(|error| MiniInferError::InvalidConfig {
            message: format!("failed to parse config {}: {error}", path.display()),
        })?;

    if file_config.format_version != 1 {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "unsupported config format version {}",
                file_config.format_version
            ),
        });
    }

    if file_config.dtype != "f32" {
        return Err(MiniInferError::InvalidConfig {
            message: format!("unsupported dtype {}", file_config.dtype),
        });
    }

    let architecture = match file_config.architecture.as_str() {
        "gpt2" => Architecture::Gpt2,
        other => {
            return Err(MiniInferError::InvalidConfig {
                message: format!("unsupported architecture {other}"),
            })
        }
    };

    let config = ModelConfig {
        architecture,
        vocab_size: file_config.vocab_size,
        max_position_embeddings: file_config.max_position_embeddings,
        hidden_size: file_config.hidden_size,
        num_layers: file_config.num_layers,
        num_heads: file_config.num_heads,
        intermediate_size: file_config.intermediate_size,
        layer_norm_epsilon: file_config.layer_norm_epsilon,
    };

    config.validate()?;
    Ok(config)
}

pub fn load_gpt2_weights(path: impl AsRef<Path>) -> Result<Gpt2Weights> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open weights {}: {error}", path.display()),
    })?;

    let file_weights: Gpt2WeightsFile =
        serde_json::from_reader(file).map_err(|error| MiniInferError::InvalidConfig {
            message: format!("failed to parse weights {}: {error}", path.display()),
        })?;

    let mut blocks = Vec::with_capacity(file_weights.blocks.len());
    for block in file_weights.blocks {
        blocks.push(Gpt2BlockWeights {
            ln_1_weight: tensor_from_file(block.ln_1_weight)?,
            ln_1_bias: tensor_from_file(block.ln_1_bias)?,
            c_attn_weight: tensor_from_file(block.c_attn_weight)?,
            c_attn_bias: tensor_from_file(block.c_attn_bias)?,
            attn_c_proj_weight: tensor_from_file(block.attn_c_proj_weight)?,
            attn_c_proj_bias: tensor_from_file(block.attn_c_proj_bias)?,
            ln_2_weight: tensor_from_file(block.ln_2_weight)?,
            ln_2_bias: tensor_from_file(block.ln_2_bias)?,
            c_fc_weight: tensor_from_file(block.c_fc_weight)?,
            c_fc_bias: tensor_from_file(block.c_fc_bias)?,
            mlp_c_proj_weight: tensor_from_file(block.mlp_c_proj_weight)?,
            mlp_c_proj_bias: tensor_from_file(block.mlp_c_proj_bias)?,
        });
    }

    Ok(Gpt2Weights {
        wte: tensor_from_file(file_weights.wte)?,
        wpe: tensor_from_file(file_weights.wpe)?,
        blocks,
        ln_f_weight: tensor_from_file(file_weights.ln_f_weight)?,
        ln_f_bias: tensor_from_file(file_weights.ln_f_bias)?,
        lm_head_weight: tensor_from_file(file_weights.lm_head_weight)?,
    })
}

fn tensor_from_file(tensor: TensorFile) -> Result<Tensor> {
    Tensor::new(tensor.shape, tensor.data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_tiny_gpt2_config() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2/config.json");

        let config = load_config(path).expect("tiny GPT-2 config should load");

        assert_eq!(config.architecture, Architecture::Gpt2);
        assert_eq!(config.vocab_size, 8);
        assert_eq!(config.max_position_embeddings, 8);
        assert_eq!(config.hidden_size, 4);
        assert_eq!(config.num_layers, 1);
        assert_eq!(config.num_heads, 2);
        assert_eq!(config.intermediate_size, 16);
        assert_eq!(config.layer_norm_epsilon, 1e-5);
    }

    #[test]
    fn loads_tiny_gpt2_weights() {
        let config_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2/config.json");
        let weights_path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2/weights.json");

        let config = load_config(config_path).expect("tiny GPT-2 config should load");
        let weights = load_gpt2_weights(weights_path).expect("tiny GPT-2 weights should load");

        weights
            .validate_shapes(&config)
            .expect("tiny GPT-2 weights should match config");
        assert_eq!(weights.wte.shape(), &[8, 4]);
        assert_eq!(weights.blocks.len(), 1);
        assert_eq!(weights.lm_head_weight.shape(), &[4, 8]);
    }
}