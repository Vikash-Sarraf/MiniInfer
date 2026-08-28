use std::{fs::File, path::Path};

use serde::Deserialize;

use crate::{
    error::{MiniInferError, Result},
    model::config::{Architecture, ModelConfig},
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
}