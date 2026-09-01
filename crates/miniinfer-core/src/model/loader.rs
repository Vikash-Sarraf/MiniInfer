use std::{fs::File, path::Path};

use serde::Deserialize;

use crate::{
    error::{MiniInferError, Result},
    model::{config::{Architecture, ModelConfig}, gpt2::{Gpt2BlockWeights, Gpt2Weights}},
    tensor::Tensor,
    tokenizer::{gpt2::Gpt2Tokenizer, tokenizer::{LoadedTokenizer, TinyTokenizer, Tokenizer}},
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

#[derive(Deserialize)]
struct VocabFile {
    tokens: Vec<String>,
}

pub enum LoadedModel {
    Gpt2 {
        config: ModelConfig,
        weights: Gpt2Weights,
        tokenizer: LoadedTokenizer,
    },
}



impl LoadedModel {
    pub fn config(&self) -> &ModelConfig {
        match self {
            LoadedModel::Gpt2 { config, .. } => config,
        }
    }

    pub fn validate(&self) -> Result<()> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => weights.validate_shapes(config),
        }
    }

    pub fn forward(&self, token_ids: &[usize]) -> Result<Tensor> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => {
                weights.forward(config, token_ids)
            }
        }
    }

    pub fn vocab(&self) -> &[String] {
        match self {
            LoadedModel::Gpt2 { tokenizer, .. } => tokenizer.vocab(),
        }
    }

    pub fn encode_prompt(&self, prompt: &str) -> Result<Vec<usize>> {
        match self {
            LoadedModel::Gpt2 { tokenizer, .. } => tokenizer.encode(prompt),
        }
    }

    pub fn decode_tokens(&self, token_ids: &[usize]) -> Result<String> {
        match self {
            LoadedModel::Gpt2 { tokenizer, .. } => tokenizer.decode(token_ids),
        }
    }
}


pub fn load_model(model_dir: impl AsRef<Path>) -> Result<LoadedModel> {
    let model_dir = model_dir.as_ref();

    let config = load_config(model_dir.join("config.json"))?;

    match &config.architecture {
        Architecture::Gpt2 => {
            let weights = load_gpt2_weights(model_dir.join("weights.json"))?;
            weights.validate_shapes(&config)?;
            let tokenizer = load_tokenizer(model_dir)?;

            if tokenizer.vocab().len() != config.vocab_size {
                return Err(MiniInferError::InvalidConfig {
                    message: format!(
                        "vocab length {} must match config vocab_size {}",
                        tokenizer.vocab().len(),
                        config.vocab_size
                    ),
                });
            }

            Ok(LoadedModel::Gpt2 { config, weights, tokenizer })
        }
    }
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

fn load_vocab(path: impl AsRef<Path>) -> Result<Vec<String>> {
    let path = path.as_ref();
    let file = File::open(path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open vocab {}: {error}", path.display()),
    })?;

    let vocab_file: VocabFile =
        serde_json::from_reader(file).map_err(|error| MiniInferError::InvalidConfig {
            message: format!("failed to parse vocab {}: {error}", path.display()),
        })?;

    Ok(vocab_file.tokens)
}

fn load_tokenizer(model_dir: &Path) -> Result<LoadedTokenizer> {
    let tokenizer_dir = model_dir.join("tokenizer");
    let gpt2_vocab_path = tokenizer_dir.join("vocab.json");
    let gpt2_merges_path = tokenizer_dir.join("merges.txt");

    if gpt2_vocab_path.exists() && gpt2_merges_path.exists() {
        let vocab_file = std::fs::read_to_string(&gpt2_vocab_path).map_err(|error| {
            MiniInferError::InvalidConfig {
                message: format!("failed to read tokenizer vocab {}: {error}", gpt2_vocab_path.display()),
            }
        })?;
        let vocab = serde_json::from_str(&vocab_file).map_err(|error| {
            MiniInferError::InvalidConfig {
                message: format!("failed to parse tokenizer vocab {}: {error}", gpt2_vocab_path.display()),
            }
        })?;
        let merges = Gpt2Tokenizer::load_merges_file(gpt2_merges_path)?;
        let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(vocab, merges)?;

        return Ok(LoadedTokenizer::Gpt2(tokenizer));
    }

    let vocab = load_vocab(model_dir.join("vocab.json"))?;
    Ok(LoadedTokenizer::Tiny(TinyTokenizer::new(vocab)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_model_dir(test_name: &str) -> std::path::PathBuf {
        let unique_suffix = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock should be after UNIX epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "miniinfer-{test_name}-{}-{unique_suffix}",
            std::process::id()
        ))
    }

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

    #[test]
    fn loads_tiny_gpt2_model() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let config_path = model_dir.join("config.json");
        let weights_path = model_dir.join("weights.json");
        let config = load_config(config_path).expect("tiny GPT-2 config should load");
        let weights = load_gpt2_weights(weights_path).expect("tiny GPT-2 weights should load");

        weights
            .validate_shapes(&config)
            .expect("tiny GPT-2 weights should match config");
    }

    #[test]
    fn loads_tiny_gpt2_model_via_loader() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");

        let config = model.config();
        assert_eq!(config.architecture, Architecture::Gpt2);
        assert_eq!(config.vocab_size, 8);
        assert_eq!(config.max_position_embeddings, 8);
        assert_eq!(config.hidden_size, 4);
        assert_eq!(config.num_layers, 1);
        assert_eq!(config.num_heads, 2);
        assert_eq!(config.intermediate_size, 16);
        assert_eq!(config.layer_norm_epsilon, 1e-5);
        assert_eq!(model.vocab().len(), config.vocab_size);

        model.validate().expect("tiny GPT-2 model should validate");
    }

    #[test]
    fn load_tokenizer_prefers_gpt2_tokenizer_artifacts_when_present() {
        let model_dir = temp_model_dir("gpt2-tokenizer-artifacts");
        let tokenizer_dir = model_dir.join("tokenizer");
        std::fs::create_dir_all(&tokenizer_dir).expect("tokenizer dir should be created");
        std::fs::write(
            tokenizer_dir.join("vocab.json"),
            r#"{"hello":0,"Ġworld":1}"#,
        )
        .expect("GPT-2 vocab fixture should be written");
        std::fs::write(
            tokenizer_dir.join("merges.txt"),
            "#version: 0.2\nh e\nhe l\nhel l\nhell o\nĠ w\nĠw o\nĠwo r\nĠwor l\nĠworl d\n",
        )
        .expect("GPT-2 merges fixture should be written");

        let tokenizer = load_tokenizer(&model_dir).expect("GPT-2 tokenizer should load");

        match tokenizer {
            LoadedTokenizer::Gpt2(tokenizer) => {
                assert_eq!(tokenizer.vocab().len(), 2);
                assert_eq!(
                    tokenizer.encode("hello world").expect("prompt should encode"),
                    vec![0, 1]
                );
            }
            LoadedTokenizer::Tiny(_) => panic!("GPT-2 tokenizer artifacts should be preferred"),
        }

        std::fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
    }
}