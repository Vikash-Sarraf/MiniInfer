use std::{collections::HashMap, fs::File, io::{Read, Seek, SeekFrom}, path::Path};

use serde::Deserialize;

use crate::{
    error::{MiniInferError, Result},
    model::{config::{Architecture, ModelConfig}, gpt2::{Gpt2BlockWeights, Gpt2Weights, LMHead}},
    ops::backend::OpsBackend,
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
    lm_head: Option<LmHeadFile>,
    lm_head_weight: Option<TensorFile>,
}

#[derive(Deserialize)]
struct LmHeadFile {
    #[serde(rename = "type")]
    head_type: String,
    weight: Option<TensorFile>,
}

#[derive(Deserialize)]
struct BinaryWeightsIndexFile {
    format_version: u32,
    dtype: String,
    endianness: String,
    tensors: HashMap<String, BinaryTensorIndexFile>,
    lm_head: Option<LmHeadFile>,
}

#[derive(Deserialize)]
struct BinaryTensorIndexFile {
    shape: Vec<usize>,
    offset_bytes: u64,
    len: usize,
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

    pub fn forward_with_backend(&self, token_ids: &[usize], backend: &dyn OpsBackend) -> Result<Tensor> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => {
                weights.forward_with_backend(config, token_ids, backend)
            }
        }
    }

    pub fn forward_last_logits(&self, token_ids: &[usize]) -> Result<Tensor> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => weights.forward_last_logits(config, token_ids),
        }
    }

    pub fn forward_last_logits_with_backend(
        &self,
        token_ids: &[usize],
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => {
                weights.forward_last_logits_with_backend(config, token_ids, backend)
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
            let weights = load_gpt2_weights_from_model_dir(model_dir, &config)?;
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

fn load_gpt2_weights_from_model_dir(model_dir: &Path, config: &ModelConfig) -> Result<Gpt2Weights> {
    let binary_index_path = model_dir.join("weights.index.json");
    let binary_data_path = model_dir.join("weights.bin");

    match (binary_index_path.exists(), binary_data_path.exists()) {
        (true, true) => load_gpt2_binary_weights(binary_index_path, binary_data_path, config),
        (false, false) => load_gpt2_weights(model_dir.join("weights.json")),
        _ => Err(MiniInferError::InvalidConfig {
            message: "binary weights require both weights.index.json and weights.bin".to_string(),
        }),
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
        lm_head_weight: load_lm_head(file_weights.lm_head, file_weights.lm_head_weight)?,
    })
}

pub fn load_gpt2_binary_weights(
    index_path: impl AsRef<Path>,
    data_path: impl AsRef<Path>,
    config: &ModelConfig,
) -> Result<Gpt2Weights> {
    let index_path = index_path.as_ref();
    let data_path = data_path.as_ref();

    let index_file = File::open(index_path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open weights index {}: {error}", index_path.display()),
    })?;
    let index: BinaryWeightsIndexFile = serde_json::from_reader(index_file).map_err(|error| {
        MiniInferError::InvalidConfig {
            message: format!("failed to parse weights index {}: {error}", index_path.display()),
        }
    })?;

    validate_binary_weight_index(&index)?;

    let mut data_file = File::open(data_path).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to open weights data {}: {error}", data_path.display()),
    })?;

    let mut blocks = Vec::with_capacity(config.num_layers);
    for layer_index in 0..config.num_layers {
        let prefix = format!("blocks.{layer_index}");
        blocks.push(Gpt2BlockWeights {
            ln_1_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_1_weight"))?,
            ln_1_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_1_bias"))?,
            c_attn_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_attn_weight"))?,
            c_attn_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_attn_bias"))?,
            attn_c_proj_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.attn_c_proj_weight"))?,
            attn_c_proj_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.attn_c_proj_bias"))?,
            ln_2_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_2_weight"))?,
            ln_2_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.ln_2_bias"))?,
            c_fc_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_fc_weight"))?,
            c_fc_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.c_fc_bias"))?,
            mlp_c_proj_weight: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.mlp_c_proj_weight"))?,
            mlp_c_proj_bias: read_binary_tensor(&mut data_file, &index, &format!("{prefix}.mlp_c_proj_bias"))?,
        });
    }

    let lm_head_weight = load_binary_lm_head(&mut data_file, &index)?;

    Ok(Gpt2Weights {
        wte: read_binary_tensor(&mut data_file, &index, "wte")?,
        wpe: read_binary_tensor(&mut data_file, &index, "wpe")?,
        blocks,
        ln_f_weight: read_binary_tensor(&mut data_file, &index, "ln_f_weight")?,
        ln_f_bias: read_binary_tensor(&mut data_file, &index, "ln_f_bias")?,
        lm_head_weight,
    })
}

fn validate_binary_weight_index(index: &BinaryWeightsIndexFile) -> Result<()> {
    if index.format_version != 1 {
        return Err(MiniInferError::InvalidConfig {
            message: format!("unsupported weights index format version {}", index.format_version),
        });
    }

    if index.dtype != "f32" {
        return Err(MiniInferError::InvalidConfig {
            message: format!("unsupported weights dtype {}", index.dtype),
        });
    }

    if index.endianness != "little" {
        return Err(MiniInferError::InvalidConfig {
            message: format!("unsupported weights endianness {}", index.endianness),
        });
    }

    Ok(())
}

fn load_binary_lm_head(data_file: &mut File, index: &BinaryWeightsIndexFile) -> Result<LMHead> {
    match index.lm_head.as_ref().map(|lm_head| lm_head.head_type.as_str()) {
        Some("tied") => Ok(LMHead::Tied),
        Some("untied") => Ok(LMHead::Untied(read_binary_tensor(data_file, index, "lm_head_weight")?)),
        Some(other) => Err(MiniInferError::InvalidConfig {
            message: format!("unsupported LM head type {other}"),
        }),
        None if index.tensors.contains_key("lm_head_weight") => {
            Ok(LMHead::Untied(read_binary_tensor(data_file, index, "lm_head_weight")?))
        }
        None => Err(MiniInferError::InvalidConfig {
            message: "weights index must specify lm_head or lm_head_weight".to_string(),
        }),
    }
}

fn read_binary_tensor(
    data_file: &mut File,
    index: &BinaryWeightsIndexFile,
    name: &str,
) -> Result<Tensor> {
    let tensor_index = index.tensors.get(name).ok_or_else(|| MiniInferError::InvalidConfig {
        message: format!("weights index is missing tensor {name}"),
    })?;
    let expected_len: usize = tensor_index.shape.iter().product();
    if tensor_index.len != expected_len {
        return Err(MiniInferError::InvalidConfig {
            message: format!(
                "tensor {name} index length {} does not match shape product {}",
                tensor_index.len, expected_len
            ),
        });
    }

    let byte_len = tensor_index.len.checked_mul(4).ok_or_else(|| MiniInferError::InvalidConfig {
        message: format!("tensor {name} byte length overflow"),
    })?;
    let mut bytes = vec![0u8; byte_len];
    data_file
        .seek(SeekFrom::Start(tensor_index.offset_bytes))
        .map_err(|error| MiniInferError::InvalidConfig {
            message: format!("failed to seek tensor {name}: {error}"),
        })?;
    data_file.read_exact(&mut bytes).map_err(|error| MiniInferError::InvalidConfig {
        message: format!("failed to read tensor {name}: {error}"),
    })?;

    let mut data = Vec::with_capacity(tensor_index.len);
    for chunk in bytes.chunks_exact(4) {
        let mut value_bytes = [0u8; 4];
        value_bytes.copy_from_slice(chunk);
        data.push(f32::from_le_bytes(value_bytes));
    }

    Tensor::new(tensor_index.shape.clone(), data)
}

fn load_lm_head(
    lm_head: Option<LmHeadFile>,
    legacy_lm_head_weight: Option<TensorFile>,
) -> Result<LMHead> {
    match (lm_head, legacy_lm_head_weight) {
        (Some(lm_head), None) if lm_head.head_type == "tied" => Ok(LMHead::Tied),
        (Some(lm_head), None) if lm_head.head_type == "untied" => {
            let weight = lm_head.weight.ok_or_else(|| MiniInferError::InvalidConfig {
                message: "untied LM head must include a weight tensor".to_string(),
            })?;
            Ok(LMHead::Untied(tensor_from_file(weight)?))
        }
        (Some(lm_head), None) => Err(MiniInferError::InvalidConfig {
            message: format!("unsupported LM head type {}", lm_head.head_type),
        }),
        (None, Some(weight)) => Ok(LMHead::Untied(tensor_from_file(weight)?)),
        (Some(_), Some(_)) => Err(MiniInferError::InvalidConfig {
            message: "weights must specify either lm_head or lm_head_weight, not both".to_string(),
        }),
        (None, None) => Err(MiniInferError::InvalidConfig {
            message: "weights must specify lm_head or lm_head_weight".to_string(),
        }),
    }
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
    use serde_json::{json, Map, Value};

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

    fn add_binary_tensor(
        tensors: &mut Map<String, Value>,
        bytes: &mut Vec<u8>,
        name: &str,
        shape: &[usize],
    ) {
        let len = shape.iter().product::<usize>();
        let offset_bytes = bytes.len();
        for index in 0..len {
            bytes.extend_from_slice(&(index as f32).to_le_bytes());
        }
        tensors.insert(
            name.to_string(),
            json!({
                "shape": shape,
                "offset_bytes": offset_bytes,
                "len": len,
            }),
        );
    }

    fn write_tiny_binary_weights(model_dir: &Path, config: &ModelConfig) {
        let mut tensors = Map::new();
        let mut bytes = Vec::new();

        add_binary_tensor(&mut tensors, &mut bytes, "wte", &[config.vocab_size, config.hidden_size]);
        add_binary_tensor(&mut tensors, &mut bytes, "wpe", &[config.max_position_embeddings, config.hidden_size]);
        for layer_index in 0..config.num_layers {
            let prefix = format!("blocks.{layer_index}");
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.ln_1_weight"), &[config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.ln_1_bias"), &[config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.c_attn_weight"), &[config.hidden_size, 3 * config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.c_attn_bias"), &[3 * config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.attn_c_proj_weight"), &[config.hidden_size, config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.attn_c_proj_bias"), &[config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.ln_2_weight"), &[config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.ln_2_bias"), &[config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.c_fc_weight"), &[config.hidden_size, config.intermediate_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.c_fc_bias"), &[config.intermediate_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.mlp_c_proj_weight"), &[config.intermediate_size, config.hidden_size]);
            add_binary_tensor(&mut tensors, &mut bytes, &format!("{prefix}.mlp_c_proj_bias"), &[config.hidden_size]);
        }
        add_binary_tensor(&mut tensors, &mut bytes, "ln_f_weight", &[config.hidden_size]);
        add_binary_tensor(&mut tensors, &mut bytes, "ln_f_bias", &[config.hidden_size]);

        let index = json!({
            "format_version": 1,
            "dtype": "f32",
            "endianness": "little",
            "lm_head": { "type": "tied" },
            "tensors": tensors,
        });

        std::fs::write(model_dir.join("weights.bin"), bytes)
            .expect("binary weights should be written");
        std::fs::write(model_dir.join("weights.index.json"), index.to_string())
            .expect("binary weights index should be written");
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
        match &weights.lm_head_weight {
            LMHead::Untied(weight) => assert_eq!(weight.shape(), &[4, 8]),
            LMHead::Tied => panic!("tiny GPT-2 weights should use explicit LM head"),
        }
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
    fn load_lm_head_accepts_tied_metadata() {
        let lm_head = load_lm_head(
            Some(LmHeadFile {
                head_type: "tied".to_string(),
                weight: None,
            }),
            None,
        )
        .expect("tied LM head should load");

        assert!(matches!(lm_head, LMHead::Tied));
    }

    #[test]
    fn load_lm_head_rejects_conflicting_metadata() {
        let err = load_lm_head(
            Some(LmHeadFile {
                head_type: "tied".to_string(),
                weight: None,
            }),
            Some(TensorFile {
                shape: vec![2, 2],
                data: vec![0.0; 4],
            }),
        )
        .expect_err("conflicting LM head metadata should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "weights must specify either lm_head or lm_head_weight, not both".to_string(),
            }
        );
    }

    #[test]
    fn load_gpt2_weights_from_model_dir_prefers_binary_weights() {
        let model_dir = temp_model_dir("binary-weights");
        std::fs::create_dir_all(&model_dir).expect("model dir should be created");
        let config = load_config(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2/config.json"),
        )
        .expect("tiny GPT-2 config should load");
        write_tiny_binary_weights(&model_dir, &config);
        std::fs::write(model_dir.join("weights.json"), "not valid json")
            .expect("legacy weights placeholder should be written");

        let weights = load_gpt2_weights_from_model_dir(&model_dir, &config)
            .expect("binary weights should load before legacy JSON");

        assert_eq!(weights.wte.shape(), &[config.vocab_size, config.hidden_size]);
        assert_eq!(weights.blocks.len(), config.num_layers);
        assert!(matches!(weights.lm_head_weight, LMHead::Tied));

        std::fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
    }

    #[test]
    fn load_gpt2_weights_from_model_dir_rejects_partial_binary_weights() {
        let model_dir = temp_model_dir("partial-binary-weights");
        std::fs::create_dir_all(&model_dir).expect("model dir should be created");
        let config = load_config(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2/config.json"),
        )
        .expect("tiny GPT-2 config should load");
        std::fs::write(model_dir.join("weights.index.json"), "{}").expect("index should be written");

        let err = match load_gpt2_weights_from_model_dir(&model_dir, &config) {
            Ok(_) => panic!("partial binary weights should fail"),
            Err(err) => err,
        };

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "binary weights require both weights.index.json and weights.bin".to_string(),
            }
        );

        std::fs::remove_dir_all(model_dir).expect("temp model dir should be removed");
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