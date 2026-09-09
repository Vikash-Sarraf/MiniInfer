use std::path::Path;

use crate::{
    error::{MiniInferError, Result},
    model::{config::{Architecture, ModelConfig}, gpt2::Gpt2Weights},
    ops::backend::{OpsBackend, ReferenceBackend},
    runtime::kv_cache::KvCache,
    tensor::Tensor,
    tokenizer::api::{LoadedTokenizer, Tokenizer},
};

pub use super::config_io::load_config;
pub use super::weights_io::{
    load_gpt2_binary_weights,
    load_gpt2_binary_weights_with_runtime,
    load_gpt2_weights,
    WeightRuntime,
};

use super::{tokenizer_io::load_tokenizer, weights_io::{load_gpt2_weights_from_model_dir_with_runtime}};

#[cfg(test)]
use crate::model::{
    gpt2::LMHead,
    weights_io::{load_lm_head, LmHeadFile, TensorFile},
};

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

    pub fn forward_next_token_with_cache_and_backend(
        &self,
        token_id: usize,
        kv_cache: &mut KvCache,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => {
                weights.forward_next_token_with_cache_and_backend(config, token_id, kv_cache, backend)
            }
        }
    }

    pub fn forward_next_token_with_cache(
        &self,
        token_id: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => {
                weights.forward_next_token_with_cache_and_backend(config, token_id, kv_cache, &backend)
            }
        }
    }

    pub fn forward_prefill_with_cache_and_backend(
        &self,
        token_ids: &[usize],
        kv_cache: &mut KvCache,
        backend: &dyn OpsBackend,
    ) -> Result<Tensor> {
        match self {
            LoadedModel::Gpt2 { config, weights, .. } => {
                weights.forward_prefill_with_cache_and_backend(config, token_ids, kv_cache, backend)
            }
        }
    }

    pub fn forward_prefill_with_cache(
        &self,
        token_ids: &[usize],
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let backend = ReferenceBackend::new();
        self.forward_prefill_with_cache_and_backend(token_ids, kv_cache, &backend)
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

pub fn load_model_with_runtime(model_dir: impl AsRef<Path>, runtime: WeightRuntime) -> Result<LoadedModel> {
    let model_dir = model_dir.as_ref();

    let config = load_config(model_dir.join("config.json"))?;

    match &config.architecture {
        Architecture::Gpt2 => {
            let weights = load_gpt2_weights_from_model_dir_with_runtime(model_dir, &config, runtime)?;
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

pub fn load_model(model_dir: impl AsRef<Path>) -> Result<LoadedModel> {
    load_model_with_runtime(model_dir, WeightRuntime::F32)
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
    fn loaded_model_forward_next_token_with_cache_returns_logits_and_updates_cache() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");
        let config = model.config();
        let mut cache = KvCache::new(
            config.num_layers,
            config.num_heads,
            config.head_dim(),
            config.max_position_embeddings,
        )
        .expect("cache should be valid");
        let backend = ReferenceBackend::new();

        let logits = model
            .forward_next_token_with_cache_and_backend(0, &mut cache, &backend)
            .expect("cached forward should succeed");

        assert_eq!(logits.shape(), &[1, config.vocab_size]);
        assert_eq!(cache.current_position().expect("position should exist"), 1);
    }

    #[test]
    fn loaded_model_forward_prefill_with_cache_returns_logits_and_updates_cache() {
        let model_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/tiny-gpt2");
        let model = load_model(model_dir).expect("tiny GPT-2 model should load");
        let config = model.config();
        let mut cache = KvCache::new(
            config.num_layers,
            config.num_heads,
            config.head_dim(),
            config.max_position_embeddings,
        )
        .expect("cache should be valid");
        let backend = ReferenceBackend::new();
        let token_ids = [0, 1];

        let logits = model
            .forward_prefill_with_cache_and_backend(&token_ids, &mut cache, &backend)
            .expect("prefill forward should succeed");

        assert_eq!(logits.shape(), &[token_ids.len(), config.vocab_size]);
        assert_eq!(cache.current_position().expect("position should exist"), token_ids.len());
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

        let weights = load_gpt2_weights_from_model_dir_with_runtime(&model_dir, &config, WeightRuntime::F32)
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

        let err = match load_gpt2_weights_from_model_dir_with_runtime(&model_dir, &config, WeightRuntime::F32) {
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