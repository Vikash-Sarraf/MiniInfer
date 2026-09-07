use std::{fs::File, path::Path};

use serde::Deserialize;

use crate::{
    error::{MiniInferError, Result},
    tokenizer::{
        api::{LoadedTokenizer, TinyTokenizer},
        gpt2::Gpt2Tokenizer,
    },
};

#[derive(Deserialize)]
struct VocabFile {
    tokens: Vec<String>,
}

pub(super) fn load_tokenizer(model_dir: &Path) -> Result<LoadedTokenizer> {
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
