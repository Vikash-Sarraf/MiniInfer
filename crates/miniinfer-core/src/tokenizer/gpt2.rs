use std::{collections::HashMap, path::Path};

use crate::error::{MiniInferError, Result};

#[derive(Debug)]
pub struct Gpt2Tokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
}

impl Gpt2Tokenizer {
    pub fn from_vocab_file(path: impl AsRef<Path>) -> Result<Self> {
        let vocab_file = std::fs::read_to_string(path).map_err(|e| MiniInferError::InvalidConfig {
            message: format!("Failed to read vocab file: {}", e),
        })?;

        let vocab_data: HashMap<String, usize> = serde_json::from_str(&vocab_file)
            .map_err(|e| 
                MiniInferError::InvalidConfig { message: format!("Invalid vocab: {}", e) 
            })?;
        Self::from_vocab_map(vocab_data)
    }

    pub fn from_vocab_map(vocab: HashMap<String, usize>) -> Result<Self> {
        let mut id_to_token = vec![String::new(); vocab.len()];

        for (token, &id) in &vocab {
            if id >= vocab.len() {
                return Err(MiniInferError::InvalidConfig {
                    message: format!("vocab id {id} is out of range for vocab length {}", vocab.len()),
                });
            }

            id_to_token[id] = token.clone();
        }

        for (id, token) in id_to_token.iter().enumerate() {
            if token.is_empty() {
                return Err(MiniInferError::InvalidConfig {
                    message: format!("vocab id {id} is missing"),
                });
            }
        }

        Ok(Self {
            token_to_id: vocab,
            id_to_token,
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_vocab() -> HashMap<String, usize> {
        HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 1),
        ])
    }

    #[test]
    fn builds_lookup_maps_from_valid_vocab() {
        let tokenizer = Gpt2Tokenizer::from_vocab_map(tiny_vocab())
            .expect("valid vocab should build tokenizer");

        assert_eq!(tokenizer.token_to_id("hello"), Some(0));
        assert_eq!(tokenizer.token_to_id("world"), Some(1));
        assert_eq!(tokenizer.id_to_token(0), Some("hello"));
        assert_eq!(tokenizer.id_to_token(1), Some("world"));
    }

    #[test]
    fn unknown_lookups_return_none() {
        let tokenizer = Gpt2Tokenizer::from_vocab_map(tiny_vocab())
            .expect("valid vocab should build tokenizer");

        assert_eq!(tokenizer.token_to_id("missing"), None);
        assert_eq!(tokenizer.id_to_token(2), None);
    }

    #[test]
    fn rejects_non_contiguous_vocab_ids() {
        let vocab = HashMap::from([
            ("hello".to_string(), 0),
            ("world".to_string(), 2),
        ]);

        let err = Gpt2Tokenizer::from_vocab_map(vocab)
            .expect_err("non-contiguous vocab IDs should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "vocab id 2 is out of range for vocab length 2".to_string(),
            }
        );
    }

    #[test]
    fn from_vocab_file_loads_valid_json() {
        let vocab_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tokenizer/gpt2-vocab.json");
        let tokenizer = Gpt2Tokenizer::from_vocab_file(vocab_path)
            .expect("valid vocab file should build tokenizer");
        assert_eq!(tokenizer.token_to_id("hello"), Some(0));
        assert_eq!(tokenizer.token_to_id("world"), Some(1));
        assert_eq!(tokenizer.id_to_token(0), Some("hello"));
        assert_eq!(tokenizer.id_to_token(1), Some("world"));
    }
}