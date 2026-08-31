use std::{collections::HashMap, path::Path};

use crate::error::{MiniInferError, Result};

#[derive(Debug)]
pub struct Gpt2Tokenizer {
    token_to_id: HashMap<String, usize>,
    id_to_token: Vec<String>,
    merges: HashMap<(String, String), usize>,
}

impl Gpt2Tokenizer {
    pub fn parse_merges_text(text: &str) -> Result<HashMap<(String, String), usize>> {
        let mut merges = HashMap::new();
        let mut rank = 0;

        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("#") {
                continue;
            }

            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() != 2 {
                return Err(MiniInferError::InvalidConfig {
                    message: format!("invalid merge line: {line}"),
                });
            }

            let left = parts[0].to_string();
            let right = parts[1].to_string();

            merges.insert((left, right), rank);
            rank += 1;
        }

        Ok(merges)
    }

    pub fn load_merges_file(path: impl AsRef<Path>) -> Result<HashMap<(String, String), usize>> {
        let merges_file = std::fs::read_to_string(path).map_err(|e| MiniInferError::InvalidConfig {
            message: format!("Failed to read merges file: {}", e),
        })?;

        Self::parse_merges_text(&merges_file)
    }

    pub fn from_vocab_file(path: impl AsRef<Path>) -> Result<Self> {
        let vocab_file = std::fs::read_to_string(path).map_err(|e| MiniInferError::InvalidConfig {
            message: format!("Failed to read vocab file: {}", e),
        })?;

        let vocab_data: HashMap<String, usize> =
            serde_json::from_str(&vocab_file).map_err(|e| MiniInferError::InvalidConfig {
                message: format!("Invalid vocab: {}", e),
            })?;
        Self::from_vocab_map(vocab_data)
    }

    pub fn from_vocab_map(vocab: HashMap<String, usize>) -> Result<Self> {
        Self::from_vocab_and_merges(vocab, HashMap::new())
    }

    pub fn from_vocab_and_merges(
        vocab: HashMap<String, usize>,
        merges: HashMap<(String, String), usize>,
    ) -> Result<Self> {
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
            merges,
        })
    }

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn id_to_token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(|s| s.as_str())
    }

    pub fn merge_rank(&self, left: &str, right: &str) -> Option<usize> {
        self.merges
            .get(&(left.to_string(), right.to_string()))
            .copied()
    }

    pub fn apply_bpe_merges(
        &self,
        pieces: &[String],
    ) -> Vec<String> {
        let mut current = pieces.to_vec();
        while let Some(next) = merge_once(&current, &self.merges) {
            current = next;
        }

        current
    }

}

fn merge_once(
    pieces: &[String],
    merges: &HashMap<(String, String), usize>,
) -> Option<Vec<String>> {
    if pieces.len() < 2 {
        return None;
    }

    let mut best_index: Option<usize> = None;
    let mut best_rank: Option<usize> = None;

    for index in 0..(pieces.len() - 1) {
        let pair = (pieces[index].clone(), pieces[index + 1].clone());

        if let Some(&rank) = merges.get(&pair) {
            if best_rank.is_none() || rank < best_rank.unwrap() {
                best_rank = Some(rank);
                best_index = Some(index);
            }
        }
    }

    let best_index = match best_index {
        Some(index) => index,
        None => return None,
    };

    let mut output = Vec::with_capacity(pieces.len() - 1);
    let mut index = 0;

    while index < pieces.len() {
        if index == best_index {
            output.push(format!("{}{}", pieces[index], pieces[index + 1]));
            index += 2;
        } else {
            output.push(pieces[index].clone());
            index += 1;
        }
    }

    Some(output)
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

    #[test]
    fn load_merges_file_loads_valid_merges_txt() {
        let merges_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/tokenizer/gpt2-merges.txt");

        let merges = Gpt2Tokenizer::load_merges_file(merges_path)
            .expect("valid merges file should parse");

        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0));
        assert_eq!(
            merges.get(&("he".to_string(), "llo".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn parse_merges_text_assigns_ranks_to_valid_merges() {
        let merges = Gpt2Tokenizer::parse_merges_text(
            "#version: 0.2\nh e\nhe llo\nĠ world\n",
        )
        .expect("valid merges should parse");

        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0));
        assert_eq!(
            merges.get(&("he".to_string(), "llo".to_string())),
            Some(&1)
        );
        assert_eq!(
            merges.get(&("Ġ".to_string(), "world".to_string())),
            Some(&2)
        );
    }

    #[test]
    fn parse_merges_text_skips_empty_lines_and_comments() {
        let merges = Gpt2Tokenizer::parse_merges_text(
            "#version: 0.2\n\nh e\n\n# comment\nhe llo\n",
        )
        .expect("valid merges should parse");

        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0));
        assert_eq!(
            merges.get(&("he".to_string(), "llo".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn parse_merges_text_rejects_malformed_lines() {
        let err = Gpt2Tokenizer::parse_merges_text("h e extra")
            .expect_err("malformed merge line should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "invalid merge line: h e extra".to_string(),
            }
        );
    }

    #[test]
    fn merge_rank_returns_rank_for_known_pair() {
        let merges = Gpt2Tokenizer::parse_merges_text("h e\nhe llo\n")
            .expect("valid merges should parse");
        let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_vocab(), merges)
            .expect("valid vocab and merges should build tokenizer");

        assert_eq!(tokenizer.merge_rank("h", "e"), Some(0));
        assert_eq!(tokenizer.merge_rank("he", "llo"), Some(1));
        assert_eq!(tokenizer.merge_rank("missing", "pair"), None);
    }

    #[test]
fn merge_once_merges_best_ranked_pair() {
    let pieces = vec![
        "h".to_string(),
        "e".to_string(),
        "l".to_string(),
        "l".to_string(),
        "o".to_string(),
    ];

    let merges = HashMap::from([
        (("h".to_string(), "e".to_string()), 0),
        (("l".to_string(), "l".to_string()), 1),
    ]);

    let output = merge_once(&pieces, &merges).expect("one merge should apply");

    assert_eq!(
        output,
        vec![
            "he".to_string(),
            "l".to_string(),
            "l".to_string(),
            "o".to_string(),
        ]
    );
}

#[test]
fn apply_bpe_merges_applies_merges_until_no_more() {
    let pieces = vec![
        "h".to_string(),
        "e".to_string(),
        "l".to_string(),
        "l".to_string(),
        "o".to_string(),
    ];

    let merges = HashMap::from([
        (("h".to_string(), "e".to_string()), 0),
        (("he".to_string(), "l".to_string()), 1),
        (("hel".to_string(), "l".to_string()), 2),
        (("hell".to_string(), "o".to_string()), 3),
    ]);

    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_vocab(), merges)
        .expect("valid vocab and merges should build tokenizer");
    let output = tokenizer.apply_bpe_merges(&pieces);

    assert_eq!(output, vec!["hello".to_string()]);
}
}