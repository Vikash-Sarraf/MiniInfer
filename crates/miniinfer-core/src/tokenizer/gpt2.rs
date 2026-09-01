use std::{collections::HashMap, path::Path, sync::LazyLock};

use crate::{error::{MiniInferError, Result}, tokenizer::tokenizer::Tokenizer};

static GPT2_PRE_TOKEN_PATTERN: LazyLock<fancy_regex::Regex> = LazyLock::new(|| {
    fancy_regex::Regex::new(
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+",
    )
    .expect("valid GPT-2 pre-token regex")
});

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
            if line.is_empty() || line.starts_with("#version") {
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

    pub fn tokenize_word(&self, word: &str) -> Vec<String> {
        let pieces: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        self.apply_bpe_merges(&pieces)
    }

    pub fn encode_word(&self, word: &str) -> Result<Vec<usize>> {
        let word = encode_bytes_to_unicode(word);
        let tokens = self.tokenize_word(&word);
        let mut token_ids = Vec::new();

        for token in tokens {
            match self.token_to_id(&token) {
                Some(id) => token_ids.push(id),
                None => return Err(MiniInferError::InvalidInput),
            }
        }

        Ok(token_ids)
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

fn pre_tokenize_text(text: &str) -> Vec<String> {
    GPT2_PRE_TOKEN_PATTERN
        .find_iter(text)
        .map(|match_result| {
            match_result
                .expect("regex match should succeed")
                .as_str()
                .to_string()
        })
        .collect()
}

fn clean_decoded_tokens(tokens: &str) -> Result<String> {
    let mapping = unicode_to_bytes();
    let mut bytes: Vec<u8> = Vec::new();

    for character in tokens.chars() {
        match mapping.get(&character) {
            Some(&byte) => bytes.push(byte),
            None => return Err(MiniInferError::InvalidInput),
        }
    }

    String::from_utf8(bytes).map_err(|_| MiniInferError::InvalidInput)
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>> {
        let mut token_ids = Vec::new();

        for word in pre_tokenize_text(text) {
            let ids = self.encode_word(&word)?;
            token_ids.extend(ids);
        }

        Ok(token_ids)
    }

    fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut words = Vec::new();

        for &token_id in token_ids {
            let token = self.id_to_token(token_id);
            match token {
                Some(token) => words.push(token.to_string()),
                None => {
                    return Err(MiniInferError::IndexOutOfBounds { index: token_id, len: self.id_to_token.len() });
                }
            }
        }

        clean_decoded_tokens(&words.concat())
    }
}

fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bytes = Vec::new();

    bytes.extend(b'!'..=b'~');
    bytes.extend(0xA1u8..=0xACu8);
    bytes.extend(0xAEu8..=0xFFu8);
    let mut codepoints: Vec<u32> = bytes.iter().map(|&b| b as u32).collect();

    let mut next_codepoint = 256u32;
    for byte in 0u8..=u8::MAX {
        if !bytes.contains(&byte) {
            bytes.push(byte);
            codepoints.push(next_codepoint);
            next_codepoint += 1;
        }
    }

    let mut byte_to_char = HashMap::new();
    for (byte, codepoint) in bytes.iter().zip(codepoints.iter()) {
        byte_to_char.insert(*byte, std::char::from_u32(*codepoint).unwrap());
    }

    byte_to_char
}

fn unicode_to_bytes() -> HashMap<char, u8> {
    let byte_to_char = bytes_to_unicode();
    let mut char_to_byte = HashMap::new();

    for (byte, character) in byte_to_char {
        char_to_byte.insert(character, byte);
    }

    char_to_byte
}

fn encode_bytes_to_unicode(text: &str) -> String {
    let byte_to_char = bytes_to_unicode();
    let mut encoded = String::new();

    for byte in text.as_bytes() {
        if let Some(&character) = byte_to_char.get(byte) {
            encoded.push(character);
        } else {
            panic!("Byte {} not found in mapping", byte);
        }
    }

    encoded
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

    fn real_gpt2_tokenizer() -> Gpt2Tokenizer {
    let vocab_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizer/gpt2-real/vocab.json");
    let merges_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tokenizer/gpt2-real/merges.txt");

    let vocab_text = std::fs::read_to_string(vocab_path)
        .expect("real GPT-2 vocab fixture should load");
    let vocab = serde_json::from_str(&vocab_text)
        .expect("real GPT-2 vocab fixture should parse");
    let merges = Gpt2Tokenizer::load_merges_file(merges_path)
        .expect("real GPT-2 merges fixture should parse");

    Gpt2Tokenizer::from_vocab_and_merges(vocab, merges)
        .expect("real GPT-2 tokenizer fixtures should build tokenizer")
}

    fn tiny_space_vocab() -> HashMap<String, usize> {
        HashMap::from([
            ("hello".to_string(), 0),
            ("Ġworld".to_string(), 1),
        ])
    }

    fn hello_world_merges() -> HashMap<(String, String), usize> {
        HashMap::from([
            (("h".to_string(), "e".to_string()), 0),
            (("he".to_string(), "l".to_string()), 1),
            (("hel".to_string(), "l".to_string()), 2),
            (("hell".to_string(), "o".to_string()), 3),
            (("Ġ".to_string(), "w".to_string()), 4),
            (("Ġw".to_string(), "o".to_string()), 5),
            (("Ġwo".to_string(), "r".to_string()), 6),
            (("Ġwor".to_string(), "l".to_string()), 7),
            (("Ġworl".to_string(), "d".to_string()), 8),
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
    fn parse_merges_text_skips_empty_lines_and_version_header() {
        let merges = Gpt2Tokenizer::parse_merges_text(
            "#version: 0.2\n\nh e\n\nhe llo\n",
        )
        .expect("valid merges should parse");

        assert_eq!(merges.get(&("h".to_string(), "e".to_string())), Some(&0));
        assert_eq!(
            merges.get(&("he".to_string(), "llo".to_string())),
            Some(&1)
        );
    }

    #[test]
    fn parse_merges_text_keeps_hash_prefixed_merge_rules() {
        let merges = Gpt2Tokenizer::parse_merges_text(
            "#version: 0.2\n# #\n## ##\n",
        )
        .expect("valid hash-prefixed merges should parse");

        assert_eq!(merges.get(&("#".to_string(), "#".to_string())), Some(&0));
        assert_eq!(
            merges.get(&("##".to_string(), "##".to_string())),
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

#[test]
fn tokenize_word_applies_bpe_merges_to_word() {
    let merges = HashMap::from([
        (("h".to_string(), "e".to_string()), 0),
        (("he".to_string(), "l".to_string()), 1),
        (("hel".to_string(), "l".to_string()), 2),
        (("hell".to_string(), "o".to_string()), 3),
    ]);

    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_vocab(), merges)
        .expect("valid vocab and merges should build tokenizer");
    let hello = tokenizer.tokenize_word("hello");

    assert_eq!(hello, vec!["hello".to_string()]);

    let cat = tokenizer.tokenize_word("cat");
    assert_eq!(cat, vec!["c".to_string(), "a".to_string(), "t".to_string()]);
}

#[test]
fn encode_word_returns_token_ids_for_known_tokens() {
    let merges = HashMap::from([
        (("h".to_string(), "e".to_string()), 0),
        (("he".to_string(), "l".to_string()), 1),
        (("hel".to_string(), "l".to_string()), 2),
        (("hell".to_string(), "o".to_string()), 3),
    ]);

    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_vocab(), merges)
        .expect("valid vocab and merges should build tokenizer");
    let token_ids = tokenizer.encode_word("hello").expect("encoding should succeed");

    assert_eq!(token_ids, vec![0]);
}

#[test]
fn gpt2_tokenizer_encode_uses_bpe_word_tokens() {
    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_space_vocab(), hello_world_merges())
        .expect("valid vocab and merges should build tokenizer");

    let token_ids = tokenizer
        .encode("hello world")
        .expect("encoding should succeed");

    assert_eq!(token_ids, vec![0, 1]);
}

#[test]
fn gpt2_tokenizer_decode_joins_tokens_with_spaces() {
    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_space_vocab(), HashMap::new())
        .expect("valid vocab should build tokenizer");

    let text = tokenizer.decode(&[0, 1]).expect("decoding should succeed");

    assert_eq!(text, "hello world");
}

#[test]
fn clean_decoded_tokens_decodes_byte_unicode_text() {
    assert_eq!(
        clean_decoded_tokens("helloĠworld").expect("byte-unicode decode should succeed"),
        "hello world"
    );
    assert_eq!(
        clean_decoded_tokens("Ġhello").expect("byte-unicode decode should succeed"),
        " hello"
    );
    assert_eq!(
        clean_decoded_tokens("hello").expect("byte-unicode decode should succeed"),
        "hello"
    );

    let encoded = encode_bytes_to_unicode("é");
    assert_eq!(
        clean_decoded_tokens(&encoded).expect("byte-unicode decode should succeed"),
        "é"
    );
}

#[test]
fn clean_decoded_tokens_rejects_invalid_byte_unicode_text() {
    assert_eq!(
        clean_decoded_tokens("hello☃"),
        Err(MiniInferError::InvalidInput)
    );
    assert_eq!(clean_decoded_tokens("ÿ"), Err(MiniInferError::InvalidInput));
}

#[test]
fn gpt2_tokenizer_encode_rejects_unknown_bpe_piece() {
    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_vocab(), HashMap::new())
        .expect("valid vocab should build tokenizer");

    let err = tokenizer
        .encode("cat")
        .expect_err("unknown BPE piece should fail");

    assert_eq!(err, MiniInferError::InvalidInput);
}

#[test]
fn gpt2_tokenizer_decode_rejects_out_of_bounds_token_id() {
    let tokenizer = Gpt2Tokenizer::from_vocab_and_merges(tiny_vocab(), HashMap::new())
        .expect("valid vocab should build tokenizer");

    let err = tokenizer
        .decode(&[0, 2])
        .expect_err("out-of-bounds token ID should fail");

    assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 2, len: 2 });
}

#[test]
fn bytes_to_unicode_maps_all_bytes() {
    let mapping = bytes_to_unicode();

    assert_eq!(mapping.len(), 256);
}

#[test]
fn bytes_to_unicode_keeps_printable_ascii_bytes() {
    let mapping = bytes_to_unicode();

    assert_eq!(mapping.get(&b'!'), Some(&'!'));
    assert_eq!(mapping.get(&b'A'), Some(&'A'));
    assert_eq!(mapping.get(&b'~'), Some(&'~'));
}

#[test]
fn bytes_to_unicode_maps_space_and_newline_to_gpt2_markers() {
    let mapping = bytes_to_unicode();

    assert_eq!(mapping.get(&b' '), Some(&'Ġ'));
    assert_eq!(mapping.get(&b'\n'), Some(&'Ċ'));
}

#[test]
fn bytes_to_unicode_outputs_unique_chars() {
    let mapping = bytes_to_unicode();
    let mut seen = std::collections::HashSet::new();

    for character in mapping.values() {
        assert!(seen.insert(*character));
    }

    assert_eq!(seen.len(), 256);
}

#[test]
fn unicode_to_bytes_reverses_gpt2_markers() {
    let mapping = unicode_to_bytes();

    assert_eq!(mapping.get(&'Ġ'), Some(&b' '));
    assert_eq!(mapping.get(&'Ċ'), Some(&b'\n'));
    assert_eq!(mapping.get(&'A'), Some(&b'A'));
    assert_eq!(mapping.get(&'~'), Some(&b'~'));
}

#[test]
fn byte_unicode_maps_round_trip_all_bytes() {
    let byte_to_char = bytes_to_unicode();
    let char_to_byte = unicode_to_bytes();

    for byte in 0u8..=u8::MAX {
        let character = byte_to_char
            .get(&byte)
            .expect("every byte should map to a character");

        assert_eq!(char_to_byte.get(character), Some(&byte));
    }
}

#[test]
fn pre_tokenize_text_splits_categories_and_preserves_spaces() {
    assert_eq!(pre_tokenize_text("hello"), vec!["hello".to_string()]);
    assert_eq!(
        pre_tokenize_text("hello world"),
        vec!["hello".to_string(), " world".to_string()]
    );
    assert_eq!(pre_tokenize_text(" hello"), vec![" hello".to_string()]);
    assert_eq!(pre_tokenize_text(""), Vec::<String>::new());
    assert_eq!(
        pre_tokenize_text("hello, world!"),
        vec![
            "hello".to_string(),
            ",".to_string(),
            " world".to_string(),
            "!".to_string(),
        ]
    );
    assert_eq!(
        pre_tokenize_text("abc123"),
        vec!["abc".to_string(), "123".to_string()]
    );
    assert_eq!(
        pre_tokenize_text("hi  there"),
        vec!["hi".to_string(), " ".to_string(), " there".to_string()]
    );
}

#[test]
fn gpt2_tokenizer_matches_reference_ids_for_common_cases() {
    let tokenizer = real_gpt2_tokenizer();

    let cases = [
        ("The quick brown fox jumped.", vec![464, 2068, 7586, 21831, 11687, 13]),
        ("I'm here", vec![40, 1101, 994]),
        ("é", vec![2634]),
        ("hi  there", vec![5303, 220, 612]),
    ];

    for (text, expected_ids) in cases {
        assert_eq!(
            tokenizer.encode(text).expect("encoding should succeed"),
            expected_ids,
            "reference ID mismatch for {text:?}"
        );
    }
}

#[test]
fn gpt2_tokenizer_decodes_reference_ids_for_common_cases() {
    let tokenizer = real_gpt2_tokenizer();

    let cases = [
        (&[464, 2068, 7586, 21831, 11687, 13][..], "The quick brown fox jumped."),
        (&[40, 1101, 994][..], "I'm here"),
        (&[2634][..], "é"),
        (&[5303, 220, 612][..], "hi  there"),
    ];

    for (token_ids, expected_text) in cases {
        assert_eq!(
            tokenizer.decode(token_ids).expect("decoding should succeed"),
            expected_text,
            "reference decode mismatch for {token_ids:?}"
        );
    }
}
}