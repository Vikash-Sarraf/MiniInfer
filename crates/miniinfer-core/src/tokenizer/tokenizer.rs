use crate::error::{Result, MiniInferError};
pub trait Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>>;
    fn decode(&self, token_ids: &[usize]) -> Result<String>;
}

pub struct TinyTokenizer {
    vocab: Vec<String>,
}

impl Tokenizer for TinyTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<usize>> {
       let mut token_ids = Vec::new();

        for word in text.split_whitespace() {
            let token_id = self.vocab.iter().position(|token| token == word);

            match token_id {
                Some(index) => token_ids.push(index),
                None => return Err(MiniInferError::InvalidInput),
            } 
        }

       Ok(token_ids)
    }

    fn decode(&self, token_ids: &[usize]) -> Result<String> {
        let mut words = Vec::new();

        for &token_id in token_ids {
            let token = self.vocab.get(token_id);
            match token {
                Some(token) => words.push(token.clone()),
                None => {
                    return Err(MiniInferError::IndexOutOfBounds { index: token_id, len: self.vocab.len() });
                }
            }
        }
        Ok(words.join(" "))
    }
}

impl TinyTokenizer {
    pub fn new(vocab: Vec<String>) -> Self {
        Self { vocab }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = TinyTokenizer::new(vocab);

        let text = "hello world";
        let token_ids = tokenizer.encode(text).expect("Encoding failed");
        assert_eq!(token_ids, vec![0, 1]);

        let decoded_text = tokenizer.decode(&token_ids).expect("Decoding failed");
        assert_eq!(decoded_text, text);
    }

    #[test]
    fn encode_reject_unknown_token() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = TinyTokenizer::new(vocab);

        let text = "hello unknown";
        let err = tokenizer
            .encode(text)
            .expect_err("unknown token should fail");

        assert_eq!(err, MiniInferError::InvalidInput);
    }

    #[test]
    fn decode_reject_out_of_bounds() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = TinyTokenizer::new(vocab);

        let token_ids = vec![0, 2]; // 2 is out of bounds
        let err = tokenizer
            .decode(&token_ids)
            .expect_err("out-of-bounds token ID should fail");

        assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 2, len: 2 });
    }

    #[test]
    fn encode_empty_string() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = TinyTokenizer::new(vocab);

        let text = "";
        let token_ids = tokenizer.encode(text).expect("Encoding failed");
        assert_eq!(token_ids, Vec::<usize>::new());
    }

    #[test]
    fn decode_empty_token_ids() {
        let vocab = vec!["hello".to_string(), "world".to_string()];
        let tokenizer = TinyTokenizer::new(vocab);

        let token_ids = Vec::<usize>::new();
        let decoded_text = tokenizer.decode(&token_ids).expect("Decoding failed");
        assert_eq!(decoded_text, "");
    }

}