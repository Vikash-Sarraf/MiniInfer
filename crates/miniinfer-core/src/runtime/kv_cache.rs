use crate::{
    error::{MiniInferError, Result},
    tensor::Tensor,
};

pub struct LayerKvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    num_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
}

pub struct KvCache {
    layers: Vec<LayerKvCache>,
}

impl LayerKvCache {
    pub fn new(num_heads: usize, head_dim: usize, max_seq_len: usize) -> Result<Self> {
        if num_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Invalid config".to_string() });
        }
        let mut keys = Vec::with_capacity(num_heads);
        let mut values = Vec::with_capacity(num_heads);
        for _ in 0..num_heads {
            keys.push(Vec::with_capacity(max_seq_len * head_dim));
            values.push(Vec::with_capacity(max_seq_len * head_dim));
        }
        let seq_len = 0;
        Ok(Self {
            keys,
            values,
            num_heads,
            head_dim,
            max_seq_len,
            seq_len,
        })
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn append(&mut self, new_keys: &[Tensor], new_values: &[Tensor]) -> Result<()> {
        if new_keys.len() != self.num_heads {
            return Err(MiniInferError::LengthMismatch { expected: self.num_heads, actual: new_keys.len() });
        }
        if new_values.len() != self.num_heads {
            return Err(MiniInferError::LengthMismatch { expected: self.num_heads, actual: new_values.len() });
        }

        if self.seq_len + 1 > self.max_seq_len {
            return Err(MiniInferError::InvalidConfig { message: "Exceeding max sequence length".to_string() });
        }

        for head in 0..self.num_heads {
            validate_head_row(&new_keys[head], self.head_dim)?;
            validate_head_row(&new_values[head], self.head_dim)?;

            self.keys[head].extend_from_slice(new_keys[head].data());
            self.values[head].extend_from_slice(new_values[head].data());
        }

        self.seq_len += 1;
        Ok(())
    }

    pub fn key_for_head(&self, head: usize) -> Result<Tensor> {
        if head >= self.num_heads {
            return Err(MiniInferError::IndexOutOfBounds { index: head, len: self.num_heads });
        }

        if self.seq_len == 0 {
            return Err(MiniInferError::EmptyInput);
        }

        Tensor::new(
            vec![self.seq_len, self.head_dim],
            self.keys[head].clone(),
        )
    }

    pub fn value_for_head(&self, head: usize) -> Result<Tensor> {
        if head >= self.num_heads {
            return Err(MiniInferError::IndexOutOfBounds { index: head, len: self.num_heads });
        }

        if self.seq_len == 0 {
            return Err(MiniInferError::EmptyInput);
        }

        Tensor::new(
            vec![self.seq_len, self.head_dim],
            self.values[head].clone(),
        )
    }

    pub fn reset(&mut self) {
        for head in 0..self.num_heads {
            self.keys[head].clear();
            self.values[head].clear();
        }
        self.seq_len = 0;
    }
}

fn validate_head_row(tensor: &Tensor, head_dim: usize) -> Result<()> {
    if tensor.shape() != [1, head_dim] {
        return Err(MiniInferError::InvalidTensorShape {
            expected: vec![1, head_dim],
            actual: tensor.shape().to_vec(),
        });
    }

    Ok(())
}

impl KvCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, max_seq_len: usize) -> Result<Self> {
        if num_layers == 0 {
            return Err(MiniInferError::InvalidConfig { message: "Number of layers must be greater than 0".to_string() });
        }
        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(LayerKvCache::new(num_heads, head_dim, max_seq_len)?);
        }
        Ok(Self { layers })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    pub fn seq_len(&self) -> Result<usize> {
        let first_layer = self.layer(0)?;
        let seq_len = first_layer.seq_len();

        for layer in &self.layers[1..] {
            if layer.seq_len() != seq_len {
                return Err(MiniInferError::InvalidConfig {
                    message: "KV cache layers must have the same sequence length".to_string(),
                });
            }
        }

        Ok(seq_len)
    }

    pub fn current_position(&self) -> Result<usize> {
        self.seq_len()
    }

    pub fn layer(&self, layer_index: usize) -> Result<&LayerKvCache> {
        self.layers
            .get(layer_index)
            .ok_or(MiniInferError::IndexOutOfBounds {
                index: layer_index,
                len: self.layers.len(),
            })
    }

    pub fn layer_mut(&mut self, layer_index: usize) -> Result<&mut LayerKvCache> {
        let len = self.layers.len();
        self.layers
            .get_mut(layer_index)
            .ok_or(MiniInferError::IndexOutOfBounds {
                index: layer_index,
                len,
            })
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(values: &[f32]) -> Tensor {
        Tensor::new(vec![1, values.len()], values.to_vec()).expect("row tensor should be valid")
    }

    fn expect_empty_input(result: Result<Tensor>) {
        match result {
            Err(MiniInferError::EmptyInput) => {}
            Err(err) => panic!("unexpected error: {err:?}"),
            Ok(_) => panic!("expected empty input error"),
        }
    }

    #[test]
    fn layer_cache_starts_empty() {
        let cache = LayerKvCache::new(2, 4, 8).expect("cache should be valid");

        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.num_heads(), 2);
        assert_eq!(cache.head_dim(), 4);
        assert_eq!(cache.max_seq_len(), 8);
        assert_eq!(cache.keys.len(), 2);
        assert_eq!(cache.values.len(), 2);
        assert_eq!(cache.keys[0].capacity(), 32);
        assert_eq!(cache.values[0].capacity(), 32);
    }

    #[test]
    fn layer_cache_rejects_zero_dimensions() {
        assert!(LayerKvCache::new(0, 4, 8).is_err());
        assert!(LayerKvCache::new(2, 0, 8).is_err());
        assert!(LayerKvCache::new(2, 4, 0).is_err());
    }

    #[test]
    fn kv_cache_creates_one_layer_per_config_layer() {
        let cache = KvCache::new(3, 2, 4, 8).expect("cache should be valid");

        assert_eq!(cache.num_layers(), 3);
        assert_eq!(cache.layer(0).expect("layer should exist").num_heads(), 2);
        assert_eq!(cache.layer(2).expect("layer should exist").head_dim(), 4);
    }

    #[test]
    fn kv_cache_rejects_invalid_layer_index() {
        let cache = KvCache::new(2, 2, 4, 8).expect("cache should be valid");

        let err = match cache.layer(2) {
            Ok(_) => panic!("out-of-bounds layer should fail"),
            Err(err) => err,
        };

        assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 2, len: 2 });
    }

    #[test]
    fn kv_cache_rejects_zero_layers() {
        assert!(KvCache::new(0, 2, 4, 8).is_err());
    }

    #[test]
    fn layer_cache_appends_one_token_per_head() {
        let mut cache = LayerKvCache::new(2, 3, 8).expect("cache should be valid");

        let keys = vec![
            Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).expect("valid key"),
            Tensor::new(vec![1, 3], vec![4.0, 5.0, 6.0]).expect("valid key"),
        ];

        let values = vec![
            Tensor::new(vec![1, 3], vec![7.0, 8.0, 9.0]).expect("valid value"),
            Tensor::new(vec![1, 3], vec![10.0, 11.0, 12.0]).expect("valid value"),
        ];

        cache.append(&keys, &values).expect("append should succeed");

        assert_eq!(cache.seq_len(), 1);
        assert_eq!(cache.key_for_head(0).expect("key should exist").data(), &[1.0, 2.0, 3.0]);
        assert_eq!(cache.value_for_head(1).expect("value should exist").data(), &[10.0, 11.0, 12.0]);
    }

    #[test]
    fn layer_cache_appends_multiple_tokens_in_order() {
        let mut cache = LayerKvCache::new(1, 2, 4).expect("cache should be valid");

        cache
            .append(&[row(&[1.0, 2.0])], &[row(&[10.0, 20.0])])
            .expect("first append should succeed");
        cache
            .append(&[row(&[3.0, 4.0])], &[row(&[30.0, 40.0])])
            .expect("second append should succeed");

        let key = cache.key_for_head(0).expect("key should exist");
        let value = cache.value_for_head(0).expect("value should exist");

        assert_eq!(cache.seq_len(), 2);
        assert_eq!(key.shape(), &[2, 2]);
        assert_eq!(key.data(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(value.shape(), &[2, 2]);
        assert_eq!(value.data(), &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn layer_cache_rejects_wrong_key_head_count() {
        let mut cache = LayerKvCache::new(2, 2, 4).expect("cache should be valid");

        let err = cache
            .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0]), row(&[5.0, 6.0])])
            .expect_err("wrong key head count should fail");

        assert_eq!(err, MiniInferError::LengthMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn layer_cache_rejects_wrong_value_head_count() {
        let mut cache = LayerKvCache::new(2, 2, 4).expect("cache should be valid");

        let err = cache
            .append(&[row(&[1.0, 2.0]), row(&[3.0, 4.0])], &[row(&[5.0, 6.0])])
            .expect_err("wrong value head count should fail");

        assert_eq!(err, MiniInferError::LengthMismatch { expected: 2, actual: 1 });
    }

    #[test]
    fn layer_cache_rejects_wrong_key_shape() {
        let mut cache = LayerKvCache::new(1, 2, 4).expect("cache should be valid");
        let bad_key = Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]).expect("valid tensor");

        let err = cache
            .append(&[bad_key], &[row(&[4.0, 5.0])])
            .expect_err("wrong key shape should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidTensorShape {
                expected: vec![1, 2],
                actual: vec![1, 3],
            }
        );
    }

    #[test]
    fn layer_cache_rejects_context_overflow() {
        let mut cache = LayerKvCache::new(1, 2, 1).expect("cache should be valid");

        cache
            .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0])])
            .expect("first append should succeed");
        let err = cache
            .append(&[row(&[5.0, 6.0])], &[row(&[7.0, 8.0])])
            .expect_err("second append should overflow");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "Exceeding max sequence length".to_string(),
            }
        );
    }

    #[test]
    fn layer_cache_reset_clears_rows() {
        let mut cache = LayerKvCache::new(1, 2, 4).expect("cache should be valid");

        cache
            .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0])])
            .expect("append should succeed");
        cache.reset();

        assert_eq!(cache.seq_len(), 0);
        assert!(cache.keys[0].is_empty());
        assert!(cache.values[0].is_empty());
        expect_empty_input(cache.key_for_head(0));
        expect_empty_input(cache.value_for_head(0));
    }

    #[test]
    fn layer_cache_empty_readback_fails() {
        let cache = LayerKvCache::new(1, 2, 4).expect("cache should be valid");

        expect_empty_input(cache.key_for_head(0));
        expect_empty_input(cache.value_for_head(0));
    }

    #[test]
    fn kv_cache_layer_mut_allows_layer_update() {
        let mut cache = KvCache::new(2, 1, 2, 4).expect("cache should be valid");

        cache
            .layer_mut(0)
            .expect("layer should exist")
            .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0])])
            .expect("append should succeed");

        let key = cache.layer(0).expect("layer should exist").key_for_head(0).expect("key should exist");
        assert_eq!(key.data(), &[1.0, 2.0]);
    }

    #[test]
    fn kv_cache_layer_mut_rejects_invalid_layer_index() {
        let mut cache = KvCache::new(1, 1, 2, 4).expect("cache should be valid");

        let err = match cache.layer_mut(1) {
            Ok(_) => panic!("out-of-bounds layer should fail"),
            Err(err) => err,
        };
        assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 1, len: 1 });
    }

    #[test]
    fn kv_cache_reset_clears_all_layers() {
        let mut cache = KvCache::new(2, 1, 2, 4).expect("cache should be valid");

        cache
            .layer_mut(0)
            .expect("layer should exist")
            .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0])])
            .expect("append should succeed");
        cache
            .layer_mut(1)
            .expect("layer should exist")
            .append(&[row(&[5.0, 6.0])], &[row(&[7.0, 8.0])])
            .expect("append should succeed");

        cache.reset();

        for layer_index in 0..cache.num_layers() {
            let layer = cache.layer(layer_index).expect("layer should exist");
            assert_eq!(layer.seq_len(), 0);
            assert!(layer.keys[0].is_empty());
            assert!(layer.values[0].is_empty());
        }
    }

    #[test]
    fn kv_cache_current_position_tracks_shared_layer_sequence_length() {
        let mut cache = KvCache::new(2, 1, 2, 4).expect("cache should be valid");

        assert_eq!(cache.current_position().expect("position should exist"), 0);

        for layer_index in 0..cache.num_layers() {
            cache
                .layer_mut(layer_index)
                .expect("layer should exist")
                .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0])])
                .expect("append should succeed");
        }

        assert_eq!(cache.seq_len().expect("seq len should exist"), 1);
        assert_eq!(cache.current_position().expect("position should exist"), 1);
    }

    #[test]
    fn kv_cache_seq_len_rejects_inconsistent_layer_lengths() {
        let mut cache = KvCache::new(2, 1, 2, 4).expect("cache should be valid");

        cache
            .layer_mut(0)
            .expect("layer should exist")
            .append(&[row(&[1.0, 2.0])], &[row(&[3.0, 4.0])])
            .expect("append should succeed");

        let err = cache.seq_len().expect_err("inconsistent cache should fail");

        assert_eq!(
            err,
            MiniInferError::InvalidConfig {
                message: "KV cache layers must have the same sequence length".to_string(),
            }
        );
    }
}

