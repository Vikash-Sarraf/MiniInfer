use crate::error::{MiniInferError, Result};

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    shape: Vec<usize>,
    data: Vec<i8>,
    scale: f32,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        if shape.is_empty() {
            return Err(MiniInferError::EmptyShape);
        }
        if shape.contains(&0) {
            return Err(MiniInferError::ZeroDimension);
        }

        let expected = shape.iter().product::<usize>();
        let actual = data.len();

        if expected != actual {
            return Err(MiniInferError::ShapeDataLengthMismatch { expected, actual });
        }

        Ok(Self { shape, data })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn get_1d(&self, index: usize) -> Result<f32> {
        let rank = self.shape.len();

        if rank != 1 {
            return Err(MiniInferError::WrongRank { expected: 1, actual: rank });
        }
        let len = self.data.len();
        if index >= len {
            return Err(MiniInferError::IndexOutOfBounds { index, len });
        }

        Ok(self.data[index])
    }

    pub fn get_2d(&self, row: usize, col: usize) -> Result<f32> {
        let rank = self.shape.len();

        if rank != 2 {
            return Err(MiniInferError::WrongRank { expected: 2, actual: rank });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];
        if row >= rows {
            return Err(MiniInferError::IndexOutOfBounds { index: row, len: rows });
        }

        if col >= cols {
            return Err(MiniInferError::IndexOutOfBounds { index: col, len: cols });
        }

        let index = row * cols + col;
        Ok(self.data[index])
    }
}

impl QuantizedTensor {
    pub fn new(shape: Vec<usize>, data: Vec<i8>, scale: f32) -> Result<Self> {
        if shape.is_empty() {
            return Err(MiniInferError::EmptyShape);
        }
        if shape.contains(&0) {
            return Err(MiniInferError::ZeroDimension);
        }

        let expected = shape.iter().product::<usize>();
        let actual = data.len();

        if expected != actual {
            return Err(MiniInferError::ShapeDataLengthMismatch { expected, actual });
        }

        if !scale.is_finite() || scale <= 0.0 {
            return Err(MiniInferError::InvalidConfig {
                message: "quantization scale must be finite and greater than zero".to_string(),
            });
        }

        Ok(Self { shape, data, scale })
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &[i8] {
        &self.data
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    pub fn dequantize(&self) -> Result<Tensor> {
        let data = self.data.iter().map(|&x| x as f32 * self.scale).collect::<Vec<f32>>();
        Tensor::new(self.shape.clone(), data)
    }

    pub fn quantize_symmetric(tensor: &Tensor) -> Result<Self> {
        let max_abs = tensor.data().iter().copied().map(f32::abs).fold(0.0, f32::max);

        if max_abs == 0.0 {
            let scale = 1.0;
            let data = vec![0i8; tensor.data().len()];
            return Self::new(tensor.shape().to_vec(), data, scale);
        }
        let scale = max_abs / 127.0;

        let data = tensor.data().iter().map(|value| {
            let quantized = (value / scale).round().clamp(-127.0, 127.0);
            quantized as i8
        }).collect::<Vec<i8>>();

        Self::new(tensor.shape().to_vec(), data, scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_dimension() {
        let err = Tensor::new(vec![2, 0], vec![]).expect_err("zero dimension should fail");
        assert_eq!(err, MiniInferError::ZeroDimension);
    }

    #[test]
    fn rejects_shape_data_length_mismatch() {
        let err = Tensor::new(vec![2, 3], vec![1.0, 2.0]).expect_err("shape-data length mismatch should fail");
        assert_eq!(err, MiniInferError::ShapeDataLengthMismatch { expected: 6, actual: 2 });
    }

    #[test]
    fn creates_valid_tensor() {
        let tensor = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("valid tensor should be created");

        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn rejects_empty_shape() {
        let err = Tensor::new(vec![], vec![]).expect_err("empty shape should fail");

        assert_eq!(err, MiniInferError::EmptyShape);
    }

    #[test]
    fn gets_1d_value() {
        let tensor = Tensor::new(vec![4], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");

        assert_eq!(tensor.get_1d(2).expect("index should exist"), 3.0);
    }

    #[test]
    fn rejects_1d_out_of_bounds(){
        let tensor = Tensor::new(vec![4], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");

        let err = tensor.get_1d(4).expect_err("index 4 should be out of bounds");

        assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 4, len: 4 })
    }

    #[test]
    fn reject_1d_on_non_1d_tensor() {
        let tensor = Tensor::new(vec![2,2], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");
        
        let err = tensor.get_1d(1).expect_err("2D tensor should be rejected");

        assert_eq!(err, MiniInferError::WrongRank { expected: 1, actual: 2 });
    }

        #[test]
    fn gets_2d_value() {
        let tensor = Tensor::new(vec![2,2], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");

        assert_eq!(tensor.get_2d(1,0).expect("index should exist"), 3.0);
    }

    #[test]
    fn rejects_2d_row_out_of_bounds(){
        let tensor = Tensor::new(vec![2,2], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");

        let err = tensor.get_2d(4,1).expect_err("row index 4 should be out of bounds");

        assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 4, len: 2 })
    } 

        #[test]
    fn rejects_2d_col_out_of_bounds(){
        let tensor = Tensor::new(vec![2,2], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");

        let err = tensor.get_2d(1,5).expect_err("col index 4 should be out of bounds");

        assert_eq!(err, MiniInferError::IndexOutOfBounds { index: 5, len: 2 })
    }  

        #[test]
    fn reject_2d_on_non_2d_tensor() {
        let tensor = Tensor::new(vec![4], vec![1.0,2.0,3.0,4.0]).expect("valid tensor created");
        
        let err = tensor.get_2d(1,1).expect_err("1D tensor should be rejected");

        assert_eq!(err, MiniInferError::WrongRank { expected: 2, actual: 1 });
    }

    #[test]
    fn creates_valid_quantized_tensor() {
        let tensor = QuantizedTensor::new(vec![2, 2], vec![-2, -1, 0, 1], 0.25)
            .expect("valid quantized tensor should be created");

        assert_eq!(tensor.shape(), &[2, 2]);
        assert_eq!(tensor.data(), &[-2, -1, 0, 1]);
        assert_eq!(tensor.scale(), 0.25);
    }

    #[test]
    fn quantized_tensor_rejects_empty_shape() {
        let err = QuantizedTensor::new(vec![], vec![], 1.0)
            .expect_err("empty quantized tensor shape should fail");

        assert_eq!(err, MiniInferError::EmptyShape);
    }

    #[test]
    fn quantized_tensor_rejects_zero_dimension() {
        let err = QuantizedTensor::new(vec![2, 0], vec![], 1.0)
            .expect_err("zero quantized tensor dimension should fail");

        assert_eq!(err, MiniInferError::ZeroDimension);
    }

    #[test]
    fn quantized_tensor_rejects_shape_data_length_mismatch() {
        let err = QuantizedTensor::new(vec![2, 3], vec![1, 2], 1.0)
            .expect_err("quantized tensor shape-data length mismatch should fail");

        assert_eq!(err, MiniInferError::ShapeDataLengthMismatch { expected: 6, actual: 2 });
    }

    #[test]
    fn quantized_tensor_rejects_zero_scale() {
        let err = QuantizedTensor::new(vec![2], vec![1, 2], 0.0)
            .expect_err("zero quantization scale should fail");

        assert_eq!(err, invalid_quantization_scale_error());
    }

    #[test]
    fn quantized_tensor_rejects_nan_scale() {
        let err = QuantizedTensor::new(vec![2], vec![1, 2], f32::NAN)
            .expect_err("NaN quantization scale should fail");

        assert_eq!(err, invalid_quantization_scale_error());
    }

    #[test]
    fn dequantizes_i8_values_to_f32_tensor() {
        let quantized = QuantizedTensor::new(
            vec![3],
            vec![-64, 32, 127],
            0.007874,
        )
        .unwrap();

        let dequantized = quantized.dequantize().unwrap();

        assert_eq!(dequantized.shape(), &[3]);
        assert_close(dequantized.data()[0], -0.503936);
        assert_close(dequantized.data()[1], 0.251968);
        assert_close(dequantized.data()[2], 0.999998);
    }

    fn invalid_quantization_scale_error() -> MiniInferError {
        MiniInferError::InvalidConfig {
            message: "quantization scale must be finite and greater than zero".to_string(),
        }
    }

    fn assert_close(actual: f32, expected: f32) {
        assert!((actual - expected).abs() < 1e-5, "actual {actual} expected {expected}");
    }

    #[test]
    fn quantizes_zero_tensor_with_scale_one() {
        let tensor = Tensor::new(vec![3], vec![0.0, 0.0, 0.0]).unwrap();

        let quantized = QuantizedTensor::quantize_symmetric(&tensor).unwrap();

        assert_eq!(quantized.shape(), &[3]);
        assert_eq!(quantized.data(), &[0, 0, 0]);
        assert_eq!(quantized.scale(), 1.0);
    }

    #[test]
    fn quantizes_symmetric_i8_values() {
        let tensor = Tensor::new(vec![3], vec![-1.0, 0.0, 1.0]).unwrap();

        let quantized = QuantizedTensor::quantize_symmetric(&tensor).unwrap();

        assert_eq!(quantized.shape(), &[3]);
        assert_close(quantized.scale(), 1.0 / 127.0);
        assert_eq!(quantized.data(), &[-127, 0, 127]);
    }

    #[test]
    fn quantize_then_dequantize_approximately_reconstructs_values() {
        let tensor = Tensor::new(vec![3], vec![-0.5, 0.25, 1.0]).unwrap();

        let quantized = QuantizedTensor::quantize_symmetric(&tensor).unwrap();
        let dequantized = quantized.dequantize().unwrap();

        assert_eq!(dequantized.shape(), tensor.shape());

        for (actual, expected) in dequantized.data().iter().zip(tensor.data()) {
            assert!(
                (actual - expected).abs() <= quantized.scale(),
                "actual {actual} expected {expected} scale {}",
                quantized.scale()
            );
        }
    }
}
