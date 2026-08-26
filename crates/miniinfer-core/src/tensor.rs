use crate::error::{MiniInferError, Result};

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        // Validate shape and data length
        if shape.is_empty() {
            return Err(MiniInferError::EmptyShape);
        }
        if shape.iter().any(|dim| *dim == 0) {
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
}