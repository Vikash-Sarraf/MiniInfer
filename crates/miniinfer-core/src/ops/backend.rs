use crate::{error::Result, ops::{matmul, softmax, helper}, tensor::Tensor};
use ndarray::ArrayView2;

pub trait OpsBackend {
    fn name(&self) -> &'static str;

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    fn softmax(&self, value: &[f32]) -> Result<Vec<f32>>;
}

pub struct ReferenceBackend;

impl ReferenceBackend {
    pub fn new() -> Self {
        ReferenceBackend
    }
}

impl OpsBackend for ReferenceBackend {
    fn name(&self) -> &'static str {
        "reference"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        matmul::matmul(a, b)
    }

    fn softmax(&self, value: &[f32]) -> Result<Vec<f32>> {
        softmax::softmax(value)
    }
}

pub struct NdArrayBackend;
impl NdArrayBackend {
    pub fn new() -> Self {
        NdArrayBackend
    }
}

impl OpsBackend for NdArrayBackend {
    fn name(&self) -> &'static str {
        "ndarray"
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
       let (m, k, n) = helper::validate_matmul_shape(a, b)?;

        let a_view = ArrayView2::from_shape((m, k), a.data()).expect("shape should be validated");
        let b_view = ArrayView2::from_shape((k, n), b.data()).expect("shape should be validated");

        let output = a_view.dot(&b_view);
        let data = output.iter().copied().collect();

        Tensor::new(vec![m, n], data)

    }

    fn softmax(&self, value: &[f32]) -> Result<Vec<f32>> { 
        softmax::softmax(value)    
    }
    
}

#[cfg(test)]
    mod tests {
    use super::*;

    #[test]
    fn reference_backend_works_through_trait() {
        let backend = ReferenceBackend::new();
        let backend: &dyn OpsBackend = &backend;

        assert_eq!(backend.name(), "reference");
    }

    #[test]
    fn reference_backend_name() {
        let backend = ReferenceBackend::new();
        assert_eq!(backend.name(), "reference");
    }

    #[test]
    fn reference_backend_matmul() {
        let backend = ReferenceBackend::new();

        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");

        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("valid tensor");

        let c = backend.matmul(&a, &b).expect("matmul should succeed");

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn reference_backend_softmax() {
        let backend = ReferenceBackend::new();

        let probs = backend.softmax(&[2.0, 1.0, 0.0]).expect("softmax should succeed");

        let sum: f32 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn ndarray_backend_works_through_trait() {
        let backend = NdArrayBackend::new();
        let backend: &dyn OpsBackend = &backend;

        assert_eq!(backend.name(), "ndarray");
    }

    #[test]
    fn ndarray_backend_name() {
        let backend = NdArrayBackend::new();
        assert_eq!(backend.name(), "ndarray");
    }

    #[test]
    fn ndarray_backend_matmul() {
        let backend = NdArrayBackend::new();

        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .expect("valid tensor");

        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
            .expect("valid tensor");

        let c = backend.matmul(&a, &b).expect("matmul should succeed");

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn ndarray_backend_softmax() {
        let backend = NdArrayBackend::new();

        let probs = backend.softmax(&[2.0, 1.0, 0.0]).expect("softmax should succeed");

        let sum: f32 = probs.iter().sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }
}