#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    I8Symmetric,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::I8Symmetric => 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_reports_size_in_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::I8Symmetric.size_in_bytes(), 1);
    }
}
