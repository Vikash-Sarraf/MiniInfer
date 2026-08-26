#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
}

impl DType {
    pub fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_is_four_bytes() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
    }
}