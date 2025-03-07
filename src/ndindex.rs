pub struct NdIndex {
    shape: Vec<usize>,
    current: Option<Vec<usize>>,
}

impl NdIndex {
    pub fn new<T: AsRef<[usize]>>(shape: T) -> Self {
        let shape_vec = shape.as_ref().to_vec();
        let current = if shape_vec.is_empty() || shape_vec.iter().any(|&dim| dim == 0) {
            None
        } else {
            Some(vec![0; shape_vec.len()])
        };

        NdIndex {
            shape: shape_vec,
            current,
        }
    }
}

impl Iterator for NdIndex {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.clone()?;

        if let Some(mut next) = self.current.clone() {
            let mut i = next.len() - 1;

            loop {
                next[i] += 1;

                if next[i] < self.shape[i] {
                    self.current = Some(next);
                    break;
                }

                next[i] = 0;
                if i == 0 {
                    self.current = None;
                    break;
                }
                i -= 1;
            }
        }

        Some(current)
    }
}

/// Create an NdIndex iterator for the given shape
pub fn ndindex<T: AsRef<[usize]>>(shape: T) -> NdIndex {
    NdIndex::new(shape)
}
