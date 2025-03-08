use crate::ndindex::ndindex;
use burn::prelude::*;
use burn::tensor::Tensor;
use burn::tensor::{Int, Numeric};

pub fn scatternd<B: Backend, const R: usize, const Q: usize, const S: usize, T>(
    data: Tensor<B, R, T>,
    indices: Tensor<B, Q, Int>,
    updates: Tensor<B, S, T>,
) -> Tensor<B, R, T>
where
    T: Numeric<B>,
{
    let k = indices.shape().dims[Q - 1];
    if R < 1 {
        panic!("data should be of rank >= 1!");
    } else if Q < 1 {
        panic!("indices should be of rank >= 1!");
    } else if S != R + Q - k - 1 {
        panic!("updates should be of rank r + q - indices.shape[-1] - 1");
    }

    let mut output = data.clone();
    let update_indices = &indices.shape().dims[0..Q - 1];
    let mut actual_indices: Vec<Tensor<B, Q, Int>> = Vec::new();
    for idx in ndindex(update_indices) {
        let indices_idx = indices
            .clone()
            .select(0, Tensor::from_data(&idx[..], &indices.device()));
        actual_indices.push(indices_idx);
    }
    let actual_indices = Tensor::cat(actual_indices, 0);

    for (idx, values) in actual_indices.iter_dim(0).zip(updates.iter_dim(0)) {
        let id = idx.reshape([1]).unsqueeze();
        let data_values = data.clone().select(0, id.clone());
        output = output.select_assign(0, id, values.unsqueeze() - data_values);
    }
    output
}
