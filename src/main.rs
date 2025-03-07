use burn::prelude::*;
use burn::tensor::{Int, Numeric};
use burn::{backend::Wgpu, tensor::Tensor};
use one_hot::scatter::scatternd;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    let device = burn::backend::wgpu::WgpuDevice::default();

    let data1 = Tensor::<MyBackend, 1>::from_data(
        TensorData::from([1.0, 2.0, 3., 4., 5., 6., 7., 8.]),
        &device,
    );

    let indices1 =
        Tensor::<MyBackend, 2, Int>::from_data(TensorData::from([[4], [3], [1], [7]]), &device);
    let updates1 =
        Tensor::<MyBackend, 1>::from_data(TensorData::from([9., 10., 11., 12.]), &device);

    let _expected1 = Tensor::<MyBackend, 1>::from_data(
        TensorData::from([1., 11., 3., 10., 9., 6., 7., 12.]),
        &device,
    );

    //let output1 = scatter(data1, indices1, updates1);
    //println!("{output}");

    let data2 = Tensor::<MyBackend, 3>::from_data(
        TensorData::from([
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            [
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
            [
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
        ]),
        &device,
    );

    let indices2 = Tensor::<MyBackend, 2, Int>::from_data(TensorData::from([[0], [2]]), &device);

    let updates2 = Tensor::<MyBackend, 3>::from_data(
        TensorData::from([
            [
                [5.0, 5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0, 8.0],
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
        ]),
        &device,
    );

    let expected2 = Tensor::<MyBackend, 3>::from_data(
        TensorData::from([
            [
                [5.0, 5.0, 5.0, 5.0],
                [6.0, 6.0, 6.0, 6.0],
                [7.0, 7.0, 7.0, 7.0],
                [8.0, 8.0, 8.0, 8.0],
            ],
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
            ],
            [
                [1.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0],
                [3.0, 3.0, 3.0, 3.0],
                [4.0, 4.0, 4.0, 4.0],
            ],
            [
                [8.0, 7.0, 6.0, 5.0],
                [4.0, 3.0, 2.0, 1.0],
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 6.0, 7.0, 8.0],
            ],
        ]),
        &device,
    );

    let output2 = scatternd(data2, indices2, updates2);
}
