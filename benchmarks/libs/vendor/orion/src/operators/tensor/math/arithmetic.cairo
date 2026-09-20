// EXTRACTED from orion @ bac0b42 (src/operators/tensor/math/arithmetic.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

use orion::operators::tensor::core::{Tensor, TensorTrait, unravel_index,};
use orion::operators::tensor::helpers::{broadcast_shape, broadcast_index_mapping, len_from_shape,};

fn add<
    T, impl TTensor: TensorTrait<T>, impl TAdd: Add<T>, impl TCopy: Copy<T>, impl TDrop: Drop<T>
>(
    self: @Tensor<T>, other: @Tensor<T>
) -> Tensor<T> {
    let broadcasted_shape = broadcast_shape(*self.shape, *other.shape);
    let mut result = array![];

    let num_elements = len_from_shape(broadcasted_shape);

    let mut n: usize = 0;
    while n != num_elements {
        let indices_broadcasted = unravel_index(n, broadcasted_shape);

        let indices_self = broadcast_index_mapping(*self.shape, indices_broadcasted);
        let indices_other = broadcast_index_mapping(*other.shape, indices_broadcasted);

        result.append(*(*self.data)[indices_self] + *(*other.data)[indices_other]);

        n += 1;
    };

    TensorTrait::<T>::new(broadcasted_shape, result.span())
}
