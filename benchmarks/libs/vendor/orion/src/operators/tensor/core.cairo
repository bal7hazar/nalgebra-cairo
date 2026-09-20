// EXTRACTED from orion @ bac0b42 (src/operators/tensor/core.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

use orion::operators::tensor::helpers::{len_from_shape, check_shape};

#[derive(Copy, Drop)]
struct Tensor<T> {
    shape: Span<usize>,
    data: Span<T>,
}

// Upstream TensorTrait declares ~106 operators; only the ones benchmarked are kept.
trait TensorTrait<T> {
    fn new(shape: Span<usize>, data: Span<T>) -> Tensor<T>;
    fn at(self: @Tensor<T>, indices: Span<usize>) -> T;
    fn add(lhs: Tensor<T>, rhs: Tensor<T>) -> Tensor<T>;
    fn matmul(self: @Tensor<T>, other: @Tensor<T>) -> Tensor<T>;
}

fn new_tensor<T>(shape: Span<usize>, data: Span<T>) -> Tensor<T> {
    check_shape::<T>(shape, data);
    Tensor::<T> { shape, data }
}

fn ravel_index(mut shape: Span<usize>, mut indices: Span<usize>) -> usize {
    assert(shape.len() == indices.len(), 'shape & indices length unequal');

    let mut raveled_index: usize = 0;
    let mut stride: usize = 1;

    loop {
        match shape.pop_back() {
            Option::Some(i) => {
                let index = *indices.pop_back().unwrap();
                raveled_index += index * stride;

                stride *= *i;
            },
            Option::None => { break; }
        };
    };

    raveled_index
}

fn unravel_index(index: usize, mut shape: Span<usize>) -> Span<usize> {
    assert(shape.len() > 0, 'shape cannot be empty');

    let mut result = ArrayTrait::new();
    let mut remainder = index;
    let mut stride = len_from_shape(shape);

    loop {
        match shape.pop_front() {
            Option::Some(i) => {
                stride /= *i;

                let coord = remainder / stride;
                remainder = remainder % stride;

                result.append(coord);
            },
            Option::None => { break; }
        };
    };

    return result.span();
}

fn stride(mut shape: Span<usize>) -> Span<usize> {
    let mut strides = ArrayTrait::new();
    let mut stride = 1;
    loop {
        match shape.pop_back() {
            Option::Some(size) => {
                strides.append(stride);
                stride *= *size;
            },
            Option::None => { break; }
        };
    };

    // PATCH: was strides.reverse().span() (alexandria ArrayTraitExt)
    orion::utils::reverse(strides.span()).span()
}

fn at_tensor<T>(self: @Tensor<T>, indices: Span<usize>) -> @T {
    assert(indices.len() == (*self.shape).len(), 'indices not match dimensions');
    let data = *self.data;

    return data.at(ravel_index(*self.shape, indices));
}
