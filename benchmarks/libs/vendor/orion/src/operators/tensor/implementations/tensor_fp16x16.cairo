// EXTRACTED from orion @ bac0b42 (src/operators/tensor/implementations/tensor_fp16x16.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

use orion::operators::tensor::core::{new_tensor, Tensor, TensorTrait, at_tensor};
use orion::operators::tensor::{math, linalg};
use orion::numbers::{NumberTrait, FP16x16};

impl FP16x16Tensor of TensorTrait<FP16x16> {
    fn new(shape: Span<usize>, data: Span<FP16x16>) -> Tensor<FP16x16> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<FP16x16>, indices: Span<usize>) -> FP16x16 {
        *at_tensor(self, indices)
    }

    fn add(lhs: Tensor<FP16x16>, rhs: Tensor<FP16x16>) -> Tensor<FP16x16> {
        math::arithmetic::add(@lhs, @rhs)
    }

    fn matmul(self: @Tensor<FP16x16>, other: @Tensor<FP16x16>) -> Tensor<FP16x16> {
        linalg::matmul::matmul(self, other)
    }
}
