// EXTRACTED from orion @ bac0b42 (src/operators/tensor/implementations/tensor_fp32x32.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

use orion::operators::tensor::core::{new_tensor, Tensor, TensorTrait, at_tensor};
use orion::operators::tensor::{math, linalg};
use orion::numbers::{NumberTrait, FP32x32};
use orion::numbers::fixed_point::implementations::fp32x32::core::FP32x32AddEq;

impl FP32x32Tensor of TensorTrait<FP32x32> {
    fn new(shape: Span<usize>, data: Span<FP32x32>) -> Tensor<FP32x32> {
        new_tensor(shape, data)
    }

    fn at(self: @Tensor<FP32x32>, indices: Span<usize>) -> FP32x32 {
        *at_tensor(self, indices)
    }

    fn add(lhs: Tensor<FP32x32>, rhs: Tensor<FP32x32>) -> Tensor<FP32x32> {
        math::arithmetic::add(@lhs, @rhs)
    }

    fn matmul(self: @Tensor<FP32x32>, other: @Tensor<FP32x32>) -> Tensor<FP32x32> {
        linalg::matmul::matmul(self, other)
    }
}
