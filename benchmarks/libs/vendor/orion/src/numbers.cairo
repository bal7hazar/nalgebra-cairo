// EXTRACTED from orion @ bac0b42 (src/numbers.cairo): only the items needed by the benchmarks are kept;
// function bodies are verbatim unless marked PATCH.

mod fixed_point;

use orion::numbers::fixed_point::core::FixedTrait;
use orion::numbers::fixed_point::implementations::fp16x16::core::{FP16x16Impl, FP16x16};
use orion::numbers::fixed_point::implementations::fp32x32::core::{FP32x32Impl, FP32x32};
use orion::numbers::fixed_point::implementations::fp64x64::core::{FP64x64Impl, FP64x64};

// Upstream NumberTrait has ~70 methods implemented for every numeric type; only zero() is used by
// matmul / MutMatrix / NullableVec.
trait NumberTrait<T, MAG> {
    fn zero() -> T;
}

impl FP16x16Number of NumberTrait<FP16x16, u32> {
    fn zero() -> FP16x16 {
        FP16x16Impl::ZERO()
    }
}

impl FP32x32Number of NumberTrait<FP32x32, u64> {
    fn zero() -> FP32x32 {
        FP32x32Impl::ZERO()
    }
}
