//! Question 5 — lazy rescaling. A Q32.32 product is `(a * b) / 2^32`; in a dot product / matrix
//! row the division (and the narrowing back to i64) can be done once on the accumulated wide
//! sum instead of once per product.
//!
//! Variants:
//! * eager          : `Fixed * Fixed` per term (i128 mul + i128 div + narrowing, each time)
//! * lazy_i128      : i128 products accumulated in i128, one signed division per row
//! * lazy_felt      : products and sum done in felt252 (1 step each, cannot overflow: |sum| <
//!                    n * 2^126), then ONE felt252 -> i128 range check + one signed division
//! * lazy_felt_floor: same, but the rescale is an unsigned `u128` div on a biased value
//!                    (floor rounding instead of truncation toward zero)
//! * eager_felt     : per-product rescale, but the product itself goes through felt252 (isolates
//!                    "felt product" gain from "single rescale" gain)
//! * eager_wide_mul / lazy_wide_mul : same as eager / lazy_i128 but the widening product is the
//!                    dedicated `i64_wide_mul` libfunc instead of the generic `i128 * i128`
//! * eager_bounded_floor / lazy_bounded_floor : `BoundedInt` kernels found by the `primitives`
//!                    benchmark package (range-check-free products and sums, one biased unsigned
//!                    `div_rem`, one narrowing; floor rounding) — the cheapest known scalar
//!                    kernel

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, downcast,
};
use core::num::traits::WideMul;
use crate::fixed::{Fixed, ONE_WIDE};
use crate::mat3::{DMatSeq, DMatSeqTrait, Mat3F};
use crate::vec3::Vec3;

pub type V3 = Vec3<Fixed>;
pub type M3 = Mat3F<Fixed>;

const BIAS: felt252 = 0x80000000000000000000000000000000; // 2^127
const BIAS_RESCALED: felt252 = 0x800000000000000000000000; // 2^95
const ONE_U128: u128 = 0x100000000;

/// i128 -> Fixed : one signed division + narrowing.
#[inline(always)]
fn rescale_i128(wide: i128) -> Fixed {
    Fixed { raw: (wide / ONE_WIDE).try_into().expect('fixed overflow') }
}

/// felt252 (signed value at 2^64 scale) -> Fixed, truncating toward zero.
#[inline(always)]
fn rescale_felt(wide: felt252) -> Fixed {
    let wide: i128 = wide.try_into().expect('fixed overflow');
    Fixed { raw: (wide / ONE_WIDE).try_into().expect('fixed overflow') }
}

/// felt252 (signed value at 2^64 scale) -> Fixed, flooring, through an unsigned division.
#[inline(always)]
fn rescale_felt_floor(wide: felt252) -> Fixed {
    let biased: u128 = (wide + BIAS).try_into().expect('fixed overflow');
    let quotient: felt252 = (biased / ONE_U128).into();
    Fixed { raw: (quotient - BIAS_RESCALED).try_into().expect('fixed overflow') }
}

#[inline(always)]
fn wide(a: Fixed, b: Fixed) -> i128 {
    a.raw.into() * b.raw.into()
}

#[inline(always)]
fn felt(a: Fixed, b: Fixed) -> felt252 {
    a.raw.into() * b.raw.into()
}

/// Single product through felt252 (no accumulation).
#[inline(always)]
fn mul_felt(a: Fixed, b: Fixed) -> Fixed {
    rescale_felt(felt(a, b))
}

// ---- cheap scalar kernels (from benchmarks/primitives/src/bounded.cairo) -----------------------

type I64Sq = BoundedInt<-0x3fffffffffffffff8000000000000000, 0x40000000000000000000000000000000>;
type I64Sq2 = BoundedInt<-0x7fffffffffffffff0000000000000000, 0x80000000000000000000000000000000>;
type I64Sq3 = BoundedInt<-0xbffffffffffffffe8000000000000000, 0xc0000000000000000000000000000000>;
type Two32 = UnitInt<0x100000000>;
type U32Rem = BoundedInt<0, 0xffffffff>;
const NZ_TWO32: NonZero<Two32> = 0x100000000;

type Offset = UnitInt<0x40000000000000000000000000000000>;
type I64SqOffset = BoundedInt<0x8000000000000000, 0x80000000000000000000000000000000>;
type I64SqOffsetShr32 = BoundedInt<0x80000000, 0x800000000000000000000000>;
type OffsetShr32 = UnitInt<0x400000000000000000000000>;

type Offset3 = UnitInt<0xc0000000000000000000000000000000>;
type I64Sq3Offset = BoundedInt<0x18000000000000000, 0x180000000000000000000000000000000>;
type I64Sq3OffsetShr32 = BoundedInt<0x180000000, 0x1800000000000000000000000>;
type Offset3Shr32 = UnitInt<0xc00000000000000000000000>;

impl MulI64I64 of MulHelper<i64, i64> {
    type Result = I64Sq;
}
impl AddOffset of AddHelper<I64Sq, Offset> {
    type Result = I64SqOffset;
}
impl DivRemI64SqOffset of DivRemHelper<I64SqOffset, Two32> {
    type DivT = I64SqOffsetShr32;
    type RemT = U32Rem;
}
impl SubOffset of SubHelper<I64SqOffsetShr32, OffsetShr32> {
    type Result = BoundedInt<-0x3fffffffffffffff80000000, 0x400000000000000000000000>;
}
impl AddISqISq of AddHelper<I64Sq, I64Sq> {
    type Result = I64Sq2;
}
impl AddISq2ISq of AddHelper<I64Sq2, I64Sq> {
    type Result = I64Sq3;
}
impl AddOffset3 of AddHelper<I64Sq3, Offset3> {
    type Result = I64Sq3Offset;
}
impl DivRemI64Sq3Offset of DivRemHelper<I64Sq3Offset, Two32> {
    type DivT = I64Sq3OffsetShr32;
    type RemT = U32Rem;
}
impl SubOffset3 of SubHelper<I64Sq3OffsetShr32, Offset3Shr32> {
    type Result = BoundedInt<-0xbffffffffffffffe80000000, 0xc00000000000000000000000>;
}

/// Eager product, dedicated widening libfunc + signed i128 division (truncation).
#[inline(always)]
fn mul_wide(a: Fixed, b: Fixed) -> Fixed {
    Fixed { raw: (a.raw.wide_mul(b.raw) / ONE_WIDE).try_into().expect('fixed overflow') }
}

/// Eager product, BoundedInt biased floor.
#[inline(always)]
fn mul_bounded(a: Fixed, b: Fixed) -> Fixed {
    let p: I64Sq = bounded_int::mul(a.raw, b.raw);
    let biased: I64SqOffset = bounded_int::add::<_, Offset>(p, 0x40000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem(biased, NZ_TWO32);
    let q = bounded_int::sub::<_, OffsetShr32>(q, 0x400000000000000000000000);
    Fixed { raw: downcast(q).expect('fixed overflow') }
}

/// Lazy 3-term row, `i64_wide_mul` products, checked i128 sums, one signed division.
#[inline(always)]
fn row_wide(a1: Fixed, a2: Fixed, a3: Fixed, b1: Fixed, b2: Fixed, b3: Fixed) -> Fixed {
    rescale_i128(a1.raw.wide_mul(b1.raw) + a2.raw.wide_mul(b2.raw) + a3.raw.wide_mul(b3.raw))
}

/// Lazy 3-term row, BoundedInt: no range check until the single biased div_rem + narrowing.
#[inline(always)]
fn row_bounded(a1: Fixed, a2: Fixed, a3: Fixed, b1: Fixed, b2: Fixed, b3: Fixed) -> Fixed {
    let p1: I64Sq = bounded_int::mul(a1.raw, b1.raw);
    let p2: I64Sq = bounded_int::mul(a2.raw, b2.raw);
    let p3: I64Sq = bounded_int::mul(a3.raw, b3.raw);
    let sum: I64Sq3 = bounded_int::add(bounded_int::add(p1, p2), p3);
    let biased: I64Sq3Offset = bounded_int::add::<
        _, Offset3,
    >(sum, 0xc0000000000000000000000000000000);
    let (q, _r) = bounded_int::div_rem(biased, NZ_TWO32);
    let q = bounded_int::sub::<_, Offset3Shr32>(q, 0xc00000000000000000000000);
    Fixed { raw: downcast(q).expect('fixed overflow') }
}

pub fn dot_eager_wide_mul(a: V3, b: V3) -> Fixed {
    mul_wide(a.x, b.x) + mul_wide(a.y, b.y) + mul_wide(a.z, b.z)
}

pub fn dot_eager_bounded_floor(a: V3, b: V3) -> Fixed {
    mul_bounded(a.x, b.x) + mul_bounded(a.y, b.y) + mul_bounded(a.z, b.z)
}

pub fn dot_lazy_wide_mul(a: V3, b: V3) -> Fixed {
    row_wide(a.x, a.y, a.z, b.x, b.y, b.z)
}

pub fn dot_lazy_bounded_floor(a: V3, b: V3) -> Fixed {
    row_bounded(a.x, a.y, a.z, b.x, b.y, b.z)
}

pub fn mul_vec_eager_wide_mul(m: M3, v: V3) -> V3 {
    Vec3 {
        x: mul_wide(m.m11, v.x) + mul_wide(m.m12, v.y) + mul_wide(m.m13, v.z),
        y: mul_wide(m.m21, v.x) + mul_wide(m.m22, v.y) + mul_wide(m.m23, v.z),
        z: mul_wide(m.m31, v.x) + mul_wide(m.m32, v.y) + mul_wide(m.m33, v.z),
    }
}

pub fn mul_vec_eager_bounded_floor(m: M3, v: V3) -> V3 {
    Vec3 {
        x: mul_bounded(m.m11, v.x) + mul_bounded(m.m12, v.y) + mul_bounded(m.m13, v.z),
        y: mul_bounded(m.m21, v.x) + mul_bounded(m.m22, v.y) + mul_bounded(m.m23, v.z),
        z: mul_bounded(m.m31, v.x) + mul_bounded(m.m32, v.y) + mul_bounded(m.m33, v.z),
    }
}

pub fn mul_vec_lazy_wide_mul(m: M3, v: V3) -> V3 {
    Vec3 {
        x: row_wide(m.m11, m.m12, m.m13, v.x, v.y, v.z),
        y: row_wide(m.m21, m.m22, m.m23, v.x, v.y, v.z),
        z: row_wide(m.m31, m.m32, m.m33, v.x, v.y, v.z),
    }
}

pub fn mul_vec_lazy_bounded_floor(m: M3, v: V3) -> V3 {
    Vec3 {
        x: row_bounded(m.m11, m.m12, m.m13, v.x, v.y, v.z),
        y: row_bounded(m.m21, m.m22, m.m23, v.x, v.y, v.z),
        z: row_bounded(m.m31, m.m32, m.m33, v.x, v.y, v.z),
    }
}

// ---- generic API sketch: the scalar exposes its wide accumulator as an associated type --------

/// What a generic `Vec3<T>::dot` needs to stay lazy: an unrescaled product, an accumulator that
/// supports `+` / `-`, and a final rescale. For plain integers / felt252 `Wide = T`.
pub trait WideScalar<T> {
    type Wide;
    fn wide_mul(a: T, b: T) -> Self::Wide;
    fn rescale(wide: Self::Wide) -> T;
}

pub impl FixedWideScalar of WideScalar<Fixed> {
    type Wide = felt252;
    #[inline(always)]
    fn wide_mul(a: Fixed, b: Fixed) -> felt252 {
        a.raw.into() * b.raw.into()
    }
    #[inline(always)]
    fn rescale(wide: felt252) -> Fixed {
        rescale_felt(wide)
    }
}

pub fn dot_lazy_generic<T, impl W: WideScalar<T>, +Add<W::Wide>, +Drop<W::Wide>, +Drop<T>>(
    a: Vec3<T>, b: Vec3<T>,
) -> T {
    W::rescale(W::wide_mul(a.x, b.x) + W::wide_mul(a.y, b.y) + W::wide_mul(a.z, b.z))
}

// ---- dot -------------------------------------------------------------------------------------

pub fn dot_eager(a: V3, b: V3) -> Fixed {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn dot_eager_felt(a: V3, b: V3) -> Fixed {
    mul_felt(a.x, b.x) + mul_felt(a.y, b.y) + mul_felt(a.z, b.z)
}

pub fn dot_lazy_i128(a: V3, b: V3) -> Fixed {
    rescale_i128(wide(a.x, b.x) + wide(a.y, b.y) + wide(a.z, b.z))
}

pub fn dot_lazy_felt(a: V3, b: V3) -> Fixed {
    rescale_felt(felt(a.x, b.x) + felt(a.y, b.y) + felt(a.z, b.z))
}

pub fn dot_lazy_felt_floor(a: V3, b: V3) -> Fixed {
    rescale_felt_floor(felt(a.x, b.x) + felt(a.y, b.y) + felt(a.z, b.z))
}

// ---- cross (each component is a difference of two products) ------------------------------------

pub fn cross_eager(a: V3, b: V3) -> V3 {
    Vec3 { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }
}

pub fn cross_lazy_i128(a: V3, b: V3) -> V3 {
    Vec3 {
        x: rescale_i128(wide(a.y, b.z) - wide(a.z, b.y)),
        y: rescale_i128(wide(a.z, b.x) - wide(a.x, b.z)),
        z: rescale_i128(wide(a.x, b.y) - wide(a.y, b.x)),
    }
}

pub fn cross_lazy_felt(a: V3, b: V3) -> V3 {
    Vec3 {
        x: rescale_felt(felt(a.y, b.z) - felt(a.z, b.y)),
        y: rescale_felt(felt(a.z, b.x) - felt(a.x, b.z)),
        z: rescale_felt(felt(a.x, b.y) - felt(a.y, b.x)),
    }
}

pub fn cross_lazy_felt_floor(a: V3, b: V3) -> V3 {
    Vec3 {
        x: rescale_felt_floor(felt(a.y, b.z) - felt(a.z, b.y)),
        y: rescale_felt_floor(felt(a.z, b.x) - felt(a.x, b.z)),
        z: rescale_felt_floor(felt(a.x, b.y) - felt(a.y, b.x)),
    }
}

// ---- mat * vec ---------------------------------------------------------------------------------

pub fn mul_vec_eager(m: M3, v: V3) -> V3 {
    Vec3 {
        x: m.m11 * v.x + m.m12 * v.y + m.m13 * v.z,
        y: m.m21 * v.x + m.m22 * v.y + m.m23 * v.z,
        z: m.m31 * v.x + m.m32 * v.y + m.m33 * v.z,
    }
}

pub fn mul_vec_lazy_i128(m: M3, v: V3) -> V3 {
    Vec3 {
        x: rescale_i128(wide(m.m11, v.x) + wide(m.m12, v.y) + wide(m.m13, v.z)),
        y: rescale_i128(wide(m.m21, v.x) + wide(m.m22, v.y) + wide(m.m23, v.z)),
        z: rescale_i128(wide(m.m31, v.x) + wide(m.m32, v.y) + wide(m.m33, v.z)),
    }
}

pub fn mul_vec_lazy_felt(m: M3, v: V3) -> V3 {
    Vec3 {
        x: rescale_felt(felt(m.m11, v.x) + felt(m.m12, v.y) + felt(m.m13, v.z)),
        y: rescale_felt(felt(m.m21, v.x) + felt(m.m22, v.y) + felt(m.m23, v.z)),
        z: rescale_felt(felt(m.m31, v.x) + felt(m.m32, v.y) + felt(m.m33, v.z)),
    }
}

pub fn mul_vec_lazy_felt_floor(m: M3, v: V3) -> V3 {
    Vec3 {
        x: rescale_felt_floor(felt(m.m11, v.x) + felt(m.m12, v.y) + felt(m.m13, v.z)),
        y: rescale_felt_floor(felt(m.m21, v.x) + felt(m.m22, v.y) + felt(m.m23, v.z)),
        z: rescale_felt_floor(felt(m.m31, v.x) + felt(m.m32, v.y) + felt(m.m33, v.z)),
    }
}

// ---- mat * mat ---------------------------------------------------------------------------------

pub fn mul_mat_eager(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31,
        m12: a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32,
        m13: a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33,
        m21: a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31,
        m22: a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32,
        m23: a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33,
        m31: a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31,
        m32: a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32,
        m33: a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33,
    }
}

#[inline(always)]
fn row_felt(a1: Fixed, a2: Fixed, a3: Fixed, b1: Fixed, b2: Fixed, b3: Fixed) -> Fixed {
    rescale_felt(felt(a1, b1) + felt(a2, b2) + felt(a3, b3))
}

#[inline(always)]
fn row_felt_floor(a1: Fixed, a2: Fixed, a3: Fixed, b1: Fixed, b2: Fixed, b3: Fixed) -> Fixed {
    rescale_felt_floor(felt(a1, b1) + felt(a2, b2) + felt(a3, b3))
}

#[inline(always)]
fn row_i128(a1: Fixed, a2: Fixed, a3: Fixed, b1: Fixed, b2: Fixed, b3: Fixed) -> Fixed {
    rescale_i128(wide(a1, b1) + wide(a2, b2) + wide(a3, b3))
}

pub fn mul_mat_lazy_i128(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: row_i128(a.m11, a.m12, a.m13, b.m11, b.m21, b.m31),
        m12: row_i128(a.m11, a.m12, a.m13, b.m12, b.m22, b.m32),
        m13: row_i128(a.m11, a.m12, a.m13, b.m13, b.m23, b.m33),
        m21: row_i128(a.m21, a.m22, a.m23, b.m11, b.m21, b.m31),
        m22: row_i128(a.m21, a.m22, a.m23, b.m12, b.m22, b.m32),
        m23: row_i128(a.m21, a.m22, a.m23, b.m13, b.m23, b.m33),
        m31: row_i128(a.m31, a.m32, a.m33, b.m11, b.m21, b.m31),
        m32: row_i128(a.m31, a.m32, a.m33, b.m12, b.m22, b.m32),
        m33: row_i128(a.m31, a.m32, a.m33, b.m13, b.m23, b.m33),
    }
}

pub fn mul_mat_lazy_felt(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: row_felt(a.m11, a.m12, a.m13, b.m11, b.m21, b.m31),
        m12: row_felt(a.m11, a.m12, a.m13, b.m12, b.m22, b.m32),
        m13: row_felt(a.m11, a.m12, a.m13, b.m13, b.m23, b.m33),
        m21: row_felt(a.m21, a.m22, a.m23, b.m11, b.m21, b.m31),
        m22: row_felt(a.m21, a.m22, a.m23, b.m12, b.m22, b.m32),
        m23: row_felt(a.m21, a.m22, a.m23, b.m13, b.m23, b.m33),
        m31: row_felt(a.m31, a.m32, a.m33, b.m11, b.m21, b.m31),
        m32: row_felt(a.m31, a.m32, a.m33, b.m12, b.m22, b.m32),
        m33: row_felt(a.m31, a.m32, a.m33, b.m13, b.m23, b.m33),
    }
}

pub fn mul_mat_lazy_felt_floor(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: row_felt_floor(a.m11, a.m12, a.m13, b.m11, b.m21, b.m31),
        m12: row_felt_floor(a.m11, a.m12, a.m13, b.m12, b.m22, b.m32),
        m13: row_felt_floor(a.m11, a.m12, a.m13, b.m13, b.m23, b.m33),
        m21: row_felt_floor(a.m21, a.m22, a.m23, b.m11, b.m21, b.m31),
        m22: row_felt_floor(a.m21, a.m22, a.m23, b.m12, b.m22, b.m32),
        m23: row_felt_floor(a.m21, a.m22, a.m23, b.m13, b.m23, b.m33),
        m31: row_felt_floor(a.m31, a.m32, a.m33, b.m11, b.m21, b.m31),
        m32: row_felt_floor(a.m31, a.m32, a.m33, b.m12, b.m22, b.m32),
        m33: row_felt_floor(a.m31, a.m32, a.m33, b.m13, b.m23, b.m33),
    }
}

pub fn mul_mat_eager_wide_mul(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: mul_wide(a.m11, b.m11) + mul_wide(a.m12, b.m21) + mul_wide(a.m13, b.m31),
        m12: mul_wide(a.m11, b.m12) + mul_wide(a.m12, b.m22) + mul_wide(a.m13, b.m32),
        m13: mul_wide(a.m11, b.m13) + mul_wide(a.m12, b.m23) + mul_wide(a.m13, b.m33),
        m21: mul_wide(a.m21, b.m11) + mul_wide(a.m22, b.m21) + mul_wide(a.m23, b.m31),
        m22: mul_wide(a.m21, b.m12) + mul_wide(a.m22, b.m22) + mul_wide(a.m23, b.m32),
        m23: mul_wide(a.m21, b.m13) + mul_wide(a.m22, b.m23) + mul_wide(a.m23, b.m33),
        m31: mul_wide(a.m31, b.m11) + mul_wide(a.m32, b.m21) + mul_wide(a.m33, b.m31),
        m32: mul_wide(a.m31, b.m12) + mul_wide(a.m32, b.m22) + mul_wide(a.m33, b.m32),
        m33: mul_wide(a.m31, b.m13) + mul_wide(a.m32, b.m23) + mul_wide(a.m33, b.m33),
    }
}

pub fn mul_mat_eager_bounded_floor(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: mul_bounded(a.m11, b.m11) + mul_bounded(a.m12, b.m21) + mul_bounded(a.m13, b.m31),
        m12: mul_bounded(a.m11, b.m12) + mul_bounded(a.m12, b.m22) + mul_bounded(a.m13, b.m32),
        m13: mul_bounded(a.m11, b.m13) + mul_bounded(a.m12, b.m23) + mul_bounded(a.m13, b.m33),
        m21: mul_bounded(a.m21, b.m11) + mul_bounded(a.m22, b.m21) + mul_bounded(a.m23, b.m31),
        m22: mul_bounded(a.m21, b.m12) + mul_bounded(a.m22, b.m22) + mul_bounded(a.m23, b.m32),
        m23: mul_bounded(a.m21, b.m13) + mul_bounded(a.m22, b.m23) + mul_bounded(a.m23, b.m33),
        m31: mul_bounded(a.m31, b.m11) + mul_bounded(a.m32, b.m21) + mul_bounded(a.m33, b.m31),
        m32: mul_bounded(a.m31, b.m12) + mul_bounded(a.m32, b.m22) + mul_bounded(a.m33, b.m32),
        m33: mul_bounded(a.m31, b.m13) + mul_bounded(a.m32, b.m23) + mul_bounded(a.m33, b.m33),
    }
}

pub fn mul_mat_lazy_wide_mul(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: row_wide(a.m11, a.m12, a.m13, b.m11, b.m21, b.m31),
        m12: row_wide(a.m11, a.m12, a.m13, b.m12, b.m22, b.m32),
        m13: row_wide(a.m11, a.m12, a.m13, b.m13, b.m23, b.m33),
        m21: row_wide(a.m21, a.m22, a.m23, b.m11, b.m21, b.m31),
        m22: row_wide(a.m21, a.m22, a.m23, b.m12, b.m22, b.m32),
        m23: row_wide(a.m21, a.m22, a.m23, b.m13, b.m23, b.m33),
        m31: row_wide(a.m31, a.m32, a.m33, b.m11, b.m21, b.m31),
        m32: row_wide(a.m31, a.m32, a.m33, b.m12, b.m22, b.m32),
        m33: row_wide(a.m31, a.m32, a.m33, b.m13, b.m23, b.m33),
    }
}

pub fn mul_mat_lazy_bounded_floor(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: row_bounded(a.m11, a.m12, a.m13, b.m11, b.m21, b.m31),
        m12: row_bounded(a.m11, a.m12, a.m13, b.m12, b.m22, b.m32),
        m13: row_bounded(a.m11, a.m12, a.m13, b.m13, b.m23, b.m33),
        m21: row_bounded(a.m21, a.m22, a.m23, b.m11, b.m21, b.m31),
        m22: row_bounded(a.m21, a.m22, a.m23, b.m12, b.m22, b.m32),
        m23: row_bounded(a.m21, a.m22, a.m23, b.m13, b.m23, b.m33),
        m31: row_bounded(a.m31, a.m32, a.m33, b.m11, b.m21, b.m31),
        m32: row_bounded(a.m31, a.m32, a.m33, b.m12, b.m22, b.m32),
        m33: row_bounded(a.m31, a.m32, a.m33, b.m13, b.m23, b.m33),
    }
}

// ---- dynamic matrix + lazy rows ----------------------------------------------------------------

/// `DMatSeq::matmul` with the inner product accumulated in felt252 and rescaled once per entry.
pub fn dmat_mul_lazy_felt(lhs: DMatSeq<Fixed>, rhs: DMatSeq<Fixed>) -> DMatSeq<Fixed> {
    assert(lhs.cols == rhs.rows && lhs.cols != 0, 'dim mismatch');
    let rhs_t = rhs.transposed();
    let mut out = array![];
    let mut lhs_rows = lhs.data;
    for _ in 0..lhs.rows {
        let row = lhs_rows.slice(0, lhs.cols);
        lhs_rows = lhs_rows.slice(lhs.cols, lhs_rows.len() - lhs.cols);
        let mut rhs_cols = rhs_t.data;
        for _ in 0..rhs.cols {
            let mut a = row;
            let mut acc: felt252 = 0;
            while let Some(x) = a.pop_front() {
                acc += felt(*x, *rhs_cols.pop_front().unwrap());
            }
            out.append(rescale_felt(acc));
        }
    }
    DMatSeq { data: out.span(), rows: lhs.rows, cols: rhs.cols }
}

#[cfg(test)]
mod tests {
    use crate::fixed::{Fixed, FixedTrait, Scalar, input};
    use crate::mat3::Mat3F;
    use crate::vec3::Vec3;
    use super::*;

    fn lit(n: i64) -> Fixed {
        Scalar::from_int(n)
    }

    fn vectors() -> (V3, V3) {
        (
            Vec3 { x: input(1), y: input(-2), z: input(3) },
            Vec3 { x: input(4), y: input(5), z: input(-6) },
        )
    }

    fn matrices() -> (M3, M3) {
        (
            Mat3F {
                m11: input(1),
                m12: input(2),
                m13: input(3),
                m21: input(0),
                m22: input(1),
                m23: input(4),
                m31: input(5),
                m32: input(6),
                m33: input(0),
            },
            Mat3F {
                m11: input(2),
                m12: input(0),
                m13: input(-1),
                m21: input(1),
                m22: input(3),
                m23: input(2),
                m31: input(-2),
                m32: input(1),
                m33: input(1),
            },
        )
    }

    fn v_is(v: V3, x: i64, y: i64, z: i64) -> bool {
        v.x == lit(x) && v.y == lit(y) && v.z == lit(z)
    }

    #[test]
    #[inline(never)]
    fn bench_lazy_dot__baseline() {
        let (a, _b) = vectors();
        assert!(a.x == lit(1));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__eager() {
        let (a, b) = vectors();
        assert!(dot_eager(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__eager_felt() {
        let (a, b) = vectors();
        assert!(dot_eager_felt(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__lazy_i128() {
        let (a, b) = vectors();
        assert!(dot_lazy_i128(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__lazy_felt() {
        let (a, b) = vectors();
        assert!(dot_lazy_felt(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__lazy_felt_generic_api() {
        let (a, b) = vectors();
        assert!(dot_lazy_generic(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__lazy_felt_floor() {
        let (a, b) = vectors();
        assert!(dot_lazy_felt_floor(a, b) == lit(-24));
    }

    #[test]
    #[inline(never)]
    fn bench_lazy_cross__baseline() {
        let (a, _b) = vectors();
        assert!(v_is(a, 1, -2, 3));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_cross__eager() {
        let (a, b) = vectors();
        assert!(v_is(cross_eager(a, b), -3, 18, 13));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_cross__lazy_i128() {
        let (a, b) = vectors();
        assert!(v_is(cross_lazy_i128(a, b), -3, 18, 13));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_cross__lazy_felt() {
        let (a, b) = vectors();
        assert!(v_is(cross_lazy_felt(a, b), -3, 18, 13));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_cross__lazy_felt_floor() {
        let (a, b) = vectors();
        assert!(v_is(cross_lazy_felt_floor(a, b), -3, 18, 13));
    }

    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__baseline() {
        let ((_m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(v, 1, -2, 3));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__eager() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_eager(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__lazy_i128() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_lazy_i128(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__lazy_felt() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_lazy_felt(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__lazy_felt_floor() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_lazy_felt_floor(m, v), 6, 10, -7));
    }

    fn m_is(r: M3) -> bool {
        r.m11 == lit(-2)
            && r.m12 == lit(9)
            && r.m13 == lit(6)
            && r.m21 == lit(-7)
            && r.m22 == lit(7)
            && r.m23 == lit(6)
            && r.m31 == lit(16)
            && r.m32 == lit(18)
            && r.m33 == lit(7)
    }

    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__baseline() {
        let (a, _b) = matrices();
        assert!(
            a.m11 == lit(1)
                && a.m12 == lit(2)
                && a.m13 == lit(3)
                && a.m21 == lit(0)
                && a.m22 == lit(1)
                && a.m23 == lit(4)
                && a.m31 == lit(5)
                && a.m32 == lit(6)
                && a.m33 == lit(0),
        );
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__eager() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_eager(a, b)));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__lazy_i128() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_lazy_i128(a, b)));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__lazy_felt() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_lazy_felt(a, b)));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__lazy_felt_floor() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_lazy_felt_floor(a, b)));
    }

    fn dynamic(m: M3) -> DMatSeq<Fixed> {
        DMatSeq {
            data: array![m.m11, m.m12, m.m13, m.m21, m.m22, m.m23, m.m31, m.m32, m.m33].span(),
            rows: 3,
            cols: 3,
        }
    }

    fn dynamic_is(m: DMatSeq<Fixed>) -> bool {
        let mut data = m.data;
        let [m11, m12, m13, m21, m22, m23, m31, m32, m33] = (*data.multi_pop_front::<9>().unwrap())
            .unbox();
        m_is(Mat3F { m11, m12, m13, m21, m22, m23, m31, m32, m33 })
    }

    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__dynamic_eager() {
        let (a, b) = matrices();
        assert!(dynamic_is(dynamic(a).matmul(dynamic(b))));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__dynamic_lazy_felt() {
        let (a, b) = matrices();
        assert!(dynamic_is(dmat_mul_lazy_felt(dynamic(a), dynamic(b))));
    }

    #[test]
    #[inline(never)]
    fn bench_lazy_dot__eager_wide_mul() {
        let (a, b) = vectors();
        assert!(dot_eager_wide_mul(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__eager_bounded_floor() {
        let (a, b) = vectors();
        assert!(dot_eager_bounded_floor(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__lazy_wide_mul() {
        let (a, b) = vectors();
        assert!(dot_lazy_wide_mul(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_dot__lazy_bounded_floor() {
        let (a, b) = vectors();
        assert!(dot_lazy_bounded_floor(a, b) == lit(-24));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__eager_wide_mul() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_eager_wide_mul(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__eager_bounded_floor() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_eager_bounded_floor(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__lazy_wide_mul() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_lazy_wide_mul(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_vec__lazy_bounded_floor() {
        let ((m, _), (v, _)) = (matrices(), vectors());
        assert!(v_is(mul_vec_lazy_bounded_floor(m, v), 6, 10, -7));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__eager_wide_mul() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_eager_wide_mul(a, b)));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__eager_bounded_floor() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_eager_bounded_floor(a, b)));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__lazy_wide_mul() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_lazy_wide_mul(a, b)));
    }
    #[test]
    #[inline(never)]
    fn bench_lazy_mul_mat__lazy_bounded_floor() {
        let (a, b) = matrices();
        assert!(m_is(mul_mat_lazy_bounded_floor(a, b)));
    }

    /// Rounding: eager truncates every product toward zero (error up to n ulp, biased), lazy
    /// truncates once. With x = 2^-32 (1 ulp) and y = 0.5: x*y truncates to 0 each time, so
    /// eager dot((x,x,x),(y,y,y)) = 0 while the exact value 1.5 ulp gives 1 ulp lazily.
    #[test]
    fn lazy_rounding_is_tighter() {
        let ulp = FixedTrait::from_raw(1);
        let half = FixedTrait::from_raw(crate::fixed::HALF);
        let a = Vec3 { x: ulp, y: ulp, z: ulp };
        let b = Vec3 { x: half, y: half, z: half };
        assert!(dot_eager(a, b).raw == 0);
        assert!(dot_lazy_i128(a, b).raw == 1);
        assert!(dot_lazy_felt(a, b).raw == 1);
        assert!(dot_lazy_felt_floor(a, b).raw == 1);
        // Negative side: truncation rounds -1.5 ulp to -1, floor rounds it to -2.
        let c = Vec3 { x: -ulp, y: -ulp, z: -ulp };
        assert!(dot_eager(c, b).raw == 0);
        assert!(dot_lazy_felt(c, b).raw == -1);
        assert!(dot_lazy_felt_floor(c, b).raw == -2);
        assert!(dot_lazy_bounded_floor(c, b).raw == -2);
        assert!(dot_lazy_bounded_floor(a, b).raw == 1);
        assert!(dot_lazy_wide_mul(c, b).raw == -1);
        // Eager floor: each -0.5 ulp product floors to -1 ulp => -3 ulp (error up to n ulp).
        assert!(dot_eager_bounded_floor(c, b).raw == -3);
        assert!(dot_eager_wide_mul(c, b).raw == 0);
    }

    /// Overflow: a felt252 accumulator cannot overflow (n * 2^126 << P), only the final value
    /// is range-checked. An i128 accumulator can overflow on an intermediate sum, but only from
    /// 4 terms up and only with operands at the very edge of the type (for <= 3 terms an
    /// intermediate overflow implies a final overflow): min*min + min*min + min*max + min*max.
    #[test]
    fn lazy_felt_survives_intermediate_overflow() {
        let min = FixedTrait::from_raw(-0x8000000000000000);
        let max = FixedTrait::from_raw(0x7fffffffffffffff);
        let sum = felt(min, min) + felt(min, min) + felt(min, max) + felt(min, max);
        assert!(rescale_felt(sum).raw == crate::fixed::ONE); // 2^64 / 2^32
        assert!(rescale_felt_floor(sum).raw == crate::fixed::ONE);
    }

    #[test]
    #[should_panic(expected: 'i128_add Overflow')]
    fn lazy_i128_intermediate_overflow_panics() {
        let min = FixedTrait::from_raw(-0x8000000000000000);
        let max = FixedTrait::from_raw(0x7fffffffffffffff);
        rescale_i128(wide(min, min) + wide(min, min) + wide(min, max) + wide(min, max));
    }
}
