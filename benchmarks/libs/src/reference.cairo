//! Hand-written comparison points: the "obvious" implementation of every measured operation, so
//! that each library figure can be read against something inside the same report.
//!
//! These are NOT the proposed nalgebra.cairo design (the `primitives/`, `scalar/` and `layout/`
//! benchmark packages explore that); they are deliberately straightforward:
//!  - fixed point = a two's complement native integer (`i32` / `i64` / `i128`), widened for
//!  mul/div;
//!  - vectors / matrices = plain structs with unrolled arithmetic.

/// Q16.16 on `i32`, widening to `i64`.
pub mod q16 {
    use core::num::traits::WideMul;

    pub const ONE: i32 = 0x10000;
    const ONE_WIDE: i64 = 0x10000;

    pub fn mul(a: i32, b: i32) -> i32 {
        (a.wide_mul(b) / ONE_WIDE).try_into().unwrap()
    }

    pub fn div(a: i32, b: i32) -> i32 {
        (a.wide_mul(ONE) / b.into()).try_into().unwrap()
    }

    pub fn dot3(a: [i32; 3], b: [i32; 3]) -> i32 {
        let [ax, ay, az] = a;
        let [bx, by, bz] = b;
        ((ax.wide_mul(bx) + ay.wide_mul(by) + az.wide_mul(bz)) / ONE_WIDE).try_into().unwrap()
    }

    /// Row-major 3x3 product, one rescale per output entry.
    pub fn matmul3(a: [i32; 9], b: [i32; 9]) -> [i32; 9] {
        let [a0, a1, a2, a3, a4, a5, a6, a7, a8] = a;
        let [b0, b1, b2, b3, b4, b5, b6, b7, b8] = b;
        [
            dot3([a0, a1, a2], [b0, b3, b6]), dot3([a0, a1, a2], [b1, b4, b7]),
            dot3([a0, a1, a2], [b2, b5, b8]), dot3([a3, a4, a5], [b0, b3, b6]),
            dot3([a3, a4, a5], [b1, b4, b7]), dot3([a3, a4, a5], [b2, b5, b8]),
            dot3([a6, a7, a8], [b0, b3, b6]), dot3([a6, a7, a8], [b1, b4, b7]),
            dot3([a6, a7, a8], [b2, b5, b8]),
        ]
    }
}

/// Q32.32 on `i64`, widening to `i128`.
pub mod q32 {
    use core::num::traits::{Sqrt, WideMul, Zero};
    use core::ops::{AddAssign, SubAssign};

    pub const ONE: i64 = 0x100000000;
    const ONE_WIDE: i128 = 0x100000000;
    const ONE_U64: u64 = 0x100000000;

    const TWO_PI: i64 = 26986075409;
    const PI: i64 = 13493037705;
    const HALF_PI: i64 = 6746518852;

    /// Newtype so that generic containers (alexandria `dot`, origami `Matrix<T>`) can be
    /// instantiated with the reference number and isolate the container overhead.
    #[derive(Copy, Drop, PartialEq, Debug)]
    pub struct Fix64 {
        pub v: i64,
    }

    #[inline]
    pub fn fix(v: i64) -> Fix64 {
        Fix64 { v }
    }

    pub fn mul(a: i64, b: i64) -> i64 {
        (a.wide_mul(b) / ONE_WIDE).try_into().unwrap()
    }

    pub fn div(a: i64, b: i64) -> i64 {
        (a.wide_mul(ONE) / b.into()).try_into().unwrap()
    }

    pub fn sqrt(a: i64) -> i64 {
        let mag: u64 = a.try_into().expect('must be positive');
        let root: u64 = Sqrt::sqrt(mag.wide_mul(ONE_U64));
        root.try_into().unwrap()
    }

    /// Odd Taylor polynomial up to x^11 (Horner in x^2) after folding into [-pi/2, pi/2].
    pub fn sin(a: i64) -> i64 {
        let mut r = a % TWO_PI;
        if r > PI {
            r -= TWO_PI;
        } else if r < -PI {
            r += TWO_PI;
        }
        if r > HALF_PI {
            r = PI - r;
        } else if r < -HALF_PI {
            r = -PI - r;
        }
        let x2 = mul(r, r);
        let mut p = mul(-108, x2) + 11836; // -1/11! , 1/9!
        p = mul(p, x2) - 852176; // -1/7!
        p = mul(p, x2) + 35791394; // 1/5!
        p = mul(p, x2) - 715827883; // -1/3!
        r + mul(r, mul(p, x2))
    }

    pub impl Fix64Add of Add<Fix64> {
        fn add(lhs: Fix64, rhs: Fix64) -> Fix64 {
            Fix64 { v: lhs.v + rhs.v }
        }
    }
    pub impl Fix64Sub of Sub<Fix64> {
        fn sub(lhs: Fix64, rhs: Fix64) -> Fix64 {
            Fix64 { v: lhs.v - rhs.v }
        }
    }
    pub impl Fix64Mul of Mul<Fix64> {
        fn mul(lhs: Fix64, rhs: Fix64) -> Fix64 {
            Fix64 { v: mul(lhs.v, rhs.v) }
        }
    }
    pub impl Fix64Div of Div<Fix64> {
        fn div(lhs: Fix64, rhs: Fix64) -> Fix64 {
            Fix64 { v: div(lhs.v, rhs.v) }
        }
    }
    pub impl Fix64Neg of Neg<Fix64> {
        fn neg(a: Fix64) -> Fix64 {
            Fix64 { v: -a.v }
        }
    }
    pub impl Fix64AddAssign of AddAssign<Fix64, Fix64> {
        fn add_assign(ref self: Fix64, rhs: Fix64) {
            self = Fix64 { v: self.v + rhs.v };
        }
    }
    pub impl Fix64SubAssign of SubAssign<Fix64, Fix64> {
        fn sub_assign(ref self: Fix64, rhs: Fix64) {
            self = Fix64 { v: self.v - rhs.v };
        }
    }
    pub impl Fix64Zero of Zero<Fix64> {
        fn zero() -> Fix64 {
            Fix64 { v: 0 }
        }
        fn is_zero(self: @Fix64) -> bool {
            *self.v == 0
        }
        fn is_non_zero(self: @Fix64) -> bool {
            *self.v != 0
        }
    }
    pub impl Fix64PartialOrd of PartialOrd<Fix64> {
        fn lt(lhs: Fix64, rhs: Fix64) -> bool {
            lhs.v < rhs.v
        }
    }

    // ---- Vectors ------------------------------------------------------------------------------

    #[derive(Copy, Drop, PartialEq, Debug)]
    pub struct Vec2 {
        pub x: i64,
        pub y: i64,
    }

    #[derive(Copy, Drop, PartialEq, Debug)]
    pub struct Vec3 {
        pub x: i64,
        pub y: i64,
        pub z: i64,
    }

    pub fn add3(a: Vec3, b: Vec3) -> Vec3 {
        Vec3 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
    }

    /// Same shape as the libraries: one rescale per product.
    pub fn dot3(a: Vec3, b: Vec3) -> i64 {
        mul(a.x, b.x) + mul(a.y, b.y) + mul(a.z, b.z)
    }

    /// Accumulate the wide products, rescale once.
    pub fn dot3_fused(a: Vec3, b: Vec3) -> i64 {
        ((a.x.wide_mul(b.x) + a.y.wide_mul(b.y) + a.z.wide_mul(b.z)) / ONE_WIDE).try_into().unwrap()
    }

    pub fn dot2_fused(a: Vec2, b: Vec2) -> i64 {
        ((a.x.wide_mul(b.x) + a.y.wide_mul(b.y)) / ONE_WIDE).try_into().unwrap()
    }

    pub fn cross3(a: Vec3, b: Vec3) -> Vec3 {
        Vec3 {
            x: mul(a.y, b.z) - mul(a.z, b.y),
            y: mul(a.z, b.x) - mul(a.x, b.z),
            z: mul(a.x, b.y) - mul(a.y, b.x),
        }
    }

    pub fn cross3_fused(a: Vec3, b: Vec3) -> Vec3 {
        Vec3 {
            x: ((a.y.wide_mul(b.z) - a.z.wide_mul(b.y)) / ONE_WIDE).try_into().unwrap(),
            y: ((a.z.wide_mul(b.x) - a.x.wide_mul(b.z)) / ONE_WIDE).try_into().unwrap(),
            z: ((a.x.wide_mul(b.y) - a.y.wide_mul(b.x)) / ONE_WIDE).try_into().unwrap(),
        }
    }

    pub fn norm3(a: Vec3) -> i64 {
        sqrt(dot3(a, a))
    }

    /// sqrt of the un-rescaled sum of squares: no rescale at all, full precision.
    pub fn norm3_fused(a: Vec3) -> i64 {
        let sum: i128 = a.x.wide_mul(a.x) + a.y.wide_mul(a.y) + a.z.wide_mul(a.z);
        let sum: u128 = sum.try_into().unwrap();
        let root: u64 = Sqrt::sqrt(sum);
        root.try_into().unwrap()
    }

    // ---- 3x3 matrices -------------------------------------------------------------------------

    #[derive(Copy, Drop, PartialEq, Debug)]
    pub struct Mat3 {
        pub r0: Vec3,
        pub r1: Vec3,
        pub r2: Vec3,
    }

    pub fn transpose3(a: Mat3) -> Mat3 {
        Mat3 {
            r0: Vec3 { x: a.r0.x, y: a.r1.x, z: a.r2.x },
            r1: Vec3 { x: a.r0.y, y: a.r1.y, z: a.r2.y },
            r2: Vec3 { x: a.r0.z, y: a.r1.z, z: a.r2.z },
        }
    }

    pub fn matadd3(a: Mat3, b: Mat3) -> Mat3 {
        Mat3 { r0: add3(a.r0, b.r0), r1: add3(a.r1, b.r1), r2: add3(a.r2, b.r2) }
    }

    pub fn matmul3(a: Mat3, b: Mat3) -> Mat3 {
        let c0 = Vec3 { x: b.r0.x, y: b.r1.x, z: b.r2.x };
        let c1 = Vec3 { x: b.r0.y, y: b.r1.y, z: b.r2.y };
        let c2 = Vec3 { x: b.r0.z, y: b.r1.z, z: b.r2.z };
        Mat3 {
            r0: Vec3 { x: dot3(a.r0, c0), y: dot3(a.r0, c1), z: dot3(a.r0, c2) },
            r1: Vec3 { x: dot3(a.r1, c0), y: dot3(a.r1, c1), z: dot3(a.r1, c2) },
            r2: Vec3 { x: dot3(a.r2, c0), y: dot3(a.r2, c1), z: dot3(a.r2, c2) },
        }
    }

    pub fn matmul3_fused(a: Mat3, b: Mat3) -> Mat3 {
        let c0 = Vec3 { x: b.r0.x, y: b.r1.x, z: b.r2.x };
        let c1 = Vec3 { x: b.r0.y, y: b.r1.y, z: b.r2.y };
        let c2 = Vec3 { x: b.r0.z, y: b.r1.z, z: b.r2.z };
        Mat3 {
            r0: Vec3 { x: dot3_fused(a.r0, c0), y: dot3_fused(a.r0, c1), z: dot3_fused(a.r0, c2) },
            r1: Vec3 { x: dot3_fused(a.r1, c0), y: dot3_fused(a.r1, c1), z: dot3_fused(a.r1, c2) },
            r2: Vec3 { x: dot3_fused(a.r2, c0), y: dot3_fused(a.r2, c1), z: dot3_fused(a.r2, c2) },
        }
    }

    pub fn matvec3_fused(a: Mat3, v: Vec3) -> Vec3 {
        Vec3 { x: dot3_fused(a.r0, v), y: dot3_fused(a.r1, v), z: dot3_fused(a.r2, v) }
    }

    /// det = r0 . (r1 x r2)
    pub fn det3(a: Mat3) -> i64 {
        dot3_fused(a.r0, cross3_fused(a.r1, a.r2))
    }

    /// Adjugate / determinant, the rows of the adjugate transpose being cross products.
    pub fn inv3(a: Mat3) -> Mat3 {
        let c0 = cross3_fused(a.r1, a.r2);
        let c1 = cross3_fused(a.r2, a.r0);
        let c2 = cross3_fused(a.r0, a.r1);
        let det = dot3_fused(a.r0, c0);
        assert(det != 0, 'not invertible');
        Mat3 {
            r0: Vec3 { x: div(c0.x, det), y: div(c1.x, det), z: div(c2.x, det) },
            r1: Vec3 { x: div(c0.y, det), y: div(c1.y, det), z: div(c2.y, det) },
            r2: Vec3 { x: div(c0.z, det), y: div(c1.z, det), z: div(c2.z, det) },
        }
    }
}

/// Q64.64 on `i128`, widening to two `u128` limbs.
pub mod q64 {
    use core::num::traits::{Sqrt, WideMul};

    pub const ONE: i128 = 0x10000000000000000;
    const ONE_U128: u128 = 0x10000000000000000;

    fn split(a: i128) -> (u128, bool) {
        if a < 0 {
            ((-a).try_into().unwrap(), true)
        } else {
            (a.try_into().unwrap(), false)
        }
    }

    fn join(mag: u128, neg: bool) -> i128 {
        let v: i128 = mag.try_into().unwrap();
        if neg {
            -v
        } else {
            v
        }
    }

    /// 128x128 -> 256 bit product, then a limb shuffle instead of a u256 division.
    pub fn mul(a: i128, b: i128) -> i128 {
        let (am, an) = split(a);
        let (bm, bn) = split(b);
        let wide: u256 = am.wide_mul(bm);
        // (high * 2^128 + low) / 2^64 = high * 2^64 + low / 2^64
        join(wide.high * ONE_U128 + wide.low / ONE_U128, an != bn)
    }

    pub fn div(a: i128, b: i128) -> i128 {
        let (am, an) = split(a);
        let (bm, bn) = split(b);
        let wide: u256 = am.wide_mul(ONE_U128);
        let q = wide / bm.into();
        assert(q.high == 0, 'overflow');
        join(q.low, an != bn)
    }

    /// Full precision: sqrt(mag * 2^64) over u256.
    pub fn sqrt(a: i128) -> i128 {
        let mag: u128 = a.try_into().expect('must be positive');
        let wide: u256 = mag.wide_mul(ONE_U128);
        let root: u128 = Sqrt::sqrt(wide);
        root.try_into().unwrap()
    }
}

/// Q32.32 on `i64` with the `BoundedInt` biased-floor kernels, copied from
/// `benchmarks/primitives/src/bounded.cairo` (`fpmul_i64_bounded_floor`, `dot3_i64_bounded`): the
/// kernel nalgebra.cairo intends to ship. The product is range-tracked (no overflow check), biased
/// to be non-negative, divided by the constant 2^32 (floor semantics, i.e. an arithmetic shift),
/// un-biased, then narrowed with a single range check.
pub mod q32_bounded {
    #[feature("bounded-int-utils")]
    use core::internal::bounded_int::{
        self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, downcast,
    };
    use super::q32::Vec3;

    type Two32 = UnitInt<0x100000000>;
    const NZ_TWO32: NonZero<Two32> = 0x100000000;
    type U32Rem = BoundedInt<0, 0xffffffff>;

    type I64Sq =
        BoundedInt<-0x3fffffffffffffff8000000000000000, 0x40000000000000000000000000000000>;
    type Offset = UnitInt<0x40000000000000000000000000000000>;
    type I64SqOffset = BoundedInt<0x8000000000000000, 0x80000000000000000000000000000000>;
    type I64SqOffsetShr32 = BoundedInt<0x80000000, 0x800000000000000000000000>;
    type OffsetShr32 = UnitInt<0x400000000000000000000000>;

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

    pub fn mul(a: i64, b: i64) -> i64 {
        let p: I64Sq = bounded_int::mul(a, b);
        let biased: I64SqOffset = bounded_int::add::<
            _, Offset,
        >(p, 0x40000000000000000000000000000000);
        let (q, _r) = bounded_int::div_rem(biased, NZ_TWO32);
        let q = bounded_int::sub::<_, OffsetShr32>(q, 0x400000000000000000000000);
        downcast(q).expect('fpmul overflow')
    }

    type I64Sq2 =
        BoundedInt<-0x7fffffffffffffff0000000000000000, 0x80000000000000000000000000000000>;
    type I64Sq3 =
        BoundedInt<-0xbffffffffffffffe8000000000000000, 0xc0000000000000000000000000000000>;
    type Offset3 = UnitInt<0xc0000000000000000000000000000000>;
    type I64Sq3Offset = BoundedInt<0x18000000000000000, 0x180000000000000000000000000000000>;
    type I64Sq3OffsetShr32 = BoundedInt<0x180000000, 0x1800000000000000000000000>;
    type Offset3Shr32 = UnitInt<0xc00000000000000000000000>;

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

    /// Fused dot product: three range-check-free products and sums, one div_rem, one narrowing.
    pub fn dot3(a: Vec3, b: Vec3) -> i64 {
        let xx: I64Sq = bounded_int::mul(a.x, b.x);
        let yy: I64Sq = bounded_int::mul(a.y, b.y);
        let zz: I64Sq = bounded_int::mul(a.z, b.z);
        let sum: I64Sq3 = bounded_int::add(bounded_int::add(xx, yy), zz);
        let biased: I64Sq3Offset = bounded_int::add::<
            _, Offset3,
        >(sum, 0xc0000000000000000000000000000000);
        let (q, _r) = bounded_int::div_rem(biased, NZ_TWO32);
        let q = bounded_int::sub::<_, Offset3Shr32>(q, 0xc00000000000000000000000);
        downcast(q).expect('dot overflow')
    }
}
