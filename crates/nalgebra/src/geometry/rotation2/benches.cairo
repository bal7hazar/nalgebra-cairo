//! Gas benchmarks of `Rotation2` (`bench_rotation2_<op>__<variant>`, net = raw - `baseline` of
//! the group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! The rotations used are `new(0.4 rad)` and `new(-1/6 rad)`, the vector `(1.5, -2.25)`; the
//! `alt_unit_complex` variants measure the same operation on `UnitComplex`, reached from the
//! matrix by `to_unit_complex` (a copy of two components).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::matrix_test_utils::{fx, p2, r2, v2};
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use crate::geometry::unit_complex::{UnitComplex, UnitComplexTrait};
use super::{Rotation2, Rotation2AngleTrait, Rotation2Trait};

/// `new(0.4)`.
fn r() -> Rotation2<Fixed> {
    r2([[3955926847, -1672539044], [1672539044, 3955926847]])
}

/// `new(-1/6)`.
fn s() -> Rotation2<Fixed> {
    r2([[4235452929, 712518464], [-712518464, 4235452929]])
}

// --- alternative implementations (losers)

/// The composition through the unit complex form: two fused kernels instead of four, but `m12`
/// is then `-floor(im)` where the matrix product gives `floor(-im)` — 1 ulp apart whenever the
/// exact value is not an integer, so it does not reproduce upstream bit for bit (the oracle
/// tolerance of `rotation2_mul` is 0).
#[inline(always)]
fn alt_mul_complex(a: Rotation2<Fixed>, b: Rotation2<Fixed>) -> Rotation2<Fixed> {
    let c = UnitComplex { re: a.matrix.m11, im: a.matrix.m21 };
    let d = UnitComplex { re: b.matrix.m11, im: b.matrix.m21 };
    (c * d).to_rotation_matrix()
}

// --- why the alternatives lost

#[test]
fn test_mul_alt_complex_differs_by_one_ulp_on_m12() {
    let (a, b) = (r(), s());
    let got = a * b;
    let alt = alt_mul_complex(a, b);
    assert!(alt.matrix.m11 == got.matrix.m11 && alt.matrix.m21 == got.matrix.m21);
    assert!(alt.matrix.m22 == got.matrix.m22);
    assert!(alt.matrix.m12 != got.matrix.m12);
    assert!(alt.abs_diff_eq(got, 1));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_rotation2_identity__baseline() {
    let _a: Fixed = black_box(fx(0x66666666));
    let e: Rotation2<Fixed> = black_box(r2([[0x100000000, 0], [0, 0x100000000]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_identity__const() {
    let _a: Fixed = black_box(fx(0x66666666));
    let e: Rotation2<Fixed> = black_box(r2([[0x100000000, 0], [0, 0x100000000]]));
    assert!(Rotation2Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix_unchecked__baseline() {
    let _m: Matrix2<Fixed> = black_box(r().matrix);
    let e: Rotation2<Fixed> = black_box(r());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix_unchecked__wrap() {
    let m: Matrix2<Fixed> = black_box(r().matrix);
    let e: Rotation2<Fixed> = black_box(r());
    assert!(Rotation2Trait::from_matrix_unchecked(m) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix__baseline() {
    let _m: Matrix2<Fixed> = black_box(
        Matrix2 { m11: fx(15823707388), m21: fx(6690156176), m12: Real::ZERO, m22: Real::ZERO },
    );
    let e: Rotation2<Fixed> = black_box(r2([[3955926847, -1672539044], [1672539044, 3955926847]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_matrix__normalize_first_column() {
    let m: Matrix2<Fixed> = black_box(
        Matrix2 { m11: fx(15823707388), m21: fx(6690156176), m12: Real::ZERO, m22: Real::ZERO },
    );
    let e: Rotation2<Fixed> = black_box(r2([[3955926847, -1672539044], [1672539044, 3955926847]]));
    assert!(Rotation2Trait::from_matrix(m) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_new__baseline() {
    let _a: Fixed = black_box(fx(0x66666666));
    let e: Rotation2<Fixed> = black_box(r());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_new__sin_cos() {
    let a: Fixed = black_box(fx(0x66666666));
    let e: Rotation2<Fixed> = black_box(r());
    assert!(Rotation2AngleTrait::new(a) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_matrix__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let e: Fixed = black_box(fx(-1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_matrix__accessor() {
    let r: Rotation2<Fixed> = black_box(r());
    let e: Fixed = black_box(fx(-1672539044));
    assert!(r.matrix().m12 == e && r.into_inner().m12 == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let e: Rotation2<Fixed> = black_box(r2([[3955926847, 1672539044], [-1672539044, 3955926847]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse__transpose() {
    let r: Rotation2<Fixed> = black_box(r());
    let e: Rotation2<Fixed> = black_box(r2([[3955926847, 1672539044], [-1672539044, 3955926847]]));
    assert!(r.inverse() == e);
    assert!(r.transpose() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_mul__baseline() {
    let _a: Rotation2<Fixed> = black_box(r());
    let _b: Rotation2<Fixed> = black_box(s());
    let e: Rotation2<Fixed> = black_box(r2([[4178578243, -993090094], [993090093, 4178578243]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_mul__matrix() {
    let a: Rotation2<Fixed> = black_box(r());
    let b: Rotation2<Fixed> = black_box(s());
    let e: Rotation2<Fixed> = black_box(r2([[4178578243, -993090094], [993090093, 4178578243]]));
    assert!(a * b == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_mul__alt_complex() {
    let a: Rotation2<Fixed> = black_box(r());
    let b: Rotation2<Fixed> = black_box(s());
    let e: Rotation2<Fixed> = black_box(r2([[4178578243, -993090093], [993090093, 4178578243]]));
    assert!(alt_mul_complex(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_transform_vector__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(9697103119, -6392026840));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_transform_vector__matrix() {
    let r: Rotation2<Fixed> = black_box(r());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(9697103119, -6392026840));
    assert!(r.transform_vector(v) == e);
    assert!(r.mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_transform_vector__alt_unit_complex() {
    let r: Rotation2<Fixed> = black_box(r());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(9697103119, -6392026840));
    assert!(r.to_unit_complex().transform_vector(v) == e);
    assert!(r.to_unit_complex().mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_transform_point__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(9697103119, -6392026840));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_transform_point__matrix() {
    let r: Rotation2<Fixed> = black_box(r());
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(9697103119, -6392026840));
    assert!(r.transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_transform_point__alt_unit_complex() {
    let r: Rotation2<Fixed> = black_box(r());
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(9697103119, -6392026840));
    assert!(r.to_unit_complex().transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse_transform_vector__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(2170677421, -11409643972));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse_transform_vector__transposed() {
    let r: Rotation2<Fixed> = black_box(r());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(2170677421, -11409643972));
    assert!(r.inverse_transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse_transform_vector__alt_inverse_then_transform() {
    let r: Rotation2<Fixed> = black_box(r());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(2170677421, -11409643972));
    assert!(r.inverse().transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse_transform_point__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(2170677421, -11409643972));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_inverse_transform_point__transposed() {
    let r: Rotation2<Fixed> = black_box(r());
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(2170677421, -11409643972));
    assert!(r.inverse_transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_to_homogeneous__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let e: Fixed = black_box(fx(-1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_to_homogeneous__expand() {
    let r: Rotation2<Fixed> = black_box(r());
    let e: Fixed = black_box(fx(-1672539044));
    let h = r.to_homogeneous();
    assert!(h.m12 == e && h.m33 == Real::ONE);
}

#[test]
#[inline(never)]
fn bench_rotation2_to_unit_complex__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let e: UnitComplex<Fixed> = black_box(UnitComplex { re: fx(3955926847), im: fx(1672539044) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_to_unit_complex__first_column() {
    let r: Rotation2<Fixed> = black_box(r());
    let e: UnitComplex<Fixed> = black_box(UnitComplex { re: fx(3955926847), im: fx(1672539044) });
    assert!(r.to_unit_complex() == e);
    let into: UnitComplex<Fixed> = r.into();
    assert!(into == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_unit_complex__baseline() {
    let _c: UnitComplex<Fixed> = black_box(UnitComplex { re: fx(3955926847), im: fx(1672539044) });
    let e: Rotation2<Fixed> = black_box(r());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_from_unit_complex__expand() {
    let c: UnitComplex<Fixed> = black_box(UnitComplex { re: fx(3955926847), im: fx(1672539044) });
    let e: Rotation2<Fixed> = black_box(r());
    let into: Rotation2<Fixed> = c.into();
    assert!(into == e);
    assert!(c.to_rotation_matrix() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_rotation_between__baseline() {
    let _a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Rotation2<Fixed> = black_box(
        r2([[-2576980378, 3435973837], [-3435973837, -2576980378]]),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_rotation_between__algebraic() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Rotation2<Fixed> = black_box(
        r2([[-2576980378, 3435973837], [-3435973837, -2576980378]]),
    );
    assert!(Rotation2Trait::rotation_between(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_rotation_between__alt_unit_complex() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: Rotation2<Fixed> = black_box(
        r2([[-2576980378, 3435973837], [-3435973837, -2576980378]]),
    );
    assert!(UnitComplexTrait::rotation_between(a, b).to_rotation_matrix() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_scaled_rotation_between__baseline() {
    let _a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let _s: Fixed = black_box(Real::HALF);
    let e: Rotation2<Fixed> = black_box(r2([[1920767767, 3841535534], [-3841535534, 1920767767]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_scaled_rotation_between__atan2_sin_cos() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let s: Fixed = black_box(Real::HALF);
    let e: Rotation2<Fixed> = black_box(r2([[1920767767, 3841535534], [-3841535534, 1920767767]]));
    assert!(Rotation2AngleTrait::scaled_rotation_between(a, b, s) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_angle__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let e: Fixed = black_box(fx(1717986918));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_angle__atan2() {
    let r: Rotation2<Fixed> = black_box(r());
    let e: Fixed = black_box(fx(1717986918));
    assert!(r.angle() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_angle_to__baseline() {
    let _a: Rotation2<Fixed> = black_box(r());
    let _b: Rotation2<Fixed> = black_box(s());
    let e: Fixed = black_box(fx(-2433814801));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_angle_to__first_columns() {
    let a: Rotation2<Fixed> = black_box(r());
    let b: Rotation2<Fixed> = black_box(s());
    let e: Fixed = black_box(fx(-2433814801));
    assert!(a.angle_to(b) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_angle_to__alt_product_then_angle() {
    let a: Rotation2<Fixed> = black_box(r());
    let b: Rotation2<Fixed> = black_box(s());
    let e: Fixed = black_box(fx(-2433814801));
    assert!((b * a.inverse()).angle() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_powf__baseline() {
    let _r: Rotation2<Fixed> = black_box(r());
    let _n: Fixed = black_box(Real::TWO);
    let e: Rotation2<Fixed> = black_box(r2([[2992332532, -3081020949], [3081020949, 2992332532]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_powf__atan2_sin_cos() {
    let r: Rotation2<Fixed> = black_box(r());
    let n: Fixed = black_box(Real::TWO);
    let e: Rotation2<Fixed> = black_box(r2([[2992332532, -3081020949], [3081020949, 2992332532]]));
    assert!(r.powf(n) == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_renormalize__baseline() {
    let _r: Rotation2<Fixed> = black_box(r2([[3955930943, -1672539044], [1672543140, 3955926847]]));
    let e: Rotation2<Fixed> = black_box(r2([[3955925999, -1672541049], [1672541049, 3955925999]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_renormalize__first_column() {
    let r: Rotation2<Fixed> = black_box(r2([[3955930943, -1672539044], [1672543140, 3955926847]]));
    let e: Rotation2<Fixed> = black_box(r2([[3955925999, -1672541049], [1672541049, 3955925999]]));
    assert!(r.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_abs_diff_eq__baseline() {
    let _a: Rotation2<Fixed> = black_box(r());
    let _b: Rotation2<Fixed> = black_box(s());
    let e: bool = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation2_abs_diff_eq__all_compared() {
    let a: Rotation2<Fixed> = black_box(r());
    let b: Rotation2<Fixed> = black_box(s());
    let e: bool = black_box(false);
    assert!(a.abs_diff_eq(b, 8) == e);
}
