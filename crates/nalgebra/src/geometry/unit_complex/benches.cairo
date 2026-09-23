//! Gas benchmarks of `UnitComplex` (`bench_unit_complex_<op>__<variant>`, net = raw - `baseline`
//! of the group), and the alternative implementations that lost (`alt_*`), kept as evidence
//! together with the tests showing why (AGENTS.md rule 8).
//!
//! The rotations used are `c = new(0.4 rad)` and `d = new(-1/6 rad)`, the vector `(1.5, -2.25)`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix_test_utils::{fx, p2, uc, v2};
use crate::base::point2::Point2;
use crate::base::vector2::Vector2;
use crate::geometry::rotation2::Rotation2;
use super::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};

/// `new(0.4)`.
fn c() -> UnitComplex<Fixed> {
    uc(3955926847, 1672539044)
}

/// `new(-1/6)`.
fn d() -> UnitComplex<Fixed> {
    uc(4235452929, -712518464)
}

// --- alternative implementations (losers)

/// `rotation_between` the way upstream writes it: normalize both vectors, take the angle of the
/// pair with `atan2`, then rebuild the rotation with `sin_cos`. Two transcendentals for a result
/// the algebraic form gets from one norm and two divisions.
#[inline(always)]
fn alt_rotation_between_atan2(a: Vector2<Fixed>, b: Vector2<Fixed>) -> UnitComplex<Fixed> {
    let na = Real::norm2(a.x, a.y);
    let nb = Real::norm2(b.x, b.y);
    if na == Real::ZERO || nb == Real::ZERO {
        return UnitComplexTrait::identity();
    }
    let (ax, ay) = (a.x / na, a.y / na);
    let (bx, by) = (b.x / nb, b.y / nb);
    let angle = Transcendental::atan2(
        Real::diff_prod(ax, by, ay, bx), Real::sum_prod2(ax, bx, ay, by),
    );
    UnitComplexAngleTrait::new(angle)
}

/// `rotation_between` with the inputs normalized first: the dot and perp products are then of
/// unit scale, so the result keeps about 4 ulp whatever the length of the inputs (instead of
/// `2 + 2/(|a|·|b|)`), at the price of two norms and four divisions.
#[inline(always)]
fn alt_rotation_between_normalized(a: Vector2<Fixed>, b: Vector2<Fixed>) -> UnitComplex<Fixed> {
    let na = Real::norm2(a.x, a.y);
    let nb = Real::norm2(b.x, b.y);
    if na == Real::ZERO || nb == Real::ZERO {
        return UnitComplexTrait::identity();
    }
    let (ax, ay) = (a.x / na, a.y / na);
    let (bx, by) = (b.x / nb, b.y / nb);
    let c = UnitComplex {
        re: Real::sum_prod2(ax, bx, ay, by), im: Real::diff_prod(ax, by, ay, bx),
    };
    c.renormalize_fast()
}

/// `renormalize_fast` written as upstream's literal `1/2 * (3 - |c|²)`: two roundings instead of
/// one for the same bits.
#[inline(always)]
fn alt_renormalize_fast_literal(c: UnitComplex<Fixed>) -> UnitComplex<Fixed> {
    let three = Real::<Fixed>::from_int(3);
    let f = Real::<Fixed>::HALF * (three - Real::norm_squared2(c.re, c.im));
    UnitComplex { re: c.re * f, im: c.im * f }
}

/// `renormalize_fast` as `c + c·(1 - |c|²)/2`, one `mul_add` per component.
#[inline(always)]
fn alt_renormalize_fast_mul_add(c: UnitComplex<Fixed>) -> UnitComplex<Fixed> {
    let g = Real::mul_add(Real::norm_squared2(c.re, c.im), -Real::<Fixed>::HALF, Real::HALF);
    UnitComplex { re: Real::mul_add(c.re, g, c.re), im: Real::mul_add(c.im, g, c.im) }
}

/// `renormalize_fast` as a `lerp` towards the corrected pair.
#[inline(always)]
fn alt_renormalize_fast_lerp(c: UnitComplex<Fixed>) -> UnitComplex<Fixed> {
    let f = Real::mul_add(
        Real::norm_squared2(c.re, c.im), -Real::<Fixed>::HALF, Real::HALF + Real::ONE,
    );
    UnitComplex {
        re: Real::lerp(Real::ZERO, c.re, f), im: Real::lerp(Real::<Fixed>::ZERO, c.im, f),
    }
}

/// `append_axisangle_linearized` closed with a Newton step instead of a division: cheaper, but
/// `|c|² - 1 = angle²` is outside the radius where one step converges (see the doc comment).
#[inline(always)]
fn alt_append_renormalize_fast(c: UnitComplex<Fixed>, angle: Fixed) -> UnitComplex<Fixed> {
    let p = UnitComplex {
        re: Real::mul_add(-c.im, angle, c.re), im: Real::mul_add(c.re, angle, c.im),
    };
    p.renormalize_fast()
}

/// The exact composition `c * new(angle)`: no linearization error at all, one `sin_cos`.
#[inline(always)]
fn alt_append_sin_cos(c: UnitComplex<Fixed>, angle: Fixed) -> UnitComplex<Fixed> {
    c * UnitComplexAngleTrait::new(angle)
}

// --- why the alternatives lost

#[test]
fn test_rotation_between_alt_atan2_agrees_on_long_vectors() {
    // Same rotation within the accuracy of atan2 + sin_cos, for 4.6x the gas.
    let (a, b) = (v2(0x180000000, -0x240000000), v2(-0x480000000, 0x40000000));
    let got = UnitComplexTrait::rotation_between(a, b);
    assert!(got == uc(-2576980378, -3435973837));
    assert!(alt_rotation_between_atan2(a, b).abs_diff_eq(got, 16));
}

#[test]
fn test_rotation_between_alt_normalized_wins_on_short_vectors() {
    // |a| = |b| ~ 0.0077: the algebraic form loses the bits of `a·b` and `a×b`, the normalized
    // one does not. The reference is the same directions 256 times longer.
    let (a, b) = (v2(0x30ec4a1, -0x41d5b92), v2(0x40a3d70, 0x2f1a9fc));
    let reference = UnitComplexTrait::rotation_between(
        v2(0x30ec4a100, -0x41d5b9200), v2(0x40a3d7000, 0x2f1a9fc00),
    );
    assert!(!UnitComplexTrait::rotation_between(a, b).abs_diff_eq(reference, 200));
    assert!(alt_rotation_between_normalized(a, b).abs_diff_eq(reference, 8));
}

#[test]
fn test_renormalize_fast_alternatives_give_the_same_bits() {
    let drifted = uc(3955930943, 1672543140);
    let got = drifted.renormalize_fast();
    assert!(alt_renormalize_fast_literal(drifted) == got);
    assert!(alt_renormalize_fast_lerp(drifted) == got);
    // `c + c·(1 - |c|²)/2` rounds the correction and the sum separately: 1 ulp apart.
    assert!(alt_renormalize_fast_mul_add(drifted).abs_diff_eq(got, 1));
}

#[test]
fn test_append_alt_renormalize_fast_leaves_a_large_norm_error() {
    // 0.01 rad: `|c|²` is off by `angle² = 1e-4`, and one Newton step leaves `3·angle⁴/4`.
    let (c, angle) = (c(), fx(0x28f5c29));
    let fast = alt_append_renormalize_fast(c, angle);
    let n = Real::norm_squared2(fast.re, fast.im);
    let e: i64 = n.raw - Real::<Fixed>::ONE.raw;
    assert!(e < 0 && e > -400);
    let exact = c.append_axisangle_linearized(angle);
    let n = Real::norm_squared2(exact.re, exact.im);
    assert!(Real::abs_diff_eq(n, Real::ONE, 2));
    // Both rotate by `atan(angle)` instead of `angle`, so they agree on the direction.
    assert!(fast.abs_diff_eq(exact, 400));
}

#[test]
fn test_append_alt_sin_cos_is_the_exact_rotation() {
    // The linearization is 1 500 ulp short of the true composition (`atan(θ) - θ = θ³/3`),
    // which buys it a `sin_cos`.
    let (c, angle) = (c(), fx(0x28f5c29));
    let exact = alt_append_sin_cos(c, angle);
    assert!(c.append_axisangle_linearized(angle).abs_diff_eq(exact, 1500));
    assert!(!c.append_axisangle_linearized(angle).abs_diff_eq(exact, 1000));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_unit_complex_identity__baseline() {
    let _a: Fixed = black_box(fx(0x66666666));
    let e: UnitComplex<Fixed> = black_box(uc(0x100000000, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_identity__const() {
    let _a: Fixed = black_box(fx(0x66666666));
    let e: UnitComplex<Fixed> = black_box(uc(0x100000000, 0));
    assert!(UnitComplexTrait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_identity__from_cos_sin_unchecked() {
    let a: Fixed = black_box(fx(0x66666666));
    let e: UnitComplex<Fixed> = black_box(uc(0x66666666, 0));
    assert!(UnitComplexTrait::from_cos_sin_unchecked(a, Real::ZERO) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_new__baseline() {
    let _a: Fixed = black_box(fx(0x66666666));
    let e: UnitComplex<Fixed> = black_box(c());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_new__sin_cos() {
    let a: Fixed = black_box(fx(0x66666666));
    let e: UnitComplex<Fixed> = black_box(c());
    assert!(UnitComplexAngleTrait::new(a) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_accessors__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let e: Vector2<Fixed> = black_box(v2(3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_accessors__complex() {
    let c: UnitComplex<Fixed> = black_box(c());
    let e: Vector2<Fixed> = black_box(v2(3955926847, 1672539044));
    assert!(c.complex() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_accessors__cos_sin_angle() {
    let c: UnitComplex<Fixed> = black_box(c());
    let e: Vector2<Fixed> = black_box(v2(3955926847, 1672539044));
    assert!(c.cos_angle() == e.x && c.sin_angle() == e.y);
    assert!(c.re() == e.x && c.im() == e.y);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let e: UnitComplex<Fixed> = black_box(uc(3955926847, -1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse__conjugate() {
    let c: UnitComplex<Fixed> = black_box(c());
    let e: UnitComplex<Fixed> = black_box(uc(3955926847, -1672539044));
    assert!(c.inverse() == e);
    assert!(c.conjugate() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_mul__baseline() {
    let _a: UnitComplex<Fixed> = black_box(c());
    let _b: UnitComplex<Fixed> = black_box(d());
    let e: UnitComplex<Fixed> = black_box(uc(4178578243, 993090093));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_mul__fused() {
    let a: UnitComplex<Fixed> = black_box(c());
    let b: UnitComplex<Fixed> = black_box(d());
    let e: UnitComplex<Fixed> = black_box(uc(4178578243, 993090093));
    assert!(a * b == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_to__baseline() {
    let _a: UnitComplex<Fixed> = black_box(c());
    let _b: UnitComplex<Fixed> = black_box(d());
    let e: UnitComplex<Fixed> = black_box(uc(3623642725, -2305636023));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_to__fused() {
    let a: UnitComplex<Fixed> = black_box(c());
    let b: UnitComplex<Fixed> = black_box(d());
    let e: UnitComplex<Fixed> = black_box(uc(3623642725, -2305636023));
    assert!(a.rotation_to(b) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_transform_vector__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(9697103119, -6392026840));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_transform_vector__fused() {
    let c: UnitComplex<Fixed> = black_box(c());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(9697103119, -6392026840));
    assert!(c.transform_vector(v) == e);
    assert!(c.mul_vec(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_transform_point__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(9697103119, -6392026840));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_transform_point__fused() {
    let c: UnitComplex<Fixed> = black_box(c());
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(9697103119, -6392026840));
    assert!(c.transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_vector__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let _v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(2170677421, -11409643972));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_vector__fused() {
    let c: UnitComplex<Fixed> = black_box(c());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(2170677421, -11409643972));
    assert!(c.inverse_transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_vector__alt_inverse_then_transform() {
    let c: UnitComplex<Fixed> = black_box(c());
    let v: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let e: Vector2<Fixed> = black_box(v2(2170677421, -11409643972));
    assert!(c.inverse().transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_point__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let _p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(2170677421, -11409643972));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_point__fused() {
    let c: UnitComplex<Fixed> = black_box(c());
    let p: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Point2<Fixed> = black_box(p2(2170677421, -11409643972));
    assert!(c.inverse_transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_to_rotation_matrix__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let e: Fixed = black_box(fx(-1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_to_rotation_matrix__expand() {
    let c: UnitComplex<Fixed> = black_box(c());
    let e: Fixed = black_box(fx(-1672539044));
    let r: Rotation2<Fixed> = c.to_rotation_matrix();
    assert!(r.matrix.m12 == e && r.matrix.m11 == c.re);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_rotation_matrix__baseline() {
    let _r: Rotation2<Fixed> = black_box(c().to_rotation_matrix());
    let e: UnitComplex<Fixed> = black_box(c());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_rotation_matrix__first_column() {
    let r: Rotation2<Fixed> = black_box(c().to_rotation_matrix());
    let e: UnitComplex<Fixed> = black_box(c());
    assert!(UnitComplexTrait::from_rotation_matrix(r) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_to_homogeneous__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let e: Fixed = black_box(fx(-1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_to_homogeneous__expand() {
    let c: UnitComplex<Fixed> = black_box(c());
    let e: Fixed = black_box(fx(-1672539044));
    let h = c.to_homogeneous();
    assert!(h.m12 == e && h.m33 == Real::ONE);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between__baseline() {
    let _a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(-2576980378, -3435973837));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between__algebraic() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(-2576980378, -3435973837));
    assert!(UnitComplexTrait::rotation_between(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between__alt_atan2() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(-2576980378, -3435973836));
    assert!(alt_rotation_between_atan2(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between__alt_normalized_inputs() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(-2576980378, -3435973837));
    assert!(alt_rotation_between_normalized(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_scaled_rotation_between__baseline() {
    let _a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let _b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let _s: Fixed = black_box(Real::HALF);
    let e: UnitComplex<Fixed> = black_box(uc(1920767767, -3841535534));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_scaled_rotation_between__atan2_sin_cos() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let s: Fixed = black_box(Real::HALF);
    let e: UnitComplex<Fixed> = black_box(uc(1920767767, -3841535534));
    assert!(UnitComplexAngleTrait::scaled_rotation_between(a, b, s) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_angle__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let e: Fixed = black_box(fx(1717986918));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_angle__atan2() {
    let c: UnitComplex<Fixed> = black_box(c());
    let e: Fixed = black_box(fx(1717986918));
    assert!(c.angle() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_angle_to__baseline() {
    let _a: UnitComplex<Fixed> = black_box(c());
    let _b: UnitComplex<Fixed> = black_box(d());
    let e: Fixed = black_box(fx(-2433814801));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_angle_to__fused_atan2() {
    let a: UnitComplex<Fixed> = black_box(c());
    let b: UnitComplex<Fixed> = black_box(d());
    let e: Fixed = black_box(fx(-2433814801));
    assert!(a.angle_to(b) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_powf__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let _n: Fixed = black_box(Real::TWO);
    let e: UnitComplex<Fixed> = black_box(uc(2992332532, 3081020949));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_powf__atan2_sin_cos() {
    let c: UnitComplex<Fixed> = black_box(c());
    let n: Fixed = black_box(Real::TWO);
    let e: UnitComplex<Fixed> = black_box(uc(2992332532, 3081020950));
    assert!(c.powf(n) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_slerp__baseline() {
    let _a: UnitComplex<Fixed> = black_box(c());
    let _b: UnitComplex<Fixed> = black_box(d());
    let _t: Fixed = black_box(fx(0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(4152447840, 1097233342));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_slerp__angle_then_compose() {
    let a: UnitComplex<Fixed> = black_box(c());
    let b: UnitComplex<Fixed> = black_box(d());
    let t: Fixed = black_box(fx(0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(4152447840, 1097233342));
    assert!(a.slerp(b, t) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize__baseline() {
    let _c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955925999, 1672541049));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize__norm_and_divisions() {
    let c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955926000, 1672541050));
    assert!(c.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize_fast__baseline() {
    let _c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955925998, 1672541049));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize_fast__mul_add() {
    let c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955925998, 1672541049));
    assert!(c.renormalize_fast() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize_fast__alt_literal() {
    let c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955925998, 1672541049));
    assert!(alt_renormalize_fast_literal(c) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize_fast__alt_mul_add_each() {
    let c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955925998, 1672541049));
    assert!(alt_renormalize_fast_mul_add(c) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize_fast__alt_lerp() {
    let c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955925998, 1672541049));
    assert!(alt_renormalize_fast_lerp(c) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_renormalize_fast__alt_exact() {
    let c: UnitComplex<Fixed> = black_box(uc(3955930943, 1672543140));
    let e: UnitComplex<Fixed> = black_box(uc(3955926000, 1672541050));
    assert!(c.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_append_axisangle_linearized__baseline() {
    let _c: UnitComplex<Fixed> = black_box(c());
    let _a: Fixed = black_box(fx(0x28f5c29));
    let e: UnitComplex<Fixed> = black_box(uc(3939004511, 1712012713));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_append_axisangle_linearized__renormalize() {
    let c: UnitComplex<Fixed> = black_box(c());
    let a: Fixed = black_box(fx(0x28f5c29));
    let e: UnitComplex<Fixed> = black_box(uc(3939004512, 1712012714));
    assert!(c.append_axisangle_linearized(a) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_append_axisangle_linearized__alt_renormalize_fast() {
    let c: UnitComplex<Fixed> = black_box(c());
    let a: Fixed = black_box(fx(0x28f5c29));
    let e: UnitComplex<Fixed> = black_box(uc(3939004496, 1712012707));
    assert!(alt_append_renormalize_fast(c, a) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_append_axisangle_linearized__alt_sin_cos() {
    let c: UnitComplex<Fixed> = black_box(c());
    let a: Fixed = black_box(fx(0x28f5c29));
    let e: UnitComplex<Fixed> = black_box(uc(3939003940, 1712014026));
    assert!(alt_append_sin_cos(c, a) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_abs_diff_eq__baseline() {
    let _a: UnitComplex<Fixed> = black_box(c());
    let _b: UnitComplex<Fixed> = black_box(d());
    let e: bool = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_abs_diff_eq__all_compared() {
    let a: UnitComplex<Fixed> = black_box(c());
    let b: UnitComplex<Fixed> = black_box(d());
    let e: bool = black_box(false);
    assert!(a.abs_diff_eq(b, 8) == e);
}
