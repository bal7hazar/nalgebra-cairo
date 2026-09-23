//! Gas benchmarks of `Isometry2` (`bench_isometry2_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together with
//! the tests showing why (AGENTS.md rule 8).
//!
//! The isometries used are `a = new((1.5, -2.25), 0.4 rad)` and `b = new((-0.75, 0.5), -1/6 rad)`
//! (hardcoded, so that the `sin_cos` of their construction is not measured), the point / vector
//! `(-2.5, 3.75)`, the translation `(1.25, -0.375)` and the rotation `UnitComplex::new(0.1)`.
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{fx, iso2, p2, uc, v2};
use crate::base::point2::{Point2, Point2Trait};
use crate::base::vector2::Vector2;
use crate::geometry::translation2::Translation2;
use crate::geometry::unit_complex::{UnitComplex, UnitComplexTrait};
use super::{Isometry2, Isometry2AngleTrait, Isometry2Trait};

/// `new((1.5, -2.25), 0.4 rad)`.
fn a() -> Isometry2<Fixed> {
    iso2(6442450944, -9663676416, 3955926847, 1672539044)
}

/// `new((-0.75, 0.5), -1/6 rad)`.
fn b() -> Isometry2<Fixed> {
    iso2(-3221225472, 2147483648, 4235452929, -712518464)
}

/// `(-2.5, 3.75)`, as a point and as a vector.
fn p() -> Point2<Fixed> {
    p2(-0x280000000, 0x3c0000000)
}

fn v() -> Vector2<Fixed> {
    v2(-0x280000000, 0x3c0000000)
}

/// `(1.25, -0.375)`.
fn t() -> Translation2<Fixed> {
    Translation2 { vector: v2(0x140000000, -0x60000000) }
}

/// `UnitComplex::new(0.1)`.
fn r() -> UnitComplex<Fixed> {
    uc(4273510349, 428781260)
}

// --- alternative implementations (losers)

/// The LOSER of `transform_point`: upstream's literal `rotation * p + translation`, i.e. two
/// fused kernels and then two `Fixed` additions, instead of the single wide accumulation of
/// `rotate_translate`. Bit for bit the same result (adding an integral number of raw units
/// commutes with the floor), 1.29x the gas: every `Fixed` addition pays an overflow check.
#[inline(always)]
fn alt_transform_point_rotate_then_add(i: Isometry2<Fixed>, q: Point2<Fixed>) -> Point2<Fixed> {
    let c = i.rotation.transform_point(q);
    Point2 { x: c.x + i.translation.vector.x, y: c.y + i.translation.vector.y }
}

// --- why the alternatives lost

/// `inv_mul` and `self.inverse() * other` are the same transform, but the second rounds the
/// intermediate `rotation⁻¹ · (-translation)` before adding the rotated translation of `other`:
/// 1 ulp apart here, and 1.31x the gas.
#[test]
fn test_inv_mul_alt_inverse_then_mul_differs_by_rounding() {
    let (x, y) = (a(), b());
    let got = x.inv_mul(y);
    let alt = x.inverse() * y;
    assert!(got.rotation == alt.rotation);
    assert!(got.translation != alt.translation);
    assert!(got.abs_diff_eq(alt, 1));
}

/// The fully fused `transform_point` gives exactly the same bits as rotating and then adding,
/// because the translation is an integral number of raw units: `floor(x + t) = floor(x) + t`.
#[test]
fn test_transform_point_fused_and_composed_agree_bit_for_bit() {
    let (x, q) = (a(), p());
    assert!(x.transform_point(q) == alt_transform_point_rotate_then_add(x, q));
    // ... and the homogeneous 3x3 product, which is the same fused form, agrees too.
    let h = x.to_homogeneous().mul_vec(q.to_homogeneous());
    assert!(h.x == x.transform_point(q).x && h.y == x.transform_point(q).y);
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_isometry2_identity__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let e: Isometry2<Fixed> = black_box(iso2(0, 0, 0x100000000, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_identity__const() {
    let _x: Isometry2<Fixed> = black_box(a());
    let e: Isometry2<Fixed> = black_box(iso2(0, 0, 0x100000000, 0));
    assert!(Isometry2Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_from_parts__baseline() {
    let _t: Translation2<Fixed> = black_box(a().translation);
    let _r: UnitComplex<Fixed> = black_box(a().rotation);
    let e: Isometry2<Fixed> = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_from_parts__wrap() {
    let t: Translation2<Fixed> = black_box(a().translation);
    let r: UnitComplex<Fixed> = black_box(a().rotation);
    let e: Isometry2<Fixed> = black_box(a());
    assert!(Isometry2Trait::from_parts(t, r) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_translation__baseline() {
    let _v: Vector2<Fixed> = black_box(t().vector);
    let e: Isometry2<Fixed> = black_box(iso2(0x140000000, -0x60000000, 0x100000000, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_translation__pure() {
    let v: Vector2<Fixed> = black_box(t().vector);
    let e: Isometry2<Fixed> = black_box(iso2(0x140000000, -0x60000000, 0x100000000, 0));
    assert!(Isometry2Trait::translation(v.x, v.y) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_from_translation__baseline() {
    let _u: Translation2<Fixed> = black_box(t());
    let e: Isometry2<Fixed> = black_box(iso2(0x140000000, -0x60000000, 0x100000000, 0));
    assert!(e == e);
}

/// `Isometry2::from_translation(t)`, which is also `t.into()` (the same code).
#[test]
#[inline(never)]
fn bench_isometry2_from_translation__pure() {
    let u: Translation2<Fixed> = black_box(t());
    let e: Isometry2<Fixed> = black_box(iso2(0x140000000, -0x60000000, 0x100000000, 0));
    assert!(Isometry2Trait::from_translation(u) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_from_rotation__baseline() {
    let _q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(0, 0, 4273510349, 428781260));
    assert!(e == e);
}

/// `Isometry2::from_rotation(r)`, which is also `r.into()` (the same code).
#[test]
#[inline(never)]
fn bench_isometry2_from_rotation__pure() {
    let q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(0, 0, 4273510349, 428781260));
    assert!(Isometry2Trait::from_rotation(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_new__baseline() {
    let _v: Vector2<Fixed> = black_box(v2(6442450944, -9663676416));
    let _angle: Fixed = black_box(fx(0x66666666));
    let e: Isometry2<Fixed> = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_new__sin_cos() {
    let v: Vector2<Fixed> = black_box(v2(6442450944, -9663676416));
    let angle: Fixed = black_box(fx(0x66666666));
    let e: Isometry2<Fixed> = black_box(a());
    assert!(Isometry2AngleTrait::new(v, angle) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_rotation__baseline() {
    let _angle: Fixed = black_box(fx(0x66666666));
    let e: Isometry2<Fixed> = black_box(iso2(0, 0, 3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_rotation__sin_cos() {
    let angle: Fixed = black_box(fx(0x66666666));
    let e: Isometry2<Fixed> = black_box(iso2(0, 0, 3955926847, 1672539044));
    assert!(Isometry2AngleTrait::rotation(angle) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inverse__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let e: Isometry2<Fixed> = black_box(iso2(-2170677422, 11409643971, 3955926847, -1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inverse__conjugate_rotate() {
    let x: Isometry2<Fixed> = black_box(a());
    let e: Isometry2<Fixed> = black_box(iso2(-2170677422, 11409643971, 3955926847, -1672539044));
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inv_mul__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _y: Isometry2<Fixed> = black_box(b());
    let e: Isometry2<Fixed> = black_box(iso2(-4301353035, 14642011678, 3623642725, -2305636023));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inv_mul__direct() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(b());
    let e: Isometry2<Fixed> = black_box(iso2(-4301353035, 14642011678, 3623642725, -2305636023));
    assert!(x.inv_mul(y) == e);
}

/// The loser: the inverse is materialised, so its translation is rotated once for nothing and
/// rounded before the composition (1 ulp apart, see the test above).
#[test]
#[inline(never)]
fn bench_isometry2_inv_mul__alt_inverse_then_mul() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(b());
    let e: Isometry2<Fixed> = black_box(iso2(-4301353036, 14642011677, 3623642725, -2305636023));
    assert!(x.inverse() * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_mul__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _y: Isometry2<Fixed> = black_box(b());
    let e: Isometry2<Fixed> = black_box(iso2(2639236286, -8940117276, 4178578243, 993090093));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_mul__compose() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(b());
    let e: Isometry2<Fixed> = black_box(iso2(2639236286, -8940117276, 4178578243, 993090093));
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_transform_point__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-9719387589, 989701650));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_transform_point__fused() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-9719387589, 989701650));
    assert!(x.transform_point(q) == e);
}

/// The loser: `rotation * p` and then two `Fixed` additions (upstream's literal form). The same
/// bits (see the test above), 1.29x the gas.
#[test]
#[inline(never)]
fn bench_isometry2_transform_point__alt_rotate_then_add() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-9719387589, 989701650));
    assert!(alt_transform_point_rotate_then_add(x, q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_transform_vector__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _w: Vector2<Fixed> = black_box(v());
    let e: Vector2<Fixed> = black_box(v2(-16161838533, 10653378066));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_transform_vector__rotate() {
    let x: Isometry2<Fixed> = black_box(a());
    let w: Vector2<Fixed> = black_box(v());
    let e: Vector2<Fixed> = black_box(v2(-16161838533, 10653378066));
    assert!(x.transform_vector(w) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inverse_transform_point__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-5788473124, 30425717258));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inverse_transform_point__subtract_rotate() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(p2(-5788473124, 30425717258));
    assert!(x.inverse_transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inverse_transform_vector__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _w: Vector2<Fixed> = black_box(v());
    let e: Vector2<Fixed> = black_box(v2(-3617795703, 19016073286));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_inverse_transform_vector__conjugate_rotate() {
    let x: Isometry2<Fixed> = black_box(a());
    let w: Vector2<Fixed> = black_box(v());
    let e: Vector2<Fixed> = black_box(v2(-3617795703, 19016073286));
    assert!(x.inverse_transform_vector(w) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_translation__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _u: Translation2<Fixed> = black_box(t());
    let e: Isometry2<Fixed> = black_box(iso2(11811160064, -11274289152, 3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_translation__add() {
    let x: Isometry2<Fixed> = black_box(a());
    let u: Translation2<Fixed> = black_box(t());
    let e: Isometry2<Fixed> = black_box(iso2(11811160064, -11274289152, 3955926847, 1672539044));
    assert!(x.append_translation(u) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_prepend_translation__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _u: Translation2<Fixed> = black_box(t());
    let e: Isometry2<Fixed> = black_box(iso2(12014561644, -9056475179, 3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_prepend_translation__rotate_add() {
    let x: Isometry2<Fixed> = black_box(a());
    let u: Translation2<Fixed> = black_box(t());
    let e: Isometry2<Fixed> = black_box(iso2(12014561644, -9056475179, 3955926847, 1672539044));
    assert!(x.prepend_translation(u) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_rotation__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(7375023358, -8972226396, 3769188402, 2059117008));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_rotation__compose_rotate() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(7375023358, -8972226396, 3769188402, 2059117008));
    assert!(x.append_rotation(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_prepend_rotation__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3769188402, 2059117008));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_prepend_rotation__compose() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3769188402, 2059117008));
    assert!(x.prepend_rotation(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_rotation_wrt_point__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _q: UnitComplex<Fixed> = black_box(r());
    let _c: Point2<Fixed> = black_box(p());
    let e: Isometry2<Fixed> = black_box(iso2(8929310716, -7819809694, 3769188402, 2059117008));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_rotation_wrt_point__shift_rotate() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: UnitComplex<Fixed> = black_box(r());
    let c: Point2<Fixed> = black_box(p());
    let e: Isometry2<Fixed> = black_box(iso2(8929310716, -7819809694, 3769188402, 2059117008));
    assert!(x.append_rotation_wrt_point(q, c) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_rotation_wrt_center__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3769188402, 2059117008));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_append_rotation_wrt_center__compose() {
    let x: Isometry2<Fixed> = black_box(a());
    let q: UnitComplex<Fixed> = black_box(r());
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3769188402, 2059117008));
    assert!(x.append_rotation_wrt_center(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_to_homogeneous__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(3955926847));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_to_homogeneous__matrix3() {
    let x: Isometry2<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(3955926847));
    let m = x.to_homogeneous();
    assert!(m.m11 == e && m.m13 == fx(6442450944));
}

#[test]
#[inline(never)]
fn bench_isometry2_renormalize__baseline() {
    let _x: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926850, 1672539042));
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926849, 1672539041));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_renormalize__exact() {
    let x: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926850, 1672539042));
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926849, 1672539041));
    assert!(x.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_renormalize_fast__baseline() {
    let _x: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926850, 1672539042));
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926848, 1672539041));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_renormalize_fast__newton() {
    let x: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926850, 1672539042));
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926848, 1672539041));
    assert!(x.renormalize_fast() == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_lerp_slerp__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _y: Isometry2<Fixed> = black_box(b());
    let _s: Fixed = black_box(fx(0x40000000));
    let e: Isometry2<Fixed> = black_box(iso2(4026531840, -6710886400, 4152447840, 1097233342));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_lerp_slerp__atan2_sin_cos() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(b());
    let s: Fixed = black_box(fx(0x40000000));
    let e: Isometry2<Fixed> = black_box(iso2(4026531840, -6710886400, 4152447840, 1097233342));
    assert!(x.lerp_slerp(y, s) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_lerp_nlerp__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _y: Isometry2<Fixed> = black_box(b());
    let _s: Fixed = black_box(fx(0x40000000));
    let e: Isometry2<Fixed> = black_box(iso2(4026531840, -6710886400, 4149247214, 1109275270));
    assert!(e == e);
}

/// The trigonometry-free interpolation: same path, cheaper parametrisation.
#[test]
#[inline(never)]
fn bench_isometry2_lerp_nlerp__lerp_normalize() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(b());
    let s: Fixed = black_box(fx(0x40000000));
    let e: Isometry2<Fixed> = black_box(iso2(4026531840, -6710886400, 4149247214, 1109275270));
    assert!(x.lerp_nlerp(y, s) == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_abs_diff_eq__baseline() {
    let _x: Isometry2<Fixed> = black_box(a());
    let _y: Isometry2<Fixed> = black_box(iso2(6442450946, -9663676415, 3955926848, 1672539044));
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_abs_diff_eq__ulps() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(iso2(6442450946, -9663676415, 3955926848, 1672539044));
    let e: bool = black_box(true);
    assert!(x.abs_diff_eq(y, 2) == e);
}
