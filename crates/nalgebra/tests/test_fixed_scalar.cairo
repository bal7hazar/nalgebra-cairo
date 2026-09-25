//! nalgebra types on the stack's scalar, fixed-cairo's `fixed::Fixed`, as a downstream crate uses
//! them (formerly the integration tests of `simba_fixed`).
//!
//! The kernels that map one-to-one onto a `fixed` call are checked against that call (`dot` is
//! `fixed::wide::dot3`, `normalize` divides by the norm with `fixed`'s `/` (to nearest), ...); the
//! composite ones (the quadratic form of `recompose`, the Hamilton product, isometries) are pinned
//! to their raw values.
//! Panics carry `fixed`'s messages.

use fixed::{Fixed, wide};
use nalgebra::base::vector3::Vector3Trait;
use nalgebra::{
    Isometry3, Isometry3Trait, Matrix3Trait, Point3, Quaternion, QuaternionTrait, SymmetricEigen3,
    SymmetricEigen3Trait, Translation3, UnitQuaternion, UnitQuaternionTrait, Vector3,
};
use simba::prelude::*;

#[inline(always)]
fn g(raw: i64) -> Fixed {
    Fixed { raw }
}

fn gv(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: g(x), y: g(y), z: g(z) }
}

fn assert_vector(a: Vector3<Fixed>, e: (i64, i64, i64), what: ByteArray) {
    let (x, y, z) = e;
    assert!(
        a.x.raw == x && a.y.raw == y && a.z.raw == z,
        "{} {} {} {}",
        what,
        a.x.raw,
        a.y.raw,
        a.z.raw,
    );
}

fn assert_quaternion(a: Quaternion<Fixed>, e: (i64, i64, i64, i64), what: ByteArray) {
    let (w, i, j, k) = e;
    assert!(
        a.w.raw == w && a.i.raw == i && a.j.raw == j && a.k.raw == k,
        "{} {} {} {} {}",
        what,
        a.w.raw,
        a.i.raw,
        a.j.raw,
        a.k.raw,
    );
}

// Raw operands shared by the tests below: two vectors, two rotations, a translation. Arbitrary
// non-round values, so that every rounding is exercised.
const AX: i64 = 0x1_8000_0000; // 1.5
const AY: i64 = -0x2_4000_0000; // -2.25
const AZ: i64 = 0x0_C000_0000; // 0.75
const BX: i64 = -0x0_6000_0000; // -0.375
const BY: i64 = 0x3_2000_0000; // 3.125
const BZ: i64 = 0x1_1000_0000; // 1.0625
const HALF_RAW: i64 = 0x8000_0000;
// Two quantised unit quaternions (norm 1 to within 1 ulp), the shape rapier stores and the shape
// the oracle generates: taken through `new_unchecked`, so that the tests measure the rotation
// kernels and not the division of `new_normalize` (which has its own test).
const QW: i64 = 3449105434;
const QI: i64 = 1293414537;
const QJ: i64 = -1724552719;
const QK: i64 = 1379642173;
const RW: i64 = 2157321895;
const RI: i64 = -2588786276;
const RJ: i64 = 2373054084;
const RK: i64 = -1208100262;

fn q() -> UnitQuaternion<Fixed> {
    UnitQuaternionTrait::new_unchecked(Quaternion { i: g(QI), j: g(QJ), k: g(QK), w: g(QW) })
}

fn r() -> UnitQuaternion<Fixed> {
    UnitQuaternionTrait::new_unchecked(Quaternion { i: g(RI), j: g(RJ), k: g(RK), w: g(RW) })
}

// --- Vector3 -------------------------------------------------------------------------------

/// `dot`, `cross`, `norm`, `norm_squared`, `lerp` are `fixed::wide` kernels, one per output.
#[test]
fn test_vector3_kernels_are_fixeds() {
    let (a, b) = (gv(AX, AY, AZ), gv(BX, BY, BZ));
    assert!(a.dot(b) == wide::dot3(a.x, b.x, a.y, b.y, a.z, b.z), "dot");
    assert!(a.norm() == wide::norm3(a.x, a.y, a.z), "norm");
    assert!(a.norm_squared() == wide::norm3_squared(a.x, a.y, a.z), "norm_squared");
    let c = a.cross(b);
    assert!(c.x == wide::mul_sub(a.y, b.z, a.z, b.y), "cross x");
    assert!(c.y == wide::mul_sub(a.z, b.x, a.x, b.z), "cross y");
    assert!(c.z == wide::mul_sub(a.x, b.y, a.y, b.x), "cross z");
    assert!(a + b == gv(AX + BX, AY + BY, AZ + BZ), "add");
    assert!(a.scale(g(HALF_RAW)) == gv(AX / 2, AY / 2, AZ / 2), "scale");
    let l = a.lerp(b, g(HALF_RAW));
    assert!(l.y == fixed::FixedTrait::lerp(a.y, b.y, g(HALF_RAW)), "lerp");
}

/// `normalize` is `unscale` by the norm, one `Real::div` per component: `fixed`'s `/`,
/// rounded to nearest, including on the negative inexact `y`.
#[test]
fn test_vector3_normalize_divides_by_the_norm() {
    let a = gv(AX, AY, AZ);
    let n = a.norm();
    let u = a.normalize();
    assert!(u == Vector3 { x: a.x / n, y: a.y / n, z: a.z / n }, "normalize");
    assert_vector(u, (2295756587, -3443634881, 1147878294), "normalize");
    assert!(a.try_normalize(Real::zero()).unwrap() == u, "try_normalize");
}

/// `unscale` by a positive inexact divisor, and `/=`: `fixed`'s `/`, rounded to nearest.
#[test]
fn test_vector3_unscale_rounds_to_nearest() {
    let k = g(0x7_0000_0000); // 7: -2.25 / 7 and -0.375 / 7 are inexact
    let a = gv(AX, AY, BX).unscale(k);
    assert_vector(a, (920350135, -1380525202, -230087534), "unscale");
    let mut c = gv(AX, AY, BX);
    c /= k;
    assert!(c == a, "/=");
}

// --- UnitQuaternion / Quaternion -------------------------------------------------------------

/// `new_normalize` on a non-unit quaternion with negative components: each component divided by
/// the norm with `fixed`'s `/`.
#[test]
fn test_unit_quaternion_new_normalize() {
    let raw = Quaternion { i: g(AX), j: g(AY), k: g(BX), w: g(BY) };
    let n = raw.norm();
    let u: UnitQuaternion<Fixed> = UnitQuaternionTrait::new_normalize(raw);
    let e = u.quaternion;
    assert!(e.i == g(AX) / n && e.j == g(AY) / n && e.k == g(BX) / n && e.w == g(BY) / n);
}

/// The Hamilton product, `transform_vector`, `inverse_transform_vector`, `renormalize_fast` and
/// `append_axisangle_linearized` (fused `Real` kernels end to end), pinned.
#[test]
fn test_unit_quaternion_kernels() {
    assert_quaternion(
        (q() * r()).quaternion, (3852976350, -1706466595, 571709618, -602027559), "product",
    );
    assert_vector(
        q().transform_vector(gv(AX, AY, AZ)),
        (8904137148, -6539827782, 4818205456),
        "transform_vector",
    );
    assert_vector(
        q().inverse_transform_vector(gv(AX, AY, AZ)),
        (3087494630, -10071360742, 5856891605),
        "inverse_transform_vector",
    );
    let mut renormalized = q();
    renormalized.renormalize_fast();
    assert_quaternion(
        renormalized.quaternion,
        (3449105434, 1293414537, -1724552720, 1379642173),
        "renormalize_fast",
    );
    assert_quaternion(
        q().append_axisangle_linearized(gv(BX, BY, BZ)).quaternion,
        (2915936134, 1918013280, 2378058494, 781103959),
        "append_axisangle_linearized",
    );
}

// --- SymmetricEigen3::recompose --------------------------------------------------------------

/// `SymmetricEigen3::recompose` (`R diag(d) Rᵀ` through the crate-internal structured quadratic
/// form, fused products) and `Matrix3::try_inverse` of the result, pinned (the same raw values as
/// the former public `SymMatrix3::quadform` / `try_inverse`, which were bit-identical to them).
#[test]
fn test_recompose_quadform_and_inverse() {
    let m = Matrix3Trait::new(g(AX), g(AY), g(AZ), g(BX), g(BY), g(BZ), g(AZ), g(BX), g(AY));
    let e = SymmetricEigen3 { eigenvalues: gv(AX, BY, BZ), eigenvectors: m };
    let s = e.recompose();
    assert!(s == s.transpose(), "recompose is symmetric");
    assert_vector(
        gv(s.m11.raw, s.m12.raw, s.m13.raw),
        (85010153472, -94359257088, 10871635968),
        "quadform row 1",
    );
    assert_vector(
        gv(s.m22.raw, s.m23.raw, s.m33.raw),
        (137129623552, -28449964032, 28613541888),
        "quadform rows 2-3",
    );
    let i = Matrix3Trait::try_inverse(s).unwrap();
    assert_vector(
        gv(i.m11.raw, i.m12.raw, i.m13.raw), (1101637383, 845641358, 422242925), "inverse row 1",
    );
    assert_vector(
        gv(i.m22.raw, i.m23.raw, i.m33.raw), (818614500, 492635554, 974075024), "inverse rows 2-3",
    );
}

// --- Isometry3 -----------------------------------------------------------------------------

/// `transform_point` (the fused `rotate_translate`) and `inv_mul` (the fused `conj_mul`), pinned.
#[test]
fn test_isometry3_kernels() {
    let a = Isometry3Trait::from_parts(Translation3 { vector: gv(BX, BY, BZ) }, q());
    let p = a.transform_point(Point3 { x: g(AX), y: g(AY), z: g(AZ) });
    assert_vector(
        gv(p.x.raw, p.y.raw, p.z.raw), (7293524412, 6881945018, 9381608208), "transform_point",
    );
    let b = Isometry3Trait::from_parts(Translation3 { vector: gv(AX, AY, AZ) }, r());
    let m: Isometry3<Fixed> = a.inv_mul(b);
    assert_quaternion(
        m.rotation.quaternion,
        (-388069562, -2451421556, 3239687848, -1338320245),
        "inv_mul rotation",
    );
    assert_vector(
        m.translation.vector, (-3658122833, -20539369763, 12819659395), "inv_mul translation",
    );
}

// --- panics: `fixed`'s messages ------------------------------------------------------------

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_overflow_in_a_kernel_panics() {
    let huge = nalgebra_testing::black_box(gv(0x40_0000_0000_0000, 0, 0));
    let _ = huge.dot(huge);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_zero_normalize_panics() {
    let _ = nalgebra_testing::black_box(gv(0, 0, 0)).normalize();
}
