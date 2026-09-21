//! nalgebra types instantiated with glam.cairo's `fixed::Fixed`, against the same computation on
//! `simba::fixed::Fixed`.
//!
//! Every test builds both objects from the SAME raw Q32.32 values and compares the results raw by
//! raw, with tolerance 0. The rule the package promises — nalgebra goes through `Real` for every
//! rounding operation, divisions included (`Real::div`), so everything is bit-identical — is
//! checked here on real types, not on scalars; `simba_fixed::conformance` has the scalar-level
//! statement. The `*_negative_inexact_is_bit_identical` tests pin the cases that differed by one
//! ulp while nalgebra still divided through the scalar's own (truncating) `/` operator.

use fixed::Fixed as Glam;
use nalgebra::base::vector3::Vector3Trait;
use nalgebra::{
    Isometry3, Isometry3Trait, Matrix3Trait, Point3, Quaternion, QuaternionTrait, SymMatrix3,
    SymMatrix3Trait, Translation3, UnitQuaternion, UnitQuaternionTrait, Vector3,
};
use simba::fixed::Fixed as Simba;
use simba_fixed::prelude::*;

#[inline(always)]
fn g(raw: i64) -> Glam {
    Glam { raw }
}

#[inline(always)]
fn s(raw: i64) -> Simba {
    Simba { raw }
}

fn gv(x: i64, y: i64, z: i64) -> Vector3<Glam> {
    Vector3 { x: g(x), y: g(y), z: g(z) }
}

fn sv(x: i64, y: i64, z: i64) -> Vector3<Simba> {
    Vector3 { x: s(x), y: s(y), z: s(z) }
}

fn assert_same_vector(a: Vector3<Glam>, b: Vector3<Simba>, what: ByteArray) {
    assert!(a.x.raw == b.x.raw, "{} x: {} vs {}", what, a.x.raw, b.x.raw);
    assert!(a.y.raw == b.y.raw, "{} y: {} vs {}", what, a.y.raw, b.y.raw);
    assert!(a.z.raw == b.z.raw, "{} z: {} vs {}", what, a.z.raw, b.z.raw);
}

fn assert_same_quaternion(a: Quaternion<Glam>, b: Quaternion<Simba>, what: ByteArray) {
    assert!(a.w.raw == b.w.raw, "{} w: {} vs {}", what, a.w.raw, b.w.raw);
    assert!(a.i.raw == b.i.raw, "{} i: {} vs {}", what, a.i.raw, b.i.raw);
    assert!(a.j.raw == b.j.raw, "{} j: {} vs {}", what, a.j.raw, b.j.raw);
    assert!(a.k.raw == b.k.raw, "{} k: {} vs {}", what, a.k.raw, b.k.raw);
}

// Raw operands shared by the tests below: two vectors, a rotation, a translation, a symmetric
// matrix. Arbitrary non-round values, so that every rounding is exercised.
const AX: i64 = 0x1_8000_0000; // 1.5
const AY: i64 = -0x2_4000_0000; // -2.25
const AZ: i64 = 0x0_C000_0000; // 0.75
const BX: i64 = -0x0_6000_0000; // -0.375
const BY: i64 = 0x3_2000_0000; // 3.125
const BZ: i64 = 0x1_1000_0000; // 1.0625
// Two quantised unit quaternions (norm 1 to within 1 ulp), the shape rapier stores and the shape
// the oracle generates: taken through `new_unchecked`, so that the comparison below measures the
// rotation kernels and not the division of `new_normalize` (which has its own test).
const QW: i64 = 3449105434;
const QI: i64 = 1293414537;
const QJ: i64 = -1724552719;
const QK: i64 = 1379642173;
const RW: i64 = 2157321895;
const RI: i64 = -2588786276;
const RJ: i64 = 2373054084;
const RK: i64 = -1208100262;

fn glam_quaternion() -> UnitQuaternion<Glam> {
    UnitQuaternionTrait::new_unchecked(Quaternion { i: g(QI), j: g(QJ), k: g(QK), w: g(QW) })
}

fn simba_quaternion() -> UnitQuaternion<Simba> {
    UnitQuaternionTrait::new_unchecked(Quaternion { i: s(QI), j: s(QJ), k: s(QK), w: s(QW) })
}

// --- Vector3 -------------------------------------------------------------------------------

/// `dot`, `cross`, `norm` and `norm_squared` go through `Real::sum_prod3` / `diff_prod` /
/// `norm3`, which are simba's kernels: bit-identical.
#[test]
fn test_vector3_dot_cross_norm_match_simba() {
    let (ga, gb) = (gv(AX, AY, AZ), gv(BX, BY, BZ));
    let (sa, sb) = (sv(AX, AY, AZ), sv(BX, BY, BZ));
    assert!(ga.dot(gb).raw == sa.dot(sb).raw, "dot");
    assert!(ga.norm().raw == sa.norm().raw, "norm");
    assert!(ga.norm_squared().raw == sa.norm_squared().raw, "norm_squared");
    assert_same_vector(ga.cross(gb), sa.cross(sb), "cross");
    assert_same_vector(ga + gb, sa + sb, "add");
    assert_same_vector(ga.scale(g(HALF_RAW)), sa.scale(s(HALF_RAW)), "scale");
    assert_same_vector(ga.lerp(gb, g(HALF_RAW)), sa.lerp(sb, s(HALF_RAW)), "lerp");
}

const HALF_RAW: i64 = 0x8000_0000;

/// Regression (WP 4.6). `normalize` is `unscale` by the norm, one `Real::div` per component, and
/// `Real::<fixed::Fixed>::div` is simba's floor: bit-identical, including `y`, whose quotient is
/// negative and inexact — the component on which glam's own truncating `/` gives one ulp more
/// (asserted too, so that the case keeps exercising the difference).
#[test]
fn test_normalize_negative_inexact_is_bit_identical() {
    let a = gv(AX, AY, AZ).normalize();
    let b = sv(AX, AY, AZ).normalize();
    assert_same_vector(a, b, "normalize");
    let n = gv(AX, AY, AZ).norm();
    assert!((g(AY) / n).raw == b.y.raw + 1, "glam's own `/` truncates y");
    let t = gv(AX, AY, AZ).try_normalize(Real::ZERO).unwrap();
    assert_same_vector(t, b, "try_normalize");
}

/// Regression (WP 4.6). `unscale` by a positive inexact divisor: bit-identical on the negative
/// components, where glam's own `/` truncates.
#[test]
fn test_unscale_negative_inexact_is_bit_identical() {
    let k = 0x7_0000_0000; // 7: -2.25 / 7 and -0.375 / 7 are inexact
    let a = gv(AX, AY, BX).unscale(g(k));
    let b = sv(AX, AY, BX).unscale(s(k));
    assert_same_vector(a, b, "unscale");
    assert!((g(AY) / g(k)).raw == b.y.raw + 1, "glam's own `/` truncates y");
    assert!((g(BX) / g(k)).raw == b.z.raw + 1, "glam's own `/` truncates z");
    let mut c = gv(AX, AY, BX);
    c /= g(k);
    assert_same_vector(c, b, "/=");
}

/// Regression (WP 4.6). `UnitQuaternion::new_normalize` on a non-unit quaternion with negative
/// components: bit-identical, where glam's own `/` used to round the negative ones up.
#[test]
fn test_new_normalize_negative_inexact_is_bit_identical() {
    let gq: UnitQuaternion<Glam> = UnitQuaternionTrait::new_normalize(
        Quaternion { i: g(AX), j: g(AY), k: g(BX), w: g(BY) },
    );
    let sq: UnitQuaternion<Simba> = UnitQuaternionTrait::new_normalize(
        Quaternion { i: s(AX), j: s(AY), k: s(BX), w: s(BY) },
    );
    assert_same_quaternion(gq.quaternion, sq.quaternion, "new_normalize");
    let n = Quaternion { i: g(AX), j: g(AY), k: g(BX), w: g(BY) }.norm();
    assert!((g(AY) / n).raw == sq.quaternion.j.raw + 1, "glam's own `/` truncates j");
}

// --- SymMatrix3 ----------------------------------------------------------------------------

/// `quadform` (`R diag(d) Rᵀ`) and `try_inverse` are pure `Real` kernels (fused products and one
/// `recip` of the determinant): bit-identical.
#[test]
fn test_sym_matrix3_quadform_and_inverse_match_simba() {
    let gr = Matrix3Trait::new(g(AX), g(AY), g(AZ), g(BX), g(BY), g(BZ), g(AZ), g(BX), g(AY));
    let sr = Matrix3Trait::new(s(AX), s(AY), s(AZ), s(BX), s(BY), s(BZ), s(AZ), s(BX), s(AY));
    let gq: SymMatrix3<Glam> = SymMatrix3Trait::quadform(gr, gv(AX, BY, BZ));
    let sq: SymMatrix3<Simba> = SymMatrix3Trait::quadform(sr, sv(AX, BY, BZ));
    assert!(gq.m11.raw == sq.m11.raw && gq.m12.raw == sq.m12.raw && gq.m13.raw == sq.m13.raw);
    assert!(gq.m22.raw == sq.m22.raw && gq.m23.raw == sq.m23.raw && gq.m33.raw == sq.m33.raw);

    let gi = gq.try_inverse().unwrap();
    let si = sq.try_inverse().unwrap();
    assert!(gi.m11.raw == si.m11.raw && gi.m12.raw == si.m12.raw && gi.m13.raw == si.m13.raw);
    assert!(gi.m22.raw == si.m22.raw && gi.m23.raw == si.m23.raw && gi.m33.raw == si.m33.raw);
}

// --- UnitQuaternion ------------------------------------------------------------------------

/// The Hamilton product, `transform_vector`, `append_axisangle_linearized` and
/// `renormalize_fast` are `Real` kernels end to end (wide accumulation, `inv_sqrt`, no `/`):
/// bit-identical.
#[test]
fn test_unit_quaternion_matches_simba() {
    let (gq, sq) = (glam_quaternion(), simba_quaternion());
    let go = UnitQuaternionTrait::new_unchecked(
        Quaternion { i: g(RI), j: g(RJ), k: g(RK), w: g(RW) },
    );
    let so = UnitQuaternionTrait::new_unchecked(
        Quaternion { i: s(RI), j: s(RJ), k: s(RK), w: s(RW) },
    );
    assert_same_quaternion((gq * go).quaternion, (sq * so).quaternion, "product");
    assert_same_vector(
        gq.transform_vector(gv(AX, AY, AZ)),
        sq.transform_vector(sv(AX, AY, AZ)),
        "transform_vector",
    );
    assert_same_vector(
        gq.inverse_transform_vector(gv(AX, AY, AZ)),
        sq.inverse_transform_vector(sv(AX, AY, AZ)),
        "inverse_transform_vector",
    );
    assert_same_quaternion(
        gq.renormalize_fast().quaternion, sq.renormalize_fast().quaternion, "renormalize_fast",
    );
}

// --- Isometry3 -----------------------------------------------------------------------------

/// `transform_point` (the fused `rotate_translate`) and `inv_mul` are `Real` kernels:
/// bit-identical.
#[test]
fn test_isometry3_matches_simba() {
    let gi = Isometry3Trait::from_parts(Translation3 { vector: gv(BX, BY, BZ) }, glam_quaternion());
    let si = Isometry3Trait::from_parts(
        Translation3 { vector: sv(BX, BY, BZ) }, simba_quaternion(),
    );
    let gp = gi.transform_point(Point3 { x: g(AX), y: g(AY), z: g(AZ) });
    let sp = si.transform_point(Point3 { x: s(AX), y: s(AY), z: s(AZ) });
    assert!(
        gp.x.raw == sp.x.raw && gp.y.raw == sp.y.raw && gp.z.raw == sp.z.raw, "transform_point",
    );

    let go = Isometry3Trait::from_parts(
        Translation3 { vector: gv(AX, AY, AZ) },
        UnitQuaternionTrait::new_unchecked(Quaternion { i: g(RI), j: g(RJ), k: g(RK), w: g(RW) }),
    );
    let so = Isometry3Trait::from_parts(
        Translation3 { vector: sv(AX, AY, AZ) },
        UnitQuaternionTrait::new_unchecked(Quaternion { i: s(RI), j: s(RJ), k: s(RK), w: s(RW) }),
    );
    let gm: Isometry3<Glam> = gi.inv_mul(go);
    let sm: Isometry3<Simba> = si.inv_mul(so);
    assert_same_quaternion(gm.rotation.quaternion, sm.rotation.quaternion, "inv_mul rotation");
    assert_same_vector(gm.translation.vector, sm.translation.vector, "inv_mul translation");
}

// --- overflow ------------------------------------------------------------------------------

/// Overflow inside a nalgebra kernel panics with SIMBA's stable message even though the scalar is
/// glam.cairo's, because the kernel is simba's: the overflow semantics carry over unchanged.
#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_overflow_in_a_kernel_panics_as_simba() {
    let huge = nalgebra_testing::black_box(gv(0x40_0000_0000_0000, 0, 0));
    let _ = huge.dot(huge);
}

/// The same overflow, with the same message, on the simba scalar: the two really do share the
/// panic, which is what lets rapier.cairo keep its `#[should_panic]` tests.
#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_overflow_in_a_kernel_panics_as_simba_reference() {
    let huge = nalgebra_testing::black_box(sv(0x40_0000_0000_0000, 0, 0));
    let _ = huge.dot(huge);
}

/// The division of `normalize` is `Real::div`, simba's kernel, so normalising a zero vector
/// panics with SIMBA's message on glam's scalar too (it was glam.cairo's `'Fixed: division by
/// zero'` while nalgebra used the scalar's own `/`): one `#[should_panic]` fits both scalars.
#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_zero_normalize_panics_as_simba() {
    let _ = nalgebra_testing::black_box(gv(0, 0, 0)).normalize();
}

/// The same call on simba's scalar, for contrast: the message is simba's.
#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_zero_normalize_panics_as_simba_reference() {
    let _ = nalgebra_testing::black_box(sv(0, 0, 0)).normalize();
}

/// `append_axisangle_linearized` renormalises exactly (`new_normalize`, i.e. `Real::div`):
/// bit-identical, component by component.
#[test]
fn test_append_axisangle_linearized_matches_simba() {
    let a = glam_quaternion().append_axisangle_linearized(gv(BX, BY, BZ)).quaternion;
    let b = simba_quaternion().append_axisangle_linearized(sv(BX, BY, BZ)).quaternion;
    assert_same_quaternion(a, b, "append_axisangle_linearized");
}
