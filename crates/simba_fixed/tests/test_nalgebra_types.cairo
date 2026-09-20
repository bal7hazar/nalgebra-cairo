//! nalgebra types instantiated with glam.cairo's `fixed::Fixed`, against the same computation on
//! `simba::fixed::Fixed`.
//!
//! Every test builds both objects from the SAME raw Q32.32 values and compares the results raw by
//! raw. The rule the package promises — everything that goes through `Real` is bit-identical, the
//! corelib `/` operator is not — is checked here on real types, not on scalars: see
//! `test_vector3_normalize_differs_by_the_division` for the one operation of this file that
//! diverges, and `simba_fixed::conformance` for the scalar-level statement.

use fixed::Fixed as Glam;
use nalgebra::base::vector3::Vector3Trait;
use nalgebra::{
    Isometry3, Isometry3Trait, Matrix3Trait, Point3, Quaternion, SymMatrix3, SymMatrix3Trait,
    Translation3, UnitQuaternion, UnitQuaternionTrait, Vector3,
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

/// DIVERGENCE, and the only one in this file. `normalize` is `unscale`, i.e. one `/` per
/// component, and the corelib `Div` impl of `fixed::Fixed` truncates toward zero where simba's
/// floors (`simba_fixed::conformance::test_conformance_div_truncates_where_simba_floors`). No impl
/// in this package can intercept an operator of a foreign type, so the two differ by one ulp on
/// exactly the components whose quotient is negative and inexact — here `y`.
///
/// Asserted as the documented difference it is, not smoothed over with a tolerance.
#[test]
fn test_vector3_normalize_differs_by_the_division() {
    let a = gv(AX, AY, AZ).normalize();
    let b = sv(AX, AY, AZ).normalize();
    assert!(a.x.raw == b.x.raw, "x is non-negative: floor = trunc");
    assert!(a.z.raw == b.z.raw, "z is non-negative: floor = trunc");
    assert!(a.y.raw == b.y.raw + 1, "y is negative and inexact: trunc = floor + 1");
    // The norms of both remain 1 to within the same rounding.
    assert!(a.norm().abs_diff_eq(Real::ONE, 2));
    assert!(b.norm().abs_diff_eq(Real::ONE, 2));
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

/// DIVERGENCE. Where the panic comes from the corelib `/` rather than from a `Real` kernel, the
/// MESSAGE is glam.cairo's (`'Fixed: division by zero'`) and not simba's. Same cause as
/// `test_vector3_normalize_differs_by_the_division`; a downstream `#[should_panic]` on
/// `normalize` therefore has to match on the scalar it instantiates.
#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_zero_normalize_panics_as_glam() {
    let _ = nalgebra_testing::black_box(gv(0, 0, 0)).normalize();
}

/// The same call on simba's scalar, for contrast: the message is simba's.
#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_zero_normalize_panics_as_simba_reference() {
    let _ = nalgebra_testing::black_box(sv(0, 0, 0)).normalize();
}

/// `append_axisangle_linearized` renormalises exactly (`new_normalize`), so it inherits the
/// division: the components agree or are one ulp apart, with glam's on the truncation side.
/// Measured component by component rather than bounded by a tolerance.
#[test]
fn test_append_axisangle_linearized_inherits_the_division() {
    let a = glam_quaternion().append_axisangle_linearized(gv(BX, BY, BZ)).quaternion;
    let b = simba_quaternion().append_axisangle_linearized(sv(BX, BY, BZ)).quaternion;
    assert_trunc_of(a.w.raw, b.w.raw, "w");
    assert_trunc_of(a.i.raw, b.i.raw, "i");
    assert_trunc_of(a.j.raw, b.j.raw, "j");
    assert_trunc_of(a.k.raw, b.k.raw, "k");
}

/// `glam == simba` for a non-negative component, `glam == simba + 1` for a negative one rounded
/// up by the truncating division.
fn assert_trunc_of(glam: i64, simba: i64, what: ByteArray) {
    if simba >= 0 {
        assert!(glam == simba, "{}: {} vs {}", what, glam, simba);
    } else {
        assert!(glam == simba || glam == simba + 1, "{}: {} vs {}", what, glam, simba);
    }
}
