//! Gas benchmarks of `UnitQuaternion` (`bench_unit_quaternion_<op>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementations that lost (`alt_*`), kept as
//! evidence together with the tests showing why (AGENTS.md rule 8).
//!
//! The inputs are the unit quaternions of the first `unit_quaternion_mul` oracle case (`a`, `b`),
//! the vector `(1.5, -2.25, 3.75)` and the rotation vector `(0.25, -0.1875, 0.125)`. Expected
//! values are the results of the kernels themselves, all of which are checked against upstream
//! nalgebra in `tests.cairo`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/unit_quaternion/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{uqt, v3t};
use crate::base::point3::Point3;
use crate::base::vector3::Vector3;
use crate::geometry::unit_quaternion::UnitQuaternionInternalTrait;
use super::{UnitQuaternion, UnitQuaternionTrait};

/// A unit quaternion of negative real part (so the sign conventions of `axis` and `angle` are
/// exercised).
fn a() -> UnitQuaternion<Fixed> {
    uqt((-1509276477, -2563574020, -2263667719, 2114881862))
}

/// Another unit quaternion.
fn b() -> UnitQuaternion<Fixed> {
    uqt((-1829744033, 968283947, 3750500405, -308145672))
}

/// `(1.5, -2.25, 3.75)`.
fn v() -> Vector3<Fixed> {
    v3t((0x180000000, -0x240000000, 0x3c0000000))
}

/// A small rotation vector `(0.25, -0.1875, 0.125)`.
fn w() -> Vector3<Fixed> {
    v3t((0x40000000, -0x30000000, 0x20000000))
}

/// `inverse_transform_vector` as it was before the conjugate's signs were folded into the
/// sandwich: three negations, then `transform_vector`. Bit for bit the same result.
fn alt_inverse_transform_vector_conjugate(
    q: UnitQuaternion<Fixed>, x: Vector3<Fixed>,
) -> Vector3<Fixed> {
    q.conjugate().transform_vector(x)
}

/// Folding the conjugate's signs into the operand order of the cross products (`(-u) × v = v ×
/// u`)
/// changes no exact product, hence no bit: the fused `inverse_transform_vector` and `conj_mul`
/// equal their conjugate-first formulations exactly.
#[test]
fn test_inverse_transform_vector_fused_matches_conjugate_then_transform() {
    assert!(a().inverse_transform_vector(v()) == alt_inverse_transform_vector_conjugate(a(), v()));
    assert!(b().inverse_transform_vector(w()) == alt_inverse_transform_vector_conjugate(b(), w()));
    assert!(a().inverse_transform_vector(w()) == alt_inverse_transform_vector_conjugate(a(), w()));
    let p = Point3 { x: v().x, y: v().y, z: v().z };
    let c = a().conjugate().transform_point(p);
    assert!(a().inverse_transform_point(p) == c);
    assert!(a().conj_mul(b()) == a().conjugate() * b());
    assert!(b().conj_mul(a()) == b().inverse() * a());
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_conj_mul__fused() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(uqt((-2063404847, 251977103, -2575182929, 2737525330)));
    assert!(q.conj_mul(r) == e);
}
