//! Unit tests of `Translation3`: exact cases (identity, integral translations), the identities a
//! translation must satisfy (`t · t⁻¹ = id`, composition = sum, `to_homogeneous` acts like
//! `transform_point`), the overflow panics, and the oracle vectors of `tools/oracle` (upstream
//! nalgebra 0.35 on the same raw inputs).
//!
//! Every operation of this type is an exact addition or negation, so the oracle tolerance is 0 and
//! the assertions below are BIT FOR BIT (`==`), not `abs_diff_eq`.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution, the
//! `translation3_*` ops of the `translation` suite) with
//! `cargo run --release -- emit-cairo translation --from vectors --max-per-dist 6 --ops
//! translation3_mul,translation3_inverse,translation3_transform_point,
//! translation3_inverse_transform_point --out <oracle.cairo>`.

use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix4::Matrix4Trait;
use crate::base::point3::{Point3, Point3Trait};
use crate::base::vector3::Vector3;
use super::{Translation3, Translation3Trait, oracle};

/// The raw value of 1.
const ONE_RAW: i64 = 0x100000000;

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

fn p3(t: (i64, i64, i64)) -> Point3<Fixed> {
    let (x, y, z) = t;
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

fn t3(t: (i64, i64, i64)) -> Translation3<Fixed> {
    Translation3 { vector: v3(t) }
}

fn id() -> Translation3<Fixed> {
    Translation3Trait::<Fixed>::identity()
}

// --- constructors and accessors

#[test]
fn test_identity_is_exact() {
    assert!(id() == t3((0, 0, 0)));
    assert!(id().vector() == Vector3 { x: Real::ZERO, y: Real::ZERO, z: Real::ZERO });
}

#[test]
fn test_new_from_vector_and_accessor() {
    let t = Translation3Trait::new(fx(3 * ONE_RAW), fx(-5 * ONE_RAW), fx(ONE_RAW));
    assert!(t == t3((3 * ONE_RAW, -5 * ONE_RAW, ONE_RAW)));
    assert!(Translation3Trait::from_vector(v3((3 * ONE_RAW, -5 * ONE_RAW, ONE_RAW))) == t);
    assert!(t.vector() == v3((3 * ONE_RAW, -5 * ONE_RAW, ONE_RAW)));
    assert!(t.vector == t.vector());
}

// --- inverse

#[test]
fn test_inverse_negates_and_is_involutive() {
    let t = t3((0x123456789, -0x9876543, 0x4000));
    assert!(t.inverse() == t3((-0x123456789, 0x9876543, -0x4000)));
    assert!(t.inverse().inverse() == t);
    assert!(id().inverse() == id());
}

#[test]
fn test_mul_inverse_is_the_identity_exactly() {
    let t = t3((0x123456789, -0x9876543, 0x4000));
    assert!(t * t.inverse() == id());
    assert!(t.inverse() * t == id());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_inverse_of_min_component_panics() {
    let v = Vector3 { x: Real::<Fixed>::MIN, y: Real::ZERO, z: Real::ZERO };
    let _ = Translation3 { vector: v }.inverse();
}

// --- composition

#[test]
fn test_mul_is_the_sum_and_commutes_bit_for_bit() {
    let (a, b) = (
        t3((0x140000000, -0x60000000, 0x280000000)), t3((-0x40000000, 0x280000000, -0x180000000)),
    );
    assert!(a * b == t3((0x100000000, 0x220000000, 0x100000000)));
    assert!(a * b == b * a);
    assert!(a * id() == a && id() * a == a);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_mul_overflow_panics() {
    let max = Vector3 { x: Real::<Fixed>::MAX, y: Real::ZERO, z: Real::ZERO };
    let one = Vector3 { x: Real::<Fixed>::ONE, y: Real::ZERO, z: Real::ZERO };
    let _ = Translation3 { vector: max } * Translation3 { vector: one };
}

// --- transforms

#[test]
fn test_transform_point_is_the_exact_sum() {
    let t = t3((0x140000000, -0x60000000, 0x280000000));
    let p = p3((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(t.transform_point(p) == p3((-0x140000000, 0x360000000, 0x340000000)));
    assert!(id().transform_point(p3((7, -9, 11))) == p3((7, -9, 11)));
}

#[test]
fn test_inverse_transform_point_undoes_transform_point() {
    let t = t3((0x123456789, -0x9876543, 0x4000));
    let p = p3((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(t.inverse_transform_point(t.transform_point(p)) == p);
    // Same result as transforming by the inverse translation, without building it.
    assert!(t.inverse_transform_point(p) == t.inverse().transform_point(p));
}

// --- homogeneous form

#[test]
fn test_to_homogeneous_is_the_identity_plus_the_last_column() {
    let t = t3((0x140000000, -0x60000000, 0x280000000));
    let m = t.to_homogeneous();
    assert!(m.m11 == Real::ONE && m.m22 == Real::ONE && m.m33 == Real::ONE && m.m44 == Real::ONE);
    assert!(m.m14 == fx(0x140000000) && m.m24 == fx(-0x60000000) && m.m34 == fx(0x280000000));
    assert!(m.m41 == Real::ZERO && m.m42 == Real::ZERO && m.m43 == Real::ZERO);
    assert!(id().to_homogeneous() == Matrix4Trait::<Fixed>::identity());
}

#[test]
fn test_to_homogeneous_acts_like_transform_point() {
    let t = t3((0x140000000, -0x60000000, 0x280000000));
    let p = p3((-0x280000000, 0x3c0000000, 0xc0000000));
    let h = t.to_homogeneous().mul_vec(p.to_homogeneous());
    let got = t.transform_point(p);
    assert!(h.x == got.x && h.y == got.y && h.z == got.z && h.w == Real::ONE);
}

// --- comparison and conversions

#[test]
fn test_abs_diff_eq_counts_raw_units() {
    let t = t3((0x140000000, -0x60000000, 0x280000000));
    assert!(t.abs_diff_eq(t3((0x140000003, -0x60000002, 0x280000001)), 3));
    assert!(!t.abs_diff_eq(t3((0x140000000, -0x60000000, 0x280000004)), 3));
    assert!(t.abs_diff_eq(t, 0));
}

#[test]
fn test_into_conversions_round_trip() {
    let v = v3((0x140000000, -0x60000000, 0x280000000));
    let t: Translation3<Fixed> = v.into();
    assert!(t == t3((0x140000000, -0x60000000, 0x280000000)));
    let back: Vector3<Fixed> = t.into();
    assert!(back == v);
}

// --- oracle vectors (upstream nalgebra 0.35, tolerance 0: bit for bit)

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::translation3_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t3(a) * t3(b) == t3(expected));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::translation3_inverse_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (t, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t3(t).inverse() == t3(expected));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::translation3_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (t, p, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t3(t).transform_point(p3(p)) == p3(expected));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::translation3_inverse_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (t, p, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t3(t).inverse_transform_point(p3(p)) == p3(expected));
    }
}

/// The homogeneous matrix of every oracle case acts on `(p, 1)` exactly like `transform_point`.
#[test]
fn test_to_homogeneous_oracle_consistency() {
    let mut cases = oracle::translation3_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (t, p, expected, _) = *case;
        let (ex, ey, ez) = expected;
        let h = t3(t).to_homogeneous().mul_vec(p3(p).to_homogeneous());
        assert!(h.x == fx(ex) && h.y == fx(ey) && h.z == fx(ez) && h.w == Real::ONE);
    }
}
