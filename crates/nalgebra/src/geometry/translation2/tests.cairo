//! Unit tests of `Translation2`: exact cases (identity, integral translations), the identities a
//! translation must satisfy (`t · t⁻¹ = id`, composition = sum, `to_homogeneous` acts like
//! `transform_point`), the overflow panics, and the oracle vectors of `tools/oracle` (upstream
//! nalgebra 0.35 on the same raw inputs).
//!
//! Every operation of this type is an exact addition or negation, so the oracle tolerance is 0 and
//! the assertions below are BIT FOR BIT (`==`), not `abs_diff_eq`.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution, the
//! `translation2_*` ops of the `translation` suite) with
//! `cargo run --release -- emit-cairo translation --from vectors --max-per-dist 6 --ops
//! translation2_mul,translation2_inverse,translation2_transform_point,
//! translation2_inverse_transform_point --out <oracle.cairo>`.

use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{ONE_RAW, fx, p2t, t2t, v2t};
use crate::base::point2::Point2Trait;
use crate::base::vector2::Vector2;
use super::{Translation2, Translation2Trait, oracle};

/// The raw value of 1.

fn id() -> Translation2<Fixed> {
    Translation2Trait::<Fixed>::identity()
}

// --- constructors and accessors

#[test]
fn test_identity_is_exact() {
    assert!(id() == t2t((0, 0)));
    assert!(id().vector() == Vector2 { x: Real::ZERO, y: Real::ZERO });
}

#[test]
fn test_new_from_vector_and_accessor() {
    let t = Translation2Trait::new(fx(3 * ONE_RAW), fx(-5 * ONE_RAW));
    assert!(t == t2t((3 * ONE_RAW, -5 * ONE_RAW)));
    assert!(Translation2Trait::from_vector(v2t((3 * ONE_RAW, -5 * ONE_RAW))) == t);
    assert!(t.vector() == v2t((3 * ONE_RAW, -5 * ONE_RAW)));
    assert!(t.vector == t.vector());
}

// --- inverse

#[test]
fn test_inverse_negates_and_is_involutive() {
    let t = t2t((0x123456789, -0x9876543));
    assert!(t.inverse() == t2t((-0x123456789, 0x9876543)));
    assert!(t.inverse().inverse() == t);
    assert!(id().inverse() == id());
}

#[test]
fn test_mul_inverse_is_the_identity_exactly() {
    let t = t2t((0x123456789, -0x9876543));
    assert!(t * t.inverse() == id());
    assert!(t.inverse() * t == id());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_inverse_of_min_component_panics() {
    let _ = Translation2 { vector: Vector2 { x: Real::<Fixed>::MIN, y: Real::ZERO } }.inverse();
}

// --- composition

#[test]
fn test_mul_is_the_sum_and_commutes_bit_for_bit() {
    let (a, b) = (t2t((0x140000000, -0x60000000)), t2t((-0x40000000, 0x280000000)));
    assert!(a * b == t2t((0x100000000, 0x220000000)));
    assert!(a * b == b * a);
    assert!(a * id() == a && id() * a == a);
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_mul_overflow_panics() {
    let big = Translation2 { vector: Vector2 { x: Real::<Fixed>::MAX, y: Real::ZERO } };
    let _ = big * Translation2 { vector: Vector2 { x: Real::<Fixed>::ONE, y: Real::ZERO } };
}

// --- transforms

#[test]
fn test_transform_point_is_the_exact_sum() {
    let t = t2t((0x140000000, -0x60000000));
    assert!(t.transform_point(p2t((0x40000000, 0x40000000))) == p2t((0x180000000, -0x20000000)));
    assert!(id().transform_point(p2t((7, -9))) == p2t((7, -9)));
}

#[test]
fn test_inverse_transform_point_undoes_transform_point() {
    let t = t2t((0x123456789, -0x9876543));
    let p = p2t((-0x280000000, 0x3c0000000));
    assert!(t.inverse_transform_point(t.transform_point(p)) == p);
    // Same result as transforming by the inverse translation, without building it.
    assert!(t.inverse_transform_point(p) == t.inverse().transform_point(p));
}

// --- homogeneous form

#[test]
fn test_to_homogeneous_is_the_identity_plus_the_last_column() {
    let t = t2t((0x140000000, -0x60000000));
    let m = t.to_homogeneous();
    assert!(m.m11 == Real::ONE && m.m22 == Real::ONE && m.m33 == Real::ONE);
    assert!(m.m12 == Real::ZERO && m.m21 == Real::ZERO);
    assert!(m.m31 == Real::ZERO && m.m32 == Real::ZERO);
    assert!(m.m13 == fx(0x140000000) && m.m23 == fx(-0x60000000));
    assert!(id().to_homogeneous() == Matrix3Trait::<Fixed>::identity());
}

#[test]
fn test_to_homogeneous_acts_like_transform_point() {
    let t = t2t((0x140000000, -0x60000000));
    let p = p2t((-0x280000000, 0x3c0000000));
    let h = t.to_homogeneous().mul_vec(p.to_homogeneous());
    let got = t.transform_point(p);
    assert!(h.x == got.x && h.y == got.y && h.z == Real::ONE);
}

// --- comparison and conversions

#[test]
fn test_abs_diff_eq_counts_raw_units() {
    let t = t2t((0x140000000, -0x60000000));
    assert!(t.abs_diff_eq(t2t((0x140000003, -0x60000002)), 3));
    assert!(!t.abs_diff_eq(t2t((0x140000004, -0x60000000)), 3));
    assert!(t.abs_diff_eq(t, 0));
}

#[test]
fn test_into_conversions_round_trip() {
    let v = v2t((0x140000000, -0x60000000));
    let t: Translation2<Fixed> = v.into();
    assert!(t == t2t((0x140000000, -0x60000000)));
    let back: Vector2<Fixed> = t.into();
    assert!(back == v);
}

// --- oracle vectors (upstream nalgebra 0.35, tolerance 0: bit for bit)

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::translation2_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t2t(a) * t2t(b) == t2t(expected));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::translation2_inverse_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (t, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t2t(t).inverse() == t2t(expected));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::translation2_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (t, p, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t2t(t).transform_point(p2t(p)) == p2t(expected));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::translation2_inverse_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (t, p, expected, tol) = *case;
        assert!(tol == 0);
        assert!(t2t(t).inverse_transform_point(p2t(p)) == p2t(expected));
    }
}

/// The homogeneous matrix of every oracle case acts on `(p, 1)` exactly like `transform_point`.
#[test]
fn test_to_homogeneous_oracle_consistency() {
    let mut cases = oracle::translation2_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (t, p, expected, _) = *case;
        let (ex, ey) = expected;
        let h = t2t(t).to_homogeneous().mul_vec(p2t(p).to_homogeneous());
        assert!(h.x == fx(ex) && h.y == fx(ey) && h.z == Real::ONE);
    }
}
