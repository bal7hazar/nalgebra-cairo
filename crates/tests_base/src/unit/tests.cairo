//! Unit tests of `Unit`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, robustness at both ends of the range, the Newton step
//! of `renormalize_fast`, and the oracle vectors of `tools/oracle` (`normalize` of upstream
//! nalgebra 0.35 on the same raw inputs, which is what `Unit::new_normalize` computes).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution)
//! as documented in its header.
//!
//! Moved from `crates/nalgebra/src/base/unit/tests.cairo` (WP 8.1c, test-only package): the tests
//! of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::unit::{Unit, Unit2Trait, Unit3Trait, Unit4Trait, UnitTrait};
use nalgebra::base::vector2::Vector2Trait;
use nalgebra::base::vector3::{Vector3, Vector3Trait};
use nalgebra::base::vector4::Vector4Trait;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, u2, u3, u4, v2, v3, v4};
use simba::scalar::Real;
use crate::unit::oracle;

const MAX: i64 = 0x7fffffffffffffff;

/// (1.5, -2.25, 3.75), of norm 4.6236 (raw 19856967406)
fn a() -> Vector3<Fixed> {
    v3(0x180000000, -0x240000000, 0x3c0000000)
}

/// (3, -4, 12), of norm 13
fn p() -> Vector3<Fixed> {
    v3(0x300000000, -0x400000000, 0xc00000000)
}

/// `p / 13`, rounded to nearest
fn np() -> Unit<Vector3<Fixed>> {
    u3(991146299, -1321528399, 3964585196)
}

/// `a / |a|`, rounded to nearest
fn na() -> Unit<Vector3<Fixed>> {
    u3(1393471397, -2090207095, 3483678492)
}

// --- construction

#[test]
fn test_new_unchecked_wraps_without_normalizing() {
    assert!(UnitTrait::new_unchecked(a()) == Unit { value: a() });
    assert!(UnitTrait::new_unchecked(a()).value == a());
}

#[test]
fn test_new_normalize_exact() {
    // (3, -4, 12) / 13, floored; the same values as `Vector3::normalize`.
    assert!(UnitTrait::new_normalize(p()) == np());
    assert!(UnitTrait::new_normalize(p()).value == p().normalize());
    assert!(UnitTrait::new_normalize(a()) == na());
}

#[test]
fn test_new_normalize_axes_are_exact() {
    assert!(UnitTrait::new_normalize(v3(-0x500000000, 0, 0)) == -Unit3Trait::<Fixed>::x_axis());
    assert!(UnitTrait::new_normalize(v3(0, 0x200000000, 0)) == Unit3Trait::<Fixed>::y_axis());
    assert!(UnitTrait::new_normalize(v3(0, 0, 0x7)) == Unit3Trait::<Fixed>::z_axis());
}

#[test]
fn test_new_normalize_2d_and_4d() {
    // (3, -4) / 5 and (1, -1, 1, -1) / 2 and (3, -4, 12, 0) / 13.
    assert!(UnitTrait::new_normalize(v2(0x300000000, -0x400000000)) == u2(2576980378, -3435973837));
    assert!(
        UnitTrait::new_normalize(
            v4(0x100000000, -0x100000000, 0x100000000, -0x100000000),
        ) == u4(0x80000000, -0x80000000, 0x80000000, -0x80000000),
    );
    assert!(
        UnitTrait::new_normalize(
            v4(0x300000000, -0x400000000, 0xc00000000, 0),
        ) == u4(991146299, -1321528399, 3964585196, 0),
    );
}

#[test]
fn test_new_normalize_tiny_is_exact() {
    // (3, -4, 0) ulp has a norm of 5 ulp: the divisions still give (0.6, -0.8, 0) floored.
    assert!(UnitTrait::new_normalize(v3(3, -4, 0)) == u3(2576980378, -3435973837, 0));
}

#[test]
fn test_new_normalize_huge_does_not_overflow() {
    // Norms of 100 000 and 141 421: their squares do not fit Q32.32, the norms do.
    assert!(UnitTrait::new_normalize(v3(0x186a000000000, 0, 0)) == Unit3Trait::<Fixed>::x_axis());
    assert!(
        UnitTrait::new_normalize(
            v3(0x186a000000000, 0x186a000000000, 0),
        ) == u3(3037000500, 3037000500, 0),
    );
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_new_normalize_zero() {
    let _ = UnitTrait::new_normalize(black_box(Vector3Trait::<Fixed>::zeros()));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_new_normalize_zero_2d() {
    let _ = UnitTrait::new_normalize(black_box(Vector2Trait::<Fixed>::zeros()));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_new_normalize_zero_4d() {
    let _ = UnitTrait::new_normalize(black_box(Vector4Trait::<Fixed>::zeros()));
}

#[test]
fn test_new_and_get_returns_the_norm() {
    let (u, n) = UnitTrait::new_and_get(p());
    assert!(u == np());
    assert!(n == fx(0xd00000000));
    let (u, n) = UnitTrait::new_and_get(v2(0x300000000, -0x400000000));
    assert!(u == u2(2576980378, -3435973837));
    assert!(n == fx(0x500000000));
    let (u, n) = UnitTrait::new_and_get(a());
    assert!(u == na());
    assert!(n == fx(0x49f9146ee));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_new_and_get_zero() {
    let _ = UnitTrait::new_and_get(black_box(Vector3Trait::<Fixed>::zeros()));
}

#[test]
fn test_try_new_some() {
    assert!(UnitTrait::try_new(p(), Real::zero()) == Some(np()));
    assert!(UnitTrait::try_new(p(), fx(0xd00000000) - Real::default_epsilon()) == Some(np()));
    assert!(
        UnitTrait::try_new(
            v3(3, -4, 0), Real::default_epsilon(),
        ) == Some(u3(2576980378, -3435973837, 0)),
    );
}

#[test]
fn test_try_new_none() {
    assert!(UnitTrait::try_new(Vector3Trait::<Fixed>::zeros(), Real::zero()) == None);
    // The norm is compared with `<=`.
    assert!(UnitTrait::try_new(p(), fx(0xd00000000)) == None);
    assert!(UnitTrait::try_new(v3(3, -4, 0), fx(5)) == None);
    assert!(UnitTrait::try_new(v2(0, 0), Real::zero()) == None);
    assert!(UnitTrait::try_new(v4(0, 0, 0, 0), Real::zero()) == None);
}

#[test]
fn test_try_new_and_get() {
    assert!(UnitTrait::try_new_and_get(p(), Real::zero()) == Some((np(), fx(0xd00000000))));
    assert!(UnitTrait::try_new_and_get(p(), fx(0xd00000000)) == None);
    assert!(UnitTrait::try_new_and_get(Vector3Trait::<Fixed>::zeros(), Real::zero()) == None);
}

#[test]
fn test_into_inner() {
    assert!(np().into_inner() == np().value);
    assert!(np().into_inner() == v3(991146299, -1321528399, 3964585196));
}

// --- products and operators

#[test]
fn test_dot_of_units() {
    assert!(Unit3Trait::<Fixed>::x_axis().dot(Unit3Trait::<Fixed>::x_axis()) == Real::one());
    assert!(Unit3Trait::<Fixed>::x_axis().dot(Unit3Trait::<Fixed>::y_axis()) == Real::zero());
    assert!(Unit3Trait::<Fixed>::x_axis().dot(-Unit3Trait::<Fixed>::x_axis()) == Real::NEG_ONE);
    // A normalized vector has a squared norm of 1 within a few ulp: here 1 ulp short.
    assert!(na().dot(na()) == fx(4294967295));
    assert!(na().dot(np()) == na().value.dot(np().value));
}

#[test]
fn test_dot_with_a_vector() {
    // The signed length of the projection of (3, -4, 12) on the axes, and of a on `na`.
    assert!(Unit3Trait::<Fixed>::x_axis().into_inner().dot(p()) == fx(0x300000000));
    assert!(Unit3Trait::<Fixed>::y_axis().into_inner().dot(p()) == fx(-0x400000000));
    assert!(Unit3Trait::<Fixed>::z_axis().into_inner().dot(p()) == fx(0xc00000000));
    assert!(na().into_inner().dot(a()) == fx(19856967404));
    assert!(u2(0x100000000, 0).into_inner().dot(v2(0x300000000, 5)) == fx(0x300000000));
    assert!(u4(0, 0, 0, 0x100000000).into_inner().dot(v4(1, 2, 3, 0x400000000)) == fx(0x400000000));
}

#[test]
fn test_scale_gives_a_vector() {
    // The scaled axis of a rotation: axis * angle.
    assert!(Unit3Trait::<Fixed>::z_axis().scale(fx(0x280000000)) == v3(0, 0, 0x280000000));
    assert!(Unit3Trait::<Fixed>::x_axis().scale(fx(-0x180000000)) == v3(-0x180000000, 0, 0));
    assert!(np().scale(Real::zero()) == Vector3Trait::<Fixed>::zeros());
    assert!(np().scale(Real::one()) == np().value);
    assert!(np().scale(fx(0xd00000000)) == np().value.scale(fx(0xd00000000)));
}

#[test]
fn test_neg_is_exact() {
    assert!(-np() == u3(-991146299, 1321528399, -3964585196));
    assert!(-(-np()) == np());
    assert!(-u2(2576980377, -3435973837) == u2(-2576980377, 3435973837));
    assert!(-(-Unit3Trait::<Fixed>::z_axis()) == Unit3Trait::<Fixed>::z_axis());
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_neg_overflow() {
    let _ = -black_box(u3(0, 0, -0x8000000000000000));
}

#[test]
fn test_abs_diff_eq_counts_ulps() {
    assert!(np().abs_diff_eq(np(), 0));
    assert!(np().abs_diff_eq(u3(991146299 + 2, -1321528399 - 2, 3964585196 + 2), 2));
    assert!(!np().abs_diff_eq(u3(991146299 + 2, -1321528399 - 2, 3964585196 + 2), 1));
    assert!(!np().abs_diff_eq(-np(), 1000));
    assert!(!u3(MAX, 0, 0).abs_diff_eq(u3(-MAX, 0, 0), 1));
}

// --- axes

#[test]
fn test_axes_are_exact() {
    assert!(Unit2Trait::<Fixed>::x_axis() == u2(0x100000000, 0));
    assert!(Unit2Trait::<Fixed>::y_axis() == u2(0, 0x100000000));
    assert!(Unit3Trait::<Fixed>::x_axis() == u3(0x100000000, 0, 0));
    assert!(Unit3Trait::<Fixed>::y_axis() == u3(0, 0x100000000, 0));
    assert!(Unit3Trait::<Fixed>::z_axis() == u3(0, 0, 0x100000000));
    assert!(Unit4Trait::<Fixed>::x_axis() == u4(0x100000000, 0, 0, 0));
    assert!(Unit4Trait::<Fixed>::y_axis() == u4(0, 0x100000000, 0, 0));
    assert!(Unit4Trait::<Fixed>::z_axis() == u4(0, 0, 0x100000000, 0));
    assert!(Unit4Trait::<Fixed>::w_axis() == u4(0, 0, 0, 0x100000000));
}

#[test]
fn test_axes_match_the_vector_axes() {
    assert!(Unit3Trait::<Fixed>::x_axis().value == Vector3Trait::<Fixed>::x());
    assert!(Unit3Trait::<Fixed>::y_axis().value == Vector3Trait::<Fixed>::y());
    assert!(Unit3Trait::<Fixed>::z_axis().value == Vector3Trait::<Fixed>::z());
    assert!(Unit2Trait::<Fixed>::x_axis().value == Vector2Trait::<Fixed>::x());
    assert!(Unit2Trait::<Fixed>::y_axis().value == Vector2Trait::<Fixed>::y());
    assert!(Unit4Trait::<Fixed>::w_axis().value == Vector4Trait::<Fixed>::w());
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp)

#[test]
fn test_new_normalize_2d_oracle() {
    let mut cases = oracle::vector2_normalize_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let ((x, y), (ex, ey), tol) = *case;
        let (u, n) = UnitTrait::new_and_get(v2(x, y));
        assert!(u.abs_diff_eq(u2(ex, ey), tol));
        assert!(UnitTrait::new_normalize(v2(x, y)) == u);
        assert!(n == v2(x, y).norm());
    }
}

#[test]
fn test_new_normalize_3d_oracle() {
    let mut cases = oracle::vector3_normalize_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let ((x, y, z), (ex, ey, ez), tol) = *case;
        let (u, n) = UnitTrait::new_and_get(v3(x, y, z));
        assert!(u.abs_diff_eq(u3(ex, ey, ez), tol));
        assert!(UnitTrait::new_normalize(v3(x, y, z)) == u);
        assert!(n == v3(x, y, z).norm());
    }
}

#[test]
fn test_new_normalize_4d_oracle() {
    let mut cases = oracle::vector4_normalize_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let ((x, y, z, w), (ex, ey, ez, ew), tol) = *case;
        let (u, n) = UnitTrait::new_and_get(v4(x, y, z, w));
        assert!(u.abs_diff_eq(u4(ex, ey, ez, ew), tol));
        assert!(UnitTrait::new_normalize(v4(x, y, z, w)) == u);
        assert!(n == v4(x, y, z, w).norm());
    }
}
