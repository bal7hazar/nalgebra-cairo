//! Unit tests of `Point2`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, extremes of the range, identities with the vector
//! operations, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the `point2_*` suite) with `cargo run --release -- emit-cairo point --from vectors
//! --max-per-dist 4 --ops <list> --out <oracle.cairo>`, `<list>` being the comma-separated
//! `point2_<op>` names of the oracle tests at the bottom of this file.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/point2/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, p2, p2t};
use crate::base::vector2::Vector2Trait;
use super::{Point2, Point2InternalTrait, Point2Trait, oracle};

const MAX: i64 = 0x7fffffffffffffff;
const MIN: i64 = -0x8000000000000000;

/// (1.5, -2.25)
fn a() -> Point2<Fixed> {
    p2(0x180000000, -0x240000000)
}

/// (-4.5, 0.25)
fn b() -> Point2<Fixed> {
    p2(-0x480000000, 0x40000000)
}

/// (5, -12), at distance 13 from the origin
fn p() -> Point2<Fixed> {
    p2(0x500000000, -0xc00000000)
}

#[test]
fn test_center_exact() {
    assert!(a().center(b()) == p2(-0x180000000, -0x100000000));
    assert!(b().center(a()) == a().center(b()));
    assert!(a().center(a()) == a());
    assert!(a().center(b()) == a().lerp(b(), Real::HALF));
}

#[test]
fn test_center_floors_odd_sums() {
    // (1 + 0) / 2 = 0.5 ulp -> 0; (-1 + 0) / 2 = -0.5 ulp -> -1; (3 - 4) / 2 -> -1.
    assert!(p2(1, -1).center(p2(0, 0)) == p2(0, -1));
    assert!(p2(3, 0).center(p2(-4, 0)) == p2(-1, 0));
}

#[test]
fn test_center_cannot_overflow() {
    // The sum of the coordinates does not fit, their exact half does.
    assert!(p2(MAX, MAX).center(p2(MAX, MIN)) == p2(MAX, -1));
    assert!(p2(MIN, 0).center(p2(MIN, 0)) == p2(MIN, 0));
}

// --- distances

#[test]
fn test_distance_exact() {
    // (5, -12) is at distance 13 (Pythagorean triple) from the origin.
    let o = Point2Trait::<Fixed>::origin();
    assert!(p().distance_squared(o) == fx(0xa900000000));
    assert!(p().distance(o) == fx(0xd00000000));
    assert!(o.distance(p()) == fx(0xd00000000));
    assert!(p().distance(p()) == Real::zero());
    assert!(p().distance_squared(p()) == Real::zero());
}

#[test]
fn test_distance_of_general_points() {
    // |a - b|² = 6² + 2.5² = 42.25 = 6.5².
    assert!(a().distance_squared(b()) == fx(0x2a40000000));
    assert!(a().distance(b()) == fx(0x680000000));
}

#[test]
fn test_distance_matches_vector_norms() {
    assert!(a().distance(b()) == a().sub_point(b()).norm());
    assert!(a().distance_squared(b()) == a().sub_point(b()).norm_squared());
}

#[test]
fn test_distance_without_intermediate_overflow() {
    // Distance 100 000: its square (1e10) does not fit Q32.32, the distance does.
    assert!(p2(0x186a000000000, 0).distance(p2(0, 0)) == fx(0x186a000000000));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_distance_squared_overflow() {
    let _ = black_box(p2(0x186a000000000, 0)).distance_squared(p2(0, 0));
}

#[test]
#[should_panic(expected: 'i64_sub Overflow')]
fn test_distance_difference_overflow() {
    let _ = black_box(p2(0, MAX)).distance(p2(0, -1));
}

#[test]
fn test_distance_squared_oracle() {
    let mut cases = oracle::point2_distance_squared_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!((p2t(p).distance_squared(p2t(q))).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_distance_oracle() {
    let mut cases = oracle::point2_distance_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!((p2t(p).distance(p2t(q))).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_center_oracle() {
    let mut cases = oracle::point2_center_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!(p2t(p).center(p2t(q)).abs_diff_eq(p2t(expected), tol));
    }
}
