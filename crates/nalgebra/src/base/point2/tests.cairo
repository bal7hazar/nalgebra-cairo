//! Unit tests of `Point2`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, extremes of the range, identities with the vector
//! operations, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the `point2_*` suite) with `cargo run --release -- emit-cairo point --from vectors
//! --max-per-dist 4 --ops <list> --out <oracle.cairo>`, `<list>` being the comma-separated
//! `point2_<op>` names of the oracle tests at the bottom of this file.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, p2, p2t, p3, v2, v2t, v3};
use crate::base::vector2::{Vector2, Vector2Trait};
use super::{Point2, Point2Trait, oracle};

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

// --- constructors and conversions

#[test]
fn test_new_sets_fields() {
    let r = Point2Trait::<Fixed>::new(fx(0x180000000), fx(-0x240000000));
    assert!(r.x == fx(0x180000000) && r.y == fx(-0x240000000));
    assert!(r == a());
}

#[test]
fn test_origin_is_zero() {
    assert!(Point2Trait::<Fixed>::origin() == p2(0, 0));
    assert!(Point2Trait::<Fixed>::origin() == Default::default());
}

#[test]
fn test_from_coordinates_and_coords_roundtrip() {
    let v = v2(0x180000000, -0x240000000);
    assert!(Point2Trait::<Fixed>::from_coordinates(v) == a());
    assert!(a().coords() == v);
    assert!(Point2Trait::<Fixed>::from_coordinates(a().coords()) == a());
}

#[test]
fn test_into_conversions() {
    let v = v2(0x180000000, -0x240000000);
    let from_vector: Point2<Fixed> = v.into();
    assert!(from_vector == a());
    let to_vector: Vector2<Fixed> = a().into();
    assert!(to_vector == v);
    let from_tuple: Point2<Fixed> = (fx(0x180000000), fx(-0x240000000)).into();
    assert!(from_tuple == a());
    let to_tuple: (Fixed, Fixed) = a().into();
    assert!(to_tuple == (fx(0x180000000), fx(-0x240000000)));
    let from_array: Point2<Fixed> = [fx(0x180000000), fx(-0x240000000)].into();
    assert!(from_array == a());
    let to_array: [Fixed; 2] = a().into();
    assert!(to_array == [fx(0x180000000), fx(-0x240000000)]);
}

#[test]
fn test_push_appends_a_coordinate() {
    assert!(a().push(fx(0x3c0000000)) == p3(0x180000000, -0x240000000, 0x3c0000000));
}

// --- homogeneous coordinates

#[test]
fn test_to_homogeneous_appends_one() {
    assert!(a().to_homogeneous() == v3(0x180000000, -0x240000000, 0x100000000));
    assert!(Point2Trait::<Fixed>::origin().to_homogeneous() == v3(0, 0, 0x100000000));
}

#[test]
fn test_from_homogeneous_divides_by_w() {
    // (1.5, -2.25) / 2 and / -2 and / 3, exactly.
    let v = |w: i64| v3(0x180000000, -0x240000000, w);
    assert!(
        Point2Trait::<
            Fixed,
        >::from_homogeneous(v(0x200000000)) == Some(p2(0xc0000000, -0x120000000)),
    );
    assert!(
        Point2Trait::<
            Fixed,
        >::from_homogeneous(v(-0x200000000)) == Some(p2(-0xc0000000, 0x120000000)),
    );
    assert!(
        Point2Trait::<Fixed>::from_homogeneous(v(0x300000000)) == Some(p2(0x80000000, -0xc0000000)),
    );
}

#[test]
fn test_from_homogeneous_rounds_toward_negative_infinity() {
    // (1.5, -2.25) / 7: exact floors.
    assert!(
        Point2Trait::<
            Fixed,
        >::from_homogeneous(
            v3(0x180000000, -0x240000000, 0x700000000),
        ) == Some(p2(920350134, -1380525203)),
    );
}

#[test]
fn test_from_homogeneous_zero_w_is_none() {
    assert!(Point2Trait::<Fixed>::from_homogeneous(v3(0x180000000, 0, 0)) == None);
    assert!(Point2Trait::<Fixed>::from_homogeneous(v3(0, 0, 0)) == None);
}

#[test]
fn test_from_homogeneous_inverts_to_homogeneous() {
    assert!(Point2Trait::<Fixed>::from_homogeneous(a().to_homogeneous()) == Some(a()));
    assert!(Point2Trait::<Fixed>::from_homogeneous(b().to_homogeneous()) == Some(b()));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_from_homogeneous_overflow() {
    // 2^30 / 2^-32 does not fit.
    let _ = Point2Trait::<Fixed>::from_homogeneous(black_box(v3(0x4000000000000000, 0, 1)));
}

// --- affine operations

#[test]
fn test_sub_point_is_a_vector() {
    // (1.5, -2.25) - (-4.5, 0.25) = (6, -2.5)
    assert!(a().sub_point(b()) == v2(0x600000000, -0x280000000));
    assert!(b().sub_point(a()) == -a().sub_point(b()));
    assert!(a().sub_point(a()) == Vector2Trait::<Fixed>::zeros());
    assert!(p().sub_point(Point2Trait::<Fixed>::origin()) == p().coords());
}

#[test]
fn test_add_vector_translates() {
    // (1.5, -2.25) + (-4.5, 0.25) = (-3, -2)
    assert!(a().add_vector(b().coords()) == p2(-0x300000000, -0x200000000));
    assert!(a().add_vector(Vector2Trait::<Fixed>::zeros()) == a());
}

#[test]
fn test_sub_vector_translates() {
    assert!(a().sub_vector(b().coords()) == p2(0x600000000, -0x280000000));
    assert!(a().sub_vector(Vector2Trait::<Fixed>::zeros()) == a());
}

#[test]
fn test_translation_identities() {
    // q = p + (q - p), p = (p + v) - v.
    assert!(a().add_vector(b().sub_point(a())) == b());
    assert!(a().add_vector(p().coords()).sub_vector(p().coords()) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sub_point_overflow() {
    let _ = black_box(p2(0, MAX)).sub_point(p2(0, -1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_add_vector_overflow() {
    let _ = black_box(p2(0, MAX)).add_vector(v2(0, 1));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_sub_vector_overflow() {
    let _ = black_box(p2(0, MIN)).sub_vector(v2(0, 1));
}

#[test]
fn test_neg_mirrors_through_origin() {
    assert!(-a() == p2(-0x180000000, 0x240000000));
    assert!(-(-a()) == a());
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_neg_overflow() {
    let _ = -black_box(p2(0, MIN));
}

#[test]
fn test_add_assign_and_sub_assign_translate() {
    let mut r = a();
    r += b().coords();
    assert!(r == a().add_vector(b().coords()));
    r -= b().coords();
    assert!(r == a());
}

#[test]
fn test_mul_assign_and_div_assign_scale() {
    let mut r = a();
    r *= fx(0x280000000);
    assert!(r == a().scale(fx(0x280000000)));
    let mut r = a();
    r /= fx(0x280000000);
    assert!(r == a().unscale(fx(0x280000000)));
}

// --- scaling

#[test]
fn test_scale_exact() {
    // (1.5, -2.25) * 2 = (3, -4.5); * -0.5 = (-0.75, 1.125)
    assert!(a().scale(fx(0x200000000)) == p2(0x300000000, -0x480000000));
    assert!(a().scale(fx(-0x80000000)) == p2(-0xc0000000, 0x120000000));
    assert!(a().scale(Real::ONE) == a());
    assert!(a().scale(Real::ZERO) == Point2Trait::<Fixed>::origin());
}

#[test]
fn test_scale_rounds_toward_negative_infinity() {
    // [3, -3] ulp * 0.5: floor, whatever the sign.
    assert!(p2(3, -3).scale(fx(0x80000000)) == p2(1, -2));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_scale_overflow() {
    let _ = black_box(p2(0, 0x4000000000000000)).scale(fx(0x200000000));
}

#[test]
fn test_unscale_exact() {
    // (1.5, -2.25) / 2 = (0.75, -1.125); / -0.5 = * -2
    assert!(a().unscale(fx(0x200000000)) == p2(0xc0000000, -0x120000000));
    assert!(a().unscale(fx(-0x80000000)) == p2(-0x300000000, 0x480000000));
    assert!(a().unscale(Real::ONE) == a());
}

#[test]
fn test_unscale_rounds_toward_negative_infinity() {
    // (1.5, -2.25) / 7: exact floors.
    assert!(a().unscale(fx(0x700000000)) == p2(920350134, -1380525203));
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_unscale_by_zero() {
    let _ = black_box(a()).unscale(Real::ZERO);
}

// --- inf / sup

#[test]
fn test_inf_sup_are_coordinate_wise() {
    let inf = p2(-0x480000000, -0x240000000);
    let sup = p2(0x180000000, 0x40000000);
    assert!(a().inf(b()) == inf);
    assert!(a().sup(b()) == sup);
    assert!(b().inf(a()) == inf);
    assert!(b().sup(a()) == sup);
    assert!(a().inf_sup(b()) == (inf, sup));
    assert!(a().inf(a()) == a() && a().sup(a()) == a());
}

// --- approximate equality

#[test]
fn test_abs_diff_eq_counts_ulps() {
    assert!(a().abs_diff_eq(a(), 0));
    assert!(a().abs_diff_eq(p2(0x180000000 + 3, -0x240000000 - 3), 3));
    assert!(!a().abs_diff_eq(p2(0x180000000 + 3, -0x240000000 - 3), 2));
    assert!(!a().abs_diff_eq(p2(0x180000000, -0x240000000 + 1), 0));
    assert!(!a().abs_diff_eq(p2(0x180000000 + 1, -0x240000000), 0));
}

#[test]
fn test_abs_diff_eq_cannot_overflow() {
    assert!(!p2(MAX, 0).abs_diff_eq(p2(MIN, 0), 1));
    assert!(p2(MAX, 0).abs_diff_eq(p2(MAX, 0), 0));
}

// --- interpolation and center

#[test]
fn test_lerp_exact() {
    // t = 0, 1 give the endpoints; t = 1/2 the midpoint; t = 2 extrapolates.
    assert!(a().lerp(b(), Real::ZERO) == a());
    assert!(a().lerp(b(), Real::ONE) == b());
    assert!(a().lerp(b(), Real::HALF) == p2(-0x180000000, -0x100000000));
    assert!(a().lerp(b(), Real::TWO) == p2(-0xa80000000, 0x2c0000000));
}

#[test]
fn test_lerp_rounds_once() {
    // t = 0.25 is exact on these inputs; t = 0.1 (429496730 raw) floors once per coordinate.
    assert!(a().lerp(b(), fx(0x40000000)) == p2(0, -0x1a0000000));
    assert!(a().lerp(b(), fx(429496730)) == p2(3865470564, -8589934591));
}

#[test]
fn test_lerp_matches_vector_lerp() {
    let t = fx(429496730);
    assert!(a().lerp(b(), t).coords() == a().coords().lerp(b().coords(), t));
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
    assert!(p().distance(p()) == Real::ZERO);
    assert!(p().distance_squared(p()) == Real::ZERO);
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
#[should_panic(expected: 'simba: overflow')]
fn test_distance_squared_overflow() {
    let _ = black_box(p2(0x186a000000000, 0)).distance_squared(p2(0, 0));
}

#[test]
#[should_panic(expected: 'simba: overflow')]
fn test_distance_difference_overflow() {
    let _ = black_box(p2(0, MAX)).distance(p2(0, -1));
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_sub_point_oracle() {
    let mut cases = oracle::point2_sub_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!(p2t(p).sub_point(p2t(q)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_add_vector_oracle() {
    let mut cases = oracle::point2_add_vector_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, v, expected, tol) = *case;
        assert!(p2t(p).add_vector(v2t(v)).abs_diff_eq(p2t(expected), tol));
    }
}

#[test]
fn test_sub_vector_oracle() {
    let mut cases = oracle::point2_sub_vector_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, v, expected, tol) = *case;
        assert!(p2t(p).sub_vector(v2t(v)).abs_diff_eq(p2t(expected), tol));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::point2_scale_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, k, expected, tol) = *case;
        assert!(p2t(p).scale(fx(k)).abs_diff_eq(p2t(expected), tol));
    }
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
