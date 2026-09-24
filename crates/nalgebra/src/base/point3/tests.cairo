//! Unit tests of `Point3`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, extremes of the range, identities with the vector
//! operations, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the `point3_*` suite) with `cargo run --release -- emit-cairo point --from vectors
//! --max-per-dist 4 --ops <list> --out <oracle.cairo>`, `<list>` being the comma-separated
//! `point3_<op>` names of the oracle tests at the bottom of this file.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, p2, p3, p3t, v3, v3t, v4};
use crate::base::vector3::{Vector3, Vector3Trait};
use super::{Point3, Point3InternalTrait, Point3Trait, oracle};

const MAX: i64 = 0x7fffffffffffffff;
const MIN: i64 = -0x8000000000000000;

/// (1.5, -2.25, 3.75)
fn a() -> Point3<Fixed> {
    p3(0x180000000, -0x240000000, 0x3c0000000)
}

/// (-4.5, 0.25, 2)
fn b() -> Point3<Fixed> {
    p3(-0x480000000, 0x40000000, 0x200000000)
}

/// (3, -4, 12), at distance 13 from the origin
fn p() -> Point3<Fixed> {
    p3(0x300000000, -0x400000000, 0xc00000000)
}

// --- constructors and conversions

#[test]
fn test_new_sets_fields() {
    let r = Point3Trait::<Fixed>::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000));
    assert!(r.x == fx(0x180000000) && r.y == fx(-0x240000000) && r.z == fx(0x3c0000000));
    assert!(r == a());
}

#[test]
fn test_origin_is_zero() {
    assert!(Point3Trait::<Fixed>::origin() == p3(0, 0, 0));
    assert!(Point3Trait::<Fixed>::origin() == Default::default());
}

#[test]
fn test_from_coordinates_and_coords_roundtrip() {
    let v = v3(0x180000000, -0x240000000, 0x3c0000000);
    assert!(Point3Trait::<Fixed>::from_coordinates(v) == a());
    assert!(a().coords() == v);
    assert!(Point3Trait::<Fixed>::from_coordinates(a().coords()) == a());
}

#[test]
fn test_into_conversions() {
    let v = v3(0x180000000, -0x240000000, 0x3c0000000);
    let from_vector: Point3<Fixed> = v.into();
    assert!(from_vector == a());
    let to_vector: Vector3<Fixed> = a().into();
    assert!(to_vector == v);
    let from_array: Point3<Fixed> = [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)].into();
    assert!(from_array == a());
    let to_array: [Fixed; 3] = a().into();
    assert!(to_array == [fx(0x180000000), fx(-0x240000000), fx(0x3c0000000)]);
}

#[test]
fn test_xy_drops_z() {
    assert!(a().xy() == p2(0x180000000, -0x240000000));
}

// --- homogeneous coordinates

#[test]
fn test_to_homogeneous_appends_one() {
    assert!(a().to_homogeneous() == v4(0x180000000, -0x240000000, 0x3c0000000, 0x100000000));
    assert!(Point3Trait::<Fixed>::origin().to_homogeneous() == v4(0, 0, 0, 0x100000000));
}

#[test]
fn test_from_homogeneous_divides_by_w() {
    // (1.5, -2.25, 3.75) / 2 and / -2 and / 3, exactly.
    let v = |w: i64| v4(0x180000000, -0x240000000, 0x3c0000000, w);
    assert!(
        Point3Trait::<
            Fixed,
        >::from_homogeneous(v(0x200000000)) == Some(p3(0xc0000000, -0x120000000, 0x1e0000000)),
    );
    assert!(
        Point3Trait::<
            Fixed,
        >::from_homogeneous(v(-0x200000000)) == Some(p3(-0xc0000000, 0x120000000, -0x1e0000000)),
    );
    assert!(
        Point3Trait::<
            Fixed,
        >::from_homogeneous(v(0x300000000)) == Some(p3(0x80000000, -0xc0000000, 0x140000000)),
    );
}

#[test]
fn test_from_homogeneous_rounds_to_nearest() {
    // (1.5, -2.25, 3.75) / 7: rounded to nearest.
    assert!(
        Point3Trait::<
            Fixed,
        >::from_homogeneous(
            v4(0x180000000, -0x240000000, 0x3c0000000, 0x700000000),
        ) == Some(p3(920350135, -1380525202, 2300875337)),
    );
    // Ties to even: 0.5, -1.5, 2.5 ulp.
    assert!(
        Point3Trait::<Fixed>::from_homogeneous(v4(1, -3, 5, 0x200000000)) == Some(p3(0, -2, 2)),
    );
}

#[test]
fn test_from_homogeneous_zero_w_is_none() {
    assert!(Point3Trait::<Fixed>::from_homogeneous(v4(0x180000000, 0, 0, 0)) == None);
    assert!(Point3Trait::<Fixed>::from_homogeneous(v4(0, 0, 0, 0)) == None);
}

#[test]
fn test_from_homogeneous_inverts_to_homogeneous() {
    assert!(Point3Trait::<Fixed>::from_homogeneous(a().to_homogeneous()) == Some(a()));
    assert!(Point3Trait::<Fixed>::from_homogeneous(b().to_homogeneous()) == Some(b()));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_from_homogeneous_overflow() {
    // 2^30 / 2^-32 does not fit.
    let _ = Point3Trait::<Fixed>::from_homogeneous(black_box(v4(0x4000000000000000, 0, 0, 1)));
}

// --- affine operations

#[test]
fn test_sub_point_is_a_vector() {
    // (1.5, -2.25, 3.75) - (-4.5, 0.25, 2) = (6, -2.5, 1.75)
    assert!(a().sub_point(b()) == v3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(b().sub_point(a()) == -a().sub_point(b()));
    assert!(a().sub_point(a()) == Vector3Trait::<Fixed>::zeros());
    assert!(p().sub_point(Point3Trait::<Fixed>::origin()) == p().coords());
}

#[test]
fn test_add_vector_translates() {
    // (1.5, -2.25, 3.75) + (-4.5, 0.25, 2) = (-3, -2, 5.75)
    assert!(a().add_vector(b().coords()) == p3(-0x300000000, -0x200000000, 0x5c0000000));
    assert!(a().add_vector(Vector3Trait::<Fixed>::zeros()) == a());
}

#[test]
fn test_sub_vector_translates() {
    assert!(a().sub_vector(b().coords()) == p3(0x600000000, -0x280000000, 0x1c0000000));
    assert!(a().sub_vector(Vector3Trait::<Fixed>::zeros()) == a());
}

#[test]
fn test_translation_identities() {
    // q = p + (q - p), p = (p + v) - v.
    assert!(a().add_vector(b().sub_point(a())) == b());
    assert!(a().add_vector(p().coords()).sub_vector(p().coords()) == a());
}

#[test]
#[should_panic(expected: 'i64_sub Overflow')]
fn test_sub_point_overflow() {
    let _ = black_box(p3(0, 0, MAX)).sub_point(p3(0, 0, -1));
}

#[test]
#[should_panic(expected: 'i64_add Overflow')]
fn test_add_vector_overflow() {
    let _ = black_box(p3(0, 0, MAX)).add_vector(v3(0, 0, 1));
}

#[test]
#[should_panic(expected: 'i64_sub Underflow')]
fn test_sub_vector_overflow() {
    let _ = black_box(p3(0, 0, MIN)).sub_vector(v3(0, 0, 1));
}

#[test]
fn test_neg_mirrors_through_origin() {
    assert!(-a() == p3(-0x180000000, 0x240000000, -0x3c0000000));
    assert!(-(-a()) == a());
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_neg_overflow() {
    let _ = -black_box(p3(0, 0, MIN));
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
    // (1.5, -2.25, 3.75) * 2 = (3, -4.5, 7.5); * -0.5 = (-0.75, 1.125, -1.875)
    assert!(a().scale(fx(0x200000000)) == p3(0x300000000, -0x480000000, 0x780000000));
    assert!(a().scale(fx(-0x80000000)) == p3(-0xc0000000, 0x120000000, -0x1e0000000));
    assert!(a().scale(Real::one()) == a());
    assert!(a().scale(Real::zero()) == Point3Trait::<Fixed>::origin());
}

#[test]
fn test_scale_rounds_toward_negative_infinity() {
    // [3, -3, 1] ulp * 0.5: floor, whatever the sign.
    assert!(p3(3, -3, 1).scale(fx(0x80000000)) == p3(1, -2, 0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale_overflow() {
    let _ = black_box(p3(0, 0, 0x4000000000000000)).scale(fx(0x200000000));
}

#[test]
fn test_unscale_exact() {
    // (1.5, -2.25, 3.75) / 2 = (0.75, -1.125, 1.875); / -0.5 = * -2
    assert!(a().unscale(fx(0x200000000)) == p3(0xc0000000, -0x120000000, 0x1e0000000));
    assert!(a().unscale(fx(-0x80000000)) == p3(-0x300000000, 0x480000000, -0x780000000));
    assert!(a().unscale(Real::one()) == a());
}

#[test]
fn test_unscale_rounds_to_nearest() {
    // (1.5, -2.25, 3.75) / 7: rounded to nearest.
    assert!(a().unscale(fx(0x700000000)) == p3(920350135, -1380525202, 2300875337));
    // Ties to even: 0.5, -1.5, 2.5 ulp.
    assert!(p3(1, -3, 5).unscale(fx(0x200000000)) == p3(0, -2, 2));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_unscale_by_zero() {
    let _ = black_box(a()).unscale(Real::zero());
}

// --- inf / sup

#[test]
fn test_inf_sup_are_coordinate_wise() {
    let inf = p3(-0x480000000, -0x240000000, 0x200000000);
    let sup = p3(0x180000000, 0x40000000, 0x3c0000000);
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
    assert!(a().abs_diff_eq(p3(0x180000000 + 3, -0x240000000 - 3, 0x3c0000000 + 3), 3));
    assert!(!a().abs_diff_eq(p3(0x180000000 + 3, -0x240000000 - 3, 0x3c0000000 + 3), 2));
    assert!(!a().abs_diff_eq(p3(0x180000000, -0x240000000, 0x3c0000000 + 1), 0));
    assert!(!a().abs_diff_eq(p3(0x180000000 + 1, -0x240000000, 0x3c0000000), 0));
    assert!(!a().abs_diff_eq(p3(0x180000000, -0x240000000 + 1, 0x3c0000000), 0));
}

#[test]
fn test_abs_diff_eq_cannot_overflow() {
    assert!(!p3(MAX, 0, 0).abs_diff_eq(p3(MIN, 0, 0), 1));
    assert!(p3(MAX, 0, 0).abs_diff_eq(p3(MAX, 0, 0), 0));
}

// --- interpolation and center

#[test]
fn test_lerp_exact() {
    // t = 0, 1 give the endpoints; t = 1/2 the midpoint; t = 2 extrapolates.
    assert!(a().lerp(b(), Real::zero()) == a());
    assert!(a().lerp(b(), Real::one()) == b());
    assert!(a().lerp(b(), Real::HALF) == p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(a().lerp(b(), Real::TWO) == p3(-0xa80000000, 0x2c0000000, 0x40000000));
}

#[test]
fn test_lerp_rounds_once() {
    // t = 0.25 is exact on these inputs; t = 0.1 (429496730 raw) floors once per coordinate.
    assert!(a().lerp(b(), fx(0x40000000)) == p3(0, -0x1a0000000, 0x350000000));
    assert!(a().lerp(b(), fx(429496730)) == p3(3865470564, -8589934591, 15354508082));
}

#[test]
fn test_lerp_matches_vector_lerp() {
    let t = fx(429496730);
    assert!(a().lerp(b(), t).coords() == a().coords().lerp(b().coords(), t));
}

#[test]
fn test_center_exact() {
    assert!(a().center(b()) == p3(-0x180000000, -0x100000000, 0x2e0000000));
    assert!(b().center(a()) == a().center(b()));
    assert!(a().center(a()) == a());
    assert!(a().center(b()) == a().lerp(b(), Real::HALF));
}

#[test]
fn test_center_floors_odd_sums() {
    // (1 + 0) / 2 = 0.5 ulp -> 0; (-1 + 0) / 2 = -0.5 ulp -> -1; (3 - 4) / 2 -> -1.
    assert!(p3(1, -1, 3).center(p3(0, 0, -4)) == p3(0, -1, -1));
}

#[test]
fn test_center_cannot_overflow() {
    // The sum of the coordinates does not fit, their exact half does.
    assert!(p3(MAX, MAX, MIN).center(p3(MAX, MIN, MIN)) == p3(MAX, -1, MIN));
}

// --- distances

#[test]
fn test_distance_exact() {
    // (3, -4, 12) is at distance 13 (Pythagorean quadruple) from the origin.
    let o = Point3Trait::<Fixed>::origin();
    assert!(p().distance_squared(o) == fx(0xa900000000));
    assert!(p().distance(o) == fx(0xd00000000));
    assert!(o.distance(p()) == fx(0xd00000000));
    assert!(p().distance(p()) == Real::zero());
    assert!(p().distance_squared(p()) == Real::zero());
}

#[test]
fn test_distance_of_general_points() {
    // |a - b|² = 6² + 2.5² + 1.75² = 45.3125, floor(sqrt) computed on the raw sum of squares.
    assert!(a().distance_squared(b()) == fx(0x2d50000000));
    assert!(a().distance(b()) == fx(28911383412));
}

#[test]
fn test_distance_matches_vector_norms() {
    assert!(a().distance(b()) == a().sub_point(b()).norm());
    assert!(a().distance_squared(b()) == a().sub_point(b()).norm_squared());
}

#[test]
fn test_distance_without_intermediate_overflow() {
    // Distance 100 000: its square (1e10) does not fit Q32.32, the distance does.
    assert!(p3(0x186a000000000, 0, 0).distance(p3(0, 0, 0)) == fx(0x186a000000000));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_distance_squared_overflow() {
    let _ = black_box(p3(0x186a000000000, 0, 0)).distance_squared(p3(0, 0, 0));
}

#[test]
#[should_panic(expected: 'i64_sub Overflow')]
fn test_distance_difference_overflow() {
    let _ = black_box(p3(0, 0, MAX)).distance(p3(0, 0, -1));
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_sub_point_oracle() {
    let mut cases = oracle::point3_sub_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!(p3t(p).sub_point(p3t(q)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_add_vector_oracle() {
    let mut cases = oracle::point3_add_vector_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, v, expected, tol) = *case;
        assert!(p3t(p).add_vector(v3t(v)).abs_diff_eq(p3t(expected), tol));
    }
}

#[test]
fn test_sub_vector_oracle() {
    let mut cases = oracle::point3_sub_vector_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, v, expected, tol) = *case;
        assert!(p3t(p).sub_vector(v3t(v)).abs_diff_eq(p3t(expected), tol));
    }
}

#[test]
fn test_scale_oracle() {
    let mut cases = oracle::point3_scale_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, k, expected, tol) = *case;
        assert!(p3t(p).scale(fx(k)).abs_diff_eq(p3t(expected), tol));
    }
}

#[test]
fn test_distance_squared_oracle() {
    let mut cases = oracle::point3_distance_squared_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!((p3t(p).distance_squared(p3t(q))).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_distance_oracle() {
    let mut cases = oracle::point3_distance_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!((p3t(p).distance(p3t(q))).abs_diff_eq(fx(expected), tol));
    }
}

#[test]
fn test_center_oracle() {
    let mut cases = oracle::point3_center_cases();
    assert!(cases.len() >= 16);
    while let Some(case) = cases.pop_front() {
        let (p, q, expected, tol) = *case;
        assert!(p3t(p).center(p3t(q)).abs_diff_eq(p3t(expected), tol));
    }
}
