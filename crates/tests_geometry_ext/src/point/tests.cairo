//! Unit tests of the point types of WP 8.4-P09a: `Point1`, `Point4`, `Point5`, `Point6` (the
//! whole API, from one template: `Point4` in depth, the other sizes on every function whose body
//! depends on the size) and the completion of `Point2` / `Point3` (`Point2ExtTrait`...). Every
//! operation is exact except the divisions and `lerp`, so the expected values are exact.

use core::num::traits::Bounded;
use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::point2::{Point2, Point2Trait};
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::base::vector4::Vector4;
use nalgebra::base::vector5::Vector5;
use nalgebra::base::vector6::Vector6;
use nalgebra::geometry::point::{
    Point2ExtTrait, Point2Index, Point2PartialOrd, Point3ExtTrait, Point3Index, Point3PartialOrd,
};
use nalgebra::geometry::point1::{Point1, Point1Trait};
use nalgebra::geometry::point4::{Point4, Point4Trait};
use nalgebra::geometry::point5::{Point5, Point5Trait};
use nalgebra::geometry::point6::{Point6, Point6Trait};
use nalgebra_tests_utils::{ONE_RAW, fx, int};
use simba::scalar::Real;

fn p4(x: i64, y: i64, z: i64, w: i64) -> Point4<Fixed> {
    Point4Trait::new(int(x), int(y), int(z), int(w))
}

fn v4(x: i64, y: i64, z: i64, w: i64) -> Vector4<Fixed> {
    Vector4 { x: int(x), y: int(y), z: int(z), w: int(w) }
}

fn p6(x: i64, y: i64, z: i64, w: i64, a: i64, b: i64) -> Point6<Fixed> {
    Point6Trait::new(int(x), int(y), int(z), int(w), int(a), int(b))
}

// --- Point4, the whole API

#[test]
fn test_point4_construction_and_parts() {
    let p = p4(1, -2, 3, 4);
    assert!(p.x == int(1) && p.y == int(-2) && p.z == int(3) && p.w == int(4));
    assert!(Point4Trait::<Fixed>::origin() == p4(0, 0, 0, 0));
    assert!(Point4Trait::from_coordinates(v4(1, -2, 3, 4)) == p);
    assert!(p.coords() == v4(1, -2, 3, 4));
    let v: Vector4<Fixed> = p.into();
    let back: Point4<Fixed> = v.into();
    assert!(back == p);
    let arr: [Fixed; 4] = p.into();
    let from_arr: Point4<Fixed> = arr.into();
    assert!(from_arr == p);
    assert!(Default::default() == p4(0, 0, 0, 0));
}

#[test]
fn test_point4_homogeneous() {
    let p = p4(1, -2, 3, 4);
    let h = p.to_homogeneous();
    assert!(h == Vector5 { x: int(1), y: int(-2), z: int(3), w: int(4), a: int(1) });
    let h2 = Vector5 { x: int(2), y: int(-4), z: int(6), w: int(8), a: int(2) };
    assert!(Point4Trait::from_homogeneous(h2) == Some(p));
    let h0 = Vector5 { x: int(2), y: int(-4), z: int(6), w: int(8), a: int(0) };
    assert!(Point4Trait::from_homogeneous(h0) == None);
}

#[test]
fn test_point4_affine_ops() {
    let (p, q) = (p4(1, -2, 3, 4), p4(5, 5, -1, 0));
    assert!(p.sub_point(q) == v4(-4, -7, 4, 4));
    assert!(p.add_vector(v4(1, 1, 1, 1)) == p4(2, -1, 4, 5));
    assert!(p.sub_vector(v4(1, 1, 1, 1)) == p4(0, -3, 2, 3));
    assert!(p.scale(int(2)) == p4(2, -4, 6, 8));
    assert!(p4(2, -4, 6, 8).unscale(int(2)) == p);
    assert!(-p == p4(-1, 2, -3, -4));
    let mut m = p;
    m += v4(1, 1, 1, 1);
    assert!(m == p4(2, -1, 4, 5));
    m -= v4(1, 1, 1, 1);
    assert!(m == p);
    m *= int(3);
    assert!(m == p4(3, -6, 9, 12));
    m /= int(3);
    assert!(m == p);
}

#[test]
fn test_point4_inf_sup_lerp() {
    let (p, q) = (p4(1, -2, 3, 4), p4(5, 5, -1, 0));
    assert!(p.inf(q) == p4(1, -2, -1, 0));
    assert!(p.sup(q) == p4(5, 5, 3, 4));
    assert!(p.inf_sup(q) == (p4(1, -2, -1, 0), p4(5, 5, 3, 4)));
    assert!(p.lerp(q, Real::zero()) == p && p.lerp(q, Real::one()) == q);
    assert!(
        p
            .lerp(
                q, Real::HALF,
            ) == Point4Trait::new(int(3), fx(ONE_RAW + ONE_RAW / 2), int(1), int(2)),
    );
}

#[test]
fn test_point4_comparisons() {
    let p = p4(1, -2, 3, 4);
    let d = Point4 { x: p.x + fx(3), ..p };
    assert!(p.abs_diff_eq(d, 3) && !p.abs_diff_eq(d, 2));
    assert!(p.relative_eq(d, 3, Real::zero()) && !p.relative_eq(d, 2, Real::zero()));
    assert!(p.relative_eq(d, 0, fx(ONE_RAW)));
    assert!(p.ulps_eq(d, 0, 3) && !p.ulps_eq(d, 2, 2));
    assert!(p.cast::<Fixed>() == p);
}

/// Upstream's component-wise partial order: every coordinate must satisfy the relation.
#[test]
fn test_point4_partial_order() {
    let (p, q) = (p4(1, 2, 3, 4), p4(2, 3, 4, 5));
    assert!(p < q && p <= q && q > p && q >= p);
    assert!(!(q < p) && !(p > q));
    assert!(p <= p && p >= p && !(p < p));
    // Incomparable: neither is below the other.
    let r = p4(0, 9, 3, 4);
    assert!(!(p <= r) && !(r <= p) && !(p < r) && !(p > r));
}

#[test]
fn test_point4_index_slice_len() {
    let mut p = p4(1, -2, 3, 4);
    assert!(p[0] == int(1) && p[1] == int(-2) && p[2] == int(3) && p[3] == int(4));
    let s = array![int(1), int(-2), int(3), int(4)];
    assert!(Point4Trait::from_slice(s.span()) == p);
    assert!(p.len() == 4 && !p.is_empty() && p.stride() == 1);
    let lo: Point4<Fixed> = Point4Trait::min_value();
    let hi: Point4<Fixed> = Point4Trait::max_value();
    assert!(lo.x == Bounded::MIN && lo.w == Bounded::MIN && hi.y == Bounded::MAX);
    assert!(lo <= p && p <= hi);
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_point4_index_out_of_bounds_panics() {
    let mut p = p4(1, -2, 3, 4);
    let _ = p[4];
}

#[test]
#[should_panic(expected: ('nalgebra: wrong slice length',))]
fn test_point4_from_short_slice_panics() {
    let s = array![int(1), int(-2), int(3)];
    let _: Point4<Fixed> = Point4Trait::from_slice(s.span());
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_point4_unscale_by_zero_panics() {
    let _ = p4(1, -2, 3, 4).unscale(Real::zero());
}

// --- the other sizes

#[test]
fn test_point1() {
    let mut p: Point1<Fixed> = Point1Trait::new(int(3));
    assert!(p.coords() == Matrix1 { x: int(3) });
    assert!(Point1Trait::from_homogeneous(p.to_homogeneous()) == Some(p));
    assert!(p.to_homogeneous().y == int(1));
    assert!(p.unscale(int(2)).x == fx(ONE_RAW + ONE_RAW / 2));
    let mut q = p;
    q /= int(3);
    assert!(q == Point1Trait::new(int(1)));
    assert!(p.len() == 1 && p[0] == int(3));
    let s = array![int(3)];
    assert!(Point1Trait::from_slice(s.span()) == p);
    let arr: [Fixed; 1] = p.into();
    let back: Point1<Fixed> = arr.into();
    assert!(back == p && q < p);
    assert!(p.relative_eq(p, 0, Real::zero()) && p.ulps_eq(p, 0, 0));
}

#[test]
fn test_point5() {
    let mut p: Point5<Fixed> = Point5Trait::new(int(1), int(2), int(3), int(4), int(5));
    let h = p.to_homogeneous();
    assert!(h == Vector6 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5), b: int(1) });
    assert!(Point5Trait::from_homogeneous(h) == Some(p));
    assert!(p.scale(int(2)).unscale(int(2)) == p);
    let mut q = p.scale(int(2));
    q /= int(2);
    assert!(q == p && p[4] == int(5) && p.len() == 5);
    let s = array![int(1), int(2), int(3), int(4), int(5)];
    assert!(Point5Trait::from_slice(s.span()) == p);
    assert!(p.lerp(p.scale(int(3)), Real::HALF) == p.scale(int(2)));
}

#[test]
fn test_point6() {
    let mut p = p6(1, 2, 3, 4, 5, 6);
    assert!(
        p.coords() == Vector6 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5), b: int(6) },
    );
    assert!(p.scale(int(2)).unscale(int(2)) == p);
    let mut q = p.scale(int(2));
    q /= int(2);
    assert!(q == p && p[5] == int(6) && p.len() == 6);
    let s = array![int(1), int(2), int(3), int(4), int(5), int(6)];
    assert!(Point6Trait::from_slice(s.span()) == p);
    assert!(p.inf(-p) == -p && p.sup(-p) == p && -p < p);
    let hi: Point6<Fixed> = Point6Trait::max_value();
    assert!(hi.b == Bounded::MAX);
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_point6_index_out_of_bounds_panics() {
    let mut p = p6(1, 2, 3, 4, 5, 6);
    let _ = p[6];
}

// --- Point2 / Point3 completion

#[test]
fn test_point2_completion() {
    let mut p: Point2<Fixed> = Point2Trait::new(int(1), int(-2));
    assert!(p[0] == int(1) && p[1] == int(-2));
    assert!(p.len() == 2 && !p.is_empty() && p.stride() == 1);
    let s = array![int(1), int(-2)];
    assert!(Point2ExtTrait::from_slice(s.span()) == p);
    let d = Point2 { x: p.x, y: p.y - fx(2) };
    assert!(p.relative_eq(d, 2, Real::zero()) && !p.relative_eq(d, 1, Real::zero()));
    assert!(p.ulps_eq(d, 0, 2) && !p.ulps_eq(d, 1, 1));
    assert!(p.cast::<Fixed>() == p);
    // `x` is equal: `d <= p` but not `d < p` (every coordinate must satisfy the relation).
    assert!(d <= p && p >= d && !(d < p) && !(p > d));
    let q: Point2<Fixed> = Point2Trait::new(int(0), int(5));
    assert!(!(p <= q) && !(q <= p));
    let lo: Point2<Fixed> = Point2ExtTrait::min_value();
    let hi: Point2<Fixed> = Point2ExtTrait::max_value();
    assert!(lo < p && p < hi && lo.y == Bounded::MIN && hi.x == Bounded::MAX);
}

#[test]
fn test_point3_completion() {
    let mut p: Point3<Fixed> = Point3Trait::new(int(1), int(-2), int(3));
    assert!(p[0] == int(1) && p[1] == int(-2) && p[2] == int(3));
    assert!(p.len() == 3 && !p.is_empty() && p.stride() == 1);
    let s = array![int(1), int(-2), int(3)];
    assert!(Point3ExtTrait::from_slice(s.span()) == p);
    let d = Point3 { z: p.z + fx(7), ..p };
    assert!(p.relative_eq(d, 7, Real::zero()) && !p.relative_eq(d, 6, Real::zero()));
    assert!(p.relative_eq(d, 0, fx(ONE_RAW / 1000)));
    assert!(p.ulps_eq(d, 0, 7) && !p.ulps_eq(d, 6, 6));
    assert!(p.cast::<Fixed>() == p);
    assert!(p <= d && !(p < d) && d >= p);
    let hi: Point3<Fixed> = Point3ExtTrait::max_value();
    let lo: Point3<Fixed> = Point3ExtTrait::min_value();
    assert!(lo < p && p < hi);
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_point3_index_out_of_bounds_panics() {
    let mut p: Point3<Fixed> = Point3Trait::new(int(1), int(-2), int(3));
    let _ = p[3];
}

#[test]
#[should_panic(expected: ('nalgebra: wrong slice length',))]
fn test_point2_from_long_slice_panics() {
    let s = array![int(1), int(-2), int(3)];
    let _: Point2<Fixed> = Point2ExtTrait::from_slice(s.span());
}
