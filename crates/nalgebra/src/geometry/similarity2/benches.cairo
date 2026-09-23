//! Gas benchmarks of `Similarity2` (`bench_similarity2_<op>__<variant>`) and the alternative
//! formulations requested by the work package.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{p2, sim2, v2};
use crate::base::point2::{Point2, Point2Trait};
use crate::base::vector2::{Vector2, Vector2Trait};
use crate::geometry::isometry2::Isometry2Trait;
use crate::geometry::unit_complex::UnitComplexTrait;
use super::{Similarity2, Similarity2Trait};

fn x() -> Similarity2<Fixed> {
    sim2(0x100000000, -0x200000000, 0, 0x100000000, 0x300000000)
}

fn p() -> Point2<Fixed> {
    p2(0x300000000, 0x400000000)
}

fn expected_point() -> Point2<Fixed> {
    p2(-0xb00000000, 0x700000000)
}

fn expected_vector() -> Vector2<Fixed> {
    v2(-0xc00000000, 0x900000000)
}

fn id() -> Similarity2<Fixed> {
    Similarity2Trait::<Fixed>::identity()
}

fn alt_transform_point_rotate_scale_add(s: Similarity2<Fixed>, q: Point2<Fixed>) -> Point2<Fixed> {
    let r = s.isometry.rotation.transform_point(q);
    Point2 {
        x: r.x * s.scaling + s.isometry.translation.vector.x,
        y: r.y * s.scaling + s.isometry.translation.vector.y,
    }
}

fn alt_inverse_transform_point_recip(s: Similarity2<Fixed>, q: Point2<Fixed>) -> Point2<Fixed> {
    let r = s.isometry.inverse_transform_point(q);
    let inv = Real::<Fixed>::recip(s.scaling);
    Point2 { x: r.x * inv, y: r.y * inv }
}

fn alt_inverse_transform_vector_recip(s: Similarity2<Fixed>, v: Vector2<Fixed>) -> Vector2<Fixed> {
    let r = s.isometry.inverse_transform_vector(v);
    let inv = Real::<Fixed>::recip(s.scaling);
    Vector2 { x: r.x * inv, y: r.y * inv }
}

#[test]
fn test_inverse_transform_point_recip_loses_ulp() {
    let got = x().inverse_transform_point(expected_point());
    let alt = alt_inverse_transform_point_recip(x(), expected_point());
    assert!(got == p());
    assert!(alt != got);
    assert!(alt.abs_diff_eq(got, 4));
}

#[test]
fn test_inv_mul_alt_inverse_then_mul_agrees_with_extra_rounding() {
    let got = x().inv_mul(x());
    let alt = x().inverse() * x();
    assert!(got.abs_diff_eq(id(), 0));
    assert!(alt.abs_diff_eq(id(), 1));
}

#[test]
#[inline(never)]
fn bench_similarity2_transform_point__baseline() {
    let _s: Similarity2<Fixed> = black_box(x());
    let _p: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(expected_point());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_transform_point__scale_translate() {
    let s: Similarity2<Fixed> = black_box(x());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(expected_point());
    assert!(s.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_transform_point__alt_rotate_scale_add() {
    let s: Similarity2<Fixed> = black_box(x());
    let q: Point2<Fixed> = black_box(p());
    let e: Point2<Fixed> = black_box(expected_point());
    assert!(alt_transform_point_rotate_scale_add(s, q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inverse_transform_point__baseline() {
    let _s: Similarity2<Fixed> = black_box(x());
    let _p: Point2<Fixed> = black_box(expected_point());
    let e: Point2<Fixed> = black_box(p());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inverse_transform_point__division() {
    let s: Similarity2<Fixed> = black_box(x());
    let q: Point2<Fixed> = black_box(expected_point());
    let e: Point2<Fixed> = black_box(p());
    assert!(s.inverse_transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inverse_transform_point__alt_reciprocal() {
    let s: Similarity2<Fixed> = black_box(x());
    let q: Point2<Fixed> = black_box(expected_point());
    let e: Point2<Fixed> = black_box(p());
    assert!(alt_inverse_transform_point_recip(s, q).abs_diff_eq(e, 4));
}

#[test]
#[inline(never)]
fn bench_similarity2_inv_mul__baseline() {
    let _s: Similarity2<Fixed> = black_box(x());
    let e: Similarity2<Fixed> = black_box(id());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inv_mul__direct() {
    let s: Similarity2<Fixed> = black_box(x());
    let e: Similarity2<Fixed> = black_box(id());
    assert!(s.inv_mul(s).abs_diff_eq(e, 0));
}

#[test]
#[inline(never)]
fn bench_similarity2_inv_mul__alt_inverse_then_mul() {
    let s: Similarity2<Fixed> = black_box(x());
    let e: Similarity2<Fixed> = black_box(id());
    assert!((s.inverse() * s).abs_diff_eq(e, 1));
}

#[test]
#[inline(never)]
fn bench_similarity2_transform_vector__baseline() {
    let _s: Similarity2<Fixed> = black_box(x());
    let _v: Vector2<Fixed> = black_box(p().coords());
    let e: Vector2<Fixed> = black_box(expected_vector());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_transform_vector__scale_after_rotate() {
    let s: Similarity2<Fixed> = black_box(x());
    let v: Vector2<Fixed> = black_box(p().coords());
    let e: Vector2<Fixed> = black_box(expected_vector());
    assert!(s.transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inverse_transform_vector__baseline() {
    let _s: Similarity2<Fixed> = black_box(x());
    let _v: Vector2<Fixed> = black_box(expected_vector());
    let e: Vector2<Fixed> = black_box(p().coords());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inverse_transform_vector__division() {
    let s: Similarity2<Fixed> = black_box(x());
    let v: Vector2<Fixed> = black_box(expected_vector());
    let e: Vector2<Fixed> = black_box(p().coords());
    assert!(s.inverse_transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_inverse_transform_vector__alt_reciprocal() {
    let s: Similarity2<Fixed> = black_box(x());
    let v: Vector2<Fixed> = black_box(expected_vector());
    let e: Vector2<Fixed> = black_box(p().coords());
    assert!(alt_inverse_transform_vector_recip(s, v).abs_diff_eq(e, 4));
}
