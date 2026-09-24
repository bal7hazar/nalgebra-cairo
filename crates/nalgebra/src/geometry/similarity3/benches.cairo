//! Gas benchmarks of `Similarity3` (`bench_similarity3_<op>__<variant>`) and the alternative
//! formulations requested by the work package.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{p3, sim3, v3};
use crate::base::point3::{Point3, Point3Trait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::isometry3::Isometry3Trait;
use crate::geometry::unit_quaternion::UnitQuaternionTrait;
use super::{Similarity3, Similarity3Trait};

fn x() -> Similarity3<Fixed> {
    sim3((0x100000000, -0x200000000, 0x100000000), (0, 0, 0x100000000, 0), 0x300000000)
}

fn p() -> Point3<Fixed> {
    p3(0x300000000, 0x400000000, 0x500000000)
}

fn expected_point() -> Point3<Fixed> {
    p3(-0x800000000, 0xa00000000, -0xe00000000)
}

fn expected_vector() -> Vector3<Fixed> {
    v3(-0x900000000, 0xc00000000, -0xf00000000)
}

fn id() -> Similarity3<Fixed> {
    Similarity3Trait::<Fixed>::identity()
}

fn alt_transform_point_rotate_scale_add(s: Similarity3<Fixed>, q: Point3<Fixed>) -> Point3<Fixed> {
    let r = s.isometry.rotation.transform_point(q);
    Point3 {
        x: r.x * s.scaling + s.isometry.translation.vector.x,
        y: r.y * s.scaling + s.isometry.translation.vector.y,
        z: r.z * s.scaling + s.isometry.translation.vector.z,
    }
}

fn alt_inverse_transform_point_recip(s: Similarity3<Fixed>, q: Point3<Fixed>) -> Point3<Fixed> {
    let r = s.isometry.inverse_transform_point(q);
    let inv = Real::<Fixed>::recip(s.scaling);
    Point3 { x: r.x * inv, y: r.y * inv, z: r.z * inv }
}

fn alt_inverse_transform_vector_recip(s: Similarity3<Fixed>, v: Vector3<Fixed>) -> Vector3<Fixed> {
    let r = s.isometry.inverse_transform_vector(v);
    let inv = Real::<Fixed>::recip(s.scaling);
    Vector3 { x: r.x * inv, y: r.y * inv, z: r.z * inv }
}

#[test]
fn test_inverse_transform_point_recip_loses_ulp() {
    let got = x().inverse_transform_point(expected_point());
    let alt = alt_inverse_transform_point_recip(x(), expected_point());
    assert!(got == p());
    assert!(alt != got);
    assert!(alt.abs_diff_eq(got, 16));
}

#[test]
#[inline(never)]
fn bench_similarity3_transform_point__baseline() {
    let _s: Similarity3<Fixed> = black_box(x());
    let _p: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(expected_point());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_transform_point__scale_translate() {
    let s: Similarity3<Fixed> = black_box(x());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(expected_point());
    assert!(s.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_transform_point__alt_rotate_scale_add() {
    let s: Similarity3<Fixed> = black_box(x());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(expected_point());
    assert!(alt_transform_point_rotate_scale_add(s, q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_inverse_transform_point__baseline() {
    let _s: Similarity3<Fixed> = black_box(x());
    let _p: Point3<Fixed> = black_box(expected_point());
    let e: Point3<Fixed> = black_box(p());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_inverse_transform_point__division() {
    let s: Similarity3<Fixed> = black_box(x());
    let q: Point3<Fixed> = black_box(expected_point());
    let e: Point3<Fixed> = black_box(p());
    assert!(s.inverse_transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_inverse_transform_point__alt_reciprocal() {
    let s: Similarity3<Fixed> = black_box(x());
    let q: Point3<Fixed> = black_box(expected_point());
    let e: Point3<Fixed> = black_box(p());
    assert!(alt_inverse_transform_point_recip(s, q).abs_diff_eq(e, 16));
}

#[test]
#[inline(never)]
fn bench_similarity3_transform_vector__baseline() {
    let _s: Similarity3<Fixed> = black_box(x());
    let _v: Vector3<Fixed> = black_box(p().coords());
    let e: Vector3<Fixed> = black_box(expected_vector());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_transform_vector__scale_after_rotate() {
    let s: Similarity3<Fixed> = black_box(x());
    let v: Vector3<Fixed> = black_box(p().coords());
    let e: Vector3<Fixed> = black_box(expected_vector());
    assert!(s.transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_inverse_transform_vector__baseline() {
    let _s: Similarity3<Fixed> = black_box(x());
    let _v: Vector3<Fixed> = black_box(expected_vector());
    let e: Vector3<Fixed> = black_box(p().coords());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_inverse_transform_vector__division() {
    let s: Similarity3<Fixed> = black_box(x());
    let v: Vector3<Fixed> = black_box(expected_vector());
    let e: Vector3<Fixed> = black_box(p().coords());
    assert!(s.inverse_transform_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_inverse_transform_vector__alt_reciprocal() {
    let s: Similarity3<Fixed> = black_box(x());
    let v: Vector3<Fixed> = black_box(expected_vector());
    let e: Vector3<Fixed> = black_box(p().coords());
    assert!(alt_inverse_transform_vector_recip(s, v).abs_diff_eq(e, 16));
}
