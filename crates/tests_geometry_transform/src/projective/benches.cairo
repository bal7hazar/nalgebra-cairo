//! Gas benchmarks of `Projective3` / `Transform3` (WP 8.4-P11a) (`bench_<group>__<variant>`, net =
//! raw - `baseline` of the group). Expected values are the results of the library kernels, checked
//! against upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::point3::Point3;
use nalgebra::geometry::affine3::Affine3;
use nalgebra::geometry::projective3::{Projective3, Projective3Trait};
use nalgebra::geometry::transform::{TransformDiv, TransformMul};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, aff3t, p3t, proj3};

/// A projective map, entries of magnitude about 1 (determinant about 3.1).
fn p() -> Projective3<Fixed> {
    proj3(
        [
            [5368709120, -2147483648, 1073741824, 2147483648],
            [1610612736, 8589934592, -3221225472, -1073741824],
            [-536870912, 2147483648, 6442450944, 536870912],
            [1073741824, 0, -536870912, 4294967296],
        ],
    )
}

fn a() -> Affine3<Fixed> {
    aff3t(
        [
            [5368709120, -2147483648, 1073741824], [1610612736, 8589934592, -3221225472],
            [-536870912, 2147483648, 6442450944],
        ],
        (15032385536, -52613349376, 171798691840),
    )
}

fn pt() -> Point3<Fixed> {
    p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW))
}

#[test]
#[inline(never)]
fn bench_projective3_try_inverse__baseline() {
    let _t: Projective3<Fixed> = black_box(p());
    let e: Matrix4<Fixed> = black_box(p().try_inverse().unwrap().into_inner());
    assert!(e == e);
}

/// `Matrix4::try_inverse` of the homogeneous matrix (upstream's formula).
#[test]
#[inline(never)]
fn bench_projective3_try_inverse__matrix4() {
    let t: Projective3<Fixed> = black_box(p());
    let e: Matrix4<Fixed> = black_box(p().try_inverse().unwrap().into_inner());
    assert!(t.try_inverse().unwrap().into_inner() == e);
}

#[test]
#[inline(never)]
fn bench_projective3_transform_point__baseline() {
    let _t: Projective3<Fixed> = black_box(p());
    let _q: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p().transform_point(pt()));
    assert!(e == e);
}

/// `Matrix4::transform_point` (upstream's formula: fused `q` and `n`, then the division).
#[test]
#[inline(never)]
fn bench_projective3_transform_point__homogeneous() {
    let t: Projective3<Fixed> = black_box(p());
    let q: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p().transform_point(pt()));
    assert!(t.transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_projective3_inverse_transform_point__baseline() {
    let _t: Projective3<Fixed> = black_box(p());
    let _q: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p().inverse_transform_point(pt()));
    assert!(e == e);
}

/// The inverse, then the transform (upstream's formula).
#[test]
#[inline(never)]
fn bench_projective3_inverse_transform_point__inverse_then_transform() {
    let t: Projective3<Fixed> = black_box(p());
    let q: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p().inverse_transform_point(pt()));
    assert!(t.inverse_transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_projective3_mul_affine3__baseline() {
    let _t: Projective3<Fixed> = black_box(p());
    let _u: Affine3<Fixed> = black_box(a());
    let r: Projective3<Fixed> = p().mul_transform(a());
    let e: Matrix4<Fixed> = black_box(r.into_inner());
    assert!(e == e);
}

/// `TransformMul::mul_transform`, the full 4x4 product (upstream's formula).
#[test]
#[inline(never)]
fn bench_projective3_mul_affine3__full_product() {
    let t: Projective3<Fixed> = black_box(p());
    let u: Affine3<Fixed> = black_box(a());
    let r: Projective3<Fixed> = p().mul_transform(a());
    let e: Matrix4<Fixed> = black_box(r.into_inner());
    let got: Projective3<Fixed> = t.mul_transform(u);
    assert!(got.into_inner() == e);
}

#[test]
#[inline(never)]
fn bench_affine3_div_projective3__baseline() {
    let _t: Projective3<Fixed> = black_box(p());
    let _u: Affine3<Fixed> = black_box(a());
    let r: Projective3<Fixed> = a().div_transform(p());
    let e: Matrix4<Fixed> = black_box(r.into_inner());
    assert!(e == e);
}

/// `TransformDiv::div_transform`: the inverse of the divisor, then the full product.
#[test]
#[inline(never)]
fn bench_affine3_div_projective3__inverse_then_product() {
    let t: Projective3<Fixed> = black_box(p());
    let u: Affine3<Fixed> = black_box(a());
    let r: Projective3<Fixed> = a().div_transform(p());
    let e: Matrix4<Fixed> = black_box(r.into_inner());
    let got: Projective3<Fixed> = u.div_transform(t);
    assert!(got.into_inner() == e);
}
