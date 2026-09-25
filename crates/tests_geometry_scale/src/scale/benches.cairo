//! Gas benchmarks of `Scale2` / `Scale3` / `Scale6` (WP 8.4-P10), `bench_scale<N>_<op>__<variant>`,
//! net = raw - `baseline` of the group, and the alternative that lost (`alt_*`), kept as evidence
//! together with the test showing that both agree (AGENTS.md rule 8).
//!
//! `try_inverse_transform_point` divides every coordinate by its factor (`Real::div`, ONE correctly
//! rounded division each); upstream's formula takes the reciprocals of the factors and multiplies
//! (`try_inverse().map(|s| s * pt)`): `alt_reciprocal_product`.

use fixed::Fixed;
use nalgebra::geometry::scale2::Scale2Trait;
use nalgebra::geometry::scale3::{Scale3, Scale3Trait};
use nalgebra::geometry::scale6::Scale6Trait;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int};
use crate::common::{err_pt3, pt2, pt3, pt6, sc2, sc3, sc6, vec3};

const ONE: i64 = 0x100000000;

fn s3() -> Scale3<Fixed> {
    sc3((2 * ONE, -4 * ONE, 8 * ONE))
}

/// The point `s3().transform_point(p3())` inverts to `p3()`.
fn p3() -> (i64, i64, i64) {
    (3 * ONE, 5 * ONE, -7 * ONE)
}

/// `try_inverse_transform_point` through the reciprocals (upstream's formula).
fn alt_reciprocal_product(
    s: Scale3<Fixed>, p: (i64, i64, i64),
) -> nalgebra::base::point3::Point3<Fixed> {
    s.try_inverse().unwrap().transform_point(pt3(p))
}

#[test]
fn test_try_inverse_transform_point_alt_reciprocal_product_agrees_within_an_ulp() {
    let s = sc3((3 * ONE, -5 * ONE, 7 * ONE));
    let p = (11 * ONE, 13 * ONE, -17 * ONE);
    let q = s.try_inverse_transform_point(pt3(p)).unwrap();
    let alt = alt_reciprocal_product(s, p);
    assert!(err_pt3(q, alt) <= 16);
    // The division is the correctly rounded quotient: `q * s` is within an ulp of `p`.
    assert!(err_pt3(s.transform_point(q), pt3(p)) <= 8);
    // Exact when the quotient is representable.
    let exact = s3().try_inverse_transform_point(s3().transform_point(pt3(p3()))).unwrap();
    assert!(
        exact == pt3(p3())
            && alt_reciprocal_product(s3(), (6 * ONE, -20 * ONE, -56 * ONE)) == pt3(p3()),
    );
}

// --- transform_point

#[test]
#[inline(never)]
fn bench_scale3_transform_point__baseline() {
    let _s = black_box(s3());
    let _p = black_box(pt3(p3()));
    let e = black_box(pt3((6 * ONE, -20 * ONE, -56 * ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale3_transform_point__product() {
    let s = black_box(s3());
    let p = black_box(pt3(p3()));
    let e = black_box(pt3((6 * ONE, -20 * ONE, -56 * ONE)));
    assert!(s.transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_scale2_transform_point__baseline() {
    let _s = black_box(sc2((2 * ONE, -4 * ONE)));
    let _p = black_box(pt2((3 * ONE, 5 * ONE)));
    let e = black_box(pt2((6 * ONE, -20 * ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale2_transform_point__product() {
    let s = black_box(sc2((2 * ONE, -4 * ONE)));
    let p = black_box(pt2((3 * ONE, 5 * ONE)));
    let e = black_box(pt2((6 * ONE, -20 * ONE)));
    assert!(s.transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_scale6_transform_point__baseline() {
    let _s = black_box(sc6((ONE, 2 * ONE, 3 * ONE, 4 * ONE, 5 * ONE, 6 * ONE)));
    let _p = black_box(pt6((ONE, ONE, ONE, ONE, ONE, ONE)));
    let e = black_box(pt6((ONE, 2 * ONE, 3 * ONE, 4 * ONE, 5 * ONE, 6 * ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale6_transform_point__product() {
    let s = black_box(sc6((ONE, 2 * ONE, 3 * ONE, 4 * ONE, 5 * ONE, 6 * ONE)));
    let p = black_box(pt6((ONE, ONE, ONE, ONE, ONE, ONE)));
    let e = black_box(pt6((ONE, 2 * ONE, 3 * ONE, 4 * ONE, 5 * ONE, 6 * ONE)));
    assert!(s.transform_point(p) == e);
}

// --- inverses

#[test]
#[inline(never)]
fn bench_scale3_try_inverse__baseline() {
    let _s = black_box(s3());
    let e = black_box(sc3((0x80000000, -0x40000000, 0x20000000)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale3_try_inverse__reciprocals() {
    let s = black_box(s3());
    let e = black_box(sc3((0x80000000, -0x40000000, 0x20000000)));
    assert!(s.try_inverse().unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_scale3_try_inverse__unchecked() {
    let s = black_box(s3());
    let e = black_box(sc3((0x80000000, -0x40000000, 0x20000000)));
    assert!(s.inverse_unchecked() == e);
}

#[test]
#[inline(never)]
fn bench_scale3_try_inverse__pseudo() {
    let s = black_box(s3());
    let e = black_box(sc3((0x80000000, -0x40000000, 0x20000000)));
    assert!(s.pseudo_inverse() == e);
}

#[test]
#[inline(never)]
fn bench_scale3_try_inverse__in_place() {
    let mut s = black_box(s3());
    let e = black_box(sc3((0x80000000, -0x40000000, 0x20000000)));
    assert!(s.try_inverse_mut() && s == e);
}

// --- inverse point transform: one division per coordinate against reciprocal then product

#[test]
#[inline(never)]
fn bench_scale3_try_inverse_transform_point__baseline() {
    let _s = black_box(s3());
    let _p = black_box(pt3((6 * ONE, -20 * ONE, -56 * ONE)));
    let e = black_box(pt3(p3()));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale3_try_inverse_transform_point__division() {
    let s = black_box(s3());
    let p = black_box(pt3((6 * ONE, -20 * ONE, -56 * ONE)));
    let e = black_box(pt3(p3()));
    assert!(s.try_inverse_transform_point(p).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_scale3_try_inverse_transform_point__alt_reciprocal_product() {
    let s = black_box(s3());
    let p = black_box((6 * ONE, -20 * ONE, -56 * ONE));
    let e = black_box(pt3(p3()));
    assert!(alt_reciprocal_product(s, p) == e);
}

// --- products, homogeneous matrix

#[test]
#[inline(never)]
fn bench_scale3_mul__baseline() {
    let _a = black_box(s3());
    let _b = black_box(s3());
    let e = black_box(sc3((4 * ONE, 16 * ONE, 64 * ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale3_mul__component_product() {
    let a = black_box(s3());
    let b = black_box(s3());
    let e = black_box(sc3((4 * ONE, 16 * ONE, 64 * ONE)));
    assert!(a * b == e);
}

#[test]
#[inline(never)]
fn bench_scale3_mul_vector__baseline() {
    let _s = black_box(s3());
    let _v = black_box(vec3((ONE, ONE, ONE)));
    let e = black_box(vec3((2 * ONE, -4 * ONE, 8 * ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale3_mul_vector__component_product() {
    let s = black_box(s3());
    let v = black_box(vec3((ONE, ONE, ONE)));
    let e = black_box(vec3((2 * ONE, -4 * ONE, 8 * ONE)));
    assert!(s.mul_vector(v) == e);
}

#[test]
#[inline(never)]
fn bench_scale3_to_homogeneous__baseline() {
    let _s = black_box(s3());
    let e = black_box(int(1));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_scale3_to_homogeneous__diagonal() {
    let s = black_box(s3());
    let m = s.to_homogeneous();
    assert!(
        m.m11 == fx(2 * ONE) && m.m22 == fx(-4 * ONE) && m.m33 == fx(8 * ONE) && m.m44 == int(1),
    );
}
