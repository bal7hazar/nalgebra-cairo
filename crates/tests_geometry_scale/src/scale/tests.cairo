//! Tests of `Scale1` .. `Scale6` (WP 8.4-P10) through the public API: the whole API on each
//! size (exact cases on powers of two, whose inverses are exact), the zero-factor behaviour of the
//! inverses, the homogeneous matrix and the oracle vectors of `tools/oracle` (suite
//! `scale_reflection`, upstream nalgebra 0.35 on the same raw inputs, tolerance in ulp).

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::matrix3::Matrix3;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::matrix5::Matrix5;
use nalgebra::base::matrix6::Matrix6;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::base::vector4::Vector4;
use nalgebra::base::vector5::Vector5;
use nalgebra::base::vector6::Vector6;
use nalgebra::geometry::point1::Point1;
use nalgebra::geometry::point4::Point4;
use nalgebra::geometry::point5::Point5;
use nalgebra::geometry::point6::Point6;
use nalgebra::geometry::scale1::{Scale1, Scale1Trait};
use nalgebra::geometry::scale2::{Scale2, Scale2Trait};
use nalgebra::geometry::scale3::{Scale3, Scale3Trait};
use nalgebra::geometry::scale4::{Scale4, Scale4Trait};
use nalgebra::geometry::scale5::{Scale5, Scale5Trait};
use nalgebra::geometry::scale6::{Scale6, Scale6Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int};
use simba::scalar::Real;
use crate::common::{
    err_mat2x1, err_mat3x1, err_mat5x1, err_pt2, err_pt3, err_pt5, pt2, pt3, pt5, sc2, sc3, sc5,
    vec2, vec3, vec5,
};
use crate::oracle;

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = nalgebra_tests_utils::excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

// --- Scale1

fn s1() -> Scale1<Fixed> {
    Scale1Trait::new(int(2))
}

fn s1_inv() -> Scale1<Fixed> {
    Scale1Trait::new(fx(2147483648))
}

fn p1() -> Point1<Fixed> {
    Point1 { x: int(1) }
}

#[test]
fn test_scale1_construction_and_conversions() {
    let s = s1();
    assert!(s.vector == Matrix1 { x: int(2) });
    assert!(Scale1Trait::<Fixed>::identity().vector == Matrix1 { x: int(1) });
    let from_vector: Scale1<Fixed> = s.vector.into();
    assert!(from_vector == s);
    let from_point: Scale1<Fixed> = (Point1 { x: int(2) }).into();
    assert!(from_point == s);
    let arr: [Fixed; 1] = s.into();
    let back: Scale1<Fixed> = arr.into();
    assert!(back == s);
    assert!(One::one() == Scale1Trait::<Fixed>::identity());
    assert!(Scale1Trait::<Fixed>::identity().is_one() && s.is_non_one());
    assert!(s.cast::<Fixed>() == s);
}

#[test]
fn test_scale1_products() {
    let s = s1();
    let p = p1();
    let e = Point1 { x: fx(8589934592) };
    assert!(s.transform_point(p) == e);
    assert!(s.mul_vector(Matrix1 { x: int(1) }) == Matrix1 { x: fx(8589934592) });
    // `s * s`: the squares of the factors.
    let sq = s * s;
    assert!(sq.vector.x == s.vector.x * s.vector.x);
    assert!(s * s1_inv() == Scale1Trait::identity() && s1_inv() * s == Scale1Trait::identity());
    let k = s.scale(int(-3));
    assert!(k.vector.x == s.vector.x * int(-3));
    let mut a = s;
    a *= s;
    assert!(a == sq);
    let mut b = s;
    b *= int(-3);
    assert!(b == k);
}

#[test]
fn test_scale1_inverse() {
    let s = s1();
    assert!(s.try_inverse().unwrap() == s1_inv());
    assert!(s.inverse_unchecked() == s1_inv());
    assert!(s.pseudo_inverse() == s1_inv());
    let mut t = s;
    assert!(t.try_inverse_mut());
    assert!(t == s1_inv());
    let p = p1();
    assert!(s.try_inverse_transform_point(s.transform_point(p)).unwrap() == p);
    // A zero factor: no inverse, the pseudo-inverse keeps a zero there, nothing changes in place.
    let z = Scale1 { vector: Matrix1 { x: int(0) } };
    assert!(z.try_inverse().is_none());
    assert!(z.try_inverse_transform_point(p).is_none());
    let mut u = z;
    assert!(!u.try_inverse_mut() && u == z);
    let pi = z.pseudo_inverse();
    assert!(pi.vector.x == int(0));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_scale1_inverse_unchecked_of_a_zero_factor_panics() {
    let z = black_box(Scale1 { vector: Matrix1 { x: int(0) } });
    let _ = z.inverse_unchecked();
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale1_transform_point_overflow_panics() {
    let s = black_box(Scale1Trait::new(int(100000)));
    let p = black_box(Point1 { x: int(100000) });
    let _ = s.transform_point(p);
}

#[test]
fn test_scale1_comparisons() {
    let s = s1();
    let d = Scale1 { vector: Matrix1 { x: int(2) + fx(5) } };
    assert!(s.abs_diff_eq(d, 5) && !s.abs_diff_eq(d, 4));
    assert!(s.relative_eq(d, 5, Real::zero()) && !s.relative_eq(d, 4, Real::zero()));
    assert!(s.ulps_eq(d, 0, 5) && !s.ulps_eq(d, 4, 4));
}

#[test]
fn test_scale1_to_homogeneous() {
    let s = s1();
    let m = s.to_homogeneous();
    assert!(m.m11 == s.vector.x);
    assert!(m.m12 == int(0));
    assert!(m.m21 == int(0));
    assert!(m.m22 == int(1));
    let converted: Matrix2<Fixed> = s.into();
    assert!(converted == m);
}

// --- Scale2

fn s2() -> Scale2<Fixed> {
    Scale2Trait::new(int(2), int(-4))
}

fn s2_inv() -> Scale2<Fixed> {
    Scale2Trait::new(fx(2147483648), fx(-1073741824))
}

fn p2() -> Point2<Fixed> {
    Point2 { x: int(1), y: int(2) }
}

#[test]
fn test_scale2_construction_and_conversions() {
    let s = s2();
    assert!(s.vector == Vector2 { x: int(2), y: int(-4) });
    assert!(Scale2Trait::<Fixed>::identity().vector == Vector2 { x: int(1), y: int(1) });
    let from_vector: Scale2<Fixed> = s.vector.into();
    assert!(from_vector == s);
    let from_point: Scale2<Fixed> = (Point2 { x: int(2), y: int(-4) }).into();
    assert!(from_point == s);
    let arr: [Fixed; 2] = s.into();
    let back: Scale2<Fixed> = arr.into();
    assert!(back == s);
    assert!(One::one() == Scale2Trait::<Fixed>::identity());
    assert!(Scale2Trait::<Fixed>::identity().is_one() && s.is_non_one());
    assert!(s.cast::<Fixed>() == s);
}

#[test]
fn test_scale2_products() {
    let s = s2();
    let p = p2();
    let e = Point2 { x: fx(8589934592), y: fx(-34359738368) };
    assert!(s.transform_point(p) == e);
    assert!(
        s
            .mul_vector(
                Vector2 { x: int(1), y: int(2) },
            ) == Vector2 { x: fx(8589934592), y: fx(-34359738368) },
    );
    // `s * s`: the squares of the factors.
    let sq = s * s;
    assert!(sq.vector.x == s.vector.x * s.vector.x);
    assert!(sq.vector.y == s.vector.y * s.vector.y);
    assert!(s * s2_inv() == Scale2Trait::identity() && s2_inv() * s == Scale2Trait::identity());
    let k = s.scale(int(-3));
    assert!(k.vector.x == s.vector.x * int(-3));
    assert!(k.vector.y == s.vector.y * int(-3));
    let mut a = s;
    a *= s;
    assert!(a == sq);
    let mut b = s;
    b *= int(-3);
    assert!(b == k);
}

#[test]
fn test_scale2_inverse() {
    let s = s2();
    assert!(s.try_inverse().unwrap() == s2_inv());
    assert!(s.inverse_unchecked() == s2_inv());
    assert!(s.pseudo_inverse() == s2_inv());
    let mut t = s;
    assert!(t.try_inverse_mut());
    assert!(t == s2_inv());
    let p = p2();
    assert!(s.try_inverse_transform_point(s.transform_point(p)).unwrap() == p);
    // A zero factor: no inverse, the pseudo-inverse keeps a zero there, nothing changes in place.
    let z = Scale2 { vector: Vector2 { x: int(2), y: int(0) } };
    assert!(z.try_inverse().is_none());
    assert!(z.try_inverse_transform_point(p).is_none());
    let mut u = z;
    assert!(!u.try_inverse_mut() && u == z);
    let pi = z.pseudo_inverse();
    assert!(pi.vector.y == int(0));
    assert!(pi.vector.x == s2_inv().vector.x);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_scale2_inverse_unchecked_of_a_zero_factor_panics() {
    let z = black_box(Scale2 { vector: Vector2 { x: int(2), y: int(0) } });
    let _ = z.inverse_unchecked();
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale2_transform_point_overflow_panics() {
    let s = black_box(Scale2Trait::new(int(100000), int(100000)));
    let p = black_box(Point2 { x: int(100000), y: int(100000) });
    let _ = s.transform_point(p);
}

#[test]
fn test_scale2_comparisons() {
    let s = s2();
    let d = Scale2 { vector: Vector2 { x: int(2), y: int(-4) + fx(5) } };
    assert!(s.abs_diff_eq(d, 5) && !s.abs_diff_eq(d, 4));
    assert!(s.relative_eq(d, 5, Real::zero()) && !s.relative_eq(d, 4, Real::zero()));
    assert!(s.ulps_eq(d, 0, 5) && !s.ulps_eq(d, 4, 4));
}

#[test]
fn test_scale2_to_homogeneous() {
    let s = s2();
    let m = s.to_homogeneous();
    assert!(m.m11 == s.vector.x);
    assert!(m.m12 == int(0));
    assert!(m.m13 == int(0));
    assert!(m.m21 == int(0));
    assert!(m.m22 == s.vector.y);
    assert!(m.m23 == int(0));
    assert!(m.m31 == int(0));
    assert!(m.m32 == int(0));
    assert!(m.m33 == int(1));
    let converted: Matrix3<Fixed> = s.into();
    assert!(converted == m);
}

// --- Scale3

fn s3() -> Scale3<Fixed> {
    Scale3Trait::new(int(2), int(-4), int(8))
}

fn s3_inv() -> Scale3<Fixed> {
    Scale3Trait::new(fx(2147483648), fx(-1073741824), fx(536870912))
}

fn p3() -> Point3<Fixed> {
    Point3 { x: int(1), y: int(2), z: int(3) }
}

#[test]
fn test_scale3_construction_and_conversions() {
    let s = s3();
    assert!(s.vector == Vector3 { x: int(2), y: int(-4), z: int(8) });
    assert!(Scale3Trait::<Fixed>::identity().vector == Vector3 { x: int(1), y: int(1), z: int(1) });
    let from_vector: Scale3<Fixed> = s.vector.into();
    assert!(from_vector == s);
    let from_point: Scale3<Fixed> = (Point3 { x: int(2), y: int(-4), z: int(8) }).into();
    assert!(from_point == s);
    let arr: [Fixed; 3] = s.into();
    let back: Scale3<Fixed> = arr.into();
    assert!(back == s);
    assert!(One::one() == Scale3Trait::<Fixed>::identity());
    assert!(Scale3Trait::<Fixed>::identity().is_one() && s.is_non_one());
    assert!(s.cast::<Fixed>() == s);
}

#[test]
fn test_scale3_products() {
    let s = s3();
    let p = p3();
    let e = Point3 { x: fx(8589934592), y: fx(-34359738368), z: fx(103079215104) };
    assert!(s.transform_point(p) == e);
    assert!(
        s
            .mul_vector(
                Vector3 { x: int(1), y: int(2), z: int(3) },
            ) == Vector3 { x: fx(8589934592), y: fx(-34359738368), z: fx(103079215104) },
    );
    // `s * s`: the squares of the factors.
    let sq = s * s;
    assert!(sq.vector.x == s.vector.x * s.vector.x);
    assert!(sq.vector.y == s.vector.y * s.vector.y);
    assert!(sq.vector.z == s.vector.z * s.vector.z);
    assert!(s * s3_inv() == Scale3Trait::identity() && s3_inv() * s == Scale3Trait::identity());
    let k = s.scale(int(-3));
    assert!(k.vector.x == s.vector.x * int(-3));
    assert!(k.vector.y == s.vector.y * int(-3));
    assert!(k.vector.z == s.vector.z * int(-3));
    let mut a = s;
    a *= s;
    assert!(a == sq);
    let mut b = s;
    b *= int(-3);
    assert!(b == k);
}

#[test]
fn test_scale3_inverse() {
    let s = s3();
    assert!(s.try_inverse().unwrap() == s3_inv());
    assert!(s.inverse_unchecked() == s3_inv());
    assert!(s.pseudo_inverse() == s3_inv());
    let mut t = s;
    assert!(t.try_inverse_mut());
    assert!(t == s3_inv());
    let p = p3();
    assert!(s.try_inverse_transform_point(s.transform_point(p)).unwrap() == p);
    // A zero factor: no inverse, the pseudo-inverse keeps a zero there, nothing changes in place.
    let z = Scale3 { vector: Vector3 { x: int(2), y: int(-4), z: int(0) } };
    assert!(z.try_inverse().is_none());
    assert!(z.try_inverse_transform_point(p).is_none());
    let mut u = z;
    assert!(!u.try_inverse_mut() && u == z);
    let pi = z.pseudo_inverse();
    assert!(pi.vector.z == int(0));
    assert!(pi.vector.x == s3_inv().vector.x);
    assert!(pi.vector.y == s3_inv().vector.y);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_scale3_inverse_unchecked_of_a_zero_factor_panics() {
    let z = black_box(Scale3 { vector: Vector3 { x: int(2), y: int(-4), z: int(0) } });
    let _ = z.inverse_unchecked();
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale3_transform_point_overflow_panics() {
    let s = black_box(Scale3Trait::new(int(100000), int(100000), int(100000)));
    let p = black_box(Point3 { x: int(100000), y: int(100000), z: int(100000) });
    let _ = s.transform_point(p);
}

#[test]
fn test_scale3_comparisons() {
    let s = s3();
    let d = Scale3 { vector: Vector3 { x: int(2), y: int(-4), z: int(8) + fx(5) } };
    assert!(s.abs_diff_eq(d, 5) && !s.abs_diff_eq(d, 4));
    assert!(s.relative_eq(d, 5, Real::zero()) && !s.relative_eq(d, 4, Real::zero()));
    assert!(s.ulps_eq(d, 0, 5) && !s.ulps_eq(d, 4, 4));
}

#[test]
fn test_scale3_to_homogeneous() {
    let s = s3();
    let m = s.to_homogeneous();
    assert!(m.m11 == s.vector.x);
    assert!(m.m12 == int(0));
    assert!(m.m13 == int(0));
    assert!(m.m14 == int(0));
    assert!(m.m21 == int(0));
    assert!(m.m22 == s.vector.y);
    assert!(m.m23 == int(0));
    assert!(m.m24 == int(0));
    assert!(m.m31 == int(0));
    assert!(m.m32 == int(0));
    assert!(m.m33 == s.vector.z);
    assert!(m.m34 == int(0));
    assert!(m.m41 == int(0));
    assert!(m.m42 == int(0));
    assert!(m.m43 == int(0));
    assert!(m.m44 == int(1));
    let converted: Matrix4<Fixed> = s.into();
    assert!(converted == m);
}

// --- Scale4

fn s4() -> Scale4<Fixed> {
    Scale4Trait::new(int(2), int(-4), int(8), fx(0x80000000))
}

fn s4_inv() -> Scale4<Fixed> {
    Scale4Trait::new(fx(2147483648), fx(-1073741824), fx(536870912), fx(8589934592))
}

fn p4() -> Point4<Fixed> {
    Point4 { x: int(1), y: int(2), z: int(3), w: int(4) }
}

#[test]
fn test_scale4_construction_and_conversions() {
    let s = s4();
    assert!(s.vector == Vector4 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000) });
    assert!(
        Scale4Trait::<Fixed>::identity()
            .vector == Vector4 { x: int(1), y: int(1), z: int(1), w: int(1) },
    );
    let from_vector: Scale4<Fixed> = s.vector.into();
    assert!(from_vector == s);
    let from_point: Scale4<Fixed> = (Point4 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000) })
        .into();
    assert!(from_point == s);
    let arr: [Fixed; 4] = s.into();
    let back: Scale4<Fixed> = arr.into();
    assert!(back == s);
    assert!(One::one() == Scale4Trait::<Fixed>::identity());
    assert!(Scale4Trait::<Fixed>::identity().is_one() && s.is_non_one());
    assert!(s.cast::<Fixed>() == s);
}

#[test]
fn test_scale4_products() {
    let s = s4();
    let p = p4();
    let e = Point4 {
        x: fx(8589934592), y: fx(-34359738368), z: fx(103079215104), w: fx(8589934592),
    };
    assert!(s.transform_point(p) == e);
    assert!(
        s
            .mul_vector(
                Vector4 { x: int(1), y: int(2), z: int(3), w: int(4) },
            ) == Vector4 {
                x: fx(8589934592), y: fx(-34359738368), z: fx(103079215104), w: fx(8589934592),
            },
    );
    // `s * s`: the squares of the factors.
    let sq = s * s;
    assert!(sq.vector.x == s.vector.x * s.vector.x);
    assert!(sq.vector.y == s.vector.y * s.vector.y);
    assert!(sq.vector.z == s.vector.z * s.vector.z);
    assert!(sq.vector.w == s.vector.w * s.vector.w);
    assert!(s * s4_inv() == Scale4Trait::identity() && s4_inv() * s == Scale4Trait::identity());
    let k = s.scale(int(-3));
    assert!(k.vector.x == s.vector.x * int(-3));
    assert!(k.vector.y == s.vector.y * int(-3));
    assert!(k.vector.z == s.vector.z * int(-3));
    assert!(k.vector.w == s.vector.w * int(-3));
    let mut a = s;
    a *= s;
    assert!(a == sq);
    let mut b = s;
    b *= int(-3);
    assert!(b == k);
}

#[test]
fn test_scale4_inverse() {
    let s = s4();
    assert!(s.try_inverse().unwrap() == s4_inv());
    assert!(s.inverse_unchecked() == s4_inv());
    assert!(s.pseudo_inverse() == s4_inv());
    let mut t = s;
    assert!(t.try_inverse_mut());
    assert!(t == s4_inv());
    let p = p4();
    assert!(s.try_inverse_transform_point(s.transform_point(p)).unwrap() == p);
    // A zero factor: no inverse, the pseudo-inverse keeps a zero there, nothing changes in place.
    let z = Scale4 { vector: Vector4 { x: int(2), y: int(-4), z: int(8), w: int(0) } };
    assert!(z.try_inverse().is_none());
    assert!(z.try_inverse_transform_point(p).is_none());
    let mut u = z;
    assert!(!u.try_inverse_mut() && u == z);
    let pi = z.pseudo_inverse();
    assert!(pi.vector.w == int(0));
    assert!(pi.vector.x == s4_inv().vector.x);
    assert!(pi.vector.y == s4_inv().vector.y);
    assert!(pi.vector.z == s4_inv().vector.z);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_scale4_inverse_unchecked_of_a_zero_factor_panics() {
    let z = black_box(Scale4 { vector: Vector4 { x: int(2), y: int(-4), z: int(8), w: int(0) } });
    let _ = z.inverse_unchecked();
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale4_transform_point_overflow_panics() {
    let s = black_box(Scale4Trait::new(int(100000), int(100000), int(100000), int(100000)));
    let p = black_box(Point4 { x: int(100000), y: int(100000), z: int(100000), w: int(100000) });
    let _ = s.transform_point(p);
}

#[test]
fn test_scale4_comparisons() {
    let s = s4();
    let d = Scale4 {
        vector: Vector4 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000) + fx(5) },
    };
    assert!(s.abs_diff_eq(d, 5) && !s.abs_diff_eq(d, 4));
    assert!(s.relative_eq(d, 5, Real::zero()) && !s.relative_eq(d, 4, Real::zero()));
    assert!(s.ulps_eq(d, 0, 5) && !s.ulps_eq(d, 4, 4));
}

#[test]
fn test_scale4_to_homogeneous() {
    let s = s4();
    let m = s.to_homogeneous();
    assert!(m.m11 == s.vector.x);
    assert!(m.m12 == int(0));
    assert!(m.m13 == int(0));
    assert!(m.m14 == int(0));
    assert!(m.m15 == int(0));
    assert!(m.m21 == int(0));
    assert!(m.m22 == s.vector.y);
    assert!(m.m23 == int(0));
    assert!(m.m24 == int(0));
    assert!(m.m25 == int(0));
    assert!(m.m31 == int(0));
    assert!(m.m32 == int(0));
    assert!(m.m33 == s.vector.z);
    assert!(m.m34 == int(0));
    assert!(m.m35 == int(0));
    assert!(m.m41 == int(0));
    assert!(m.m42 == int(0));
    assert!(m.m43 == int(0));
    assert!(m.m44 == s.vector.w);
    assert!(m.m45 == int(0));
    assert!(m.m51 == int(0));
    assert!(m.m52 == int(0));
    assert!(m.m53 == int(0));
    assert!(m.m54 == int(0));
    assert!(m.m55 == int(1));
    let converted: Matrix5<Fixed> = s.into();
    assert!(converted == m);
}

// --- Scale5

fn s5() -> Scale5<Fixed> {
    Scale5Trait::new(int(2), int(-4), int(8), fx(0x80000000), int(16))
}

fn s5_inv() -> Scale5<Fixed> {
    Scale5Trait::new(fx(2147483648), fx(-1073741824), fx(536870912), fx(8589934592), fx(268435456))
}

fn p5() -> Point5<Fixed> {
    Point5 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5) }
}

#[test]
fn test_scale5_construction_and_conversions() {
    let s = s5();
    assert!(
        s.vector == Vector5 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16) },
    );
    assert!(
        Scale5Trait::<Fixed>::identity()
            .vector == Vector5 { x: int(1), y: int(1), z: int(1), w: int(1), a: int(1) },
    );
    let from_vector: Scale5<Fixed> = s.vector.into();
    assert!(from_vector == s);
    let from_point: Scale5<Fixed> = (Point5 {
        x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16),
    })
        .into();
    assert!(from_point == s);
    let arr: [Fixed; 5] = s.into();
    let back: Scale5<Fixed> = arr.into();
    assert!(back == s);
    assert!(One::one() == Scale5Trait::<Fixed>::identity());
    assert!(Scale5Trait::<Fixed>::identity().is_one() && s.is_non_one());
    assert!(s.cast::<Fixed>() == s);
}

#[test]
fn test_scale5_products() {
    let s = s5();
    let p = p5();
    let e = Point5 {
        x: fx(8589934592),
        y: fx(-34359738368),
        z: fx(103079215104),
        w: fx(8589934592),
        a: fx(343597383680),
    };
    assert!(s.transform_point(p) == e);
    assert!(
        s
            .mul_vector(
                Vector5 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5) },
            ) == Vector5 {
                x: fx(8589934592),
                y: fx(-34359738368),
                z: fx(103079215104),
                w: fx(8589934592),
                a: fx(343597383680),
            },
    );
    // `s * s`: the squares of the factors.
    let sq = s * s;
    assert!(sq.vector.x == s.vector.x * s.vector.x);
    assert!(sq.vector.y == s.vector.y * s.vector.y);
    assert!(sq.vector.z == s.vector.z * s.vector.z);
    assert!(sq.vector.w == s.vector.w * s.vector.w);
    assert!(sq.vector.a == s.vector.a * s.vector.a);
    assert!(s * s5_inv() == Scale5Trait::identity() && s5_inv() * s == Scale5Trait::identity());
    let k = s.scale(int(-3));
    assert!(k.vector.x == s.vector.x * int(-3));
    assert!(k.vector.y == s.vector.y * int(-3));
    assert!(k.vector.z == s.vector.z * int(-3));
    assert!(k.vector.w == s.vector.w * int(-3));
    assert!(k.vector.a == s.vector.a * int(-3));
    let mut a = s;
    a *= s;
    assert!(a == sq);
    let mut b = s;
    b *= int(-3);
    assert!(b == k);
}

#[test]
fn test_scale5_inverse() {
    let s = s5();
    assert!(s.try_inverse().unwrap() == s5_inv());
    assert!(s.inverse_unchecked() == s5_inv());
    assert!(s.pseudo_inverse() == s5_inv());
    let mut t = s;
    assert!(t.try_inverse_mut());
    assert!(t == s5_inv());
    let p = p5();
    assert!(s.try_inverse_transform_point(s.transform_point(p)).unwrap() == p);
    // A zero factor: no inverse, the pseudo-inverse keeps a zero there, nothing changes in place.
    let z = Scale5 {
        vector: Vector5 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(0) },
    };
    assert!(z.try_inverse().is_none());
    assert!(z.try_inverse_transform_point(p).is_none());
    let mut u = z;
    assert!(!u.try_inverse_mut() && u == z);
    let pi = z.pseudo_inverse();
    assert!(pi.vector.a == int(0));
    assert!(pi.vector.x == s5_inv().vector.x);
    assert!(pi.vector.y == s5_inv().vector.y);
    assert!(pi.vector.z == s5_inv().vector.z);
    assert!(pi.vector.w == s5_inv().vector.w);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_scale5_inverse_unchecked_of_a_zero_factor_panics() {
    let z = black_box(
        Scale5 {
            vector: Vector5 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(0) },
        },
    );
    let _ = z.inverse_unchecked();
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale5_transform_point_overflow_panics() {
    let s = black_box(
        Scale5Trait::new(int(100000), int(100000), int(100000), int(100000), int(100000)),
    );
    let p = black_box(
        Point5 { x: int(100000), y: int(100000), z: int(100000), w: int(100000), a: int(100000) },
    );
    let _ = s.transform_point(p);
}

#[test]
fn test_scale5_comparisons() {
    let s = s5();
    let d = Scale5 {
        vector: Vector5 { x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16) + fx(5) },
    };
    assert!(s.abs_diff_eq(d, 5) && !s.abs_diff_eq(d, 4));
    assert!(s.relative_eq(d, 5, Real::zero()) && !s.relative_eq(d, 4, Real::zero()));
    assert!(s.ulps_eq(d, 0, 5) && !s.ulps_eq(d, 4, 4));
}

#[test]
fn test_scale5_to_homogeneous() {
    let s = s5();
    let m = s.to_homogeneous();
    assert!(m.m11 == s.vector.x);
    assert!(m.m12 == int(0));
    assert!(m.m13 == int(0));
    assert!(m.m14 == int(0));
    assert!(m.m15 == int(0));
    assert!(m.m16 == int(0));
    assert!(m.m21 == int(0));
    assert!(m.m22 == s.vector.y);
    assert!(m.m23 == int(0));
    assert!(m.m24 == int(0));
    assert!(m.m25 == int(0));
    assert!(m.m26 == int(0));
    assert!(m.m31 == int(0));
    assert!(m.m32 == int(0));
    assert!(m.m33 == s.vector.z);
    assert!(m.m34 == int(0));
    assert!(m.m35 == int(0));
    assert!(m.m36 == int(0));
    assert!(m.m41 == int(0));
    assert!(m.m42 == int(0));
    assert!(m.m43 == int(0));
    assert!(m.m44 == s.vector.w);
    assert!(m.m45 == int(0));
    assert!(m.m46 == int(0));
    assert!(m.m51 == int(0));
    assert!(m.m52 == int(0));
    assert!(m.m53 == int(0));
    assert!(m.m54 == int(0));
    assert!(m.m55 == s.vector.a);
    assert!(m.m56 == int(0));
    assert!(m.m61 == int(0));
    assert!(m.m62 == int(0));
    assert!(m.m63 == int(0));
    assert!(m.m64 == int(0));
    assert!(m.m65 == int(0));
    assert!(m.m66 == int(1));
    let converted: Matrix6<Fixed> = s.into();
    assert!(converted == m);
}

// --- Scale6

fn s6() -> Scale6<Fixed> {
    Scale6Trait::new(int(2), int(-4), int(8), fx(0x80000000), int(16), int(-1))
}

fn s6_inv() -> Scale6<Fixed> {
    Scale6Trait::new(
        fx(2147483648),
        fx(-1073741824),
        fx(536870912),
        fx(8589934592),
        fx(268435456),
        fx(-4294967296),
    )
}

fn p6() -> Point6<Fixed> {
    Point6 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5), b: int(6) }
}

#[test]
fn test_scale6_construction_and_conversions() {
    let s = s6();
    assert!(
        s
            .vector == Vector6 {
                x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16), b: int(-1),
            },
    );
    assert!(
        Scale6Trait::<Fixed>::identity()
            .vector == Vector6 { x: int(1), y: int(1), z: int(1), w: int(1), a: int(1), b: int(1) },
    );
    let from_vector: Scale6<Fixed> = s.vector.into();
    assert!(from_vector == s);
    let from_point: Scale6<Fixed> = (Point6 {
        x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16), b: int(-1),
    })
        .into();
    assert!(from_point == s);
    let arr: [Fixed; 6] = s.into();
    let back: Scale6<Fixed> = arr.into();
    assert!(back == s);
    assert!(One::one() == Scale6Trait::<Fixed>::identity());
    assert!(Scale6Trait::<Fixed>::identity().is_one() && s.is_non_one());
    assert!(s.cast::<Fixed>() == s);
}

#[test]
fn test_scale6_products() {
    let s = s6();
    let p = p6();
    let e = Point6 {
        x: fx(8589934592),
        y: fx(-34359738368),
        z: fx(103079215104),
        w: fx(8589934592),
        a: fx(343597383680),
        b: fx(-25769803776),
    };
    assert!(s.transform_point(p) == e);
    assert!(
        s
            .mul_vector(
                Vector6 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5), b: int(6) },
            ) == Vector6 {
                x: fx(8589934592),
                y: fx(-34359738368),
                z: fx(103079215104),
                w: fx(8589934592),
                a: fx(343597383680),
                b: fx(-25769803776),
            },
    );
    // `s * s`: the squares of the factors.
    let sq = s * s;
    assert!(sq.vector.x == s.vector.x * s.vector.x);
    assert!(sq.vector.y == s.vector.y * s.vector.y);
    assert!(sq.vector.z == s.vector.z * s.vector.z);
    assert!(sq.vector.w == s.vector.w * s.vector.w);
    assert!(sq.vector.a == s.vector.a * s.vector.a);
    assert!(sq.vector.b == s.vector.b * s.vector.b);
    assert!(s * s6_inv() == Scale6Trait::identity() && s6_inv() * s == Scale6Trait::identity());
    let k = s.scale(int(-3));
    assert!(k.vector.x == s.vector.x * int(-3));
    assert!(k.vector.y == s.vector.y * int(-3));
    assert!(k.vector.z == s.vector.z * int(-3));
    assert!(k.vector.w == s.vector.w * int(-3));
    assert!(k.vector.a == s.vector.a * int(-3));
    assert!(k.vector.b == s.vector.b * int(-3));
    let mut a = s;
    a *= s;
    assert!(a == sq);
    let mut b = s;
    b *= int(-3);
    assert!(b == k);
}

#[test]
fn test_scale6_inverse() {
    let s = s6();
    assert!(s.try_inverse().unwrap() == s6_inv());
    assert!(s.inverse_unchecked() == s6_inv());
    assert!(s.pseudo_inverse() == s6_inv());
    let mut t = s;
    assert!(t.try_inverse_mut());
    assert!(t == s6_inv());
    let p = p6();
    assert!(s.try_inverse_transform_point(s.transform_point(p)).unwrap() == p);
    // A zero factor: no inverse, the pseudo-inverse keeps a zero there, nothing changes in place.
    let z = Scale6 {
        vector: Vector6 {
            x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16), b: int(0),
        },
    };
    assert!(z.try_inverse().is_none());
    assert!(z.try_inverse_transform_point(p).is_none());
    let mut u = z;
    assert!(!u.try_inverse_mut() && u == z);
    let pi = z.pseudo_inverse();
    assert!(pi.vector.b == int(0));
    assert!(pi.vector.x == s6_inv().vector.x);
    assert!(pi.vector.y == s6_inv().vector.y);
    assert!(pi.vector.z == s6_inv().vector.z);
    assert!(pi.vector.w == s6_inv().vector.w);
    assert!(pi.vector.a == s6_inv().vector.a);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_scale6_inverse_unchecked_of_a_zero_factor_panics() {
    let z = black_box(
        Scale6 {
            vector: Vector6 {
                x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16), b: int(0),
            },
        },
    );
    let _ = z.inverse_unchecked();
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_scale6_transform_point_overflow_panics() {
    let s = black_box(
        Scale6Trait::new(
            int(100000), int(100000), int(100000), int(100000), int(100000), int(100000),
        ),
    );
    let p = black_box(
        Point6 {
            x: int(100000),
            y: int(100000),
            z: int(100000),
            w: int(100000),
            a: int(100000),
            b: int(100000),
        },
    );
    let _ = s.transform_point(p);
}

#[test]
fn test_scale6_comparisons() {
    let s = s6();
    let d = Scale6 {
        vector: Vector6 {
            x: int(2), y: int(-4), z: int(8), w: fx(0x80000000), a: int(16), b: int(-1) + fx(5),
        },
    };
    assert!(s.abs_diff_eq(d, 5) && !s.abs_diff_eq(d, 4));
    assert!(s.relative_eq(d, 5, Real::zero()) && !s.relative_eq(d, 4, Real::zero()));
    assert!(s.ulps_eq(d, 0, 5) && !s.ulps_eq(d, 4, 4));
}

// --- oracle vectors

#[test]
fn test_scale2_mul_oracle() {
    let mut cases = oracle::scale2_mul_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = sc2(a) * sc2(b);
        worst =
            core::cmp::max(worst, report("scale2_mul", n, err_mat2x1(got.vector, vec2(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale2_transform_point_oracle() {
    let mut cases = oracle::scale2_transform_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, p, e, tol) = *case;
        let got = sc2(s).transform_point(pt2(p));
        let e = pt2(e);
        worst = core::cmp::max(worst, report("scale2_transform_point", n, err_pt2(got, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale2_mul_vector_oracle() {
    let mut cases = oracle::scale2_mul_vector_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, v, e, tol) = *case;
        let got = sc2(s).mul_vector(vec2(v));
        worst =
            core::cmp::max(worst, report("scale2_mul_vector", n, err_mat2x1(got, vec2(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale2_scale_oracle() {
    let mut cases = oracle::scale2_scale_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, k, e, tol) = *case;
        let got = sc2(s).scale(fx(k));
        worst =
            core::cmp::max(worst, report("scale2_scale", n, err_mat2x1(got.vector, vec2(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale2_try_inverse_oracle() {
    let mut cases = oracle::scale2_try_inverse_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, e, tol) = *case;
        let got = sc2(s).try_inverse().unwrap();
        worst =
            core::cmp::max(
                worst, report("scale2_try_inverse", n, err_mat2x1(got.vector, vec2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale2_pseudo_inverse_oracle() {
    let mut cases = oracle::scale2_pseudo_inverse_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, e, tol) = *case;
        let got = sc2(s).pseudo_inverse();
        worst =
            core::cmp::max(
                worst, report("scale2_pseudo_inverse", n, err_mat2x1(got.vector, vec2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale2_try_inverse_transform_point_oracle() {
    let mut cases = oracle::scale2_try_inverse_transform_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, p, e, tol) = *case;
        let got = sc2(s).try_inverse_transform_point(pt2(p)).unwrap();
        worst =
            core::cmp::max(
                worst, report("scale2_try_inverse_transform_point", n, err_pt2(got, pt2(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_mul_oracle() {
    let mut cases = oracle::scale3_mul_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = sc3(a) * sc3(b);
        worst =
            core::cmp::max(worst, report("scale3_mul", n, err_mat3x1(got.vector, vec3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_transform_point_oracle() {
    let mut cases = oracle::scale3_transform_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, p, e, tol) = *case;
        let got = sc3(s).transform_point(pt3(p));
        let e = pt3(e);
        worst = core::cmp::max(worst, report("scale3_transform_point", n, err_pt3(got, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_mul_vector_oracle() {
    let mut cases = oracle::scale3_mul_vector_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, v, e, tol) = *case;
        let got = sc3(s).mul_vector(vec3(v));
        worst =
            core::cmp::max(worst, report("scale3_mul_vector", n, err_mat3x1(got, vec3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_scale_oracle() {
    let mut cases = oracle::scale3_scale_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, k, e, tol) = *case;
        let got = sc3(s).scale(fx(k));
        worst =
            core::cmp::max(worst, report("scale3_scale", n, err_mat3x1(got.vector, vec3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_try_inverse_oracle() {
    let mut cases = oracle::scale3_try_inverse_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, e, tol) = *case;
        let got = sc3(s).try_inverse().unwrap();
        worst =
            core::cmp::max(
                worst, report("scale3_try_inverse", n, err_mat3x1(got.vector, vec3(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_pseudo_inverse_oracle() {
    let mut cases = oracle::scale3_pseudo_inverse_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, e, tol) = *case;
        let got = sc3(s).pseudo_inverse();
        worst =
            core::cmp::max(
                worst, report("scale3_pseudo_inverse", n, err_mat3x1(got.vector, vec3(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale3_try_inverse_transform_point_oracle() {
    let mut cases = oracle::scale3_try_inverse_transform_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, p, e, tol) = *case;
        let got = sc3(s).try_inverse_transform_point(pt3(p)).unwrap();
        worst =
            core::cmp::max(
                worst, report("scale3_try_inverse_transform_point", n, err_pt3(got, pt3(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_mul_oracle() {
    let mut cases = oracle::scale5_mul_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let got = sc5(a) * sc5(b);
        worst =
            core::cmp::max(worst, report("scale5_mul", n, err_mat5x1(got.vector, vec5(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_transform_point_oracle() {
    let mut cases = oracle::scale5_transform_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, p, e, tol) = *case;
        let got = sc5(s).transform_point(pt5(p));
        let e = pt5(e);
        worst = core::cmp::max(worst, report("scale5_transform_point", n, err_pt5(got, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_mul_vector_oracle() {
    let mut cases = oracle::scale5_mul_vector_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, v, e, tol) = *case;
        let got = sc5(s).mul_vector(vec5(v));
        worst =
            core::cmp::max(worst, report("scale5_mul_vector", n, err_mat5x1(got, vec5(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_scale_oracle() {
    let mut cases = oracle::scale5_scale_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, k, e, tol) = *case;
        let got = sc5(s).scale(fx(k));
        worst =
            core::cmp::max(worst, report("scale5_scale", n, err_mat5x1(got.vector, vec5(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_try_inverse_oracle() {
    let mut cases = oracle::scale5_try_inverse_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, e, tol) = *case;
        let got = sc5(s).try_inverse().unwrap();
        worst =
            core::cmp::max(
                worst, report("scale5_try_inverse", n, err_mat5x1(got.vector, vec5(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_pseudo_inverse_oracle() {
    let mut cases = oracle::scale5_pseudo_inverse_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, e, tol) = *case;
        let got = sc5(s).pseudo_inverse();
        worst =
            core::cmp::max(
                worst, report("scale5_pseudo_inverse", n, err_mat5x1(got.vector, vec5(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_scale5_try_inverse_transform_point_oracle() {
    let mut cases = oracle::scale5_try_inverse_transform_point_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (s, p, e, tol) = *case;
        let got = sc5(s).try_inverse_transform_point(pt5(p)).unwrap();
        worst =
            core::cmp::max(
                worst, report("scale5_try_inverse_transform_point", n, err_pt5(got, pt5(e)), tol),
            );
        n += 1;
    }
    assert!(worst == 0);
}

/// Negative and fractional factors: the reciprocals are exact.
#[test]
fn test_scale3_try_inverse_of_negative_factors() {
    let s = sc3((-2 * 0x100000000, -0x80000000, 4 * 0x100000000));
    let e = sc3((-0x80000000, -2 * 0x100000000, 0x40000000));
    assert!(s.try_inverse().unwrap() == e && s.pseudo_inverse() == e);
}
