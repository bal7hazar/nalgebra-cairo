//! Unit tests of the translation types of WP 8.4-P09a: `Translation1`, `Translation4`,
//! `Translation5`, `Translation6` (the whole API, from one template) and the completion of
//! `Translation2` / `Translation3` (division, `One`, conversions, the heterogeneous products with
//! isometries, similarities and rotations, the approximate comparisons, `cast`). Everything is an
//! exact addition, subtraction or negation: the expected values are exact.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::vector4::Vector4;
use nalgebra::base::vector6::Vector6;
use nalgebra::geometry::isometry2::{Isometry2, Isometry2Trait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::point1::{Point1, Point1Trait};
use nalgebra::geometry::point4::{Point4, Point4Trait};
use nalgebra::geometry::point6::Point6Trait;
use nalgebra::geometry::similarity2::Similarity2;
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::translation1::{Translation1, Translation1Trait};
use nalgebra::geometry::translation2::{Translation2, Translation2Trait};
use nalgebra::geometry::translation3::{Translation3, Translation3Trait};
use nalgebra::geometry::translation4::{Translation4, Translation4Trait};
use nalgebra::geometry::translation5::{Translation5, Translation5Trait};
use nalgebra::geometry::translation6::{Translation6, Translation6Trait};
use nalgebra::geometry::unit_complex::UnitComplexTrait;
use nalgebra::geometry::unit_quaternion::UnitQuaternionTrait;
use nalgebra_tests_utils::{fx, int, iso2, iso3, p2, p3, sim2, sim3, t2, t3, uc, uq, v2i, v3i};
use simba::scalar::Real;

fn t4(x: i64, y: i64, z: i64, w: i64) -> Translation4<Fixed> {
    Translation4Trait::new(int(x), int(y), int(z), int(w))
}

// --- Translation4, the whole API

#[test]
fn test_translation4_api() {
    let t = t4(1, -2, 3, 4);
    assert!(t.vector == Vector4 { x: int(1), y: int(-2), z: int(3), w: int(4) });
    assert!(Translation4Trait::<Fixed>::identity() == t4(0, 0, 0, 0));
    assert!(Translation4Trait::from_vector(t.vector) == t);
    assert!(t.inverse() == t4(-1, 2, -3, -4));
    let p: Point4<Fixed> = Point4Trait::new(int(10), int(10), int(10), int(10));
    assert!(t.transform_point(p) == Point4Trait::new(int(11), int(8), int(13), int(14)));
    assert!(t.inverse_transform_point(t.transform_point(p)) == p);
    let h = t.to_homogeneous();
    assert!(h.m15 == int(1) && h.m25 == int(-2) && h.m45 == int(4) && h.m55 == int(1));
    assert!(h.m11 == int(1) && h.m12 == int(0) && h.m51 == int(0));
    assert!(t * t.inverse() == Translation4Trait::identity());
    assert!(t / t == Translation4Trait::identity());
    assert!(t4(5, 5, 5, 5) / t == t4(4, 7, 2, 1));
    let v: Translation4<Fixed> = t.vector.into();
    assert!(v == t);
    let from_point: Translation4<Fixed> = Point4Trait::new(int(1), int(-2), int(3), int(4)).into();
    assert!(from_point == t);
    let arr: [Fixed; 4] = t.into();
    let back: Translation4<Fixed> = arr.into();
    assert!(back == t);
    assert!(One::one() == Translation4Trait::<Fixed>::identity());
    assert!(Translation4Trait::<Fixed>::identity().is_one() && t.is_non_one());
    assert!(Default::default() == Translation4Trait::<Fixed>::identity());
}

#[test]
fn test_translation4_comparisons_and_cast() {
    let t = t4(1, -2, 3, 4);
    let d = Translation4 { vector: Vector4 { w: t.vector.w + fx(5), ..t.vector } };
    assert!(t.abs_diff_eq(d, 5) && !t.abs_diff_eq(d, 4));
    assert!(t.relative_eq(d, 5, Real::zero()) && !t.relative_eq(d, 4, Real::zero()));
    assert!(t.ulps_eq(d, 0, 5) && !t.ulps_eq(d, 4, 4));
    assert!(t.cast::<Fixed>() == t);
}

// --- the other sizes

#[test]
fn test_translation1() {
    let t: Translation1<Fixed> = Translation1Trait::new(int(3));
    let p: Point1<Fixed> = Point1Trait::new(int(4));
    assert!(t.transform_point(p) == Point1Trait::new(int(7)));
    let h = t.to_homogeneous();
    assert!(h == Matrix2 { m11: int(1), m21: int(0), m12: int(3), m22: int(1) });
    assert!(t / t == Translation1Trait::identity() && t.vector == Matrix1 { x: int(3) });
    let arr: [Fixed; 1] = t.into();
    let back: Translation1<Fixed> = arr.into();
    assert!(back == t && t.relative_eq(t, 0, Real::zero()));
}

#[test]
fn test_translation5() {
    let t: Translation5<Fixed> = Translation5Trait::new(int(1), int(2), int(3), int(4), int(5));
    let h = t.to_homogeneous();
    assert!(h.m16 == int(1) && h.m56 == int(5) && h.m66 == int(1) && h.m55 == int(1));
    assert!(h.m65 == int(0) && h.m21 == int(0));
    assert!(t * t.inverse() == Translation5Trait::identity());
    assert!(t.ulps_eq(t, 0, 0));
}

#[test]
fn test_translation6() {
    let t: Translation6<Fixed> = Translation6Trait::new(
        int(1), int(2), int(3), int(4), int(5), int(6),
    );
    assert!(
        t.vector == Vector6 { x: int(1), y: int(2), z: int(3), w: int(4), a: int(5), b: int(6) },
    );
    let p = Point6Trait::new(int(1), int(1), int(1), int(1), int(1), int(1));
    assert!(t.inverse_transform_point(t.transform_point(p)) == p);
    assert!(t / t == Translation6Trait::identity() && (t * t).vector.b == int(12));
    let arr: [Fixed; 6] = t.into();
    let back: Translation6<Fixed> = arr.into();
    assert!(back == t && t.cast::<Fixed>() == t);
}

// --- Translation2 / Translation3 completion

#[test]
fn test_translation2_completion() {
    let t = t2(0x100000000, -0x200000000);
    assert!(t / t == Translation2Trait::identity());
    let one: Translation2<Fixed> = One::one();
    assert!(one == Translation2Trait::identity() && one.is_one() && t.is_non_one());
    let p = p2(0x100000000, -0x200000000);
    let from_point: Translation2<Fixed> = p.into();
    assert!(from_point == t);
    let arr: [Fixed; 2] = t.into();
    let back: Translation2<Fixed> = arr.into();
    assert!(back == t);
    let d = Translation2 {
        vector: nalgebra::base::vector2::Vector2 { x: t.vector.x + fx(1), y: t.vector.y },
    };
    assert!(t.relative_eq(d, 1, Real::zero()) && !t.relative_eq(d, 0, Real::zero()));
    assert!(t.ulps_eq(d, 0, 1) && t.cast::<Fixed>() == t);
    // Heterogeneous products.
    let c = uc(0, 0x100000000);
    let iso = t.mul_unit_complex(c);
    assert!(iso == Isometry2 { rotation: c, translation: t });
    let i = iso2(0x300000000, 0x100000000, 0, 0x100000000);
    let ti = t.mul_isometry(i);
    assert!(ti.rotation == i.rotation && ti.translation.vector == v2i(4, -1));
    let s = sim2(0x300000000, 0x100000000, 0, 0x100000000, 0x200000000);
    let ts = t.mul_similarity(s);
    assert!(ts.isometry == ti && ts.scaling == int(2));
    // Conversions.
    let as_iso: Isometry2<Fixed> = t.into();
    assert!(as_iso.rotation == UnitComplexTrait::identity() && as_iso.translation == t);
    let as_sim: Similarity2<Fixed> = t.into();
    assert!(as_sim.isometry == as_iso && as_sim.scaling == Real::one());
    // `t * iso` transforms like the composition.
    let q = Point2 { x: int(1), y: int(2) };
    assert!(ti.transform_point(q) == t.transform_point(i.transform_point(q)));
}

#[test]
fn test_translation3_completion() {
    let t = t3(0x100000000, -0x200000000, 0x300000000);
    assert!(t / t3(0, 0x100000000, 0) == t3(0x100000000, -0x300000000, 0x300000000));
    let one: Translation3<Fixed> = One::one();
    assert!(one == Translation3Trait::identity());
    let from_point: Translation3<Fixed> = p3(0x100000000, -0x200000000, 0x300000000).into();
    assert!(from_point == t);
    let arr: [Fixed; 3] = t.into();
    let back: Translation3<Fixed> = arr.into();
    assert!(back == t && t.relative_eq(t, 0, Real::zero()) && t.ulps_eq(t, 0, 0));
    let q = uq(0x80000000, 0x80000000, 0x80000000, 0x80000000);
    assert!(t.mul_unit_quaternion(q) == Isometry3 { rotation: q, translation: t });
    // `Isometry3::abs_diff_eq` compares the rotations up to their sign, like upstream.
    assert!(Isometry3 { rotation: -q, translation: t }.abs_diff_eq(t.mul_unit_quaternion(q), 0));
    let i = iso3((0, 0, 0x100000000), (0x80000000, 0x80000000, 0x80000000, 0x80000000));
    let ti = t.mul_isometry(i);
    assert!(ti.rotation == i.rotation && ti.translation.vector == v3i(1, -2, 4));
    let s = sim3((0, 0, 0x100000000), (0x80000000, 0x80000000, 0x80000000, 0x80000000), 0x40000000);
    assert!(t.mul_similarity(s) == Similarity3 { isometry: ti, scaling: fx(0x40000000) });
    let as_iso: Isometry3<Fixed> = t.into();
    assert!(as_iso.rotation == UnitQuaternionTrait::identity() && as_iso.translation == t);
    let as_sim: Similarity3<Fixed> = t.into();
    assert!(as_sim.isometry == as_iso && as_sim.scaling == Real::one());
    let p = Point3 { x: int(1), y: int(2), z: int(3) };
    assert!(ti.transform_point(p) == t.transform_point(i.transform_point(p)));
}

#[test]
#[should_panic]
fn test_translation4_div_overflow_panics() {
    let t = Translation4 {
        vector: Vector4 { x: int(0), y: int(0), z: int(0), w: fx(-0x7fffffffffffffff) },
    };
    let _ = t / t4(0, 0, 0, 1);
}
