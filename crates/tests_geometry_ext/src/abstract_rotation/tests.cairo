//! Unit tests of `AbstractRotation`: every method of the four implementations agrees bit for bit
//! with the rotation's own method (the trait only forwards), called through generic code the way
//! upstream's `Isometry<T, R, D>` uses it.

use fixed::Fixed;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::abstract_rotation::AbstractRotation;
use nalgebra::geometry::rotation2::{Rotation2, Rotation2Trait};
use nalgebra::geometry::rotation3::{Rotation3, Rotation3Trait};
use nalgebra::geometry::unit_complex::{UnitComplex, UnitComplexTrait};
use nalgebra::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use nalgebra_tests_utils::{ONE_RAW, fx, p2, p3, u2, u3, uc, uq, v2, v3};

/// `r` applied then undone through the trait only (`r⁻¹ (r v)`, exact for the rotations of the
/// tests), and whether the in-place inverse equals `inverse`.
fn round_trip<
    R,
    impl A: AbstractRotation<R>,
    +Copy<R>,
    +Drop<R>,
    +PartialEq<R>,
    +Copy<A::Vector>,
    +Drop<A::Vector>,
>(
    r: R, v: A::Vector,
) -> (A::Vector, bool) {
    let back = A::inverse_transform_vector(r, A::transform_vector(r, v));
    let mut m = r;
    A::inverse_mut(ref m);
    (back, m == A::inverse(r))
}

#[test]
fn test_rotation2() {
    let r = Rotation2 {
        matrix: Matrix2 { m11: fx(0), m21: fx(ONE_RAW), m12: fx(-ONE_RAW), m22: fx(0) },
    };
    let v = v2(0x100000000, 0x200000000);
    let p = p2(0x100000000, 0x200000000);
    let u = u2(0, ONE_RAW);
    assert!(AbstractRotation::transform_vector(r, v) == Rotation2Trait::transform_vector(r, v));
    assert!(AbstractRotation::transform_point(r, p) == Rotation2Trait::transform_point(r, p));
    assert!(
        AbstractRotation::inverse_transform_point(
            r, p,
        ) == Rotation2Trait::inverse_transform_point(r, p),
    );
    assert!(
        AbstractRotation::inverse_transform_unit_vector(r, u)
            .value == Rotation2Trait::inverse_transform_vector(r, u.value),
    );
    let id: Rotation2<Fixed> = AbstractRotation::identity();
    assert!(id == Rotation2Trait::identity());
    let (back, ok) = round_trip(r, v);
    assert!(ok && back == v);
}

#[test]
fn test_rotation3() {
    let q = uq(0x80000000, 0x80000000, 0x80000000, 0x80000000);
    let r: Rotation3<Fixed> = q.into();
    let v = v3(0x100000000, 0x200000000, -0x300000000);
    let p = p3(0x100000000, 0x200000000, -0x300000000);
    let u: Unit<Vector3<Fixed>> = u3(0, 0, ONE_RAW);
    assert!(AbstractRotation::transform_vector(r, v) == Rotation3Trait::transform_vector(r, v));
    assert!(AbstractRotation::transform_point(r, p) == Rotation3Trait::transform_point(r, p));
    assert!(
        AbstractRotation::inverse_transform_point(
            r, p,
        ) == Rotation3Trait::inverse_transform_point(r, p),
    );
    assert!(
        AbstractRotation::inverse_transform_unit_vector(r, u)
            .value == Rotation3Trait::inverse_transform_vector(r, u.value),
    );
    let (back, ok) = round_trip(r, v);
    assert!(ok && back == v);
}

#[test]
fn test_unit_complex() {
    let c = uc(0, ONE_RAW);
    let v = v2(0x100000000, 0x200000000);
    let p: Point2<Fixed> = p2(0x100000000, 0x200000000);
    assert!(AbstractRotation::transform_vector(c, v) == UnitComplexTrait::transform_vector(c, v));
    assert!(AbstractRotation::transform_point(c, p) == UnitComplexTrait::transform_point(c, p));
    let id: UnitComplex<Fixed> = AbstractRotation::identity();
    assert!(id == UnitComplexTrait::identity());
    let (back, ok) = round_trip(c, v);
    assert!(ok && back == v);
    let w: Vector2<Fixed> = AbstractRotation::inverse_transform_vector(c, v);
    assert!(w == UnitComplexTrait::inverse_transform_vector(c, v));
}

#[test]
fn test_unit_quaternion() {
    let q = uq(0x80000000, 0x80000000, 0x80000000, 0x80000000);
    let v = v3(0x100000000, 0x200000000, -0x300000000);
    let p: Point3<Fixed> = p3(0x100000000, 0x200000000, -0x300000000);
    assert!(
        AbstractRotation::transform_vector(q, v) == UnitQuaternionTrait::transform_vector(q, v),
    );
    assert!(
        AbstractRotation::inverse_transform_point(
            q, p,
        ) == UnitQuaternionTrait::inverse_transform_point(q, p),
    );
    let id: UnitQuaternion<Fixed> = AbstractRotation::identity();
    assert!(id == UnitQuaternionTrait::identity());
    let (back, ok) = round_trip(q, v);
    assert!(ok && back == v);
}
