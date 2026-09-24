//! `AbstractRotation`: the operations every rotation type provides (upstream
//! `nalgebra::AbstractRotation`, `geometry/abstract_rotation.rs`), implemented by `Rotation2`,
//! `Rotation3`, `UnitComplex` and `UnitQuaternion` like upstream.
//!
//! Upstream's `AbstractRotation<T, D>` is generic over the dimension and names the vector and
//! point types through it; the Cairo trait is generic over the rotation type `R` and names them
//! with the associated types `Vector` and `Point` (`Vector2` / `Point2` for the 2D rotations,
//! `Vector3` / `Point3` for the 3D ones). Every method forwards to the rotation's own method of the
//! same name (no arithmetic here): the costs and roundings are those of `Rotation2Trait` & co.
//!
//! The methods have the names of the inherent methods: with both `AbstractRotation` and, say,
//! `Rotation3Trait` imported, `r.inverse()` is ambiguous — import the trait alone in generic
//! code, or call it by path (`AbstractRotation::inverse(r)`).

use simba::scalar::Real;
use crate::base::point2::Point2;
use crate::base::point3::Point3;
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use super::rotation2::{Rotation2, Rotation2Trait};
use super::rotation3::{Rotation3, Rotation3Trait};
use super::unit_complex::{UnitComplex, UnitComplexTrait};
use super::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};

/// The operations of a rotation of a `Vector` / `Point` space, with upstream's method set.
/// Upstream: `AbstractRotation<T, D>`.
pub trait AbstractRotation<R> {
    /// The vectors the rotation acts on (`Vector2<T>` or `Vector3<T>`).
    type Vector;
    /// The points the rotation acts on (`Point2<T>` or `Point3<T>`).
    type Point;
    /// The identity rotation. Upstream: `AbstractRotation::identity`.
    fn identity() -> R;
    /// The inverse rotation. Upstream: `AbstractRotation::inverse`.
    fn inverse(self: R) -> R;
    /// Inverts the rotation in place. Upstream: `AbstractRotation::inverse_mut`.
    fn inverse_mut(ref self: R);
    /// `self * v`. Upstream: `AbstractRotation::transform_vector`.
    fn transform_vector(self: R, v: Self::Vector) -> Self::Vector;
    /// `self * p`. Upstream: `AbstractRotation::transform_point`.
    fn transform_point(self: R, p: Self::Point) -> Self::Point;
    /// `self⁻¹ * v`. Upstream: `AbstractRotation::inverse_transform_vector`.
    fn inverse_transform_vector(self: R, v: Self::Vector) -> Self::Vector;
    /// `self⁻¹ * v` for a unit vector, not renormalised. Upstream:
    /// `AbstractRotation::inverse_transform_unit_vector`.
    fn inverse_transform_unit_vector(self: R, v: Unit<Self::Vector>) -> Unit<Self::Vector>;
    /// `self⁻¹ * p`. Upstream: `AbstractRotation::inverse_transform_point`.
    fn inverse_transform_point(self: R, p: Self::Point) -> Self::Point;
}

/// `AbstractRotation` for `Rotation2`: forwards to `Rotation2Trait`. Upstream: `AbstractRotation<T,
/// 2> for Rotation2`.
pub impl Rotation2AbstractRotation<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of AbstractRotation<Rotation2<T>> {
    type Vector = Vector2<T>;
    type Point = Point2<T>;

    #[inline(always)]
    fn identity() -> Rotation2<T> {
        Rotation2Trait::identity()
    }

    #[inline(always)]
    fn inverse(self: Rotation2<T>) -> Rotation2<T> {
        Rotation2Trait::inverse(self)
    }

    #[inline(always)]
    fn inverse_mut(ref self: Rotation2<T>) {
        self = Rotation2Trait::inverse(self);
    }

    #[inline(always)]
    fn transform_vector(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T> {
        Rotation2Trait::transform_vector(self, v)
    }

    #[inline(always)]
    fn transform_point(self: Rotation2<T>, p: Point2<T>) -> Point2<T> {
        Rotation2Trait::transform_point(self, p)
    }

    #[inline(always)]
    fn inverse_transform_vector(self: Rotation2<T>, v: Vector2<T>) -> Vector2<T> {
        Rotation2Trait::inverse_transform_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_unit_vector(self: Rotation2<T>, v: Unit<Vector2<T>>) -> Unit<Vector2<T>> {
        Rotation2Trait::inverse_transform_unit_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_point(self: Rotation2<T>, p: Point2<T>) -> Point2<T> {
        Rotation2Trait::inverse_transform_point(self, p)
    }
}

/// `AbstractRotation` for `Rotation3`: forwards to `Rotation3Trait`. Upstream: `AbstractRotation<T,
/// 3> for Rotation3`.
pub impl Rotation3AbstractRotation<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of AbstractRotation<Rotation3<T>> {
    type Vector = Vector3<T>;
    type Point = Point3<T>;

    #[inline(always)]
    fn identity() -> Rotation3<T> {
        Rotation3Trait::identity()
    }

    #[inline(always)]
    fn inverse(self: Rotation3<T>) -> Rotation3<T> {
        Rotation3Trait::inverse(self)
    }

    #[inline(always)]
    fn inverse_mut(ref self: Rotation3<T>) {
        self = Rotation3Trait::inverse(self);
    }

    #[inline(always)]
    fn transform_vector(self: Rotation3<T>, v: Vector3<T>) -> Vector3<T> {
        Rotation3Trait::transform_vector(self, v)
    }

    #[inline(always)]
    fn transform_point(self: Rotation3<T>, p: Point3<T>) -> Point3<T> {
        Rotation3Trait::transform_point(self, p)
    }

    #[inline(always)]
    fn inverse_transform_vector(self: Rotation3<T>, v: Vector3<T>) -> Vector3<T> {
        Rotation3Trait::inverse_transform_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_unit_vector(self: Rotation3<T>, v: Unit<Vector3<T>>) -> Unit<Vector3<T>> {
        Rotation3Trait::inverse_transform_unit_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_point(self: Rotation3<T>, p: Point3<T>) -> Point3<T> {
        Rotation3Trait::inverse_transform_point(self, p)
    }
}

/// `AbstractRotation` for `UnitComplex`: forwards to `UnitComplexTrait`. Upstream:
/// `AbstractRotation<T, 2> for UnitComplex`.
pub impl UnitComplexAbstractRotation<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of AbstractRotation<UnitComplex<T>> {
    type Vector = Vector2<T>;
    type Point = Point2<T>;

    #[inline(always)]
    fn identity() -> UnitComplex<T> {
        UnitComplexTrait::identity()
    }

    #[inline(always)]
    fn inverse(self: UnitComplex<T>) -> UnitComplex<T> {
        UnitComplexTrait::inverse(self)
    }

    #[inline(always)]
    fn inverse_mut(ref self: UnitComplex<T>) {
        self = UnitComplexTrait::inverse(self);
    }

    #[inline(always)]
    fn transform_vector(self: UnitComplex<T>, v: Vector2<T>) -> Vector2<T> {
        UnitComplexTrait::transform_vector(self, v)
    }

    #[inline(always)]
    fn transform_point(self: UnitComplex<T>, p: Point2<T>) -> Point2<T> {
        UnitComplexTrait::transform_point(self, p)
    }

    #[inline(always)]
    fn inverse_transform_vector(self: UnitComplex<T>, v: Vector2<T>) -> Vector2<T> {
        UnitComplexTrait::inverse_transform_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: UnitComplex<T>, v: Unit<Vector2<T>>,
    ) -> Unit<Vector2<T>> {
        UnitComplexTrait::inverse_transform_unit_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_point(self: UnitComplex<T>, p: Point2<T>) -> Point2<T> {
        UnitComplexTrait::inverse_transform_point(self, p)
    }
}

/// `AbstractRotation` for `UnitQuaternion`: forwards to `UnitQuaternionTrait`. Upstream:
/// `AbstractRotation<T, 3> for UnitQuaternion`.
pub impl UnitQuaternionAbstractRotation<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of AbstractRotation<UnitQuaternion<T>> {
    type Vector = Vector3<T>;
    type Point = Point3<T>;

    #[inline(always)]
    fn identity() -> UnitQuaternion<T> {
        UnitQuaternionTrait::identity()
    }

    #[inline(always)]
    fn inverse(self: UnitQuaternion<T>) -> UnitQuaternion<T> {
        UnitQuaternionTrait::inverse(self)
    }

    #[inline(always)]
    fn inverse_mut(ref self: UnitQuaternion<T>) {
        self = UnitQuaternionTrait::inverse(self);
    }

    #[inline(always)]
    fn transform_vector(self: UnitQuaternion<T>, v: Vector3<T>) -> Vector3<T> {
        UnitQuaternionTrait::transform_vector(self, v)
    }

    #[inline(always)]
    fn transform_point(self: UnitQuaternion<T>, p: Point3<T>) -> Point3<T> {
        UnitQuaternionTrait::transform_point(self, p)
    }

    #[inline(always)]
    fn inverse_transform_vector(self: UnitQuaternion<T>, v: Vector3<T>) -> Vector3<T> {
        UnitQuaternionTrait::inverse_transform_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_unit_vector(
        self: UnitQuaternion<T>, v: Unit<Vector3<T>>,
    ) -> Unit<Vector3<T>> {
        UnitQuaternionTrait::inverse_transform_unit_vector(self, v)
    }

    #[inline(always)]
    fn inverse_transform_point(self: UnitQuaternion<T>, p: Point3<T>) -> Point3<T> {
        UnitQuaternionTrait::inverse_transform_point(self, p)
    }
}
