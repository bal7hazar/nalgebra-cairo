//! In `nalgebra_geometry2`: the impl `Rotation2InternalImpl`. This module is split over packages;
//! the other parts are in `nalgebra_types3`.
//!
//! Internal, no stability promise: the crate-private items of `geometry::rotation2` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_types2::base::vector2::Vector2;
use nalgebra_types2::geometry::rotation2::Rotation2;
use nalgebra_types2::geometry::translation2::Translation2;
use simba::scalar::Real;
use crate::geometry::isometry_matrix2::IsometryMatrix2;
use crate::geometry::rotation2::Rotation2Trait;

/// Crate-internal by-value forms of the in-place `renormalize` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub impl Rotation2InternalImpl<
    T, impl R: Real<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Rotation2InternalTrait<T> {
    /// `iso.inverse()` (`IsometryMatrix2::inverse`: the transposed rotation and
    /// `rotationᵀ · (-translation)`), with the bounds of `Rotation2Trait`.
    #[inline(always)]
    fn inverse_isometry(iso: IsometryMatrix2<T>) -> IsometryMatrix2<T> {
        let t = iso.translation.vector;
        IsometryMatrix2 {
            rotation: Rotation2Trait::inverse(iso.rotation),
            translation: Translation2 {
                vector: Rotation2Trait::inverse_transform_vector(
                    iso.rotation, Vector2 { x: -t.x, y: -t.y },
                ),
            },
        }
    }

    /// `self` renormalized exactly (`Rotation2Trait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: Rotation2<T>) -> Rotation2<T> {
        let mut r = self;
        Rotation2Trait::renormalize(ref r);
        r
    }
}
