//! `Translation2..4` <-> `Vec2..4` (upstream `third_party/glam/common/glam_translation.rs`).
//!
//! The offset of the translation, in order, exact, no arithmetic. Upstream's `f64` (`DVec*`) and
//! aligned (`Vec3A`) impls stay excluded (`interop`): glam-cairo has neither.
//!
//! Impls of `Into` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_translation::Translation3FromVec3;` (or `use nalgebra_glam::prelude::*;`).

use fixed::Fixed;
use glam::{Vec2, Vec3, Vec4};
use nalgebra::{Translation2, Translation3, Translation4, Vector2, Vector3, Vector4};

/// The `Translation2<Fixed>` by `Vec2`. Exact. Upstream: `From<Vec2> for Translation2<f32>`.
pub impl Translation2FromVec2 of Into<Vec2, Translation2<Fixed>> {
    #[inline(always)]
    fn into(self: Vec2) -> Translation2<Fixed> {
        Translation2 { vector: Vector2 { x: self.x, y: self.y } }
    }
}

/// The `Vec2` of the offset of a `Translation2<Fixed>`. Exact. Upstream: `From<Translation2<f32>>
/// for Vec2`.
pub impl Translation2IntoVec2 of Into<Translation2<Fixed>, Vec2> {
    #[inline(always)]
    fn into(self: Translation2<Fixed>) -> Vec2 {
        Vec2 { x: self.vector.x, y: self.vector.y }
    }
}

/// The `Translation3<Fixed>` by `Vec3`. Exact. Upstream: `From<Vec3> for Translation3<f32>`.
pub impl Translation3FromVec3 of Into<Vec3, Translation3<Fixed>> {
    #[inline(always)]
    fn into(self: Vec3) -> Translation3<Fixed> {
        Translation3 { vector: Vector3 { x: self.x, y: self.y, z: self.z } }
    }
}

/// The `Vec3` of the offset of a `Translation3<Fixed>`. Exact. Upstream: `From<Translation3<f32>>
/// for Vec3`.
pub impl Translation3IntoVec3 of Into<Translation3<Fixed>, Vec3> {
    #[inline(always)]
    fn into(self: Translation3<Fixed>) -> Vec3 {
        Vec3 { x: self.vector.x, y: self.vector.y, z: self.vector.z }
    }
}

/// The `Translation4<Fixed>` by `Vec4`. Exact. Upstream: `From<Vec4> for Translation4<f32>`.
pub impl Translation4FromVec4 of Into<Vec4, Translation4<Fixed>> {
    #[inline(always)]
    fn into(self: Vec4) -> Translation4<Fixed> {
        Translation4 { vector: Vector4 { x: self.x, y: self.y, z: self.z, w: self.w } }
    }
}

/// The `Vec4` of the offset of a `Translation4<Fixed>`. Exact. Upstream: `From<Translation4<f32>>
/// for Vec4`.
pub impl Translation4IntoVec4 of Into<Translation4<Fixed>, Vec4> {
    #[inline(always)]
    fn into(self: Translation4<Fixed>) -> Vec4 {
        Vec4 { x: self.vector.x, y: self.vector.y, z: self.vector.z, w: self.vector.w }
    }
}
