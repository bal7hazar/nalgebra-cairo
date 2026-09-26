//! `Point2..4` <-> the glam vectors (upstream `third_party/glam/common/glam_point.rs`).
//!
//! `Vec2..4` <-> `Point2..4<Fixed>`, `IVec2..4` <-> `Point2..4<i32>`, `UVec2..4` <->
//! `Point2..4<u32>`
//! and `BVec2..4` <-> `Point2..4<bool>`: the components in order, exact, no arithmetic. glam-cairo
//! has no `Vec3A` (the aligned type of upstream's `Vec3A` impls) nor `f64` vectors (`DVec*`):
//! those impls stay excluded (`interop`).
//!
//! Impls of `Into` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_point::Point3FromVec3;` (or the whole prelude, `use
//! nalgebra_glam::prelude::*;`).

use fixed::Fixed;
use glam::{BVec2, BVec3, BVec4, IVec2, IVec3, IVec4, UVec2, UVec3, UVec4, Vec2, Vec3, Vec4};
use nalgebra::{Point2, Point3, Point4};

/// The `Point2<Fixed>` with the components of `Vec2`, in order (`x, y`). Exact.
/// Upstream: `From<Vec2> for Point2<f32>`.
pub impl Point2FromVec2 of Into<Vec2, Point2<Fixed>> {
    #[inline(always)]
    fn into(self: Vec2) -> Point2<Fixed> {
        Point2 { x: self.x, y: self.y }
    }
}

/// The `Vec2` with the components of `Point2<Fixed>`, in order. Exact. Upstream:
/// `From<Point2<f32>> for Vec2`.
pub impl Point2IntoVec2 of Into<Point2<Fixed>, Vec2> {
    #[inline(always)]
    fn into(self: Point2<Fixed>) -> Vec2 {
        Vec2 { x: self.x, y: self.y }
    }
}

/// The `Point3<Fixed>` with the components of `Vec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<Vec3> for Point3<f32>`.
pub impl Point3FromVec3 of Into<Vec3, Point3<Fixed>> {
    #[inline(always)]
    fn into(self: Vec3) -> Point3<Fixed> {
        Point3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Vec3` with the components of `Point3<Fixed>`, in order. Exact. Upstream:
/// `From<Point3<f32>> for Vec3`.
pub impl Point3IntoVec3 of Into<Point3<Fixed>, Vec3> {
    #[inline(always)]
    fn into(self: Point3<Fixed>) -> Vec3 {
        Vec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Point4<Fixed>` with the components of `Vec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<Vec4> for Point4<f32>`.
pub impl Point4FromVec4 of Into<Vec4, Point4<Fixed>> {
    #[inline(always)]
    fn into(self: Vec4) -> Point4<Fixed> {
        Point4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Vec4` with the components of `Point4<Fixed>`, in order. Exact. Upstream:
/// `From<Point4<f32>> for Vec4`.
pub impl Point4IntoVec4 of Into<Point4<Fixed>, Vec4> {
    #[inline(always)]
    fn into(self: Point4<Fixed>) -> Vec4 {
        Vec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Point2<i32>` with the components of `IVec2`, in order (`x, y`). Exact.
/// Upstream: `From<IVec2> for Point2<i32>`.
pub impl Point2FromIVec2 of Into<IVec2, Point2<i32>> {
    #[inline(always)]
    fn into(self: IVec2) -> Point2<i32> {
        Point2 { x: self.x, y: self.y }
    }
}

/// The `IVec2` with the components of `Point2<i32>`, in order. Exact. Upstream:
/// `From<Point2<i32>> for IVec2`.
pub impl Point2IntoIVec2 of Into<Point2<i32>, IVec2> {
    #[inline(always)]
    fn into(self: Point2<i32>) -> IVec2 {
        IVec2 { x: self.x, y: self.y }
    }
}

/// The `Point3<i32>` with the components of `IVec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<IVec3> for Point3<i32>`.
pub impl Point3FromIVec3 of Into<IVec3, Point3<i32>> {
    #[inline(always)]
    fn into(self: IVec3) -> Point3<i32> {
        Point3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `IVec3` with the components of `Point3<i32>`, in order. Exact. Upstream:
/// `From<Point3<i32>> for IVec3`.
pub impl Point3IntoIVec3 of Into<Point3<i32>, IVec3> {
    #[inline(always)]
    fn into(self: Point3<i32>) -> IVec3 {
        IVec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Point4<i32>` with the components of `IVec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<IVec4> for Point4<i32>`.
pub impl Point4FromIVec4 of Into<IVec4, Point4<i32>> {
    #[inline(always)]
    fn into(self: IVec4) -> Point4<i32> {
        Point4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `IVec4` with the components of `Point4<i32>`, in order. Exact. Upstream:
/// `From<Point4<i32>> for IVec4`.
pub impl Point4IntoIVec4 of Into<Point4<i32>, IVec4> {
    #[inline(always)]
    fn into(self: Point4<i32>) -> IVec4 {
        IVec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Point2<u32>` with the components of `UVec2`, in order (`x, y`). Exact.
/// Upstream: `From<UVec2> for Point2<u32>`.
pub impl Point2FromUVec2 of Into<UVec2, Point2<u32>> {
    #[inline(always)]
    fn into(self: UVec2) -> Point2<u32> {
        Point2 { x: self.x, y: self.y }
    }
}

/// The `UVec2` with the components of `Point2<u32>`, in order. Exact. Upstream:
/// `From<Point2<u32>> for UVec2`.
pub impl Point2IntoUVec2 of Into<Point2<u32>, UVec2> {
    #[inline(always)]
    fn into(self: Point2<u32>) -> UVec2 {
        UVec2 { x: self.x, y: self.y }
    }
}

/// The `Point3<u32>` with the components of `UVec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<UVec3> for Point3<u32>`.
pub impl Point3FromUVec3 of Into<UVec3, Point3<u32>> {
    #[inline(always)]
    fn into(self: UVec3) -> Point3<u32> {
        Point3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `UVec3` with the components of `Point3<u32>`, in order. Exact. Upstream:
/// `From<Point3<u32>> for UVec3`.
pub impl Point3IntoUVec3 of Into<Point3<u32>, UVec3> {
    #[inline(always)]
    fn into(self: Point3<u32>) -> UVec3 {
        UVec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Point4<u32>` with the components of `UVec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<UVec4> for Point4<u32>`.
pub impl Point4FromUVec4 of Into<UVec4, Point4<u32>> {
    #[inline(always)]
    fn into(self: UVec4) -> Point4<u32> {
        Point4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `UVec4` with the components of `Point4<u32>`, in order. Exact. Upstream:
/// `From<Point4<u32>> for UVec4`.
pub impl Point4IntoUVec4 of Into<Point4<u32>, UVec4> {
    #[inline(always)]
    fn into(self: Point4<u32>) -> UVec4 {
        UVec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `Point2<bool>` with the components of `BVec2`, in order (`x, y`). Exact.
/// Upstream: `From<BVec2> for Point2<bool>`.
pub impl Point2FromBVec2 of Into<BVec2, Point2<bool>> {
    #[inline(always)]
    fn into(self: BVec2) -> Point2<bool> {
        Point2 { x: self.x, y: self.y }
    }
}

/// The `BVec2` with the components of `Point2<bool>`, in order. Exact. Upstream:
/// `From<Point2<bool>> for BVec2`.
pub impl Point2IntoBVec2 of Into<Point2<bool>, BVec2> {
    #[inline(always)]
    fn into(self: Point2<bool>) -> BVec2 {
        BVec2 { x: self.x, y: self.y }
    }
}

/// The `Point3<bool>` with the components of `BVec3`, in order (`x, y, z`). Exact.
/// Upstream: `From<BVec3> for Point3<bool>`.
pub impl Point3FromBVec3 of Into<BVec3, Point3<bool>> {
    #[inline(always)]
    fn into(self: BVec3) -> Point3<bool> {
        Point3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `BVec3` with the components of `Point3<bool>`, in order. Exact. Upstream:
/// `From<Point3<bool>> for BVec3`.
pub impl Point3IntoBVec3 of Into<Point3<bool>, BVec3> {
    #[inline(always)]
    fn into(self: Point3<bool>) -> BVec3 {
        BVec3 { x: self.x, y: self.y, z: self.z }
    }
}

/// The `Point4<bool>` with the components of `BVec4`, in order (`x, y, z, w`). Exact.
/// Upstream: `From<BVec4> for Point4<bool>`.
pub impl Point4FromBVec4 of Into<BVec4, Point4<bool>> {
    #[inline(always)]
    fn into(self: BVec4) -> Point4<bool> {
        Point4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}

/// The `BVec4` with the components of `Point4<bool>`, in order. Exact. Upstream:
/// `From<Point4<bool>> for BVec4`.
pub impl Point4IntoBVec4 of Into<Point4<bool>, BVec4> {
    #[inline(always)]
    fn into(self: Point4<bool>) -> BVec4 {
        BVec4 { x: self.x, y: self.y, z: self.z, w: self.w }
    }
}
