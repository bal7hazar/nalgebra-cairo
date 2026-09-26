//! `UnitComplex` <-> `Mat2` (upstream `third_party/glam/common/glam_unit_complex.rs`).
//!
//! `Mat2` -> `UnitComplex` reads the FIRST COLUMN of the matrix (`x_axis`, glam is column-major:
//! the `(cos, sin)` pair of a rotation) and normalizes it, like upstream's
//! `UnitComplex::new_normalize(Complex::new(e.x_axis.x, e.x_axis.y))` (`Real::norm2` floored once,
//! two correctly rounded divisions); it panics with `Fixed: division by zero` when that column is
//! zero (upstream: NaN). `UnitComplex` -> `Mat2` is the rotation matrix `[[re, -im], [im, re]]`,
//! exact. glam-cairo has no `DMat2` (`f64`): those impls stay excluded (`interop`).
//!
//! Impls of `Into` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_unit_complex::UnitComplexFromMat2;` (or `use nalgebra_glam::prelude::*;`).

use fixed::Fixed;
use glam::{Mat2, Vec2};
use nalgebra::{UnitComplex, UnitComplexTrait, Vector2};

/// The rotation matrix of a `UnitComplex<Fixed>`, `x_axis = (re, im)`, `y_axis = (-im, re)`.
/// Exact. Upstream: `From<UnitComplex<f32>> for Mat2` (`to_rotation_matrix().into_inner()`).
pub impl UnitComplexIntoMat2 of Into<UnitComplex<Fixed>, Mat2> {
    #[inline(always)]
    fn into(self: UnitComplex<Fixed>) -> Mat2 {
        Mat2 { x_axis: Vec2 { x: self.re, y: self.im }, y_axis: Vec2 { x: -self.im, y: self.re } }
    }
}

/// The unit complex of the direction of the first column of a `Mat2` (see the module
/// documentation for the rounding and the panic). Upstream: `From<Mat2> for UnitComplex<f32>`.
pub impl UnitComplexFromMat2 of Into<Mat2, UnitComplex<Fixed>> {
    #[inline(always)]
    fn into(self: Mat2) -> UnitComplex<Fixed> {
        UnitComplexTrait::new_normalize(Vector2 { x: self.x_axis.x, y: self.x_axis.y })
    }
}
