//! Package-local helpers: the component-wise errors of projections against the oracle's raw
//! entries, in raw units, and the report of an oracle case.

use fixed::Fixed;
use nalgebra::base::point3::Point3;
use nalgebra::geometry::orthographic3::{Orthographic3, Orthographic3Trait};
use nalgebra::geometry::perspective3::{Perspective3, Perspective3Trait};
use nalgebra_tests_utils::{ONE_RAW, excess, max_ulp_diff4, ortho6t, p3, pers4t};

/// Largest error in raw units over the WHOLE matrix of `got` against the perspective of entries
/// `e` (so a stray non-structural entry is caught too).
pub fn pers_err(got: Perspective3<Fixed>, e: (i64, i64, i64, i64)) -> u128 {
    max_ulp_diff4(got.into_inner(), pers4t(e).into_inner())
}

/// Largest error in raw units over the WHOLE matrix of `got` against the orthographic projection
/// of entries `e`.
pub fn ortho_err(got: Orthographic3<Fixed>, e: (i64, i64, i64, i64, i64, i64)) -> u128 {
    max_ulp_diff4(got.into_inner(), ortho6t(e).into_inner())
}

/// The excess of `err` over `tol`, printed when positive.
pub fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

/// `Point3` from integer coordinates.
pub fn p3i(x: i64, y: i64, z: i64) -> Point3<Fixed> {
    p3(x * ONE_RAW, y * ONE_RAW, z * ONE_RAW)
}
