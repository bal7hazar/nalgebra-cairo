//! Package-local helpers: the report of an oracle case, the errors of points in raw units.

use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra_tests_utils::{ONE_RAW, excess, fx, p3, ulp_diff};

/// The excess of `err` over `tol`, printed when positive.
pub fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

/// Largest component-wise error in raw units of the point `p` against the raw tuple `e`.
pub fn p2_err(p: Point2<Fixed>, e: (i64, i64)) -> u128 {
    let (x, y) = e;
    core::cmp::max(ulp_diff(p.x, fx(x)), ulp_diff(p.y, fx(y)))
}

/// Largest component-wise error in raw units of the point `p` against the raw tuple `e`.
pub fn p3_err(p: Point3<Fixed>, e: (i64, i64, i64)) -> u128 {
    let (x, y, z) = e;
    core::cmp::max(core::cmp::max(ulp_diff(p.x, fx(x)), ulp_diff(p.y, fx(y))), ulp_diff(p.z, fx(z)))
}

/// `Point3` from integer coordinates.
pub fn p3i(x: i64, y: i64, z: i64) -> Point3<Fixed> {
    p3(x * ONE_RAW, y * ONE_RAW, z * ONE_RAW)
}
