//! Package-local helpers of the oracle assertions: the excess over the tolerance (printed) and the
//! component-wise errors of the poses, in raw units.

use core::cmp::max;
use fixed::Fixed;
use nalgebra::geometry::isometry2::Isometry2;
use nalgebra::geometry::isometry3::Isometry3;
use nalgebra::geometry::isometry_matrix2::IsometryMatrix2;
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3;
use nalgebra::geometry::similarity2::Similarity2;
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::similarity_matrix2::SimilarityMatrix2;
use nalgebra::geometry::similarity_matrix3::SimilarityMatrix3;
use nalgebra_tests_utils::{
    excess, iso2t, iso3t, isom2t, isom3t, max_ulp_diff2, max_ulp_diff3, max_ulp_diff_q,
    max_ulp_diff_uc, max_ulp_diff_v2, max_ulp_diff_v3, sim2t, sim3t, ulp_diff,
};

/// The excess of `err` over `tol`, printed when positive.
pub fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

pub fn isom2_err(got: IsometryMatrix2<Fixed>, t: (i64, i64), r: [[i64; 2]; 2]) -> u128 {
    let e = isom2t(t, r);
    max(
        max_ulp_diff_v2(got.translation.vector, e.translation.vector),
        max_ulp_diff2(got.rotation.matrix, e.rotation.matrix),
    )
}

pub fn isom3_err(got: IsometryMatrix3<Fixed>, t: (i64, i64, i64), r: [[i64; 3]; 3]) -> u128 {
    let e = isom3t(t, r);
    max(
        max_ulp_diff_v3(got.translation.vector, e.translation.vector),
        max_ulp_diff3(got.rotation.matrix, e.rotation.matrix),
    )
}

pub fn simm2_err(got: SimilarityMatrix2<Fixed>, t: (i64, i64), r: [[i64; 2]; 2], s: i64) -> u128 {
    max(isom2_err(got.isometry, t, r), ulp_diff(got.scaling, nalgebra_tests_utils::fx(s)))
}

pub fn simm3_err(
    got: SimilarityMatrix3<Fixed>, t: (i64, i64, i64), r: [[i64; 3]; 3], s: i64,
) -> u128 {
    max(isom3_err(got.isometry, t, r), ulp_diff(got.scaling, nalgebra_tests_utils::fx(s)))
}

pub fn iso2_err(got: Isometry2<Fixed>, e: ((i64, i64), (i64, i64))) -> u128 {
    let e = iso2t(e);
    max(
        max_ulp_diff_v2(got.translation.vector, e.translation.vector),
        max_ulp_diff_uc(got.rotation, e.rotation),
    )
}

pub fn iso3_err(got: Isometry3<Fixed>, e: ((i64, i64, i64), (i64, i64, i64, i64))) -> u128 {
    let e = iso3t(e);
    max(
        max_ulp_diff_v3(got.translation.vector, e.translation.vector),
        max_ulp_diff_q(got.rotation.quaternion, e.rotation.quaternion),
    )
}

pub fn sim2_err(got: Similarity2<Fixed>, e: ((i64, i64), (i64, i64), i64)) -> u128 {
    let e = sim2t(e);
    max(
        max(
            max_ulp_diff_v2(got.isometry.translation.vector, e.isometry.translation.vector),
            max_ulp_diff_uc(got.isometry.rotation, e.isometry.rotation),
        ),
        ulp_diff(got.scaling, e.scaling),
    )
}

pub fn sim3_err(got: Similarity3<Fixed>, e: ((i64, i64, i64), (i64, i64, i64, i64), i64)) -> u128 {
    let e = sim3t(e);
    max(
        max(
            max_ulp_diff_v3(got.isometry.translation.vector, e.isometry.translation.vector),
            max_ulp_diff_q(got.isometry.rotation.quaternion, e.isometry.rotation.quaternion),
        ),
        ulp_diff(got.scaling, e.scaling),
    )
}
