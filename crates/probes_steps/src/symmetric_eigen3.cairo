//! Probe of `SymmetricEigen3`: the eigendecomposition of a symmetric matrix. Input: a case of the
//! oracle `symmetric_eigen3_eigenvalues_spd` (upstream nalgebra on f64,
//! `nalgebra_tests_linalg::oracle_symmetric_eigen`); the asserted value is the exact result of the
//! kernel at this head (its eigenvalues agree with the oracle within its tolerance, checked when
//! the probe was written; the eigenvectors follow the sign convention of `SymmetricEigen3`).

use fixed::Fixed;
use nalgebra_linalg_svd_eigen3::linalg::symmetric_eigen3::{
    Matrix3SymmetricEigenTrait, SymmetricEigen3,
};
use nalgebra_testing::black_box;
use nalgebra_types3::base::matrix3::Matrix3;
use crate::builders::{m3, v3};
use crate::eq::SymmetricEigen3Eq;

fn a() -> Matrix3<Fixed> {
    m3(
        [
            [4317911657, 360208987, -398750113], [360208987, 4088348811, -684321863],
            [-398750113, -684321863, 4267097999],
        ],
    )
}

fn expected() -> SymmetricEigen3<Fixed> {
    SymmetricEigen3 {
        eigenvalues: v3((3487496255, 3989392718, 5196469485)),
        eigenvectors: m3(
            [
                [-52729894, 3663649488, 2240900727], [3240927190, -1436497133, 2424790921],
                [2817863479, 1720724411, -2746906807],
            ],
        ),
    }
}

#[test]
#[inline(never)]
fn probe_symmetric_eigen3__baseline() {
    let _i = black_box(a());
    let e = black_box(expected());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_symmetric_eigen3__op() {
    let a = black_box(a());
    let e = black_box(expected());
    assert!(a.symmetric_eigen() == e);
}
