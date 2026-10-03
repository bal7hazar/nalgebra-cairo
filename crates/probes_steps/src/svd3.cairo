//! Probe of `Svd3`: the singular value decomposition with both `u` and `v_t`. Input: a case of the
//! oracle `svd3_singular_values` (upstream nalgebra on f64, `nalgebra_tests_linalg::oracle_svd`);
//! the asserted value is the exact result of the kernel at this head (its singular values agree
//! with the oracle within its tolerance, checked when the probe was written; the vectors follow the
//! sign convention of `Svd3`).

use fixed::Fixed;
use nalgebra_linalg_svd_eigen3::linalg::svd3::{Matrix3SvdTrait, Svd3};
use nalgebra_testing::black_box;
use nalgebra_types3::base::matrix3::Matrix3;
use crate::builders::{m3, v3};
use crate::eq::Svd3Eq;

fn a() -> Matrix3<Fixed> {
    m3(
        [
            [-163144510, -950958694, -1188067349], [1679219748, 749014508, -1278117059],
            [887659363, -1272926972, 1082801323],
        ],
    )
}

fn expected() -> Svd3<Fixed> {
    Svd3 {
        u: Option::Some(
            m3(
                [
                    [767737278, 101867359, -4224564664], [3836997256, 1782185344, 740278016],
                    [-1770529754, 3906428359, -227565259],
                ],
            ),
        ),
        singular_values: v3((2347686549, 1785679889, 1495755202)),
        v_t: Option::Some(
            m3(
                [
                    [2021684760, 1873178203, -3294045876], [3608507586, -2091408996, 1025390410],
                    [1156809327, 3250223976, 2558237746],
                ],
            ),
        ),
    }
}

#[test]
#[inline(never)]
fn probe_svd3__baseline() {
    let _i = black_box(a());
    let e = black_box(expected());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_svd3__op() {
    let a = black_box(a());
    let e = black_box(expected());
    assert!(a.svd(true, true) == e);
}
