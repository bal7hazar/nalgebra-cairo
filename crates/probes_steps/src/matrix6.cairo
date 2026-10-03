//! Probes of `Matrix6`: `mul_vec`. The matrix is the first input of the oracle case 3 of
//! `lu6_solve` (fractional entries); the expected vector is computed on exact integers (one floor
//! of the sum of six products per component, the kernel's definition).

use fixed::Fixed;
use nalgebra_core::base::matrix_mul::MatrixMul;
use nalgebra_testing::black_box;
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::base::vector6::Vector6;
use crate::builders::{m6, v6};

fn a() -> Matrix6<Fixed> {
    m6(
        [
            [-2527254097, 4325213708, -1213189640, 4008510819, 2405075077, 466428599],
            [-700750028, -3536112306, 1731465811, 244560054, 2632174365, 1874080084],
            [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
            [-617580144, 580405440, -3475076872, -924247759, -985644408, 938584824],
            [-1076219818, -1905997652, -1595683778, 2067806730, -3245944919, 3717734456],
            [-308280913, -120179493, 2005545225, -1090342151, 932840990, 2332476429],
        ],
    )
}

fn x() -> Vector6<Fixed> {
    v6((3579353502, 7767965432, 6840630971, -6086016248, -4735434050, -2809324573))
}

fn ax() -> Vector6<Fixed> {
    v6((-4852678061, -8696246748, -733717169, -3217262024, -8668622223, 1710827457))
}

#[test]
#[inline(never)]
fn probe_matrix6_mul_vec__baseline() {
    let _i = black_box((a(), x()));
    let e = black_box(ax());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_matrix6_mul_vec__op() {
    let (p, v) = black_box((a(), x()));
    let e = black_box(ax());
    assert!(p.mul_mat(v) == e);
}
