//! Probes of `Matrix3`: `mul`, `mul_vec`, `transpose`, `determinant`, `try_inverse`. Inputs and
//! expected values are those of the gas benches of `nalgebra_tests_base::matrix3`.

use fixed::Fixed;
use nalgebra_core::base::matrix_mul::MatrixMul;
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_testing::black_box;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use crate::builders::{fx, m3, v3};

fn a() -> Matrix3<Fixed> {
    m3(
        [
            [-8333418062, -3562322882, -7141719772], [-6195210852, 8037214559, 5406550886],
            [2873393302, -6638673079, 4752733287],
        ],
    )
}

fn b() -> Matrix3<Fixed> {
    m3(
        [
            [-6403338741, -8222594018, -3661741761], [4499486460, 3456238901, 7403023557],
            [-5213305701, 2927835069, 3069760649],
        ],
    )
}

fn x() -> Vector3<Fixed> {
    v3((6422282562, 6202159288, 2324644860))
}

/// A matrix of determinant well away from 0 (the inverse takes its prescaled path).
fn c() -> Matrix3<Fixed> {
    m3(
        [
            [1651849619, 3942926787, -4111385247], [-2706174340, -1356311774, -3161153896],
            [1713407532, -2388220103, 1097729905],
        ],
    )
}

fn ab() -> Matrix3<Fixed> {
    m3(
        [
            [17361027083, 8218990867, -4139846565], [11093767635, 22013840869, 22999422663],
            [-17007668927, -7603403072, -10495568584],
        ],
    )
}

fn at() -> Matrix3<Fixed> {
    m3(
        [
            [-8333418062, -6195210852, 2873393302], [-3562322882, 8037214559, -6638673079],
            [-7141719772, 5406550886, 4752733287],
        ],
    )
}

fn c_inv() -> Matrix3<Fixed> {
    m3(
        [
            [2746796817, -1668618017, 5482570468], [743254844, -2691902150, -4968171083],
            [-2670352862, -3252013208, -2561850149],
        ],
    )
}

#[test]
#[inline(never)]
fn probe_matrix3_mul__baseline() {
    let _i = black_box((a(), b()));
    let e = black_box(ab());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_mul__op() {
    let (p, q) = black_box((a(), b()));
    let e = black_box(ab());
    assert!(p * q == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_mul_vec__baseline() {
    let _i = black_box((a(), x()));
    let e = black_box(v3((-21470622535, 5268724875, -2717586978)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_mul_vec__op() {
    let (p, v) = black_box((a(), x()));
    let e = black_box(v3((-21470622535, 5268724875, -2717586978)));
    assert!(p.mul_mat(v) == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_transpose__baseline() {
    let _i = black_box(a());
    let e = black_box(at());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_transpose__op() {
    let p = black_box(a());
    let e = black_box(at());
    assert!(p.transpose() == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_determinant__baseline() {
    let _i = black_box(a());
    let e = black_box(fx(-49139065034));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_determinant__op() {
    let p = black_box(a());
    let e = black_box(fx(-49139065034));
    assert!(p.determinant() == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_try_inverse__baseline() {
    let _i = black_box(c());
    let e = black_box(c_inv());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_matrix3_try_inverse__op() {
    let p = black_box(c());
    let e = black_box(c_inv());
    assert!(p.try_inverse().unwrap() == e);
}
