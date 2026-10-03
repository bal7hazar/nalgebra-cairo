//! Probes of `Cholesky3` and `Cholesky6`: the factorisation of a symmetric positive-definite matrix
//! and the solve with a factor. Inputs: a case of the oracle `cholesky{3,6}_solve` (upstream
//! nalgebra on f64, `nalgebra_tests_linalg::oracle_cholesky`); the asserted values are the exact
//! results of the kernels at this head (they agree with the oracle within its tolerance, checked
//! when the probes were written).

use fixed::Fixed;
use nalgebra_linalg3::linalg::cholesky::{Cholesky3, Cholesky3Trait, Matrix3CholeskyTrait};
use nalgebra_linalg6::linalg::cholesky::{Cholesky6, Cholesky6Trait, Matrix6CholeskyTrait};
use nalgebra_testing::black_box;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::base::vector6::Vector6;
use crate::builders::{fx, m3, m6, v3, v6};
use crate::eq::{Cholesky3Eq, Cholesky6Eq};

fn a3() -> Matrix3<Fixed> {
    m3(
        [
            [1900012299, -85944448, -86359314], [-85944448, 1871146221, -627940431],
            [-86359314, -627940431, 1916592464],
        ],
    )
}

fn b3() -> Vector3<Fixed> {
    v3((2792725566, 5137014102, 3010252812))
}

fn f3() -> Cholesky3<Fixed> {
    Cholesky3 {
        l11: fx(2856657257),
        l21: fx(-129216969),
        l31: fx(-129840718),
        l22: fx(2831927752),
        l32: fx(-958273473),
        l33: fx(2701213669),
    }
}

fn x3() -> Vector3<Fixed> {
    v3((7616023173, 16313868946, 12433936943))
}

fn a6() -> Matrix6<Fixed> {
    m6(
        [
            [3654480394, 90933453, -304406952, -302438381, -235454993, -273500147],
            [90933453, 4075594332, -1110029388, 810457968, 432815, 165140043],
            [-304406952, -1110029388, 4932250777, 743241296, -713496402, -664760589],
            [-302438381, 810457968, 743241296, 3475255779, 101179924, 635633272],
            [-235454993, 432815, -713496402, 101179924, 4388679014, -240687438],
            [-273500147, 165140043, -664760589, 635633272, -240687438, 4212214819],
        ],
    )
}

fn b6() -> Vector6<Fixed> {
    v6((2829439348, -7782455719, 7478168046, -4127194961, 6320029132, 7729783779))
}

fn f6() -> Cholesky6<Fixed> {
    Cholesky6 {
        l11: fx(3961801834),
        l21: fx(98580450),
        l31: fx(-330005881),
        l41: fx(-327871764),
        l51: fx(-255255446),
        l61: fx(-296499986),
        l22: fx(4182681706),
        l32: fx(-1132050709),
        l42: fx(839942519),
        l52: fx(6460478),
        l62: fx(176561411),
        l33: fx(4448978881),
        l43: fx(906917088),
        l53: fx(-706086949),
        l63: fx(-618815088),
        l44: fx(3645628612),
        l54: fx(270408592),
        l64: fx(835445133),
        l55: fx(4267593721),
        l65: fx(-415554408),
        l66: fx(4088844893),
    }
}

fn x6() -> Vector6<Fixed> {
    v6((5035766652, -4473761745, 9902458876, -8132566764, 8893777070, 11682206543))
}

#[test]
#[inline(never)]
fn probe_cholesky3_factor__baseline() {
    let _i = black_box(a3());
    let e = black_box(f3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_cholesky3_factor__op() {
    let a = black_box(a3());
    let e = black_box(f3());
    assert!(a.cholesky().unwrap() == e);
}

#[test]
#[inline(never)]
fn probe_cholesky3_solve__baseline() {
    let _i = black_box((f3(), b3()));
    let e = black_box(x3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_cholesky3_solve__op() {
    let (f, b) = black_box((f3(), b3()));
    let e = black_box(x3());
    assert!(f.solve(b) == e);
}

#[test]
#[inline(never)]
fn probe_cholesky6_factor__baseline() {
    let _i = black_box(a6());
    let e = black_box(f6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_cholesky6_factor__op() {
    let a = black_box(a6());
    let e = black_box(f6());
    assert!(a.cholesky().unwrap() == e);
}

#[test]
#[inline(never)]
fn probe_cholesky6_solve__baseline() {
    let _i = black_box((f6(), b6()));
    let e = black_box(x6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_cholesky6_solve__op() {
    let (f, b) = black_box((f6(), b6()));
    let e = black_box(x6());
    assert!(f.solve(b) == e);
}
