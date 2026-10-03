//! Probes of `Lu3` and `Lu6`: the LU factorisation with partial pivoting and the solve with a
//! factor. Inputs: a case of the oracle `lu{3,6}_solve` (upstream nalgebra on f64,
//! `nalgebra_tests_linalg::lu`); the asserted values are the exact results of the kernels at this
//! head (they agree with the oracle within its tolerance, checked when the probes were written).

use fixed::Fixed;
use nalgebra_linalg3::linalg::lu::lu3::{Lu3, Lu3Trait, Matrix3LuTrait};
use nalgebra_static6_wide::linalg::lu::Perm6;
use nalgebra_static6_wide::linalg::lu::lu6::{Lu6, Lu6Trait, Matrix6LuTrait};
use nalgebra_testing::black_box;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::linalg::lu::Perm3;
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::base::vector6::Vector6;
use crate::builders::{m3, m6, v3, v6};
use crate::eq::{Lu3Eq, Lu6Eq};

fn a3() -> Matrix3<Fixed> {
    m3(
        [
            [-2414118097, 417657389, 4595817171], [6298117444, -2725477524, -1338161353],
            [2614037897, 2690897069, 3051067169],
        ],
    )
}

fn b3() -> Vector3<Fixed> {
    v3((7757329492, -3622378744, 4147284176))
}

fn f3() -> Lu3<Fixed> {
    Lu3 {
        lu: m3(
            [
                [6298117444, -2725477524, -1338161353], [1782629076, 3822108355, 3606471942],
                [-1646294844, -704614971, 4674552574],
            ],
        ),
        p: Perm3 { p1: 2, p2: 3 },
    }
}

fn x3() -> Vector3<Fixed> {
    v3((-1035334030, 24604678, 6703439321))
}

fn a6() -> Matrix6<Fixed> {
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

fn b6() -> Vector6<Fixed> {
    v6((3579353502, 7767965432, 6840630971, -6086016248, -4735434050, -2809324573))
}

fn f6() -> Lu6<Fixed> {
    Lu6 {
        lu: m6(
            [
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [2625460419, 4958875604, -941745570, 5107545028, 4074451472, 218945968],
                [641578631, 636814518, -3269112062, -1412974283, -1181820814, 845644872],
                [1118040540, -1417102435, 2352776612, 4995058989, -543304442, 3221342200],
                [727980405, -2910510775, -1535245700, 3014081670, 5814961884, -4536658],
                [320260371, -37142392, -2667692284, -1538893104, 179474863, 3983829884],
            ],
        ),
        p: Perm6 { p1: 3, p2: 3, p3: 4, p4: 5, p5: 5 },
    }
}

fn x6() -> Vector6<Fixed> {
    v6((-11208158820, -7248812512, 3383102671, 4212967583, 4206763124, -9649832811))
}

#[test]
#[inline(never)]
fn probe_lu3_factor__baseline() {
    let _i = black_box(a3());
    let e = black_box(f3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_lu3_factor__op() {
    let a = black_box(a3());
    let e = black_box(f3());
    assert!(a.lu() == e);
}

#[test]
#[inline(never)]
fn probe_lu3_solve__baseline() {
    let _i = black_box((f3(), b3()));
    let e = black_box(x3());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_lu3_solve__op() {
    let (f, b) = black_box((f3(), b3()));
    let e = black_box(x3());
    assert!(f.solve(b).unwrap() == e);
}

#[test]
#[inline(never)]
fn probe_lu6_factor__baseline() {
    let _i = black_box(a6());
    let e = black_box(f6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_lu6_factor__op() {
    let a = black_box(a6());
    let e = black_box(f6());
    assert!(a.lu() == e);
}

#[test]
#[inline(never)]
fn probe_lu6_solve__baseline() {
    let _i = black_box((f6(), b6()));
    let e = black_box(x6());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_lu6_solve__op() {
    let (f, b) = black_box((f6(), b6()));
    let e = black_box(x6());
    assert!(f.solve(b).unwrap() == e);
}
