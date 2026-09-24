//! Gas of `Udu{2,3,4,6}::new`: the `LDLᵀ` kernel of the reversed matrix plus the reversals
//! (moves).
//! Inputs are oracle `ldlt{n}_l_d` cases (SPD).

use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{m2, m3, m4, m6};
use crate::linalg::oracle_udu;
use super::{Udu2Trait, Udu3Trait, Udu4Trait, Udu6Trait};

#[test]
#[inline(never)]
fn bench_udu2_new__baseline() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt2_l_d_cases()[0];
    let _p = black_box(m2(a));
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_udu2_new__reversed_ldlt() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt2_l_d_cases()[0];
    let p = black_box(m2(a));
    let e = black_box(true);
    assert!(Udu2Trait::new(p).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_udu3_new__baseline() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt3_l_d_cases()[0];
    let _p = black_box(m3(a));
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_udu3_new__reversed_ldlt() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt3_l_d_cases()[0];
    let p = black_box(m3(a));
    let e = black_box(true);
    assert!(Udu3Trait::new(p).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_udu4_new__baseline() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt4_l_d_cases()[0];
    let _p = black_box(m4(a));
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_udu4_new__reversed_ldlt() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt4_l_d_cases()[0];
    let p = black_box(m4(a));
    let e = black_box(true);
    assert!(Udu4Trait::new(p).is_some() == e);
}

#[test]
#[inline(never)]
fn bench_udu6_new__baseline() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt6_l_d_cases()[0];
    let _p = black_box(m6(a));
    let e = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_udu6_new__reversed_ldlt() {
    let (a, _l, _d, _tol) = *oracle_udu::ldlt6_l_d_cases()[0];
    let p = black_box(m6(a));
    let e = black_box(true);
    assert!(Udu6Trait::new(p).is_some() == e);
}
