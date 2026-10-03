//! Probes of the two scalar kernels the decomposition packages call far more often than any
//! shape operation (counted over `linalg_core`, `linalg2..6`, `linalg_pivot2..6`,
//! `linalg_spectral2..6` and `linalg_svd_eigen2..6`: ~3,900 `Real::mul_add` and ~4,400
//! `wide_add_prod` / `wide_sub_prod` calls, ~900 `wide_rescale`): `mul_add` and a fused
//! sum of three products (`wide_add_prod` x 3 then `wide_rescale`). The expected values come from
//! exact integers: the product floors once (`mul_add`: `c + floor(a * b / 2^32)`; the sum:
//! `floor((a1 b1 + a2 b2 + a3 b3) / 2^32)`).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::builders::fx;

fn m() -> (Fixed, Fixed, Fixed) {
    (fx(-8333418062), fx(6422282562), fx(5406550886))
}

fn p() -> (Fixed, Fixed, Fixed, Fixed, Fixed, Fixed) {
    (
        fx(-3562322882),
        fx(6202159288),
        fx(-7141719772),
        fx(2324644860),
        fx(2873393302),
        fx(-6638673079),
    )
}

#[test]
#[inline(never)]
fn probe_scalar_mul_add__baseline() {
    let _i = black_box(m());
    let e = black_box(fx(-7054443998));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_scalar_mul_add__op() {
    let (x, y, z) = black_box(m());
    let e = black_box(fx(-7054443998));
    assert!(Real::mul_add(x, y, z) == e);
}

#[test]
#[inline(never)]
fn probe_scalar_wide_dot3__baseline() {
    let _i = black_box(p());
    let e = black_box(fx(-13450992962));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn probe_scalar_wide_dot3__op() {
    let (a1, b1, a2, b2, a3, b3) = black_box(p());
    let e = black_box(fx(-13450992962));
    let w = Real::wide_add_prod(Real::<Fixed>::wide_zero(), a1, b1);
    let w = Real::wide_add_prod(w, a2, b2);
    let w = Real::wide_add_prod(w, a3, b3);
    assert!(Real::wide_rescale(w) == e);
}
