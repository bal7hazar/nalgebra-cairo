//! Gas benchmarks of `UnitComplex` (`bench_unit_complex_<op>__<variant>`, net = raw - `baseline`
//! of the group), and the alternative implementations that lost (`alt_*`), kept as evidence
//! together with the tests showing why (AGENTS.md rule 8).
//!
//! The rotations used are `c = new(0.4 rad)` and `d = new(-1/6 rad)`, the vector `(1.5, -2.25)`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/unit_complex/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{uc, v2};
use crate::base::vector2::Vector2;
use crate::geometry::unit_complex::UnitComplexInternalTrait;
use super::{UnitComplex, UnitComplexTrait};

/// `new(0.4)`.
fn c() -> UnitComplex<Fixed> {
    uc(3955926847, 1672539044)
}

/// `rotation_between` with the inputs normalized first: the dot and perp products are then of
/// unit scale, so the result keeps about 4 ulp whatever the length of the inputs (instead of
/// `2 + 2/(|a|·|b|)`), at the price of two norms and four divisions.
#[inline(always)]
fn alt_rotation_between_normalized(a: Vector2<Fixed>, b: Vector2<Fixed>) -> UnitComplex<Fixed> {
    let na = Real::norm2(a.x, a.y);
    let nb = Real::norm2(b.x, b.y);
    if na == Real::zero() || nb == Real::zero() {
        return UnitComplexTrait::identity();
    }
    let (ax, ay) = (a.x / na, a.y / na);
    let (bx, by) = (b.x / nb, b.y / nb);
    let c = UnitComplex {
        re: Real::sum_prod2(ax, bx, ay, by), im: Real::diff_prod(ax, by, ay, bx),
    };
    c.renormalized_fast()
}

/// `renormalize_fast` written as upstream's literal `1/2 * (3 - |c|²)`: two roundings instead of
/// one for the same bits.
#[inline(always)]
fn alt_renormalize_fast_literal(c: UnitComplex<Fixed>) -> UnitComplex<Fixed> {
    let three = Real::<Fixed>::from_int(3);
    let f = Real::<Fixed>::HALF * (three - Real::norm_squared2(c.re, c.im));
    UnitComplex { re: c.re * f, im: c.im * f }
}

/// `renormalize_fast` as `c + c·(1 - |c|²)/2`, one `mul_add` per component.
#[inline(always)]
fn alt_renormalize_fast_mul_add(c: UnitComplex<Fixed>) -> UnitComplex<Fixed> {
    let g = Real::mul_add(Real::norm_squared2(c.re, c.im), -Real::<Fixed>::HALF, Real::HALF);
    UnitComplex { re: Real::mul_add(c.re, g, c.re), im: Real::mul_add(c.im, g, c.im) }
}

/// `renormalize_fast` as a `lerp` towards the corrected pair.
#[inline(always)]
fn alt_renormalize_fast_lerp(c: UnitComplex<Fixed>) -> UnitComplex<Fixed> {
    let f = Real::mul_add(
        Real::norm_squared2(c.re, c.im), -Real::<Fixed>::HALF, Real::HALF + Real::one(),
    );
    UnitComplex {
        re: Real::lerp(Real::zero(), c.re, f), im: Real::lerp(Real::<Fixed>::zero(), c.im, f),
    }
}

#[test]
fn test_rotation_between_alt_normalized_wins_on_short_vectors() {
    // |a| = |b| ~ 0.0077: the algebraic form loses the bits of `a·b` and `a×b`, the normalized
    // one does not. The reference is the same directions 256 times longer.
    let (a, b) = (v2(0x30ec4a1, -0x41d5b92), v2(0x40a3d70, 0x2f1a9fc));
    let reference = UnitComplexTrait::rotation_between(
        v2(0x30ec4a100, -0x41d5b9200), v2(0x40a3d7000, 0x2f1a9fc00),
    );
    assert!(!UnitComplexTrait::rotation_between(a, b).abs_diff_eq(reference, 200));
    assert!(alt_rotation_between_normalized(a, b).abs_diff_eq(reference, 8));
}

#[test]
fn test_renormalize_fast_alternatives_give_the_same_bits() {
    let drifted = uc(3955930943, 1672543140);
    let got = drifted.renormalized_fast();
    assert!(alt_renormalize_fast_literal(drifted) == got);
    assert!(alt_renormalize_fast_lerp(drifted) == got);
    // `c + c·(1 - |c|²)/2` rounds the correction and the sum separately: 1 ulp apart.
    assert!(alt_renormalize_fast_mul_add(drifted).abs_diff_eq(got, 1));
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between__alt_normalized_inputs() {
    let a: Vector2<Fixed> = black_box(v2(0x180000000, -0x240000000));
    let b: Vector2<Fixed> = black_box(v2(-0x480000000, 0x40000000));
    let e: UnitComplex<Fixed> = black_box(uc(-2576980378, -3435973837));
    assert!(alt_rotation_between_normalized(a, b) == e);
}
