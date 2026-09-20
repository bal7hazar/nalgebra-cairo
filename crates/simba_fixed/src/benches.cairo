//! Gas benchmarks: what a nalgebra kernel costs on glam.cairo's scalar versus on simba's.
//!
//! Test-only module. Each group runs the SAME kernel twice, once instantiated at `fixed::Fixed`
//! (`via_fixed`) and once at `simba::fixed::Fixed` (`via_simba`), on operands hidden behind
//! `black_box`, with a `baseline` variant that loads the inputs and asserts without computing.
//!
//! Target: zero overhead. `crate::convert` is a `struct_deconstruct` / `struct_construct` pair,
//! which the compiler folds away, so `via_fixed` and `via_simba` must come out equal for every
//! kernel this package delegates (`sum_prod3`, `norm3`, the `Wide` accumulator, ...). Where they
//! differ, the two variants are genuinely different code and the gas table says which one the
//! caller pays for: `mul` (glam's `Mul` impl, which nalgebra reaches through the corelib operator,
//! not through `Real`) and `sin_cos` (glam's generated polynomial, DESIGN D6 versus glam.cairo's
//! own tables).

use fixed::trig::TrigTrait;
use fixed::{Fixed as Glam, wide as gwide};
use nalgebra_testing::black_box;
use simba::fixed::{Fixed as Simba, fused, transcendental as strans};
use simba::scalar::{Real, Transcendental};
use crate::convert::{from_simba, to_simba};
use crate::real::FixedReal;
use crate::transcendental::FixedTranscendental;

// 1.5, -2.25, 0.75 and -0.375, 3.125, 1.0625 as raw Q32.32, and 0.7 radian.
const AX: i64 = 0x1_8000_0000;
const AY: i64 = -0x2_4000_0000;
const AZ: i64 = 0x0_C000_0000;
const BX: i64 = -0x0_6000_0000;
const BY: i64 = 0x3_2000_0000;
const BZ: i64 = 0x1_1000_0000;
const ANGLE: i64 = 3006477107;

// --- conversion ---------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_simba_fixed_convert__baseline() {
    let a = black_box(Glam { raw: AX });
    assert!(a.raw == AX);
}

/// The whole cost of this package's indirection, measured: `Glam -> Simba -> Glam`.
#[test]
#[inline(never)]
fn bench_simba_fixed_convert__roundtrip() {
    let a = black_box(Glam { raw: AX });
    assert!(from_simba(to_simba(a)).raw == AX);
}

// --- mul --------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_simba_fixed_mul__baseline() {
    let _a = black_box(Glam { raw: AX });
    let _b = black_box(Glam { raw: BY });
    let e = black_box(0x4_B000_0000_i64);
    assert!(e == e);
}

/// glam.cairo's own `Mul` impl: what `a * b` costs in generic nalgebra code on `fixed::Fixed`
/// (the corelib operator is NOT routed through `Real`).
#[test]
#[inline(never)]
fn bench_simba_fixed_mul__via_fixed() {
    let a = black_box(Glam { raw: AX });
    let b = black_box(Glam { raw: BY });
    let e = black_box(0x4_B000_0000_i64);
    assert!((a * b).raw == e);
}

#[test]
#[inline(never)]
fn bench_simba_fixed_mul__via_simba() {
    let a = black_box(Simba { raw: AX });
    let b = black_box(Simba { raw: BY });
    let e = black_box(0x4_B000_0000_i64);
    assert!((a * b).raw == e);
}

// --- sum_prod3 --------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_simba_fixed_sum_prod3__baseline() {
    let _a = black_box((Glam { raw: AX }, Glam { raw: AY }, Glam { raw: AZ }));
    let _b = black_box((Glam { raw: BX }, Glam { raw: BY }, Glam { raw: BZ }));
    let e = black_box(-0x6_CC00_0000_i64);
    assert!(e == e);
}

/// `Real::sum_prod3` at `fixed::Fixed`: three conversions in, one out, simba's kernel in between.
#[test]
#[inline(never)]
fn bench_simba_fixed_sum_prod3__via_fixed() {
    let (ax, ay, az) = black_box((Glam { raw: AX }, Glam { raw: AY }, Glam { raw: AZ }));
    let (bx, by, bz) = black_box((Glam { raw: BX }, Glam { raw: BY }, Glam { raw: BZ }));
    let e = black_box(-0x6_CC00_0000_i64);
    assert!(Real::sum_prod3(ax, bx, ay, by, az, bz).raw == e);
}

#[test]
#[inline(never)]
fn bench_simba_fixed_sum_prod3__via_simba() {
    let (ax, ay, az) = black_box((Simba { raw: AX }, Simba { raw: AY }, Simba { raw: AZ }));
    let (bx, by, bz) = black_box((Simba { raw: BX }, Simba { raw: BY }, Simba { raw: BZ }));
    let e = black_box(-0x6_CC00_0000_i64);
    assert!(fused::sum_prod3(ax, bx, ay, by, az, bz).raw == e);
}

/// glam.cairo's own fused dot product, for reference: the same figure means a rapier user loses
/// nothing by going through nalgebra's trait.
#[test]
#[inline(never)]
fn bench_simba_fixed_sum_prod3__via_glam_native() {
    let (ax, ay, az) = black_box((Glam { raw: AX }, Glam { raw: AY }, Glam { raw: AZ }));
    let (bx, by, bz) = black_box((Glam { raw: BX }, Glam { raw: BY }, Glam { raw: BZ }));
    let e = black_box(-0x6_CC00_0000_i64);
    assert!(gwide::dot3(ax, bx, ay, by, az, bz).raw == e);
}

// --- norm3 ------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_simba_fixed_norm3__baseline() {
    let _a = black_box((Glam { raw: AX }, Glam { raw: AY }, Glam { raw: AZ }));
    let e = black_box(0x2_CE65_F1A1_i64);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_simba_fixed_norm3__via_fixed() {
    let (x, y, z) = black_box((Glam { raw: AX }, Glam { raw: AY }, Glam { raw: AZ }));
    let e = black_box(0x2_CE65_F1A1_i64);
    assert!(Real::norm3(x, y, z).raw == e);
}

#[test]
#[inline(never)]
fn bench_simba_fixed_norm3__via_simba() {
    let (x, y, z) = black_box((Simba { raw: AX }, Simba { raw: AY }, Simba { raw: AZ }));
    let e = black_box(0x2_CE65_F1A1_i64);
    assert!(fused::norm3(x, y, z).raw == e);
}

#[test]
#[inline(never)]
fn bench_simba_fixed_norm3__via_glam_native() {
    let (x, y, z) = black_box((Glam { raw: AX }, Glam { raw: AY }, Glam { raw: AZ }));
    let e = black_box(0x2_CE65_F1A1_i64);
    assert!(gwide::norm3(x, y, z).raw == e);
}

// --- sin_cos ----------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_simba_fixed_sin_cos__baseline() {
    let a = black_box(Glam { raw: ANGLE });
    assert!(a.raw == ANGLE);
}

/// `Transcendental::sin_cos` at `fixed::Fixed`, i.e. glam.cairo's `TrigTrait` (DELIBERATE, see
/// `crate::transcendental`): a different generated polynomial from simba's, so a different price.
#[test]
#[inline(never)]
fn bench_simba_fixed_sin_cos__via_fixed() {
    let a = black_box(Glam { raw: ANGLE });
    let (s, c) = Transcendental::sin_cos(a);
    assert!(s.raw != 0 && c.raw != 0);
}

#[test]
#[inline(never)]
fn bench_simba_fixed_sin_cos__via_simba() {
    let a = black_box(Simba { raw: ANGLE });
    let (s, c) = strans::sin_cos(a);
    assert!(s.raw != 0 && c.raw != 0);
}

/// glam.cairo's `TrigTrait` called directly, to show the forward is free.
#[test]
#[inline(never)]
fn bench_simba_fixed_sin_cos__via_glam_native() {
    let a = black_box(Glam { raw: ANGLE });
    let (s, c) = TrigTrait::sin_cos(a);
    assert!(s.raw != 0 && c.raw != 0);
}
