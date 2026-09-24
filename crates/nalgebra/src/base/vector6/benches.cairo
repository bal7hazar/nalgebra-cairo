//! Gas benchmarks of `Vector6` (`bench_vector6_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, v6, v6_head, v6_tail};
use crate::base::vector3::Vector3Trait;
use super::{Vector6, Vector6Trait};

// --- alternative implementations (losers)

/// `dot` as two `sum_prod3`, one per block: one rescale per half instead of one for the whole
/// row. Two floor roundings, so up to 1 ulp away from the exact floor — and dearer, because a
/// `sum_prod3` pays a full rescale while an extra `wide_add_prod` does not.
#[inline(always)]
fn alt_dot_two_sum_prod3(a: Vector6<Fixed>, b: Vector6<Fixed>) -> Fixed {
    Vector3Trait::dot(v6_head(a), v6_head(b)) + Vector3Trait::dot(v6_tail(a), v6_tail(b))
}

/// `norm_squared` as the sum of the two blocks' squared norms: same two-rounding defect.
#[inline(always)]
fn alt_norm_squared_two_blocks(a: Vector6<Fixed>) -> Fixed {
    Vector3Trait::norm_squared(v6_head(a)) + Vector3Trait::norm_squared(v6_tail(a))
}

/// `norm` through `norm_squared`: the squared norm must fit Q32.32, so it overflows above a norm
/// of about 46 340, where the shipped `wide_sqrt` of the unscaled accumulator still works.
#[inline(always)]
fn alt_norm_via_norm_squared(a: Vector6<Fixed>) -> Fixed {
    Real::sqrt(Vector6Trait::norm_squared(a))
}

/// `dot` with one rounding and one overflow check per product (what AGENTS.md rule 4 forbids).
#[inline(always)]
fn alt_dot_unfused(a: Vector6<Fixed>, b: Vector6<Fixed>) -> Fixed {
    a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w + a.a * b.a + a.b * b.b
}

// --- why the alternatives lost

#[test]
fn test_dot_alt_two_sum_prod3_rounds_twice() {
    // Two products of 2^-33: the exact sum is 1 ulp, each half floors to 0.
    let u = v6(0x80000000, 0, 0, 0x80000000, 0, 0);
    let w = v6(1, 0, 0, 1, 0, 0);
    assert!(u.dot(w) == fx(1));
    assert!(alt_dot_two_sum_prod3(u, w) == fx(0));
}

#[test]
fn test_norm_squared_alt_two_blocks_rounds_twice() {
    // Six components of 2^-17: each block sums three squares to 3 * 2^30 raw, which floors to 0;
    // the exact sum of the six, 6 * 2^30, is 1 ulp.
    let u = v6(0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000);
    assert!(u.norm_squared() == fx(1));
    assert!(alt_norm_squared_two_blocks(u) == fx(0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_norm_alt_via_norm_squared_overflows_on_huge() {
    // Six components of 1e6, norm 2 449 489.7: the squared norm (6e12) does not fit Q32.32, the
    // norm does.
    const M: i64 = 1000000 * 0x100000000;
    let v = v6(M, M, M, M, M, M);
    assert!(v.norm() == fx(10520478337141201));
    let _ = alt_norm_via_norm_squared(black_box(v));
}

#[test]
fn test_dot_alt_unfused_is_less_accurate() {
    // Six products of 2^-33 - epsilon: every one floors to 0 on its own, the exact sum is 2 ulp.
    let u = v6(0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000);
    let w = v6(1, 1, 1, 1, 1, 1);
    assert!(u.dot(w) == fx(3));
    assert!(alt_dot_unfused(u, w) == fx(0));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_vector6_new__baseline() {
    let _x: Fixed = black_box(fx(0x180000000));
    let e: Vector6<Fixed> = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_new__new() {
    let x: Fixed = black_box(fx(0x180000000));
    let e: Vector6<Fixed> = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let r = Vector6Trait::new(
        x, fx(-0x240000000), fx(0x3c0000000), fx(-0x480000000), fx(0x40000000), fx(0x200000000),
    );
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_vector6_fill__baseline() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector6<Fixed> = black_box(v6(0, 0, 0, 0, 0, 0));
    assert!(black_box(e) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_fill__zeros() {
    let _s: Fixed = black_box(fx(0x280000000));
    let e: Vector6<Fixed> = black_box(v6(0, 0, 0, 0, 0, 0));
    assert!(black_box(Vector6Trait::<Fixed>::zeros()) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_add__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x300000000, -0x200000000, 0x5c0000000, -0x300000000, -0x200000000, 0x5c0000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_add__operator() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x300000000, -0x200000000, 0x5c0000000, -0x300000000, -0x200000000, 0x5c0000000),
    );
    assert!(a + b == e);
}

#[test]
#[inline(never)]
fn bench_vector6_add__assign() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x300000000, -0x200000000, 0x5c0000000, -0x300000000, -0x200000000, 0x5c0000000),
    );
    let mut acc = a;
    acc += b;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_vector6_sub__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(0x600000000, -0x280000000, 0x1c0000000, -0x600000000, 0x280000000, -0x1c0000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_sub__operator() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(0x600000000, -0x280000000, 0x1c0000000, -0x600000000, 0x280000000, -0x1c0000000),
    );
    assert!(a - b == e);
}

#[test]
#[inline(never)]
fn bench_vector6_sub__assign() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(0x600000000, -0x280000000, 0x1c0000000, -0x600000000, 0x280000000, -0x1c0000000),
    );
    let mut acc = a;
    acc -= b;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_vector6_neg__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(
        v6(-0x180000000, 0x240000000, -0x3c0000000, 0x480000000, -0x40000000, -0x200000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_neg__operator() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(
        v6(-0x180000000, 0x240000000, -0x3c0000000, 0x480000000, -0x40000000, -0x200000000),
    );
    assert!(-a == e);
}

#[test]
#[inline(never)]
fn bench_vector6_scale__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _k = black_box(fx(0x200000000));
    let e = black_box(
        v6(0x300000000, -0x480000000, 0x780000000, -0x900000000, 0x80000000, 0x400000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_scale__scale() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let k = black_box(fx(0x200000000));
    let e = black_box(
        v6(0x300000000, -0x480000000, 0x780000000, -0x900000000, 0x80000000, 0x400000000),
    );
    assert!(a.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_scale__mul_assign() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let k = black_box(fx(0x200000000));
    let e = black_box(
        v6(0x300000000, -0x480000000, 0x780000000, -0x900000000, 0x80000000, 0x400000000),
    );
    let mut acc = a;
    acc *= k;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_vector6_unscale__baseline() {
    let _a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let _k = black_box(fx(0x600000000));
    let e = black_box(v6(1431655765, -1431655766, 1431655765, -1431655766, 1431655765, 2863311530));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_unscale__unscale() {
    let a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let k = black_box(fx(0x600000000));
    let e = black_box(v6(1431655765, -1431655765, 1431655765, -1431655765, 1431655765, 2863311531));
    assert!(a.unscale(k) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_unscale__div_assign() {
    let a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let k = black_box(fx(0x600000000));
    let e = black_box(v6(1431655765, -1431655765, 1431655765, -1431655765, 1431655765, 2863311531));
    let mut acc = a;
    acc /= k;
    assert!(acc == e);
}

#[test]
#[inline(never)]
fn bench_vector6_component_mul__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x6c0000000, -0x90000000, 0x780000000, -0x6c0000000, -0x90000000, 0x780000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_component_mul__component_mul() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x6c0000000, -0x90000000, 0x780000000, -0x6c0000000, -0x90000000, 0x780000000),
    );
    assert!(a.component_mul(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_abs__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(
        v6(0x180000000, 0x240000000, 0x3c0000000, 0x480000000, 0x40000000, 0x200000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_abs__componentwise() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(
        v6(0x180000000, 0x240000000, 0x3c0000000, 0x480000000, 0x40000000, 0x200000000),
    );
    assert!(a.abs() == e);
}

#[test]
#[inline(never)]
fn bench_vector6_inf_sup__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x480000000, -0x240000000, 0x200000000, -0x480000000, -0x240000000, 0x200000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_inf_sup__inf() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(-0x480000000, -0x240000000, 0x200000000, -0x480000000, -0x240000000, 0x200000000),
    );
    assert!(a.inf(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_inf_sup__sup() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(
        v6(0x180000000, 0x40000000, 0x3c0000000, 0x180000000, 0x40000000, 0x3c0000000),
    );
    assert!(a.sup(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_sum__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(fx(0xc0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_sum__sum() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(fx(0xc0000000));
    assert!(a.sum() == e);
}

#[test]
#[inline(never)]
fn bench_vector6_dot__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(fx(0x60000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_dot__wide() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(fx(0x60000000));
    assert!(a.dot(b) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_dot__alt_two_sum_prod3() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(fx(0x60000000));
    assert!(alt_dot_two_sum_prod3(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_dot__alt_unfused() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let e = black_box(fx(0x60000000));
    assert!(alt_dot_unfused(a, b) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_norm_squared__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(fx(0x2db0000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_norm_squared__wide() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(fx(0x2db0000000));
    assert!(a.norm_squared() == e);
}

#[test]
#[inline(never)]
fn bench_vector6_norm_squared__alt_two_blocks() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let e = black_box(fx(0x2db0000000));
    assert!(alt_norm_squared_two_blocks(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_norm__baseline() {
    let _a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let e = black_box(fx(0x600000000));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_norm__wide_sqrt() {
    let a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let e = black_box(fx(0x600000000));
    assert!(a.norm() == e);
}

#[test]
#[inline(never)]
fn bench_vector6_norm__alt_via_norm_squared() {
    let a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let e = black_box(fx(0x600000000));
    assert!(alt_norm_via_norm_squared(a) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_normalize__baseline() {
    let _a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let e = black_box(v6(1431655765, -1431655766, 1431655765, -1431655766, 1431655765, 2863311530));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_normalize__unscale_by_norm() {
    let a = black_box(
        v6(0x200000000, -0x200000000, 0x200000000, -0x200000000, 0x200000000, 0x400000000),
    );
    let e = black_box(v6(1431655765, -1431655765, 1431655765, -1431655765, 1431655765, 2863311531));
    assert!(a.normalize() == e);
}

#[test]
#[inline(never)]
fn bench_vector6_lerp__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let _t = black_box(fx(0x80000000));
    let e = black_box(
        v6(-0x180000000, -0x100000000, 0x2e0000000, -0x180000000, -0x100000000, 0x2e0000000),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_vector6_lerp__fused() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(-0x480000000, 0x40000000, 0x200000000, 0x180000000, -0x240000000, 0x3c0000000),
    );
    let t = black_box(fx(0x80000000));
    let e = black_box(
        v6(-0x180000000, -0x100000000, 0x2e0000000, -0x180000000, -0x100000000, 0x2e0000000),
    );
    assert!(a.lerp(b, t) == e);
}

#[test]
#[inline(never)]
fn bench_vector6_abs_diff_eq__baseline() {
    let _a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let _b = black_box(
        v6(0x180000001, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_vector6_abs_diff_eq__all_compared() {
    let a = black_box(
        v6(0x180000000, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    let b = black_box(
        v6(0x180000001, -0x240000000, 0x3c0000000, -0x480000000, 0x40000000, 0x200000000),
    );
    assert!(a.abs_diff_eq(b, 1));
}
