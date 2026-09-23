//! Gas benchmarks of `Quaternion` (`bench_quaternion_<op>__<variant>`, net = raw - `baseline` of
//! the group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{ONE_RAW, fx, int, q};
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use super::{Quaternion, QuaternionTrait};


/// `1 + 2i - 3j + 4k`, of squared norm 30 (norm 5.477).
fn a() -> Quaternion<Fixed> {
    q(ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW, 4 * ONE_RAW)
}

/// `-2 + i + 5j - k`, of squared norm 31.
fn b() -> Quaternion<Fixed> {
    q(-2 * ONE_RAW, ONE_RAW, 5 * ONE_RAW, -ONE_RAW)
}

// --- alternative implementations (losers)

/// The Hamilton product through `Real::sum_prod4`, the three minus signs carried by negated
/// operands of `lhs` instead of being folded into the accumulation: the same bits, the same number
/// of products, plus three negations — and a component equal to `MIN` panics. Out of line like
/// the shipped operator, so the two figures are comparable.
fn alt_mul_sum_prod4(l: Quaternion<Fixed>, r: Quaternion<Fixed>) -> Quaternion<Fixed> {
    let Quaternion { i: ai, j: aj, k: ak, w: aw } = l;
    let Quaternion { i: bi, j: bj, k: bk, w: bw } = r;
    let (ni, nj, nk) = (-ai, -aj, -ak);
    Quaternion {
        i: Real::sum_prod4(aw, bi, ai, bw, aj, bk, nk, bj),
        j: Real::sum_prod4(aw, bj, ni, bk, aj, bw, ak, bi),
        k: Real::sum_prod4(aw, bk, ai, bj, nj, bi, ak, bw),
        w: Real::sum_prod4(aw, bw, ni, bi, nj, bj, nk, bk),
    }
}

/// The Hamilton product with one rounding and one overflow check per product (what AGENTS.md
/// rule 4 forbids): four times the roundings, and it overflows on intermediate products.
#[inline(always)]
fn alt_mul_unfused(l: Quaternion<Fixed>, r: Quaternion<Fixed>) -> Quaternion<Fixed> {
    Quaternion {
        w: l.w * r.w - l.i * r.i - l.j * r.j - l.k * r.k,
        i: l.w * r.i + l.i * r.w + l.j * r.k - l.k * r.j,
        j: l.w * r.j - l.i * r.k + l.j * r.w + l.k * r.i,
        k: l.w * r.k + l.i * r.j - l.j * r.i + l.k * r.w,
    }
}

/// `normalize` through the reciprocal of the norm: one division and four products instead of four
/// divisions.
#[inline(always)]
fn alt_normalize_recip(x: Quaternion<Fixed>) -> Quaternion<Fixed> {
    let f = Real::recip(x.norm());
    x.scale(f)
}

/// `try_inverse` through the reciprocal of the squared norm: one division and four products.
#[inline(always)]
fn alt_try_inverse_recip(x: Quaternion<Fixed>) -> Option<Quaternion<Fixed>> {
    let n2 = x.norm_squared();
    if n2 == Real::ZERO {
        None
    } else {
        Some(x.conjugate().scale(Real::recip(n2)))
    }
}

// --- why the alternatives lost

/// The wide accumulation and the negated operands give the same bits (the accumulation is exact
/// either way and negating an operand is exact): only the gas differs, and the shipped form wins.
#[test]
fn test_mul_alt_wide_is_bit_identical() {
    assert!(alt_mul_sum_prod4(a(), b()) == a() * b());
    let u = q(2576980377, 1145324612, 2290649224, -2290649225);
    let v = q(2147483648, 2147483648, 2147483648, -2147483648);
    assert!(alt_mul_sum_prod4(u, v) == u * v);
    assert!(alt_mul_sum_prod4(v, u) == v * u);
    let t = q(56756, 56756, 56756, 56756);
    assert!(alt_mul_sum_prod4(t, t) == t * t);
}

/// The one input where the two forms differ: the alternative negates three components of `lhs`, so
/// a component equal to the scalar's `MIN` panics, where the shipped wide accumulation returns the
/// exact product.
#[test]
fn test_mul_alt_sum_prod4_cannot_take_min() {
    let m = q(0, -0x8000000000000000, 0, 0);
    assert!(m * QuaternionTrait::<Fixed>::identity() == m);
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_mul_alt_sum_prod4_of_min_panics() {
    let m = q(0, -0x8000000000000000, 0, 0);
    let _ = alt_mul_sum_prod4(black_box(m), black_box(QuaternionTrait::<Fixed>::identity()));
}

#[test]
fn test_mul_alt_unfused_is_less_accurate() {
    // Four products of 0.75 ulp each: the fused kernel sums them exactly before flooring, the
    // unfused version floors each one to zero.
    let t = q(56756, 56756, 56756, 56756);
    assert!((t * t).w == fx(-2));
    assert!(alt_mul_unfused(t, t).w == fx(0));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_mul_alt_unfused_overflows_on_intermediates() {
    // |i| = 50 000: i·i does not fit, although the result (0) does.
    let t = q(0, 50000 * ONE_RAW, 0, 0);
    let u = q(0, 50000 * ONE_RAW, 0, 0);
    let _ = alt_mul_unfused(black_box(t), black_box(u));
}

#[test]
fn test_normalize_alt_recip_is_less_accurate() {
    // |a| = 5.477: the rounded reciprocal shifts the last bit of the longest components.
    let exact = a().normalize();
    let approx = alt_normalize_recip(a());
    assert!(!approx.abs_diff_eq(exact, 0));
    assert!(approx.abs_diff_eq(exact, 1));
}

#[test]
fn test_try_inverse_alt_recip_is_less_accurate() {
    let exact = a().try_inverse().unwrap();
    let approx = alt_try_inverse_recip(a()).unwrap();
    // 2 ulp apart with `fixed`'s correctly rounded division and reciprocal (3 with the former
    // floor).
    assert!(!approx.abs_diff_eq(exact, 1));
    assert!(approx.abs_diff_eq(exact, 2));
    assert!(alt_try_inverse_recip(QuaternionTrait::<Fixed>::zero()) == None);
}

// --- constructors and parts

#[test]
#[inline(never)]
fn bench_quaternion_new__baseline() {
    let _w = black_box(int(1));
    let _i = black_box(int(2));
    let _j = black_box(int(-3));
    let _k = black_box(int(4));
    let e = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_new__new() {
    let w = black_box(int(1));
    let i = black_box(int(2));
    let j = black_box(int(-3));
    let k = black_box(int(4));
    let e = black_box(a());
    assert!(QuaternionTrait::new(w, i, j, k) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_identity__baseline() {
    let e = black_box(q(ONE_RAW, 0, 0, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_identity__const() {
    let e = black_box(q(ONE_RAW, 0, 0, 0));
    assert!(QuaternionTrait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_from_parts__baseline() {
    let _s = black_box(int(1));
    let _v = black_box(Vector3 { x: int(2), y: int(-3), z: int(4) });
    let e = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_from_parts__struct() {
    let s = black_box(int(1));
    let v = black_box(Vector3 { x: int(2), y: int(-3), z: int(4) });
    let e = black_box(a());
    assert!(QuaternionTrait::from_parts(s, v) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_from_imag__baseline() {
    let _v = black_box(Vector3 { x: int(2), y: int(-3), z: int(4) });
    let e = black_box(q(0, 2 * ONE_RAW, -3 * ONE_RAW, 4 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_from_imag__struct() {
    let v = black_box(Vector3 { x: int(2), y: int(-3), z: int(4) });
    let e = black_box(q(0, 2 * ONE_RAW, -3 * ONE_RAW, 4 * ONE_RAW));
    assert!(QuaternionTrait::from_imag(v) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_vector__baseline() {
    let _x = black_box(a());
    let e = black_box(Vector3 { x: int(2), y: int(-3), z: int(4) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_vector__struct() {
    let x = black_box(a());
    let e = black_box(Vector3 { x: int(2), y: int(-3), z: int(4) });
    assert!(x.vector() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_as_vector__baseline() {
    let _x = black_box(a());
    let e = black_box(Vector4 { x: int(2), y: int(-3), z: int(4), w: int(1) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_as_vector__struct() {
    let x = black_box(a());
    let e = black_box(Vector4 { x: int(2), y: int(-3), z: int(4), w: int(1) });
    assert!(x.as_vector() == e);
}

// --- algebra

#[test]
#[inline(never)]
fn bench_quaternion_add__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let e = black_box(q(-ONE_RAW, 3 * ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_add__op() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(-ONE_RAW, 3 * ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW));
    assert!(x + y == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sub__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let e = black_box(q(3 * ONE_RAW, ONE_RAW, -8 * ONE_RAW, 5 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sub__op() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(3 * ONE_RAW, ONE_RAW, -8 * ONE_RAW, 5 * ONE_RAW));
    assert!(x - y == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_neg__baseline() {
    let _x = black_box(a());
    let e = black_box(q(-ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW, -4 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_neg__op() {
    let x = black_box(a());
    let e = black_box(q(-ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW, -4 * ONE_RAW));
    assert!(-x == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_mul__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let e = black_box(q(15 * ONE_RAW, -20 * ONE_RAW, 17 * ONE_RAW, 4 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_mul__fused_wide() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(15 * ONE_RAW, -20 * ONE_RAW, 17 * ONE_RAW, 4 * ONE_RAW));
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_mul__alt_sum_prod4() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(15 * ONE_RAW, -20 * ONE_RAW, 17 * ONE_RAW, 4 * ONE_RAW));
    assert!(alt_mul_sum_prod4(x, y) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_mul__alt_unfused() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(15 * ONE_RAW, -20 * ONE_RAW, 17 * ONE_RAW, 4 * ONE_RAW));
    assert!(alt_mul_unfused(x, y) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_conj_mul__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let e = black_box(q(-19 * ONE_RAW, 22 * ONE_RAW, -7 * ONE_RAW, -6 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_conj_mul__fused() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(-19 * ONE_RAW, 22 * ONE_RAW, -7 * ONE_RAW, -6 * ONE_RAW));
    assert!(x.conj_mul(y) == e);
}

/// The formulation `conj_mul` replaces: three negations, then the Hamilton product.
#[test]
#[inline(never)]
fn bench_quaternion_conj_mul__alt_conjugate_then_mul() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(-19 * ONE_RAW, 22 * ONE_RAW, -7 * ONE_RAW, -6 * ONE_RAW));
    assert!(x.conjugate() * y == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_scale__baseline() {
    let _x = black_box(a());
    let _k = black_box(int(2));
    let e = black_box(q(2 * ONE_RAW, 4 * ONE_RAW, -6 * ONE_RAW, 8 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_scale__products() {
    let x = black_box(a());
    let k = black_box(int(2));
    let e = black_box(q(2 * ONE_RAW, 4 * ONE_RAW, -6 * ONE_RAW, 8 * ONE_RAW));
    assert!(x.scale(k) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_unscale__baseline() {
    let _x = black_box(a());
    let _k = black_box(int(2));
    let e = black_box(q(ONE_RAW / 2, ONE_RAW, -3 * ONE_RAW / 2, 2 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_unscale__divisions() {
    let x = black_box(a());
    let k = black_box(int(2));
    let e = black_box(q(ONE_RAW / 2, ONE_RAW, -3 * ONE_RAW / 2, 2 * ONE_RAW));
    assert!(x.unscale(k) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_conjugate__baseline() {
    let _x = black_box(a());
    let e = black_box(q(ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW, -4 * ONE_RAW));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_conjugate__neg() {
    let x = black_box(a());
    let e = black_box(q(ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW, -4 * ONE_RAW));
    assert!(x.conjugate() == e);
}

// --- norms and products

#[test]
#[inline(never)]
fn bench_quaternion_norm__baseline() {
    let _x = black_box(a());
    let e = black_box(fx(23524504717));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_norm__norm4() {
    let x = black_box(a());
    let e = black_box(fx(23524504717));
    assert!(x.norm() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_norm_squared__baseline() {
    let _x = black_box(a());
    let e = black_box(int(30));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_norm_squared__fused() {
    let x = black_box(a());
    let e = black_box(int(30));
    assert!(x.norm_squared() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_dot__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let e = black_box(int(-19));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_dot__fused() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(int(-19));
    assert!(x.dot(y) == e);
}

// --- normalization and inverse

#[test]
#[inline(never)]
fn bench_quaternion_normalize__baseline() {
    let _x = black_box(a());
    let e = black_box(q(784150157, 1568300314, -2352450472, 3136600629));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_normalize__divisions() {
    let x = black_box(a());
    let e = black_box(q(784150157, 1568300314, -2352450471, 3136600629));
    assert!(x.normalize() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_normalize__alt_recip() {
    let x = black_box(a());
    let e = black_box(q(784150157, 1568300314, -2352450471, 3136600628));
    assert!(alt_normalize_recip(x) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_try_inverse__baseline() {
    let _x = black_box(a());
    let e = black_box(q(143165576, -286331154, 429496729, -572662307));
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_quaternion_try_inverse__divisions() {
    let x = black_box(a());
    let e = black_box(q(143165576, -286331153, 429496729, -572662306));
    assert!(x.try_inverse() == Some(e));
}

#[test]
#[inline(never)]
fn bench_quaternion_try_inverse__alt_recip() {
    let x = black_box(a());
    let e = black_box(q(143165576, -286331152, 429496728, -572662304));
    assert!(alt_try_inverse_recip(x) == Some(e));
}

// --- interpolation and comparison

#[test]
#[inline(never)]
fn bench_quaternion_lerp__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let _t = black_box(fx(ONE_RAW / 4));
    let e = black_box(q(1073741824, 7516192768, -4294967296, 11811160064));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_lerp__fused() {
    let x = black_box(a());
    let y = black_box(b());
    let t = black_box(fx(ONE_RAW / 4));
    let e = black_box(q(1073741824, 7516192768, -4294967296, 11811160064));
    assert!(x.lerp(y, t) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_abs_diff_eq__baseline() {
    let _x = black_box(a());
    let _y = black_box(b());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_abs_diff_eq__components() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(false);
    assert!(x.abs_diff_eq(y, 4) == e);
}
