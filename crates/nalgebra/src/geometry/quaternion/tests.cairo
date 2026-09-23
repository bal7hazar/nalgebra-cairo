//! Unit tests of `Quaternion`: the argument / storage conventions, exact integer cases, the
//! algebra identities (`i·j = k`, `q · q⁻¹ = 1`), panics, and the oracle vectors of
//! `tools/oracle`
//! (upstream nalgebra 0.35 on the same raw inputs, tolerance in ulp).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo quaternion --from vectors --max-per-dist 4 \
//!     --ops quaternion_mul,quaternion_conjugate,quaternion_norm,quaternion_normalize,\
//! quaternion_try_inverse --out crates/nalgebra/src/geometry/quaternion/oracle.cairo
//! ```

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{ONE_RAW, fx, int, qi, qt, v3};
use crate::base::vector4::Vector4;
use super::{Quaternion, QuaternionTrait, oracle};

const MIN: i64 = -0x8000000000000000;

/// `1 + 2i - 3j + 4k`, of squared norm 30.
fn a() -> Quaternion<Fixed> {
    qi(1, 2, -3, 4)
}

/// `-2 + i + 5j - k`, of squared norm 31.
fn b() -> Quaternion<Fixed> {
    qi(-2, 1, 5, -1)
}

// --- conventions

#[test]
fn test_new_is_w_first() {
    let x = QuaternionTrait::new(int(1), int(2), int(3), int(4));
    assert!(x.w == int(1) && x.i == int(2) && x.j == int(3) && x.k == int(4));
    assert!(x == Quaternion { i: int(2), j: int(3), k: int(4), w: int(1) });
}

/// Upstream stores `coords: Vector4 = [i, j, k, w]` and serialises that order; so do we.
#[test]
fn test_serde_is_imag_first() {
    let mut out = array![];
    QuaternionTrait::new(int(1), int(2), int(3), int(4)).serialize(ref out);
    let one: felt252 = 0x100000000;
    assert!(out == array![2 * one, 3 * one, 4 * one, 1 * one]);
    let mut span = out.span();
    let back: Quaternion<Fixed> = Serde::deserialize(ref span).unwrap();
    assert!(back == QuaternionTrait::new(int(1), int(2), int(3), int(4)));
}

#[test]
fn test_as_vector_is_imag_first() {
    assert!(a().as_vector() == Vector4 { x: int(2), y: int(-3), z: int(4), w: int(1) });
    let back: Quaternion<Fixed> = a().as_vector().into();
    assert!(back == a());
    assert!(QuaternionTrait::from_vector(a().as_vector()) == a());
    let coords: Vector4<Fixed> = a().into();
    assert!(coords == a().as_vector());
}

// --- constructors and parts

#[test]
fn test_identity_zero_and_from_real() {
    assert!(QuaternionTrait::<Fixed>::identity() == qi(1, 0, 0, 0));
    assert!(QuaternionTrait::<Fixed>::zero() == qi(0, 0, 0, 0));
    assert!(QuaternionTrait::from_real(int(-5)) == qi(-5, 0, 0, 0));
    assert!(QuaternionTrait::<Fixed>::identity() == Default::default() + qi(1, 0, 0, 0));
}

#[test]
fn test_from_parts_and_from_imag() {
    assert!(QuaternionTrait::from_parts(int(1), v3(int(2).raw, int(-3).raw, int(4).raw)) == a());
    assert!(QuaternionTrait::from_imag(v3(int(2).raw, int(-3).raw, int(4).raw)) == qi(0, 2, -3, 4));
}

#[test]
fn test_scalar_vector_imag() {
    assert!(a().scalar() == int(1));
    assert!(a().vector() == v3(int(2).raw, int(-3).raw, int(4).raw));
    assert!(a().imag() == a().vector());
    assert!(QuaternionTrait::from_parts(a().scalar(), a().vector()) == a());
}

// --- addition, subtraction, negation

#[test]
fn test_add_sub_neg_are_exact() {
    assert!(a() + b() == qi(-1, 3, 2, 3));
    assert!(a() - b() == qi(3, 1, -8, 5));
    assert!(-a() == qi(-1, -2, 3, -4));
    assert!(a() + (-a()) == QuaternionTrait::zero());
    assert!(a() - a() == QuaternionTrait::zero());
}

#[test]
#[should_panic(expected: 'i64_add Overflow')]
fn test_add_overflow_panics() {
    let big = qt((0x7fffffffffffffff, 0, 0, 0));
    let _ = black_box(big) + black_box(qt((1, 0, 0, 0)));
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_neg_min_panics() {
    let _ = -black_box(qt((MIN, 0, 0, 0)));
}

// --- Hamilton product

#[test]
fn test_mul_exact_integer_case() {
    assert!(a() * b() == qi(15, -20, 17, 4));
    // Non-commutative: only the real part and the sign of the vector part are shared.
    assert!(b() * a() == qi(15, 14, 5, -22));
}

#[test]
fn test_mul_basis_relations() {
    let one = QuaternionTrait::<Fixed>::identity();
    let i = qi(0, 1, 0, 0);
    let j = qi(0, 0, 1, 0);
    let k = qi(0, 0, 0, 1);
    assert!(i * j == k);
    assert!(j * k == i);
    assert!(k * i == j);
    assert!(j * i == -k);
    assert!(i * i == -one);
    assert!(j * j == -one);
    assert!(k * k == -one);
    assert!(i * j * k == -one);
    assert!(one * a() == a());
    assert!(a() * one == a());
}

/// The oracle expectation is the EXACT floor of the product (computed on `i128` raws), and one
/// rescale per output component reproduces it bit for bit: the tolerance of 1 ulp is not used.
#[test]
fn test_mul_oracle_is_bit_exact() {
    let mut cases = oracle::quaternion_mul_cases();
    assert!(cases.len() == 16);
    while let Some(case) = cases.pop_front() {
        let (ra, rb, expected, _tol) = *case;
        assert!(qt(ra) * qt(rb) == qt(expected));
    }
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_mul_overflow_panics() {
    // 50 000² > 2^31: the real part does not fit.
    let big = qt((50000 * ONE_RAW, 0, 0, 0));
    let _ = black_box(big) * black_box(big);
}

/// The norm is multiplicative, so the product of two unit quaternions is a unit quaternion (within
/// the rounding of the four components).
#[test]
fn test_mul_preserves_the_norm() {
    // Rotation of angle 2·acos(0.6) about (1, 2, -2) / 3, and of 2·pi/3 about (1, 1, -1) /
    // sqrt(3).
    let u = qt((2576980377, 1145324612, 2290649224, -2290649225));
    let v = qt((2147483648, 2147483648, 2147483648, -2147483648));
    assert!(u.norm().abs_diff_eq(Real::ONE, 1));
    assert!(v.norm() == Real::ONE);
    assert!((u * v).norm().abs_diff_eq(Real::ONE, 1));
    assert!(u * v == qt((-1574821342, 1861152495, 1861152494, -3006477107)));
}

// --- conjugate, norms, dot

#[test]
fn test_conjugate_is_exact() {
    assert!(a().conjugate() == qi(1, -2, 3, -4));
    assert!(a().conjugate().conjugate() == a());
    // q * conj(q) = |q|² and conj(a*b) = conj(b) * conj(a).
    assert!(a() * a().conjugate() == qi(30, 0, 0, 0));
    assert!((a() * b()).conjugate() == b().conjugate() * a().conjugate());
}

#[test]
fn test_conjugate_oracle() {
    let mut cases = oracle::quaternion_conjugate_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(tol == 0);
        assert!(qt(rq).conjugate() == qt(expected));
    }
}

// --- conj_mul

#[test]
fn test_conj_mul_exact_integer_case() {
    assert!(a().conj_mul(b()) == qi(-19, 22, -7, -6));
    assert!(a().conj_mul(b()) == a().conjugate() * b());
    assert!(b().conj_mul(a()) == b().conjugate() * a());
}

/// `conj(q) · q = |q|²`, a real quaternion, exactly for integer components.
#[test]
fn test_conj_mul_self_is_the_squared_norm() {
    assert!(a().conj_mul(a()) == qi(30, 0, 0, 0));
    assert!(b().conj_mul(b()) == qi(31, 0, 0, 0));
    let one = QuaternionTrait::<Fixed>::identity();
    assert!(one.conj_mul(a()) == a());
    assert!(a().conj_mul(one) == a().conjugate());
}

/// Bit-identical to `conjugate() * other` on every oracle input of the Hamilton product, in both
/// orders.
#[test]
fn test_conj_mul_oracle_matches_conjugate_then_mul() {
    let mut cases = oracle::quaternion_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, _expected, _tol) = *case;
        assert!(qt(ra).conj_mul(qt(rb)) == qt(ra).conjugate() * qt(rb));
        assert!(qt(rb).conj_mul(qt(ra)) == qt(rb).conjugate() * qt(ra));
    }
}

/// The one behavioural difference: a component equal to `MIN` is never negated, so it does not
/// panic (`conjugate()` does, see `test_neg_min_panics`). `-(MIN · 2^-32)` floors to `2^31` raw.
#[test]
fn test_conj_mul_min_component_does_not_panic() {
    let m = black_box(qt((0, MIN, 0, 0)));
    let r = black_box(qt((1, 0, 0, 0)));
    assert!(m.conj_mul(r) == qt((0, 0x80000000, 0, 0)));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_conj_mul_overflow_panics() {
    // 50 000² > 2^31: the real part does not fit.
    let big = qt((50000 * ONE_RAW, 0, 0, 0));
    let _ = black_box(big).conj_mul(black_box(big));
}

#[test]
fn test_norm_and_norm_squared_exact() {
    assert!(a().norm_squared() == int(30));
    assert!(qi(1, 1, 1, 1).norm() == int(2));
    assert!(qi(0, 3, 4, 0).norm() == int(5));
    assert!(QuaternionTrait::<Fixed>::identity().norm() == Real::ONE);
    assert!(a().norm() == a().norm_squared().sqrt());
}

#[test]
fn test_norm_oracle() {
    let mut cases = oracle::quaternion_norm_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(Real::abs_diff_eq(qt(rq).norm(), fx(expected), tol));
    }
}

/// `norm` has no intermediate overflow; `norm_squared` does above a norm of about 46 340.
#[test]
fn test_norm_huge_does_not_overflow() {
    assert!(qt((0x186a000000000, 0, 0, 0)).norm() == int(100000));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_norm_squared_overflow_panics() {
    let _ = black_box(qt((0x186a000000000, 0, 0, 0))).norm_squared();
}

#[test]
fn test_dot_is_fused() {
    assert!(a().dot(b()) == int(2 * 1 + (-3) * 5 + 4 * (-1) + 1 * (-2)));
    assert!(a().dot(a()) == a().norm_squared());
    // Four products of 0.5 ulp: the fused kernel sums them exactly before flooring.
    let tiny = qt((46341, 46341, 46341, 0));
    assert!(tiny.dot(tiny) == fx(1));
}

// --- scaling, normalization, inverse

#[test]
fn test_scale_and_unscale() {
    assert!(a().scale(int(2)) == qi(2, 4, -6, 8));
    assert!(a().scale(Real::ZERO) == QuaternionTrait::zero());
    assert!(a().unscale(int(2)) == qt((ONE_RAW / 2, ONE_RAW, -3 * ONE_RAW / 2, 2 * ONE_RAW)));
    // Floor division: -1 / 2 = -0.5 exactly, -1 / 3 rounds down.
    assert!(qi(-1, 0, 0, 0).unscale(int(3)) == qt((-1431655765, 0, 0, 0)));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_unscale_by_zero_panics() {
    let _ = black_box(a()).unscale(Real::ZERO);
}

#[test]
fn test_normalize_exact_and_oracle() {
    // (0, 3, 4, 0) / 5 and (1, 1, 1, 1) / 2 are exact.
    assert!(qi(0, 3, 4, 0).normalize() == qt((0, 2576980377, 3435973836, 0)));
    assert!(qi(1, 1, 1, 1).normalize() == qt((0x80000000, 0x80000000, 0x80000000, 0x80000000)));
    let mut cases = oracle::quaternion_normalize_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(qt(rq).normalize().abs_diff_eq(qt(expected), tol));
        assert!(qt(rq).normalize().norm().abs_diff_eq(Real::ONE, 4));
    }
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_normalize_zero_panics() {
    let _ = black_box(QuaternionTrait::<Fixed>::zero()).normalize();
}

#[test]
fn test_try_inverse_exact() {
    // 1/2 = 0.5: (2, 0, 0, 0)⁻¹ = (0.5, 0, 0, 0).
    assert!(qi(2, 0, 0, 0).try_inverse() == Some(qt((0x80000000, 0, 0, 0))));
    assert!(QuaternionTrait::<Fixed>::identity().try_inverse() == Some(qi(1, 0, 0, 0)));
    // A unit quaternion: the inverse is the conjugate.
    let u = qt((2147483648, 2147483648, 2147483648, -2147483648));
    assert!(u.try_inverse() == Some(u.conjugate()));
}

#[test]
fn test_try_inverse_product_is_identity() {
    let mut cases = oracle::quaternion_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        let inv = qt(rq).try_inverse().unwrap();
        assert!(inv.abs_diff_eq(qt(expected), tol));
        // q · q⁻¹ = 1 within the rounding of the inverse times the magnitude of q.
        let one = qt(rq) * inv;
        let scale: u64 = 4 + qt(rq).norm().abs().raw.try_into().unwrap() / 0x100000000;
        assert!(one.abs_diff_eq(QuaternionTrait::identity(), tol * scale + 4));
    }
}

#[test]
fn test_try_inverse_none_on_zero() {
    assert!(QuaternionTrait::<Fixed>::zero().try_inverse() == None);
    // A quaternion so short that its squared norm floors to zero (norm < 2^-16).
    assert!(qt((1, -1, 0, 0)).try_inverse() == None);
}

// --- interpolation and comparison

#[test]
fn test_lerp_endpoints_are_exact() {
    assert!(a().lerp(b(), Real::ZERO) == a());
    assert!(a().lerp(b(), Real::ONE) == b());
    assert!(a().lerp(b(), Real::HALF) == qt((-0x80000000, 0x180000000, 0x100000000, 0x180000000)));
    // Not clamped: t = 2 extrapolates.
    assert!(a().lerp(b(), int(2)) == qi(-5, 0, 13, -6));
}

#[test]
fn test_abs_diff_eq_counts_ulps() {
    assert!(a().abs_diff_eq(a(), 0));
    let drifted = qt((int(1).raw + 2, int(2).raw - 2, int(-3).raw + 1, int(4).raw));
    assert!(a().abs_diff_eq(drifted, 2));
    assert!(!a().abs_diff_eq(drifted, 1));
    // q and -q are the same rotation but not close.
    assert!(!a().abs_diff_eq(-a(), 1000));
}
