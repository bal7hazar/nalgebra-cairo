//! Unit tests of `UnitComplex`: exact cases (identity, quarter turns, axes), the identities a
//! rotation must satisfy (`c · c⁻¹ = 1`, `R · Rᵀ = I`, `c → R → c`), panics on
//! degenerate inputs, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same
//! raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution,
//! the `unit_complex_*` ops of the `rotation2` suite) with
//! `cargo run --release -- emit-cairo rotation2 --from vectors --max-per-dist 6 --out
//! <oracle.cairo> --ops <list>`, `<list>` being the comma-separated op names of the oracle tests
//! at the bottom of this file.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{ONE_RAW, fx, m2, max_ulp_diff_uc, p2t, r2, uct, ulp_diff, v2t};
use crate::base::vector2::{Vector2, Vector2Trait};
use crate::geometry::rotation2::Rotation2Trait;
use super::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait, oracle};

/// The raw value of 1.

fn id() -> UnitComplex<Fixed> {
    UnitComplexTrait::<Fixed>::identity()
}

/// The quarter turn `(0, 1)`, an exactly representable rotation.
fn quarter() -> UnitComplex<Fixed> {
    uct((0, ONE_RAW))
}

// --- constructors and accessors

#[test]
fn test_identity_is_exact() {
    let c = id();
    assert!(c.re == Real::ONE && c.im == Real::ZERO);
    assert!(c.angle() == Real::ZERO);
}

#[test]
fn test_from_cos_sin_unchecked_sets_fields() {
    let c = UnitComplexTrait::from_cos_sin_unchecked(fx(0x80000000), fx(-0x40000000));
    assert!(c.re == fx(0x80000000) && c.im == fx(-0x40000000));
    assert!(c == UnitComplex { re: fx(0x80000000), im: fx(-0x40000000) });
}

#[test]
fn test_new_cardinal_angles() {
    // sin_cos is accurate to 0.53 ulp (DESIGN D6): the components land within 1 ulp of the exact
    // (0, 1), (-1, 0), (0, -1).
    assert!(UnitComplexAngleTrait::<Fixed>::new(Real::ZERO) == id());
    assert!(max_ulp_diff_uc(UnitComplexAngleTrait::<Fixed>::new(Real::FRAC_PI_2), quarter()) <= 1);
    assert!(
        max_ulp_diff_uc(UnitComplexAngleTrait::<Fixed>::new(Real::PI), uct((-ONE_RAW, 0))) <= 1,
    );
    assert!(
        max_ulp_diff_uc(
            UnitComplexAngleTrait::<Fixed>::new(-Real::<Fixed>::FRAC_PI_2), uct((0, -ONE_RAW)),
        ) <= 1,
    );
}

#[test]
fn test_new_is_unit_within_two_ulp() {
    // Each component is floored once, so `re² + im²` is 1 within about 2 ulp.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x123456789));
    let n = Real::norm_squared2(c.re, c.im);
    assert!(ulp_diff(n, Real::ONE) <= 2);
}

#[test]
fn test_from_angle_is_new() {
    let a = fx(-0x1c0000000);
    assert!(UnitComplexAngleTrait::<Fixed>::from_angle(a) == UnitComplexAngleTrait::new(a));
}

#[test]
fn test_accessors() {
    let c = uct((0x80000000, -0x40000000));
    assert!(c.re() == fx(0x80000000) && c.im() == fx(-0x40000000));
    assert!(c.cos_angle() == c.re && c.sin_angle() == c.im);
    assert!(c.complex() == v2t((0x80000000, -0x40000000)));
}

// --- conjugate, inverse, composition

#[test]
fn test_conjugate_and_inverse_agree_and_are_exact() {
    let c = uct((0x80000000, -0x40000000));
    assert!(c.conjugate() == uct((0x80000000, 0x40000000)));
    assert!(c.inverse() == c.conjugate());
    assert!(c.inverse().inverse() == c);
}

#[test]
#[should_panic(expected: 'i64_neg Underflow')]
fn test_inverse_of_min_imaginary_part_panics() {
    let _ = black_box(UnitComplex { re: Real::<Fixed>::ZERO, im: Real::MIN }).inverse();
}

#[test]
fn test_mul_identity_is_neutral_and_exact() {
    let c = uct((0x80000000, -0x40000000));
    assert!(c * id() == c);
    assert!(id() * c == c);
}

#[test]
fn test_mul_quarter_turns_are_exact() {
    // i * i = -1, i * (-1) = -i, and four quarter turns are the identity, bit for bit.
    let i = quarter();
    assert!(i * i == uct((-ONE_RAW, 0)));
    assert!(i * i * i == uct((0, -ONE_RAW)));
    assert!(i * i * i * i == id());
}

#[test]
fn test_mul_is_commutative_bit_for_bit() {
    // The product of complex numbers is symmetric in its two operands, floors included.
    let (a, b) = (uct((0x80000000, -0xdd6a9c1)), uct((-0x5a827999, 0xb504f334)));
    assert!(a * b == b * a);
}

#[test]
fn test_mul_inverse_is_identity_within_two_ulp() {
    // `c · c⁻¹` is exactly `(re² + im², 0)`: the identity up to the drift of the norm.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let p = c * c.inverse();
    assert!(p.im == Real::ZERO);
    assert!(ulp_diff(p.re, Real::ONE) <= 2);
    assert!(p.abs_diff_eq(id(), 2));
}

#[test]
fn test_mul_adds_angles() {
    let (a, b) = (fx(0x66666666), fx(-0x2aaaaaaa)); // 0.4 and -0.1666 rad
    let got = UnitComplexAngleTrait::<Fixed>::new(a) * UnitComplexAngleTrait::new(b);
    assert!(max_ulp_diff_uc(got, UnitComplexAngleTrait::<Fixed>::new(a + b)) <= 8);
}

#[test]
fn test_rotation_to_and_angle_to() {
    let (a, b) = (
        UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666)),
        UnitComplexAngleTrait::<Fixed>::new(fx(0x1999999a)),
    );
    // a * a.rotation_to(b) = b, within the rounding of two compositions.
    assert!((a * a.rotation_to(b)).abs_diff_eq(b, 4));
    assert!(ulp_diff(a.angle_to(b), a.rotation_to(b).angle()) <= 1);
    assert!(ulp_diff(a.angle_to(b), fx(0x1999999a) - fx(0x66666666)) <= 16);
}

// --- transforms

#[test]
fn test_transform_vector_quarter_turn_is_exact() {
    // (x, y) -> (-y, x), with no rounding at all.
    let v = v2t((0x300000000, -0x400000000));
    assert!(quarter().transform_vector(v) == v2t((0x400000000, 0x300000000)));
    assert!(quarter().mul_vec(v) == quarter().transform_vector(v));
    assert!(id().transform_vector(v) == v);
}

#[test]
fn test_transform_point_matches_transform_vector() {
    let c = uct((0x80000000, -0xdd6a9c1));
    let p = p2t((0x300000000, -0x400000000));
    let r = c.transform_point(p);
    let v = c.transform_vector(v2t((0x300000000, -0x400000000)));
    assert!(r.x == v.x && r.y == v.y);
}

#[test]
fn test_inverse_transform_is_the_inverse_rotation() {
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let v = v2t((0x300000000, -0x400000000));
    // Bit for bit the transform by the conjugate, and a round trip within a few ulp.
    assert!(c.inverse_transform_vector(v) == c.inverse().transform_vector(v));
    assert!(c.inverse_transform_vector(c.transform_vector(v)).abs_diff_eq(v, 4));
    let p = p2t((0x300000000, -0x400000000));
    assert!(c.inverse_transform_point(p) == c.inverse().transform_point(p));
}

#[test]
fn test_transform_vector_preserves_length() {
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let v = v2t((0x300000000, -0x400000000)); // norm 5
    let r = c.transform_vector(v);
    assert!(ulp_diff(Real::norm2(r.x, r.y), fx(0x500000000)) <= 2);
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_transform_vector_overflow_panics() {
    // (1.6e9, 1.6e9) turned by -π/4 lands at (2.26e9, 0), past the 2.147e9 of Q32.32.
    let c = black_box(UnitComplexAngleTrait::<Fixed>::new(-Real::<Fixed>::FRAC_PI_4));
    let _ = c.transform_vector(black_box(v2t((0x6000000000000000, 0x6000000000000000))));
}

// --- matrix conversions

#[test]
fn test_to_rotation_matrix_and_back_is_exact() {
    let c = uct((0x80000000, -0xdd6a9c1));
    let r = c.to_rotation_matrix();
    assert!(r.matrix == m2([[0x80000000, 0xdd6a9c1], [-0xdd6a9c1, 0x80000000]]));
    assert!(UnitComplexTrait::from_rotation_matrix(r) == c);
    assert!(r.to_unit_complex() == c);
}

#[test]
fn test_rotation_matrix_is_orthogonal() {
    // R · Rᵀ = I: the columns are unit within the drift of `new`, so 2 ulp on the diagonal and
    // exactly 0 off it (the off-diagonal terms cancel exactly, `re·im - im·re`).
    let r = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a)).to_rotation_matrix();
    let p = r * r.inverse();
    assert!(p.matrix.m12 == Real::ZERO && p.matrix.m21 == Real::ZERO);
    assert!(ulp_diff(p.matrix.m11, Real::ONE) <= 2);
    assert!(ulp_diff(p.matrix.m22, Real::ONE) <= 2);
}

#[test]
fn test_to_homogeneous() {
    let c = uct((0x80000000, -0xdd6a9c1));
    let h = c.to_homogeneous();
    assert!(h.m11 == c.re && h.m22 == c.re);
    assert!(h.m21 == c.im && h.m12 == -c.im);
    assert!(h.m13 == Real::ZERO && h.m23 == Real::ZERO);
    assert!(h.m31 == Real::ZERO && h.m32 == Real::ZERO);
    assert!(h.m33 == Real::ONE);
    // The homogeneous matrix carries the same rotation block as the 2x2 one.
    assert!(h.m11 == c.to_rotation_matrix().matrix.m11);
    assert!(h.m12 == c.to_rotation_matrix().matrix.m12);
}

// --- rotation_between

#[test]
fn test_rotation_between_axes_is_exact() {
    let (x, y) = (v2t((0x300000000, 0)), v2t((0, 0x500000000)));
    assert!(UnitComplexTrait::rotation_between(x, y) == quarter());
    assert!(UnitComplexTrait::rotation_between(y, x) == uct((0, -ONE_RAW)));
    assert!(UnitComplexTrait::rotation_between(x, x) == id());
    assert!(UnitComplexTrait::rotation_between(x, v2t((-0x100000000, 0))) == uct((-ONE_RAW, 0)));
}

#[test]
fn test_rotation_between_maps_a_to_b() {
    let (a, b) = (v2t((0x300000000, -0x400000000)), v2t((0x180000000, 0x200000000)));
    // |a| = 5, |b| = 2.5: the rotation takes a to b scaled by |a| / |b|.
    let c = UnitComplexTrait::rotation_between(a, b);
    let got = c.transform_vector(a);
    assert!(got.abs_diff_eq(v2t((0x300000000, 0x400000000)), 4));
}

#[test]
fn test_rotation_between_zero_vector_is_identity() {
    let z = Vector2 { x: Real::<Fixed>::ZERO, y: Real::ZERO };
    assert!(UnitComplexTrait::rotation_between(z, v2t((0x300000000, 0))) == id());
    assert!(UnitComplexTrait::rotation_between(v2t((0x300000000, 0)), z) == id());
    assert!(UnitComplexTrait::rotation_between(z, z) == id());
}

#[test]
fn test_rotation_between_short_vectors_lose_precision() {
    // The same two directions, once at |a| = |b| ~ 2 and once 256 times shorter. `a·b` and `a×b`
    // are floored at 2^-32 either way, then blown up by 1 / (|a|·|b|), which the shortening
    // multiplies by 2^16. Documented in `UnitComplexTrait::rotation_between`.
    let long = UnitComplexTrait::rotation_between(
        v2t((0x30ec4a100, -0x41d5b9200)), v2t((0x40a3d7000, 0x2f1a9fc00)),
    );
    let short = UnitComplexTrait::rotation_between(
        v2t((0x30ec4a1, -0x41d5b92)), v2t((0x40a3d70, 0x2f1a9fc)),
    );
    assert!(short.abs_diff_eq(long, 8000));
    assert!(!short.abs_diff_eq(long, 200));
    // Directions whose dot and perp products are exact are exact at any length: (3, -4) to
    // (4, 3) is the quarter turn, bit for bit, even 256 times shorter.
    let (a, b) = (v2t((0x300000000, -0x400000000)), v2t((0x400000000, 0x300000000)));
    assert!(UnitComplexTrait::rotation_between(a, b) == quarter());
    let (a, b) = (v2t((0x3000000, -0x4000000)), v2t((0x4000000, 0x3000000)));
    assert!(UnitComplexTrait::rotation_between(a, b) == quarter());
}

#[test]
fn test_scaled_rotation_between() {
    let (x, y) = (v2t((0x300000000, 0)), v2t((0, 0x500000000)));
    // Half of a quarter turn is an eighth of a turn.
    let c = UnitComplexAngleTrait::scaled_rotation_between(x, y, Real::HALF);
    assert!(c.abs_diff_eq(UnitComplexAngleTrait::<Fixed>::new(Real::FRAC_PI_4), 4));
    // s = 1 is `rotation_between`, up to the trigonometry.
    let c = UnitComplexAngleTrait::scaled_rotation_between(x, y, Real::ONE);
    assert!(c.abs_diff_eq(quarter(), 4));
    let z = Vector2 { x: Real::<Fixed>::ZERO, y: Real::ZERO };
    assert!(UnitComplexAngleTrait::scaled_rotation_between(z, z, Real::ONE) == id());
}

// --- angle, powf, slerp

#[test]
fn test_angle_round_trip() {
    let a = fx(0x1f0a3d70a);
    assert!(ulp_diff(UnitComplexAngleTrait::<Fixed>::new(a).angle(), a) <= 12);
    assert!(quarter().angle() == Real::FRAC_PI_2);
    assert!(uct((0, -ONE_RAW)).angle() == -Real::<Fixed>::FRAC_PI_2);
    assert!(ulp_diff(uct((-ONE_RAW, 0)).angle(), Real::PI) <= 12);
}

#[test]
fn test_powf() {
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666)); // 0.4 rad
    assert!(c.powf(Real::ZERO) == id());
    assert!(c.powf(Real::ONE).abs_diff_eq(c, 16));
    assert!(c.powf(Real::TWO).abs_diff_eq(c * c, 32));
    assert!(c.powf(Real::NEG_ONE).abs_diff_eq(c.inverse(), 16));
}

#[test]
fn test_powf_takes_the_principal_angle() {
    // 3π/4 doubled is -π/2, not 3π/2: `angle()` lands in (-π, π] like upstream.
    let c = UnitComplexAngleTrait::<Fixed>::new(Real::FRAC_PI_4 + Real::FRAC_PI_2);
    assert!(c.powf(Real::TWO).abs_diff_eq(uct((0, -ONE_RAW)), 16));
}

#[test]
fn test_slerp_endpoints_and_midpoint() {
    let a = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666));
    let b = UnitComplexAngleTrait::<Fixed>::new(fx(-0x33333333));
    // t = 0 gives `a` back (the interpolated rotation is the identity, whose product is exact
    // up to the norm drift), t = 1 gives `b`.
    assert!(a.slerp(b, Real::ZERO).abs_diff_eq(a, 4));
    assert!(a.slerp(b, Real::ONE).abs_diff_eq(b, 32));
    // The midpoint is the half angle.
    let mid = a.slerp(b, Real::HALF);
    assert!(mid.abs_diff_eq(UnitComplexAngleTrait::<Fixed>::new(fx(0x1999999a)), 32));
}

#[test]
fn test_slerp_takes_the_shortest_arc() {
    // From -3π/4 to 3π/4 the short way is through π, not through 0.
    let a = UnitComplexAngleTrait::<Fixed>::new(-Real::<Fixed>::FRAC_PI_4 - Real::FRAC_PI_2);
    let b = UnitComplexAngleTrait::<Fixed>::new(Real::FRAC_PI_4 + Real::FRAC_PI_2);
    assert!(a.slerp(b, Real::HALF).abs_diff_eq(uct((-ONE_RAW, 0)), 32));
}

// --- renormalization

#[test]
fn test_renormalize_is_exact_on_a_unit_pair() {
    assert!(quarter().renormalize() == quarter());
    assert!(id().renormalize() == id());
}

#[test]
fn test_renormalize_recovers_from_a_large_drift() {
    // A unit pair scaled by 4 (exact: a power of two): renormalize divides it back, which
    // `renormalize_fast` cannot do (`|c|² - 1 = 15` is far outside its convergence radius).
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let four = Real::<Fixed>::from_int(4);
    let drifted = UnitComplex { re: c.re * four, im: c.im * four };
    let n = drifted.renormalize();
    assert!(ulp_diff(Real::norm_squared2(n.re, n.im), Real::ONE) <= 2);
    assert!(n.abs_diff_eq(c, 2));
}

#[test]
fn test_renormalize_fast_converges_quadratically() {
    // A pair about 2^-20 off the unit circle: `|c|² = 1 + e` with `e ~ 2.7e-6`, and one Newton
    // step leaves `3e²/4 ~ 5e-12`, i.e. 0 ulp. The step costs no division and no square root.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let drifted = UnitComplex { re: c.re + fx(0x1000), im: c.im + fx(0x1000) };
    assert!(ulp_diff(Real::norm_squared2(drifted.re, drifted.im), Real::ONE) > 1000);
    let n = drifted.renormalize_fast();
    assert!(ulp_diff(Real::norm_squared2(n.re, n.im), Real::ONE) <= 2);
    // And the exact renormalization agrees to a few ulp.
    assert!(n.abs_diff_eq(drifted.renormalize(), 4));
}

#[test]
fn test_renormalize_fast_leaves_the_square_of_a_large_drift() {
    // 1e-4 off the unit circle: `e ~ 2.8e-4`, one step leaves `3e²/4 ~ 6e-8` = 250 ulp on the
    // squared norm, where `renormalize` would land within 2. The error is squared at every step,
    // so a second step would fix it; `renormalize` is the right tool above 2^-16.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let drifted = UnitComplex { re: c.re + fx(0x68db8), im: c.im + fx(0x68db8) };
    let once = drifted.renormalize_fast();
    let e = ulp_diff(Real::norm_squared2(once.re, once.im), Real::ONE);
    assert!(e > 2 && e < 400);
    let twice = once.renormalize_fast();
    assert!(ulp_diff(Real::norm_squared2(twice.re, twice.im), Real::ONE) <= 2);
    let exact = drifted.renormalize();
    assert!(ulp_diff(Real::norm_squared2(exact.re, exact.im), Real::ONE) <= 2);
}

#[test]
fn test_renormalize_fast_of_zero_stays_zero() {
    let z = UnitComplex { re: Real::<Fixed>::ZERO, im: Real::ZERO };
    assert!(z.renormalize_fast() == z);
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_renormalize_of_zero_panics() {
    let z = UnitComplex { re: Real::<Fixed>::ZERO, im: Real::ZERO };
    let _ = black_box(z).renormalize();
}

// --- linearized composition

#[test]
fn test_append_axisangle_linearized_small_angle() {
    // 0.01 rad appended to a rotation: the linearization rotates by atan(θ) instead of θ, i.e.
    // 3.3e-7 rad short, which is 1500 ulp on the components.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666));
    let step = fx(0x28f5c29); // 0.01
    let got = c.append_axisangle_linearized(step);
    let expected = c * UnitComplexAngleTrait::<Fixed>::new(step);
    assert!(got.abs_diff_eq(expected, 1500));
    // Whatever the error on the angle, the result is a unit complex number again.
    assert!(ulp_diff(Real::norm_squared2(got.re, got.im), Real::ONE) <= 2);
}

#[test]
fn test_append_axisangle_linearized_zero_angle() {
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666));
    assert!(c.append_axisangle_linearized(Real::ZERO).abs_diff_eq(c, 2));
}

#[test]
fn test_append_axisangle_linearized_accumulates() {
    // 64 steps of 1/64 rad: the norm stays unit (each step renormalizes), while the angle falls
    // short of 1 rad by 64 · θ³/3 = 8.1e-5, i.e. about 350 000 ulp on the components. That is
    // the systematic error of the linearization, not drift.
    let mut c = id();
    let step = fx(0x4000000); // 1/64
    for _ in 0_u8..64 {
        c = c.append_axisangle_linearized(step);
    }
    assert!(ulp_diff(Real::norm_squared2(c.re, c.im), Real::ONE) <= 2);
    let exact = UnitComplexAngleTrait::<Fixed>::new(Real::ONE);
    assert!(c.abs_diff_eq(exact, 400000));
    assert!(!c.abs_diff_eq(exact, 100000));
}

// --- approximate equality

#[test]
fn test_abs_diff_eq() {
    let c = uct((0x80000000, -0xdd6a9c1));
    assert!(c.abs_diff_eq(c, 0));
    assert!(c.abs_diff_eq(UnitComplex { re: c.re + fx(3), im: c.im - fx(3) }, 3));
    assert!(!c.abs_diff_eq(UnitComplex { re: c.re + fx(4), im: c.im }, 3));
    // The opposite pair is the same rotation turned by 2π, and is NOT `abs_diff_eq` to it.
    assert!(!id().abs_diff_eq(UnitComplex { re: -Real::<Fixed>::ONE, im: Real::ZERO }, 1000));
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_new_oracle() {
    let mut cases = oracle::unit_complex_new_cases();
    assert!(cases.len() >= 7);
    while let Some(case) = cases.pop_front() {
        let (angle, expected, tol) = *case;
        let got = UnitComplexAngleTrait::<Fixed>::new(fx(angle));
        assert!(got.abs_diff_eq(uct(expected), tol));
    }
}

#[test]
fn test_angle_oracle() {
    let mut cases = oracle::unit_complex_angle_cases();
    assert!(cases.len() >= 9);
    while let Some(case) = cases.pop_front() {
        let (c, expected, tol) = *case;
        assert!(Real::abs_diff_eq(uct(c).angle(), fx(expected), tol));
    }
}

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::unit_complex_mul_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!((uct(a) * uct(b)).abs_diff_eq(uct(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::unit_complex_inverse_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (c, expected, tol) = *case;
        assert!(uct(c).inverse().abs_diff_eq(uct(expected), tol));
        assert!(uct(c).conjugate().abs_diff_eq(uct(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::unit_complex_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (c, v, expected, tol) = *case;
        assert!(uct(c).transform_vector(v2t(v)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_vector_oracle() {
    let mut cases = oracle::unit_complex_inverse_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (c, v, expected, tol) = *case;
        assert!(uct(c).inverse_transform_vector(v2t(v)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_to_rotation_matrix_oracle() {
    let mut cases = oracle::unit_complex_to_rotation_matrix_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (c, expected, tol) = *case;
        assert!(uct(c).to_rotation_matrix().abs_diff_eq(r2(expected), tol));
    }
}

#[test]
fn test_from_rotation_matrix_oracle() {
    let mut cases = oracle::unit_complex_from_rotation_matrix_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (r, expected, tol) = *case;
        assert!(UnitComplexTrait::from_rotation_matrix(r2(r)).abs_diff_eq(uct(expected), tol));
    }
}

#[test]
fn test_rotation_between_oracle() {
    let mut cases = oracle::unit_complex_rotation_between_cases();
    assert!(cases.len() >= 18);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(UnitComplexTrait::rotation_between(v2t(a), v2t(b)).abs_diff_eq(uct(expected), tol));
    }
}

#[test]
fn test_angle_to_oracle() {
    let mut cases = oracle::unit_complex_angle_to_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!(Real::abs_diff_eq(uct(a).angle_to(uct(b)), fx(expected), tol));
    }
}

#[test]
fn test_slerp_oracle() {
    let mut cases = oracle::unit_complex_slerp_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (a, b, t, expected, tol) = *case;
        assert!(uct(a).slerp(uct(b), fx(t)).abs_diff_eq(uct(expected), tol));
    }
}
