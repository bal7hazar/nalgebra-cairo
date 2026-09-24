//! Unit tests of `Rotation2`: exact cases (identity, quarter turns, axes), the identities a
//! rotation matrix must satisfy (`R · Rᵀ = I`, `det R = 1`, `R → c → R`), panics on
//! degenerate inputs, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same
//! raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution,
//! the `rotation2_*` ops of the `rotation2` suite) with
//! `cargo run --release -- emit-cairo rotation2 --from vectors --max-per-dist 6 --out
//! <oracle.cairo> --ops rotation2_new,rotation2_angle,rotation2_mul,rotation2_inverse,
//! rotation2_transform_vector`.
//!
//! Moved from `crates/nalgebra/src/geometry/rotation2/tests.cairo` (WP 8.1c, test-only package):
//! the tests of crate-internal items stay there.

use fixed::Fixed;
use nalgebra::base::matrix2::{Matrix2, Matrix2Trait};
use nalgebra::base::vector2::{Vector2, Vector2Trait};
use nalgebra::geometry::rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
use nalgebra::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, fx, m2, p2t, r2, uct, ulp_diff, v2t};
use simba::scalar::Real;
use crate::rotation2::oracle;

/// The raw value of 1.
fn id() -> Rotation2<Fixed> {
    Rotation2Trait::<Fixed>::identity()
}

/// The quarter turn `[[0, -1], [1, 0]]`, an exactly representable rotation.
fn quarter() -> Rotation2<Fixed> {
    r2([[0, -ONE_RAW], [ONE_RAW, 0]])
}

// --- constructors and accessors

#[test]
fn test_identity_is_exact() {
    let r = id();
    assert!(r.matrix == Matrix2Trait::<Fixed>::identity());
    assert!(r.angle() == Real::zero());
    assert!(r.matrix.determinant() == Real::one());
}

#[test]
fn test_from_matrix_unchecked_and_accessors() {
    let m = m2([[0x80000000, -0xdd6a9c1], [0xdd6a9c1, 0x80000000]]);
    let r = Rotation2Trait::from_matrix_unchecked(m);
    assert!(r.matrix() == m);
    assert!(r.into_inner() == m);
    assert!(r.matrix.m11 == fx(0x80000000) && r.matrix.m21 == fx(0xdd6a9c1));
}

#[test]
fn test_new_cardinal_angles() {
    assert!(Rotation2AngleTrait::<Fixed>::new(Real::zero()) == id());
    let q = Rotation2AngleTrait::<Fixed>::new(Real::frac_pi_2());
    assert!(q.abs_diff_eq(quarter(), 1));
    assert!(Rotation2AngleTrait::<Fixed>::new(Real::pi()).abs_diff_eq(quarter() * quarter(), 2));
}

#[test]
fn test_new_matches_unit_complex_new() {
    let a = fx(0x1f0a3d70a);
    let r = Rotation2AngleTrait::<Fixed>::new(a);
    assert!(r == UnitComplexAngleTrait::<Fixed>::new(a).to_rotation_matrix());
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_from_matrix_of_a_zero_first_column_panics() {
    let m = Matrix2 {
        m11: Real::<Fixed>::zero(), m21: Real::zero(), m12: Real::one(), m22: Real::one(),
    };
    let _ = Rotation2Trait::from_matrix(black_box(m));
}

// --- inverse, composition

#[test]
fn test_inverse_is_the_transpose_and_exact() {
    let r = r2([[0x80000000, -0xdd6a9c1], [0xdd6a9c1, 0x80000000]]);
    assert!(r.inverse() == r.transpose());
    assert!(r.inverse().matrix == r.matrix.transpose());
    assert!(r.inverse().inverse() == r);
    assert!(quarter().inverse() == r2([[0, ONE_RAW], [-ONE_RAW, 0]]));
}

#[test]
fn test_mul_identity_is_neutral_and_exact() {
    let r = r2([[0x80000000, -0xdd6a9c1], [0xdd6a9c1, 0x80000000]]);
    assert!(r * id() == r);
    assert!(id() * r == r);
}

#[test]
fn test_mul_quarter_turns_are_exact() {
    let q = quarter();
    assert!(q * q == r2([[-ONE_RAW, 0], [0, -ONE_RAW]]));
    assert!(q * q * q * q == id());
}

#[test]
fn test_mul_inverse_is_identity_within_two_ulp() {
    // R · Rᵀ = I: the off-diagonal terms cancel exactly, the diagonal carries the norm drift.
    let r = Rotation2AngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let p = r * r.inverse();
    assert!(p.matrix.m12 == Real::zero() && p.matrix.m21 == Real::zero());
    assert!(ulp_diff(p.matrix.m11, Real::one()) <= 2 && ulp_diff(p.matrix.m22, Real::one()) <= 2);
    assert!(p.abs_diff_eq(id(), 2));
}

#[test]
fn test_determinant_is_one() {
    let r = Rotation2AngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    assert!(ulp_diff(r.matrix.determinant(), Real::one()) <= 2);
}

#[test]
fn test_mul_matches_the_unit_complex_product_up_to_one_ulp() {
    // The matrix product floors `-(sin)` where the complex product negates `floor(sin)`: the two
    // agree on `m11` / `m22` bit for bit and differ by at most 1 ulp on `m12`
    // (`bench_rotation2_mul__alt_complex`).
    let (a, b) = (
        Rotation2AngleTrait::<Fixed>::new(fx(0x66666666)),
        Rotation2AngleTrait::<Fixed>::new(fx(-0x2aaaaaaa)),
    );
    let m = a * b;
    let c = (UnitComplexTrait::from_rotation_matrix(a) * UnitComplexTrait::from_rotation_matrix(b))
        .to_rotation_matrix();
    assert!(m.matrix.m11 == c.matrix.m11 && m.matrix.m22 == c.matrix.m22);
    assert!(m.matrix.m21 == c.matrix.m21);
    assert!(ulp_diff(m.matrix.m12, c.matrix.m12) <= 1);
}

// --- transforms

#[test]
fn test_transform_vector_quarter_turn_is_exact() {
    let v = v2t((0x300000000, -0x400000000));
    assert!(quarter().transform_vector(v) == v2t((0x400000000, 0x300000000)));
    assert!(id().transform_vector(v) == v);
}

#[test]
fn test_transform_matches_the_unit_complex_bit_for_bit() {
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let r = c.to_rotation_matrix();
    let v = v2t((0x300000000, -0x400000000));
    assert!(r.transform_vector(v) == c.transform_vector(v));
    assert!(r.inverse_transform_vector(v) == c.inverse_transform_vector(v));
    let p = p2t((0x300000000, -0x400000000));
    assert!(r.transform_point(p) == c.transform_point(p));
    assert!(r.inverse_transform_point(p) == c.inverse_transform_point(p));
}

#[test]
fn test_inverse_transform_is_the_transposed_product() {
    let r = Rotation2AngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let v = v2t((0x300000000, -0x400000000));
    assert!(r.inverse_transform_vector(v) == r.inverse().transform_vector(v));
    assert!(r.inverse_transform_vector(r.transform_vector(v)).abs_diff_eq(v, 4));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_transform_vector_overflow_panics() {
    let r = black_box(Rotation2AngleTrait::<Fixed>::new(-Real::<Fixed>::frac_pi_4()));
    let _ = r.transform_vector(black_box(v2t((0x6000000000000000, 0x6000000000000000))));
}

// --- conversions

#[test]
fn test_unit_complex_round_trip_is_exact() {
    let c = uct((0x80000000, -0xdd6a9c1));
    let r: Rotation2<Fixed> = c.into();
    assert!(r == c.to_rotation_matrix());
    assert!(UnitComplexTrait::from_rotation_matrix(r) == c);
    let back: UnitComplex<Fixed> = r.into();
    assert!(back == c);
    assert!(UnitComplexTrait::from_rotation_matrix(r) == c);
}

#[test]
fn test_to_homogeneous() {
    let r = r2([[0x80000000, -0xdd6a9c1], [0xdd6a9c1, 0x80000000]]);
    let h = r.to_homogeneous();
    assert!(h.m11 == r.matrix.m11 && h.m12 == r.matrix.m12);
    assert!(h.m21 == r.matrix.m21 && h.m22 == r.matrix.m22);
    assert!(h.m13 == Real::zero() && h.m23 == Real::zero());
    assert!(h.m31 == Real::zero() && h.m32 == Real::zero() && h.m33 == Real::one());
    assert!(h == UnitComplexTrait::from_rotation_matrix(r).to_homogeneous());
}

// --- rotation_between, angles

#[test]
fn test_rotation_between_axes_is_exact() {
    let (x, y) = (v2t((0x300000000, 0)), v2t((0, 0x500000000)));
    assert!(Rotation2Trait::rotation_between(x, y) == quarter());
    assert!(Rotation2Trait::rotation_between(x, x) == id());
    let z = Vector2 { x: Real::<Fixed>::zero(), y: Real::zero() };
    assert!(Rotation2Trait::rotation_between(z, x) == id());
    // Same rotation as the unit complex form, expanded.
    let (a, b) = (v2t((0x30ec4a100, -0x41d5b9200)), v2t((0x40a3d7000, 0x2f1a9fc00)));
    assert!(
        Rotation2Trait::rotation_between(a, b) == UnitComplexTrait::rotation_between(a, b)
            .to_rotation_matrix(),
    );
}

#[test]
fn test_scaled_rotation_between() {
    let (x, y) = (v2t((0x300000000, 0)), v2t((0, 0x500000000)));
    let r = Rotation2AngleTrait::scaled_rotation_between(x, y, Real::HALF);
    assert!(r.abs_diff_eq(Rotation2AngleTrait::<Fixed>::new(Real::frac_pi_4()), 4));
    let z = Vector2 { x: Real::<Fixed>::zero(), y: Real::zero() };
    assert!(Rotation2AngleTrait::scaled_rotation_between(z, z, Real::one()) == id());
}

#[test]
fn test_angle_round_trip_and_angle_to() {
    let a = fx(0x1f0a3d70a);
    assert!(ulp_diff(Rotation2AngleTrait::<Fixed>::new(a).angle(), a) <= 12);
    assert!(quarter().angle() == Real::frac_pi_2());
    let (x, y) = (
        Rotation2AngleTrait::<Fixed>::new(fx(0x66666666)),
        Rotation2AngleTrait::<Fixed>::new(fx(0x1999999a)),
    );
    assert!(
        ulp_diff(
            x.angle_to(y),
            UnitComplexTrait::from_rotation_matrix(x)
                .angle_to(UnitComplexTrait::from_rotation_matrix(y)),
        ) == 0,
    );
    assert!(ulp_diff(x.angle_to(y), fx(0x1999999a) - fx(0x66666666)) <= 16);
}

#[test]
fn test_powf() {
    let r = Rotation2AngleTrait::<Fixed>::new(fx(0x66666666));
    assert!(r.powf(Real::zero()) == id());
    assert!(r.powf(Real::one()).abs_diff_eq(r, 16));
    assert!(r.powf(Real::TWO).abs_diff_eq(r * r, 32));
    assert!(r.powf(Real::NEG_ONE).abs_diff_eq(r.inverse(), 16));
}

// --- approximate equality

#[test]
fn test_abs_diff_eq() {
    let r = r2([[0x80000000, -0xdd6a9c1], [0xdd6a9c1, 0x80000000]]);
    assert!(r.abs_diff_eq(r, 0));
    assert!(r.abs_diff_eq(r2([[0x80000003, -0xdd6a9c1], [0xdd6a9c1, 0x80000000]]), 3));
    assert!(!r.abs_diff_eq(r2([[0x80000000, -0xdd6a9c1], [0xdd6a9c5, 0x80000000]]), 3));
}

// --- oracle vectors (upstream nalgebra on the same raw inputs; `tol` in ulp, 0 = bit for bit)

#[test]
fn test_new_oracle() {
    let mut cases = oracle::rotation2_new_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (angle, expected, tol) = *case;
        assert!(Rotation2AngleTrait::<Fixed>::new(fx(angle)).abs_diff_eq(r2(expected), tol));
    }
}

#[test]
fn test_angle_oracle() {
    let mut cases = oracle::rotation2_angle_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (r, expected, tol) = *case;
        assert!(Real::abs_diff_eq(r2(r).angle(), fx(expected), tol));
    }
}

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::rotation2_mul_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (a, b, expected, tol) = *case;
        assert!((r2(a) * r2(b)).abs_diff_eq(r2(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::rotation2_inverse_cases();
    assert!(cases.len() >= 6);
    while let Some(case) = cases.pop_front() {
        let (r, expected, tol) = *case;
        assert!(r2(r).inverse().abs_diff_eq(r2(expected), tol));
        assert!(r2(r).transpose().abs_diff_eq(r2(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::rotation2_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (r, v, expected, tol) = *case;
        assert!(r2(r).transform_vector(v2t(v)).abs_diff_eq(v2t(expected), tol));
    }
}
