//! Unit tests of the WP 8.4-P09a completion of `Rotation2`: the oracle vectors of `tools/oracle`
//! (suite `rotation_matrix_completion`), exact cases, identities with `UnitComplex` and the
//! panics.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::vector2::Vector2;
use nalgebra::geometry::isometry2::Isometry2;
use nalgebra::geometry::rotation2::{Rotation2, Rotation2AngleTrait, Rotation2Trait};
use nalgebra::geometry::similarity2::Similarity2;
use nalgebra::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use nalgebra_tests_utils::{
    ONE_RAW, excess, fx, m2, max_ulp_diff2, max_ulp_diff_uc, r2, u2, uct, v2i,
};
use simba::scalar::Real;
use crate::oracle;

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

fn rot_err(a: Rotation2<Fixed>, b: Rotation2<Fixed>) -> u128 {
    max_ulp_diff2(a.matrix, b.matrix)
}

/// The rotation of 0.75 rad.
fn a() -> Rotation2<Fixed> {
    Rotation2AngleTrait::new(fx(0xc0000000))
}

/// The rotation of -2 rad.
fn b() -> Rotation2<Fixed> {
    Rotation2AngleTrait::new(fx(-0x200000000))
}

// --- oracle vectors

#[test]
fn test_from_matrix_oracle() {
    let mut cases = oracle::rotation2_from_matrix_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let got = Rotation2Trait::from_matrix(m2(m));
        worst = core::cmp::max(worst, report("from_matrix", n, rot_err(got, r2(e)), tol));
        // Bit for bit the unit complex closed form.
        assert!(got == UnitComplexAngleTrait::from_matrix(m2(m)).to_rotation_matrix());
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_slerp_oracle() {
    let mut cases = oracle::rotation2_slerp_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (x, y, t, e, tol) = *case;
        let got = r2(x).slerp(r2(y), fx(t));
        worst = core::cmp::max(worst, report("slerp", n, rot_err(got, r2(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

/// `rotation_to`, `/` and the products with unit complex numbers floor the exact results once.
#[test]
fn test_exact_ops_oracle() {
    let mut cases = oracle::rotation2_rotation_to_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, _) = *case;
        assert!(r2(x).rotation_to(r2(y)) == r2(e));
        assert!(r2(y) / r2(x) == r2(e));
    }
    let mut cases = oracle::rotation2_div_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, _) = *case;
        assert!(r2(x) / r2(y) == r2(e));
    }
    let mut cases = oracle::rotation2_mul_unit_complex_cases();
    while let Some(case) = cases.pop_front() {
        let (r, c, e, _) = *case;
        assert!(r2(r).mul_unit_complex(uct(c)) == uct(e));
    }
    let mut cases = oracle::rotation2_div_unit_complex_cases();
    while let Some(case) = cases.pop_front() {
        let (r, c, e, _) = *case;
        assert!(r2(r).div_unit_complex(uct(c)) == uct(e));
    }
}

// --- construction

#[test]
fn test_from_matrix_eps() {
    let m = m2([[ONE_RAW * 3, ONE_RAW], [-ONE_RAW, ONE_RAW * 2]]);
    let id = Rotation2Trait::identity();
    let closed = Rotation2Trait::from_matrix(m);
    assert!(Rotation2AngleTrait::from_matrix_eps(m, Real::default_epsilon(), 0, id) == closed);
    // The bounded iteration from the identity converges to the closed form.
    let it = Rotation2AngleTrait::from_matrix_eps(m, Real::default_epsilon(), 32, id);
    assert!(rot_err(it, closed) <= 8);
    // Exact on an exact rotation.
    let quarter = r2([[0, -ONE_RAW], [ONE_RAW, 0]]);
    assert!(Rotation2Trait::from_matrix(quarter.matrix) == quarter);
}

#[test]
fn test_basis_and_parts() {
    let r = a();
    let m = r.matrix;
    let basis = [Vector2 { x: m.m11, y: m.m21 }, Vector2 { x: m.m12, y: m.m22 }];
    assert!(Rotation2Trait::from_basis_unchecked(basis) == r);
    assert!(r.unwrap() == m);
}

#[test]
fn test_scaled_axis_round_trip() {
    let r: Rotation2<Fixed> = Rotation2AngleTrait::from_scaled_axis(Matrix1 { x: fx(0xc0000000) });
    assert!(r == a());
    assert!(a().scaled_axis() == Matrix1 { x: a().angle() });
}

// --- interpolation, composition

#[test]
fn test_slerp_cases() {
    assert!(a().slerp(b(), Real::zero()) == a());
    assert!(rot_err(a().slerp(b(), Real::one()), b()) <= 4);
    // The unit complex slerp, expanded.
    let c = UnitComplexAngleTrait::slerp(a().into(), b().into(), Real::HALF);
    assert!(a().slerp(b(), Real::HALF) == c.to_rotation_matrix());
}

#[test]
fn test_rotation_to_and_div() {
    assert!(rot_err(a().rotation_to(b()) * a(), b()) <= 2);
    assert!(b() / a() == a().rotation_to(b()));
    assert!(rot_err(a() / a(), Rotation2Trait::identity()) <= 1);
}

#[test]
fn test_unit_complex_products() {
    let c: UnitComplex<Fixed> = b().into();
    let r = a();
    let ra: UnitComplex<Fixed> = r.into();
    assert!(r.mul_unit_complex(c) == ra * c);
    assert!(r.div_unit_complex(c) == ra / c);
    assert!(max_ulp_diff_uc(r.mul_unit_complex(c) / c, ra) <= 2);
}

#[test]
fn test_unit_vector_transforms() {
    let u = u2(0, ONE_RAW);
    assert!(a().transform_unit_vector(u).value == a().transform_vector(u.value));
    assert!(a().inverse_transform_unit_vector(u).value == a().inverse_transform_vector(u.value));
}

// --- comparisons, casts, trait impls

#[test]
fn test_relative_ulps_eq() {
    let r = a();
    let m = r.matrix;
    let drifted = Rotation2 { matrix: Matrix2 { m12: m.m12 - fx(5), ..m } };
    assert!(r.relative_eq(r, 0, Real::zero()) && r.ulps_eq(r, 0, 0));
    assert!(!r.relative_eq(drifted, 4, Real::zero()) && r.relative_eq(drifted, 5, Real::zero()));
    assert!(!r.ulps_eq(drifted, 4, 4) && r.ulps_eq(drifted, 0, 5));
}

#[test]
fn test_trait_impls() {
    let id: Rotation2<Fixed> = Rotation2Trait::identity();
    assert!(Default::default() == id);
    assert!(One::one() == id && id.is_one() && a().is_non_one());
    assert!(a().cast::<Fixed>() == a());
    let mut r = a();
    let m = r.matrix;
    assert!(r[(0, 0)] == m.m11 && r[(0, 1)] == m.m12 && r[(1, 0)] == m.m21 && r[(1, 1)] == m.m22);
    let iso: Isometry2<Fixed> = a().into();
    let c: UnitComplex<Fixed> = a().into();
    assert!(iso.rotation == c && iso.translation.vector == v2i(0, 0));
    let sim: Similarity2<Fixed> = a().into();
    assert!(sim.isometry == iso && sim.scaling == Real::one());
    assert!(UnitComplexTrait::from_rotation_matrix(a()) == c);
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_index_out_of_bounds_panics() {
    let mut r = a();
    let _ = r[(2, 0)];
}
