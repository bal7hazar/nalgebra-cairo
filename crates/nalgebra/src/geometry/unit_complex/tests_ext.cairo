//! Unit tests of the WP 8.4-P08 completion of `UnitComplex`: the division, the heterogeneous
//! operators (rotations, translations, isometries, similarities), `from_complex`,
//! `rotation_between_axis`, `from_matrix_eps`, the `Vector1` forms (`from_scaled_axis`,
//! `scaled_axis`, `axis_angle`) and the trait impls: exact cases, identities, the iteration
//! measurement and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs, tolerance in ulp).
//!
//! `oracle_ext.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo unit_complex_completion --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/unit_complex/oracle_ext.cairo
//! ```

use core::num::traits::One;
use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{
    ONE_RAW, excess, fx, int, m2, max_ulp_diff_uc, max_ulp_diff_v2, r2, uct, ulp_diff, v2i, v2t,
};
use crate::base::unit::Unit;
use crate::base::vector2::Vector2;
use crate::geometry::isometry2::Isometry2;
use crate::geometry::similarity2::Similarity2;
use crate::geometry::translation2::Translation2;
use super::{
    FROM_MATRIX_MAX_ITER, UnitComplex, UnitComplexAngleInternalTrait, UnitComplexAngleTrait,
    UnitComplexTrait, oracle_ext,
};

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

fn iso(t: ((i64, i64), (i64, i64))) -> Isometry2<Fixed> {
    let (v, r) = t;
    Isometry2 { rotation: uct(r), translation: Translation2 { vector: v2t(v) } }
}

fn sim(t: ((i64, i64), (i64, i64), i64)) -> Similarity2<Fixed> {
    let (v, r, s) = t;
    Similarity2 { isometry: iso((v, r)), scaling: fx(s) }
}

fn iso_err(a: Isometry2<Fixed>, b: Isometry2<Fixed>) -> u128 {
    core::cmp::max(
        max_ulp_diff_uc(a.rotation, b.rotation),
        max_ulp_diff_v2(a.translation.vector, b.translation.vector),
    )
}

/// The rotation by 2.2 rad.
fn c() -> UnitComplex<Fixed> {
    UnitComplexAngleTrait::new(fx(9448928051))
}

// --- oracle vectors

#[test]
fn test_div_oracle_is_bit_exact() {
    let mut cases = oracle_ext::unit_complex_div_cases();
    while let Some(case) = cases.pop_front() {
        let (a, b, e, _) = *case;
        let got = uct(a) / uct(b);
        assert!(got == uct(e));
        assert!(got == uct(b).rotation_to(uct(a)));
    }
}

#[test]
fn test_mul_rotation_oracle_is_bit_exact() {
    let mut cases = oracle_ext::unit_complex_mul_rotation_cases();
    while let Some(case) = cases.pop_front() {
        let (a, r, e, _) = *case;
        assert!(uct(a).mul_rotation(r2(r)) == uct(e));
    }
}

#[test]
fn test_div_rotation_oracle_is_bit_exact() {
    let mut cases = oracle_ext::unit_complex_div_rotation_cases();
    while let Some(case) = cases.pop_front() {
        let (a, r, e, _) = *case;
        assert!(uct(a).div_rotation(r2(r)) == uct(e));
    }
}

#[test]
fn test_mul_translation_oracle() {
    let mut cases = oracle_ext::unit_complex_mul_translation_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, t, e, tol) = *case;
        let got = uct(a).mul_translation(Translation2 { vector: v2t(t) });
        worst = core::cmp::max(worst, report("mul_translation", n, iso_err(got, iso(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mul_isometry_oracle() {
    let mut cases = oracle_ext::unit_complex_mul_isometry_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, i, e, tol) = *case;
        let got = uct(a).mul_isometry(iso(i));
        worst = core::cmp::max(worst, report("mul_isometry", n, iso_err(got, iso(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mul_similarity_oracle() {
    let mut cases = oracle_ext::unit_complex_mul_similarity_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, s, e, tol) = *case;
        let got = uct(a).mul_similarity(sim(s));
        let expected = sim(e);
        let err = core::cmp::max(
            iso_err(got.isometry, expected.isometry), ulp_diff(got.scaling, expected.scaling),
        );
        worst = core::cmp::max(worst, report("mul_similarity", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_from_complex_oracle() {
    let mut cases = oracle_ext::unit_complex_from_complex_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (q, e, tol) = *case;
        let got = UnitComplexTrait::from_complex(v2t(q));
        worst = core::cmp::max(worst, report("from_complex", n, max_ulp_diff_uc(got, uct(e)), tol));
        let (again, norm) = UnitComplexTrait::from_complex_and_get(v2t(q));
        assert!(again == got);
        let (x, y) = q;
        assert!(norm == Real::norm2(fx(x), fx(y)));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_rotation_between_axis_oracle() {
    let mut cases = oracle_ext::unit_complex_rotation_between_axis_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (a, b, e, tol) = *case;
        let (ua, ub) = (Unit { value: v2t(a) }, Unit { value: v2t(b) });
        let got = UnitComplexTrait::rotation_between_axis(ua, ub);
        worst =
            core::cmp::max(
                worst, report("rotation_between_axis", n, max_ulp_diff_uc(got, uct(e)), tol),
            );
        assert!(
            UnitComplexAngleTrait::scaled_rotation_between_axis(ua, ub, Real::one())
                .abs_diff_eq(got, tol),
        );
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_from_matrix_oracle() {
    let mut cases = oracle_ext::unit_complex_from_matrix_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let got = UnitComplexAngleTrait::from_matrix(m2(m));
        worst = core::cmp::max(worst, report("from_matrix", n, max_ulp_diff_uc(got, uct(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

/// The 2D Müller iteration (`max_iter > 0`) reaches the closed form: from the identity, with
/// `eps` = 32 ulp, 23 of the 24 oracle cases stop in 4 to 12 iterations (the step `tan(φ - θ)`
/// converges cubically once the error is below π/2), and the last one oscillates at the noise
/// floor until the bound, 3 ulp from the closed form.
#[test]
fn test_from_matrix_eps_iterations_on_the_oracle_set() {
    let mut cases = oracle_ext::unit_complex_from_matrix_cases();
    let (mut worst, mut most, mut stopped) = (0, 0, 0);
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let (got, iterations) = UnitComplexAngleInternalTrait::from_matrix_eps_count(
            m2(m), fx(32), 0, UnitComplexTrait::identity(),
        );
        if iterations < FROM_MATRIX_MAX_ITER {
            stopped += 1;
            most = core::cmp::max(most, iterations);
        }
        worst = core::cmp::max(worst, excess(max_ulp_diff_uc(got, uct(e)), tol.into()));
    }
    println!("unit_complex from_matrix_eps: {stopped} of 24 stop, in at most {most} iterations");
    assert!(stopped == 23 && most == 12);
    assert!(worst == 0);
}

#[test]
fn test_from_matrix_exact_cases() {
    // A rotation matrix gives its own rotation; a scaled one too.
    let r = c().to_rotation_matrix().matrix;
    assert!(UnitComplexAngleTrait::from_matrix(r).abs_diff_eq(c(), 2));
    let (re, im) = (c().re, c().im);
    let r3 = crate::base::matrix2::Matrix2 {
        m11: re * int(3), m21: im * int(3), m12: -im * int(3), m22: re * int(3),
    };
    assert!(UnitComplexAngleTrait::from_matrix(r3).abs_diff_eq(c(), 2));
    // The zero matrix has no rotation part: the identity.
    assert!(
        UnitComplexAngleTrait::from_matrix(m2([[0, 0], [0, 0]])) == UnitComplexTrait::identity(),
    );
    // One explicit iteration from the identity is upstream's first step, not the answer.
    let one = UnitComplexAngleTrait::from_matrix_eps(r, Real::default_epsilon(), 1, One::one());
    assert!(!one.abs_diff_eq(c(), 64));
}

// --- the Vector1 forms

#[test]
fn test_scaled_axis_forms() {
    let angle = c().angle();
    assert!(c().scaled_axis() == angle);
    assert!(UnitComplexAngleTrait::from_scaled_axis(angle) == UnitComplexAngleTrait::new(angle));
    assert!(c().axis_angle() == Some((Real::one(), angle)));
    assert!(c().inverse().axis_angle() == Some((Real::NEG_ONE, angle)));
    assert!(UnitComplexTrait::<Fixed>::identity().axis_angle().is_none());
}

// --- constructors, unit vectors, operators

#[test]
fn test_from_basis_unchecked_takes_the_first_column() {
    let basis = [Vector2 { x: c().re, y: c().im }, Vector2 { x: -c().im, y: c().re }];
    assert!(UnitComplexTrait::from_basis_unchecked(basis) == c());
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_from_complex_of_zero_panics() {
    let _ = UnitComplexTrait::from_complex(v2i(0, 0));
}

#[test]
fn test_unit_vector_transforms() {
    let v = Unit { value: v2t((ONE_RAW, 0)) };
    assert!(c().transform_unit_vector(v).value == c().transform_vector(v.value));
    assert!(c().inverse_transform_unit_vector(v).value == c().inverse_transform_vector(v.value));
}

#[test]
fn test_div_undoes_mul() {
    let d = uct((3037000500, 3037000500));
    assert!(((c() * d) / d).abs_diff_eq(c(), 2));
    assert!((c() / c()).abs_diff_eq(UnitComplexTrait::identity(), 1));
    assert!(c().div_rotation(d.to_rotation_matrix()) == c() / d);
    assert!(c().mul_rotation(d.to_rotation_matrix()) == c() * d);
}

#[test]
fn test_heterogeneous_products_compose() {
    let t = Translation2 { vector: v2i(1, -2) };
    let i = c().mul_translation(t);
    assert!(i.rotation == c() && i.translation.vector == c().transform_vector(v2i(1, -2)));
    let j = Isometry2 { rotation: c(), translation: t };
    let k = c().mul_isometry(j);
    assert!(k.rotation == c() * c());
    let s = Similarity2 { isometry: j, scaling: int(3) };
    let p = c().mul_similarity(s);
    assert!(p.isometry == k && p.scaling == int(3));
}

// --- trait impls and comparisons

#[test]
fn test_default_and_one_are_the_identity() {
    let d: UnitComplex<Fixed> = Default::default();
    let o: UnitComplex<Fixed> = One::one();
    assert!(d == UnitComplexTrait::identity() && o == d);
    assert!(o.is_one() && !c().is_one());
}

#[test]
fn test_into_isometry_and_similarity() {
    let i: Isometry2<Fixed> = c().into();
    assert!(i.rotation == c() && i.translation.vector == Vector2 { x: fx(0), y: fx(0) });
    let s: Similarity2<Fixed> = c().into();
    assert!(s.isometry == i && s.scaling == Real::one());
}

#[test]
fn test_cast_relative_and_ulps_eq() {
    let x: UnitComplex<Fixed> = c().cast();
    assert!(x == c());
    let y = UnitComplex { re: c().re + fx(2), im: c().im };
    assert!(c().relative_eq(y, 2, Real::zero()) && !c().relative_eq(y, 1, Real::zero()));
    assert!(c().ulps_eq(y, 0, 2) && !c().ulps_eq(y, 1, 1));
}
