//! Unit tests of `Similarity2`: exact cases, identities, zero-scale panics, append/prepend
//! semantics, homogeneous layout, and oracle vectors from upstream nalgebra 0.35.

use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::point2::{Point2, Point2Trait};
use crate::base::vector2::{Vector2, Vector2Trait};
use crate::geometry::isometry2::{Isometry2, Isometry2Trait};
use crate::geometry::translation2::{Translation2, Translation2Trait};
use crate::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use super::{Similarity2, Similarity2AngleTrait, Similarity2Trait, oracle};

const ONE_RAW: i64 = 0x100000000;

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn v2(t: (i64, i64)) -> Vector2<Fixed> {
    let (x, y) = t;
    Vector2 { x: fx(x), y: fx(y) }
}

fn p2(t: (i64, i64)) -> Point2<Fixed> {
    let (x, y) = t;
    Point2 { x: fx(x), y: fx(y) }
}

fn uc(t: (i64, i64)) -> UnitComplex<Fixed> {
    let (re, im) = t;
    UnitComplex { re: fx(re), im: fx(im) }
}

fn m3(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = rows;
    Matrix3 {
        m11: fx(m11),
        m21: fx(m21),
        m31: fx(m31),
        m12: fx(m12),
        m22: fx(m22),
        m32: fx(m32),
        m13: fx(m13),
        m23: fx(m23),
        m33: fx(m33),
    }
}

fn sim(t: ((i64, i64), (i64, i64), i64)) -> Similarity2<Fixed> {
    let (tr, rot, scaling) = t;
    Similarity2 {
        isometry: Isometry2 { rotation: uc(rot), translation: Translation2 { vector: v2(tr) } },
        scaling: fx(scaling),
    }
}

fn id() -> Similarity2<Fixed> {
    Similarity2Trait::<Fixed>::identity()
}

fn a() -> Similarity2<Fixed> {
    Similarity2AngleTrait::new(v2((0x180000000, -0x240000000)), fx(0x66666666), fx(0x280000000))
}

fn b() -> Similarity2<Fixed> {
    Similarity2AngleTrait::new(v2((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa), fx(0x180000000))
}

fn quarter() -> UnitComplex<Fixed> {
    uc((0, ONE_RAW))
}

#[test]
fn test_identity_and_constructors_are_exact() {
    let i = id();
    assert!(i.isometry == Isometry2Trait::<Fixed>::identity());
    assert!(i.scaling == Real::ONE);
    assert!(i.transform_point(p2((0x123, -0x456))) == p2((0x123, -0x456)));

    let t = Translation2Trait::new(fx(0x180000000), fx(-0x240000000));
    let r = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666));
    let s = fx(0x280000000);
    let from_parts = Similarity2Trait::from_parts(t, r, s);
    let from_iso = Similarity2Trait::from_isometry(Isometry2Trait::from_parts(t, r), s);
    assert!(from_parts == a() && from_iso == a());
    assert!(Similarity2Trait::from_scaling(s).scaling() == s);
    assert!(from_parts.with_scaling(fx(0x180000000)).scaling() == fx(0x180000000));
    let pure: Similarity2<Fixed> = Isometry2Trait::from_parts(t, r).into();
    assert!(pure.scaling == Real::ONE && pure.isometry == from_parts.isometry);
}

#[test]
fn test_pure_scaling_and_quarter_turn_are_exact() {
    let s = Similarity2Trait::<Fixed>::from_scaling(fx(0x300000000));
    assert!(s.transform_point(p2((ONE_RAW, -2 * ONE_RAW))) == p2((3 * ONE_RAW, -6 * ONE_RAW)));
    assert!(s.transform_vector(v2((ONE_RAW, -2 * ONE_RAW))) == v2((3 * ONE_RAW, -6 * ONE_RAW)));

    let q = Similarity2Trait::from_parts(
        Translation2Trait::new(fx(ONE_RAW), Real::ZERO), quarter(), fx(0x200000000),
    );
    assert!(q.transform_point(p2((ONE_RAW, 0))) == p2((ONE_RAW, 2 * ONE_RAW)));
    assert!(q.transform_vector(v2((0, ONE_RAW))) == v2((-2 * ONE_RAW, 0)));
}

#[test]
fn test_inverse_composes_to_identity_and_undoes_points() {
    let x = a();
    assert!((x * x.inverse()).abs_diff_eq(id(), 8));
    assert!((x.inverse() * x).abs_diff_eq(id(), 8));
    let p = p2((-0x280000000, 0x3c0000000));
    assert!(x.inverse_transform_point(x.transform_point(p)).abs_diff_eq(p, 16));
    let v = v2((-0x280000000, 0x3c0000000));
    assert!(x.inverse_transform_vector(x.transform_vector(v)).abs_diff_eq(v, 16));
}

#[test]
fn test_mul_and_inv_mul_match_actions() {
    let (x, y, p) = (a(), b(), p2((-0x280000000, 0x3c0000000)));
    assert!((x * y).transform_point(p).abs_diff_eq(x.transform_point(y.transform_point(p)), 16));
    assert!(x.inv_mul(y).abs_diff_eq(x.inverse() * y, 16));
    assert!(x.inv_mul(x).abs_diff_eq(id(), 8));
}

#[test]
fn test_scaling_append_and_prepend_match_upstream_order() {
    let x = a();
    let s = fx(0x180000000);
    assert!(
        x
            .prepend_scaling(s)
            .transform_point(p2((ONE_RAW, 0)))
            .abs_diff_eq(x.transform_point(p2((0x180000000, 0))), 4),
    );
    let p = p2((ONE_RAW, 0));
    assert!(x.append_scaling(s).transform_point(p).abs_diff_eq(x.transform_point(p).scale(s), 1));
}

#[test]
fn test_append_and_prepend_translation_rotation_match_composition() {
    let (x, t, r) = (a(), Translation2Trait::new(fx(0x140000000), fx(-0x60000000)), quarter());
    let ti: Similarity2<Fixed> = Similarity2Trait::from_isometry(t.into(), Real::ONE);
    let ri: Similarity2<Fixed> = Similarity2Trait::from_isometry(r.into(), Real::ONE);
    assert!(x.append_translation(t) == ti * x);
    assert!(x.prepend_translation(t).abs_diff_eq(x * ti, 1));
    assert!(x.append_rotation(r) == ri * x);
    assert!(x.prepend_rotation(r) == x * ri);
}

#[test]
fn test_to_homogeneous_layout_and_action() {
    let x = a();
    let m = x.to_homogeneous();
    assert!(m.m11 == x.isometry.rotation.re * x.scaling);
    assert!(m.m12 == -x.isometry.rotation.im * x.scaling);
    assert!(m.m13 == x.isometry.translation.vector.x);
    assert!(m.m23 == x.isometry.translation.vector.y);
    assert!(id().to_homogeneous() == Matrix3Trait::<Fixed>::identity());
    let p = p2((-0x280000000, 0x3c0000000));
    let h = m.mul_vec(p.to_homogeneous());
    assert!(Point2 { x: h.x, y: h.y }.abs_diff_eq(x.transform_point(p), 4));
}

#[test]
fn test_append_rotation_wrt_point_and_center() {
    let (x, r, p) = (a(), quarter(), p2((0x180000000, -0x80000000)));
    let y = x.append_rotation_wrt_point(r, p);
    let shift: Similarity2<Fixed> = Similarity2Trait::from_isometry(
        Translation2Trait::new(p.x, p.y).into(), Real::ONE,
    );
    let back: Similarity2<Fixed> = Similarity2Trait::from_isometry(
        Translation2Trait::new(-p.x, -p.y).into(), Real::ONE,
    );
    let ri: Similarity2<Fixed> = Similarity2Trait::from_isometry(r.into(), Real::ONE);
    assert!(y.abs_diff_eq(shift * ri * back * x, 4));
    assert!(x.append_rotation_wrt_center(r).isometry.translation == x.isometry.translation);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_parts_zero_scaling_panics() {
    Similarity2Trait::<
        Fixed,
    >::from_parts(Translation2Trait::identity(), UnitComplexTrait::identity(), Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_isometry_zero_scaling_panics() {
    Similarity2Trait::<Fixed>::from_isometry(Isometry2Trait::identity(), Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_scaling_zero_panics() {
    Similarity2Trait::<Fixed>::from_scaling(Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_with_scaling_zero_panics() {
    a().with_scaling(Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_prepend_scaling_zero_panics() {
    a().prepend_scaling(Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_append_scaling_zero_panics() {
    a().append_scaling(Real::ZERO);
}

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::similarity2_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!((sim(x) * sim(y)).abs_diff_eq(sim(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::similarity2_inverse_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(sim(x).inverse().abs_diff_eq(sim(expected), tol));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::similarity2_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(sim(x).transform_point(p2(p)).abs_diff_eq(p2(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::similarity2_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(sim(x).transform_vector(v2(v)).abs_diff_eq(v2(expected), tol));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::similarity2_inverse_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(sim(x).inverse_transform_point(p2(p)).abs_diff_eq(p2(expected), tol));
    }
}
