//! Unit tests of `Similarity2`: exact cases, identities, zero-scale panics, append/prepend
//! semantics, homogeneous layout, and oracle vectors from upstream nalgebra 0.35.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{ONE_RAW, fx, p2t, sim2t, uct, v2t};
use crate::base::point2::{Point2, Point2Trait};
use crate::base::vector2::Vector2Trait;
use crate::geometry::isometry2::Isometry2Trait;
use crate::geometry::translation2::Translation2Trait;
use crate::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use super::{Similarity2, Similarity2AngleTrait, Similarity2Trait, oracle};


fn id() -> Similarity2<Fixed> {
    Similarity2Trait::<Fixed>::identity()
}

fn a() -> Similarity2<Fixed> {
    Similarity2AngleTrait::new(v2t((0x180000000, -0x240000000)), fx(0x66666666), fx(0x280000000))
}

fn b() -> Similarity2<Fixed> {
    Similarity2AngleTrait::new(v2t((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa), fx(0x180000000))
}

fn quarter() -> UnitComplex<Fixed> {
    uct((0, ONE_RAW))
}

#[test]
fn test_identity_and_constructors_are_exact() {
    let i = id();
    assert!(i.isometry == Isometry2Trait::<Fixed>::identity());
    assert!(i.scaling == Real::one());
    assert!(i.transform_point(p2t((0x123, -0x456))) == p2t((0x123, -0x456)));

    let t = Translation2Trait::new(fx(0x180000000), fx(-0x240000000));
    let r = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666));
    let s = fx(0x280000000);
    let from_parts = Similarity2Trait::from_parts(t, r, s);
    let from_iso = Similarity2Trait::from_isometry(Isometry2Trait::from_parts(t, r), s);
    assert!(from_parts == a() && from_iso == a());
    assert!(Similarity2Trait::from_scaling(s).scaling() == s);
    assert!(
        {
            let mut m = from_parts;
            m.set_scaling(fx(0x180000000));
            m
        }.scaling() == fx(0x180000000),
    );
    let pure = Similarity2Trait::from_isometry(Isometry2Trait::from_parts(t, r), Real::one());
    assert!(pure.scaling == Real::one() && pure.isometry == from_parts.isometry);
}

#[test]
fn test_pure_scaling_and_quarter_turn_are_exact() {
    let s = Similarity2Trait::<Fixed>::from_scaling(fx(0x300000000));
    assert!(s.transform_point(p2t((ONE_RAW, -2 * ONE_RAW))) == p2t((3 * ONE_RAW, -6 * ONE_RAW)));
    assert!(s.transform_vector(v2t((ONE_RAW, -2 * ONE_RAW))) == v2t((3 * ONE_RAW, -6 * ONE_RAW)));

    let q = Similarity2Trait::from_parts(
        Translation2Trait::new(fx(ONE_RAW), Real::zero()), quarter(), fx(0x200000000),
    );
    assert!(q.transform_point(p2t((ONE_RAW, 0))) == p2t((ONE_RAW, 2 * ONE_RAW)));
    assert!(q.transform_vector(v2t((0, ONE_RAW))) == v2t((-2 * ONE_RAW, 0)));
}

#[test]
fn test_inverse_composes_to_identity_and_undoes_points() {
    let x = a();
    assert!((x * x.inverse()).abs_diff_eq(id(), 8));
    assert!((x.inverse() * x).abs_diff_eq(id(), 8));
    let p = p2t((-0x280000000, 0x3c0000000));
    assert!(x.inverse_transform_point(x.transform_point(p)).abs_diff_eq(p, 16));
    let v = v2t((-0x280000000, 0x3c0000000));
    assert!(x.inverse_transform_vector(x.transform_vector(v)).abs_diff_eq(v, 16));
}

#[test]
fn test_mul_matches_actions() {
    let (x, y, p) = (a(), b(), p2t((-0x280000000, 0x3c0000000)));
    assert!((x * y).transform_point(p).abs_diff_eq(x.transform_point(y.transform_point(p)), 16));
}

#[test]
fn test_scaling_append_and_prepend_match_upstream_order() {
    let x = a();
    let s = fx(0x180000000);
    assert!(
        x
            .prepend_scaling(s)
            .transform_point(p2t((ONE_RAW, 0)))
            .abs_diff_eq(x.transform_point(p2t((0x180000000, 0))), 4),
    );
    let p = p2t((ONE_RAW, 0));
    assert!(x.append_scaling(s).transform_point(p).abs_diff_eq(x.transform_point(p).scale(s), 1));
}

#[test]
fn test_append_and_prepend_translation_rotation_match_composition() {
    let (x, t, r) = (a(), Translation2Trait::new(fx(0x140000000), fx(-0x60000000)), quarter());
    let ti: Similarity2<Fixed> = Similarity2Trait::from_isometry(t.into(), Real::one());
    let ri: Similarity2<Fixed> = Similarity2Trait::from_isometry(
        Isometry2Trait::from_parts(Translation2Trait::identity(), r), Real::one(),
    );
    assert!({
        let mut m = x;
        m.append_translation_mut(t);
        m
    } == ti * x);
    assert!(x.mul_translation(t).abs_diff_eq(x * ti, 1));
    assert!({
        let mut m = x;
        m.append_rotation_mut(r);
        m
    } == ri * x);
    assert!(x.mul_unit_complex(r) == x * ri);
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
    let p = p2t((-0x280000000, 0x3c0000000));
    let h = m.mul_vec(p.to_homogeneous());
    assert!(Point2 { x: h.x, y: h.y }.abs_diff_eq(x.transform_point(p), 4));
}

#[test]
fn test_append_rotation_wrt_point_and_center() {
    let (x, r, p) = (a(), quarter(), p2t((0x180000000, -0x80000000)));
    let y = {
        let mut m = x;
        m.append_rotation_wrt_point_mut(r, p);
        m
    };
    let shift: Similarity2<Fixed> = Similarity2Trait::from_isometry(
        Translation2Trait::new(p.x, p.y).into(), Real::one(),
    );
    let back: Similarity2<Fixed> = Similarity2Trait::from_isometry(
        Translation2Trait::new(-p.x, -p.y).into(), Real::one(),
    );
    let ri: Similarity2<Fixed> = Similarity2Trait::from_isometry(
        Isometry2Trait::from_parts(Translation2Trait::identity(), r), Real::one(),
    );
    assert!(y.abs_diff_eq(shift * ri * back * x, 4));
    assert!(
        {
            let mut m = x;
            m.append_rotation_wrt_center_mut(r);
            m
        }
            .isometry
            .translation == x
            .isometry
            .translation,
    );
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_parts_zero_scaling_panics() {
    Similarity2Trait::<
        Fixed,
    >::from_parts(Translation2Trait::identity(), UnitComplexTrait::identity(), Real::zero());
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_isometry_zero_scaling_panics() {
    Similarity2Trait::<Fixed>::from_isometry(Isometry2Trait::identity(), Real::zero());
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_scaling_zero_panics() {
    Similarity2Trait::<Fixed>::from_scaling(Real::zero());
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_with_scaling_zero_panics() {
    {
        let mut m = a();
        m.set_scaling(Real::zero());
        m
    };
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_prepend_scaling_zero_panics() {
    a().prepend_scaling(Real::zero());
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_append_scaling_zero_panics() {
    a().append_scaling(Real::zero());
}

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::similarity2_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!((sim2t(x) * sim2t(y)).abs_diff_eq(sim2t(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::similarity2_inverse_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(sim2t(x).inverse().abs_diff_eq(sim2t(expected), tol));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::similarity2_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(sim2t(x).transform_point(p2t(p)).abs_diff_eq(p2t(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::similarity2_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(sim2t(x).transform_vector(v2t(v)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::similarity2_inverse_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(sim2t(x).inverse_transform_point(p2t(p)).abs_diff_eq(p2t(expected), tol));
    }
}
