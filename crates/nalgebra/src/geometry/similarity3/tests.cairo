//! Unit tests of `Similarity3`: exact cases, identities, zero-scale panics, append/prepend
//! semantics, homogeneous layout, and oracle vectors from upstream nalgebra 0.35.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix4::Matrix4Trait;
use crate::base::matrix_test_utils::{ONE_RAW, fx, p3t, sim3t, uqt, v3t};
use crate::base::point3::{Point3, Point3Trait};
use crate::base::vector3::Vector3Trait;
use crate::geometry::isometry3::Isometry3Trait;
use crate::geometry::translation3::Translation3Trait;
use crate::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use super::{Similarity3, Similarity3AngleTrait, Similarity3Trait, oracle};


fn id() -> Similarity3<Fixed> {
    Similarity3Trait::<Fixed>::identity()
}

fn a() -> Similarity3<Fixed> {
    Similarity3AngleTrait::new(
        v3t((0x180000000, -0x240000000, 0x3c0000000)),
        v3t((0x40000000, -0x30000000, 0x20000000)),
        fx(0x280000000),
    )
}

fn b() -> Similarity3<Fixed> {
    Similarity3AngleTrait::new(
        v3t((-0xc0000000, 0x80000000, 0x140000000)),
        v3t((-0x80000000, 0x60000000, 0xe0000000)),
        fx(0x180000000),
    )
}

fn half_turn_y() -> UnitQuaternion<Fixed> {
    uqt((0, 0, ONE_RAW, 0))
}

#[test]
fn test_identity_and_constructors_are_exact() {
    let i = id();
    assert!(i.isometry == Isometry3Trait::<Fixed>::identity());
    assert!(i.scaling == Real::ONE);
    assert!(i.transform_point(p3t((0x123, -0x456, 0x789))) == p3t((0x123, -0x456, 0x789)));

    let t = Translation3Trait::new(fx(0x180000000), fx(-0x240000000), fx(0x3c0000000));
    let r = UnitQuaternionAngleTrait::<
        Fixed,
    >::from_scaled_axis(v3t((0x40000000, -0x30000000, 0x20000000)));
    let s = fx(0x280000000);
    let from_parts = Similarity3Trait::from_parts(t, r, s);
    let from_iso = Similarity3Trait::from_isometry(Isometry3Trait::from_parts(t, r), s);
    assert!(from_parts == a() && from_iso == a());
    assert!(Similarity3Trait::from_scaling(s).scaling() == s);
    assert!(from_parts.with_scaling(fx(0x180000000)).scaling() == fx(0x180000000));
    let pure: Similarity3<Fixed> = Isometry3Trait::from_parts(t, r).into();
    assert!(pure.scaling == Real::ONE && pure.isometry == from_parts.isometry);
}

#[test]
fn test_pure_scaling_and_half_turn_are_exact() {
    let s = Similarity3Trait::<Fixed>::from_scaling(fx(0x300000000));
    assert!(
        s
            .transform_point(
                p3t((ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW)),
            ) == p3t((3 * ONE_RAW, -6 * ONE_RAW, 9 * ONE_RAW)),
    );
    let q = Similarity3Trait::from_parts(
        Translation3Trait::new(fx(ONE_RAW), Real::ZERO, Real::ZERO), half_turn_y(), fx(0x200000000),
    );
    assert!(
        q
            .transform_point(
                p3t((ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW)),
            ) == p3t((-ONE_RAW, 4 * ONE_RAW, -6 * ONE_RAW)),
    );
}

#[test]
fn test_inverse_composes_to_identity_and_undoes_points() {
    let x = a();
    assert!((x * x.inverse()).abs_diff_eq(id(), 64));
    assert!((x.inverse() * x).abs_diff_eq(id(), 64));
    let p = p3t((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(x.inverse_transform_point(x.transform_point(p)).abs_diff_eq(p, 64));
    let v = v3t((-0x280000000, 0x3c0000000, 0xc0000000));
    assert!(x.inverse_transform_vector(x.transform_vector(v)).abs_diff_eq(v, 64));
}

#[test]
fn test_mul_and_inv_mul_match_actions() {
    let (x, y, p) = (a(), b(), p3t((-0x280000000, 0x3c0000000, 0xc0000000)));
    assert!((x * y).transform_point(p).abs_diff_eq(x.transform_point(y.transform_point(p)), 96));
    assert!(x.inv_mul(y).abs_diff_eq(x.inverse() * y, 32));
    assert!(x.inv_mul(x).abs_diff_eq(id(), 16));
}

#[test]
fn test_scaling_append_and_prepend_match_upstream_order() {
    let x = a();
    let s = fx(0x180000000);
    assert!(
        x
            .prepend_scaling(s)
            .transform_point(p3t((ONE_RAW, 0, 0)))
            .abs_diff_eq(x.transform_point(p3t((0x180000000, 0, 0))), 4),
    );
    let p = p3t((ONE_RAW, 0, 0));
    assert!(x.append_scaling(s).transform_point(p).abs_diff_eq(x.transform_point(p).scale(s), 4));
}

#[test]
fn test_append_and_prepend_translation_rotation_match_composition() {
    let (x, t, r) = (
        a(),
        Translation3Trait::new(fx(0x140000000), fx(-0x60000000), fx(0x280000000)),
        half_turn_y(),
    );
    let ti: Similarity3<Fixed> = Similarity3Trait::from_isometry(t.into(), Real::ONE);
    let ri: Similarity3<Fixed> = Similarity3Trait::from_isometry(r.into(), Real::ONE);
    assert!(x.append_translation(t) == ti * x);
    assert!(x.prepend_translation(t).abs_diff_eq(x * ti, 4));
    assert!(x.append_rotation(r) == ri * x);
    assert!(x.prepend_rotation(r) == x * ri);
}

#[test]
fn test_to_homogeneous_layout_and_action() {
    let x = a();
    let m = x.to_homogeneous();
    let r = x.isometry.rotation.to_rotation_matrix().matrix;
    assert!(m.m11 == r.m11 * x.scaling && m.m22 == r.m22 * x.scaling);
    assert!(m.m14 == x.isometry.translation.vector.x && m.m44 == Real::ONE);
    assert!(id().to_homogeneous() == Matrix4Trait::<Fixed>::identity());
    let p = p3t((-0x280000000, 0x3c0000000, 0xc0000000));
    let h = m.mul_vec(p.to_homogeneous());
    assert!(Point3 { x: h.x, y: h.y, z: h.z }.abs_diff_eq(x.transform_point(p), 16));
}

#[test]
fn test_append_rotation_wrt_point_and_center() {
    let (x, r, p) = (a(), half_turn_y(), p3t((0x180000000, -0x80000000, 0x40000000)));
    let y = x.append_rotation_wrt_point(r, p);
    let shift: Similarity3<Fixed> = Similarity3Trait::from_isometry(
        Translation3Trait::new(p.x, p.y, p.z).into(), Real::ONE,
    );
    let back: Similarity3<Fixed> = Similarity3Trait::from_isometry(
        Translation3Trait::new(-p.x, -p.y, -p.z).into(), Real::ONE,
    );
    let ri: Similarity3<Fixed> = Similarity3Trait::from_isometry(r.into(), Real::ONE);
    assert!(y.abs_diff_eq(shift * ri * back * x, 8));
    assert!(x.append_rotation_wrt_center(r).isometry.translation == x.isometry.translation);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_parts_zero_scaling_panics() {
    Similarity3Trait::<
        Fixed,
    >::from_parts(Translation3Trait::identity(), UnitQuaternionTrait::identity(), Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_isometry_zero_scaling_panics() {
    Similarity3Trait::<Fixed>::from_isometry(Isometry3Trait::identity(), Real::ZERO);
}

#[test]
#[should_panic(expected: 'nalgebra: zero scale')]
fn test_from_scaling_zero_panics() {
    Similarity3Trait::<Fixed>::from_scaling(Real::ZERO);
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
    let mut cases = oracle::similarity3_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!((sim3t(x) * sim3t(y)).abs_diff_eq(sim3t(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::similarity3_inverse_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(sim3t(x).inverse().abs_diff_eq(sim3t(expected), tol));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::similarity3_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(sim3t(x).transform_point(p3t(p)).abs_diff_eq(p3t(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::similarity3_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(sim3t(x).transform_vector(v3t(v)).abs_diff_eq(v3t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::similarity3_inverse_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(sim3t(x).inverse_transform_point(p3t(p)).abs_diff_eq(p3t(expected), tol));
    }
}
