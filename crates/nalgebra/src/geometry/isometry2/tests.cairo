//! Unit tests of `Isometry2`: exact cases (identity, pure translations, quarter turns), the
//! identities a rigid-body transform must satisfy (`iso · iso⁻¹ = id`, `inv_mul(a, b) =
//! a⁻¹ · b`, `to_homogeneous` acts like `transform_point`, append / prepend against the
//! composition), and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs).
//!
//! Stated tolerances: the translation part of a pure translation is carried BIT FOR BIT
//! (`floor(x + t) = floor(x) + t` for an integral `t` in raw units); everything that goes through
//! a rotation is floored once per component, which the oracle bounds at 2 ulp for the whole
//! `isometry2` suite. `inv_mul(a, b)` and `a.inverse() * b` agree within 2 ulp but not bit for bit
//! (the second rounds one more intermediate): see `benches.cairo`.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution, the
//! whole `isometry2` suite) with
//! `cargo run --release -- emit-cairo isometry2 --from vectors --max-per-dist 6 --out
//! <oracle.cairo>`.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::MatrixMul;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{ONE_RAW, fx, iso2t, m3, p2t, uct, v2t};
use crate::base::point2::{Point2, Point2Trait};
use crate::base::vector2::{Vector2, Vector2Trait};
use crate::geometry::isometry2::Isometry2InternalTrait;
use crate::geometry::translation2::{Translation2, Translation2Trait};
use crate::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use super::{Isometry2, Isometry2AngleTrait, Isometry2Trait, oracle};

/// The raw value of 1.

fn id() -> Isometry2<Fixed> {
    Isometry2Trait::<Fixed>::identity()
}

/// `new((1.5, -2.25), 0.4 rad)`.
fn a() -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(v2t((0x180000000, -0x240000000)), fx(0x66666666))
}

/// `new((-0.75, 0.5), -1/6 rad)`.
fn b() -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(v2t((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa))
}

/// The quarter turn, exactly representable.
fn quarter() -> UnitComplex<Fixed> {
    uct((0, ONE_RAW))
}

// --- construction

#[test]
fn test_identity_is_exact() {
    let i = id();
    assert!(i.rotation == UnitComplexTrait::<Fixed>::identity());
    assert!(i.translation == Translation2Trait::<Fixed>::identity());
    assert!(i.transform_point(p2t((0x123, -0x456))) == p2t((0x123, -0x456)));
}

#[test]
fn test_from_parts_and_the_constructors_agree() {
    let t = Translation2Trait::new(fx(0x180000000), fx(-0x240000000));
    let r = UnitComplexAngleTrait::<Fixed>::new(fx(0x66666666));
    let i = Isometry2Trait::from_parts(t, r);
    assert!(i == a());
    assert!(i.translation == t && i.rotation == r);
    let pure_t: Isometry2<Fixed> = t.into();
    let pure_r = Isometry2Trait::from_parts(Translation2Trait::identity(), r);
    assert!(Isometry2Trait::translation(fx(0x180000000), fx(-0x240000000)) == pure_t);
    assert!(Isometry2AngleTrait::<Fixed>::rotation(fx(0x66666666)) == pure_r);
}

#[test]
fn test_pure_translation_moves_points_exactly() {
    let t = Isometry2Trait::translation(fx(0x140000000), fx(-0x60000000));
    assert!(t.rotation == UnitComplexTrait::<Fixed>::identity());
    // The rotation is the exact identity, so the point comes out bit for bit shifted.
    assert!(
        t.transform_point(p2t((-0x280000000, 0x3c0000000))) == p2t((-0x140000000, 0x360000000)),
    );
    assert!(t.transform_vector(v2t((7, -9))) == v2t((7, -9)));
}

#[test]
fn test_quarter_turn_is_exact() {
    let i = Isometry2Trait::from_parts(
        Translation2Trait::new(fx(ONE_RAW), Real::zero()), quarter(),
    );
    // (1, 0) turns into (0, 1), then the translation adds (1, 0).
    assert!(i.transform_point(p2t((ONE_RAW, 0))) == p2t((ONE_RAW, ONE_RAW)));
    assert!(i.transform_vector(v2t((0, ONE_RAW))) == v2t((-ONE_RAW, 0)));
    assert!(i.inverse_transform_point(p2t((ONE_RAW, ONE_RAW))) == p2t((ONE_RAW, 0)));
}

// --- inverse, composition, inv_mul

#[test]
fn test_inverse_composes_to_the_identity() {
    let i = a();
    assert!((i * i.inverse()).abs_diff_eq(id(), 2));
    assert!((i.inverse() * i).abs_diff_eq(id(), 2));
    assert!(i.inverse().inverse().abs_diff_eq(i, 2));
}

#[test]
fn test_inverse_of_a_pure_translation_is_exact() {
    let t = Isometry2Trait::translation(fx(0x140000000), fx(-0x60000000));
    assert!(t.inverse() == Isometry2Trait::translation(fx(-0x140000000), fx(0x60000000)));
}

#[test]
fn test_inverse_transform_point_undoes_transform_point() {
    let (i, p) = (a(), p2t((-0x280000000, 0x3c0000000)));
    assert!(i.inverse_transform_point(i.transform_point(p)).abs_diff_eq(p, 4));
    // and it agrees with the materialised inverse, within the extra rounding the latter pays.
    assert!(i.inverse_transform_point(p).abs_diff_eq(i.inverse().transform_point(p), 4));
}

#[test]
fn test_inverse_transform_vector_is_the_conjugate_rotation() {
    let (i, v) = (a(), v2t((-0x280000000, 0x3c0000000)));
    assert!(i.inverse_transform_vector(v) == i.rotation.inverse_transform_vector(v));
    assert!(i.inverse_transform_vector(i.transform_vector(v)).abs_diff_eq(v, 4));
}

#[test]
fn test_mul_is_the_composition_of_the_actions() {
    let (x, y, p) = (a(), b(), p2t((-0x280000000, 0x3c0000000)));
    assert!((x * y).transform_point(p).abs_diff_eq(x.transform_point(y.transform_point(p)), 4));
    assert!(x * id() == x);
    assert!((id() * x) == x);
}

#[test]
fn test_inv_mul_is_the_inverse_times_other_within_two_ulp() {
    let (x, y) = (a(), b());
    // Same transform, but `inv_mul` rounds one intermediate less (see `benches.cairo`).
    assert!(x.inv_mul(y).abs_diff_eq(x.inverse() * y, 2));
    assert!(x.inv_mul(x).abs_diff_eq(id(), 2));
    // The relative pose maps `y`'s frame into `x`'s.
    let p = p2t((-0x280000000, 0x3c0000000));
    assert!(
        x
            .inv_mul(y)
            .transform_point(p)
            .abs_diff_eq(x.inverse_transform_point(y.transform_point(p)), 4),
    );
}

// --- append / prepend

#[test]
fn test_append_and_prepend_translation_match_the_composition() {
    let (i, t) = (a(), Translation2Trait::new(fx(0x140000000), fx(-0x60000000)));
    let ti: Isometry2<Fixed> = t.into();
    assert!({
        let mut m = i;
        m.append_translation_mut(t);
        m
    } == ti * i);
    assert!(i.mul_translation(t) == i * ti);
}

#[test]
fn test_append_and_prepend_rotation_match_the_composition() {
    let (i, r) = (a(), UnitComplexAngleTrait::<Fixed>::new(fx(0x1999999a)));
    let ri = Isometry2Trait::from_parts(Translation2Trait::identity(), r);
    assert!({
        let mut m = i;
        m.append_rotation_mut(r);
        m
    } == ri * i);
    assert!(i.mul_unit_complex(r) == i * ri);
}

#[test]
fn test_append_rotation_wrt_point_fixes_that_point() {
    let (i, r, p) = (a(), quarter(), p2t((0x180000000, -0x80000000)));
    let j = {
        let mut m = i;
        m.append_rotation_wrt_point_mut(r, p);
        m
    };
    // T(p) · R(r) · T(-p) · self, the definition.
    let shift: Isometry2<Fixed> = Translation2Trait::new(p.x, p.y).into();
    let back: Isometry2<Fixed> = Translation2Trait::new(-p.x, -p.y).into();
    let ri = Isometry2Trait::from_parts(Translation2Trait::identity(), r);
    assert!(j.abs_diff_eq(shift * ri * back * i, 2));
    assert!(j.rotation == r * i.rotation);
}

#[test]
fn test_append_rotation_wrt_center_keeps_the_translation() {
    let (i, r) = (a(), quarter());
    let j = {
        let mut m = i;
        m.append_rotation_wrt_center_mut(r);
        m
    };
    assert!(j.translation == i.translation);
    assert!(j.rotation == r * i.rotation);
    // Rotating about the isometry's own centre is `append_rotation_wrt_point` at that centre.
    let c = Point2 { x: i.translation.vector.x, y: i.translation.vector.y };
    assert!(j == {
        let mut m = i;
        m.append_rotation_wrt_point_mut(r, c);
        m
    });
}

// --- homogeneous form

#[test]
fn test_to_homogeneous_layout() {
    let i = a();
    let m = i.to_homogeneous();
    assert!(m.m11 == i.rotation.re && m.m21 == i.rotation.im);
    assert!(m.m12 == -i.rotation.im && m.m22 == i.rotation.re);
    assert!(m.m13 == i.translation.vector.x && m.m23 == i.translation.vector.y);
    assert!(m.m31 == Real::zero() && m.m32 == Real::zero() && m.m33 == Real::one());
    assert!(id().to_homogeneous() == Matrix3Trait::<Fixed>::identity());
}

/// The 3x3 product `M · (p, 1)` is the fully fused form of `transform_point` — the same bits,
/// since adding an integral translation commutes with the floor.
#[test]
fn test_to_homogeneous_acts_like_transform_point_bit_for_bit() {
    let (i, p) = (a(), p2t((-0x280000000, 0x3c0000000)));
    let h = i.to_homogeneous().mul_mat(p.to_homogeneous());
    let got = i.transform_point(p);
    assert!(h.x == got.x && h.y == got.y && h.z == Real::one());
}

// --- renormalisation, comparison, interpolation

#[test]
fn test_renormalize_restores_the_rotation_and_keeps_the_translation() {
    // A rotation scaled by 4 (exact): both renormalisations bring it back, the translation is
    // untouched.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let four = Real::<Fixed>::from_int(4);
    let drifted = Isometry2Trait::from_parts(
        Translation2Trait::new(fx(0x180000000), fx(-0x240000000)),
        UnitComplex { re: c.re * four, im: c.im * four },
    );
    let r = drifted.renormalize();
    assert!(r.translation == drifted.translation);
    assert!(r.rotation.abs_diff_eq(c, 2));
    // `renormalize_fast` is a single Newton step: valid near 1, so apply it to a small drift.
    let small = Isometry2Trait::from_parts(
        drifted.translation, UnitComplex { re: c.re + fx(3), im: c.im - fx(2) },
    );
    assert!(small.renormalize_fast().rotation.abs_diff_eq(small.renormalize().rotation, 2));
    assert!(small.renormalize_fast().translation == small.translation);
}

#[test]
fn test_abs_diff_eq_covers_both_parts() {
    let i = a();
    let j = Isometry2Trait::from_parts(
        Translation2 {
            vector: Vector2 {
                x: i.translation.vector.x + fx(2), y: i.translation.vector.y - fx(1),
            },
        },
        UnitComplex { re: i.rotation.re + fx(1), im: i.rotation.im },
    );
    assert!(i.abs_diff_eq(j, 2));
    assert!(!i.abs_diff_eq(j, 1));
    assert!(i.abs_diff_eq(i, 0));
}

#[test]
fn test_lerp_nlerp_endpoints_and_midpoint() {
    let (x, y) = (a(), b());
    // `t = 0` and `t = 1` give the endpoints back, up to the renormalisation of the rotation.
    assert!(x.lerp_nlerp(y, Real::zero()).abs_diff_eq(x, 2));
    assert!(x.lerp_nlerp(y, Real::one()).abs_diff_eq(y, 2));
    // The translation is the exact `lerp`, whatever the rotation does.
    let h = x.lerp_nlerp(y, Real::HALF);
    assert!(
        h
            .translation
            .vector
            .x == Real::lerp(x.translation.vector.x, y.translation.vector.x, Real::HALF),
    );
    // The interpolated rotation is unit and lies between the two (half the angle to `y`).
    assert!(Real::abs_diff_eq(Real::norm_squared2(h.rotation.re, h.rotation.im), Real::one(), 2));
    let half_angle = x.rotation.angle_to(y.rotation) * Real::HALF;
    assert!(Real::abs_diff_eq(x.rotation.angle_to(h.rotation), half_angle, 1048576));
}

#[test]
fn test_lerp_slerp_endpoints_and_agreement_with_lerp_nlerp() {
    let (x, y) = (a(), b());
    assert!(x.lerp_slerp(y, Real::zero()).abs_diff_eq(x, 2));
    assert!(x.lerp_slerp(y, Real::one()).abs_diff_eq(y, 8));
    // The two interpolations differ by the parametrisation only: the angle to `x` is the same
    // at the midpoint (both walk half the arc there).
    let (s, n) = (x.lerp_slerp(y, Real::HALF), x.lerp_nlerp(y, Real::HALF));
    assert!(s.translation == n.translation);
    assert!(s.rotation.abs_diff_eq(n.rotation, 65536));
}

// --- oracle vectors (upstream nalgebra 0.35)

#[test]
fn test_mul_oracle() {
    let mut cases = oracle::isometry2_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!((iso2t(x) * iso2t(y)).abs_diff_eq(iso2t(expected), tol));
    }
}

#[test]
fn test_inverse_oracle() {
    let mut cases = oracle::isometry2_inverse_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(iso2t(x).inverse().abs_diff_eq(iso2t(expected), tol));
    }
}

#[test]
fn test_inv_mul_oracle() {
    let mut cases = oracle::isometry2_inv_mul_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, y, expected, tol) = *case;
        assert!(iso2t(x).inv_mul(iso2t(y)).abs_diff_eq(iso2t(expected), tol));
    }
}

#[test]
fn test_transform_point_oracle() {
    let mut cases = oracle::isometry2_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(iso2t(x).transform_point(p2t(p)).abs_diff_eq(p2t(expected), tol));
    }
}

#[test]
fn test_transform_vector_oracle() {
    let mut cases = oracle::isometry2_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(iso2t(x).transform_vector(v2t(v)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_point_oracle() {
    let mut cases = oracle::isometry2_inverse_transform_point_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, p, expected, tol) = *case;
        assert!(iso2t(x).inverse_transform_point(p2t(p)).abs_diff_eq(p2t(expected), tol));
    }
}

#[test]
fn test_inverse_transform_vector_oracle() {
    let mut cases = oracle::isometry2_inverse_transform_vector_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, v, expected, tol) = *case;
        assert!(iso2t(x).inverse_transform_vector(v2t(v)).abs_diff_eq(v2t(expected), tol));
    }
}

#[test]
fn test_to_homogeneous_oracle() {
    let mut cases = oracle::isometry2_to_homogeneous_cases();
    assert!(cases.len() >= 24);
    while let Some(case) = cases.pop_front() {
        let (x, expected, tol) = *case;
        assert!(tol == 0);
        assert!(iso2t(x).to_homogeneous() == m3(expected));
    }
}
