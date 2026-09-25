//! Unit tests of the category rules and the standard traits of the transform types (WP 8.4-P11a):
//! upstream's `TCategoryMul` table through `TransformMul` / `TransformDiv`, `set_category`, the
//! widening `Into` and the checked `TryInto` (`check_homogeneous_invariants`), the conversions of
//! the geometry types, the divisions of upstream (`Rotation / Transform` is `r⁻¹ * t`), `Index`,
//! `One`, `Default`, the approximate comparisons, `Hash`, `Serde`, and `Perspective3` /
//! `Orthographic3::to_projective`.

use core::hash::{HashStateExTrait, HashStateTrait};
use core::num::traits::One;
use core::poseidon::PoseidonTrait;
use fixed::Fixed;
use nalgebra::base::matrix3::{Matrix3, Matrix3Trait};
use nalgebra::base::matrix4::{Matrix4, Matrix4Trait};
use nalgebra::geometry::affine2::{Affine2, Affine2Trait};
use nalgebra::geometry::affine3::{Affine3, Affine3Trait};
use nalgebra::geometry::isometry2::Isometry2Trait;
use nalgebra::geometry::isometry3::Isometry3Trait;
use nalgebra::geometry::isometry_matrix2::IsometryMatrix2Trait;
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3Trait;
use nalgebra::geometry::orthographic3::Orthographic3Trait;
use nalgebra::geometry::perspective3::Perspective3Trait;
use nalgebra::geometry::projective2::{Projective2, Projective2Trait};
use nalgebra::geometry::projective3::{Projective3, Projective3Trait};
use nalgebra::geometry::rotation2::Rotation2Trait;
use nalgebra::geometry::rotation3::Rotation3Trait;
use nalgebra::geometry::scale2::Scale2Trait;
use nalgebra::geometry::scale3::Scale3Trait;
use nalgebra::geometry::similarity2::Similarity2Trait;
use nalgebra::geometry::similarity3::Similarity3Trait;
use nalgebra::geometry::similarity_matrix2::SimilarityMatrix2Trait;
use nalgebra::geometry::similarity_matrix3::SimilarityMatrix3Trait;
use nalgebra::geometry::transform::{TransformDiv, TransformMul, TransformSetCategory};
use nalgebra::geometry::transform2::{Transform2, Transform2Trait};
use nalgebra::geometry::transform3::{Transform3, Transform3Trait};
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::Translation3Trait;
use nalgebra::geometry::unit_complex::UnitComplexTrait;
use nalgebra::geometry::unit_dual_quaternion::{UnitDualQuaternion, UnitDualQuaternionTrait};
use nalgebra::geometry::unit_quaternion::UnitQuaternionTrait;
use nalgebra_tests_utils::{
    ONE_RAW, aff2t, aff3t, fx, int, iso2t, iso3t, isom2t, isom3t, m3i, m4i, ortho6t, pers4t, proj3,
    sim2t, sim3t, simm2t, simm3t, t2, t3, uct, uqt,
};

fn a() -> Affine3<Fixed> {
    aff3t(
        [[2 * ONE_RAW, ONE_RAW, 0], [0, ONE_RAW, -ONE_RAW], [ONE_RAW, 0, 3 * ONE_RAW]],
        (ONE_RAW, -2 * ONE_RAW, 3 * ONE_RAW),
    )
}

fn p() -> Projective3<Fixed> {
    proj3(
        [
            [2 * ONE_RAW, 0, ONE_RAW, 0], [0, ONE_RAW, 0, -ONE_RAW], [ONE_RAW, 0, 3 * ONE_RAW, 0],
            [0, ONE_RAW, 0, 2 * ONE_RAW],
        ],
    )
}

fn g() -> Transform3<Fixed> {
    Transform3Trait::from_matrix_unchecked(
        m4i([[1, 2, 0, 0], [2, 4, 0, 0], [0, 0, 1, 0], [0, 0, 1, 1]]),
    )
}

/// The rotation of 1 radian about `(1, 2, 2) / 3`.
fn q() -> nalgebra::geometry::unit_quaternion::UnitQuaternion<Fixed> {
    uqt((3769188403, 686372336, 1372744673, 1372744673))
}

// --- upstream's TCategoryMul table

#[test]
fn test_category_of_products() {
    let (am, pm, gm) = (a().into_inner(), p().into_inner(), g().into_inner());
    // TAffine * TAffine => TAffine, ... (the output types are the assertions).
    let x: Affine3<Fixed> = a().mul_transform(a());
    assert!(x.into_inner() == am * am && x == a() * a());
    let x: Projective3<Fixed> = a().mul_transform(p());
    assert!(x.into_inner() == am * pm);
    let x: Transform3<Fixed> = a().mul_transform(g());
    assert!(x.into_inner() == am * gm);
    let x: Projective3<Fixed> = p().mul_transform(a());
    assert!(x.into_inner() == pm * am);
    let x: Projective3<Fixed> = p().mul_transform(p());
    assert!(x.into_inner() == pm * pm && x == p() * p());
    let x: Transform3<Fixed> = p().mul_transform(g());
    assert!(x.into_inner() == pm * gm);
    let x: Transform3<Fixed> = g().mul_transform(a());
    assert!(x.into_inner() == gm * am);
    let x: Transform3<Fixed> = g().mul_transform(p());
    assert!(x.into_inner() == gm * pm);
    let x: Transform3<Fixed> = g().mul_transform(g());
    assert!(x.into_inner() == gm * gm && x == g() * g());
}

#[test]
fn test_category_of_quotients() {
    let (am, pm, gm) = (a().into_inner(), p().into_inner(), g().into_inner());
    let (ai, pi) = (a().inverse().into_inner(), p().inverse().into_inner());
    let x: Affine3<Fixed> = a().div_transform(a());
    assert!(x.into_inner() == am * ai && x == a() / a());
    let x: Projective3<Fixed> = a().div_transform(p());
    assert!(x.into_inner() == am * pi);
    let x: Projective3<Fixed> = p().div_transform(a());
    assert!(x.into_inner() == pm * ai);
    let x: Projective3<Fixed> = p().div_transform(p());
    assert!(x.into_inner() == pm * pi && x == p() / p());
    let x: Transform3<Fixed> = g().div_transform(a());
    assert!(x.into_inner() == gm * ai);
    let x: Transform3<Fixed> = g().div_transform(p());
    assert!(x.into_inner() == gm * pi);
}

#[test]
fn test_geometry_times_transform_keeps_the_category() {
    let r = q().to_rotation_matrix();
    let x: Projective3<Fixed> = r.mul_transform(p());
    assert!(x.into_inner() == r.to_homogeneous() * p().into_inner());
    let x: Transform3<Fixed> = q().mul_transform(g());
    assert!(x.into_inner() == q().to_homogeneous() * g().into_inner());
    let tr = t3(ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW);
    let x: Projective3<Fixed> = tr.mul_transform(p());
    assert!(x.into_inner() == tr.to_homogeneous() * p().into_inner());
    // The same-category products of a transform by a geometry type keep its category.
    assert!(p().mul_rotation(r).into_inner() == p().into_inner() * r.to_homogeneous());
    assert!(g().mul_translation(tr).into_inner() == g().into_inner() * tr.to_homogeneous());
}

#[test]
fn test_upstream_divisions_by_a_transform() {
    // Upstream's `Rotation / Transform`, `UnitQuaternion / Transform`, `Translation / Transform`
    // are `self.inverse() * rhs` (sic).
    let r = q().to_rotation_matrix();
    let x: Transform3<Fixed> = r.div_transform(g());
    assert!(x.into_inner() == r.inverse().to_homogeneous() * g().into_inner());
    let x: Affine3<Fixed> = q().div_transform(a());
    assert!(x.into_inner() == q().inverse().to_homogeneous() * a().into_inner());
    let tr = t3(ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW);
    let x: Projective3<Fixed> = tr.div_transform(p());
    assert!(x.into_inner() == tr.inverse().to_homogeneous() * p().into_inner());
    let r2 = uct((2576980378, 3435973837)).to_rotation_matrix();
    let b = aff2t([[2 * ONE_RAW, ONE_RAW], [-ONE_RAW, 3 * ONE_RAW]], (ONE_RAW, 0));
    let x: Affine2<Fixed> = r2.div_transform(b);
    assert!(x.into_inner() == r2.inverse().to_homogeneous() * b.into_inner());
    let x: Affine2<Fixed> = t2(ONE_RAW, 0).div_transform(b);
    assert!(x.into_inner() == t2(-ONE_RAW, 0).to_homogeneous() * b.into_inner());
}

// --- set_category, Into, TryInto

#[test]
fn test_set_category_and_widening() {
    let x: Projective3<Fixed> = a().set_category();
    assert!(x.into_inner() == a().into_inner());
    let x: Transform3<Fixed> = a().set_category();
    assert!(x.into_inner() == a().into_inner());
    let x: Affine3<Fixed> = a().set_category();
    assert!(x == a());
    let x: Transform3<Fixed> = p().set_category();
    assert!(x.into_inner() == p().into_inner());
    let x: Transform3<Fixed> = g().set_category();
    assert!(x == g());
    let x: Projective3<Fixed> = a().into();
    assert!(x.into_inner() == a().into_inner());
    let x: Transform3<Fixed> = p().into();
    assert!(x.into_inner() == p().into_inner());
    let b = aff2t([[2 * ONE_RAW, ONE_RAW], [-ONE_RAW, 3 * ONE_RAW]], (ONE_RAW, 0));
    let x: Transform2<Fixed> = b.set_category();
    let y: Projective2<Fixed> = b.into();
    assert!(x.into_inner() == b.into_inner() && y.into_inner() == b.into_inner());
}

#[test]
fn test_checked_narrowing() {
    // General -> projective: invertible only.
    let x: Option<Projective3<Fixed>> = g().try_into();
    assert!(x.is_none());
    let pg: Transform3<Fixed> = p().into();
    let x: Option<Projective3<Fixed>> = pg.try_into();
    assert!(x.unwrap() == p());
    // -> affine: the last row EXACTLY (0, 0, 0, 1), and invertible.
    let x: Option<Affine3<Fixed>> = p().try_into();
    assert!(x.is_none());
    let ap: Projective3<Fixed> = a().into();
    let x: Option<Affine3<Fixed>> = ap.try_into();
    assert!(x.unwrap() == a());
    let mut m = a().into_inner();
    m.m44 = fx(ONE_RAW + 1);
    let x: Option<Affine3<Fixed>> = m.try_into();
    assert!(x.is_none());
    let sing = m4i([[1, 2, 0, 0], [2, 4, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]);
    let x: Option<Affine3<Fixed>> = sing.try_into();
    assert!(x.is_none());
    let x: Option<Projective3<Fixed>> = sing.try_into();
    assert!(x.is_none());
    // Every matrix is a general transform.
    let x: Option<Transform3<Fixed>> = sing.try_into();
    assert!(x.unwrap().into_inner() == sing);
    let x: Option<Affine3<Fixed>> = a().into_inner().try_into();
    assert!(x.unwrap() == a());
    // 2D.
    let m2 = m3i([[2, 1, 1], [-1, 3, 0], [0, 0, 1]]);
    let x: Option<Affine2<Fixed>> = m2.try_into();
    assert!(x.unwrap().into_inner() == m2);
    let x: Option<Affine2<Fixed>> = m3i([[2, 1, 1], [-1, 3, 0], [0, 1, 1]]).try_into();
    assert!(x.is_none());
    let x: Option<Projective2<Fixed>> = m3i([[1, 2, 0], [2, 4, 0], [0, 0, 1]]).try_into();
    assert!(x.is_none());
    let t2g: Transform2<Fixed> = Transform2Trait::from_matrix_unchecked(m2);
    let x: Option<Affine2<Fixed>> = t2g.try_into();
    let y: Option<Projective2<Fixed>> = t2g.try_into();
    assert!(x.is_some() && y.is_some());
    let x: Option<Transform2<Fixed>> = m2.try_into();
    assert!(x.is_some());
}

#[test]
fn test_into_matrix() {
    let m: Matrix4<Fixed> = a().into();
    assert!(m == a().into_inner());
    let m: Matrix4<Fixed> = p().into();
    assert!(m == p().into_inner());
    let m: Matrix4<Fixed> = g().into();
    assert!(m == g().into_inner());
    let b = aff2t([[2 * ONE_RAW, ONE_RAW], [-ONE_RAW, 3 * ONE_RAW]], (ONE_RAW, 0));
    let m: Matrix3<Fixed> = b.into();
    assert!(m == b.into_inner());
    let pb: Projective2<Fixed> = b.into();
    let m: Matrix3<Fixed> = pb.into();
    let tb: Transform2<Fixed> = b.into();
    let n: Matrix3<Fixed> = tb.into();
    assert!(m == n);
}

#[test]
fn test_geometry_types_into_transforms_3d() {
    let r = q().to_rotation_matrix();
    let tr = t3(ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW);
    let iso = iso3t(
        ((ONE_RAW, 2 * ONE_RAW, -ONE_RAW), (3769188403, 686372336, 1372744673, 1372744673)),
    );
    let isom = isom3t((ONE_RAW, 0, -ONE_RAW), [[0, -ONE_RAW, 0], [ONE_RAW, 0, 0], [0, 0, ONE_RAW]]);
    let sim = sim3t(
        (
            (ONE_RAW, 2 * ONE_RAW, -ONE_RAW),
            (3769188403, 686372336, 1372744673, 1372744673),
            3 * ONE_RAW / 2,
        ),
    );
    let simm = simm3t(
        (ONE_RAW, 0, -ONE_RAW), [[0, -ONE_RAW, 0], [ONE_RAW, 0, 0], [0, 0, ONE_RAW]], 2 * ONE_RAW,
    );
    let s = Scale3Trait::new(int(2), int(3), int(4));
    let dq: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::from_isometry(iso);
    let x: Affine3<Fixed> = r.into();
    assert!(x.into_inner() == r.to_homogeneous());
    let x: Affine3<Fixed> = q().into();
    assert!(x.into_inner() == q().to_homogeneous());
    let x: Affine3<Fixed> = tr.into();
    assert!(x.into_inner() == tr.to_homogeneous());
    let x: Affine3<Fixed> = iso.into();
    assert!(x.into_inner() == iso.to_homogeneous());
    let x: Affine3<Fixed> = isom.into();
    assert!(x.into_inner() == isom.to_homogeneous());
    let x: Affine3<Fixed> = sim.into();
    assert!(x.into_inner() == sim.to_homogeneous());
    let x: Affine3<Fixed> = simm.into();
    assert!(x.into_inner() == simm.to_homogeneous());
    let x: Affine3<Fixed> = s.into();
    assert!(x.into_inner() == s.to_homogeneous());
    let x: Affine3<Fixed> = dq.into();
    assert!(x.into_inner() == dq.to_homogeneous());
    // Into every category (upstream: `C: SuperTCategoryOf<TAffine>`).
    let x: Projective3<Fixed> = iso.into();
    let y: Transform3<Fixed> = iso.into();
    assert!(x.into_inner() == iso.to_homogeneous() && y.into_inner() == iso.to_homogeneous());
    let x: Projective3<Fixed> = dq.into();
    let y: Transform3<Fixed> = s.into();
    assert!(x.into_inner() == dq.to_homogeneous() && y.into_inner() == s.to_homogeneous());
}

#[test]
fn test_geometry_types_into_transforms_2d() {
    let c = uct((2576980378, 3435973837));
    let r = c.to_rotation_matrix();
    let tr = t2(ONE_RAW, 2 * ONE_RAW);
    let iso = iso2t(((ONE_RAW / 4, -3 * ONE_RAW / 2), (2576980378, 3435973837)));
    let isom = isom2t((ONE_RAW, 0), [[0, -ONE_RAW], [ONE_RAW, 0]]);
    let sim = sim2t(((ONE_RAW, 2 * ONE_RAW), (2576980378, 3435973837), 3 * ONE_RAW / 2));
    let simm = simm2t((ONE_RAW, 0), [[0, -ONE_RAW], [ONE_RAW, 0]], 2 * ONE_RAW);
    let s = Scale2Trait::new(int(2), int(3));
    let x: Affine2<Fixed> = r.into();
    assert!(x.into_inner() == r.to_homogeneous());
    let x: Affine2<Fixed> = c.into();
    assert!(x.into_inner() == c.to_homogeneous());
    let x: Affine2<Fixed> = tr.into();
    assert!(x.into_inner() == tr.to_homogeneous());
    let x: Affine2<Fixed> = iso.into();
    assert!(x.into_inner() == iso.to_homogeneous());
    let x: Affine2<Fixed> = isom.into();
    assert!(x.into_inner() == isom.to_homogeneous());
    let x: Affine2<Fixed> = sim.into();
    assert!(x.into_inner() == sim.to_homogeneous());
    let x: Affine2<Fixed> = simm.into();
    assert!(x.into_inner() == simm.to_homogeneous());
    let x: Affine2<Fixed> = s.into();
    assert!(x.into_inner() == s.to_homogeneous());
    let x: Projective2<Fixed> = c.into();
    let y: Transform2<Fixed> = sim.into();
    assert!(x.into_inner() == c.to_homogeneous() && y.into_inner() == sim.to_homogeneous());
}

// --- standard traits

#[test]
fn test_identity_one_default_index() {
    let i: Affine3<Fixed> = One::one();
    assert!(i.is_one() && !a().is_one() && a().is_non_one());
    let d: Affine3<Fixed> = Default::default();
    assert!(d == i && i.into_inner() == Matrix4Trait::identity());
    let d: Projective2<Fixed> = Default::default();
    let o: Projective2<Fixed> = One::one();
    assert!(d == o && d.into_inner() == Matrix3Trait::identity());
    let d: Transform3<Fixed> = Default::default();
    assert!(d.is_one());
    let x = a();
    assert!(x[(0, 0)] == int(2) && x[(0, 3)] == int(1) && x[(2, 2)] == int(3));
    assert!(x[(3, 3)] == int(1) && x[(3, 0)] == int(0) && x[(1, 3)] == int(-2));
    let y = p();
    assert!(y[(3, 1)] == int(1) && y[(3, 3)] == int(2));
    let b: Affine2<Fixed> = aff2t([[2 * ONE_RAW, ONE_RAW], [-ONE_RAW, 3 * ONE_RAW]], (ONE_RAW, 0));
    assert!(b[(1, 0)] == int(-1) && b[(0, 2)] == int(1) && b[(2, 2)] == int(1));
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_index_out_of_bounds_panics() {
    let x = a();
    let _v = x[(4, 0)];
}

#[test]
fn test_approx_comparisons() {
    let x = a();
    let mut m = x.into_inner();
    m.m14 = fx(m.m14.raw + 3);
    let y: Affine3<Fixed> = Affine3Trait::from_matrix_unchecked(m);
    assert!(x.abs_diff_eq(y, 3) && !x.abs_diff_eq(y, 2));
    assert!(x.relative_eq(y, 3, fx(0)) && !x.relative_eq(y, 2, fx(0)));
    assert!(x.ulps_eq(y, 0, 3) && !x.ulps_eq(y, 0, 2));
    assert!(p().abs_diff_eq(p(), 0) && g().ulps_eq(g(), 0, 0));
}

#[test]
fn test_hash_follows_equality() {
    let h1 = PoseidonTrait::new().update_with(a()).finalize();
    let h2 = PoseidonTrait::new().update_with(a()).finalize();
    let h3 = PoseidonTrait::new().update_with(a() * a()).finalize();
    assert!(h1 == h2 && h1 != h3);
    // The hash of the matrix (upstream hashes its matrix).
    assert!(h1 == PoseidonTrait::new().update_with(a().into_inner()).finalize());
}

#[test]
fn test_serde_is_the_matrix() {
    let mut out = array![];
    a().serialize(ref out);
    let mut expected = array![];
    a().into_inner().serialize(ref expected);
    assert!(out == expected);
    let mut span = out.span();
    let back: Affine3<Fixed> = Serde::deserialize(ref span).unwrap();
    assert!(back == a());
    let mut out = array![];
    p().serialize(ref out);
    let mut span = out.span();
    let back: Projective3<Fixed> = Serde::deserialize(ref span).unwrap();
    assert!(back == p());
}

#[test]
fn test_projections_to_projective() {
    let pers = pers4t((5583457485, 7443676160, -4402341478, -869730079));
    assert!(pers.to_projective().into_inner() == pers.into_inner());
    assert!(pers.as_projective() == pers.to_projective());
    let ortho = ortho6t((4294967296, 429496730, 8589934592, -429496730, -4294967296, 0));
    assert!(ortho.to_projective().into_inner() == ortho.into_inner());
    assert!(ortho.as_projective() == ortho.to_projective());
    // A perspective is invertible: its projective inverse (whole-matrix) is its closed-form
    // inverse within a few ulp (9 here).
    let inv = pers.to_projective().inverse().into_inner();
    let d = nalgebra_tests_utils::max_ulp_diff4(inv, pers.inverse());
    assert!(d <= 16);
}
