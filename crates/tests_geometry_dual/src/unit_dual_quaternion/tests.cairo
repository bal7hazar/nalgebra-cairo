//! Unit tests of `UnitDualQuaternion` (WP 8.4-P12) and of the products of `UnitQuaternion`,
//! `Translation3` and `Isometry3` with it: the oracle vectors of `tools/oracle` (suite
//! `dual_quaternion`) and exact cases.

use core::cmp::max;
use core::hash::{HashStateExTrait, HashStateTrait};
use core::num::traits::One;
use core::poseidon::PoseidonTrait;
use fixed::Fixed;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::base::unit::Unit;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::dual_quaternion::{DualQuaternion, DualQuaternionTrait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::quaternion::QuaternionTrait;
use nalgebra::geometry::rotation3::Rotation3;
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::translation3::{Translation3, Translation3Trait};
use nalgebra::geometry::unit_dual_quaternion::{
    Isometry3DualQuaternionTrait, Translation3DualQuaternionTrait, UnitDualQuaternion,
    UnitDualQuaternionAngleTrait, UnitDualQuaternionTrait, UnitQuaternionDualQuaternionTrait,
};
use nalgebra::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use nalgebra_tests_utils::{
    ONE_RAW, fx, iso3, iso3t, m4, max_ulp_diff4, max_ulp_diff_q, max_ulp_diff_v3, p3, p3t, t3, t3t,
    u3, uq, uqt, v3, v3t,
};
use crate::common::{dq_err, report, udqt};
use crate::oracle;

/// The rotation of the `Isometry3` bench pose `a3` (about 0.26 rad).
fn ra() -> UnitQuaternion<Fixed> {
    uq(4234293283, 534340439, -400755330, 267170219)
}

/// The rotation of the `Isometry3` bench pose `b3` (about 1.1 rad).
fn rb() -> UnitQuaternion<Fixed> {
    uq(3689020097, -1022754606, 767065954, 1789820560)
}

/// `from_parts((1.5, -2.25, 3.75), ra)`.
fn a() -> UnitDualQuaternion<Fixed> {
    UnitDualQuaternionTrait::from_parts(t3(0x180000000, -0x240000000, 0x3c0000000), ra())
}

/// `from_parts((-0.75, 0.5, 1.25), rb)`.
fn b() -> UnitDualQuaternion<Fixed> {
    UnitDualQuaternionTrait::from_parts(t3(-0xc0000000, 0x80000000, 0x140000000), rb())
}

fn v3_err(got: Vector3<Fixed>, e: (i64, i64, i64)) -> u128 {
    max_ulp_diff_v3(got, v3t(e))
}

// --- oracle vectors

#[test]
fn test_unit_dual_quaternion_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::unit_dual_quaternion_from_parts_cases();
    while let Some(case) = cases.pop_front() {
        let (t, r, e, tol) = *case;
        let got = UnitDualQuaternionTrait::from_parts(t3t(t), uqt(r));
        worst = max(worst, report("from_parts", n, dq_err(got.dual_quaternion, e), tol));
        assert!(
            UnitDualQuaternionTrait::from_isometry(
                Isometry3Trait::from_parts(t3t(t), uqt(r)),
            ) == got,
        );
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_translation_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        worst = max(worst, report("translation", n, v3_err(udqt(x).translation().vector, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_to_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        let got = udqt(x).to_isometry();
        let e = iso3t(e);
        let err = max(
            max_ulp_diff_v3(got.translation.vector, e.translation.vector),
            max_ulp_diff_q(got.rotation.quaternion, e.rotation.quaternion),
        );
        worst = max(worst, report("to_isometry", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, tol) = *case;
        worst = max(worst, report("mul", n, dq_err((udqt(x) * udqt(y)).dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        worst = max(worst, report("inverse", n, dq_err(udqt(x).inverse().dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_div_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, tol) = *case;
        worst = max(worst, report("div", n, dq_err((udqt(x) / udqt(y)).dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, p, e, tol) = *case;
        let got = udqt(x).transform_point(p3t(p));
        worst = max(worst, report("transform_point", n, v3_err(got.coords(), e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, p, e, tol) = *case;
        let got = udqt(x).inverse_transform_point(p3t(p));
        worst = max(worst, report("inverse_transform_point", n, v3_err(got.coords(), e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (x, v, e, tol) = *case;
        let got = udqt(x).transform_vector(v3t(v));
        worst = max(worst, report("transform_vector", n, v3_err(got, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_to_homogeneous_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        let err = max_ulp_diff4(udqt(x).to_homogeneous(), m4(e));
        worst = max(worst, report("to_homogeneous", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_unit_dual_quaternion_interpolation_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::unit_dual_quaternion_nlerp_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, t, e, tol) = *case;
        let got = udqt(x).nlerp(udqt(y), fx(t));
        worst = max(worst, report("nlerp", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_sclerp_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, t, e, tol) = *case;
        let got = udqt(x).sclerp(udqt(y), fx(t));
        worst = max(worst, report("sclerp", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mixed_products_oracle() {
    let (mut worst, mut n) = (0, 0);
    let mut cases = oracle::unit_dual_quaternion_mul_unit_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (x, r, e, tol) = *case;
        let got = udqt(x).mul_unit_quaternion(uqt(r));
        worst = max(worst, report("mul_uq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_div_unit_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (x, r, e, tol) = *case;
        let got = udqt(x).div_unit_quaternion(uqt(r));
        worst = max(worst, report("div_uq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_quaternion_mul_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (r, x, e, tol) = *case;
        let got = uqt(r).mul_unit_dual_quaternion(udqt(x));
        worst = max(worst, report("uq mul_udq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_quaternion_div_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (r, x, e, tol) = *case;
        let got = uqt(r).div_unit_dual_quaternion(udqt(x));
        worst = max(worst, report("uq div_udq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_mul_translation_cases();
    while let Some(case) = cases.pop_front() {
        let (x, t, e, tol) = *case;
        let got = udqt(x).mul_translation(t3t(t));
        worst = max(worst, report("mul_translation", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_div_translation_cases();
    while let Some(case) = cases.pop_front() {
        let (x, t, e, tol) = *case;
        let got = udqt(x).div_translation(t3t(t));
        worst = max(worst, report("div_translation", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::translation_mul_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (t, x, e, tol) = *case;
        let got = t3t(t).mul_unit_dual_quaternion(udqt(x));
        worst = max(worst, report("t mul_udq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::translation_div_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (t, x, e, tol) = *case;
        let got = t3t(t).div_unit_dual_quaternion(udqt(x));
        worst = max(worst, report("t div_udq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_mul_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (x, i, e, tol) = *case;
        let got = udqt(x).mul_isometry(iso3t(i));
        worst = max(worst, report("mul_iso", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::unit_dual_quaternion_div_isometry_cases();
    while let Some(case) = cases.pop_front() {
        let (x, i, e, tol) = *case;
        let got = udqt(x).div_isometry(iso3t(i));
        worst = max(worst, report("div_iso", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_mul_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (i, x, e, tol) = *case;
        let got = iso3t(i).mul_unit_dual_quaternion(udqt(x));
        worst = max(worst, report("iso mul_udq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    let mut cases = oracle::isometry_div_unit_dual_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (i, x, e, tol) = *case;
        let got = iso3t(i).div_unit_dual_quaternion(udqt(x));
        worst = max(worst, report("iso div_udq", n, dq_err(got.dual_quaternion, e), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- construction and the `Unit` wrapper

#[test]
fn test_identity_default_one() {
    let i: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::identity();
    assert!(i.dual_quaternion == DualQuaternionTrait::identity());
    let d: UnitDualQuaternion<Fixed> = Default::default();
    let o: UnitDualQuaternion<Fixed> = One::one();
    assert!(d == i && o == i && o.is_one() && !a().is_one() && a().is_non_one());
    assert!(a() * i == a() && i * a() == a());
}

#[test]
fn test_from_parts_parts_round_trip() {
    let x = a();
    assert!(x.rotation() == ra());
    assert!(x.dual_quaternion.real == ra().quaternion);
    // translation() recovers the translation within 2 ulp.
    assert!(
        max_ulp_diff_v3(x.translation().vector, v3(0x180000000, -0x240000000, 0x3c0000000)) <= 2,
    );
    let iso = x.to_isometry();
    assert!(iso.rotation == ra() && iso.translation == x.translation());
    let back: Isometry3<Fixed> = x.into();
    assert!(back == iso);
}

#[test]
fn test_from_parts_of_integers_is_exact() {
    // Identity rotation: dual = (0, t / 2) exactly.
    let x = UnitDualQuaternionTrait::from_parts(
        t3(2 * ONE_RAW, -4 * ONE_RAW, 6 * ONE_RAW), uq(ONE_RAW, 0, 0, 0),
    );
    assert!(x.dual_quaternion.dual == nalgebra_tests_utils::qi(0, 1, -2, 3));
    assert!(x.translation() == t3(2 * ONE_RAW, -4 * ONE_RAW, 6 * ONE_RAW));
}

#[test]
fn test_from_rotation_and_conversions() {
    let r = UnitDualQuaternionTrait::from_rotation(ra());
    assert!(r.dual_quaternion == DualQuaternionTrait::from_real(ra().quaternion));
    let q: UnitDualQuaternion<Fixed> = ra().into();
    assert!(q == r);
    let m: UnitDualQuaternion<Fixed> = ra().to_rotation_matrix().into();
    assert!(
        m == UnitDualQuaternionTrait::from_rotation(
            UnitQuaternionTrait::from_rotation_matrix(ra().to_rotation_matrix()),
        ),
    );
    let rot: Rotation3<Fixed> = ra().to_rotation_matrix();
    let _m2: UnitDualQuaternion<Fixed> = rot.into();
    let t: UnitDualQuaternion<Fixed> = t3(ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW).into();
    assert!(
        t == UnitDualQuaternionTrait::from_parts(
            t3(ONE_RAW, 2 * ONE_RAW, 3 * ONE_RAW), UnitQuaternionTrait::identity(),
        ),
    );
    let iso = iso3(
        (0x180000000, -0x240000000, 0x3c0000000), (4234293283, 534340439, -400755330, 267170219),
    );
    let fi: UnitDualQuaternion<Fixed> = iso.into();
    assert!(fi == a());
    let h: Matrix4<Fixed> = a().into();
    assert!(h == a().to_homogeneous() && h == a().to_isometry().to_homogeneous());
    let s: Similarity3<Fixed> = a().into();
    assert!(s.isometry == a().to_isometry() && s.scaling == fx(ONE_RAW));
}

#[test]
fn test_unit_wrapper_methods() {
    let dq = a().dual_quaternion;
    assert!(UnitDualQuaternionTrait::new_unchecked(dq) == a());
    assert!(a().into_inner() == dq && a().dual_quaternion() == dq);
    let big = dq.scale(fx(3 * ONE_RAW));
    let n = UnitDualQuaternionTrait::new_normalize(big);
    assert!(n.dual_quaternion == big.normalize());
    assert!(n.abs_diff_eq(a(), 8));
    let t = UnitDualQuaternionTrait::try_new(big, fx(ONE_RAW)).unwrap();
    assert!(t == n);
    assert!(UnitDualQuaternionTrait::try_new(big, fx(4 * ONE_RAW)).is_none());
    let c: UnitDualQuaternion<Fixed> = a().cast();
    assert!(c == a());
}

#[test]
fn test_renormalize() {
    let mut x = UnitDualQuaternion { dual_quaternion: a().dual_quaternion.scale(fx(2 * ONE_RAW)) };
    let n = x.renormalize();
    assert!(n.raw >= 2 * ONE_RAW - 4 && n.raw <= 2 * ONE_RAW + 4);
    assert!(x.abs_diff_eq(a(), 8));
    // One Newton step on a norm off by 2^-20.
    let mut y = UnitDualQuaternion {
        dual_quaternion: a().dual_quaternion.scale(fx(ONE_RAW + ONE_RAW / 1048576)),
    };
    y.renormalize_fast();
    assert!(y.abs_diff_eq(a(), 16));
}

// --- conjugate, inverse, composition

#[test]
fn test_conjugate() {
    let c = a().conjugate();
    assert!(c.dual_quaternion == a().dual_quaternion.conjugate());
    let mut m = a();
    m.conjugate_mut();
    assert!(m == c);
}

#[test]
fn test_inverse() {
    let inv = a().inverse();
    assert!(inv.dual_quaternion.real == a().dual_quaternion.real.conjugate());
    let one: UnitDualQuaternion<Fixed> = One::one();
    assert!((a() * inv).abs_diff_eq(one, 8) && (inv * a()).abs_diff_eq(one, 8));
    // For a unit dual quaternion the inverse is the conjugate (up to rounding).
    assert!(inv.dual_quaternion.dual.abs_diff_eq(a().dual_quaternion.dual.conjugate(), 8));
    let mut m = a();
    m.inverse_mut();
    assert!(m == inv);
}

#[test]
fn test_composition_matches_the_isometries() {
    let p = (a() * b()).to_isometry();
    let e = a().to_isometry() * b().to_isometry();
    assert!(max_ulp_diff_v3(p.translation.vector, e.translation.vector) <= 8);
    assert!(p.rotation.abs_diff_eq(e.rotation, 4));
    assert!((a() / b()).abs_diff_eq(a() * b().inverse(), 0));
    let x = a().isometry_to(b());
    assert!((x * a()).abs_diff_eq(b(), 8));
}

#[test]
fn test_neg_is_the_same_transform() {
    let n = -a();
    assert!(n.dual_quaternion == -a().dual_quaternion);
    assert!(n.abs_diff_eq(a(), 0));
    let p = p3(ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW);
    assert!(n.transform_point(p) == a().transform_point(p));
}

// --- transforms

#[test]
fn test_transforms_match_the_isometry() {
    let p = p3(0x280000000, -0x100000000, 0x40000000);
    let iso = a().to_isometry();
    let got = a().transform_point(p);
    assert!(max_ulp_diff_v3(got.coords(), iso.transform_point(p).coords()) <= 8);
    let back = a().inverse_transform_point(got);
    assert!(max_ulp_diff_v3(back.coords(), p.coords()) <= 16);
    let v = v3(0x280000000, -0x100000000, 0x40000000);
    assert!(a().transform_vector(v) == ra().transform_vector(v));
    assert!(a().inverse_transform_vector(v) == ra().inverse_transform_vector(v));
    let u = u3(0, 0, ONE_RAW);
    let tu: Unit<Vector3<Fixed>> = a().transform_unit_vector(u);
    assert!(tu.value == ra().transform_vector(u.value));
    let iu = a().inverse_transform_unit_vector(u);
    assert!(iu.value == ra().inverse_transform_vector(u.value));
}

#[test]
fn test_identity_transform_point_is_exact() {
    let i: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::identity();
    let p = p3(0x280000000, -0x100000000, 0x40000000);
    assert!(i.transform_point(p) == p && i.inverse_transform_point(p) == p);
    let t: UnitDualQuaternion<Fixed> = t3(ONE_RAW, 2 * ONE_RAW, -ONE_RAW).into();
    let moved: Point3<Fixed> = t.transform_point(p);
    assert!(moved == p3(0x380000000, ONE_RAW, 0x40000000 - ONE_RAW));
}

// --- interpolation

#[test]
fn test_lerp_nlerp() {
    assert!(a().lerp(b(), fx(0)) == a().dual_quaternion);
    assert!(a().lerp(b(), fx(ONE_RAW)) == b().dual_quaternion);
    let n = a().nlerp(b(), fx(ONE_RAW / 2));
    assert!(n.dual_quaternion == a().lerp(b(), fx(ONE_RAW / 2)).normalize());
    assert!(a().nlerp(b(), fx(0)).abs_diff_eq(a(), 4));
}

#[test]
fn test_sclerp_endpoints() {
    assert!(a().sclerp(b(), fx(0)).abs_diff_eq(a(), 64));
    assert!(a().sclerp(b(), fx(ONE_RAW)).abs_diff_eq(b(), 64));
    // Shortest path: interpolating towards `-b` gives the same transforms.
    assert!(a().sclerp(-b(), fx(ONE_RAW / 2)).abs_diff_eq(a().sclerp(b(), fx(ONE_RAW / 2)), 64));
}

#[test]
fn test_sclerp_pure_translation_is_linear() {
    // Equal rotations: the translations are linearly interpolated, the rotation kept.
    let x: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::from_parts(t3(0, 0, 0), ra());
    let y: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::from_parts(
        t3(4 * ONE_RAW, -2 * ONE_RAW, 0), ra(),
    );
    let h = x.sclerp(y, fx(ONE_RAW / 4));
    assert!(h.rotation() == ra());
    assert!(max_ulp_diff_v3(h.translation().vector, v3(ONE_RAW, -ONE_RAW / 2, 0)) <= 4);
}

#[test]
fn test_sclerp_screw_about_z() {
    // A quarter turn about z with a translation of 2 along z: halfway is an eighth of a turn and
    // a translation of 1 along z.
    let s = 3037000499; // sqrt(1/2)
    let x: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::identity();
    let y = UnitDualQuaternionTrait::from_parts(t3(0, 0, 2 * ONE_RAW), uq(s, 0, 0, s));
    let h = x.sclerp(y, fx(ONE_RAW / 2));
    let e = UnitDualQuaternionTrait::from_parts(
        t3(0, 0, ONE_RAW), uq(3968032378, 0, 0, 1643612827),
    );
    assert!(h.abs_diff_eq(e, 64));
}

#[test]
fn test_try_sclerp_orthogonal_rotations_is_none() {
    let x: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::identity();
    // A half turn about x: real parts (1, 0, 0, 0) and (0, 1, 0, 0) are orthogonal.
    let y = UnitDualQuaternionTrait::from_rotation(uq(0, ONE_RAW, 0, 0));
    assert!(x.try_sclerp(y, fx(ONE_RAW / 2), fx(1)).is_none());
}

#[test]
#[should_panic(expected: 'nalgebra: ambiguous sclerp')]
fn test_sclerp_orthogonal_rotations_panics() {
    let x: UnitDualQuaternion<Fixed> = UnitDualQuaternionTrait::identity();
    let y = UnitDualQuaternionTrait::from_rotation(uq(0, ONE_RAW, 0, 0));
    let _ = x.sclerp(y, fx(ONE_RAW / 2));
}

// --- comparisons, hash

#[test]
fn test_approx_comparisons() {
    let x = a();
    assert!(x.abs_diff_eq(-x, 0) && x.relative_eq(-x, 0, fx(0)) && x.ulps_eq(-x, 0, 0));
    assert!(!x.abs_diff_eq(b(), 1000) && !x.relative_eq(b(), 0, fx(1)) && !x.ulps_eq(b(), 0, 1000));
}

#[test]
fn test_hash_follows_equality() {
    let h1 = PoseidonTrait::new().update_with(a()).finalize();
    let h2 = PoseidonTrait::new().update_with(a()).finalize();
    let h3 = PoseidonTrait::new().update_with(b()).finalize();
    assert!(h1 == h2 && h1 != h3);
}

// --- heterogeneous products, exact relations

#[test]
fn test_mixed_products_match_the_general_product() {
    let r = rb();
    assert!(a().mul_unit_quaternion(r) == a() * UnitDualQuaternionTrait::from_rotation(r));
    assert!(
        a().div_unit_quaternion(r) == a() * UnitDualQuaternionTrait::from_rotation(r.inverse()),
    );
    assert!(r.mul_unit_dual_quaternion(a()) == UnitDualQuaternionTrait::from_rotation(r) * a());
    assert!(
        r.div_unit_dual_quaternion(a()) == UnitDualQuaternionTrait::from_rotation(r)
            * a().inverse(),
    );
    let t: Translation3<Fixed> = t3(ONE_RAW, -2 * ONE_RAW, ONE_RAW / 2);
    let ft: UnitDualQuaternion<Fixed> = t.into();
    assert!(a().mul_translation(t).abs_diff_eq(a() * ft, 1));
    assert!(t.mul_unit_dual_quaternion(a()).abs_diff_eq(ft * a(), 1));
    assert!(a().div_translation(t) == a().mul_translation(t.inverse()));
    assert!(t.div_unit_dual_quaternion(a()) == t.mul_unit_dual_quaternion(a().inverse()));
    let iso = b().to_isometry();
    let fi: UnitDualQuaternion<Fixed> = iso.into();
    assert!(a().mul_isometry(iso) == a() * fi && a().div_isometry(iso) == a() / fi);
    assert!(
        iso.mul_unit_dual_quaternion(a()) == fi
            * a() && iso.div_unit_dual_quaternion(a()) == fi
            / a(),
    );
    let dq: DualQuaternion<Fixed> = b().dual_quaternion;
    assert!(a().mul_dual_quaternion(dq) == a().dual_quaternion * dq);
}
