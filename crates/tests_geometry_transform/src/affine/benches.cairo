//! Gas benchmarks of `Affine2` / `Affine3` (WP 8.4-P11a) (`bench_<group>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementations (`alt_*`, AGENTS.md rule 8).
//!
//! Expected values are the results of the library kernels themselves (computed in the baseline
//! and in every variant alike, so `net` is the cost of ONE call of the measured variant), all of
//! which are checked against upstream nalgebra in `tests.cairo`; each alternative is checked
//! against the library (bit for bit where it claims to be, against the oracle otherwise).

use fixed::Fixed;
use nalgebra::base::cg::Matrix4CgTrait;
use nalgebra::base::matrix2::Matrix2Trait;
use nalgebra::base::matrix3::{Matrix3, Matrix3Trait};
use nalgebra::base::matrix4::{Matrix4, Matrix4Trait};
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::geometry::affine2::{Affine2, Affine2Trait};
use nalgebra::geometry::affine3::{Affine3, Affine3Trait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::rotation3::{Rotation3, Rotation3Trait};
use nalgebra::geometry::transform::TransformMul;
use nalgebra::geometry::translation3::{Translation3, Translation3Trait};
use nalgebra::geometry::unit_quaternion::UnitQuaternionTrait;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{
    ONE_RAW, aff2t, aff3t, iso3t, m3, m4, max_ulp_diff3, max_ulp_diff4, p2t, p3t, t3, uqt,
};
use simba::scalar::Real;
use crate::common::report;
use crate::oracle;

/// The linear block `[[1.25, -0.5, 0.25], [0.375, 2, -0.75], [-0.125, 0.5, 1.5]]`, translation
/// `(3.5, -12.25, 40)`.
fn a() -> Affine3<Fixed> {
    aff3t(
        [
            [5368709120, -2147483648, 1073741824], [1610612736, 8589934592, -3221225472],
            [-536870912, 2147483648, 6442450944],
        ],
        (15032385536, -52613349376, 171798691840),
    )
}

/// The linear block `[[1.25, -0.5], [0.375, 2]]`, translation `(3.5, -12.25)`.
fn a2() -> Affine2<Fixed> {
    aff2t([[5368709120, -2147483648], [1610612736, 8589934592]], (15032385536, -52613349376))
}

/// The rotation of 1 radian about `(1, 2, 2) / 3`, translation `(1, 2, -1)`.
fn iso() -> Isometry3<Fixed> {
    iso3t(((ONE_RAW, 2 * ONE_RAW, -ONE_RAW), (3769188403, 686372336, 1372744673, 1372744673)))
}

fn r() -> Rotation3<Fixed> {
    uqt((3769188403, 686372336, 1372744673, 1372744673)).to_rotation_matrix()
}

fn tr() -> Translation3<Fixed> {
    t3(ONE_RAW / 3, -5 * ONE_RAW, 7 * ONE_RAW / 4)
}

// --- alternative implementations

/// The affine inverse by blocks WITHOUT the refinement: `l = m[:3, :3]⁻¹`
/// (`Matrix3::try_inverse`), the translation `l * (-t)` (one fused sum of products per
/// coordinate), the exact last row `(0, 0, 0, 1)`. The one-ulp roundings of `l` are multiplied
/// by the translation (`test_affine_try_inverse_candidates_error`).
fn alt_affine3_inverse_block(t: Affine3<Fixed>) -> Option<Matrix4<Fixed>> {
    let m = t.into_inner();
    let lin = Matrix3 {
        m11: m.m11,
        m21: m.m21,
        m31: m.m31,
        m12: m.m12,
        m22: m.m22,
        m32: m.m32,
        m13: m.m13,
        m23: m.m23,
        m33: m.m33,
    };
    let l = lin.try_inverse()?;
    let z = Real::zero();
    Some(
        Matrix4 {
            m11: l.m11,
            m21: l.m21,
            m31: l.m31,
            m41: z,
            m12: l.m12,
            m22: l.m22,
            m32: l.m32,
            m42: z,
            m13: l.m13,
            m23: l.m23,
            m33: l.m33,
            m43: z,
            m14: Real::sum_prod3(l.m11, -m.m14, l.m12, -m.m24, l.m13, -m.m34),
            m24: Real::sum_prod3(l.m21, -m.m14, l.m22, -m.m24, l.m23, -m.m34),
            m34: Real::sum_prod3(l.m31, -m.m14, l.m32, -m.m24, l.m33, -m.m34),
            m44: Real::one(),
        },
    )
}

/// The 2D affine inverse by blocks without refinement (`Matrix2::try_inverse`, then `l * (-t)`).
fn alt_affine2_inverse_block(t: Affine2<Fixed>) -> Option<Matrix3<Fixed>> {
    let m = t.into_inner();
    let l = nalgebra::base::matrix2::Matrix2 { m11: m.m11, m21: m.m21, m12: m.m12, m22: m.m22 }
        .try_inverse()?;
    let z = Real::zero();
    Some(
        Matrix3 {
            m11: l.m11,
            m21: l.m21,
            m31: z,
            m12: l.m12,
            m22: l.m22,
            m32: z,
            m13: Real::sum_prod2(l.m11, -m.m13, l.m12, -m.m23),
            m23: Real::sum_prod2(l.m21, -m.m13, l.m22, -m.m23),
            m33: Real::one(),
        },
    )
}

/// Upstream's formula: the whole homogeneous matrix inverted (`Matrix4::try_inverse`).
fn alt_affine3_inverse_full(t: Affine3<Fixed>) -> Option<Matrix4<Fixed>> {
    t.into_inner().try_inverse()
}

/// Upstream's formula in 2D (`Matrix3::try_inverse`).
fn alt_affine2_inverse_full(t: Affine2<Fixed>) -> Option<Matrix3<Fixed>> {
    t.into_inner().try_inverse()
}

/// The affine `transform_point` through the general homogeneous kernel
/// (`Matrix4::transform_point`: the normaliser `m[3, :] . (p, 1) = 1` computed and divided by).
fn alt_affine3_transform_point_homogeneous(t: Affine3<Fixed>, p: Point3<Fixed>) -> Point3<Fixed> {
    Matrix4CgTrait::transform_point(t.into_inner(), p)
}

#[test]
fn test_affine_try_inverse_candidates_error() {
    // Both alternatives leave oracle cases outside their tolerance: the plain block formula on
    // `medium` inputs (the error of `l` times `|t|`), the whole-matrix inverse on `small` ones
    // (no pre-scaling, few significant bits in the determinant). The library (refined block
    // formula) passes every case (`test_affine_oracle_try_inverse`).
    let (mut block, mut full, mut n) = (0, 0, 0);
    let mut cases = oracle::affine2_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, e, tol) = *case;
        let a = aff2t(l, t);
        let err = max_ulp_diff3(alt_affine2_inverse_block(a).unwrap(), m3(e));
        block = core::cmp::max(block, report("alt_block2", n, err, tol));
        let err = max_ulp_diff3(alt_affine2_inverse_full(a).unwrap(), m3(e));
        full = core::cmp::max(full, report("alt_full2", n, err, tol));
        n += 1;
    }
    let mut cases = oracle::affine3_try_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (l, t, e, tol) = *case;
        let a = aff3t(l, t);
        let err = max_ulp_diff4(alt_affine3_inverse_block(a).unwrap(), m4(e));
        block = core::cmp::max(block, report("alt_block3", n, err, tol));
        let err = max_ulp_diff4(alt_affine3_inverse_full(a).unwrap(), m4(e));
        full = core::cmp::max(full, report("alt_full3", n, err, tol));
        n += 1;
    }
    assert!(block > 0 && full > 0);
}

#[test]
fn test_affine_alts_agree() {
    let p = p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW));
    assert!(alt_affine3_transform_point_homogeneous(a(), p) == a().transform_point(p));
    // The three inverses agree within a few ulp on this well-scaled case.
    let lib = a().inverse().into_inner();
    assert!(max_ulp_diff4(lib, alt_affine3_inverse_block(a()).unwrap()) <= 8);
    assert!(max_ulp_diff4(lib, alt_affine3_inverse_full(a()).unwrap()) <= 8);
    assert!(lib.m41 == Real::zero() && lib.m44 == Real::one());
}

// --- benchmarks

#[test]
#[inline(never)]
fn bench_affine3_try_inverse__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let e: Fixed = black_box(Real::one());
    assert!(e == e);
}

/// The refined block formula (winner: the only candidate within the oracle tolerance).
#[test]
#[inline(never)]
fn bench_affine3_try_inverse__refined_block() {
    let t: Affine3<Fixed> = black_box(a());
    let e: Fixed = black_box(Real::one());
    assert!(t.try_inverse().unwrap().into_inner().m44 == e);
}

/// Upstream's whole-matrix inverse (loser: outside the oracle tolerance on `small` inputs).
#[test]
#[inline(never)]
fn bench_affine3_try_inverse__alt_full_matrix() {
    let t: Affine3<Fixed> = black_box(a());
    let e: Fixed = black_box(Real::one());
    assert!(alt_affine3_inverse_full(t).unwrap().m44 == e);
}

/// The block formula without refinement (loser: outside the oracle tolerance on `medium`
/// inputs).
#[test]
#[inline(never)]
fn bench_affine3_try_inverse__alt_block() {
    let t: Affine3<Fixed> = black_box(a());
    let e: Fixed = black_box(Real::one());
    assert!(alt_affine3_inverse_block(t).unwrap().m44 == e);
}

#[test]
#[inline(never)]
fn bench_affine2_try_inverse__baseline() {
    let _t: Affine2<Fixed> = black_box(a2());
    let e: Fixed = black_box(Real::one());
    assert!(e == e);
}

/// The refined block formula (winner).
#[test]
#[inline(never)]
fn bench_affine2_try_inverse__refined_block() {
    let t: Affine2<Fixed> = black_box(a2());
    let e: Fixed = black_box(Real::one());
    assert!(t.try_inverse().unwrap().into_inner().m33 == e);
}

/// Upstream's whole-matrix inverse (loser: outside the oracle tolerance on `small` inputs in 3D,
/// and dearer).
#[test]
#[inline(never)]
fn bench_affine2_try_inverse__alt_full_matrix() {
    let t: Affine2<Fixed> = black_box(a2());
    let e: Fixed = black_box(Real::one());
    assert!(alt_affine2_inverse_full(t).unwrap().m33 == e);
}

/// The block formula without refinement (loser: outside the oracle tolerance on `medium`
/// inputs).
#[test]
#[inline(never)]
fn bench_affine2_try_inverse__alt_block() {
    let t: Affine2<Fixed> = black_box(a2());
    let e: Fixed = black_box(Real::one());
    assert!(alt_affine2_inverse_block(t).unwrap().m33 == e);
}

#[test]
#[inline(never)]
fn bench_affine3_transform_point__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let _p: Point3<Fixed> = black_box(p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW)));
    let e: Point3<Fixed> = black_box(
        a().transform_point(p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW))),
    );
    assert!(e == e);
}

/// No normalizer (winner, upstream's `TAffine`).
#[test]
#[inline(never)]
fn bench_affine3_transform_point__affine() {
    let t: Affine3<Fixed> = black_box(a());
    let p: Point3<Fixed> = black_box(p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW)));
    let e: Point3<Fixed> = black_box(
        a().transform_point(p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW))),
    );
    assert!(t.transform_point(p) == e);
}

/// Through the general homogeneous kernel (same bits for an affine matrix).
#[test]
#[inline(never)]
fn bench_affine3_transform_point__alt_homogeneous() {
    let t: Affine3<Fixed> = black_box(a());
    let p: Point3<Fixed> = black_box(p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW)));
    let e: Point3<Fixed> = black_box(
        a().transform_point(p3t((ONE_RAW / 2, -3 * ONE_RAW, 7 * ONE_RAW))),
    );
    assert!(alt_affine3_transform_point_homogeneous(t, p) == e);
}

#[test]
#[inline(never)]
fn bench_affine2_transform_point__baseline() {
    let _t: Affine2<Fixed> = black_box(a2());
    let _p: Point2<Fixed> = black_box(p2t((ONE_RAW / 2, -3 * ONE_RAW)));
    let e: Point2<Fixed> = black_box(a2().transform_point(p2t((ONE_RAW / 2, -3 * ONE_RAW))));
    assert!(e == e);
}

/// No normalizer (upstream's `TAffine`).
#[test]
#[inline(never)]
fn bench_affine2_transform_point__affine() {
    let t: Affine2<Fixed> = black_box(a2());
    let p: Point2<Fixed> = black_box(p2t((ONE_RAW / 2, -3 * ONE_RAW)));
    let e: Point2<Fixed> = black_box(a2().transform_point(p2t((ONE_RAW / 2, -3 * ONE_RAW))));
    assert!(t.transform_point(p) == e);
}

#[test]
#[inline(never)]
fn bench_affine3_mul_rotation__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let _r: Rotation3<Fixed> = black_box(r());
    let e: Matrix4<Fixed> = black_box(a().mul_rotation(r()).into_inner());
    assert!(e == e);
}

/// The structured kernel (winner, bit-identical).
#[test]
#[inline(never)]
fn bench_affine3_mul_rotation__structured() {
    let t: Affine3<Fixed> = black_box(a());
    let rot: Rotation3<Fixed> = black_box(r());
    let e: Matrix4<Fixed> = black_box(a().mul_rotation(r()).into_inner());
    assert!(t.mul_rotation(rot).into_inner() == e);
}

/// Upstream's literal `matrix * rotation.to_homogeneous()` (full 4x4 product, same bits).
#[test]
#[inline(never)]
fn bench_affine3_mul_rotation__alt_full_product() {
    let t: Affine3<Fixed> = black_box(a());
    let rot: Rotation3<Fixed> = black_box(r());
    let e: Matrix4<Fixed> = black_box(a().mul_rotation(r()).into_inner());
    assert!(t.into_inner() * rot.to_homogeneous() == e);
}

#[test]
#[inline(never)]
fn bench_affine3_mul_translation__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let _tr: Translation3<Fixed> = black_box(tr());
    let e: Matrix4<Fixed> = black_box(a().mul_translation(tr()).into_inner());
    assert!(e == e);
}

/// `Matrix4::prepend_translation` (winner, bit-identical).
#[test]
#[inline(never)]
fn bench_affine3_mul_translation__structured() {
    let t: Affine3<Fixed> = black_box(a());
    let s: Translation3<Fixed> = black_box(tr());
    let e: Matrix4<Fixed> = black_box(a().mul_translation(tr()).into_inner());
    assert!(t.mul_translation(s).into_inner() == e);
}

/// Upstream's literal `matrix * translation.to_homogeneous()`.
#[test]
#[inline(never)]
fn bench_affine3_mul_translation__alt_full_product() {
    let t: Affine3<Fixed> = black_box(a());
    let s: Translation3<Fixed> = black_box(tr());
    let e: Matrix4<Fixed> = black_box(a().mul_translation(tr()).into_inner());
    assert!(t.into_inner() * s.to_homogeneous() == e);
}

#[test]
#[inline(never)]
fn bench_affine3_mul_isometry__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let _i: Isometry3<Fixed> = black_box(iso());
    let e: Matrix4<Fixed> = black_box(a().mul_isometry(iso()).into_inner());
    assert!(e == e);
}

/// The structured kernel after `to_homogeneous` (winner, bit-identical).
#[test]
#[inline(never)]
fn bench_affine3_mul_isometry__structured() {
    let t: Affine3<Fixed> = black_box(a());
    let i: Isometry3<Fixed> = black_box(iso());
    let e: Matrix4<Fixed> = black_box(a().mul_isometry(iso()).into_inner());
    assert!(t.mul_isometry(i).into_inner() == e);
}

/// Upstream's literal `matrix * isometry.to_homogeneous()`.
#[test]
#[inline(never)]
fn bench_affine3_mul_isometry__alt_full_product() {
    let t: Affine3<Fixed> = black_box(a());
    let i: Isometry3<Fixed> = black_box(iso());
    let e: Matrix4<Fixed> = black_box(a().mul_isometry(iso()).into_inner());
    assert!(t.into_inner() * i.to_homogeneous() == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_mul_affine3__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let _i: Isometry3<Fixed> = black_box(iso());
    let p: Affine3<Fixed> = iso().mul_transform(a());
    let e: Matrix4<Fixed> = black_box(p.into_inner());
    assert!(e == e);
}

/// `TransformMul::mul_transform`, the structured kernel (winner, bit-identical).
#[test]
#[inline(never)]
fn bench_isometry3_mul_affine3__structured() {
    let t: Affine3<Fixed> = black_box(a());
    let i: Isometry3<Fixed> = black_box(iso());
    let p: Affine3<Fixed> = iso().mul_transform(a());
    let e: Matrix4<Fixed> = black_box(p.into_inner());
    let got: Affine3<Fixed> = i.mul_transform(t);
    assert!(got.into_inner() == e);
}

/// Upstream's literal `isometry.to_homogeneous() * matrix`.
#[test]
#[inline(never)]
fn bench_isometry3_mul_affine3__alt_full_product() {
    let t: Affine3<Fixed> = black_box(a());
    let i: Isometry3<Fixed> = black_box(iso());
    let p: Affine3<Fixed> = iso().mul_transform(a());
    let e: Matrix4<Fixed> = black_box(p.into_inner());
    assert!(i.to_homogeneous() * t.into_inner() == e);
}

#[test]
#[inline(never)]
fn bench_affine3_mul__baseline() {
    let _t: Affine3<Fixed> = black_box(a());
    let _u: Affine3<Fixed> = black_box(a().inverse());
    let e: Matrix4<Fixed> = black_box((a() * a().inverse()).into_inner());
    assert!(e == e);
}

/// `a * b`, the full 4x4 product (upstream's formula).
#[test]
#[inline(never)]
fn bench_affine3_mul__full_product() {
    let t: Affine3<Fixed> = black_box(a());
    let u: Affine3<Fixed> = black_box(a().inverse());
    let e: Matrix4<Fixed> = black_box((a() * a().inverse()).into_inner());
    assert!((t * u).into_inner() == e);
}
