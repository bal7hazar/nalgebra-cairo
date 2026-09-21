//! Gas benchmarks of `UnitQuaternion` (`bench_unit_quaternion_<op>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementations that lost (`alt_*`), kept as
//! evidence together with the tests showing why (AGENTS.md rule 8).
//!
//! The inputs are the unit quaternions of the first `unit_quaternion_mul` oracle case (`a`, `b`),
//! the vector `(1.5, -2.25, 3.75)` and the rotation vector `(0.25, -0.1875, 0.125)`. Expected
//! values are the results of the kernels themselves, all of which are checked against upstream
//! nalgebra in `tests.cairo`.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix_test_utils::{fx, int, m3, u3t, uqt, v3t};
use crate::base::point3::Point3;
use crate::base::unit::Unit;
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::quaternion::{Quaternion, QuaternionTrait};
use crate::geometry::rotation3::{Rotation3, Rotation3Trait};
use super::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};

/// A unit quaternion of negative real part (so the sign conventions of `axis` and `angle` are
/// exercised).
fn a() -> UnitQuaternion<Fixed> {
    uqt((-1509276477, -2563574020, -2263667719, 2114881862))
}

/// Another unit quaternion.
fn b() -> UnitQuaternion<Fixed> {
    uqt((-1829744033, 968283947, 3750500405, -308145672))
}

/// `(1.5, -2.25, 3.75)`.
fn v() -> Vector3<Fixed> {
    v3t((0x180000000, -0x240000000, 0x3c0000000))
}

/// A small rotation vector `(0.25, -0.1875, 0.125)`.
fn w() -> Vector3<Fixed> {
    v3t((0x40000000, -0x30000000, 0x20000000))
}

/// A unit vector: `(1.5, -2.25, 3.75)` normalized.
fn axis() -> Unit<Vector3<Fixed>> {
    u3t((1393471396, -2090207096, 3483678492))
}

/// The rotation matrix of `a`.
fn rot() -> Rotation3<Fixed> {
    Rotation3 {
        matrix: m3(
            [
                [-173945351, 4188633151, -933723411], [1215906026, -848092610, -4031011726],
                [-4115587397, -427592469, -1151455223],
            ],
        ),
    }
}

// --- alternative implementations (losers)

/// `to_rotation_matrix` with the `1 - 2(j² + k²)` diagonal (two products instead of four): it
/// assumes `|q|` is EXACTLY 1, which a fixed-point unit quaternion never is.
fn alt_to_rotation_matrix_one_minus(q: UnitQuaternion<Fixed>) -> Rotation3<Fixed> {
    let Quaternion { i, j, k, w } = q.quaternion;
    let (i2, j2, k2) = (i + i, j + j, k + k);
    let one = Real::<Fixed>::ONE;
    Rotation3 {
        matrix: Matrix3 {
            m11: Real::wide_rescale(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), one), j2, j),
                    k2,
                    k,
                ),
            ),
            m21: Real::sum_prod2(i2, j, w, k2),
            m31: Real::diff_prod(i2, k, w, j2),
            m12: Real::diff_prod(i2, j, w, k2),
            m22: Real::wide_rescale(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), one), i2, i),
                    k2,
                    k,
                ),
            ),
            m32: Real::sum_prod2(w, i2, j2, k),
            m13: Real::sum_prod2(w, j2, i2, k),
            m23: Real::diff_prod(j2, k, w, i2),
            m33: Real::wide_rescale(
                Real::wide_sub_prod(
                    Real::wide_sub_prod(Real::wide_add(Real::<Fixed>::wide_zero(), one), i2, i),
                    j2,
                    j,
                ),
            ),
        },
    }
}

/// `transform_vector` as the naive sandwich `q · (0, v) · q⁻¹`: two full Hamilton products, 32
/// products instead of 15.
fn alt_transform_vector_sandwich(q: UnitQuaternion<Fixed>, x: Vector3<Fixed>) -> Vector3<Fixed> {
    let p = QuaternionTrait::from_imag(x);
    let r = q.quaternion * p * q.quaternion.conjugate();
    Vector3 { x: r.i, y: r.j, z: r.k }
}

/// `inverse_transform_vector` as it was before the conjugate's signs were folded into the
/// sandwich: three negations, then `transform_vector`. Bit for bit the same result.
fn alt_inverse_transform_vector_conjugate(
    q: UnitQuaternion<Fixed>, x: Vector3<Fixed>,
) -> Vector3<Fixed> {
    q.conjugate().transform_vector(x)
}

/// `transform_vector` through the rotation matrix: 24 + 9 products, but the matrix can be reused.
fn alt_transform_vector_via_matrix(q: UnitQuaternion<Fixed>, x: Vector3<Fixed>) -> Vector3<Fixed> {
    q.to_rotation_matrix().transform_vector(x)
}

/// `from_rotation_matrix` multiplying by the reciprocal of the denominator instead of dividing
/// three times (trace branch only, enough to measure).
fn alt_from_rotation_matrix_recip(r: Rotation3<Fixed>) -> UnitQuaternion<Fixed> {
    let m = r.matrix;
    let tr = m.m11 + m.m22 + m.m33;
    let d = Real::sqrt(Real::ONE + tr);
    let f = Real::recip(d + d);
    UnitQuaternion {
        quaternion: Quaternion {
            i: (m.m32 - m.m23) * f,
            j: (m.m13 - m.m31) * f,
            k: (m.m21 - m.m12) * f,
            w: d * Real::HALF,
        },
    }
}

/// `from_scaled_axis` normalizing the half vector first: three divisions instead of one.
fn alt_from_scaled_axis_unit_axis(axisangle: Vector3<Fixed>) -> UnitQuaternion<Fixed> {
    let h = Vector3 {
        x: axisangle.x * Real::HALF, y: axisangle.y * Real::HALF, z: axisangle.z * Real::HALF,
    };
    let n = Real::norm3(h.x, h.y, h.z);
    if n == Real::ZERO {
        return UnitQuaternionTrait::identity();
    }
    let (s, c) = Transcendental::sin_cos(n);
    let u = Vector3 { x: h.x / n, y: h.y / n, z: h.z / n };
    UnitQuaternion { quaternion: Quaternion { i: u.x * s, j: u.y * s, k: u.z * s, w: c } }
}

/// `angle` as `2·acos(|w|)`: cheaper than `2·atan2(|imag|, |w|)` but it loses half of the digits
/// near `0` and `π`, where `acos` has an unbounded derivative.
fn alt_angle_acos(q: UnitQuaternion<Fixed>) -> Fixed {
    let half = Transcendental::acos(Real::clamp(Real::abs(q.quaternion.w), Real::ZERO, Real::ONE));
    half + half
}

/// `scaled_axis` with the common factor `angle / |imag|` computed once: one division instead of
/// three, at the price of rounding the factor first.
fn alt_scaled_axis_factor(q: UnitQuaternion<Fixed>) -> Vector3<Fixed> {
    let c = q.quaternion;
    let n = Real::norm3(c.i, c.j, c.k);
    if n == Real::ZERO {
        return Vector3Trait::zeros();
    }
    let half = Transcendental::atan2(n, Real::abs(c.w));
    let f = (half + half) / n;
    if Real::is_negative(c.w) {
        Vector3 { x: -c.i * f, y: -c.j * f, z: -c.k * f }
    } else {
        Vector3 { x: c.i * f, y: c.j * f, z: c.k * f }
    }
}

// --- why the alternatives lost

/// The `1 - 2(j² + k²)` diagonal differs from upstream's `w² + i² - j² - k²` by the norm
/// defect of the quaternion: a few ulp for a freshly built rotation, more for one that has drifted.
#[test]
fn test_to_rotation_matrix_alt_one_minus_differs_by_the_norm_defect() {
    let got = alt_to_rotation_matrix_one_minus(a());
    let exact = a().to_rotation_matrix();
    assert!(got.abs_diff_eq(exact, 2));
    // A quaternion scaled by 1 + 2^-12 is still "unit" for the wrapper, but the two diagonals then
    // disagree by about 2^-11 (2 097 152 ulp): upstream's form stays exactly `|q|²` times the true
    // rotation matrix (so `renormalize` on the quaternion fixes it), the `1 - 2(...)` form mixes a
    // scaled off-diagonal with an unscaled diagonal and is no longer a similarity.
    let s = Real::<Fixed>::ONE + fx(0x100000);
    let drifted = UnitQuaternionTrait::new_unchecked(a().quaternion.scale(s));
    let approx = alt_to_rotation_matrix_one_minus(drifted);
    let good = drifted.to_rotation_matrix();
    assert!(!approx.abs_diff_eq(good, 0x100000));
    assert!(good.matrix.abs_diff_eq(exact.matrix.scale(s * s), 8));
    assert!(!approx.matrix.abs_diff_eq(exact.matrix.scale(s * s), 0x100000));
}

#[test]
fn test_transform_vector_alts_agree() {
    // The sandwich and the matrix path compute the same rotation, less accurately (more roundings)
    // and dearer.
    let exact = a().transform_vector(v());
    assert!(alt_transform_vector_sandwich(a(), v()).abs_diff_eq(exact, 8));
    assert!(alt_transform_vector_via_matrix(a(), v()).abs_diff_eq(exact, 8));
}

/// Folding the conjugate's signs into the operand order of the cross products (`(-u) × v = v ×
/// u`)
/// changes no exact product, hence no bit: the fused `inverse_transform_vector` and `conj_mul`
/// equal their conjugate-first formulations exactly.
#[test]
fn test_inverse_transform_vector_fused_matches_conjugate_then_transform() {
    assert!(a().inverse_transform_vector(v()) == alt_inverse_transform_vector_conjugate(a(), v()));
    assert!(b().inverse_transform_vector(w()) == alt_inverse_transform_vector_conjugate(b(), w()));
    assert!(a().inverse_transform_vector(w()) == alt_inverse_transform_vector_conjugate(a(), w()));
    let p = Point3 { x: v().x, y: v().y, z: v().z };
    let c = a().conjugate().transform_point(p);
    assert!(a().inverse_transform_point(p) == c);
    assert!(a().conj_mul(b()) == a().conjugate() * b());
    assert!(b().conj_mul(a()) == b().inverse() * a());
}

/// The reciprocal of the denominator costs one division instead of three but rounds the factor
/// first: the components come out within a few ulp of the exactly divided ones. `denom >= 2` here,
/// so the loss stays small — the three divisions are kept because they are exact and the branch
/// already pays a square root.
#[test]
fn test_from_rotation_matrix_alt_recip_is_within_a_few_ulp() {
    // The trace branch (a small rotation), the one the alternative implements.
    let small = UnitQuaternionAngleTrait::from_axis_angle(axis(), fx(0x40000000))
        .to_rotation_matrix();
    let exact = UnitQuaternionTrait::from_rotation_matrix(small);
    let approx = alt_from_rotation_matrix_recip(small);
    assert!(approx.abs_diff_eq(exact, 4));
    assert!(exact.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    assert!(approx.quaternion.norm().abs_diff_eq(Real::ONE, 4));
}

/// Normalizing the half vector first (three divisions) gives the same bits as the shipped single
/// division on this input, and at most 1 ulp elsewhere: the two extra divisions buy nothing.
#[test]
fn test_from_scaled_axis_alt_unit_axis_is_barely_more_accurate() {
    let exact = UnitQuaternionAngleTrait::from_scaled_axis(w());
    let alt = alt_from_scaled_axis_unit_axis(w());
    assert!(alt.abs_diff_eq(exact, 1));
    assert!(alt.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    // A larger rotation vector (|v| = 3): still within 2 ulp.
    let big = v3t((0x200000000, -0x180000000, 0x100000000));
    assert!(
        alt_from_scaled_axis_unit_axis(big)
            .abs_diff_eq(UnitQuaternionAngleTrait::from_scaled_axis(big), 2),
    );
}

/// `2·acos(|w|)` collapses near the identity: at an angle of 2^-20 rad it is off by 4 000 ulp,
/// where `2·atan2` is exact to the last bits.
#[test]
fn test_angle_alt_acos_loses_precision_near_zero() {
    let tiny = UnitQuaternionAngleTrait::from_axis_angle(axis(), fx(0x1000));
    assert!(tiny.angle().abs_diff_eq(fx(0x1000), 4));
    assert!(!alt_angle_acos(tiny).abs_diff_eq(fx(0x1000), 4000));
    // Away from the ends both agree.
    assert!(alt_angle_acos(a()).abs_diff_eq(a().angle(), 64));
}

#[test]
fn test_scaled_axis_alt_factor_is_less_accurate() {
    let exact = a().scaled_axis();
    let alt = alt_scaled_axis_factor(a());
    assert!(alt.abs_diff_eq(exact, 8));
    assert!(!alt.abs_diff_eq(exact, 0));
}

// --- construction and parts

#[test]
#[inline(never)]
fn bench_unit_quaternion_identity__baseline() {
    let e = black_box(uqt((0x100000000, 0, 0, 0)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_identity__const() {
    let e = black_box(uqt((0x100000000, 0, 0, 0)));
    assert!(UnitQuaternionTrait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_unchecked__baseline() {
    let _q = black_box(a().quaternion);
    let e = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_unchecked__struct() {
    let q = black_box(a().quaternion);
    let e = black_box(a());
    assert!(UnitQuaternionTrait::new_unchecked(q) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_normalize__baseline() {
    let _q = black_box(QuaternionTrait::new(int(1), int(2), int(-3), int(4)));
    let e = black_box(uqt((784150157, 1568300314, -2352450472, 3136600629)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_normalize__divisions() {
    let q = black_box(QuaternionTrait::new(int(1), int(2), int(-3), int(4)));
    let e = black_box(uqt((784150157, 1568300314, -2352450472, 3136600629)));
    assert!(UnitQuaternionTrait::new_normalize(q) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_imag__baseline() {
    let _q = black_box(a());
    let e = black_box(v3t((-2563574020, -2263667719, 2114881862)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_imag__struct() {
    let q = black_box(a());
    let e = black_box(v3t((-2563574020, -2263667719, 2114881862)));
    assert!(q.imag() == e);
}

// --- conjugate, inverse, composition

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse__baseline() {
    let _q = black_box(a());
    let e = black_box(uqt((-1509276477, 2563574020, 2263667719, -2114881862)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse__conjugate() {
    let q = black_box(a());
    let e = black_box(uqt((-1509276477, 2563574020, 2263667719, -2114881862)));
    assert!(q.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let e = black_box(uqt((3349370227, -932498320, -60712365, -2520956970)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul__hamilton() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(uqt((3349370227, -932498320, -60712365, -2520956970)));
    assert!(q * r == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_conj_mul__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let e = black_box(uqt((-2063404847, 251977103, -2575182929, 2737525330)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_conj_mul__fused() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(uqt((-2063404847, 251977103, -2575182929, 2737525330)));
    assert!(q.conj_mul(r) == e);
}

/// The formulation `conj_mul` replaces: `self.inverse() * other` (three negations, then the
/// Hamilton product).
#[test]
#[inline(never)]
fn bench_unit_quaternion_conj_mul__alt_conjugate_then_mul() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(uqt((-2063404847, 251977103, -2575182929, 2737525330)));
    assert!(q.inverse() * r == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_to__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let e = black_box(uqt((-2063404847, -3116768394, -1989449985, -718990994)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_to__conjugate_product() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(uqt((-2063404847, -3116768394, -1989449985, -718990994)));
    assert!(q.rotation_to(r) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_dot__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let e = black_box(fx(-2063404847));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_dot__fused() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(fx(-2063404847));
    assert!(q.dot(r) == e);
}

// --- renormalization

#[test]
#[inline(never)]
fn bench_unit_quaternion_renormalize__baseline() {
    let _q = black_box(a());
    let e = black_box(uqt((-1509276478, -2563574021, -2263667720, 2114881862)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_renormalize__exact() {
    let q = black_box(a());
    let e = black_box(uqt((-1509276478, -2563574021, -2263667720, 2114881862)));
    assert!(q.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_renormalize_fast__baseline() {
    let _q = black_box(a());
    let e = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_renormalize_fast__newton() {
    let q = black_box(a());
    let e = black_box(a());
    assert!(q.renormalize_fast() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_renormalize_fast__alt_exact() {
    let q = black_box(a());
    let e = black_box(uqt((-1509276478, -2563574021, -2263667720, 2114881862)));
    assert!(q.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_abs_diff_eq__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_abs_diff_eq__components() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(false);
    assert!(q.abs_diff_eq(r, 4) == e);
}

// --- axis and angle

#[test]
#[inline(never)]
fn bench_unit_quaternion_axis__baseline() {
    let _q = black_box(a());
    let e = black_box(u3t((2738208059, 2417871746, -2258950401)));
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_axis__try_new() {
    let q = black_box(a());
    let e = black_box(u3t((2738208059, 2417871746, -2258950401)));
    assert!(q.axis() == Some(e));
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_angle__baseline() {
    let _q = black_box(a());
    let e = black_box(fx(10408630470));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_angle__atan2() {
    let q = black_box(a());
    let e = black_box(fx(10408630470));
    assert!(q.angle() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_angle__alt_acos() {
    let q = black_box(a());
    let e = black_box(fx(10408630470));
    assert!(alt_angle_acos(q) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_angle_to__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let e = black_box(fx(9188295436));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_angle_to__product_and_atan2() {
    let q = black_box(a());
    let r = black_box(b());
    let e = black_box(fx(9188295436));
    assert!(q.angle_to(r) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_axis__baseline() {
    let _q = black_box(a());
    let e = black_box(v3t((6635905205, 5859586765, -5474449130)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_axis__divisions() {
    let q = black_box(a());
    let e = black_box(v3t((6635905205, 5859586765, -5474449130)));
    assert!(q.scaled_axis() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_axis__alt_factor() {
    let q = black_box(a());
    let e = black_box(v3t((6635905207, 5859586766, -5474449129)));
    assert!(alt_scaled_axis_factor(q) == e);
}

// --- rotation matrix conversions

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_rotation_matrix__baseline() {
    let _q = black_box(a());
    let e = black_box(rot());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_rotation_matrix__fused() {
    let q = black_box(a());
    let e = black_box(rot());
    assert!(q.to_rotation_matrix() == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_rotation_matrix__alt_one_minus() {
    let q = black_box(a());
    let e = black_box(
        Rotation3 {
            matrix: m3(
                [
                    [-173945351, 4188633151, -933723411], [1215906026, -848092609, -4031011726],
                    [-4115587397, -427592469, -1151455223],
                ],
            ),
        },
    );
    assert!(alt_to_rotation_matrix_one_minus(q) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_rotation_matrix__baseline() {
    let _r = black_box(rot());
    let e = black_box(uqt((1509276477, 2563574020, 2263667718, -2114881863)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_rotation_matrix__shepperd() {
    let r = black_box(rot());
    let e = black_box(uqt((1509276477, 2563574020, 2263667718, -2114881863)));
    assert!(UnitQuaternionTrait::from_rotation_matrix(r) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_homogeneous__baseline() {
    let _q = black_box(a());
    let e = black_box(fx(-173945351));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_homogeneous__matrix4() {
    let q = black_box(a());
    let e = black_box(fx(-173945351));
    let h = q.to_homogeneous();
    assert!(h.m11 == e);
    assert!(h.m44 == Real::ONE);
}

// --- transforms

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector__baseline() {
    let _q = black_box(a());
    let _x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector__expanded() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    assert!(q.transform_vector(x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector__alt_via_matrix() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226561, -9529255127)));
    assert!(alt_transform_vector_via_matrix(q, x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector__alt_sandwich() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    assert!(alt_transform_vector_sandwich(q, x) == e);
}

/// Two vectors with the same rotation: the matrix conversion already wins here.
#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector_x2__baseline() {
    let _q = black_box(a());
    let _x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector_x2__expanded() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    let r1 = q.transform_vector(x);
    let r2 = q.transform_vector(Vector3 { x: x.y, y: x.z, z: x.x });
    assert!(r1 == e);
    assert!(r2 != r1);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector_x2__via_matrix() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226561, -9529255127)));
    let m = q.to_rotation_matrix();
    let r1 = m.transform_vector(x);
    let r2 = m.transform_vector(Vector3 { x: x.y, y: x.z, z: x.x });
    assert!(r1 == e);
    assert!(r2 != r1);
}

/// Three vectors with the same rotation: the matrix is clearly ahead.
#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector_x3__baseline() {
    let _q = black_box(a());
    let _x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector_x3__expanded() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226564, -9529255125)));
    let r1 = q.transform_vector(x);
    let r2 = q.transform_vector(Vector3 { x: x.y, y: x.z, z: x.x });
    let r3 = q.transform_vector(Vector3 { x: x.z, y: x.x, z: x.y });
    assert!(r1 == e);
    assert!(r2 != r3);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_vector_x3__via_matrix() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226561, -9529255127)));
    let m = q.to_rotation_matrix();
    let r1 = m.transform_vector(x);
    let r2 = m.transform_vector(Vector3 { x: x.y, y: x.z, z: x.x });
    let r3 = m.transform_vector(Vector3 { x: x.z, y: x.x, z: x.y });
    assert!(r1 == e);
    assert!(r2 != r3);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse_transform_vector__baseline() {
    let _q = black_box(a());
    let _x = black_box(v());
    let e = black_box(v3t((-18430159324, 6587686339, 3351234183)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse_transform_vector__folded() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-18430159324, 6587686339, 3351234183)));
    assert!(q.inverse_transform_vector(x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse_transform_vector__alt_conjugate_then_transform() {
    let q = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-18430159324, 6587686339, 3351234183)));
    assert!(alt_inverse_transform_vector_conjugate(q, x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_point__baseline() {
    let _q = black_box(a());
    let _p = black_box(Point3 { x: v().x, y: v().y, z: v().z });
    let e = black_box(Point3 { x: fx(-13186805408), y: fx(-11384226564), z: fx(-9529255125) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_point__expanded() {
    let q = black_box(a());
    let p = black_box(Point3 { x: v().x, y: v().y, z: v().z });
    let e = black_box(Point3 { x: fx(-13186805408), y: fx(-11384226564), z: fx(-9529255125) });
    assert!(q.transform_point(p) == e);
}

// --- rotation between two vectors

#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_between__baseline() {
    let _x = black_box(v());
    let _y = black_box(w());
    let e = black_box(uqt((4089636155, 611444112, 1087011755, 407629408)));
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_between__algebraic() {
    let x = black_box(v());
    let y = black_box(w());
    let e = black_box(uqt((4089636155, 611444112, 1087011755, 407629408)));
    assert!(UnitQuaternionTrait::rotation_between(x, y) == Some(e));
}

/// Upstream's trigonometric path (`acos` then `from_axis_angle`), which `scaled_rotation_between`
/// still needs in order to scale the angle. It happens to give the same bits as the algebraic form
/// on this input.
#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_between__alt_axis_angle() {
    let x = black_box(v());
    let y = black_box(w());
    let e = black_box(uqt((4089636155, 611444112, 1087011755, 407629408)));
    assert!(UnitQuaternionAngleTrait::scaled_rotation_between(x, y, Real::ONE) == Some(e));
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_rotation_between__baseline() {
    let _x = black_box(v());
    let _y = black_box(w());
    let e = black_box(uqt((4243324028, 309442838, 550120601, 206295225)));
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_rotation_between__acos() {
    let x = black_box(v());
    let y = black_box(w());
    let e = black_box(uqt((4243324028, 309442838, 550120601, 206295225)));
    assert!(UnitQuaternionAngleTrait::scaled_rotation_between(x, y, Real::HALF) == Some(e));
}

// --- axis-angle and Euler-angle constructors

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_axis_angle__baseline() {
    let _u = black_box(axis());
    let _t = black_box(fx(0x180000000));
    let e = black_box(uqt((3142579763, 949844114, -1424766174, 2374610287)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_axis_angle__sin_cos() {
    let u = black_box(axis());
    let t = black_box(fx(0x180000000));
    let e = black_box(uqt((3142579763, 949844114, -1424766174, 2374610287)));
    assert!(UnitQuaternionAngleTrait::from_axis_angle(u, t) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_scaled_axis__baseline() {
    let _x = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_scaled_axis__one_division() {
    let x = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(UnitQuaternionAngleTrait::from_scaled_axis(x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_scaled_axis__alt_unit_axis() {
    let x = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(alt_from_scaled_axis_unit_axis(x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_euler_angles__baseline() {
    let _r = black_box(fx(0x40000000));
    let _p = black_box(fx(-0x30000000));
    let _y = black_box(fx(0x20000000));
    let e = black_box(uqt((4231328319, 556998235, -364849220, 315028144)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_euler_angles__three_sin_cos() {
    let r = black_box(fx(0x40000000));
    let p = black_box(fx(-0x30000000));
    let y = black_box(fx(0x20000000));
    let e = black_box(uqt((4231328319, 556998235, -364849220, 315028144)));
    assert!(UnitQuaternionAngleTrait::from_euler_angles(r, p, y) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_euler_angles__baseline() {
    let _q = black_box(a());
    let e = black_box(fx(-11965891172));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_euler_angles__five_entries() {
    let q = black_box(a());
    let e = black_box(fx(-11965891172));
    let (roll, pitch, yaw) = q.euler_angles();
    assert!(roll == e);
    assert!(pitch == fx(5500842971));
    assert!(yaw == fx(7356808382));
}

// --- interpolation, powers, integration

#[test]
#[inline(never)]
fn bench_unit_quaternion_nlerp__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let _t = black_box(fx(0x40000000));
    let e = black_box(uqt((-2383027037, -2519790274, -1139680148, 2262678139)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_nlerp__lerp_normalize() {
    let q = black_box(a());
    let r = black_box(b());
    let t = black_box(fx(0x40000000));
    let e = black_box(uqt((-2383027037, -2519790274, -1139680148, 2262678139)));
    assert!(q.nlerp(r, t) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_slerp__baseline() {
    let _q = black_box(a());
    let _r = black_box(b());
    let _t = black_box(fx(0x40000000));
    let e = black_box(uqt((-685896197, -2393123540, -2985529559, 1826434633)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_slerp__acos_sin() {
    let q = black_box(a());
    let r = black_box(b());
    let t = black_box(fx(0x40000000));
    let e = black_box(uqt((-685896197, -2393123540, -2985529559, 1826434633)));
    assert!(q.slerp(r, t) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_powf__baseline() {
    let _q = black_box(a());
    let _n = black_box(Real::<Fixed>::HALF);
    let e = black_box(uqt((3530512512, 1559329776, 1376907571, -1286406493)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_powf__axis_angle() {
    let q = black_box(a());
    let n = black_box(Real::<Fixed>::HALF);
    let e = black_box(uqt((3530512512, 1559329776, 1376907571, -1286406493)));
    assert!(q.powf(n) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_append_axisangle_linearized__baseline() {
    let _q = black_box(a());
    let _x = black_box(w());
    let e = black_box(uqt((-1511968454, -2770073696, -2511442522, 1476497089)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_append_axisangle_linearized__reduced_product() {
    let q = black_box(a());
    let x = black_box(w());
    let e = black_box(uqt((-1511968454, -2770073696, -2511442522, 1476497089)));
    assert!(q.append_axisangle_linearized(x) == e);
}

/// The exact alternative to the linearized update, for reference: a `from_scaled_axis` (one
/// `sin_cos`) and a full Hamilton product.
#[test]
#[inline(never)]
fn bench_unit_quaternion_append_axisangle_linearized__alt_exact() {
    let q = black_box(a());
    let x = black_box(w());
    let e = black_box(uqt((-1511794594, -2771652604, -2513444042, 1470293193)));
    assert!(UnitQuaternionAngleTrait::from_scaled_axis(x) * q == e);
}
