//! Gas benchmarks of the WP 8.4-P08 completion of `UnitQuaternion`
//! (`bench_unit_quaternion_<op>__<variant>`, net = raw - `baseline` of the group), and the
//! alternative implementations that lost (`alt_*` variants, upstream's literal formulations),
//! kept as evidence together with the tests showing why (AGENTS.md rule 8; `ext_tests.cairo`).
//!
//! The inputs are the unit quaternions `a`, `b` of `benches.cairo`, the vector `(1.5, -2.25,
//! 3.75)`, the isometry `(v, b)`, the similarity `((v, b), 2)` and a well-conditioned matrix.
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `ext_tests.cairo`.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix3::Matrix3;
use crate::base::matrix_test_utils::{fx, iso3t, m3i, qt, sim3t, u3t, uqt, v3t};
use crate::base::unit::Unit;
use crate::base::vector3::Vector3;
use crate::geometry::isometry3::{Isometry3, Isometry3Trait};
use crate::geometry::rotation3::Rotation3;
use crate::geometry::similarity3::{Similarity3, Similarity3Trait};
use crate::geometry::translation3::Translation3;
use super::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait};

/// A unit quaternion of negative real part (the `a` of `benches.cairo`).
fn a() -> UnitQuaternion<Fixed> {
    uqt((-1509276477, -2563574020, -2263667719, 2114881862))
}

/// Another unit quaternion (the `b` of `benches.cairo`).
fn b() -> UnitQuaternion<Fixed> {
    uqt((-1829744033, 968283947, 3750500405, -308145672))
}

/// Two rotations close to `a` (for `mean_of`).
fn c() -> UnitQuaternion<Fixed> {
    uqt((-1459276477, -2563574020, -2293667719, 2114881862))
}

fn d() -> UnitQuaternion<Fixed> {
    uqt((-1509276477, -2533574020, -2263667719, 2144881862))
}

/// The rotation matrix of `b`.
fn rot_b() -> Rotation3<Fixed> {
    b().to_rotation_matrix()
}

/// `(1.5, -2.25, 3.75)`.
fn v() -> Vector3<Fixed> {
    v3t((0x180000000, -0x240000000, 0x3c0000000))
}

/// `(0, 1, 0.25)`.
fn up() -> Vector3<Fixed> {
    v3t((0, 0x100000000, 0x40000000))
}

/// A small rotation vector `(0.25, -0.1875, 0.125)`.
fn w() -> Vector3<Fixed> {
    v3t((0x40000000, -0x30000000, 0x20000000))
}

/// `v` normalised.
fn axis() -> Unit<Vector3<Fixed>> {
    u3t((1393471396, -2090207096, 3483678492))
}

/// `(0.6, 0.8, 0)`.
fn axis2() -> Unit<Vector3<Fixed>> {
    u3t((2576980378, 3435973837, 0))
}

fn iso() -> Isometry3<Fixed> {
    Isometry3 { rotation: b(), translation: Translation3 { vector: v() } }
}

fn sim() -> Similarity3<Fixed> {
    Similarity3 { isometry: iso(), scaling: fx(0x200000000) }
}

/// A well-conditioned matrix.
fn mat() -> Matrix3<Fixed> {
    m3i([[2, -1, 0], [1, 2, -1], [0, 1, 3]])
}

/// The columns of `b`'s rotation matrix.
fn basis() -> [Vector3<Fixed>; 3] {
    let m = rot_b().matrix;
    [
        Vector3 { x: m.m11, y: m.m21, z: m.m31 }, Vector3 { x: m.m12, y: m.m22, z: m.m32 },
        Vector3 { x: m.m13, y: m.m23, z: m.m33 },
    ]
}


// --- div

#[test]
#[inline(never)]
fn bench_unit_quaternion_div__baseline() {
    let _a = black_box(a());
    let _b = black_box(b());
    let e = black_box(uqt((-2063404847, 3116768393, 1989449984, 718990993)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div__fused() {
    let a = black_box(a());
    let b = black_box(b());
    let e = black_box(uqt((-2063404847, 3116768393, 1989449984, 718990993)));
    assert!(a / b == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div__alt_inverse_then_mul() {
    let a = black_box(a());
    let b = black_box(b());
    let e = black_box(uqt((-2063404847, 3116768393, 1989449984, 718990993)));
    assert!(a * b.inverse() == e);
}

// --- mul_rotation

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_rotation__baseline() {
    let _a = black_box(a());
    let _r = black_box(rot_b());
    let e = black_box(uqt((3349370227, -932498320, -60712365, -2520956970)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_rotation__convert_then_mul() {
    let a = black_box(a());
    let r = black_box(rot_b());
    let e = black_box(uqt((3349370227, -932498320, -60712365, -2520956970)));
    assert!(a.mul_rotation(r) == e);
}

// --- div_rotation

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_rotation__baseline() {
    let _a = black_box(a());
    let _r = black_box(rot_b());
    let e = black_box(uqt((-2063404847, 3116768393, 1989449984, 718990993)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_rotation__convert_then_mul_conj() {
    let a = black_box(a());
    let r = black_box(rot_b());
    let e = black_box(uqt((-2063404847, 3116768393, 1989449984, 718990993)));
    assert!(a.div_rotation(r) == e);
}

// --- mul_translation

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_translation__baseline() {
    let _a = black_box(a());
    let _t = black_box(Translation3 { vector: v() });
    let e = black_box(
        iso3t(
            (
                (-13186805408, -11384226564, -9529255125),
                (-1509276477, -2563574020, -2263667719, 2114881862),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_translation__rotate() {
    let a = black_box(a());
    let t = black_box(Translation3 { vector: v() });
    let e = black_box(
        iso3t(
            (
                (-13186805408, -11384226564, -9529255125),
                (-1509276477, -2563574020, -2263667719, 2114881862),
            ),
        ),
    );
    assert!(a.mul_translation(t) == e);
}

// --- mul_isometry

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_isometry__baseline() {
    let _a = black_box(a());
    let _i = black_box(iso());
    let e = black_box(
        iso3t(
            (
                (-13186805408, -11384226564, -9529255125),
                (3349370227, -932498320, -60712365, -2520956970),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_isometry__compose() {
    let a = black_box(a());
    let i = black_box(iso());
    let e = black_box(
        iso3t(
            (
                (-13186805408, -11384226564, -9529255125),
                (3349370227, -932498320, -60712365, -2520956970),
            ),
        ),
    );
    assert!(a.mul_isometry(i) == e);
}

// --- div_isometry

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_isometry__baseline() {
    let _a = black_box(a());
    let _i = black_box(iso());
    let e = black_box(
        iso3t(
            (
                (7989423761, -18078854477, -1903492270),
                (-2063404847, 3116768393, 1989449984, 718990993),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_isometry__fused() {
    let a = black_box(a());
    let i = black_box(iso());
    let e = black_box(
        iso3t(
            (
                (7989423761, -18078854477, -1903492270),
                (-2063404847, 3116768393, 1989449984, 718990993),
            ),
        ),
    );
    assert!(a.div_isometry(i) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_isometry__alt_inverse_then_mul() {
    let a = black_box(a());
    let i = black_box(iso());
    let e = black_box(
        iso3t(
            (
                (7989423767, -18078854485, -1903492250),
                (-2063404847, 3116768393, 1989449984, 718990993),
            ),
        ),
    );
    assert!(a.mul_isometry(i.inverse()) == e);
}

// --- mul_similarity

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_similarity__baseline() {
    let _a = black_box(a());
    let _s = black_box(sim());
    let e = black_box(
        sim3t(
            (
                (-13186805408, -11384226564, -9529255125),
                (3349370227, -932498320, -60712365, -2520956970),
                8589934592,
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mul_similarity__compose() {
    let a = black_box(a());
    let s = black_box(sim());
    let e = black_box(
        sim3t(
            (
                (-13186805408, -11384226564, -9529255125),
                (3349370227, -932498320, -60712365, -2520956970),
                8589934592,
            ),
        ),
    );
    assert!(a.mul_similarity(s) == e);
}

// --- div_similarity

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_similarity__baseline() {
    let _a = black_box(a());
    let _s = black_box(sim());
    let e = black_box(
        sim3t(
            (
                (3994711880, -9039427238, -951746135),
                (-2063404847, 3116768393, 1989449984, 718990993),
                2147483648,
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_similarity__fused() {
    let a = black_box(a());
    let s = black_box(sim());
    let e = black_box(
        sim3t(
            (
                (3994711880, -9039427238, -951746135),
                (-2063404847, 3116768393, 1989449984, 718990993),
                2147483648,
            ),
        ),
    );
    assert!(a.div_similarity(s) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_div_similarity__alt_inverse_then_mul() {
    let a = black_box(a());
    let s = black_box(sim());
    let e = black_box(
        sim3t(
            (
                (3994711883, -9039427244, -951746125),
                (-2063404847, 3116768393, 1989449984, 718990993),
                2147483648,
            ),
        ),
    );
    assert!(a.mul_similarity(s.inverse()) == e);
}

// --- rotation_between_axis

#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_between_axis__baseline() {
    let _u = black_box(axis());
    let _w = black_box(axis2());
    let e = black_box(uqt((2725416998, -2195962704, 1646972028, 1866568298)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_rotation_between_axis__algebraic() {
    let u = black_box(axis());
    let w = black_box(axis2());
    let e = black_box(uqt((2725416998, -2195962704, 1646972028, 1866568298)));
    assert!(UnitQuaternionTrait::rotation_between_axis(u, w).unwrap() == e);
}

// --- scaled_rotation_between_axis

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_rotation_between_axis__baseline() {
    let _u = black_box(axis());
    let _w = black_box(axis2());
    let _s = black_box(fx(0x80000000));
    let e = black_box(uqt((3882803172, -1214533365, 910900022, 1032353359)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_scaled_rotation_between_axis__acos() {
    let u = black_box(axis());
    let w = black_box(axis2());
    let s = black_box(fx(0x80000000));
    let e = black_box(uqt((3882803172, -1214533365, 910900022, 1032353359)));
    assert!(UnitQuaternionAngleTrait::scaled_rotation_between_axis(u, w, s).unwrap() == e);
}

// --- face_towards

#[test]
#[inline(never)]
fn bench_unit_quaternion_face_towards__baseline() {
    let _d = black_box(v());
    let _u = black_box(up());
    let e = black_box(uqt((4087096511, 1096004728, 735529113, -12519501)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_face_towards__rotation3() {
    let d = black_box(v());
    let u = black_box(up());
    let e = black_box(uqt((4087096511, 1096004728, 735529113, -12519501)));
    assert!(UnitQuaternionTrait::face_towards(d, u) == e);
}

// --- new_observer_frames

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_observer_frames__baseline() {
    let _d = black_box(v());
    let _u = black_box(up());
    let e = black_box(uqt((4087096511, 1096004728, 735529113, -12519501)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_observer_frames__face_towards() {
    let d = black_box(v());
    let u = black_box(up());
    let e = black_box(uqt((4087096511, 1096004728, 735529113, -12519501)));
    assert!(UnitQuaternionTrait::new_observer_frames(d, u) == e);
}

// --- look_at_rh

#[test]
#[inline(never)]
fn bench_unit_quaternion_look_at_rh__baseline() {
    let _d = black_box(v());
    let _u = black_box(up());
    let e = black_box(uqt((-735529113, -12519501, -4087096511, -1096004728)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_look_at_rh__face_towards() {
    let d = black_box(v());
    let u = black_box(up());
    let e = black_box(uqt((-735529113, -12519501, -4087096511, -1096004728)));
    assert!(UnitQuaternionTrait::look_at_rh(d, u) == e);
}

// --- look_at_lh

#[test]
#[inline(never)]
fn bench_unit_quaternion_look_at_lh__baseline() {
    let _d = black_box(v());
    let _u = black_box(up());
    let e = black_box(uqt((4087096511, -1096004728, -735529113, 12519501)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_look_at_lh__face_towards() {
    let d = black_box(v());
    let u = black_box(up());
    let e = black_box(uqt((4087096511, -1096004728, -735529113, 12519501)));
    assert!(UnitQuaternionTrait::look_at_lh(d, u) == e);
}

// --- from_matrix

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_matrix__baseline() {
    let _m = black_box(mat());
    let e = black_box(uqt((4122288484, 761768116, 0, 934446931)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_matrix__closed_form() {
    let m = black_box(mat());
    let e = black_box(uqt((4122288484, 761768116, 0, 934446931)));
    assert!(UnitQuaternionAngleTrait::from_matrix(m) == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_matrix__alt_iterate() {
    let m = black_box(mat());
    let e = black_box(uqt((4122288485, 761768125, -1, 934446920)));
    assert!(
        UnitQuaternionAngleTrait::from_matrix_eps(
            m, fx(32), 64, UnitQuaternionTrait::identity(),
        ) == e,
    );
}

// --- from_matrix_eps

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_matrix_eps__baseline() {
    let _m = black_box(mat());
    let e = black_box(uqt((4122400056, 761869347, -1, 933872022)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_matrix_eps__muller_8() {
    let m = black_box(mat());
    let e = black_box(uqt((4122400056, 761869347, -1, 933872022)));
    assert!(
        UnitQuaternionAngleTrait::from_matrix_eps(
            m, fx(32), 8, UnitQuaternionTrait::identity(),
        ) == e,
    );
}

// --- mean_of

#[test]
#[inline(never)]
fn bench_unit_quaternion_mean_of__baseline() {
    let _qs = black_box(array![a(), c(), d()].span());
    let e = black_box(uqt((1493153395, 2554507498, 2274495771, -2125652242)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_mean_of__squarings() {
    let qs = black_box(array![a(), c(), d()].span());
    let e = black_box(uqt((1493153395, 2554507498, 2274495771, -2125652242)));
    assert!(UnitQuaternionTrait::mean_of(qs) == e);
}

// --- ln

#[test]
#[inline(never)]
fn bench_unit_quaternion_ln__baseline() {
    let _a = black_box(a());
    let e = black_box(qt((0, 6635905207, 5859586768, -5474449130)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_ln__scaled_axis() {
    let a = black_box(a());
    let e = black_box(qt((0, 6635905207, 5859586768, -5474449130)));
    assert!(a.ln() == e);
}

// --- exp

#[test]
#[inline(never)]
fn bench_unit_quaternion_exp__baseline() {
    let _a = black_box(a());
    let e = black_box(qt((1791757316, -1551757823, -1370221443, 1280159826)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_exp__quaternion_exp() {
    let a = black_box(a());
    let e = black_box(qt((1791757316, -1551757823, -1370221443, 1280159826)));
    assert!(a.exp() == e);
}

// --- new

#[test]
#[inline(never)]
fn bench_unit_quaternion_new__baseline() {
    let _w = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new__from_scaled_axis() {
    let w = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(UnitQuaternionAngleTrait::new(w) == e);
}

// --- new_eps

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_eps__baseline() {
    let _w = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_new_eps__threshold() {
    let w = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(UnitQuaternionAngleTrait::new_eps(w, fx(0x1000)) == e);
}

// --- from_scaled_axis_eps

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_scaled_axis_eps__baseline() {
    let _w = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_scaled_axis_eps__new_eps() {
    let w = black_box(w());
    let e = black_box(uqt((4234293283, 534340439, -400755330, 267170219)));
    assert!(UnitQuaternionAngleTrait::from_scaled_axis_eps(w, fx(0x1000)) == e);
}

// --- to_euler_angles

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_euler_angles__baseline() {
    let _a = black_box(a());
    let e = black_box(v3t((-11965891174, 5500842972, 7356808383)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_to_euler_angles__euler_angles() {
    let a = black_box(a());
    let e = black_box(v3t((-11965891174, 5500842972, 7356808383)));
    assert!({
        let (r, p, y) = a.to_euler_angles();
        v3t((r.raw, p.raw, y.raw))
    } == e);
}

// --- from_quaternion

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_quaternion__baseline() {
    let _q = black_box(a().quaternion);
    let e = black_box(uqt((-1509276477, -2563574021, -2263667720, 2114881862)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_quaternion__normalize() {
    let q = black_box(a().quaternion);
    let e = black_box(uqt((-1509276477, -2563574021, -2263667720, 2114881862)));
    assert!(UnitQuaternionTrait::from_quaternion(q) == e);
}

// --- from_basis_unchecked

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_basis_unchecked__baseline() {
    let _bs = black_box(basis());
    let e = black_box(uqt((-1829744033, 968283947, 3750500405, -308145672)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_from_basis_unchecked__shepperd() {
    let bs = black_box(basis());
    let e = black_box(uqt((-1829744033, 968283947, 3750500405, -308145672)));
    assert!(UnitQuaternionTrait::from_basis_unchecked(bs) == e);
}

// --- cast

#[test]
#[inline(never)]
fn bench_unit_quaternion_cast__baseline() {
    let _a = black_box(a());
    let e = black_box(uqt((-1509276477, -2563574020, -2263667719, 2114881862)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_cast__into() {
    let a = black_box(a());
    let e = black_box(uqt((-1509276477, -2563574020, -2263667719, 2114881862)));
    assert!({
        let r: UnitQuaternion<Fixed> = a.cast();
        r
    } == e);
}

// --- lerp

#[test]
#[inline(never)]
fn bench_unit_quaternion_lerp__baseline() {
    let _a = black_box(a());
    let _b = black_box(b());
    let e = black_box(qt((-1589393366, -1680609529, -760125688, 1509124978)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_lerp__quaternion_lerp() {
    let a = black_box(a());
    let b = black_box(b());
    let e = black_box(qt((-1589393366, -1680609529, -760125688, 1509124978)));
    assert!(a.lerp(b, fx(0x40000000)) == e);
}

// --- relative_eq

#[test]
#[inline(never)]
fn bench_unit_quaternion_relative_eq__baseline() {
    let _a = black_box(a());
    let _b = black_box(b());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_relative_eq__components() {
    let a = black_box(a());
    let b = black_box(b());
    let e = black_box(false);
    assert!(a.relative_eq(b, 4, fx(0x100)) == e);
}

// --- ulps_eq

#[test]
#[inline(never)]
fn bench_unit_quaternion_ulps_eq__baseline() {
    let _a = black_box(a());
    let _b = black_box(b());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_ulps_eq__components() {
    let a = black_box(a());
    let b = black_box(b());
    let e = black_box(false);
    assert!(a.ulps_eq(b, 4, 4) == e);
}

// --- transform_unit_vector

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_unit_vector__baseline() {
    let _a = black_box(a());
    let _u = black_box(axis());
    let e = black_box(v3t((-2852243085, -2462353884, -2061132411)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_transform_unit_vector__transform_vector() {
    let a = black_box(a());
    let u = black_box(axis());
    let e = black_box(v3t((-2852243085, -2462353884, -2061132411)));
    assert!(a.transform_unit_vector(u).value == e);
}

// --- inverse_transform_unit_vector

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse_transform_unit_vector__baseline() {
    let _a = black_box(a());
    let _u = black_box(axis());
    let e = black_box(v3t((-3986355517, 1424885119, 724855962)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_inverse_transform_unit_vector__inverse_transform_vector() {
    let a = black_box(a());
    let u = black_box(axis());
    let e = black_box(v3t((-3986355517, 1424885119, 724855962)));
    assert!(a.inverse_transform_unit_vector(u).value == e);
}

// --- default

#[test]
#[inline(never)]
fn bench_unit_quaternion_default__baseline() {
    let e = black_box(uqt((4294967296, 0, 0, 0)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_default__const() {
    let e = black_box(uqt((4294967296, 0, 0, 0)));
    assert!(Default::<UnitQuaternion<Fixed>>::default() == e);
}

// --- one

#[test]
#[inline(never)]
fn bench_unit_quaternion_one__baseline() {
    let e = black_box(uqt((4294967296, 0, 0, 0)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_one__const() {
    let e = black_box(uqt((4294967296, 0, 0, 0)));
    assert!(One::<UnitQuaternion<Fixed>>::one() == e);
}

// --- into_isometry

#[test]
#[inline(never)]
fn bench_unit_quaternion_into_isometry__baseline() {
    let _a = black_box(a());
    let e = black_box(iso3t(((0, 0, 0), (-1509276477, -2563574020, -2263667719, 2114881862))));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_into_isometry__struct() {
    let a = black_box(a());
    let e = black_box(iso3t(((0, 0, 0), (-1509276477, -2563574020, -2263667719, 2114881862))));
    assert!({
        let r: Isometry3<Fixed> = a.into();
        r
    } == e);
}

// --- into_similarity

#[test]
#[inline(never)]
fn bench_unit_quaternion_into_similarity__baseline() {
    let _a = black_box(a());
    let e = black_box(
        sim3t(((0, 0, 0), (-1509276477, -2563574020, -2263667719, 2114881862), 4294967296)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_quaternion_into_similarity__struct() {
    let a = black_box(a());
    let e = black_box(
        sim3t(((0, 0, 0), (-1509276477, -2563574020, -2263667719, 2114881862), 4294967296)),
    );
    assert!({
        let r: Similarity3<Fixed> = a.into();
        r
    } == e);
}
