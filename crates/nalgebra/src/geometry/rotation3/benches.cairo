//! Gas benchmarks of `Rotation3` (`bench_rotation3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together with
//! the tests showing why (AGENTS.md rule 8).
//!
//! The inputs are the rotation matrices of the two unit quaternions of the first
//! `unit_quaternion_mul` oracle case (`a`, `b`), the vector `(1.5, -2.25, 3.75)` and the rotation
//! vector `(0.25, -0.1875, 0.125)`. Expected values are the results of the kernels themselves, all
//! of which are checked against upstream nalgebra in `tests.cairo`.

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::{Real, Transcendental};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix_test_utils::{fx, r3, u3t, v3t};
use crate::base::point3::Point3;
use crate::base::unit::{Unit, Unit3Trait, UnitTrait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::quaternion::Quaternion;
use crate::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use super::{Rotation3, Rotation3AngleTrait, Rotation3Trait};

/// The rotation matrix of the unit quaternion `q` below.
fn a() -> Rotation3<Fixed> {
    r3(
        [
            [-173945351, 4188633151, -933723411], [1215906026, -848092610, -4031011726],
            [-4115587397, -427592469, -1151455223],
        ],
    )
}

/// Another rotation matrix.
fn b() -> Rotation3<Fixed> {
    r3(
        [
            [-2299358607, 1428519203, -3334520499], [1953624673, 3814159183, 286852618],
            [3056639446, -1363182554, -2691734142],
        ],
    )
}

/// The unit quaternion of `a` (up to the sign).
fn q() -> UnitQuaternion<Fixed> {
    UnitQuaternion {
        quaternion: Quaternion {
            i: fx(-2563574020), j: fx(-2263667719), k: fx(2114881862), w: fx(-1509276477),
        },
    }
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

// --- alternative implementations (losers)

/// `from_axis_angle` through the quaternion: `sin_cos` of the HALF angle, 3 products, then 24
/// products for the matrix — against one `sin_cos` and 21 products for Rodrigues' formula.
fn alt_from_axis_angle_quaternion(u: Unit<Vector3<Fixed>>, angle: Fixed) -> Rotation3<Fixed> {
    UnitQuaternionAngleTrait::from_axis_angle(u, angle).to_rotation_matrix()
}

/// `angle` through the quaternion: `2·atan2` instead of `acos((trace - 1) / 2)`, at the price of a
/// square root and three divisions.
fn alt_angle_quaternion(r: Rotation3<Fixed>) -> Fixed {
    UnitQuaternionTrait::from_rotation_matrix(r).angle()
}

/// `renormalize` through the quaternion: `from_rotation_matrix`, one Newton step, back to a matrix.
fn alt_renormalize_quaternion(r: Rotation3<Fixed>) -> Rotation3<Fixed> {
    UnitQuaternionTrait::from_rotation_matrix(r).renormalize_fast().to_rotation_matrix()
}

/// `renormalize` by one Newton step of the polar decomposition, `R · (3I - RᵀR) / 2`: two 3x3
/// products (54 products) against two norms and six divisions for Gram-Schmidt.
fn alt_renormalize_newton(r: Rotation3<Fixed>) -> Rotation3<Fixed> {
    let m = r.matrix;
    let s = m.tr_mul(m);
    let three = Real::<Fixed>::TWO + Real::ONE;
    let c = Matrix3 {
        m11: Real::mul_add(-s.m11, Real::HALF, three * Real::HALF),
        m21: -s.m21 * Real::HALF,
        m31: -s.m31 * Real::HALF,
        m12: -s.m12 * Real::HALF,
        m22: Real::mul_add(-s.m22, Real::HALF, three * Real::HALF),
        m32: -s.m32 * Real::HALF,
        m13: -s.m13 * Real::HALF,
        m23: -s.m23 * Real::HALF,
        m33: Real::mul_add(-s.m33, Real::HALF, three * Real::HALF),
    };
    Rotation3 { matrix: m * c }
}

/// `rotation_between` through upstream's trigonometric path (`acos` then Rodrigues' formula).
fn alt_rotation_between_axis_angle(
    x: Vector3<Fixed>, y: Vector3<Fixed>,
) -> Option<Rotation3<Fixed>> {
    let nx = x.try_normalize(Real::ZERO);
    let ny = y.try_normalize(Real::ZERO);
    match (nx, ny) {
        (
            Some(ux), Some(uy),
        ) => {
            let c = ux.cross(uy);
            let d = Real::clamp(Vector3Trait::dot(ux, uy), Real::NEG_ONE, Real::ONE);
            match UnitTrait::try_new(c, Real::ZERO) {
                Some(u) => Some(Rotation3AngleTrait::from_axis_angle(u, Transcendental::acos(d))),
                None => Option::None,
            }
        },
        _ => Some(Rotation3Trait::identity()),
    }
}

// --- why the alternatives lost

/// Going through the quaternion gives the same rotation within a few ulp, but it pays the half
/// angle and a second conversion.
#[test]
fn test_from_axis_angle_alt_quaternion_agrees() {
    let angle = fx(0x180000000);
    let direct = Rotation3AngleTrait::from_axis_angle(axis(), angle);
    let through = alt_from_axis_angle_quaternion(axis(), angle);
    assert!(through.abs_diff_eq(direct, 16));
}

/// `acos((trace - 1) / 2)` loses precision near the identity, where `2·atan2` on the quaternion
/// keeps it: `cos θ = 1 - θ²/2` is 1 within one ulp as soon as `θ < 2^-16`, so `acos` returns
/// exactly 0, while the quaternion path (whose imaginary part is proportional to `θ`) is accurate
/// to 2 ulp. At `θ = 2^-10` the error of `acos` is 1 024 ulp (the derivative of `acos` is `1/θ`)
/// and the quaternion path is still within 2. Upstream's form is kept as the cheap default (it
/// never needs a conversion) and this alternative is documented on `angle`.
#[test]
fn test_angle_alt_quaternion_is_more_accurate_near_zero() {
    // θ = 2^-17: `acos` collapses to zero, the quaternion path is within 2 ulp.
    let tiny = Rotation3AngleTrait::from_axis_angle(axis(), fx(0x8000));
    assert!(tiny.angle() == Real::ZERO);
    assert!(alt_angle_quaternion(tiny).abs_diff_eq(fx(0x8000), 4));
    // θ = 2^-10: 1 024 ulp against 2.
    let small = Rotation3AngleTrait::from_axis_angle(axis(), fx(0x400000));
    assert!(!small.angle().abs_diff_eq(fx(0x400000), 1000));
    assert!(small.angle().abs_diff_eq(fx(0x400000), 1024));
    assert!(alt_angle_quaternion(small).abs_diff_eq(fx(0x400000), 4));
    // Away from the identity both agree.
    assert!(alt_angle_quaternion(a()).abs_diff_eq(a().angle(), 64));
}

/// All three renormalizations restore orthonormality on a drifted matrix; Gram-Schmidt is the
/// cheapest (see the benchmarks) and the most accurate of the three here.
#[test]
fn test_renormalize_alts_restore_orthonormality() {
    let step = Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), fx(0x123456789));
    let mut r = Rotation3Trait::<Fixed>::identity();
    for _ in 0_u32..64 {
        r = r * step;
    }
    let gs = r.renormalize();
    let qn = alt_renormalize_quaternion(r);
    let nw = alt_renormalize_newton(r);
    assert!((gs.matrix * gs.matrix.transpose()).is_identity(4));
    assert!((qn.matrix * qn.matrix.transpose()).is_identity(16));
    assert!((nw.matrix * nw.matrix.transpose()).is_identity(16));
    assert!(gs.abs_diff_eq(qn, 64));
    assert!(gs.abs_diff_eq(nw, 64));
}

/// A Newton step of the polar decomposition only halves the error, so it does not fix a matrix
/// scaled by 1 + 1e-3, where Gram-Schmidt does in one pass.
#[test]
fn test_renormalize_alt_newton_only_converges() {
    let scaled = Rotation3 { matrix: a().matrix.scale(Real::ONE + fx(4294967)) };
    assert!((scaled.renormalize().matrix * scaled.renormalize().matrix.transpose()).is_identity(4));
    let once = alt_renormalize_newton(scaled);
    assert!(!(once.matrix * once.matrix.transpose()).is_identity(0x1000));
}

#[test]
fn test_rotation_between_alt_axis_angle_agrees() {
    let direct = Rotation3Trait::rotation_between(v(), w()).unwrap();
    let through = alt_rotation_between_axis_angle(v(), w()).unwrap();
    assert!(through.abs_diff_eq(direct, 32));
}


// --- constructors and parts

#[test]
#[inline(never)]
fn bench_rotation3_identity__baseline() {
    let e = black_box(Rotation3 { matrix: Matrix3Trait::<Fixed>::identity() });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_identity__const() {
    let e = black_box(Rotation3 { matrix: Matrix3Trait::<Fixed>::identity() });
    assert!(Rotation3Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_matrix__baseline() {
    let _r = black_box(a());
    let e = black_box(a().matrix);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_matrix__struct() {
    let r = black_box(a());
    let e = black_box(a().matrix);
    assert!(r.matrix() == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_inverse__baseline() {
    let _r = black_box(a());
    let e = black_box(
        r3(
            [
                [-173945351, 1215906026, -4115587397], [4188633151, -848092610, -427592469],
                [-933723411, -4031011726, -1151455223],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_inverse__transpose() {
    let r = black_box(a());
    let e = black_box(
        r3(
            [
                [-173945351, 1215906026, -4115587397], [4188633151, -848092610, -427592469],
                [-933723411, -4031011726, -1151455223],
            ],
        ),
    );
    assert!(r.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_to_homogeneous__baseline() {
    let _r = black_box(a());
    let e = black_box(fx(-173945351));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_to_homogeneous__matrix4() {
    let r = black_box(a());
    let e = black_box(fx(-173945351));
    let h = r.to_homogeneous();
    assert!(h.m11 == e);
    assert!(h.m44 == Real::ONE);
}

// --- composition and transforms

#[test]
#[inline(never)]
fn bench_rotation3_mul__baseline() {
    let _r = black_box(a());
    let _s = black_box(b());
    let e = black_box(
        r3(
            [
                [1333869062, 3958229158, 999979653], [-3905503099, 930668262, 1525662545],
                [1189362409, -1383120590, 3888333648],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_mul__matrix_product() {
    let r = black_box(a());
    let s = black_box(b());
    let e = black_box(
        r3(
            [
                [1333869062, 3958229158, 999979653], [-3905503099, 930668262, 1525662545],
                [1189362409, -1383120590, 3888333648],
            ],
        ),
    );
    assert!(r * s == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_transform_vector__baseline() {
    let _r = black_box(a());
    let _x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226561, -9529255127)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_transform_vector__mul_vec() {
    let r = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-13186805408, -11384226561, -9529255127)));
    assert!(r.transform_vector(x) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_inverse_transform_vector__baseline() {
    let _r = black_box(a());
    let _x = black_box(v());
    let e = black_box(v3t((-18430159324, 6587686340, 3351234180)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_inverse_transform_vector__tr_mul_vec() {
    let r = black_box(a());
    let x = black_box(v());
    let e = black_box(v3t((-18430159324, 6587686340, 3351234180)));
    assert!(r.inverse_transform_vector(x) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_transform_point__baseline() {
    let _r = black_box(a());
    let _p = black_box(Point3 { x: v().x, y: v().y, z: v().z });
    let e = black_box(Point3 { x: fx(-13186805408), y: fx(-11384226561), z: fx(-9529255127) });
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_transform_point__mul_vec() {
    let r = black_box(a());
    let p = black_box(Point3 { x: v().x, y: v().y, z: v().z });
    let e = black_box(Point3 { x: fx(-13186805408), y: fx(-11384226561), z: fx(-9529255127) });
    assert!(r.transform_point(p) == e);
}

// --- conversions with `UnitQuaternion`

#[test]
#[inline(never)]
fn bench_rotation3_to_unit_quaternion__baseline() {
    let _r = black_box(a());
    let e = black_box(
        UnitQuaternion {
            quaternion: Quaternion {
                i: fx(2563574020), j: fx(2263667718), k: fx(-2114881863), w: fx(1509276477),
            },
        },
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_to_unit_quaternion__shepperd() {
    let r = black_box(a());
    let e = black_box(
        UnitQuaternion {
            quaternion: Quaternion {
                i: fx(2563574020), j: fx(2263667718), k: fx(-2114881863), w: fx(1509276477),
            },
        },
    );
    assert!(r.to_unit_quaternion() == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_unit_quaternion__baseline() {
    let _q = black_box(q());
    let e = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_unit_quaternion__fused() {
    let x = black_box(q());
    let e = black_box(a());
    assert!(Rotation3Trait::from_unit_quaternion(x) == e);
}

// --- axis and angle

#[test]
#[inline(never)]
fn bench_rotation3_axis__baseline() {
    let _r = black_box(a());
    let e = black_box(u3t((2738208059, 2417871746, -2258950401)));
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_axis__antisymmetric_part() {
    let r = black_box(a());
    let e = black_box(u3t((2738208059, 2417871746, -2258950401)));
    assert!(r.axis() == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_angle__baseline() {
    let _r = black_box(a());
    let e = black_box(fx(10408630471));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_angle__acos_trace() {
    let r = black_box(a());
    let e = black_box(fx(10408630471));
    assert!(r.angle() == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_angle__alt_quaternion() {
    let r = black_box(a());
    let e = black_box(fx(10408630470));
    assert!(alt_angle_quaternion(r) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_scaled_axis__baseline() {
    let _r = black_box(a());
    let e = black_box(v3t((6635905205, 5859586766, -5474449130)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_scaled_axis__axis_times_angle() {
    let r = black_box(a());
    let e = black_box(v3t((6635905205, 5859586766, -5474449130)));
    assert!(r.scaled_axis() == e);
}

// --- axis-angle and Euler-angle constructors

#[test]
#[inline(never)]
fn bench_rotation3_from_axis_angle__baseline() {
    let _u = black_box(axis());
    let _t = black_box(fx(0x180000000));
    let e = black_box(
        r3(
            [
                [723935370, -4105133935, -1034667594], [2844769725, 1249087125, -2965435993],
                [3135274604, -185474531, 2929572735],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_axis_angle__rodrigues() {
    let u = black_box(axis());
    let t = black_box(fx(0x180000000));
    let e = black_box(
        r3(
            [
                [723935370, -4105133935, -1034667594], [2844769725, 1249087125, -2965435993],
                [3135274604, -185474531, 2929572735],
            ],
        ),
    );
    assert!(Rotation3AngleTrait::from_axis_angle(u, t) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_axis_angle__alt_quaternion() {
    let u = black_box(axis());
    let t = black_box(fx(0x180000000));
    let e = black_box(
        r3(
            [
                [723935370, -4105133935, -1034667595], [2844769725, 1249087127, -2965435993],
                [3135274606, -185474532, 2929572735],
            ],
        ),
    );
    assert!(alt_from_axis_angle_quaternion(u, t) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_scaled_axis__baseline() {
    let _x = black_box(w());
    let e = black_box(
        r3(
            [
                [4186940974, -626508540, -723710167], [427075329, 4128772954, -1103442173],
                [856665639, 1003725567, 4087224369],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_scaled_axis__norm_and_rodrigues() {
    let x = black_box(w());
    let e = black_box(
        r3(
            [
                [4186940974, -626508540, -723710167], [427075329, 4128772954, -1103442173],
                [856665639, 1003725567, 4087224369],
            ],
        ),
    );
    assert!(Rotation3AngleTrait::from_scaled_axis(x) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_euler_angles__baseline() {
    let _r = black_box(fx(0x40000000));
    let _p = black_box(fx(-0x30000000));
    let _y = black_box(fx(0x20000000));
    let e = black_box(
        r3(
            [
                [4186767317, -715352540, -637176780], [526088819, 4104283764, -1151012341],
                [800596063, 1043968198, 4088510782],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_euler_angles__three_sin_cos() {
    let r = black_box(fx(0x40000000));
    let p = black_box(fx(-0x30000000));
    let y = black_box(fx(0x20000000));
    let e = black_box(
        r3(
            [
                [4186767317, -715352540, -637176780], [526088819, 4104283764, -1151012341],
                [800596063, 1043968198, 4088510782],
            ],
        ),
    );
    assert!(Rotation3AngleTrait::from_euler_angles(r, p, y) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_euler_angles__baseline() {
    let _r = black_box(a());
    let e = black_box(fx(-11965891172));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_euler_angles__asin_atan2() {
    let r = black_box(a());
    let e = black_box(fx(-11965891172));
    let (roll, pitch, yaw) = r.euler_angles();
    assert!(roll == e);
    assert!(pitch == fx(5500842971));
    assert!(yaw == fx(7356808382));
}

// --- construction from vectors

#[test]
#[inline(never)]
fn bench_rotation3_rotation_between__baseline() {
    let _x = black_box(v());
    let _y = black_box(w());
    let e = black_box(
        r3(
            [
                [3667369366, -466783079, 2186151768], [1085783774, 4043498262, -958091575],
                [-1954026508, 1370758705, 3570650507],
            ],
        ),
    );
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_rotation_between__algebraic_quaternion() {
    let x = black_box(v());
    let y = black_box(w());
    let e = black_box(
        r3(
            [
                [3667369366, -466783079, 2186151768], [1085783774, 4043498262, -958091575],
                [-1954026508, 1370758705, 3570650507],
            ],
        ),
    );
    assert!(Rotation3Trait::rotation_between(x, y) == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_rotation_between__alt_axis_angle() {
    let x = black_box(v());
    let y = black_box(w());
    let e = black_box(
        r3(
            [
                [3667369367, -466783079, 2186151769], [1085783775, 4043498262, -958091576],
                [-1954026509, 1370758706, 3570650508],
            ],
        ),
    );
    assert!(alt_rotation_between_axis_angle(x, y) == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_scaled_rotation_between__baseline() {
    let _x = black_box(v());
    let _y = black_box(w());
    let e = black_box(
        r3(
            [
                [4134225491, -328359477, 1116737979], [486899339, 4230560477, -558597492],
                [-1057285532, 664290733, 4109453637],
            ],
        ),
    );
    assert!(Some(e) == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_scaled_rotation_between__acos_rodrigues() {
    let x = black_box(v());
    let y = black_box(w());
    let e = black_box(
        r3(
            [
                [4134225491, -328359477, 1116737979], [486899339, 4230560477, -558597492],
                [-1057285532, 664290733, 4109453637],
            ],
        ),
    );
    assert!(Rotation3AngleTrait::scaled_rotation_between(x, y, Real::HALF) == Some(e));
}

#[test]
#[inline(never)]
fn bench_rotation3_face_towards__baseline() {
    let _d = black_box(v());
    let _u = black_box(w());
    let e = black_box(
        r3(
            [
                [-2001464795, 3535409489, 1393471396], [-3558159633, -1190495032, -2090207096],
                [-1334309867, -2128460815, 3483678492],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_face_towards__two_normalizations() {
    let d = black_box(v());
    let u = black_box(w());
    let e = black_box(
        r3(
            [
                [-2001464795, 3535409489, 1393471396], [-3558159633, -1190495032, -2090207096],
                [-1334309867, -2128460815, 3483678492],
            ],
        ),
    );
    assert!(Rotation3Trait::face_towards(d, u) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_look_at_rh__baseline() {
    let _d = black_box(v());
    let _u = black_box(w());
    let e = black_box(
        r3(
            [
                [2001464792, 3558159631, 1334309858], [3535409484, -1190495033, -2128460813],
                [-1393471397, 2090207095, -3483678493],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_look_at_rh__transpose() {
    let d = black_box(v());
    let u = black_box(w());
    let e = black_box(
        r3(
            [
                [2001464792, 3558159631, 1334309858], [3535409484, -1190495033, -2128460813],
                [-1393471397, 2090207095, -3483678493],
            ],
        ),
    );
    assert!(Rotation3Trait::look_at_rh(d, u) == e);
}

// --- renormalization

#[test]
#[inline(never)]
fn bench_rotation3_renormalize__baseline() {
    let _r = black_box(a());
    let e = black_box(
        r3(
            [
                [-173945352, 4188633151, -933723414], [1215906026, -848092611, -4031011727],
                [-4115587398, -427592472, -1151455222],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_renormalize__gram_schmidt() {
    let r = black_box(a());
    let e = black_box(
        r3(
            [
                [-173945352, 4188633151, -933723414], [1215906026, -848092611, -4031011727],
                [-4115587398, -427592472, -1151455222],
            ],
        ),
    );
    assert!(r.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_renormalize__alt_quaternion() {
    let r = black_box(a());
    let e = black_box(a());
    assert!(alt_renormalize_quaternion(r).abs_diff_eq(e, 8));
}

#[test]
#[inline(never)]
fn bench_rotation3_renormalize__alt_newton() {
    let r = black_box(a());
    let e = black_box(a());
    assert!(alt_renormalize_newton(r).abs_diff_eq(e, 8));
}

// --- comparison

#[test]
#[inline(never)]
fn bench_rotation3_abs_diff_eq__baseline() {
    let _r = black_box(a());
    let _s = black_box(b());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_abs_diff_eq__entries() {
    let r = black_box(a());
    let s = black_box(b());
    let e = black_box(false);
    assert!(r.abs_diff_eq(s, 4) == e);
}
