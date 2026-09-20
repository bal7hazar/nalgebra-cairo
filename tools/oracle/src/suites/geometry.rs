//! `nalgebra::geometry`: UnitComplex / Rotation2, Quaternion / UnitQuaternion, Rotation3,
//! Translation2/3, Isometry2/3, Similarity2/3.

use super::base::{mat_mul, mat_mul_vec};
use super::{
    flat, flat_c, flat_iso2, flat_iso3, flat_q, flat_sim2, flat_sim3, iso2, iso3, quat, ring, sim2,
    sim3, sm, sv, ucomplex, uquat, Ring,
};
use crate::engine::{
    fangle, fc, feuler, fiso2, fiso3, fm, fq, fs, fsim2, fsim3, fv, iangle, iiso2, iiso3, iq,
    isim2, isim3, it, iu, iuc, iuq, iv, with, Input, Op, Suite, Tol,
};
use crate::gen::{Dist, Gen};
use nalgebra::{
    Point2, Point3, Rotation2, Rotation3, Translation, Unit, UnitComplex, UnitQuaternion, Vector3,
};
use std::f64::consts::PI;

/// Two or three rounding stages on values of magnitude <= the inputs.
const ROTATE: Tol = Tol::Sens { k: 3.0, base: 2.0 };
/// One atan2 / acos (8 ulp, DESIGN D6) plus a few rounding stages.
const INVERSE_TRIG: Tol = Tol::Sens { k: 4.0, base: 24.0 };
/// sin / cos (2 ulp each) combined by a few products.
const TRIG: Tol = Tol::Sens { k: 6.0, base: 8.0 };

fn irot(name: &str, n: usize) -> Input {
    with(fm(name, n, n), Gen::Rot(n))
}

fn ieuler(name: &str) -> Input {
    with(
        feuler(name),
        Gen::Group(vec![
            Gen::Range(-PI, PI),
            Gen::Range(-1.45, 1.45),
            Gen::Range(-PI, PI),
        ]),
    )
}

fn rot2(x: &[f64]) -> Rotation2<f64> {
    Rotation2::from_matrix_unchecked(sm::<f64, 2, 2>(x))
}

fn rot3(x: &[f64]) -> Rotation3<f64> {
    Rotation3::from_matrix_unchecked(sm::<f64, 3, 3>(x))
}

fn v3(x: &[f64]) -> Vector3<f64> {
    sv::<f64, 3>(x)
}

/// Rejects nearly (anti)parallel pairs: `rotation_between` is singular there.
fn well_separated<const N: usize>(x: &[f64]) -> bool {
    let (a, b) = (sv::<f64, N>(x), sv::<f64, N>(&x[N..]));
    let cos = a.dot(&b) / (a.norm() * b.norm());
    cos.abs() < 0.95
}

/// Quaternion slerp is singular for (anti)parallel inputs and flips its arc when the dot product
/// changes sign: keep away from both.
fn slerp_ok(dot: f64) -> bool {
    (0.05..0.95).contains(&dot.abs())
}

/// Rejects the gimbal-lock neighbourhood of an euler triple.
fn euler_ok((roll, pitch, yaw): (f64, f64, f64)) -> Option<Vec<f64>> {
    (pitch.abs() < 1.45).then_some(vec![roll, pitch, yaw])
}

// Bilinear kernels on complex numbers, exact on i128 raws ----------------------------------------

/// `(re, im) * (re, im)`.
fn complex_mul<T: Ring>(x: &[T]) -> Vec<T> {
    vec![x[0] * x[2] - x[1] * x[3], x[0] * x[3] + x[1] * x[2]]
}
/// Rotation of `(x, y)` by `(re, im)`.
fn complex_rotate<T: Ring>(x: &[T]) -> Vec<T> {
    vec![x[0] * x[2] - x[1] * x[3], x[1] * x[2] + x[0] * x[3]]
}
/// Rotation of `(x, y)` by the conjugate of `(re, im)`.
fn complex_inverse_rotate<T: Ring>(x: &[T]) -> Vec<T> {
    vec![x[0] * x[2] + x[1] * x[3], x[0] * x[3] - x[1] * x[2]]
}

/// Hamilton product of `(w, i, j, k)` quaternions.
fn hamilton<T: Ring>(x: &[T]) -> Vec<T> {
    let (a, b) = (&x[..4], &x[4..8]);
    vec![
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// The exact kernels above are hand-written; this checks them against upstream in f64.
fn checked(
    pair: (crate::engine::EvalFn, crate::engine::ExactFn),
    upstream: impl Fn(&[f64]) -> Vec<f64> + 'static,
) -> (crate::engine::EvalFn, crate::engine::ExactFn) {
    let (eval, exact) = pair;
    let eval = Box::new(move |x: &[f64]| {
        let ours = eval(x)?;
        let theirs = upstream(x);
        let scale = x.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        for (o, t) in ours.iter().zip(&theirs) {
            assert!((o - t).abs() <= 1.0e-12 * scale * scale, "kernel mismatch");
        }
        Some(theirs)
    });
    (eval, exact)
}

fn rotation2_ops() -> Vec<Op> {
    vec![
        Op::new("unit_complex_new", "UnitComplex::new(angle)")
            .input(iangle("angle"))
            .out(fc("rotation"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(4))
            .eval(|x| Some(flat_c(&UnitComplex::new(x[0]))))
            .special(&[0.0], 0),
        Op::new("unit_complex_angle", "c.angle()")
            .input(iuc("c"))
            .out(fangle("angle"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(12))
            .eval(|x| Some(vec![ucomplex(x).angle()]))
            .special(&[1.0, 0.0], 0)
            .special(&[0.0, 1.0], 12)
            .special(&[-1.0, 0.0], 12),
        Op::new("unit_complex_mul", "a * b")
            .input(iuc("a"))
            .input(iuc("b"))
            .out(fc("product"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(complex_mul), |x| {
                flat_c(&(ucomplex(x) * ucomplex(&x[2..])))
            })),
        Op::new("unit_complex_inverse", "c.inverse() (= c.conjugate())")
            .input(iuc("c"))
            .out(fc("inverse"))
            .dists(&Dist::UNIT)
            .eval(|x| Some(flat_c(&ucomplex(x).inverse()))),
        Op::new("unit_complex_transform_vector", "c.transform_vector(&v)")
            .input(iuc("c"))
            .input(iv("v", 2))
            .out(fv("rotated", 2))
            .ring(checked(ring!(complex_rotate), |x| {
                flat(&ucomplex(x).transform_vector(&sv::<f64, 2>(&x[2..])))
            })),
        Op::new(
            "unit_complex_inverse_transform_vector",
            "c.inverse_transform_vector(&v)",
        )
        .input(iuc("c"))
        .input(iv("v", 2))
        .out(fv("rotated", 2))
        .ring(checked(ring!(complex_inverse_rotate), |x| {
            flat(&ucomplex(x).inverse_transform_vector(&sv::<f64, 2>(&x[2..])))
        })),
        Op::new("unit_complex_to_rotation_matrix", "c.to_rotation_matrix()")
            .input(iuc("c"))
            .out(fm("matrix", 2, 2))
            .dists(&Dist::UNIT)
            .eval(|x| Some(flat(ucomplex(x).to_rotation_matrix().matrix()))),
        Op::new(
            "unit_complex_from_rotation_matrix",
            "UnitComplex::from_rotation_matrix(&r)",
        )
        .input(irot("r", 2))
        .out(fc("rotation"))
        .dists(&Dist::UNIT)
        .eval(|x| Some(flat_c(&UnitComplex::from_rotation_matrix(&rot2(x))))),
        Op::new(
            "unit_complex_rotation_between",
            "UnitComplex::rotation_between(&a, &b)",
        )
        .input(iv("a", 2))
        .input(iv("b", 2))
        .out(fc("rotation"))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 4.0, base: 8.0 })
        .eval(|x| {
            Some(flat_c(&UnitComplex::rotation_between(
                &sv::<f64, 2>(x),
                &sv::<f64, 2>(&x[2..]),
            )))
        }),
        Op::new(
            "unit_complex_angle_to",
            "a.angle_to(&b) (branch cut at +-pi avoided)",
        )
        .input(iuc("a"))
        .input(iuc("b"))
        .out(fangle("angle"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 1.0, base: 14.0 })
        .eval(|x| {
            let angle = ucomplex(x).angle_to(&ucomplex(&x[2..]));
            (angle.abs() < 3.1).then_some(vec![angle])
        }),
        Op::new(
            "unit_complex_slerp",
            "a.slerp(&b, t) = a * UnitComplex::new((b / a).angle() * t): shortest arc, \
             |(b / a).angle()| < 3",
        )
        .input(iuc("a"))
        .input(iuc("b"))
        .input(it("t"))
        .out(fc("slerp"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 4.0, base: 32.0 })
        .eval(|x| {
            let (a, b) = (ucomplex(x), ucomplex(&x[2..]));
            // The arc flips when the relative angle crosses +-pi.
            (a.angle_to(&b).abs() < 3.0).then(|| flat_c(&a.slerp(&b, x[4])))
        }),
        Op::new("rotation2_new", "Rotation2::new(angle)")
            .input(iangle("angle"))
            .out(fm("matrix", 2, 2))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(4))
            .eval(|x| Some(flat(Rotation2::new(x[0]).matrix()))),
        Op::new("rotation2_angle", "r.angle()")
            .input(irot("r", 2))
            .out(fangle("angle"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(12))
            .eval(|x| Some(vec![rot2(x).angle()])),
        Op::new("rotation2_mul", "a * b")
            .input(irot("a", 2))
            .input(irot("b", 2))
            .out(fm("product", 2, 2))
            .dists(&Dist::UNIT)
            .ring(ring!(mat_mul, 2)),
        Op::new("rotation2_inverse", "r.inverse() (= transpose)")
            .input(irot("r", 2))
            .out(fm("inverse", 2, 2))
            .dists(&Dist::UNIT)
            .eval(|x| Some(flat(rot2(x).inverse().matrix()))),
        Op::new(
            "rotation2_transform_vector",
            "r.transform_vector(&v) = r * v",
        )
        .input(irot("r", 2))
        .input(iv("v", 2))
        .out(fv("rotated", 2))
        .ring(ring!(mat_mul_vec, 2)),
    ]
}

fn quaternion_ops() -> Vec<Op> {
    vec![
        Op::new("quaternion_mul", "a * b (Hamilton product)")
            .input(iq("a"))
            .input(iq("b"))
            .out(fq("product"))
            .ring(checked(ring!(hamilton), |x| {
                flat_q(&(quat(x) * quat(&x[4..])))
            }))
            .tol(Tol::Ulp(1)),
        Op::new("quaternion_conjugate", "q.conjugate()")
            .input(iq("q"))
            .out(fq("conjugate"))
            .eval(|x| Some(flat_q(&quat(x).conjugate()))),
        Op::new("quaternion_norm", "q.norm()")
            .input(iq("q"))
            .out(fs("norm"))
            .tol(Tol::Ulp(2))
            .eval(|x| Some(vec![quat(x).norm()])),
        Op::new("quaternion_normalize", "q.normalize()")
            .input(iq("q"))
            .out(fq("unit"))
            .tol(Tol::Sens { k: 2.0, base: 2.0 })
            .eval(|x| Some(flat_q(&quat(x).normalize()))),
        Op::new("quaternion_try_inverse", "q.try_inverse().unwrap()")
            .input(iq("q"))
            .out(fq("inverse"))
            .dists(&Dist::NO_LARGE)
            .tol(ROTATE)
            .eval(|x| quat(x).try_inverse().map(|q| flat_q(&q))),
        Op::new("unit_quaternion_mul", "a * b")
            .input(iuq("a"))
            .input(iuq("b"))
            .out(fq("product"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(hamilton), |x| {
                flat_q(&(uquat(x) * uquat(&x[4..])).into_inner())
            }))
            .tol(Tol::Ulp(1)),
        Op::new("unit_quaternion_inverse", "q.inverse() (= q.conjugate())")
            .input(iuq("q"))
            .out(fq("inverse"))
            .dists(&Dist::UNIT)
            .eval(|x| Some(flat_q(&uquat(x).inverse()))),
        Op::new(
            "unit_quaternion_from_axis_angle",
            "UnitQuaternion::from_axis_angle(&axis, angle)",
        )
        .input(iu("axis", 3))
        .input(iangle("angle"))
        .out(fq("rotation"))
        .dists(&Dist::UNIT)
        .tol(Tol::Ulp(6))
        .eval(|x| {
            let axis = Unit::new_unchecked(v3(x));
            Some(flat_q(&UnitQuaternion::from_axis_angle(&axis, x[3])))
        }),
        Op::new(
            "unit_quaternion_from_scaled_axis",
            "UnitQuaternion::from_scaled_axis(axisangle)",
        )
        .input(with(fv("axisangle", 3), Gen::ScaledAxis))
        .out(fq("rotation"))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| Some(flat_q(&UnitQuaternion::from_scaled_axis(v3(x))))),
        Op::new("unit_quaternion_scaled_axis", "q.scaled_axis()")
            .input(iuq("q"))
            .out(fv("axisangle", 3))
            .dists(&Dist::UNIT)
            .tol(INVERSE_TRIG)
            .eval(|x| Some(flat(&uquat(x).scaled_axis()))),
        Op::new("unit_quaternion_angle", "q.angle() in [0, pi]")
            .input(iuq("q"))
            .out(fangle("angle"))
            .dists(&Dist::UNIT)
            .tol(INVERSE_TRIG)
            .eval(|x| Some(vec![uquat(x).angle()])),
        Op::new("unit_quaternion_angle_to", "a.angle_to(&b)")
            .input(iuq("a"))
            .input(iuq("b"))
            .out(fangle("angle"))
            .dists(&Dist::UNIT)
            .tol(INVERSE_TRIG)
            .eval(|x| Some(vec![uquat(x).angle_to(&uquat(&x[4..]))])),
        Op::new(
            "unit_quaternion_to_rotation_matrix",
            "q.to_rotation_matrix()",
        )
        .input(iuq("q"))
        .out(fm("matrix", 3, 3))
        .dists(&Dist::UNIT)
        .tol(Tol::Ulp(4))
        .eval(|x| Some(flat(uquat(x).to_rotation_matrix().matrix()))),
        Op::new(
            "unit_quaternion_from_rotation_matrix",
            "UnitQuaternion::from_rotation_matrix(&r) (sign as upstream: compare up to sign if \
             your branch selection differs)",
        )
        .input(irot("r", 3))
        .out(fq("rotation"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 4.0, base: 4.0 })
        .eval(|x| Some(flat_q(&UnitQuaternion::from_rotation_matrix(&rot3(x))))),
        Op::new(
            "unit_quaternion_transform_vector",
            "q.transform_vector(&v) = q * v",
        )
        .input(iuq("q"))
        .input(iv("v", 3))
        .out(fv("rotated", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&uquat(x).transform_vector(&v3(&x[4..]))))),
        Op::new(
            "unit_quaternion_inverse_transform_vector",
            "q.inverse_transform_vector(&v)",
        )
        .input(iuq("q"))
        .input(iv("v", 3))
        .out(fv("rotated", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&uquat(x).inverse_transform_vector(&v3(&x[4..]))))),
        Op::new(
            "unit_quaternion_slerp",
            "a.slerp(&b, t): shortest arc (upstream negates b when a.dot(b) < 0), \
             0.05 < |a.dot(b)| < 0.95",
        )
        .input(iuq("a"))
        .input(iuq("b"))
        .input(it("t"))
        .out(fq("slerp"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 4.0, base: 32.0 })
        .eval(|x| {
            let (a, b) = (uquat(x), uquat(&x[4..]));
            slerp_ok(a.coords.dot(&b.coords)).then(|| flat_q(&a.slerp(&b, x[8])))
        }),
        Op::new(
            "unit_quaternion_nlerp",
            "a.nlerp(&b, t): plain lerp + normalize, no shortest-arc flip; a.dot(b) > -0.9",
        )
        .input(iuq("a"))
        .input(iuq("b"))
        .input(it("t"))
        .out(fq("nlerp"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 3.0, base: 2.0 })
        .eval(|x| {
            let (a, b) = (uquat(x), uquat(&x[4..]));
            (a.coords.dot(&b.coords) > -0.9).then(|| flat_q(&a.nlerp(&b, x[8])))
        }),
        Op::new(
            "unit_quaternion_rotation_between",
            "UnitQuaternion::rotation_between(&a, &b).unwrap()",
        )
        .input(iv("a", 3))
        .input(iv("b", 3))
        .out(fq("rotation"))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 16.0 })
        .eval(|x| {
            if !well_separated::<3>(x) {
                return None;
            }
            UnitQuaternion::rotation_between(&v3(x), &v3(&x[3..])).map(|q| flat_q(&q))
        }),
        Op::new(
            "unit_quaternion_euler_angles",
            "q.euler_angles() -> (roll, pitch, yaw), |pitch| < 1.45",
        )
        .input(iuq("q"))
        .out(feuler("euler"))
        .dists(&Dist::UNIT)
        .tol(INVERSE_TRIG)
        .eval(|x| euler_ok(uquat(x).euler_angles())),
        Op::new(
            "unit_quaternion_from_euler_angles",
            "UnitQuaternion::from_euler_angles(roll, pitch, yaw)",
        )
        .input(ieuler("euler"))
        .out(fq("rotation"))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| Some(flat_q(&UnitQuaternion::from_euler_angles(x[0], x[1], x[2])))),
        Op::new(
            "unit_quaternion_append_axisangle_linearized",
            "q.append_axisangle_linearized(&axisangle), |axisangle_i| < 1",
        )
        .input(iuq("q"))
        .input(iv("axisangle", 3))
        .out(fq("rotation"))
        .dists(&[Dist::Small])
        .tol(Tol::Sens { k: 3.0, base: 2.0 })
        .eval(|x| Some(flat_q(&uquat(x).append_axisangle_linearized(&v3(&x[4..]))))),
    ]
}

fn rotation3_ops() -> Vec<Op> {
    vec![
        Op::new(
            "rotation3_from_axis_angle",
            "Rotation3::from_axis_angle(&axis, angle)",
        )
        .input(iu("axis", 3))
        .input(iangle("angle"))
        .out(fm("matrix", 3, 3))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| {
            let axis = Unit::new_unchecked(v3(x));
            Some(flat(Rotation3::from_axis_angle(&axis, x[3]).matrix()))
        }),
        Op::new(
            "rotation3_from_scaled_axis",
            "Rotation3::from_scaled_axis(axisangle)",
        )
        .input(with(fv("axisangle", 3), Gen::ScaledAxis))
        .out(fm("matrix", 3, 3))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| Some(flat(Rotation3::from_scaled_axis(v3(x)).matrix()))),
        Op::new("rotation3_scaled_axis", "r.scaled_axis()")
            .input(irot("r", 3))
            .out(fv("axisangle", 3))
            .dists(&Dist::UNIT)
            .tol(INVERSE_TRIG)
            .eval(|x| Some(flat(&rot3(x).scaled_axis()))),
        Op::new("rotation3_angle", "r.angle() in [0, pi]")
            .input(irot("r", 3))
            .out(fangle("angle"))
            .dists(&Dist::UNIT)
            .tol(INVERSE_TRIG)
            .eval(|x| Some(vec![rot3(x).angle()])),
        Op::new(
            "rotation3_euler_angles",
            "r.euler_angles() -> (roll, pitch, yaw), |pitch| < 1.45",
        )
        .input(irot("r", 3))
        .out(feuler("euler"))
        .dists(&Dist::UNIT)
        .tol(INVERSE_TRIG)
        .eval(|x| euler_ok(rot3(x).euler_angles())),
        Op::new(
            "rotation3_from_euler_angles",
            "Rotation3::from_euler_angles(roll, pitch, yaw)",
        )
        .input(ieuler("euler"))
        .out(fm("matrix", 3, 3))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| {
            Some(flat(
                Rotation3::from_euler_angles(x[0], x[1], x[2]).matrix(),
            ))
        }),
        Op::new("rotation3_mul", "a * b")
            .input(irot("a", 3))
            .input(irot("b", 3))
            .out(fm("product", 3, 3))
            .dists(&Dist::UNIT)
            .ring(ring!(mat_mul, 3)),
        Op::new("rotation3_inverse", "r.inverse() (= transpose)")
            .input(irot("r", 3))
            .out(fm("inverse", 3, 3))
            .dists(&Dist::UNIT)
            .eval(|x| Some(flat(rot3(x).inverse().matrix()))),
        Op::new(
            "rotation3_transform_vector",
            "r.transform_vector(&v) = r * v",
        )
        .input(irot("r", 3))
        .input(iv("v", 3))
        .out(fv("rotated", 3))
        .ring(ring!(mat_mul_vec, 3)),
        Op::new(
            "rotation3_rotation_between",
            "Rotation3::rotation_between(&a, &b).unwrap()",
        )
        .input(iv("a", 3))
        .input(iv("b", 3))
        .out(fm("matrix", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 16.0 })
        .eval(|x| {
            if !well_separated::<3>(x) {
                return None;
            }
            Rotation3::rotation_between(&v3(x), &v3(&x[3..])).map(|r| flat(r.matrix()))
        }),
        Op::new(
            "rotation3_face_towards",
            "Rotation3::face_towards(&dir, &up)",
        )
        .input(iv("dir", 3))
        .input(iv("up", 3))
        .out(fm("matrix", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 8.0 })
        .eval(|x| {
            well_separated::<3>(x)
                .then(|| flat(Rotation3::face_towards(&v3(x), &v3(&x[3..])).matrix()))
        }),
        Op::new("rotation3_look_at_rh", "Rotation3::look_at_rh(&dir, &up)")
            .input(iv("dir", 3))
            .input(iv("up", 3))
            .out(fm("matrix", 3, 3))
            .dists(&Dist::NO_LARGE)
            .tol(Tol::Sens { k: 6.0, base: 8.0 })
            .eval(|x| {
                well_separated::<3>(x)
                    .then(|| flat(Rotation3::look_at_rh(&v3(x), &v3(&x[3..])).matrix()))
            }),
    ]
}

fn translation_ops<const N: usize>() -> Vec<Op> {
    let p = format!("translation{N}");
    let t = |x: &[f64]| Translation::<f64, N>::from(sv::<f64, N>(x));
    let pt = |x: &[f64]| nalgebra::Point::<f64, N>::from(sv::<f64, N>(x));
    vec![
        Op::new(format!("{p}_mul"), "a * b (composition = sum)")
            .input(iv("a", N))
            .input(iv("b", N))
            .out(fv("product", N))
            .eval(move |x| Some(flat(&(t(x) * t(&x[N..])).vector))),
        Op::new(format!("{p}_inverse"), "t.inverse()")
            .input(iv("t", N))
            .out(fv("inverse", N))
            .eval(move |x| Some(flat(&t(x).inverse().vector))),
        Op::new(format!("{p}_transform_point"), "t.transform_point(&p)")
            .input(iv("t", N))
            .input(iv("p", N))
            .out(fv("point", N))
            .eval(move |x| Some(flat(&t(x).transform_point(&pt(&x[N..])).coords))),
        Op::new(
            format!("{p}_inverse_transform_point"),
            "t.inverse_transform_point(&p)",
        )
        .input(iv("t", N))
        .input(iv("p", N))
        .out(fv("point", N))
        .eval(move |x| Some(flat(&t(x).inverse_transform_point(&pt(&x[N..])).coords))),
    ]
}

fn isometry2_ops() -> Vec<Op> {
    // Rotations by a unit complex are single fused kernels: a constant budget is enough.
    let tol = || Tol::Ulp(2);
    let p2 = |x: &[f64]| Point2::new(x[0], x[1]);
    vec![
        Op::new("isometry2_mul", "a * b")
            .input(iiso2("a"))
            .input(iiso2("b"))
            .out(fiso2("product"))
            .tol(tol())
            .eval(|x| Some(flat_iso2(&(iso2(x) * iso2(&x[4..]))))),
        Op::new("isometry2_inverse", "a.inverse()")
            .input(iiso2("a"))
            .out(fiso2("inverse"))
            .tol(tol())
            .eval(|x| Some(flat_iso2(&iso2(x).inverse()))),
        Op::new("isometry2_inv_mul", "a.inv_mul(&b) = a.inverse() * b")
            .input(iiso2("a"))
            .input(iiso2("b"))
            .out(fiso2("product"))
            .tol(tol())
            .eval(|x| Some(flat_iso2(&iso2(x).inv_mul(&iso2(&x[4..]))))),
        Op::new("isometry2_transform_point", "a.transform_point(&p) = a * p")
            .input(iiso2("a"))
            .input(iv("p", 2))
            .out(fv("point", 2))
            .tol(tol())
            .eval(move |x| Some(flat(&iso2(x).transform_point(&p2(&x[4..])).coords))),
        Op::new(
            "isometry2_transform_vector",
            "a.transform_vector(&v) = a * v",
        )
        .input(iiso2("a"))
        .input(iv("v", 2))
        .out(fv("vector", 2))
        .tol(tol())
        .eval(|x| Some(flat(&iso2(x).transform_vector(&sv::<f64, 2>(&x[4..]))))),
        Op::new(
            "isometry2_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .input(iiso2("a"))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .tol(tol())
        .eval(move |x| Some(flat(&iso2(x).inverse_transform_point(&p2(&x[4..])).coords))),
        Op::new(
            "isometry2_inverse_transform_vector",
            "a.inverse_transform_vector(&v)",
        )
        .input(iiso2("a"))
        .input(iv("v", 2))
        .out(fv("vector", 2))
        .tol(tol())
        .eval(|x| {
            Some(flat(
                &iso2(x).inverse_transform_vector(&sv::<f64, 2>(&x[4..])),
            ))
        }),
        Op::new("isometry2_to_homogeneous", "a.to_homogeneous()")
            .input(iiso2("a"))
            .out(fm("matrix", 3, 3))
            .eval(|x| Some(flat(&iso2(x).to_homogeneous()))),
    ]
}

fn isometry3_ops() -> Vec<Op> {
    let p3 = |x: &[f64]| Point3::new(x[0], x[1], x[2]);
    vec![
        Op::new("isometry3_mul", "a * b")
            .input(iiso3("a"))
            .input(iiso3("b"))
            .out(fiso3("product"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso3(&(iso3(x) * iso3(&x[7..]))))),
        Op::new("isometry3_inverse", "a.inverse()")
            .input(iiso3("a"))
            .out(fiso3("inverse"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso3(&iso3(x).inverse()))),
        Op::new("isometry3_inv_mul", "a.inv_mul(&b) = a.inverse() * b")
            .input(iiso3("a"))
            .input(iiso3("b"))
            .out(fiso3("product"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso3(&iso3(x).inv_mul(&iso3(&x[7..]))))),
        Op::new("isometry3_transform_point", "a.transform_point(&p) = a * p")
            .input(iiso3("a"))
            .input(iv("p", 3))
            .out(fv("point", 3))
            .tol(ROTATE)
            .eval(move |x| Some(flat(&iso3(x).transform_point(&p3(&x[7..])).coords))),
        Op::new(
            "isometry3_transform_vector",
            "a.transform_vector(&v) = a * v",
        )
        .input(iiso3("a"))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&iso3(x).transform_vector(&v3(&x[7..]))))),
        Op::new(
            "isometry3_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .input(iiso3("a"))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(ROTATE)
        .eval(move |x| Some(flat(&iso3(x).inverse_transform_point(&p3(&x[7..])).coords))),
        Op::new(
            "isometry3_inverse_transform_vector",
            "a.inverse_transform_vector(&v)",
        )
        .input(iiso3("a"))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&iso3(x).inverse_transform_vector(&v3(&x[7..]))))),
        Op::new("isometry3_to_homogeneous", "a.to_homogeneous()")
            .input(iiso3("a"))
            .out(fm("matrix", 4, 4))
            .tol(Tol::Ulp(4))
            .eval(|x| Some(flat(&iso3(x).to_homogeneous()))),
        Op::new(
            "isometry3_lerp_slerp",
            "a.lerp_slerp(&b, t), 0.05 < |a.rotation.dot(b.rotation)| < 0.95",
        )
        .input(iiso3("a"))
        .input(iiso3("b"))
        .input(it("t"))
        .out(fiso3("interpolated"))
        .tol(Tol::Sens { k: 4.0, base: 32.0 })
        .eval(|x| {
            let (a, b) = (iso3(x), iso3(&x[7..]));
            slerp_ok(a.rotation.coords.dot(&b.rotation.coords))
                .then(|| flat_iso3(&a.lerp_slerp(&b, x[14])))
        }),
    ]
}

fn similarity_ops() -> Vec<Op> {
    let p2 = |x: &[f64]| Point2::new(x[0], x[1]);
    let p3 = |x: &[f64]| Point3::new(x[0], x[1], x[2]);
    vec![
        Op::new("similarity2_mul", "a * b")
            .input(isim2("a"))
            .input(isim2("b"))
            .out(fsim2("product"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim2(&(sim2(x) * sim2(&x[5..]))))),
        Op::new("similarity2_inverse", "a.inverse()")
            .input(isim2("a"))
            .out(fsim2("inverse"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim2(&sim2(x).inverse()))),
        Op::new(
            "similarity2_transform_point",
            "a.transform_point(&p) = a * p",
        )
        .input(isim2("a"))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .tol(ROTATE)
        .eval(move |x| Some(flat(&sim2(x).transform_point(&p2(&x[5..])).coords))),
        Op::new(
            "similarity2_transform_vector",
            "a.transform_vector(&v) = a * v",
        )
        .input(isim2("a"))
        .input(iv("v", 2))
        .out(fv("vector", 2))
        .tol(ROTATE)
        .eval(|x| Some(flat(&sim2(x).transform_vector(&sv::<f64, 2>(&x[5..]))))),
        Op::new(
            "similarity2_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .input(isim2("a"))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .tol(ROTATE)
        .eval(move |x| Some(flat(&sim2(x).inverse_transform_point(&p2(&x[5..])).coords))),
        Op::new("similarity3_mul", "a * b")
            .input(isim3("a"))
            .input(isim3("b"))
            .out(fsim3("product"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim3(&(sim3(x) * sim3(&x[8..]))))),
        Op::new("similarity3_inverse", "a.inverse()")
            .input(isim3("a"))
            .out(fsim3("inverse"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim3(&sim3(x).inverse()))),
        Op::new(
            "similarity3_transform_point",
            "a.transform_point(&p) = a * p",
        )
        .input(isim3("a"))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(ROTATE)
        .eval(move |x| Some(flat(&sim3(x).transform_point(&p3(&x[8..])).coords))),
        Op::new(
            "similarity3_transform_vector",
            "a.transform_vector(&v) = a * v",
        )
        .input(isim3("a"))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&sim3(x).transform_vector(&v3(&x[8..]))))),
        Op::new(
            "similarity3_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .input(isim3("a"))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(ROTATE)
        .eval(move |x| Some(flat(&sim3(x).inverse_transform_point(&p3(&x[8..])).coords))),
    ]
}

pub fn suites() -> Vec<Suite> {
    vec![
        Suite {
            name: "rotation2",
            description: "UnitComplex and Rotation2: construction, angle, product, inverse, \
                          vector transforms, matrix conversions, rotation_between, slerp",
            ops: rotation2_ops(),
        },
        Suite {
            name: "quaternion",
            description: "Quaternion and UnitQuaternion: product, conjugate, inverse, axis-angle, \
                          scaled axis, rotation matrix conversions, vector transforms, slerp, nlerp, \
                          rotation_between, euler angles, linearized axis-angle update",
            ops: quaternion_ops(),
        },
        Suite {
            name: "rotation3",
            description: "Rotation3: axis-angle, scaled axis, euler angles, product, inverse, vector \
                          transform, rotation_between, face_towards, look_at_rh",
            ops: rotation3_ops(),
        },
        Suite {
            name: "translation",
            description: "Translation2 / Translation3: composition, inverse, point transforms",
            ops: {
                let mut ops = translation_ops::<2>();
                ops.extend(translation_ops::<3>());
                ops
            },
        },
        Suite {
            name: "isometry2",
            description: "Isometry2 (translation + unit complex): mul, inverse, inv_mul, point and \
                          vector transforms and their inverses, to_homogeneous",
            ops: isometry2_ops(),
        },
        Suite {
            name: "isometry3",
            description: "Isometry3 (translation + unit quaternion): mul, inverse, inv_mul, point and \
                          vector transforms and their inverses, to_homogeneous, lerp_slerp",
            ops: isometry3_ops(),
        },
        Suite {
            name: "similarity",
            description: "Similarity2 / Similarity3 (isometry + uniform scaling in [0.25, 4]): mul, \
                          inverse, point and vector transforms",
            ops: similarity_ops(),
        },
    ]
}
