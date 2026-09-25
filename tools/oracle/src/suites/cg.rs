//! WP 8.3-P07: the homogeneous / computer-graphics helpers of upstream `base/cg.rs`, suite `cg`:
//! `Matrix3::new_rotation`, the `Matrix4` rotation constructors (`new_rotation`,
//! `from_scaled_axis`, `from_axis_angle`, `from_euler_angles`, `new_rotation_wrt_point`), the
//! observer / view matrices (`face_towards`, `look_at_rh`, `look_at_lh`), and the homogeneous
//! `transform_point` / `transform_vector` of `Matrix3` / `Matrix4`.
//!
//! The scalings, translations and their compositions are exact compositions of floored products
//! (modelled bit for bit in `tools/shapegen/tests_cg.py`); the transforms divide by the
//! homogeneous normaliser, drawn away from zero here (`|n| >= 1/4`) so that the quotients fit.

use super::{flat, sm, sv};
use crate::engine::{feuler, fm, fv, iangle, im, iu, iv, with, Input, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{Matrix3, Matrix4, Point2, Point3, Unit, Vector3};
use std::f64::consts::PI;

/// sin / cos (2 ulp each) combined by a few products (the rotation matrices).
const TRIG: Tol = Tol::Sens { k: 6.0, base: 8.0 };
/// Normalisations and cross products (the observer frame).
const FRAME: Tol = Tol::Sens { k: 8.0, base: 16.0 };
/// The view transforms: the translation `rotation · (-eye)` multiplies the few-ulp error of the
/// normalised frame by `|eye|`.
const VIEW: Tol = Tol::SensMag {
    k: 8.0,
    base: 16.0,
    mag: 32.0,
};
/// One fused sum of products per component and per normaliser, then one division: the rounding
/// errors of `q` and `n` amplified by `1 / n` and `q / n^2` (first order, `Sens`).
const TRANSFORM: Tol = Tol::Sens { k: 3.0, base: 2.0 };
/// `transform_vector` divides BEFORE the product (upstream's order, `m * (v / n)`): the half-ulp
/// rounding of each quotient is multiplied by the entries of `m`, an absolute error of up to
/// `sum_k |m_ik| / 2` ulp that a fixed-point result cannot hide (f64 keeps it relative).
const TRANSFORM_VECTOR: Tol = Tol::SensMag {
    k: 3.0,
    base: 2.0,
    mag: 2.0,
};

fn ieuler() -> Input {
    with(
        feuler("euler"),
        Gen::Group(vec![
            Gen::Range(-PI, PI),
            Gen::Range(-1.45, 1.45),
            Gen::Range(-PI, PI),
        ]),
    )
}

fn v3(x: &[f64]) -> Vector3<f64> {
    sv::<f64, 3>(x)
}

fn p3(x: &[f64]) -> Point3<f64> {
    Point3::new(x[0], x[1], x[2])
}

/// Rejects the degenerate observer frames: `target - eye` and `up` nearly (anti)parallel or null.
fn frame_ok(x: &[f64]) -> bool {
    let (eye, target, up) = (v3(x), v3(&x[3..]), v3(&x[6..]));
    let dir = target - eye;
    let cos = dir.dot(&up) / (dir.norm() * up.norm());
    dir.norm() > 1.0e-3 && up.norm() > 1.0e-3 && cos.abs() < 0.95
}

/// The homogeneous normaliser away from zero (`|n| >= 1/4`).
fn normaliser_ok(n: f64) -> bool {
    n.abs() >= 0.25
}

fn rotation_ops() -> Vec<Op> {
    vec![
        Op::new("matrix3_new_rotation", "Matrix3::new_rotation(angle)")
            .input(iangle("angle"))
            .out(fm("matrix", 3, 3))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(4))
            .eval(|x| Some(flat(&Matrix3::new_rotation(x[0])))),
        Op::new("matrix4_new_rotation", "Matrix4::new_rotation(axisangle)")
            .input(with(fv("axisangle", 3), Gen::ScaledAxis))
            .out(fm("matrix", 4, 4))
            .dists(&Dist::UNIT)
            .tol(TRIG)
            .eval(|x| Some(flat(&Matrix4::new_rotation(v3(x))))),
        Op::new(
            "matrix4_from_scaled_axis",
            "Matrix4::from_scaled_axis(axisangle)",
        )
        .input(with(fv("axisangle", 3), Gen::ScaledAxis))
        .out(fm("matrix", 4, 4))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| Some(flat(&Matrix4::from_scaled_axis(v3(x))))),
        Op::new(
            "matrix4_from_axis_angle",
            "Matrix4::from_axis_angle(&axis, angle)",
        )
        .input(iu("axis", 3))
        .input(iangle("angle"))
        .out(fm("matrix", 4, 4))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| {
            let axis = Unit::new_unchecked(v3(x));
            Some(flat(&Matrix4::from_axis_angle(&axis, x[3])))
        }),
        Op::new(
            "matrix4_from_euler_angles",
            "Matrix4::from_euler_angles(roll, pitch, yaw)",
        )
        .input(ieuler())
        .out(fm("matrix", 4, 4))
        .dists(&Dist::UNIT)
        .tol(TRIG)
        .eval(|x| Some(flat(&Matrix4::from_euler_angles(x[0], x[1], x[2])))),
        Op::new(
            "matrix4_new_rotation_wrt_point",
            "Matrix4::new_rotation_wrt_point(axisangle, pt)",
        )
        .input(with(fv("axisangle", 3), Gen::ScaledAxis))
        .input(iv("pt", 3))
        .out(fm("matrix", 4, 4))
        .dists(&Dist::UNIT)
        .tol(VIEW)
        .eval(|x| Some(flat(&Matrix4::new_rotation_wrt_point(v3(x), p3(&x[3..]))))),
    ]
}

fn frame_ops() -> Vec<Op> {
    let frame = |name: &'static str, doc: &'static str, tol: Tol, f: fn(&[f64]) -> Matrix4<f64>| {
        Op::new(name, doc)
            .input(iv("eye", 3))
            .input(iv("target", 3))
            .input(iv("up", 3))
            .out(fm("matrix", 4, 4))
            .tol(tol)
            .eval(move |x| frame_ok(x).then(|| flat(&f(x))))
    };
    vec![
        frame(
            "matrix4_face_towards",
            "Matrix4::face_towards(&eye, &target, &up)",
            FRAME,
            |x| Matrix4::face_towards(&p3(x), &p3(&x[3..]), &v3(&x[6..])),
        ),
        frame(
            "matrix4_look_at_rh",
            "Matrix4::look_at_rh(&eye, &target, &up)",
            VIEW,
            |x| Matrix4::look_at_rh(&p3(x), &p3(&x[3..]), &v3(&x[6..])),
        ),
        frame(
            "matrix4_look_at_lh",
            "Matrix4::look_at_lh(&eye, &target, &up)",
            VIEW,
            |x| Matrix4::look_at_lh(&p3(x), &p3(&x[3..]), &v3(&x[6..])),
        ),
    ]
}

fn transform_ops() -> Vec<Op> {
    vec![
        Op::new(
            "matrix3_transform_point",
            "m.transform_point(&p), |m[2, :] . (p, 1)| >= 1/4",
        )
        .input(im("m", 3, 3))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .tol(TRANSFORM)
        .eval(|x| {
            let m: Matrix3<f64> = sm(x);
            let p = Point2::new(x[9], x[10]);
            let n = m[(2, 0)] * p.x + m[(2, 1)] * p.y + m[(2, 2)];
            normaliser_ok(n).then(|| flat(&m.transform_point(&p).coords))
        }),
        Op::new(
            "matrix4_transform_point",
            "m.transform_point(&p), |m[3, :] . (p, 1)| >= 1/4",
        )
        .input(im("m", 4, 4))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(TRANSFORM)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            let p = p3(&x[16..]);
            let n = m[(3, 0)] * p.x + m[(3, 1)] * p.y + m[(3, 2)] * p.z + m[(3, 3)];
            normaliser_ok(n).then(|| flat(&m.transform_point(&p).coords))
        }),
        Op::new(
            "matrix3_transform_vector",
            "m.transform_vector(&v), |m[2, :2] . v| >= 1/4",
        )
        .input(im("m", 3, 3))
        .input(iv("v", 2))
        .out(fv("vector", 2))
        .tol(TRANSFORM_VECTOR)
        .eval(|x| {
            let m: Matrix3<f64> = sm(x);
            let v = sv::<f64, 2>(&x[9..]);
            let n = m[(2, 0)] * v.x + m[(2, 1)] * v.y;
            normaliser_ok(n).then(|| flat(&m.transform_vector(&v)))
        }),
        Op::new(
            "matrix4_transform_vector",
            "m.transform_vector(&v), |m[3, :3] . v| >= 1/4",
        )
        .input(im("m", 4, 4))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .tol(TRANSFORM_VECTOR)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            let v = v3(&x[16..]);
            let n = m[(3, 0)] * v.x + m[(3, 1)] * v.y + m[(3, 2)] * v.z;
            normaliser_ok(n).then(|| flat(&m.transform_vector(&v)))
        }),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut ops = rotation_ops();
    ops.extend(frame_ops());
    ops.extend(transform_ops());
    vec![Suite {
        name: "cg",
        description:
            "Homogeneous / computer-graphics helpers (base/cg.rs): Matrix3::new_rotation, \
                      the Matrix4 rotation constructors, face_towards, look_at_rh, look_at_lh, \
                      transform_point and transform_vector of Matrix3 / Matrix4",
        ops,
    }]
}
