//! WP 8.4-P08: the completion of `Quaternion`, `UnitQuaternion` and `UnitComplex` — the
//! transcendental quaternion functions, the divisions, the polar decomposition, `from_matrix`,
//! `mean_of`, the observer frames and the heterogeneous operators.
//!
//! WP 8.4-P09a: the completion of the rotation matrices `Rotation2` / `Rotation3` (suite
//! `rotation_matrix_completion`): `new`, `powf`, `angle_to`, `axis_angle`, `rotation_to`, `/`,
//! `euler_angles_ordered`, `from_matrix`, `slerp`, `look_at_lh` and the products with unit
//! quaternions / unit complex numbers.

use super::{
    flat, flat_c, flat_iso2, flat_iso3, flat_q, flat_sim2, flat_sim3, iso2, iso3, quat, ring, sim2,
    sim3, sm, sv, ucomplex, uquat, Ring,
};
use crate::engine::{
    fangle, fc, fiso2, fiso3, fm, fq, fquats, fs, fsim2, fsim3, fv, iiso2, iiso3, iq, isim2, isim3,
    it, iu, iuc, iuq, iv, with, Input, Op, Suite, Tol,
};
use crate::gen::{Dist, Gen};
use nalgebra::{
    Complex, Matrix2, Matrix3, Quaternion, Rotation2, Rotation3, Translation2, Translation3, Unit,
    UnitComplex, UnitQuaternion, Vector2, Vector3,
};
use std::f64::consts::PI;

/// A few rounding stages on values of magnitude <= the inputs.
const ROTATE: Tol = Tol::Sens { k: 3.0, base: 2.0 };
/// One or two transcendental calls combined by a few products and divisions.
const TRANS: Tol = Tol::Sens { k: 8.0, base: 16.0 };
/// Upstream's compositions of `ln`, `sqrt` and products (every stage rounds).
const COMPOSED: Tol = Tol::Sens {
    k: 32.0,
    base: 64.0,
};

fn v2(x: &[f64]) -> Vector2<f64> {
    sv::<f64, 2>(x)
}

fn v3(x: &[f64]) -> Vector3<f64> {
    sv::<f64, 3>(x)
}

fn rot2(x: &[f64]) -> Rotation2<f64> {
    Rotation2::from_matrix_unchecked(sm::<f64, 2, 2>(x))
}

fn rot3(x: &[f64]) -> Rotation3<f64> {
    Rotation3::from_matrix_unchecked(sm::<f64, 3, 3>(x))
}

fn irot(name: &str, n: usize) -> Input {
    with(fm(name, n, n), Gen::Rot(n))
}

/// `n` unit quaternions.
fn iuqs(name: &str, n: usize) -> Input {
    with(fquats(name, n), Gen::Group(vec![Gen::U(4); n]))
}

/// A real exponent in [-2, 2].
fn iexp(name: &str) -> Input {
    with(fs(name), Gen::Range(-2.0, 2.0))
}

/// Rejects (nearly) parallel pairs: the observer frames and `rotation_between` are singular.
fn separated3(a: &Vector3<f64>, b: &Vector3<f64>) -> bool {
    let cos = a.dot(b) / (a.norm() * b.norm());
    cos.abs() < 0.95
}

// Exact bilinear kernels on i128 raws ----------------------------------------------------------

/// `a * conj(b)` for `(w, i, j, k)` quaternions.
fn hamilton_conj<T: Ring>(x: &[T]) -> Vec<T> {
    let (a, b) = (&x[..4], &x[4..8]);
    vec![
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3],
        -a[0] * b[1] + a[1] * b[0] - a[2] * b[3] + a[3] * b[2],
        -a[0] * b[2] + a[1] * b[3] + a[2] * b[0] - a[3] * b[1],
        -a[0] * b[3] - a[1] * b[2] + a[2] * b[1] + a[3] * b[0],
    ]
}

/// `q * q` for a `(w, i, j, k)` quaternion.
fn square<T: Ring>(x: &[T]) -> Vec<T> {
    let q = &x[..4];
    vec![
        q[0] * q[0] - q[1] * q[1] - q[2] * q[2] - q[3] * q[3],
        q[0] * q[1] + q[0] * q[1],
        q[0] * q[2] + q[0] * q[2],
        q[0] * q[3] + q[0] * q[3],
    ]
}

/// `(a * b + b * a) / 2 = (a0 b0 - a·b, a0 b + b0 a)`.
fn inner<T: Ring>(x: &[T]) -> Vec<T> {
    let (a, b) = (&x[..4], &x[4..8]);
    vec![
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + b[0] * a[1],
        a[0] * b[2] + b[0] * a[2],
        a[0] * b[3] + b[0] * a[3],
    ]
}

/// `(a * b - b * a) / 2 = (0, a × b)`.
fn outer<T: Ring>(x: &[T]) -> Vec<T> {
    let (a, b) = (&x[..4], &x[4..8]);
    vec![
        T::zero(),
        a[2] * b[3] - a[3] * b[2],
        a[3] * b[1] - a[1] * b[3],
        a[1] * b[2] - a[2] * b[1],
    ]
}

/// `a * conj(b)` for `(re, im)` complex numbers.
fn complex_div<T: Ring>(x: &[T]) -> Vec<T> {
    vec![x[0] * x[2] + x[1] * x[3], x[1] * x[2] - x[0] * x[3]]
}

/// `c * (m11, m21)`, `c = (re, im)` and a row-major 2x2 rotation matrix.
fn complex_mul_rot<T: Ring>(x: &[T]) -> Vec<T> {
    let (re, im) = (x[2], x[4]);
    vec![x[0] * re - x[1] * im, x[0] * im + x[1] * re]
}

/// `(m11, m21) * c` for a row-major 2x2 rotation matrix and `c = (re, im)`.
fn rot_mul_complex<T: Ring>(x: &[T]) -> Vec<T> {
    let (a, b, re, im) = (x[0], x[2], x[4], x[5]);
    vec![a * re - b * im, a * im + b * re]
}

/// `(m11, m21) * conj(c)`.
fn rot_div_complex<T: Ring>(x: &[T]) -> Vec<T> {
    let (a, b, re, im) = (x[0], x[2], x[4], x[5]);
    vec![a * re + b * im, b * re - a * im]
}

/// `a * bᵀ` for two row-major `n x n` matrices (`Rotation::rotation_to` / `/`).
fn mat_mul_tr<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    let (a, b) = (&x[..N * N], &x[N * N..2 * N * N]);
    let mut out = Vec::with_capacity(N * N);
    for i in 0..N {
        for j in 0..N {
            let mut acc = T::zero();
            for k in 0..N {
                acc += a[i * N + k] * b[j * N + k];
            }
            out.push(acc);
        }
    }
    out
}

/// `c * conj((m11, m21))`.
fn complex_div_rot<T: Ring>(x: &[T]) -> Vec<T> {
    let (re, im) = (x[2], x[4]);
    vec![x[0] * re + x[1] * im, x[1] * re - x[0] * im]
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

// Closest rotations: upstream's iteration, cross-checked against the polar decomposition ---------

/// Upstream's `from_matrix_eps` (bounded here: upstream's `from_matrix` loops until convergence),
/// accepted only when it agrees with the rotation maximising `tr(Rᵀ m)` computed by an SVD.
fn closest_rotation3(m: &Matrix3<f64>) -> Option<UnitQuaternion<f64>> {
    let q = UnitQuaternion::from_matrix_eps(m, f64::EPSILON, 1000, UnitQuaternion::identity());
    let svd = m.svd(true, true);
    let (u, vt) = (svd.u?, svd.v_t?);
    let mut d = Matrix3::identity();
    d[(2, 2)] = (u * vt).determinant().signum();
    let best = u * d * vt;
    let gap = (q.to_rotation_matrix().matrix() - best).amax();
    // A distinct smallest singular value: the maximiser is unique and well conditioned.
    let s = svd.singular_values;
    let conditioned = (s[1] - s[2]).abs() > 0.05 * s[0] && (s[0] - s[1]).abs() > 0.05 * s[0];
    (gap < 1e-9 && conditioned).then_some(q)
}

/// Upstream's `UnitComplex::from_matrix_eps`, accepted when it agrees with the closed form
/// `atan2(m21 - m12, m11 + m22)`.
fn closest_rotation2(m: &Matrix2<f64>) -> Option<UnitComplex<f64>> {
    let c = UnitComplex::from_matrix_eps(m, f64::EPSILON, 1000, UnitComplex::identity());
    let (a, b) = (m[(1, 0)] - m[(0, 1)], m[(0, 0)] + m[(1, 1)]);
    let best = UnitComplex::new(a.atan2(b));
    let conditioned = a.hypot(b) > 0.1 * m.amax();
    ((c.angle_to(&best)).abs() < 1e-9 && conditioned).then_some(c)
}

/// Upstream's `mean_of`, un-permuted: upstream builds `Quaternion::new(v[0], v[1], v[2], v[3])`
/// from an eigenvector stored as `(i, j, k, w)`, so its `(w, i, j, k)` is our `(i, j, k, w)`.
/// Sign: the largest component positive (the eigenvector's sign is arbitrary). Rejected when the
/// two largest eigenvalues are close (the mean is then ill-defined).
fn mean_of(qs: &[UnitQuaternion<f64>]) -> Option<Vec<f64>> {
    let m: nalgebra::Matrix4<f64> = qs
        .iter()
        .map(|q| q.as_vector() * q.as_vector().transpose())
        .sum();
    let mut ev: Vec<f64> = m.symmetric_eigenvalues().iter().copied().collect();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if ev[1] > 0.8 * ev[0] {
        return None;
    }
    let p = UnitQuaternion::mean_of(qs.iter().copied());
    let mut out = vec![p.k, p.w, p.i, p.j];
    let big = out
        .iter()
        .fold(0.0f64, |acc, v| if v.abs() > acc.abs() { *v } else { acc });
    if big < 0.0 {
        out.iter_mut().for_each(|v| *v = -*v);
    }
    Some(out)
}

// Suites -----------------------------------------------------------------------------------------

fn quaternion_function_ops() -> Vec<Op> {
    let unary = |name: &'static str, doc: &'static str, dists: &[Dist], tol: Tol| {
        Op::new(name, doc)
            .input(iq("q"))
            .out(fq("result"))
            .dists(dists)
            .tol(tol)
    };
    let small_unit = [Dist::Small, Dist::Unit];
    vec![
        unary("quaternion_exp", "q.exp()", &small_unit, TRANS)
            .eval(|x| Some(flat_q(&quat(x).exp()))),
        unary("quaternion_ln", "q.ln()", &Dist::NO_LARGE, TRANS)
            .eval(|x| Some(flat_q(&quat(x).ln()))),
        Op::new("quaternion_powf", "q.powf(n), n in [-2, 2]")
            .input(iq("q"))
            .input(iexp("n"))
            .out(fq("result"))
            .dists(&small_unit)
            .tol(COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).powf(x[4])))),
        unary(
            "quaternion_sqrt",
            "q.sqrt() (principal root)",
            &Dist::ALL,
            TRANS,
        )
        .eval(|x| Some(flat_q(&quat(x).sqrt()))),
        unary(
            "quaternion_squared",
            "q.squared() = q * q",
            &Dist::NO_LARGE,
            Tol::Exact,
        )
        .ring(checked(ring!(square), |x| flat_q(&quat(x).squared()))),
        unary(
            "quaternion_half",
            "q.half() = q / 2 (nearest; the oracle floors: 1 ulp)",
            &Dist::ALL,
            Tol::Ulp(1),
        )
        .eval(|x| Some(flat_q(&quat(x).half()))),
        Op::new("quaternion_inner", "a.inner(&b) = (a * b + b * a) / 2")
            .input(iq("a"))
            .input(iq("b"))
            .out(fq("result"))
            .dists(&Dist::NO_LARGE)
            .ring(checked(ring!(inner), |x| {
                flat_q(&quat(x).inner(&quat(&x[4..])))
            })),
        Op::new("quaternion_outer", "a.outer(&b) = (a * b - b * a) / 2")
            .input(iq("a"))
            .input(iq("b"))
            .out(fq("result"))
            .dists(&Dist::NO_LARGE)
            .ring(checked(ring!(outer), |x| {
                flat_q(&quat(x).outer(&quat(&x[4..])))
            })),
        Op::new(
            "quaternion_mul_conj",
            "a * b.conjugate() (the kernel of right_div and /)",
        )
        .input(iq("a"))
        .input(iq("b"))
        .out(fq("result"))
        .ring(checked(ring!(hamilton_conj), |x| {
            flat_q(&(quat(x) * quat(&x[4..]).conjugate()))
        })),
        Op::new(
            "quaternion_right_div",
            "a.right_div(&b).unwrap() = a * b^-1",
        )
        .input(iq("a"))
        .input(iq("b"))
        .out(fq("result"))
        .dists(&Dist::NO_LARGE)
        .tol(ROTATE)
        .eval(|x| quat(x).right_div(&quat(&x[4..])).map(|q| flat_q(&q))),
        Op::new("quaternion_left_div", "a.left_div(&b).unwrap() = b^-1 * a")
            .input(iq("a"))
            .input(iq("b"))
            .out(fq("result"))
            .dists(&Dist::NO_LARGE)
            .tol(ROTATE)
            .eval(|x| quat(x).left_div(&quat(&x[4..])).map(|q| flat_q(&q))),
        Op::new("quaternion_project", "a.project(&b).unwrap()")
            .input(iq("a"))
            .input(iq("b"))
            .out(fq("result"))
            .dists(&Dist::NO_LARGE)
            .tol(ROTATE)
            .eval(|x| quat(x).project(&quat(&x[4..])).map(|q| flat_q(&q))),
        Op::new("quaternion_reject", "a.reject(&b).unwrap()")
            .input(iq("a"))
            .input(iq("b"))
            .out(fq("result"))
            .dists(&Dist::NO_LARGE)
            .tol(ROTATE)
            .eval(|x| quat(x).reject(&quat(&x[4..])).map(|q| flat_q(&q))),
        Op::new(
            "quaternion_polar_decomposition",
            "q.polar_decomposition() -> (norm, half angle, axis.unwrap())",
        )
        .input(iq("q"))
        .out(fs("norm"))
        .out(fangle("angle"))
        .out(fv("axis", 3))
        .dists(&Dist::NO_LARGE)
        .tol(TRANS)
        .eval(|x| {
            let (n, angle, axis) = quat(x).polar_decomposition();
            let mut out = vec![n, angle];
            out.extend(flat(&axis?.into_inner()));
            Some(out)
        }),
        Op::new(
            "quaternion_from_polar_decomposition",
            "Quaternion::from_polar_decomposition(scale, theta, axis)",
        )
        .input(with(fs("scale"), Gen::Range(0.25, 4.0)))
        .input(with(fangle("theta"), Gen::Range(-PI, PI)))
        .input(iu("axis", 3))
        .out(fq("result"))
        .dists(&Dist::UNIT)
        .tol(TRANS)
        .eval(|x| {
            let axis = Unit::new_unchecked(v3(&x[2..]));
            Some(flat_q(&Quaternion::from_polar_decomposition(
                x[0], x[1], axis,
            )))
        }),
        unary("quaternion_cos", "q.cos()", &small_unit, TRANS)
            .eval(|x| Some(flat_q(&quat(x).cos()))),
        unary("quaternion_sin", "q.sin()", &small_unit, TRANS)
            .eval(|x| Some(flat_q(&quat(x).sin()))),
        unary("quaternion_tan", "q.tan()", &[Dist::Small], COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).tan()))),
        unary("quaternion_sinh", "q.sinh()", &small_unit, TRANS)
            .eval(|x| Some(flat_q(&quat(x).sinh()))),
        unary("quaternion_cosh", "q.cosh()", &small_unit, TRANS)
            .eval(|x| Some(flat_q(&quat(x).cosh()))),
        unary("quaternion_tanh", "q.tanh()", &[Dist::Small], COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).tanh()))),
        unary("quaternion_acos", "q.acos()", &small_unit, COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).acos()))),
        unary("quaternion_asin", "q.asin()", &small_unit, COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).asin()))),
        unary("quaternion_atan", "q.atan()", &small_unit, COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).atan()))),
        unary("quaternion_asinh", "q.asinh()", &small_unit, COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).asinh()))),
        unary("quaternion_acosh", "q.acosh()", &small_unit, COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).acosh()))),
        unary("quaternion_atanh", "q.atanh()", &small_unit, COMPOSED)
            .eval(|x| Some(flat_q(&quat(x).atanh()))),
    ]
}

fn unit_quaternion_completion_ops() -> Vec<Op> {
    vec![
        Op::new("unit_quaternion_div", "a / b = a * b.inverse()")
            .input(iuq("a"))
            .input(iuq("b"))
            .out(fq("result"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(hamilton_conj), |x| {
                flat_q(&(uquat(x) / uquat(&x[4..])).into_inner())
            })),
        Op::new("unit_quaternion_mul_rotation", "q * r (Rotation3)")
            .input(iuq("q"))
            .input(irot("r", 3))
            .out(fq("result"))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 4.0, base: 4.0 })
            .eval(|x| Some(flat_q(&(uquat(x) * rot3(&x[4..])).into_inner()))),
        Op::new("unit_quaternion_div_rotation", "q / r (Rotation3)")
            .input(iuq("q"))
            .input(irot("r", 3))
            .out(fq("result"))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 4.0, base: 4.0 })
            .eval(|x| Some(flat_q(&(uquat(x) / rot3(&x[4..])).into_inner()))),
        Op::new("unit_quaternion_mul_translation", "q * t (an Isometry3)")
            .input(iuq("q"))
            .input(iv("t", 3))
            .out(fiso3("result"))
            .tol(ROTATE)
            .eval(|x| {
                let t = Translation3::new(x[4], x[5], x[6]);
                Some(flat_iso3(&(uquat(x) * t)))
            }),
        Op::new("unit_quaternion_mul_isometry", "q * iso")
            .input(iuq("q"))
            .input(iiso3("iso"))
            .out(fiso3("result"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso3(&(uquat(x) * iso3(&x[4..]))))),
        Op::new(
            "unit_quaternion_div_isometry",
            "q / iso = q * iso.inverse()",
        )
        .input(iuq("q"))
        .input(iiso3("iso"))
        .out(fiso3("result"))
        .tol(ROTATE)
        .eval(|x| Some(flat_iso3(&(uquat(x) / iso3(&x[4..]))))),
        Op::new("unit_quaternion_mul_similarity", "q * sim")
            .input(iuq("q"))
            .input(isim3("sim"))
            .out(fsim3("result"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim3(&(uquat(x) * sim3(&x[4..]))))),
        Op::new(
            "unit_quaternion_div_similarity",
            "q / sim = q * sim.inverse()",
        )
        .input(iuq("q"))
        .input(isim3("sim"))
        .out(fsim3("result"))
        .tol(ROTATE)
        .eval(|x| Some(flat_sim3(&(uquat(x) / sim3(&x[4..]))))),
        Op::new(
            "unit_quaternion_rotation_between_axis",
            "UnitQuaternion::rotation_between_axis(&a, &b).unwrap(), |a.dot(b)| < 0.95",
        )
        .input(iu("a", 3))
        .input(iu("b", 3))
        .out(fq("rotation"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 6.0, base: 16.0 })
        .eval(|x| {
            let (a, b) = (v3(x), v3(&x[3..]));
            if !separated3(&a, &b) {
                return None;
            }
            let (a, b) = (Unit::new_unchecked(a), Unit::new_unchecked(b));
            UnitQuaternion::rotation_between_axis(&a, &b).map(|q| flat_q(&q))
        }),
        Op::new(
            "unit_quaternion_scaled_rotation_between_axis",
            "UnitQuaternion::scaled_rotation_between_axis(&a, &b, s).unwrap(), s in [0, 1]",
        )
        .input(iu("a", 3))
        .input(iu("b", 3))
        .input(it("s"))
        .out(fq("rotation"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 6.0, base: 32.0 })
        .eval(|x| {
            let (a, b) = (v3(x), v3(&x[3..]));
            if !separated3(&a, &b) {
                return None;
            }
            let (a, b) = (Unit::new_unchecked(a), Unit::new_unchecked(b));
            UnitQuaternion::scaled_rotation_between_axis(&a, &b, x[6]).map(|q| flat_q(&q))
        }),
        Op::new(
            "unit_quaternion_face_towards",
            "UnitQuaternion::face_towards(&dir, &up), dir and up not parallel",
        )
        .input(iv("dir", 3))
        .input(iv("up", 3))
        .out(fq("rotation"))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 16.0 })
        .eval(|x| {
            let (d, u) = (v3(x), v3(&x[3..]));
            separated3(&d, &u).then(|| flat_q(&UnitQuaternion::face_towards(&d, &u)))
        }),
        Op::new(
            "unit_quaternion_look_at_rh",
            "UnitQuaternion::look_at_rh(&dir, &up), dir and up not parallel",
        )
        .input(iv("dir", 3))
        .input(iv("up", 3))
        .out(fq("rotation"))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 16.0 })
        .eval(|x| {
            let (d, u) = (v3(x), v3(&x[3..]));
            separated3(&d, &u).then(|| flat_q(&UnitQuaternion::look_at_rh(&d, &u)))
        }),
        Op::new(
            "unit_quaternion_look_at_lh",
            "UnitQuaternion::look_at_lh(&dir, &up), dir and up not parallel",
        )
        .input(iv("dir", 3))
        .input(iv("up", 3))
        .out(fq("rotation"))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 16.0 })
        .eval(|x| {
            let (d, u) = (v3(x), v3(&x[3..]));
            separated3(&d, &u).then(|| flat_q(&UnitQuaternion::look_at_lh(&d, &u)))
        }),
        Op::new(
            "unit_quaternion_from_matrix",
            "UnitQuaternion::from_matrix(&m) (upstream's iteration, checked against the SVD \
             maximiser of tr(R^T m); sign as upstream from the identity guess)",
        )
        .input(with(fm("m", 3, 3), Gen::WellCond(3)))
        .out(fq("rotation"))
        .dists(&[Dist::Small, Dist::Unit, Dist::Medium])
        .tol(Tol::Sens { k: 4.0, base: 64.0 })
        .eval(|x| closest_rotation3(&sm::<f64, 3, 3>(x)).map(|q| flat_q(&q))),
        Op::new(
            "unit_quaternion_mean_of",
            "UnitQuaternion::mean_of([q1, q2, q3]) un-permuted (see the suite doc), largest \
             component positive, second eigenvalue below 0.8 of the first",
        )
        .input(iuqs("qs", 3))
        .out(fq("mean"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 8.0, base: 16.0 })
        .eval(|x| mean_of(&[uquat(x), uquat(&x[4..]), uquat(&x[8..])])),
        Op::new(
            "unit_quaternion_ln",
            "q.ln() = (0, axis * angle) (upstream's full angle)",
        )
        .input(iuq("q"))
        .out(fq("result"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 4.0, base: 24.0 })
        .eval(|x| Some(flat_q(&uquat(x).ln()))),
        Op::new("unit_quaternion_exp", "q.exp() (of the quaternion)")
            .input(iuq("q"))
            .out(fq("result"))
            .dists(&Dist::UNIT)
            .tol(TRANS)
            .eval(|x| Some(flat_q(&uquat(x).exp()))),
    ]
}

fn unit_complex_completion_ops() -> Vec<Op> {
    vec![
        Op::new("unit_complex_div", "a / b = a * b.inverse()")
            .input(iuc("a"))
            .input(iuc("b"))
            .out(fc("result"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(complex_div), |x| {
                flat_c(&(ucomplex(x) / ucomplex(&x[2..])))
            })),
        Op::new("unit_complex_mul_rotation", "c * r (Rotation2)")
            .input(iuc("c"))
            .input(irot("r", 2))
            .out(fc("result"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(complex_mul_rot), |x| {
                flat_c(&(ucomplex(x) * rot2(&x[2..])))
            })),
        Op::new("unit_complex_div_rotation", "c / r (Rotation2)")
            .input(iuc("c"))
            .input(irot("r", 2))
            .out(fc("result"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(complex_div_rot), |x| {
                flat_c(&(ucomplex(x) / rot2(&x[2..])))
            })),
        Op::new("unit_complex_mul_translation", "c * t (an Isometry2)")
            .input(iuc("c"))
            .input(iv("t", 2))
            .out(fiso2("result"))
            .tol(Tol::Ulp(2))
            .eval(|x| {
                let t = Translation2::new(x[2], x[3]);
                Some(flat_iso2(&(ucomplex(x) * t)))
            }),
        Op::new("unit_complex_mul_isometry", "c * iso")
            .input(iuc("c"))
            .input(iiso2("iso"))
            .out(fiso2("result"))
            .tol(Tol::Ulp(2))
            .eval(|x| Some(flat_iso2(&(ucomplex(x) * iso2(&x[2..]))))),
        Op::new("unit_complex_mul_similarity", "c * sim")
            .input(iuc("c"))
            .input(isim2("sim"))
            .out(fsim2("result"))
            .tol(Tol::Ulp(2))
            .eval(|x| Some(flat_sim2(&(ucomplex(x) * sim2(&x[2..]))))),
        Op::new(
            "unit_complex_from_complex",
            "UnitComplex::from_complex(Complex::new(re, im))",
        )
        .input(iv("q", 2))
        .out(fc("rotation"))
        .tol(Tol::Sens { k: 2.0, base: 2.0 })
        .eval(|x| Some(flat_c(&UnitComplex::from_complex(Complex::new(x[0], x[1]))))),
        Op::new(
            "unit_complex_rotation_between_axis",
            "UnitComplex::rotation_between_axis(&a, &b)",
        )
        .input(iu("a", 2))
        .input(iu("b", 2))
        .out(fc("rotation"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 4.0, base: 16.0 })
        .eval(|x| {
            let (a, b) = (Unit::new_unchecked(v2(x)), Unit::new_unchecked(v2(&x[2..])));
            Some(flat_c(&UnitComplex::rotation_between_axis(&a, &b)))
        }),
        Op::new(
            "unit_complex_from_matrix",
            "UnitComplex::from_matrix(&m) (upstream's iteration, checked against the closed form)",
        )
        .input(with(fm("m", 2, 2), Gen::WellCond(2)))
        .out(fc("rotation"))
        .dists(&[Dist::Small, Dist::Unit, Dist::Medium])
        .tol(Tol::Sens { k: 4.0, base: 32.0 })
        .eval(|x| closest_rotation2(&sm::<f64, 2, 2>(x)).map(|c| flat_c(&c))),
    ]
}

/// The angle between two rotation matrices, in f64.
fn rot3_angle_between(a: &Rotation3<f64>, b: &Rotation3<f64>) -> f64 {
    a.angle_to(b)
}

/// `r.euler_angles_ordered(seq, extrinsic)` when well conditioned: observable, the middle angle
/// away from its range ends (where `acos` is ill-conditioned) and the outer ones away from `±π`
/// (where the wrap-around of a rounded angle is ambiguous). Upstream asserts `n1 ⟂ n2` and
/// `n3 ⟂ n1`, so only the six Tait-Bryan sequences are accepted (a symmetric one such as `zxz`
/// panics upstream, and in the Cairo port).
fn euler_ordered(
    r: &Rotation3<f64>,
    seq: [Unit<Vector3<f64>>; 3],
    extrinsic: bool,
) -> Option<Vec<f64>> {
    let symmetric = seq[0] == seq[2];
    let (angles, observable) = r.euler_angles_ordered(seq, extrinsic);
    let mid = angles[1];
    let mid_ok = if symmetric {
        mid > 0.1 && mid < PI - 0.1
    } else {
        mid.abs() < PI / 2.0 - 0.1
    };
    let outer_ok = angles[0].abs() < PI - 0.01 && angles[2].abs() < PI - 0.01;
    (observable && mid_ok && outer_ok).then(|| angles.to_vec())
}

fn rotation_matrix_completion_ops() -> Vec<Op> {
    let (x, y, z) = (Vector3::x_axis(), Vector3::y_axis(), Vector3::z_axis());
    vec![
        Op::new(
            "rotation2_from_matrix",
            "Rotation2::from_matrix(&m) (upstream's iteration, checked against the closed form)",
        )
        .input(with(fm("m", 2, 2), Gen::WellCond(2)))
        .out(fm("rotation", 2, 2))
        .dists(&[Dist::Small, Dist::Unit, Dist::Medium])
        .tol(Tol::Sens { k: 4.0, base: 32.0 })
        .eval(|x| {
            let m = sm::<f64, 2, 2>(x);
            let r = Rotation2::from_matrix_eps(&m, f64::EPSILON, 1000, Rotation2::identity());
            closest_rotation2(&m).map(|_| flat(r.matrix()))
        }),
        Op::new("rotation2_slerp", "a.slerp(&b, t), angle between them < 3")
            .input(irot("a", 2))
            .input(irot("b", 2))
            .input(it("t"))
            .out(fm("rotation", 2, 2))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 6.0, base: 32.0 })
            .eval(|x| {
                let (a, b) = (rot2(x), rot2(&x[4..]));
                (a.angle_to(&b).abs() < 3.0).then(|| flat(a.slerp(&b, x[8]).matrix()))
            }),
        Op::new("rotation2_rotation_to", "a.rotation_to(&b) = b * a^-1")
            .input(irot("a", 2))
            .input(irot("b", 2))
            .out(fm("rotation", 2, 2))
            .dists(&Dist::UNIT)
            .ring(checked(
                (
                    Box::new(|x: &[f64]| {
                        let mut y = x[4..8].to_vec();
                        y.extend_from_slice(&x[..4]);
                        Some(mat_mul_tr::<f64, 2>(&y))
                    }) as crate::engine::EvalFn,
                    Box::new(|x: &[i128]| {
                        let mut y = x[4..8].to_vec();
                        y.extend_from_slice(&x[..4]);
                        Some(mat_mul_tr::<i128, 2>(&y))
                    }) as crate::engine::ExactFn,
                ),
                |x| flat(rot2(x).rotation_to(&rot2(&x[4..])).matrix()),
            )),
        Op::new("rotation2_div", "a / b = a * b^-1")
            .input(irot("a", 2))
            .input(irot("b", 2))
            .out(fm("rotation", 2, 2))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(mat_mul_tr, 2), |x| {
                flat((rot2(x) / rot2(&x[4..])).matrix())
            })),
        Op::new("rotation2_mul_unit_complex", "r * c (a UnitComplex)")
            .input(irot("r", 2))
            .input(iuc("c"))
            .out(fc("result"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(rot_mul_complex), |x| {
                flat_c(&(rot2(x) * ucomplex(&x[4..])))
            })),
        Op::new("rotation2_div_unit_complex", "r / c (a UnitComplex)")
            .input(irot("r", 2))
            .input(iuc("c"))
            .out(fc("result"))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(rot_div_complex), |x| {
                flat_c(&(rot2(x) / ucomplex(&x[4..])))
            })),
        Op::new("rotation3_new", "Rotation3::new(axisangle)")
            .input(with(fv("axisangle", 3), Gen::ScaledAxis))
            .out(fm("rotation", 3, 3))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 6.0, base: 8.0 })
            .eval(|x| Some(flat(Rotation3::new(v3(x)).matrix()))),
        Op::new("rotation3_powf", "r.powf(n), n in [-2, 2], angle of r < 3")
            .input(irot("r", 3))
            .input(iexp("n"))
            .out(fm("rotation", 3, 3))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 8.0, base: 32.0 })
            .eval(|x| {
                let r = rot3(x);
                (r.angle() > 0.05 && r.angle() < 3.0).then(|| flat(r.powf(x[9]).matrix()))
            }),
        Op::new("rotation3_angle_to", "a.angle_to(&b), in [0.05, 3]")
            .input(irot("a", 3))
            .input(irot("b", 3))
            .out(fangle("angle"))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 4.0, base: 24.0 })
            .eval(|x| {
                let angle = rot3_angle_between(&rot3(x), &rot3(&x[9..]));
                (angle > 0.05 && angle < 3.0).then_some(vec![angle])
            }),
        Op::new(
            "rotation3_axis_angle",
            "r.axis_angle().unwrap(), angle in [0.05, 3]",
        )
        .input(irot("r", 3))
        .out(fv("axis", 3))
        .out(fangle("angle"))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 4.0, base: 24.0 })
        .eval(|x| {
            let (axis, angle) = rot3(x).axis_angle()?;
            let mut out = flat(&axis.into_inner());
            out.push(angle);
            (angle > 0.05 && angle < 3.0).then_some(out)
        }),
        Op::new("rotation3_rotation_to", "a.rotation_to(&b) = b * a^-1")
            .input(irot("a", 3))
            .input(irot("b", 3))
            .out(fm("rotation", 3, 3))
            .dists(&Dist::UNIT)
            .ring(checked(
                (
                    Box::new(|x: &[f64]| {
                        let mut y = x[9..18].to_vec();
                        y.extend_from_slice(&x[..9]);
                        Some(mat_mul_tr::<f64, 3>(&y))
                    }) as crate::engine::EvalFn,
                    Box::new(|x: &[i128]| {
                        let mut y = x[9..18].to_vec();
                        y.extend_from_slice(&x[..9]);
                        Some(mat_mul_tr::<i128, 3>(&y))
                    }) as crate::engine::ExactFn,
                ),
                |x| flat(rot3(x).rotation_to(&rot3(&x[9..])).matrix()),
            )),
        Op::new("rotation3_div", "a / b = a * b^-1")
            .input(irot("a", 3))
            .input(irot("b", 3))
            .out(fm("rotation", 3, 3))
            .dists(&Dist::UNIT)
            .ring(checked(ring!(mat_mul_tr, 3), |x| {
                flat((rot3(x) / rot3(&x[9..])).matrix())
            })),
        Op::new(
            "rotation3_euler_angles_ordered_zyx",
            "r.euler_angles_ordered([z, y, x], false) (intrinsic), well-conditioned cases",
        )
        .input(irot("r", 3))
        .out(fv("angles", 3))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 8.0, base: 64.0 })
        .eval(move |x_| euler_ordered(&rot3(x_), [z, y, x], false)),
        Op::new(
            "rotation3_euler_angles_ordered_xyz_extrinsic",
            "r.euler_angles_ordered([x, y, z], true) (extrinsic), well-conditioned cases",
        )
        .input(irot("r", 3))
        .out(fv("angles", 3))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 8.0, base: 64.0 })
        .eval(move |x_| euler_ordered(&rot3(x_), [x, y, z], true)),
        Op::new(
            "rotation3_euler_angles_ordered_xzy",
            "r.euler_angles_ordered([x, z, y], false) (intrinsic), well-conditioned cases",
        )
        .input(irot("r", 3))
        .out(fv("angles", 3))
        .dists(&Dist::UNIT)
        .tol(Tol::Sens { k: 8.0, base: 64.0 })
        .eval(move |x_| euler_ordered(&rot3(x_), [x, z, y], false)),
        Op::new(
            "rotation3_from_matrix",
            "Rotation3::from_matrix(&m) (upstream's iteration, checked against the SVD maximiser)",
        )
        .input(with(fm("m", 3, 3), Gen::WellCond(3)))
        .out(fm("rotation", 3, 3))
        .dists(&[Dist::Small, Dist::Unit, Dist::Medium])
        .tol(Tol::Sens { k: 4.0, base: 64.0 })
        .eval(|x| {
            let m = sm::<f64, 3, 3>(x);
            closest_rotation3(&m)?;
            let r = Rotation3::from_matrix_eps(&m, f64::EPSILON, 1000, Rotation3::identity());
            Some(flat(r.matrix()))
        }),
        Op::new("rotation3_slerp", "a.slerp(&b, t), angle between them < 3")
            .input(irot("a", 3))
            .input(irot("b", 3))
            .input(it("t"))
            .out(fm("rotation", 3, 3))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 8.0, base: 32.0 })
            .eval(|x| {
                let (a, b) = (rot3(x), rot3(&x[9..]));
                (rot3_angle_between(&a, &b) < 3.0).then(|| flat(a.slerp(&b, x[18]).matrix()))
            }),
        Op::new(
            "rotation3_look_at_lh",
            "Rotation3::look_at_lh(&dir, &up), dir and up not parallel",
        )
        .input(iv("dir", 3))
        .input(iv("up", 3))
        .out(fm("rotation", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::Sens { k: 6.0, base: 8.0 })
        .eval(|x| {
            let (d, u) = (v3(x), v3(&x[3..]));
            separated3(&d, &u).then(|| flat(Rotation3::look_at_lh(&d, &u).matrix()))
        }),
        Op::new("rotation3_mul_unit_quaternion", "r * q (a UnitQuaternion)")
            .input(irot("r", 3))
            .input(iuq("q"))
            .out(fq("result"))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 4.0, base: 4.0 })
            .eval(|x| Some(flat_q(&(rot3(x) * uquat(&x[9..])).into_inner()))),
        Op::new("rotation3_div_unit_quaternion", "r / q (a UnitQuaternion)")
            .input(irot("r", 3))
            .input(iuq("q"))
            .out(fq("result"))
            .dists(&Dist::UNIT)
            .tol(Tol::Sens { k: 4.0, base: 4.0 })
            .eval(|x| Some(flat_q(&(rot3(x) / uquat(&x[9..])).into_inner()))),
    ]
}

pub fn suites() -> Vec<Suite> {
    vec![
        Suite {
            name: "quaternion_functions",
            description:
                "Quaternion completion: exp, ln, powf, sqrt, squared, half, inner, outer, \
                          divisions, projections, polar decomposition, trigonometric and \
                          hyperbolic functions and their inverses",
            ops: quaternion_function_ops(),
        },
        Suite {
            name: "unit_quaternion_completion",
            description: "UnitQuaternion completion: division, heterogeneous products with \
                          rotations, translations, isometries and similarities, \
                          rotation_between_axis, observer frames, from_matrix, mean_of, ln, exp",
            ops: unit_quaternion_completion_ops(),
        },
        Suite {
            name: "unit_complex_completion",
            description: "UnitComplex completion: division, heterogeneous products with \
                          rotations, translations, isometries and similarities, from_complex, \
                          rotation_between_axis, from_matrix",
            ops: unit_complex_completion_ops(),
        },
        Suite {
            name: "rotation_matrix_completion",
            description: "Rotation2 / Rotation3 completion: from_matrix, slerp, rotation_to, \
                          division, products with unit complex numbers and unit quaternions, new, \
                          powf, angle_to, axis_angle, euler_angles_ordered, look_at_lh",
            ops: rotation_matrix_completion_ops(),
        },
    ]
}
