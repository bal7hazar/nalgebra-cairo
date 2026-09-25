//! WP 8.4-P12: `DualQuaternion` and `UnitDualQuaternion`, suite `dual_quaternion`.
//!
//! The products whose Cairo kernels floor each output component once (the dual-quaternion product
//! and the products by a unit quaternion) are evaluated exactly on `i128` raws: their expected
//! values are the exact floors, tolerance 0. The other ops are evaluated by upstream in f64; their
//! tolerance follows the rounding stages of the Cairo kernels (a few fused products, divisions,
//! the transcendental calls of `sclerp`).

use super::{flat, flat_iso3, flat_q, iso3, quat, sv, uquat};
use crate::engine::{fdq, fiso3, fm, fv, idq, iiso3, it, iudq, iuq, iv, Op, Suite, Tol};
use crate::gen::Dist;
use nalgebra::{DualQuaternion, Point3, Translation3, Unit, UnitDualQuaternion};

/// Chains of two or three fused products on unit-scale factors: every rounding is an absolute
/// ulp (fixed point), multiplied by factors of magnitude about 1 afterwards, whatever the
/// magnitude of the translation. (The sensitivity models of the other suites would count the
/// derivatives with respect to the rotation, of the size of the translation, as rounding
/// errors, which these kernels never commit on their exact inputs.)
const CHAINED: Tol = Tol::Ulp(8);
/// A rounded unit-scale factor (the norm of `nlerp`, the screw of `sclerp`) multiplies the dual
/// part: an error proportional to the magnitude of the inputs.
const NLERP: Tol = Tol::SensMag {
    k: 0.0,
    base: 8.0,
    mag: 4.0,
};
/// `acos`, `sin_cos` and the divisions by `|v|` of the screw parameters (a few ulp on the unit
/// screw), then the product of the rounded screw by the dual part of the first transform: an error
/// proportional to the magnitude of the inputs (measured: at most 2.4 ulp per unit of the largest
/// input).
const SCLERP: Tol = Tol::SensMag {
    k: 0.0,
    base: 64.0,
    mag: 8.0,
};

// Values <-> flat slices ------------------------------------------------------------------------

/// `(real (w, i, j, k), dual (w, i, j, k))`.
fn dq(x: &[f64]) -> DualQuaternion<f64> {
    DualQuaternion::from_real_and_dual(quat(x), quat(&x[4..]))
}

/// Taken as unit without renormalisation (what the Cairo side does).
fn udq(x: &[f64]) -> UnitDualQuaternion<f64> {
    Unit::new_unchecked(dq(x))
}

fn flat_dq(d: &DualQuaternion<f64>) -> Vec<f64> {
    let mut out = flat_q(&d.real);
    out.extend(flat_q(&d.dual));
    out
}

fn flat_udq(d: &UnitDualQuaternion<f64>) -> Vec<f64> {
    flat_dq(d.as_ref())
}

fn t3(x: &[f64]) -> Translation3<f64> {
    Translation3::new(x[0], x[1], x[2])
}

fn p3(x: &[f64]) -> Point3<f64> {
    Point3::new(x[0], x[1], x[2])
}

// Exact kernels on i128 raws (outputs at scale 2^64) ---------------------------------------------

type Q = [i128; 4];

fn q_of(x: &[i128]) -> Q {
    [x[0], x[1], x[2], x[3]]
}

/// Hamilton product, `(w, i, j, k)`.
fn ham(a: Q, b: Q) -> Q {
    let [aw, ai, aj, ak] = a;
    let [bw, bi, bj, bk] = b;
    [
        aw * bw - ai * bi - aj * bj - ak * bk,
        aw * bi + ai * bw + aj * bk - ak * bj,
        aw * bj - ai * bk + aj * bw + ak * bi,
        aw * bk + ai * bj - aj * bi + ak * bw,
    ]
}

fn conj(a: Q) -> Q {
    [a[0], -a[1], -a[2], -a[3]]
}

fn add(a: Q, b: Q) -> Q {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

/// `(ar, ad) * (br, bd) = (ar·br, ar·bd + ad·br)`.
fn dq_mul_exact(x: &[i128]) -> Option<Vec<i128>> {
    let (ar, ad, br, bd) = (q_of(x), q_of(&x[4..]), q_of(&x[8..]), q_of(&x[12..]));
    let mut out = ham(ar, br).to_vec();
    out.extend(add(ham(ar, bd), ham(ad, br)));
    Some(out)
}

/// `(ar, ad) * q = (ar·q, ad·q)`.
fn udq_mul_uq_exact(x: &[i128]) -> Option<Vec<i128>> {
    let (ar, ad, q) = (q_of(x), q_of(&x[4..]), q_of(&x[8..]));
    let mut out = ham(ar, q).to_vec();
    out.extend(ham(ad, q));
    Some(out)
}

/// `(ar, ad) / q = (ar·q*, ad·q*)`.
fn udq_div_uq_exact(x: &[i128]) -> Option<Vec<i128>> {
    let (ar, ad, q) = (q_of(x), q_of(&x[4..]), q_of(&x[8..]));
    let mut out = ham(ar, conj(q)).to_vec();
    out.extend(ham(ad, conj(q)));
    Some(out)
}

/// `q * (br, bd) = (q·br, q·bd)`.
fn uq_mul_udq_exact(x: &[i128]) -> Option<Vec<i128>> {
    let (q, br, bd) = (q_of(x), q_of(&x[4..]), q_of(&x[8..]));
    let mut out = ham(q, br).to_vec();
    out.extend(ham(q, bd));
    Some(out)
}

fn exact(
    f64_eval: impl Fn(&[f64]) -> Option<Vec<f64>> + 'static,
    exact_eval: fn(&[i128]) -> Option<Vec<i128>>,
) -> (crate::engine::EvalFn, crate::engine::ExactFn) {
    (Box::new(f64_eval), Box::new(exact_eval))
}

// Ops ----------------------------------------------------------------------------------------------

fn dual_quaternion_ops() -> Vec<Op> {
    vec![
        Op::new(
            "dual_quaternion_mul",
            "a * b = (a.real * b.real, a.real * b.dual + a.dual * b.real)",
        )
        .input(idq("a"))
        .input(idq("b"))
        .out(fdq("product"))
        .dists(&Dist::NO_LARGE)
        .ring(exact(
            |x| Some(flat_dq(&(dq(x) * dq(&x[8..])))),
            dq_mul_exact,
        )),
        Op::new("dual_quaternion_normalize", "dq.normalize()")
            .input(idq("dq"))
            .out(fdq("normalized"))
            .tol(Tol::Sens { k: 3.0, base: 4.0 })
            .eval(|x| Some(flat_dq(&dq(x).normalize()))),
        Op::new("dual_quaternion_try_inverse", "dq.try_inverse().unwrap()")
            .input(idq("dq"))
            .out(fdq("inverse"))
            .dists(&[Dist::Unit, Dist::Medium])
            .tol(Tol::Sens { k: 6.0, base: 8.0 })
            .eval(|x| dq(x).try_inverse().map(|i| flat_dq(&i))),
        Op::new("dual_quaternion_lerp", "a.lerp(&b, t)")
            .input(idq("a"))
            .input(idq("b"))
            .input(it("t"))
            .out(fdq("lerp"))
            .tol(Tol::Ulp(2))
            .eval(|x| Some(flat_dq(&dq(x).lerp(&dq(&x[8..]), x[16])))),
    ]
}

fn unit_dual_quaternion_ops() -> Vec<Op> {
    vec![
        Op::new(
            "unit_dual_quaternion_from_parts",
            "UnitDualQuaternion::from_parts(t, r)",
        )
        .input(iv("t", 3))
        .input(iuq("r"))
        .out(fdq("dq"))
        .tol(Tol::Ulp(1))
        .eval(|x| {
            Some(flat_udq(&UnitDualQuaternion::from_parts(
                t3(x),
                uquat(&x[3..]),
            )))
        }),
        Op::new("unit_dual_quaternion_translation", "dq.translation()")
            .input(iudq("dq"))
            .out(fv("translation", 3))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(flat(&udq(x).translation().vector))),
        Op::new("unit_dual_quaternion_mul", "a * b")
            .input(iudq("a"))
            .input(iudq("b"))
            .out(fdq("product"))
            .ring(exact(
                |x| Some(flat_udq(&(udq(x) * udq(&x[8..])))),
                dq_mul_exact,
            )),
        Op::new("unit_dual_quaternion_inverse", "dq.inverse()")
            .input(iudq("dq"))
            .out(fdq("inverse"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&udq(x).inverse()))),
        Op::new("unit_dual_quaternion_div", "a / b")
            .input(iudq("a"))
            .input(iudq("b"))
            .out(fdq("quotient"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(udq(x) / udq(&x[8..]))))),
        Op::new(
            "unit_dual_quaternion_transform_point",
            "dq.transform_point(&p) = dq * p",
        )
        .input(iudq("dq"))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(CHAINED)
        .eval(|x| Some(flat(&udq(x).transform_point(&p3(&x[8..])).coords))),
        Op::new(
            "unit_dual_quaternion_inverse_transform_point",
            "dq.inverse_transform_point(&p)",
        )
        .input(iudq("dq"))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(CHAINED)
        .eval(|x| Some(flat(&udq(x).inverse_transform_point(&p3(&x[8..])).coords))),
        Op::new(
            "unit_dual_quaternion_transform_vector",
            "dq.transform_vector(&v) = dq * v",
        )
        .input(iudq("dq"))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .tol(CHAINED)
        .eval(|x| Some(flat(&udq(x).transform_vector(&sv::<f64, 3>(&x[8..]))))),
        Op::new("unit_dual_quaternion_to_homogeneous", "dq.to_homogeneous()")
            .input(iudq("dq"))
            .out(fm("matrix", 4, 4))
            .tol(CHAINED)
            .eval(|x| Some(flat(&udq(x).to_homogeneous()))),
        Op::new("unit_dual_quaternion_nlerp", "a.nlerp(&b, t)")
            .input(iudq("a"))
            .input(iudq("b"))
            .input(it("t"))
            .out(fdq("nlerp"))
            .tol(NLERP)
            .eval(|x| Some(flat_udq(&udq(x).nlerp(&udq(&x[8..]), x[16])))),
        Op::new("unit_dual_quaternion_sclerp", "a.sclerp(&b, t)")
            .input(iudq("a"))
            .input(iudq("b"))
            .input(it("t"))
            .out(fdq("sclerp"))
            .tol(SCLERP)
            .eval(|x| {
                udq(x)
                    .try_sclerp(&udq(&x[8..]), x[16], 1.0 / 4_294_967_296.0)
                    .map(|r| flat_udq(&r))
            }),
    ]
}

fn mixed_ops() -> Vec<Op> {
    vec![
        Op::new("unit_dual_quaternion_mul_unit_quaternion", "dq * q")
            .input(iudq("dq"))
            .input(iuq("q"))
            .out(fdq("product"))
            .ring(exact(
                |x| Some(flat_udq(&(udq(x) * uquat(&x[8..])))),
                udq_mul_uq_exact,
            )),
        Op::new("unit_dual_quaternion_div_unit_quaternion", "dq / q")
            .input(iudq("dq"))
            .input(iuq("q"))
            .out(fdq("quotient"))
            .ring(exact(
                |x| Some(flat_udq(&(udq(x) / uquat(&x[8..])))),
                udq_div_uq_exact,
            )),
        Op::new("unit_quaternion_mul_unit_dual_quaternion", "q * dq")
            .input(iuq("q"))
            .input(iudq("dq"))
            .out(fdq("product"))
            .ring(exact(
                |x| Some(flat_udq(&(uquat(x) * udq(&x[4..])))),
                uq_mul_udq_exact,
            )),
        Op::new("unit_quaternion_div_unit_dual_quaternion", "q / dq")
            .input(iuq("q"))
            .input(iudq("dq"))
            .out(fdq("quotient"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(uquat(x) / udq(&x[4..]))))),
        Op::new("unit_dual_quaternion_mul_translation", "dq * t")
            .input(iudq("dq"))
            .input(iv("t", 3))
            .out(fdq("product"))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(flat_udq(&(udq(x) * t3(&x[8..]))))),
        Op::new("unit_dual_quaternion_div_translation", "dq / t")
            .input(iudq("dq"))
            .input(iv("t", 3))
            .out(fdq("quotient"))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(flat_udq(&(udq(x) / t3(&x[8..]))))),
        Op::new("translation_mul_unit_dual_quaternion", "t * dq")
            .input(iv("t", 3))
            .input(iudq("dq"))
            .out(fdq("product"))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(flat_udq(&(t3(x) * udq(&x[3..]))))),
        Op::new("translation_div_unit_dual_quaternion", "t / dq")
            .input(iv("t", 3))
            .input(iudq("dq"))
            .out(fdq("quotient"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(t3(x) / udq(&x[3..]))))),
        Op::new("unit_dual_quaternion_mul_isometry", "dq * iso")
            .input(iudq("dq"))
            .input(iiso3("iso"))
            .out(fdq("product"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(udq(x) * iso3(&x[8..]))))),
        Op::new("unit_dual_quaternion_div_isometry", "dq / iso")
            .input(iudq("dq"))
            .input(iiso3("iso"))
            .out(fdq("quotient"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(udq(x) / iso3(&x[8..]))))),
        Op::new("isometry_mul_unit_dual_quaternion", "iso * dq")
            .input(iiso3("iso"))
            .input(iudq("dq"))
            .out(fdq("product"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(iso3(x) * udq(&x[7..]))))),
        Op::new("isometry_div_unit_dual_quaternion", "iso / dq")
            .input(iiso3("iso"))
            .input(iudq("dq"))
            .out(fdq("quotient"))
            .tol(CHAINED)
            .eval(|x| Some(flat_udq(&(iso3(x) / udq(&x[7..]))))),
        Op::new("unit_dual_quaternion_to_isometry", "dq.to_isometry()")
            .input(iudq("dq"))
            .out(fiso3("iso"))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(flat_iso3(&udq(x).to_isometry()))),
        Op::new(
            "dual_quaternion_div_unit_dual_quaternion",
            "a / b (a general, b unit)",
        )
        .input(idq("a"))
        .input(iudq("b"))
        .out(fdq("quotient"))
        .dists(&Dist::NO_LARGE)
        .tol(NLERP)
        .eval(|x| Some(flat_dq(&(dq(x) / udq(&x[8..]))))),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut ops = dual_quaternion_ops();
    ops.extend(unit_dual_quaternion_ops());
    ops.extend(mixed_ops());
    vec![Suite {
        name: "dual_quaternion",
        description:
            "DualQuaternion (product, normalize, try_inverse, lerp) and UnitDualQuaternion \
              (from_parts, translation, products, inverse, divisions, point transforms, \
              to_homogeneous, nlerp, sclerp, the products with unit quaternions, translations \
              and isometries)",
        ops,
    }]
}
