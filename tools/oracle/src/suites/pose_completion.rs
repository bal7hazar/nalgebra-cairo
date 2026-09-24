//! WP 8.4-P09b: the completion of the poses (suite `pose_completion`) — the rotation-matrix
//! isometries / similarities `IsometryMatrix2/3` and `SimilarityMatrix2/3`, the new operators of
//! `Isometry2/3` / `Similarity2/3` (`/`, `rotation_wrt_point`, `look_at_lh`, the products with
//! each other and with rotations), the operators of `Rotation2/3` whose outputs are the
//! rotation-matrix poses, and the two fidelity fixes of the WP: the exponential (and `sinh` /
//! `cosh`) of a REAL quaternion, which upstream returns as the identity, and
//! `Rotation2::renormalize` (upstream's `from_matrix_eps` iteration).

use super::{
    flat, flat_iso2, flat_iso3, flat_q, flat_sim2, flat_sim3, iso2, iso3, quat, sim2, sim3, sm, sv,
    ucomplex, uquat,
};
use crate::engine::{
    fiso2, fiso3, fm, fq, fs, fsim2, fsim3, fv, iiso2, iiso3, isim2, isim3, it, iuc, iuq, iv, with,
    Input, Op, Suite, Tol,
};
use crate::gen::Gen;
use nalgebra::{
    IsometryMatrix2, IsometryMatrix3, Point2, Point3, Rotation2, Rotation3, SimilarityMatrix2,
    SimilarityMatrix3, Translation2, Translation3, UnitQuaternion, Vector3,
};

/// A few rounding stages on values of magnitude <= the inputs.
const ROTATE: Tol = Tol::Sens { k: 3.0, base: 2.0 };
/// Normalisations, cross products and a quaternion conversion (the observer frames).
const FRAME: Tol = Tol::Sens { k: 8.0, base: 16.0 };
/// The view transforms: the translation `rotation · (-eye)` multiplies the few-ulp error of the
/// normalised frame by `|eye|`.
const VIEW: Tol = Tol::SensMag {
    k: 8.0,
    base: 16.0,
    mag: 32.0,
};
/// One `acos` / `atan2` and a few `sin` (the spherical interpolations).
const SLERP: Tol = Tol::Sens { k: 4.0, base: 32.0 };
/// `exp` and `sin_cos` combined by a few products.
const TRANS: Tol = Tol::Sens { k: 8.0, base: 16.0 };

// Inputs / outputs ---------------------------------------------------------------------------------

fn irot(name: &str, n: usize) -> Input {
    with(fm(name, n, n), Gen::Rot(n))
}

/// A similarity scaling factor in [0.25, 4].
fn iscale(name: &str) -> Input {
    with(fs(name), Gen::Range(0.25, 4.0))
}

/// A real quaternion `(w, 0, 0, 0)`, `w` in [-3, 3].
fn ireal(name: &str) -> Input {
    with(
        fq(name),
        Gen::Group(vec![
            Gen::Range(-3.0, 3.0),
            Gen::Range(0.0, 0.0),
            Gen::Range(0.0, 0.0),
            Gen::Range(0.0, 0.0),
        ]),
    )
}

/// A drift of each entry of a rotation matrix, in [-2^-10, 2^-10].
fn idrift(name: &str) -> Input {
    let d = 1.0 / 1024.0;
    with(fm(name, 2, 2), Gen::Group(vec![Gen::Range(-d, d); 4]))
}

fn rot2(x: &[f64]) -> Rotation2<f64> {
    Rotation2::from_matrix_unchecked(sm::<f64, 2, 2>(x))
}

fn rot3(x: &[f64]) -> Rotation3<f64> {
    Rotation3::from_matrix_unchecked(sm::<f64, 3, 3>(x))
}

fn p2(x: &[f64]) -> Point2<f64> {
    Point2::new(x[0], x[1])
}

fn p3(x: &[f64]) -> Point3<f64> {
    Point3::new(x[0], x[1], x[2])
}

fn v3(x: &[f64]) -> Vector3<f64> {
    sv::<f64, 3>(x)
}

/// `(tx, ty, m11, m12, m21, m22)`: translation, then the row-major rotation matrix.
fn isom2(x: &[f64]) -> IsometryMatrix2<f64> {
    IsometryMatrix2::from_parts(Translation2::new(x[0], x[1]), rot2(&x[2..]))
}

/// `(tx, ty, tz, m11, .., m33)`.
fn isom3(x: &[f64]) -> IsometryMatrix3<f64> {
    IsometryMatrix3::from_parts(Translation3::new(x[0], x[1], x[2]), rot3(&x[3..]))
}

/// `(tx, ty, m11, .., m22, scaling)`.
fn simm2(x: &[f64]) -> SimilarityMatrix2<f64> {
    SimilarityMatrix2::from_isometry(isom2(x), x[6])
}

/// `(tx, ty, tz, m11, .., m33, scaling)`.
fn simm3(x: &[f64]) -> SimilarityMatrix3<f64> {
    SimilarityMatrix3::from_isometry(isom3(x), x[12])
}

fn flat_isom2(iso: &IsometryMatrix2<f64>) -> Vec<f64> {
    let mut out = flat(&iso.translation.vector);
    out.extend(flat(iso.rotation.matrix()));
    out
}

fn flat_isom3(iso: &IsometryMatrix3<f64>) -> Vec<f64> {
    let mut out = flat(&iso.translation.vector);
    out.extend(flat(iso.rotation.matrix()));
    out
}

fn flat_simm2(sim: &SimilarityMatrix2<f64>) -> Vec<f64> {
    let mut out = flat_isom2(&sim.isometry);
    out.push(sim.scaling());
    out
}

fn flat_simm3(sim: &SimilarityMatrix3<f64>) -> Vec<f64> {
    let mut out = flat_isom3(&sim.isometry);
    out.push(sim.scaling());
    out
}

/// An `IsometryMatrix{n}` as two inputs: the translation `{name}t` and the rotation `{name}r`.
fn iisom(name: &str, n: usize) -> [Input; 2] {
    [iv(&format!("{name}t"), n), irot(&format!("{name}r"), n)]
}

/// A `SimilarityMatrix{n}` as three inputs: translation, rotation, scaling.
fn isimm(name: &str, n: usize) -> [Input; 3] {
    let [t, r] = iisom(name, n);
    [t, r, iscale(&format!("{name}s"))]
}

trait Inputs {
    fn inputs<const K: usize>(self, inputs: [Input; K]) -> Self;
    fn pose_out(self, n: usize, scaled: bool) -> Self;
}

impl Inputs for Op {
    fn inputs<const K: usize>(self, inputs: [Input; K]) -> Op {
        inputs.into_iter().fold(self, |op, i| op.input(i))
    }

    /// The translation, the rotation matrix and (for a similarity) the scaling.
    fn pose_out(self, n: usize, scaled: bool) -> Op {
        let op = self.out(fv("translation", n)).out(fm("rotation", n, n));
        if scaled {
            op.out(fs("scaling"))
        } else {
            op
        }
    }
}

/// The quaternion slerp behind `Rotation3::slerp` is singular for (anti)parallel inputs and flips
/// its arc when the dot product changes sign: keep away from both.
fn slerp_ok(a: &Rotation3<f64>, b: &Rotation3<f64>) -> bool {
    let dot = UnitQuaternion::from_rotation_matrix(a)
        .coords
        .dot(&UnitQuaternion::from_rotation_matrix(b).coords);
    (0.05..0.95).contains(&dot.abs())
}

/// Rejects the singular observer frames: `up` (nearly) parallel to `target - eye`, or `eye` too
/// close to `target`.
fn frame_ok(x: &[f64]) -> bool {
    let (eye, target, up) = (v3(x), v3(&x[3..]), v3(&x[6..]));
    let dir = target - eye;
    let cos = dir.dot(&up) / (dir.norm() * up.norm());
    dir.norm() > 1.0e-3 && up.norm() > 1.0e-3 && cos.abs() < 0.95
}

// Ops ----------------------------------------------------------------------------------------------

fn isometry_matrix_ops() -> Vec<Op> {
    vec![
        Op::new("isometry_matrix2_mul", "a * b")
            .inputs(iisom("a", 2))
            .inputs(iisom("b", 2))
            .pose_out(2, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom2(&(isom2(x) * isom2(&x[6..]))))),
        Op::new("isometry_matrix2_inverse", "a.inverse()")
            .inputs(iisom("a", 2))
            .pose_out(2, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom2(&isom2(x).inverse()))),
        Op::new("isometry_matrix2_inv_mul", "a.inv_mul(&b)")
            .inputs(iisom("a", 2))
            .inputs(iisom("b", 2))
            .pose_out(2, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom2(&isom2(x).inv_mul(&isom2(&x[6..]))))),
        Op::new("isometry_matrix2_div", "a / b")
            .inputs(iisom("a", 2))
            .inputs(iisom("b", 2))
            .pose_out(2, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom2(&(isom2(x) / isom2(&x[6..]))))),
        Op::new("isometry_matrix2_transform_point", "a * p")
            .inputs(iisom("a", 2))
            .input(iv("p", 2))
            .out(fv("point", 2))
            .tol(ROTATE)
            .eval(|x| Some(flat(&(isom2(x) * p2(&x[6..])).coords))),
        Op::new(
            "isometry_matrix2_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .inputs(iisom("a", 2))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .tol(ROTATE)
        .eval(|x| Some(flat(&isom2(x).inverse_transform_point(&p2(&x[6..])).coords))),
        Op::new(
            "isometry_matrix2_lerp_slerp",
            "a.lerp_slerp(&b, t), the rotations less than 170 degrees apart",
        )
        .inputs(iisom("a", 2))
        .inputs(iisom("b", 2))
        .input(it("t"))
        .pose_out(2, false)
        .tol(SLERP)
        .eval(|x| {
            let (a, b) = (isom2(x), isom2(&x[6..]));
            (a.rotation.angle_to(&b.rotation).abs() < 2.96)
                .then(|| flat_isom2(&a.lerp_slerp(&b, x[12])))
        }),
        Op::new("isometry_matrix3_mul", "a * b")
            .inputs(iisom("a", 3))
            .inputs(iisom("b", 3))
            .pose_out(3, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom3(&(isom3(x) * isom3(&x[12..]))))),
        Op::new("isometry_matrix3_inverse", "a.inverse()")
            .inputs(iisom("a", 3))
            .pose_out(3, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom3(&isom3(x).inverse()))),
        Op::new("isometry_matrix3_inv_mul", "a.inv_mul(&b)")
            .inputs(iisom("a", 3))
            .inputs(iisom("b", 3))
            .pose_out(3, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom3(&isom3(x).inv_mul(&isom3(&x[12..]))))),
        Op::new("isometry_matrix3_div", "a / b")
            .inputs(iisom("a", 3))
            .inputs(iisom("b", 3))
            .pose_out(3, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom3(&(isom3(x) / isom3(&x[12..]))))),
        Op::new("isometry_matrix3_transform_point", "a * p")
            .inputs(iisom("a", 3))
            .input(iv("p", 3))
            .out(fv("point", 3))
            .tol(ROTATE)
            .eval(|x| Some(flat(&(isom3(x) * p3(&x[12..])).coords))),
        Op::new(
            "isometry_matrix3_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .inputs(iisom("a", 3))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&isom3(x).inverse_transform_point(&p3(&x[12..])).coords))),
        Op::new(
            "isometry_matrix3_face_towards",
            "IsometryMatrix3::face_towards(&eye, &target, &up)",
        )
        .input(iv("eye", 3))
        .input(iv("target", 3))
        .input(iv("up", 3))
        .pose_out(3, false)
        .tol(FRAME)
        .eval(|x| {
            frame_ok(x).then(|| {
                flat_isom3(&IsometryMatrix3::face_towards(
                    &p3(x),
                    &p3(&x[3..]),
                    &v3(&x[6..]),
                ))
            })
        }),
        Op::new(
            "isometry_matrix3_look_at_rh",
            "IsometryMatrix3::look_at_rh(&eye, &target, &up)",
        )
        .input(iv("eye", 3))
        .input(iv("target", 3))
        .input(iv("up", 3))
        .pose_out(3, false)
        .tol(VIEW)
        .eval(|x| {
            frame_ok(x).then(|| {
                flat_isom3(&IsometryMatrix3::look_at_rh(
                    &p3(x),
                    &p3(&x[3..]),
                    &v3(&x[6..]),
                ))
            })
        }),
        Op::new(
            "isometry_matrix3_look_at_lh",
            "IsometryMatrix3::look_at_lh(&eye, &target, &up)",
        )
        .input(iv("eye", 3))
        .input(iv("target", 3))
        .input(iv("up", 3))
        .pose_out(3, false)
        .tol(VIEW)
        .eval(|x| {
            frame_ok(x).then(|| {
                flat_isom3(&IsometryMatrix3::look_at_lh(
                    &p3(x),
                    &p3(&x[3..]),
                    &v3(&x[6..]),
                ))
            })
        }),
        Op::new(
            "isometry_matrix3_lerp_slerp",
            "a.lerp_slerp(&b, t), 0.05 < |dot of the quaternions| < 0.95",
        )
        .inputs(iisom("a", 3))
        .inputs(iisom("b", 3))
        .input(it("t"))
        .pose_out(3, false)
        .tol(SLERP)
        .eval(|x| {
            let (a, b) = (isom3(x), isom3(&x[12..]));
            slerp_ok(&a.rotation, &b.rotation).then(|| flat_isom3(&a.lerp_slerp(&b, x[24])))
        }),
    ]
}

fn similarity_matrix_ops() -> Vec<Op> {
    vec![
        Op::new("similarity_matrix2_mul", "a * b")
            .inputs(isimm("a", 2))
            .inputs(isimm("b", 2))
            .pose_out(2, true)
            .tol(ROTATE)
            .eval(|x| Some(flat_simm2(&(simm2(x) * simm2(&x[7..]))))),
        Op::new("similarity_matrix2_inverse", "a.inverse()")
            .inputs(isimm("a", 2))
            .pose_out(2, true)
            .tol(ROTATE)
            .eval(|x| Some(flat_simm2(&simm2(x).inverse()))),
        Op::new("similarity_matrix2_transform_point", "a * p")
            .inputs(isimm("a", 2))
            .input(iv("p", 2))
            .out(fv("point", 2))
            .tol(ROTATE)
            .eval(|x| Some(flat(&(simm2(x) * p2(&x[7..])).coords))),
        Op::new(
            "similarity_matrix2_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .inputs(isimm("a", 2))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .tol(ROTATE)
        .eval(|x| Some(flat(&simm2(x).inverse_transform_point(&p2(&x[7..])).coords))),
        Op::new("similarity_matrix3_mul", "a * b")
            .inputs(isimm("a", 3))
            .inputs(isimm("b", 3))
            .pose_out(3, true)
            .tol(ROTATE)
            .eval(|x| Some(flat_simm3(&(simm3(x) * simm3(&x[13..]))))),
        Op::new("similarity_matrix3_inverse", "a.inverse()")
            .inputs(isimm("a", 3))
            .pose_out(3, true)
            .tol(ROTATE)
            .eval(|x| Some(flat_simm3(&simm3(x).inverse()))),
        Op::new("similarity_matrix3_transform_point", "a * p")
            .inputs(isimm("a", 3))
            .input(iv("p", 3))
            .out(fv("point", 3))
            .tol(ROTATE)
            .eval(|x| Some(flat(&(simm3(x) * p3(&x[13..])).coords))),
        Op::new(
            "similarity_matrix3_inverse_transform_point",
            "a.inverse_transform_point(&p)",
        )
        .inputs(isimm("a", 3))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(ROTATE)
        .eval(|x| Some(flat(&simm3(x).inverse_transform_point(&p3(&x[13..])).coords))),
    ]
}

/// The new operators of `Isometry2/3` / `Similarity2/3` (unit complex / quaternion rotations).
fn pose_ops() -> Vec<Op> {
    vec![
        Op::new("isometry2_div", "a / b")
            .input(iiso2("a"))
            .input(iiso2("b"))
            .out(fiso2("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso2(&(iso2(x) / iso2(&x[4..]))))),
        Op::new("isometry2_div_unit_complex", "a / r")
            .input(iiso2("a"))
            .input(iuc("r"))
            .out(fiso2("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso2(&(iso2(x) / ucomplex(&x[4..]))))),
        Op::new(
            "isometry2_rotation_wrt_point",
            "Isometry2::rotation_wrt_point(r, p)",
        )
        .input(iuc("r"))
        .input(iv("p", 2))
        .out(fiso2("isometry"))
        .tol(ROTATE)
        .eval(|x| {
            Some(flat_iso2(&nalgebra::Isometry2::rotation_wrt_point(
                ucomplex(x),
                p2(&x[2..]),
            )))
        }),
        Op::new("isometry2_mul_similarity", "a * s")
            .input(iiso2("a"))
            .input(isim2("s"))
            .out(fsim2("product"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim2(&(iso2(x) * sim2(&x[4..]))))),
        Op::new("similarity2_div_isometry", "s / a")
            .input(isim2("s"))
            .input(iiso2("a"))
            .out(fsim2("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim2(&(sim2(x) / iso2(&x[5..]))))),
        Op::new("similarity2_div", "a / b")
            .input(isim2("a"))
            .input(isim2("b"))
            .out(fsim2("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim2(&(sim2(x) / sim2(&x[5..]))))),
        Op::new("isometry3_div", "a / b")
            .input(iiso3("a"))
            .input(iiso3("b"))
            .out(fiso3("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso3(&(iso3(x) / iso3(&x[7..]))))),
        Op::new("isometry3_div_unit_quaternion", "a / r")
            .input(iiso3("a"))
            .input(iuq("r"))
            .out(fiso3("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_iso3(&(iso3(x) / uquat(&x[7..]))))),
        Op::new(
            "isometry3_rotation_wrt_point",
            "Isometry3::rotation_wrt_point(r, p)",
        )
        .input(iuq("r"))
        .input(iv("p", 3))
        .out(fiso3("isometry"))
        .tol(ROTATE)
        .eval(|x| {
            Some(flat_iso3(&nalgebra::Isometry3::rotation_wrt_point(
                uquat(x),
                p3(&x[4..]),
            )))
        }),
        Op::new(
            "isometry3_look_at_lh",
            "Isometry3::look_at_lh(&eye, &target, &up)",
        )
        .input(iv("eye", 3))
        .input(iv("target", 3))
        .input(iv("up", 3))
        .out(fiso3("isometry"))
        .tol(VIEW)
        .eval(|x| {
            frame_ok(x).then(|| {
                flat_iso3(&nalgebra::Isometry3::look_at_lh(
                    &p3(x),
                    &p3(&x[3..]),
                    &v3(&x[6..]),
                ))
            })
        }),
        Op::new("isometry3_div_similarity", "a / s")
            .input(iiso3("a"))
            .input(isim3("s"))
            .out(fsim3("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim3(&(iso3(x) / sim3(&x[7..]))))),
        Op::new("similarity3_mul_isometry", "s * a")
            .input(isim3("s"))
            .input(iiso3("a"))
            .out(fsim3("product"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim3(&(sim3(x) * iso3(&x[8..]))))),
        Op::new("similarity3_div", "a / b")
            .input(isim3("a"))
            .input(isim3("b"))
            .out(fsim3("quotient"))
            .tol(ROTATE)
            .eval(|x| Some(flat_sim3(&(sim3(x) / sim3(&x[8..]))))),
    ]
}

/// The operators of `Rotation2/3` / `Translation2/3` whose outputs are the rotation-matrix poses.
fn rotation_pose_ops() -> Vec<Op> {
    vec![
        Op::new("rotation2_mul_isometry", "r * a (a: IsometryMatrix2)")
            .input(irot("r", 2))
            .inputs(iisom("a", 2))
            .pose_out(2, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom2(&(rot2(x) * isom2(&x[4..]))))),
        Op::new("rotation2_div_similarity", "r / s (s: SimilarityMatrix2)")
            .input(irot("r", 2))
            .inputs(isimm("s", 2))
            .pose_out(2, true)
            .tol(ROTATE)
            .eval(|x| Some(flat_simm2(&(rot2(x) / simm2(&x[4..]))))),
        Op::new("rotation3_mul_translation", "r * t")
            .input(irot("r", 3))
            .input(iv("t", 3))
            .pose_out(3, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom3(&(rot3(x) * Translation3::from(v3(&x[9..])))))),
        Op::new("rotation3_div_isometry", "r / a (a: IsometryMatrix3)")
            .input(irot("r", 3))
            .inputs(iisom("a", 3))
            .pose_out(3, false)
            .tol(ROTATE)
            .eval(|x| Some(flat_isom3(&(rot3(x) / isom3(&x[9..]))))),
        Op::new("rotation3_mul_similarity", "r * s (s: SimilarityMatrix3)")
            .input(irot("r", 3))
            .inputs(isimm("s", 3))
            .pose_out(3, true)
            .tol(ROTATE)
            .eval(|x| Some(flat_simm3(&(rot3(x) * simm3(&x[9..]))))),
    ]
}

/// The fidelity fixes of the WP.
fn fidelity_ops() -> Vec<Op> {
    vec![
        Op::new(
            "quaternion_exp_real",
            "q.exp() of a real quaternion: the identity upstream",
        )
        .input(ireal("q"))
        .out(fq("exp"))
        .eval(|x| Some(flat_q(&quat(x).exp()))),
        Op::new(
            "quaternion_sinh_real",
            "q.sinh() of a real quaternion: zero upstream",
        )
        .input(ireal("q"))
        .out(fq("sinh"))
        .eval(|x| Some(flat_q(&quat(x).sinh()))),
        Op::new(
            "quaternion_cosh_real",
            "q.cosh() of a real quaternion: the identity upstream",
        )
        .input(ireal("q"))
        .out(fq("cosh"))
        .eval(|x| Some(flat_q(&quat(x).cosh()))),
        Op::new(
            "quaternion_exp_nearly_real",
            "q.exp() of a quaternion whose imaginary part is a few ulp",
        )
        .input(with(
            fq("q"),
            Gen::Group(vec![
                Gen::Range(-3.0, 3.0),
                Gen::Range(2.0e-9, 1.0e-8),
                Gen::Range(-1.0e-8, 1.0e-8),
                Gen::Range(-1.0e-8, 1.0e-8),
            ]),
        ))
        .out(fq("exp"))
        .tol(TRANS)
        .eval(|x| Some(flat_q(&quat(x).exp()))),
        Op::new(
            "rotation2_renormalize",
            "(r + drift).renormalize(): upstream's from_matrix_eps(m, eps, 0, guess)",
        )
        .input(irot("r", 2))
        .input(idrift("drift"))
        .out(fm("renormalized", 2, 2))
        .tol(ROTATE)
        .eval(|x| {
            let m = sm::<f64, 2, 2>(x) + sm::<f64, 2, 2>(&x[4..]);
            let mut r = Rotation2::from_matrix_unchecked(m);
            r.renormalize();
            Some(flat(r.matrix()))
        }),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut ops = isometry_matrix_ops();
    ops.extend(similarity_matrix_ops());
    ops.extend(pose_ops());
    ops.extend(rotation_pose_ops());
    ops.extend(fidelity_ops());
    vec![Suite {
        name: "pose_completion",
        description: "IsometryMatrix2/3 and SimilarityMatrix2/3 (products, inverses, inv_mul, \
                      divisions, point transforms, observer frames, lerp_slerp), the new operators \
                      of Isometry2/3 and Similarity2/3 (divisions, rotation_wrt_point, look_at_lh, \
                      mixed products), the Rotation operators with rotation-matrix poses, the \
                      exponential / sinh / cosh of real quaternions and Rotation2::renormalize",
        ops,
    }]
}
