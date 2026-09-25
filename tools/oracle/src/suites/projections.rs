//! WP 8.4-P11b: `Perspective3` and `Orthographic3` (construction, accessors, setters,
//! `project_point`, `project_vector`, `unproject_point`, `inverse`), suite `projections`.
//!
//! Except for the constructors, the ops take the structural entries of the projection matrix
//! directly (`(m11, m22, m33, m34)` for a perspective, `(m11, m14, m22, m24, m33, m34)` for an
//! orthographic projection, built with `from_matrix_unchecked`), so each op is checked on its own
//! and not through the rounding of a constructor. Every Cairo kernel is a short chain of
//! correctly rounded quotients and floored products (one fused `mul_add` where upstream sums a
//! product and a term), so the tolerances are `Sens` budgets: a few one-ulp roundings on
//! intermediates, amplified like input perturbations.

use crate::engine::{fm, fs, fv, iv, with, Input, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{Matrix4, Orthographic3, Perspective3, Point3, Vector3};

use super::flat;

/// A quotient and a product per output, amplified by the sensitivity.
const SHORT: Tol = Tol::Sens { k: 2.0, base: 2.0 };
/// Chains of up to four roundings (constructors, setters, `fovy`'s `atan`).
const CHAIN: Tol = Tol::Sens { k: 4.0, base: 4.0 };
/// One correctly rounded quotient (or one fused `mul_add`) per output against the floor of the
/// exact value.
const ONE: Tol = Tol::Ulp(1);

// Inputs ---------------------------------------------------------------------------------------------

/// `(m11, m22, m33, m34)` of a perspective: `m11, m22` in [0.3, 4] (fields of view from 28° to
/// 147°, any aspect ratio in between), `m33` in [-1.5, -1.001], `m34` in [-40, -0.02] (znear
/// in about [0.01, 20], zfar beyond it).
fn ipers(name: &str) -> Input {
    with(
        fv(name, 4),
        Gen::Group(vec![
            Gen::Range(0.3, 4.0),
            Gen::Range(0.3, 4.0),
            Gen::Range(-1.5, -1.001),
            Gen::Range(-40.0, -0.02),
        ]),
    )
}

/// `(m11, m14, m22, m24, m33, m34)` of an orthographic projection: scales in [0.01, 4] (`m33`
/// negative, as for `znear < zfar`), offsets in [-3, 3].
fn iortho(name: &str) -> Input {
    with(
        fv(name, 6),
        Gen::Group(vec![
            Gen::Range(0.01, 4.0),
            Gen::Range(-3.0, 3.0),
            Gen::Range(0.01, 4.0),
            Gen::Range(-3.0, 3.0),
            Gen::Range(-4.0, -0.01),
            Gen::Range(-3.0, 3.0),
        ]),
    )
}

fn iaspect(name: &str) -> Input {
    with(fs(name), Gen::Range(0.3, 3.0))
}

fn ifov(name: &str) -> Input {
    with(fs(name), Gen::Range(0.3, 2.5))
}

fn iznear(name: &str) -> Input {
    with(fs(name), Gen::Range(0.01, 10.0))
}

fn izfar(name: &str) -> Input {
    with(fs(name), Gen::Range(20.0, 1000.0))
}

/// A plane or corner offset in [-50, 50].
fn ioffset(name: &str) -> Input {
    with(fs(name), Gen::Range(-50.0, 50.0))
}

// Builders -------------------------------------------------------------------------------------------

#[rustfmt::skip]
fn pers(x: &[f64]) -> Perspective3<f64> {
    Perspective3::from_matrix_unchecked(Matrix4::new(
        x[0], 0.0, 0.0, 0.0,
        0.0, x[1], 0.0, 0.0,
        0.0, 0.0, x[2], x[3],
        0.0, 0.0, -1.0, 0.0,
    ))
}

#[rustfmt::skip]
fn ortho(x: &[f64]) -> Orthographic3<f64> {
    Orthographic3::from_matrix_unchecked(Matrix4::new(
        x[0], 0.0, 0.0, x[1],
        0.0, x[2], 0.0, x[3],
        0.0, 0.0, x[4], x[5],
        0.0, 0.0, 0.0, 1.0,
    ))
}

fn pers_entries(p: &Perspective3<f64>) -> Vec<f64> {
    let m = p.as_matrix();
    vec![m[(0, 0)], m[(1, 1)], m[(2, 2)], m[(2, 3)]]
}

fn ortho_entries(o: &Orthographic3<f64>) -> Vec<f64> {
    let m = o.as_matrix();
    vec![
        m[(0, 0)],
        m[(0, 3)],
        m[(1, 1)],
        m[(1, 3)],
        m[(2, 2)],
        m[(2, 3)],
    ]
}

fn pt(x: &[f64]) -> Point3<f64> {
    Point3::new(x[0], x[1], x[2])
}

fn vec3(x: &[f64]) -> Vector3<f64> {
    Vector3::new(x[0], x[1], x[2])
}

// Perspective3 ---------------------------------------------------------------------------------------

/// A perspective setter: `(proj, args) -> proj`.
fn pers_setter(name: &str, arg: Input, f: fn(&mut Perspective3<f64>, f64)) -> Op {
    Op::new(
        format!("perspective3_{name}"),
        format!("p.{name}(x), entries (m11, m22, m33, m34) before and after"),
    )
    .input(ipers("proj"))
    .input(arg)
    .out(fv("proj", 4))
    .tol(CHAIN)
    .dists(&Dist::UNIT)
    .eval(move |x| {
        let mut p = pers(x);
        f(&mut p, x[4]);
        Some(pers_entries(&p))
    })
}

fn perspective_ops() -> Vec<Op> {
    vec![
        Op::new(
            "perspective3_new",
            "Perspective3::new(aspect, fovy, znear, zfar).into_inner()",
        )
        .input(iaspect("aspect"))
        .input(ifov("fovy"))
        .input(iznear("znear"))
        .input(izfar("zfar"))
        .out(fm("matrix", 4, 4))
        .tol(CHAIN)
        .dists(&Dist::UNIT)
        .eval(|x| {
            Some(flat(
                &Perspective3::new(x[0], x[1], x[2], x[3]).into_inner(),
            ))
        }),
        Op::new(
            "perspective3_accessors",
            "(p.aspect(), p.fovy(), p.znear(), p.zfar()), p = entries (m11, m22, m33, m34)",
        )
        .input(ipers("proj"))
        .out(fs("aspect"))
        .out(fs("fovy"))
        .out(fs("znear"))
        .out(fs("zfar"))
        .tol(Tol::Ulp(4))
        .dists(&Dist::UNIT)
        .eval(|x| {
            let p = pers(x);
            Some(vec![p.aspect(), p.fovy(), p.znear(), p.zfar()])
        }),
        Op::new(
            "perspective3_inverse",
            "p.inverse(), p = entries (m11, m22, m33, m34)",
        )
        .input(ipers("proj"))
        .out(fm("inverse", 4, 4))
        .tol(ONE)
        .dists(&Dist::UNIT)
        .eval(|x| Some(flat(&pers(x).inverse()))),
        Op::new(
            "perspective3_project_point",
            "p.project_point(&pt), p = entries (m11, m22, m33, m34)",
        )
        .input(ipers("proj"))
        .input(iv("pt", 3))
        .out(fv("projected", 3))
        .tol(SHORT)
        .dists(&Dist::NO_LARGE)
        .eval(|x| Some(flat(&pers(x).project_point(&pt(&x[4..])).coords))),
        Op::new(
            "perspective3_project_vector",
            "p.project_vector(&v), p = entries (m11, m22, m33, m34)",
        )
        .input(ipers("proj"))
        .input(iv("v", 3))
        .out(fv("projected", 3))
        .tol(SHORT)
        .dists(&Dist::NO_LARGE)
        .eval(|x| Some(flat(&pers(x).project_vector(&vec3(&x[4..]))))),
        Op::new(
            "perspective3_unproject_point",
            "p.unproject_point(&pt), p = entries (m11, m22, m33, m34)",
        )
        .input(ipers("proj"))
        .input(iv("pt", 3))
        .out(fv("unprojected", 3))
        .tol(Tol::Model(
            Box::new(|x, _| {
                2.0 + ((0.5 * x[4].abs() + 1.0) / x[0]).max((0.5 * x[5].abs() + 1.0) / x[1])
            }),
            "2 + max_i (|p_i| / 2 + 1) / m_ii: the rounding of w = m34 / (z + m33) (1/2 ulp) \
             amplified by |p_i| / m_ii, the floored product p_i * w divided by m_ii, the final \
             quotient and the floor of the expected value",
        ))
        .dists(&Dist::NO_LARGE)
        .eval(|x| Some(flat(&pers(x).unproject_point(&pt(&x[4..])).coords))),
        pers_setter("set_aspect", iaspect("aspect"), |p, v| p.set_aspect(v)).tol(ONE),
        pers_setter("set_fovy", ifov("fovy"), |p, v| p.set_fovy(v)),
        pers_setter("set_znear", iznear("znear"), |p, v| p.set_znear(v)),
        pers_setter("set_zfar", izfar("zfar"), |p, v| p.set_zfar(v)),
        Op::new(
            "perspective3_set_znear_and_zfar",
            "p.set_znear_and_zfar(znear, zfar), entries (m11, m22, m33, m34) before and after",
        )
        .input(ipers("proj"))
        .input(iznear("znear"))
        .input(izfar("zfar"))
        .out(fv("proj", 4))
        .tol(Tol::Ulp(2))
        .dists(&Dist::UNIT)
        .eval(|x| {
            let mut p = pers(x);
            p.set_znear_and_zfar(x[4], x[5]);
            Some(pers_entries(&p))
        }),
    ]
}

// Orthographic3 --------------------------------------------------------------------------------------

/// An orthographic setter of one value: `(proj, x) -> proj`. `kept` is the offset the setter
/// keeps (upstream asserts that the two differ).
fn ortho_setter(
    name: &str,
    arg: Input,
    kept: fn(&Orthographic3<f64>) -> f64,
    f: fn(&mut Orthographic3<f64>, f64),
) -> Op {
    Op::new(
        format!("orthographic3_{name}"),
        format!("o.{name}(x), entries (m11, m14, m22, m24, m33, m34) before and after"),
    )
    .input(iortho("proj"))
    .input(arg)
    .out(fv("proj", 6))
    .tol(CHAIN)
    .dists(&Dist::UNIT)
    .eval(move |x| {
        let mut o = ortho(x);
        if (kept(&o) - x[6]).abs() < 0.01 {
            return None;
        }
        f(&mut o, x[6]);
        Some(ortho_entries(&o))
    })
}

/// An orthographic setter of two values: `(proj, lo, hi) -> proj`.
fn ortho_setter2(name: &str, f: fn(&mut Orthographic3<f64>, f64, f64)) -> Op {
    Op::new(
        format!("orthographic3_{name}"),
        format!("o.{name}(lo, hi), entries (m11, m14, m22, m24, m33, m34) before and after"),
    )
    .input(iortho("proj"))
    .input(ioffset("lo"))
    .input(ioffset("hi"))
    .out(fv("proj", 6))
    .tol(ONE)
    .dists(&Dist::UNIT)
    .eval(move |x| {
        if (x[6] - x[7]).abs() < 0.01 {
            return None;
        }
        let mut o = ortho(x);
        f(&mut o, x[6], x[7]);
        Some(ortho_entries(&o))
    })
}

fn orthographic_ops() -> Vec<Op> {
    vec![
        Op::new(
            "orthographic3_new",
            "Orthographic3::new(left, right, bottom, top, znear, zfar).into_inner()",
        )
        .input(ioffset("left"))
        .input(ioffset("right"))
        .input(ioffset("bottom"))
        .input(ioffset("top"))
        .input(ioffset("znear"))
        .input(ioffset("zfar"))
        .out(fm("matrix", 4, 4))
        .tol(ONE)
        .dists(&Dist::UNIT)
        .eval(|x| {
            if (x[0] - x[1]).abs() < 0.01
                || (x[2] - x[3]).abs() < 0.01
                || (x[4] - x[5]).abs() < 0.01
            {
                return None;
            }
            Some(flat(
                &Orthographic3::new(x[0], x[1], x[2], x[3], x[4], x[5]).into_inner(),
            ))
        }),
        Op::new(
            "orthographic3_from_fov",
            "Orthographic3::from_fov(aspect, vfov, znear, zfar).into_inner()",
        )
        .input(iaspect("aspect"))
        .input(ifov("vfov"))
        .input(iznear("znear"))
        .input(izfar("zfar"))
        .out(fm("matrix", 4, 4))
        .tol(CHAIN)
        .dists(&Dist::UNIT)
        .eval(|x| {
            Some(flat(
                &Orthographic3::from_fov(x[0], x[1], x[2], x[3]).into_inner(),
            ))
        }),
        Op::new(
            "orthographic3_accessors",
            "(o.left(), o.right(), o.bottom(), o.top(), o.znear(), o.zfar()), o = entries \
             (m11, m14, m22, m24, m33, m34)",
        )
        .input(iortho("proj"))
        .out(fv("planes", 6))
        .tol(ONE)
        .dists(&Dist::UNIT)
        .eval(|x| {
            let o = ortho(x);
            Some(vec![
                o.left(),
                o.right(),
                o.bottom(),
                o.top(),
                o.znear(),
                o.zfar(),
            ])
        }),
        Op::new(
            "orthographic3_inverse",
            "o.inverse(), o = entries (m11, m14, m22, m24, m33, m34)",
        )
        .input(iortho("proj"))
        .out(fm("inverse", 4, 4))
        .tol(ONE)
        .dists(&Dist::UNIT)
        .eval(|x| Some(flat(&ortho(x).inverse()))),
        Op::new(
            "orthographic3_project_point",
            "o.project_point(&pt), o = entries (m11, m14, m22, m24, m33, m34)",
        )
        .input(iortho("proj"))
        .input(iv("pt", 3))
        .out(fv("projected", 3))
        .tol(ONE)
        .dists(&Dist::ALL)
        .eval(|x| Some(flat(&ortho(x).project_point(&pt(&x[6..])).coords))),
        Op::new(
            "orthographic3_project_vector",
            "o.project_vector(&v), o = entries (m11, m14, m22, m24, m33, m34)",
        )
        .input(iortho("proj"))
        .input(iv("v", 3))
        .out(fv("projected", 3))
        .tol(ONE)
        .dists(&Dist::ALL)
        .eval(|x| Some(flat(&ortho(x).project_vector(&vec3(&x[6..]))))),
        Op::new(
            "orthographic3_unproject_point",
            "o.unproject_point(&pt), o = entries (m11, m14, m22, m24, m33, m34)",
        )
        .input(iortho("proj"))
        .input(iv("pt", 3))
        .out(fv("unprojected", 3))
        .tol(ONE)
        .dists(&Dist::NO_LARGE)
        .eval(|x| Some(flat(&ortho(x).unproject_point(&pt(&x[6..])).coords))),
        ortho_setter(
            "set_left",
            ioffset("left"),
            |o| o.right(),
            |o, v| o.set_left(v),
        ),
        ortho_setter(
            "set_right",
            ioffset("right"),
            |o| o.left(),
            |o, v| o.set_right(v),
        ),
        ortho_setter(
            "set_bottom",
            ioffset("bottom"),
            |o| o.top(),
            |o, v| o.set_bottom(v),
        ),
        ortho_setter(
            "set_top",
            ioffset("top"),
            |o| o.bottom(),
            |o, v| o.set_top(v),
        ),
        ortho_setter(
            "set_znear",
            ioffset("znear"),
            |o| o.zfar(),
            |o, v| o.set_znear(v),
        ),
        ortho_setter(
            "set_zfar",
            ioffset("zfar"),
            |o| o.znear(),
            |o, v| o.set_zfar(v),
        ),
        ortho_setter2("set_left_and_right", |o, l, r| o.set_left_and_right(l, r)),
        ortho_setter2("set_bottom_and_top", |o, b, t| o.set_bottom_and_top(b, t)),
        ortho_setter2("set_znear_and_zfar", |o, n, f| o.set_znear_and_zfar(n, f)),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut ops = perspective_ops();
    ops.extend(orthographic_ops());
    vec![Suite {
        name: "projections",
        description: "Perspective3 and Orthographic3 (new, from_fov, accessors, inverse, \
                      project_point, project_vector, unproject_point, setters)",
        ops,
    }]
}
