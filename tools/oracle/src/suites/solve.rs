//! WP 8.5-P14a, suite `solve`: the triangular solves of `linalg/solve.rs` (lower, upper and the
//! transposed forms, vector and 3-column right-hand sides, `solve_lower_triangular_with_diag_mut`),
//! the Givens rotations of `linalg/givens.rs` (`new`, `cancel_x`, `cancel_y`, `rotate`,
//! `rotate_rows`), `reflection_axis_mut` (`linalg/householder.rs`) and the multi-column
//! `solve_mut` / `q_tr_mul` / `ln_determinant` of the LU, QR and Cholesky decompositions.
//!
//! The triangular operands are the triangles of an SPD matrix (`Gen::Spd`: positive diagonal,
//! well conditioned); upstream reads only the triangle a solve needs, so the whole matrix is the
//! input and the Cairo side passes it as it is.

use super::{dm, flat};
use crate::engine::{fm, fs, fv, im, iu, iv, with, Input, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::linalg::givens::GivensRotation;
use nalgebra::linalg::householder::reflection_axis_mut;
use nalgebra::{DMatrix, DVector, Matrix2x3, Matrix3x2, Vector2};

/// `2n` rounding stages per output scalar, amplified like input perturbations (the `linalg`
/// suites' model).
fn tol(n: usize) -> Tol {
    Tol::Sens {
        k: 2.0 * n as f64,
        base: 4.0,
    }
}

/// A few rounded quotients and one rounded norm (the Givens constructors, the Householder axis).
const QUOTIENTS: Tol = Tol::Sens { k: 4.0, base: 4.0 };

/// One fused sum of two products per entry, with a rotation unit only within a few ulp.
const ROTATE: Tol = Tol::SensMag {
    k: 2.0,
    base: 2.0,
    mag: 4.0,
};

/// `Qᵀ b` and the QR solve: the rounded orthonormal factor applied to `b` adds an error
/// proportional to `|b|` (the engine's model of rounded unit-scale factors).
fn qr_tol(n: usize) -> Tol {
    Tol::SensMag {
        k: 2.0 * n as f64,
        base: 4.0,
        mag: 4.0,
    }
}

/// Right-hand columns of the matrix cases.
const COLS: usize = 3;

fn ispd(n: usize) -> Input {
    with(fm("a", n, n), Gen::Spd(n))
}

fn iwell(n: usize) -> Input {
    with(fm("a", n, n), Gen::WellCond(n))
}

fn vec_at(x: &[f64], off: usize, n: usize) -> DVector<f64> {
    DVector::from_column_slice(&x[off..off + n])
}

fn mat_at(x: &[f64], off: usize, r: usize, c: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(r, c, &x[off..off + r * c])
}

fn triangular_ops(n: usize) -> Vec<Op> {
    type Solve = fn(&DMatrix<f64>, &DMatrix<f64>) -> Option<DMatrix<f64>>;
    let forms: [(&str, &str, Solve); 4] = [
        ("lower", "a.solve_lower_triangular(&b)", |a, b| {
            a.solve_lower_triangular(b)
        }),
        ("upper", "a.solve_upper_triangular(&b)", |a, b| {
            a.solve_upper_triangular(b)
        }),
        ("tr_lower", "a.tr_solve_lower_triangular(&b)", |a, b| {
            a.tr_solve_lower_triangular(b)
        }),
        ("tr_upper", "a.tr_solve_upper_triangular(&b)", |a, b| {
            a.tr_solve_upper_triangular(b)
        }),
    ];
    let mut ops = Vec::new();
    for (name, doc, solve) in forms {
        ops.push(
            Op::new(
                format!("solve{n}_{name}"),
                format!("{doc} (vector b; a SPD, only its triangle is read)"),
            )
            .input(ispd(n))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::NO_LARGE)
            .tol(tol(n))
            .eval(move |x| solve(&dm(x, n), &mat_at(x, n * n, n, 1)).map(|m| flat(&m))),
        );
    }
    if n == 3 || n == 6 {
        for (name, doc, solve) in [forms[0], forms[1]] {
            ops.push(
                Op::new(
                    format!("solve{n}_{name}_matrix"),
                    format!("{doc} (b: {n}x{COLS} matrix)"),
                )
                .input(ispd(n))
                .input(im("b", n, COLS))
                .out(fm("x", n, COLS))
                .dists(&Dist::NO_LARGE)
                .tol(tol(n))
                .eval(move |x| solve(&dm(x, n), &mat_at(x, n * n, n, COLS)).map(|m| flat(&m))),
            );
        }
    }
    ops.push(
        Op::new(
            format!("solve{n}_lower_with_diag"),
            "a.solve_lower_triangular_with_diag_mut(&mut b, diag); b (the strict lower triangle \
             of a with diag on the diagonal; the result is diag * x)",
        )
        .input(ispd(n))
        .input(iv("b", n))
        .input(with(fs("diag"), Gen::Range(0.5, 2.0)))
        .out(fv("b", n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            let mut b = vec_at(x, n * n, n);
            dm(x, n)
                .solve_lower_triangular_with_diag_mut(&mut b, x[n * n + n])
                .then(|| flat(&b))
        }),
    );
    ops
}

fn givens_ops() -> Vec<Op> {
    fn out(g: GivensRotation<f64>, r: f64) -> Vec<f64> {
        vec![g.c(), g.s(), r]
    }
    vec![
        Op::new("givens_new", "GivensRotation::new(c, s): (c, s, r)")
            .input(with(fv("cs", 2), Gen::V(2)))
            .out(fv("c_s_r", 3))
            .dists(&Dist::ALL)
            .tol(QUOTIENTS)
            .eval(|x| {
                let (g, r) = GivensRotation::new(x[0], x[1]);
                Some(out(g, r))
            }),
        Op::new(
            "givens_cancel_y",
            "GivensRotation::cancel_y(&v).unwrap(): (c, s, r)",
        )
        .input(iv("v", 2))
        .out(fv("c_s_r", 3))
        .dists(&Dist::ALL)
        .tol(QUOTIENTS)
        .eval(|x| GivensRotation::cancel_y(&Vector2::new(x[0], x[1])).map(|(g, r)| out(g, r))),
        Op::new(
            "givens_cancel_x",
            "GivensRotation::cancel_x(&v).unwrap(): (c, s, r)",
        )
        .input(iv("v", 2))
        .out(fv("c_s_r", 3))
        .dists(&Dist::ALL)
        .tol(QUOTIENTS)
        .eval(|x| GivensRotation::cancel_x(&Vector2::new(x[0], x[1])).map(|(g, r)| out(g, r))),
        Op::new(
            "givens_rotate",
            "GivensRotation::new_unchecked(c, s).rotate(&mut m) on a 2x3 m ((c, s) unit)",
        )
        .input(iu("cs", 2))
        .input(im("m", 2, 3))
        .out(fm("m", 2, 3))
        .dists(&Dist::NO_LARGE)
        .tol(ROTATE)
        .eval(|x| {
            let mut m = Matrix2x3::from_row_slice(&x[2..8]);
            GivensRotation::new_unchecked(x[0], x[1]).rotate(&mut m);
            Some(flat(&m))
        }),
        Op::new(
            "givens_rotate_rows",
            "GivensRotation::new_unchecked(c, s).rotate_rows(&mut m) on a 3x2 m ((c, s) unit)",
        )
        .input(iu("cs", 2))
        .input(im("m", 3, 2))
        .out(fm("m", 3, 2))
        .dists(&Dist::NO_LARGE)
        .tol(ROTATE)
        .eval(|x| {
            let mut m = Matrix3x2::from_row_slice(&x[2..8]);
            GivensRotation::new_unchecked(x[0], x[1]).rotate_rows(&mut m);
            Some(flat(&m))
        }),
    ]
}

fn householder_op(n: usize) -> Op {
    Op::new(
        format!("reflection_axis{n}"),
        "reflection_axis_mut(&mut v): (axis, r) (v nonzero)",
    )
    .input(iv("v", n))
    .out(fv("axis", n))
    .out(fs("r"))
    .dists(&Dist::NO_LARGE)
    .tol(QUOTIENTS)
    .eval(move |x| {
        let mut v = vec_at(x, 0, n);
        let (r, ok) = reflection_axis_mut(&mut v);
        ok.then(|| {
            let mut out = flat(&v);
            out.push(r);
            out
        })
    })
}

fn decomposition_ops(n: usize) -> Vec<Op> {
    let mut ops = vec![
        Op::new(
            format!("lu{n}_solve_matrix"),
            format!("a.lu().solve(&b).unwrap() (b: {n}x{COLS} matrix)"),
        )
        .input(iwell(n))
        .input(im("b", n, COLS))
        .out(fm("x", n, COLS))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            dm(x, n)
                .lu()
                .solve(&mat_at(x, n * n, n, COLS))
                .map(|m| flat(&m))
        }),
        Op::new(
            format!("cholesky{n}_solve_matrix"),
            format!("a.cholesky().unwrap().solve(&b) (b: {n}x{COLS} matrix)"),
        )
        .input(ispd(n))
        .input(im("b", n, COLS))
        .out(fm("x", n, COLS))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            dm(x, n)
                .cholesky()
                .map(|c| flat(&c.solve(&mat_at(x, n * n, n, COLS))))
        }),
        Op::new(
            format!("cholesky{n}_ln_determinant"),
            "a.cholesky().unwrap().ln_determinant()",
        )
        .input(ispd(n))
        .out(fs("ln_determinant"))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| dm(x, n).cholesky().map(|c| vec![c.ln_determinant()])),
    ];
    if n <= 4 {
        ops.push(
            Op::new(
                format!("qr{n}_q_tr_mul"),
                format!("a.qr().q_tr_mul(&mut b); b (b: {n}x{COLS} matrix)"),
            )
            .input(iwell(n))
            .input(im("b", n, COLS))
            .out(fm("qtb", n, COLS))
            .dists(&Dist::NO_LARGE)
            .tol(qr_tol(n))
            .eval(move |x| {
                let mut b = mat_at(x, n * n, n, COLS);
                dm(x, n).qr().q_tr_mul(&mut b);
                Some(flat(&b))
            }),
        );
        ops.push(
            Op::new(
                format!("qr{n}_solve_matrix"),
                format!("a.qr().solve(&b).unwrap() (b: {n}x{COLS} matrix)"),
            )
            .input(iwell(n))
            .input(im("b", n, COLS))
            .out(fm("x", n, COLS))
            .dists(&Dist::NO_LARGE)
            .tol(qr_tol(n))
            .eval(move |x| {
                dm(x, n)
                    .qr()
                    .solve(&mat_at(x, n * n, n, COLS))
                    .map(|m| flat(&m))
            }),
        );
    }
    ops
}

pub fn suites() -> Vec<Suite> {
    let mut ops: Vec<Op> = (2..=6).flat_map(triangular_ops).collect();
    ops.extend(givens_ops());
    ops.extend((2..=6).map(householder_op));
    ops.extend([2, 3, 4, 6].into_iter().flat_map(decomposition_ops));
    vec![Suite {
        name: "solve",
        description: "Triangular solves (linalg/solve.rs), Givens rotations, the Householder axis \
                      and the multi-column LU / QR / Cholesky solves (WP 8.5-P14a)",
        ops,
    }]
}
