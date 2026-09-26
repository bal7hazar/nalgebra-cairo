//! WP 8.6-P20 / P18, suite `sparse`: the legacy sparse module (`CsMatrix` products and sums, the
//! sparse triangular solves, `CsCholesky`) and the vector convolutions (`linalg/convolution.rs`).
//!
//! upstream's `sparse` feature needs `std` (the oracle builds nalgebra without it, see
//! Cargo.toml), so the sparse ops are evaluated on the DENSE matrices with the same entries,
//! which is what the sparse algorithms compute: the product and the sum exactly (i128 path), the
//! solves with upstream's dense `solve_lower_triangular` / `tr_solve_lower_triangular`, the
//! factor with upstream's dense `Cholesky` (the sparse factor has the same entries, the fill-in
//! included; the Cairo side compares its dense form).
//!
//! Operands are sparse by construction: a fixed PATTERN (1D / 2D Laplacian stencils, a
//! rectangular pattern) selects the entries of a dense random input, on both sides (the Cairo
//! test builds the `CsMatrix` from those entries with `from_triplet`). The SPD matrices of the
//! Cholesky ops are weighted graph Laplacians plus a positive diagonal: `A = sum_e w_e (e_i -
//! e_j)(e_i - e_j)ᵀ + diag(s)` with `w = |input|`, `s = |input| + 1/4`, assembled from triplets
//! (the Cairo side sums the duplicates, exactly).
//!
//! Convolutions: upstream's `convolve_*` need a `RealField`, so the exact path is a hand-written
//! kernel on `i128` raws, cross-checked against upstream in f64 by the engine.

use super::flat;
use crate::engine::{fm, im, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{DMatrix, DVector};

use super::Ring;

/// The pattern of the `n x n` 1D Laplacian (tridiagonal), row-major order.
fn lap1d(n: usize) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            if i.abs_diff(j) <= 1 {
                out.push((i, j));
            }
        }
    }
    out
}

/// The pattern of the 2D Laplacian of a `g x g` grid (5-point stencil), row-major order.
fn lap2d(g: usize) -> Vec<(usize, usize)> {
    let n = g * g;
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..n {
            let (ri, ci, rj, cj) = (i / g, i % g, j / g, j % g);
            if ri.abs_diff(rj) + ci.abs_diff(cj) <= 1 {
                out.push((i, j));
            }
        }
    }
    out
}

/// The edges `(i, j)`, `i < j`, of a symmetric pattern.
fn edges(pattern: &[(usize, usize)]) -> Vec<(usize, usize)> {
    pattern.iter().copied().filter(|(i, j)| i < j).collect()
}

/// A 5x7 and a 7x4 pattern (the rectangular product).
const RECT_A: [(usize, usize); 12] = [
    (0, 0),
    (0, 3),
    (1, 1),
    (1, 6),
    (2, 2),
    (2, 4),
    (3, 0),
    (3, 5),
    (4, 1),
    (4, 3),
    (4, 6),
    (2, 6),
];
const RECT_B: [(usize, usize); 10] = [
    (0, 0),
    (1, 2),
    (2, 1),
    (3, 3),
    (3, 0),
    (4, 2),
    (5, 1),
    (6, 3),
    (6, 0),
    (2, 3),
];

/// The dense `r x c` matrix of the entries of the row-major `x` on `pattern`, zeros elsewhere.
fn masked<T: Ring>(x: &[T], r: usize, c: usize, pattern: &[(usize, usize)]) -> DMatrix<T> {
    let mut m = DMatrix::from_element(r, c, T::zero());
    for &(i, j) in pattern {
        m[(i, j)] = x[i * c + j];
    }
    m
}

fn lower(pattern: &[(usize, usize)]) -> Vec<(usize, usize)> {
    pattern.iter().copied().filter(|(i, j)| i >= j).collect()
}

fn mul<T: Ring>(
    x: &[T],
    m: usize,
    k: usize,
    n: usize,
    pa: &[(usize, usize)],
    pb: &[(usize, usize)],
) -> Vec<T> {
    flat(&(masked(x, m, k, pa) * masked(&x[m * k..], k, n, pb)))
}

fn add<T: Ring>(x: &[T], n: usize, pa: &[(usize, usize)], pb: &[(usize, usize)]) -> Vec<T> {
    flat(&(masked(x, n, n, pa) + masked(&x[n * n..], n, n, pb)))
}

fn ring_pair(
    f: fn(&[f64]) -> Vec<f64>,
    g: fn(&[i128]) -> Vec<i128>,
) -> (crate::engine::EvalFn, crate::engine::ExactFn) {
    (Box::new(move |x| Some(f(x))), Box::new(move |x| Some(g(x))))
}

/// `k` rounding stages per output, amplified like input perturbations.
fn sens(k: f64) -> Tol {
    Tol::Sens { k, base: 4.0 }
}

/// The SPD graph Laplacian of `pattern` (`n x n`) from the weights `x[..e]` and shifts
/// `x[e..e + n]` (see the module doc).
fn laplacian(x: &[f64], n: usize, pattern: &[(usize, usize)]) -> DMatrix<f64> {
    let es = edges(pattern);
    let mut a = DMatrix::from_element(n, n, 0.0);
    for (k, &(i, j)) in es.iter().enumerate() {
        let w = x[k].abs();
        a[(i, j)] -= w;
        a[(j, i)] -= w;
        a[(i, i)] += w;
        a[(j, j)] += w;
    }
    for i in 0..n {
        a[(i, i)] += x[es.len() + i].abs() + 0.25;
    }
    a
}

fn cholesky_op(name: &str, n: usize, pattern: Vec<(usize, usize)>) -> Op {
    let e = edges(&pattern).len();
    Op::new(
        name,
        format!(
            "CsCholesky::new(&a).l() on the SPD Laplacian of a {n}-node pattern ({e} edges); \
             upstream's dense Cholesky of the same matrix, L row-major"
        ),
    )
    .input(im("w", e, 1))
    .input(im("s", n, 1))
    .out(fm("l", n, n))
    .dists(&Dist::NO_LARGE)
    .tol(sens(4.0 * n as f64))
    .eval(move |x| {
        let a = laplacian(x, n, &pattern);
        a.cholesky().map(|c| flat(&c.l()))
    })
}

type Solve = fn(&DMatrix<f64>, &DMatrix<f64>) -> Option<DMatrix<f64>>;

fn solve_op(name: &str, doc: &str, g: usize, solve: Solve) -> Op {
    let n = g * g;
    let pattern = lower(&lap2d(g));
    Op::new(
        name,
        format!("{doc} (L: the lower 2D-Laplacian pattern of a {g}x{g} grid, entries of an SPD a)"),
    )
    .input(crate::engine::with(fm("a", n, n), Gen::Spd(n)))
    .input(im("b", n, 1))
    .out(fm("x", n, 1))
    .dists(&Dist::NO_LARGE)
    .tol(sens(2.0 * n as f64))
    .eval(move |x| {
        let l = masked(x, n, n, &pattern);
        let b = DMatrix::from_row_slice(n, 1, &x[n * n..n * n + n]);
        solve(&l, &b).map(|m| flat(&m))
    })
}

/// The rows reached from `b[0]` and `b[4]` only (the sparse right-hand side of `_cs`).
const CS_RHS: [usize; 2] = [0, 4];

// --- convolutions -----------------------------------------------------------------------------

fn conv_full<T: Ring>(x: &[T], k: &[T]) -> Vec<T> {
    let (n, m) = (x.len(), k.len());
    (0..n + m - 1)
        .map(|i| {
            let mut s = T::zero();
            for u in 0..n {
                if i >= u && i - u < m {
                    s += x[u] * k[i - u];
                }
            }
            s
        })
        .collect()
}

fn conv_same<T: Ring>(x: &[T], k: &[T]) -> Vec<T> {
    let (n, m) = (x.len(), k.len());
    (0..n)
        .map(|i| {
            let mut s = T::zero();
            for j in 0..m {
                if i + j >= 1 && i + j < n + 1 {
                    s += x[i + j - 1] * k[m - j - 1];
                }
            }
            s
        })
        .collect()
}

fn conv_valid<T: Ring>(x: &[T], k: &[T]) -> Vec<T> {
    let (n, m) = (x.len(), k.len());
    (0..n - m + 1)
        .map(|i| {
            let mut s = T::zero();
            for j in 0..m {
                s += x[i + j] * k[m - j - 1];
            }
            s
        })
        .collect()
}

fn conv_op(form: &'static str, n: usize, m: usize) -> Op {
    let out = match form {
        "full" => n + m - 1,
        "same" => n,
        _ => n - m + 1,
    };
    let upstream = move |x: &[f64]| -> Vec<f64> {
        let v = DVector::from_column_slice(&x[..n]);
        let k = DVector::from_column_slice(&x[n..n + m]);
        let r = match form {
            "full" => v.convolve_full(k),
            "same" => v.convolve_same(k),
            _ => v.convolve_valid(k),
        };
        r.as_slice().to_vec()
    };
    let exact = move |x: &[i128]| -> Vec<i128> {
        match form {
            "full" => conv_full(&x[..n], &x[n..n + m]),
            "same" => conv_same(&x[..n], &x[n..n + m]),
            _ => conv_valid(&x[..n], &x[n..n + m]),
        }
    };
    Op::new(
        format!("convolve_{form}{n}_{m}"),
        format!("v.convolve_{form}(k) (v: {n} components, k: {m})"),
    )
    .input(im("v", n, 1))
    .input(im("k", m, 1))
    .out(fm("c", out, 1))
    .dists(&Dist::NO_LARGE)
    .ring((
        Box::new(move |x| Some(upstream(x))),
        Box::new(move |x| Some(exact(x))),
    ))
}

fn ops() -> Vec<Op> {
    let mut ops = vec![
        Op::new(
            "cs_mul_lap1d8",
            "a * b (CsMatrix): two 8x8 tridiagonal patterns, a pentadiagonal product",
        )
        .input(im("a", 8, 8))
        .input(im("b", 8, 8))
        .out(fm("ab", 8, 8))
        .dists(&Dist::NO_LARGE)
        .ring(ring_pair(
            |x| mul(x, 8, 8, 8, &lap1d(8), &lap1d(8)),
            |x| mul(x, 8, 8, 8, &lap1d(8), &lap1d(8)),
        )),
        Op::new(
            "cs_mul_rect",
            "a * b (CsMatrix): a 5x7 pattern of 12 entries times a 7x4 pattern of 10 entries",
        )
        .input(im("a", 5, 7))
        .input(im("b", 7, 4))
        .out(fm("ab", 5, 4))
        .dists(&Dist::NO_LARGE)
        .ring(ring_pair(
            |x| mul(x, 5, 7, 4, &RECT_A, &RECT_B),
            |x| mul(x, 5, 7, 4, &RECT_A, &RECT_B),
        )),
        Op::new(
            "cs_add_lap2d9",
            "a + b (CsMatrix): a 3x3-grid 2D-Laplacian pattern plus a 9x9 tridiagonal one",
        )
        .input(im("a", 9, 9))
        .input(im("b", 9, 9))
        .out(fm("s", 9, 9))
        .dists(&Dist::NO_LARGE)
        .ring(ring_pair(
            |x| add(x, 9, &lap2d(3), &lap1d(9)),
            |x| add(x, 9, &lap2d(3), &lap1d(9)),
        ))
        .degree(1),
        solve_op(
            "cs_solve_lower_lap2d9",
            "l.solve_lower_triangular(&b) (CsMatrix)",
            3,
            |l, b| l.solve_lower_triangular(b),
        ),
        solve_op(
            "cs_tr_solve_lower_lap2d9",
            "l.tr_solve_lower_triangular(&b) (CsMatrix)",
            3,
            |l, b| l.tr_solve_lower_triangular(b),
        ),
        {
            let pattern = lower(&lap2d(3));
            Op::new(
                "cs_solve_lower_cs_lap2d9",
                "l.solve_lower_triangular_cs(&b) (CsMatrix, b sparse: its entries 0 and 4 only), \
                 as a dense vector",
            )
            .input(crate::engine::with(fm("a", 9, 9), Gen::Spd(9)))
            .input(im("b", 9, 1))
            .out(fm("x", 9, 1))
            .dists(&Dist::NO_LARGE)
            .tol(sens(18.0))
            .eval(move |x| {
                let l = masked(x, 9, 9, &pattern);
                let mut b = DMatrix::from_element(9, 1, 0.0);
                for i in CS_RHS {
                    b[(i, 0)] = x[81 + i];
                }
                l.solve_lower_triangular(&b).map(|m| flat(&m))
            })
        },
        cholesky_op("cs_cholesky_lap1d16", 16, lap1d(16)),
        cholesky_op("cs_cholesky_lap2d16", 16, lap2d(4)),
        cholesky_op("cs_cholesky_lap2d9", 9, lap2d(3)),
    ];
    ops.extend([
        conv_op("full", 7, 3),
        conv_op("same", 7, 4),
        conv_op("valid", 16, 5),
        conv_op("full", 6, 3),
        conv_op("same", 6, 6),
    ]);
    ops
}

pub fn suites() -> Vec<Suite> {
    vec![Suite {
        name: "sparse",
        description: "Sparse matrices (CsMatrix products, sums, triangular solves, CsCholesky on \
                      SPD graph Laplacians) against upstream's dense forms, and vector \
                      convolutions",
        ops: ops(),
    }]
}
