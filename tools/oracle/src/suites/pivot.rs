//! WP 8.5-P15, suite `pivot`: the pivoted factorisations of upstream `nalgebra::linalg` —
//! `FullPivLU` and `ColPivQR` on every static shape, `LBLT` on the squares 1..6.
//!
//! Upstream runs the same generic code on static and dynamic matrices; `DMatrix` is used so that
//! one definition covers every shape. The permutations are emitted as the permuted index vector
//! (`p.permute_rows(&mut [0, 1, .., n - 1])`, exact integers): the Cairo tests apply their
//! permutation to the same vector, so a different pivot choice is a mismatch of `2^32` raw.
//!
//! Three kinds of inputs: well-conditioned (`WellCondRect`: factors, solve, inverse,
//! determinant, compared entry by entry), EXACTLY rank-deficient small-integer matrices
//! (`RankDeficient`: only the rank is emitted — the factors past the rank are rounding noise, and
//! so are the pivots chosen among it; the Cairo tests check the rank-revealing property and the
//! reconstruction), and near-singular square systems (`NearSingular`, FLAGGED: loose tolerance).
//! `LBLT` adds zero-diagonal symmetric matrices, which force 2x2 pivot blocks.

use super::flat;
use crate::engine::{fm, fs, fv, iv, with, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{DMatrix, DVector};

const NEAR_SINGULAR_CAP: u64 = 1 << 44;

/// Rank threshold of the rank-deficient ops (exact integer inputs: the true singular values are
/// either 0 or at least ~0.1).
const RANK_EPS: f64 = 1.0e-6;

fn shapes() -> Vec<(usize, usize)> {
    (1..=6).flat_map(|r| (1..=6).map(move |c| (r, c))).collect()
}

fn name(r: usize, c: usize) -> String {
    if r == c {
        format!("{r}")
    } else {
        format!("{r}x{c}")
    }
}

fn mat_at(x: &[f64], off: usize, r: usize, c: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(r, c, &x[off..off + r * c])
}

fn vec_at(x: &[f64], off: usize, n: usize) -> DVector<f64> {
    DVector::from_column_slice(&x[off..off + n])
}

fn indices(n: usize) -> DVector<f64> {
    DVector::from_fn(n, |i, _| i as f64)
}

/// `2 (r + c)` rounding stages per output scalar (elimination and back substitution), amplified
/// like input perturbations.
fn lu_tol(r: usize, c: usize) -> Tol {
    Tol::Sens {
        k: 2.0 * (r + c) as f64,
        base: 4.0,
    }
}

/// Householder reflections: rounded unit-scale factors applied to the input (`SensMag`).
fn qr_tol(r: usize, c: usize) -> Tol {
    let k = r.min(c) as f64;
    Tol::SensMag {
        k: 4.0 * k,
        base: 8.0,
        mag: 4.0 * k,
    }
}

fn sq_tol(n: usize) -> Tol {
    Tol::Sens {
        k: 4.0 * n as f64,
        base: 8.0,
    }
}

fn full_piv_lu_ops(r: usize, c: usize) -> Vec<Op> {
    let s = name(r, c);
    let mut ops = vec![Op::new(
        format!("full_piv_lu{s}"),
        format!(
            "a.full_piv_lu(): (lu {r}x{c} packed, p = P [0..{r}), q = Q [0..{c}) as permuted \
                 index vectors), well-conditioned a"
        ),
    )
    .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
    .out(fm("lu", r, c))
    .out(fv("p", r))
    .out(fv("q", c))
    .dists(&Dist::NO_LARGE)
    .tol(lu_tol(r, c))
    .eval(move |x| {
        let f = mat_at(x, 0, r, c).full_piv_lu();
        let mut p = indices(r);
        f.p().permute_rows(&mut p);
        let mut q = indices(c);
        f.q().permute_rows(&mut q);
        let mut out = flat(f.lu_internal());
        out.extend(flat(&p));
        out.extend(flat(&q));
        Some(out)
    })];
    if r.min(c) >= 2 {
        ops.push(
            Op::new(
                format!("full_piv_lu{s}_rank"),
                format!(
                    "a.rank({RANK_EPS:e}), a EXACTLY rank-deficient (small integer entries): the \
                     number of pivots of a.full_piv_lu() that are not rounding noise"
                ),
            )
            .input(with(fm("a", r, c), Gen::RankDeficient(r, c)))
            .out(fs("rank"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(0))
            .eval(move |x| Some(vec![mat_at(x, 0, r, c).rank(RANK_EPS) as f64])),
        );
    }
    if r == c {
        let n = r;
        ops.push(
            Op::new(
                format!("full_piv_lu{n}_solve"),
                "a.full_piv_lu().solve(&b).unwrap()",
            )
            .input(with(fm("a", n, n), Gen::WellCond(n)))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| {
                mat_at(x, 0, n, n)
                    .full_piv_lu()
                    .solve(&vec_at(x, n * n, n))
                    .map(|x| flat(&x))
            }),
        );
        ops.push(
            Op::new(
                format!("full_piv_lu{n}_inverse"),
                "a.full_piv_lu().try_inverse().unwrap()",
            )
            .input(with(fm("a", n, n), Gen::WellCond(n)))
            .out(fm("inverse", n, n))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| {
                mat_at(x, 0, n, n)
                    .full_piv_lu()
                    .try_inverse()
                    .map(|m| flat(&m))
            }),
        );
        ops.push(
            Op::new(
                format!("full_piv_lu{n}_determinant"),
                "a.full_piv_lu().determinant()",
            )
            .input(with(fm("a", n, n), Gen::WellCond(n)))
            .out(fs("determinant"))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| Some(vec![mat_at(x, 0, n, n).full_piv_lu().determinant()])),
        );
        if n >= 2 {
            ops.push(
                Op::new(
                    format!("full_piv_lu{n}_solve_near_singular"),
                    "a.full_piv_lu().solve(&b).unwrap(), one singular value in [1e-4, 1e-2] \
                     (FLAGGED: loose tolerance)",
                )
                .input(with(fm("a", n, n), Gen::NearSingular(n)))
                .input(iv("b", n))
                .out(fv("x", n))
                .dists(&Dist::UNIT)
                .cap(NEAR_SINGULAR_CAP)
                .tol(sq_tol(n))
                .eval(move |x| {
                    mat_at(x, 0, n, n)
                        .full_piv_lu()
                        .solve(&vec_at(x, n * n, n))
                        .map(|x| flat(&x))
                }),
            );
        }
    }
    ops
}

fn col_piv_qr_ops(r: usize, c: usize) -> Vec<Op> {
    let s = name(r, c);
    let k = r.min(c);
    let mut ops = vec![Op::new(
        format!("col_piv_qr{s}"),
        format!(
            "a.col_piv_qr(): (q {r}x{k}, r {k}x{c} with diag(r) >= 0, p = P [0..{c}) as a \
             permuted index vector), well-conditioned a"
        ),
    )
    .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
    .out(fm("q", r, k))
    .out(fm("r", k, c))
    .out(fv("p", c))
    .dists(&Dist::NO_LARGE)
    .tol(qr_tol(r, c))
    .eval(move |x| {
        let f = mat_at(x, 0, r, c).col_piv_qr();
        let mut p = indices(c);
        f.p().permute_rows(&mut p);
        let mut out = flat(&f.q());
        out.extend(flat(&f.r()));
        out.extend(flat(&p));
        Some(out)
    })];
    if k >= 2 {
        ops.push(
            Op::new(
                format!("col_piv_qr{s}_rank"),
                format!(
                    "a.rank({RANK_EPS:e}), a EXACTLY rank-deficient (small integer entries): the \
                     number of diagonal entries of a.col_piv_qr().r() that are not rounding noise"
                ),
            )
            .input(with(fm("a", r, c), Gen::RankDeficient(r, c)))
            .out(fs("rank"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(0))
            .eval(move |x| Some(vec![mat_at(x, 0, r, c).rank(RANK_EPS) as f64])),
        );
    }
    if r == c {
        let n = r;
        ops.push(
            Op::new(
                format!("col_piv_qr{n}_solve"),
                "a.col_piv_qr().solve(&b).unwrap()",
            )
            .input(with(fm("a", n, n), Gen::WellCond(n)))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| {
                mat_at(x, 0, n, n)
                    .col_piv_qr()
                    .solve(&vec_at(x, n * n, n))
                    .map(|x| flat(&x))
            }),
        );
        ops.push(
            Op::new(
                format!("col_piv_qr{n}_determinant"),
                "a.col_piv_qr().determinant()",
            )
            .input(with(fm("a", n, n), Gen::WellCond(n)))
            .out(fs("determinant"))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| Some(vec![mat_at(x, 0, n, n).col_piv_qr().determinant()])),
        );
        if n >= 2 {
            ops.push(
                Op::new(
                    format!("col_piv_qr{n}_solve_near_singular"),
                    "a.col_piv_qr().solve(&b).unwrap(), one singular value in [1e-4, 1e-2] \
                     (FLAGGED: loose tolerance)",
                )
                .input(with(fm("a", n, n), Gen::NearSingular(n)))
                .input(iv("b", n))
                .out(fv("x", n))
                .dists(&Dist::UNIT)
                .cap(NEAR_SINGULAR_CAP)
                .tol(sq_tol(n))
                .eval(move |x| {
                    mat_at(x, 0, n, n)
                        .col_piv_qr()
                        .solve(&vec_at(x, n * n, n))
                        .map(|x| flat(&x))
                }),
            );
        }
    }
    ops
}

fn lblt_ops(n: usize) -> Vec<Op> {
    let kinds: [(&str, &str, Gen); 2] = [
        ("", "symmetric (indefinite) a", Gen::Sym(n)),
        (
            "_zero_diag",
            "symmetric a with an exactly zero diagonal (2x2 pivot blocks)",
            Gen::SymZeroDiag(n),
        ),
    ];
    let mut ops = Vec::new();
    for (suffix, doc, gen) in kinds {
        if n == 1 && suffix == "_zero_diag" {
            continue;
        }
        ops.push(
            Op::new(
                format!("lblt{n}{suffix}"),
                format!("(a.lblt().d(), a.lblt().l_permuted()), {doc}"),
            )
            .input(with(fm("a", n, n), gen.clone()))
            .out(fm("d", n, n))
            .out(fm("l_permuted", n, n))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| {
                let f = mat_at(x, 0, n, n).lblt();
                let mut out = flat(&f.d());
                out.extend(flat(&f.l_permuted()));
                Some(out)
            }),
        );
        ops.push(
            Op::new(
                format!("lblt{n}{suffix}_solve"),
                format!("a.lblt().solve(&b).unwrap(), {doc}"),
            )
            .input(with(fm("a", n, n), gen.clone()))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::NO_LARGE)
            .tol(sq_tol(n))
            .eval(move |x| {
                let b = DMatrix::from_column_slice(n, 1, &x[n * n..n * n + n]);
                mat_at(x, 0, n, n).lblt().solve(&b).map(|x| flat(&x))
            }),
        );
        ops.push(
            Op::new(
                format!("lblt{n}{suffix}_determinant"),
                format!("a.lblt().determinant(), {doc}"),
            )
            .input(with(fm("a", n, n), gen))
            .out(fs("determinant"))
            // independent entries: a `medium` determinant does not fit Q32.32
            .dists(&[Dist::Small, Dist::Unit])
            .tol(sq_tol(n))
            .eval(move |x| Some(vec![mat_at(x, 0, n, n).lblt().determinant()])),
        );
    }
    ops
}

pub fn suites() -> Vec<Suite> {
    let mut ops: Vec<Op> = shapes()
        .into_iter()
        .flat_map(|(r, c)| full_piv_lu_ops(r, c))
        .collect();
    ops.extend(shapes().into_iter().flat_map(|(r, c)| col_piv_qr_ops(r, c)));
    ops.extend((1..=6).flat_map(lblt_ops));
    vec![Suite {
        name: "pivot",
        description: "FullPivLU and ColPivQR of every shape, LBLT 1..6: factors, permutations, \
                      solve, inverse, determinant, rank of rank-deficient inputs (WP 8.5-P15)",
        ops,
    }]
}
