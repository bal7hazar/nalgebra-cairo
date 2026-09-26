//! WP 8.5-P14b, suite `spectral`: the symmetric eigen decomposition of the sizes 4..6, the SVD of
//! every static shape (singular values of well-conditioned, exactly rank-deficient and flagged
//! near-singular matrices; rank, pseudo-inverse and polar decomposition), the QR factors of the
//! shapes P14a did not cover, the Cholesky rank-one update / downdate and column insertion /
//! removal, and `wilkinson_shift`.
//!
//! Upstream runs the same generic code on static and dynamic matrices; `DMatrix` is used so that
//! one definition covers every shape. Eigenvalues are sorted ASCENDING and singular values
//! DESCENDING (the order of the Cairo types); eigenvectors, `U` and `V` are not emitted (sign /
//! order ambiguous): the Cairo tests check them by reconstruction. The polar factors, the
//! pseudo-inverse, the QR factors (upstream's unpacked, non-negative `diag(r)`) and the updated
//! Cholesky factors are unique and compared entry by entry.

use super::flat;
use crate::engine::{fm, fs, fv, is, iv, with, Input, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{DMatrix, DVector};

/// Iterative / squared-matrix algorithms (Jacobi sweeps, SVD through `M^T M`): the `linalg`
/// suite's `SPECTRAL` budget.
const SPECTRAL: Tol = Tol::SensMag {
    k: 16.0,
    base: 8.0,
    mag: 16.0,
};

const NEAR_SINGULAR_CAP: u64 = 1 << 44;

/// `pseudo_inverse` / `rank` threshold of the ops below (well above the rounding noise of a
/// Q32.32 decomposition, far below the smallest singular value of the non-deficient inputs).
pub const EPS: f64 = 1.0e-4;

/// Every static shape (rows, cols), 1..6 x 1..6.
fn shapes() -> Vec<(usize, usize)> {
    (1..=6)
        .flat_map(|r| (1..=6).map(move |c| (r, c)))
        .collect()
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

fn sorted(mut values: Vec<f64>, descending: bool) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    if descending {
        values.reverse();
    }
    values
}

fn eigen_ops(n: usize) -> Vec<Op> {
    let kinds: [(&str, &str, Gen); 4] = [
        ("", "symmetric matrices, independent entries mirrored", Gen::Sym(n)),
        ("_spd", "symmetric positive-definite matrices", Gen::Spd(n)),
        (
            "_clustered",
            "SPD with clustered eigenvalues (relative gaps 0, 1e-6, 1e-3)",
            Gen::Clustered(n),
        ),
        (
            "_deficient",
            "B^T B of an integer k x n factor, k < n: exact zero eigenvalues",
            Gen::SymDeficient(n),
        ),
    ];
    kinds
        .into_iter()
        .map(|(suffix, doc, gen)| {
            let dists: &[Dist] = if matches!(gen, Gen::SymDeficient(_)) {
                &Dist::UNIT
            } else {
                &Dist::NO_LARGE
            };
            Op::new(
                format!("symmetric_eigen{n}_eigenvalues{suffix}"),
                format!(
                    "a.symmetric_eigen().eigenvalues sorted ASCENDING, {doc} (check the \
                     eigenvectors by reconstruction)"
                ),
            )
            .input(with(fm("a", n, n), gen))
            .out(fv("eigenvalues", n))
            .dists(dists)
            .tol(SPECTRAL)
            .eval(move |x| {
                Some(sorted(
                    flat(&mat_at(x, 0, n, n).symmetric_eigen().eigenvalues),
                    false,
                ))
            })
        })
        .collect()
}

fn svd_ops(r: usize, c: usize) -> Vec<Op> {
    let k = r.min(c);
    let s = name(r, c);
    let singular_values = move |x: &[f64]| {
        Some(sorted(
            flat(&mat_at(x, 0, r, c).singular_values()),
            true,
        ))
    };
    let mut ops = vec![Op::new(
        format!("svd{s}_singular_values"),
        "a.singular_values() sorted DESCENDING (well-conditioned a; check U, V by \
         reconstruction)",
    )
    .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
    .out(fv("singular_values", k))
    .dists(&Dist::NO_LARGE)
    .tol(SPECTRAL)
    .eval(singular_values)];
    if k >= 2 {
        ops.push(
            Op::new(
                format!("svd{s}_singular_values_deficient"),
                "a.singular_values() sorted DESCENDING, a EXACTLY rank-deficient (small \
                 integer entries)",
            )
            .input(with(fm("a", r, c), Gen::RankDeficient(r, c)))
            .out(fv("singular_values", k))
            .dists(&Dist::UNIT)
            .tol(SPECTRAL)
            .eval(singular_values),
        );
        ops.push(
            Op::new(
                format!("svd{s}_rank_deficient"),
                format!("a.rank({EPS:e}), a EXACTLY rank-deficient (small integer entries)"),
            )
            .input(with(fm("a", r, c), Gen::RankDeficient(r, c)))
            .out(fs("rank"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(0))
            .eval(move |x| Some(vec![mat_at(x, 0, r, c).rank(EPS) as f64])),
        );
        ops.push(
            Op::new(
                format!("svd{s}_singular_values_near_singular"),
                "a.singular_values() sorted DESCENDING, one singular value in [1e-4, 1e-2] \
                 (FLAGGED: loose tolerance)",
            )
            .input(with(fm("a", r, c), Gen::NearSingularRect(r, c)))
            .out(fv("singular_values", k))
            .dists(&Dist::UNIT)
            .cap(NEAR_SINGULAR_CAP)
            .tol(SPECTRAL)
            .eval(singular_values),
        );
    }
    ops.push(
        Op::new(
            format!("svd{s}_pseudo_inverse"),
            format!("a.pseudo_inverse({EPS:e}).unwrap() ({c}x{r}, well-conditioned a)"),
        )
        .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
        .out(fm("pseudo_inverse", c, r))
        .dists(&Dist::NO_LARGE)
        .tol(SPECTRAL)
        .eval(move |x| {
            mat_at(x, 0, r, c)
                .pseudo_inverse(EPS)
                .ok()
                .map(|m| flat(&m))
        }),
    );
    ops.push(
        Op::new(
            format!("svd{s}_polar"),
            format!(
                "a.polar(): (p {r}x{r} symmetric positive semi-definite, u {r}x{c} with \
                 orthonormal rows or columns), a = p * u (well-conditioned a; unique)"
            ),
        )
        .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
        .out(fm("p", r, r))
        .out(fm("u", r, c))
        .dists(&Dist::NO_LARGE)
        .tol(SPECTRAL)
        .eval(move |x| {
            let (p, u) = mat_at(x, 0, r, c).polar();
            let mut out = flat(&p);
            out.extend(flat(&u));
            Some(out)
        }),
    );
    ops
}

/// The QR shapes P14a did not cover (it has 2, 3 and 4 in the `qr` suite).
fn qr_ops(r: usize, c: usize) -> Vec<Op> {
    let k = r.min(c);
    let s = name(r, c);
    let mut ops = vec![Op::new(
        format!("qr{s}_q_r"),
        format!(
            "a.qr(): (q {r}x{k}, r {k}x{c}), upstream's unpacked factors (diag(r) >= 0; \
             well-conditioned a)"
        ),
    )
    .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
    .out(fm("q", r, k))
    .out(fm("r", k, c))
    .dists(&Dist::NO_LARGE)
    .tol(Tol::SensMag {
        k: 2.0 * k as f64,
        base: 4.0,
        mag: 2.0 * k as f64,
    })
    .eval(move |x| {
        let qr = mat_at(x, 0, r, c).qr();
        let mut out = flat(&qr.q());
        out.extend(flat(&qr.r()));
        Some(out)
    })];
    if r == c {
        ops.push(
            Op::new(format!("qr{s}_solve"), "a.qr().solve(&b).unwrap()")
                .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
                .input(iv("b", r))
                .out(fv("x", r))
                .dists(&Dist::NO_LARGE)
                .tol(Tol::Sens {
                    k: 2.0 * r as f64,
                    base: 4.0,
                })
                .eval(move |x| {
                    let b = DVector::from_column_slice(&x[r * r..r * r + r]);
                    mat_at(x, 0, r, r).qr().solve(&b).map(|x| flat(&x))
                }),
        );
    }
    ops
}

fn ispd(n: usize) -> Input {
    with(fm("a", n, n), Gen::Spd(n))
}

/// `2n` rounding stages per output scalar (the `linalg` suite's Cholesky model), doubled: an
/// update re-derives every entry of the factor.
fn chol_tol(n: usize) -> Tol {
    Tol::Sens {
        k: 4.0 * n as f64,
        base: 8.0,
    }
}

fn cholesky_ops(n: usize) -> Vec<Op> {
    let mut ops = vec![
        Op::new(
            format!("cholesky{n}_rank_one_update"),
            "l of a.cholesky().rank_one_update(&x, sigma), sigma in [0.25, 2] (the factor of a \
             + sigma x x^T)",
        )
        .input(ispd(n))
        .input(iv("x", n))
        .input(with(fs("sigma"), Gen::Range(0.25, 2.0)))
        .out(fm("l", n, n))
        .dists(&Dist::NO_LARGE)
        .tol(chol_tol(n))
        .eval(move |x| {
            let mut chol = mat_at(x, 0, n, n).cholesky()?;
            chol.rank_one_update(&DVector::from_column_slice(&x[n * n..n * n + n]), x[n * n + n]);
            let l = chol.l();
            l.iter().all(|v| v.is_finite()).then(|| flat(&l))
        }),
        Op::new(
            format!("cholesky{n}_rank_one_downdate"),
            "l of a.cholesky().rank_one_update(&x, sigma), sigma in [-0.5, -0.05] (the factor \
             of a - |sigma| x x^T; cases that leave a indefinite are resampled)",
        )
        .input(ispd(n))
        .input(iv("x", n))
        .input(with(fs("sigma"), Gen::Range(-0.5, -0.05)))
        .out(fm("l", n, n))
        .dists(&Dist::NO_LARGE)
        .tol(chol_tol(n))
        .eval(move |x| {
            let a = mat_at(x, 0, n, n);
            let v = DVector::from_column_slice(&x[n * n..n * n + n]);
            let sigma = x[n * n + n];
            // Keep a well-conditioned result: the updated matrix must stay safely definite.
            let updated = &a + &v * v.transpose() * sigma;
            let e = updated.clone().symmetric_eigen().eigenvalues.min();
            if e <= 0.05 * a.clone().symmetric_eigen().eigenvalues.min() {
                return None;
            }
            let mut chol = a.cholesky()?;
            chol.rank_one_update(&v, sigma);
            let l = chol.l();
            l.iter().all(|v| v.is_finite()).then(|| flat(&l))
        }),
    ];
    if n == 2 || n == 3 {
        for j in 0..=n {
            let m = n + 1;
            ops.push(
                Op::new(
                    format!("cholesky{n}_insert_column_j{j}"),
                    format!(
                        "l of chol(a without row / column {j}).insert_column({j}, a.column({j})): \
                         the factor of the {m}x{m} SPD a"
                    ),
                )
                .input(ispd(m))
                .out(fm("l", m, m))
                .dists(&Dist::NO_LARGE)
                .tol(chol_tol(m))
                .eval(move |x| {
                    let a = mat_at(x, 0, m, m);
                    let sub = a.clone().remove_row(j).remove_column(j);
                    let chol = sub.cholesky()?;
                    let l = chol.insert_column(j, a.column(j).clone_owned()).l();
                    l.iter().all(|v| v.is_finite()).then(|| flat(&l))
                }),
            );
        }
    }
    if n == 3 || n == 4 {
        for j in 0..n {
            let m = n - 1;
            ops.push(
                Op::new(
                    format!("cholesky{n}_remove_column_j{j}"),
                    format!("l of a.cholesky().remove_column({j}): a {m}x{m} factor"),
                )
                .input(ispd(n))
                .out(fm("l", m, m))
                .dists(&Dist::NO_LARGE)
                .tol(chol_tol(n))
                .eval(move |x| {
                    let l = mat_at(x, 0, n, n).cholesky()?.remove_column(j).l();
                    l.iter().all(|v| v.is_finite()).then(|| flat(&l))
                }),
            );
        }
    }
    ops
}

fn wilkinson_ops() -> Vec<Op> {
    vec![Op::new(
        "wilkinson_shift",
        "nalgebra::linalg::wilkinson_shift(tmm, tnn, tmn): the eigenvalue of [[tmm, tmn], [tmn, \
         tnn]] closest to tnn",
    )
    .input(is("tmm"))
    .input(is("tnn"))
    .input(is("tmn"))
    .out(fs("shift"))
    .dists(&Dist::NO_LARGE)
    .tol(Tol::Sens { k: 4.0, base: 4.0 })
    .eval(|x| Some(vec![nalgebra::linalg::wilkinson_shift(x[0], x[1], x[2])]))]
}

pub fn suites() -> Vec<Suite> {
    let mut ops: Vec<Op> = (4..=6).flat_map(eigen_ops).collect();
    ops.extend(shapes().into_iter().flat_map(|(r, c)| svd_ops(r, c)));
    ops.extend(
        shapes()
            .into_iter()
            .filter(|&(r, c)| !(r == c && (2..=4).contains(&r)))
            .flat_map(|(r, c)| qr_ops(r, c)),
    );
    ops.extend([2, 3, 4, 6].into_iter().flat_map(cholesky_ops));
    ops.extend(wilkinson_ops());
    vec![Suite {
        name: "spectral",
        description: "SymmetricEigen 4..6, SVD / rank / pseudo-inverse / polar of every shape, QR \
                      of the remaining shapes, Cholesky updates, wilkinson_shift (WP 8.5-P14b)",
        ops,
    }]
}
