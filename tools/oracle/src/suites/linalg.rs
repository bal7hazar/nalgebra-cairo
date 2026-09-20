//! `nalgebra::linalg`: Cholesky, LU, UDU / LDLT (2..6), SymmetricEigen and SVD (2, 3), QR (2..4).
//!
//! Upstream runs the same generic code on static and dynamic matrices; `DMatrix` is used here so
//! that one definition covers every size.

use super::{dm, flat};
use crate::engine::{fm, fs, fv, iv, with, Input, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{DMatrix, DVector, UDU};

/// `2n` rounding stages per output scalar, amplified like input perturbations.
fn tol(n: usize) -> Tol {
    Tol::Sens {
        k: 2.0 * n as f64,
        base: 4.0,
    }
}

/// Iterative / squared-matrix algorithms (Jacobi sweeps, SVD through `M^T M`).
const SPECTRAL: Tol = Tol::SensMag {
    k: 16.0,
    base: 8.0,
    mag: 16.0,
};

const NEAR_SINGULAR_CAP: u64 = 1 << 44;

fn ispd(n: usize) -> Input {
    with(fm("a", n, n), Gen::Spd(n))
}

fn iwell(n: usize) -> Input {
    with(fm("a", n, n), Gen::WellCond(n))
}

fn rhs(x: &[f64], n: usize) -> DVector<f64> {
    DVector::from_column_slice(&x[n * n..n * n + n])
}

/// `L` (unit lower) and `D` of `A = L D L^T`, derived from upstream's Cholesky factor
/// (`L_chol = L * sqrt(D)`); upstream only ships the `U D U^T` variant.
fn ldlt(a: DMatrix<f64>) -> Option<(DMatrix<f64>, DVector<f64>)> {
    let n = a.nrows();
    let mut l = a.cholesky()?.l();
    let mut d = DVector::zeros(n);
    for j in 0..n {
        let diag = l[(j, j)];
        d[j] = diag * diag;
        for i in 0..n {
            l[(i, j)] /= diag;
        }
    }
    Some((l, d))
}

fn cholesky_ops(n: usize) -> Vec<Op> {
    vec![
        Op::new(
            format!("cholesky{n}_l"),
            "a.cholesky().unwrap().l() (lower triangular, zeros above the diagonal)",
        )
        .input(ispd(n))
        .out(fm("l", n, n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| dm(x, n).cholesky().map(|c| flat(&c.l()))),
        Op::new(
            format!("cholesky{n}_solve"),
            "a.cholesky().unwrap().solve(&b)",
        )
        .input(ispd(n))
        .input(iv("b", n))
        .out(fv("x", n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| dm(x, n).cholesky().map(|c| flat(&c.solve(&rhs(x, n))))),
        Op::new(
            format!("cholesky{n}_inverse"),
            "a.cholesky().unwrap().inverse()",
        )
        .input(ispd(n))
        .out(fm("inverse", n, n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| dm(x, n).cholesky().map(|c| flat(&c.inverse()))),
    ]
}

fn udu_ops(n: usize) -> Vec<Op> {
    vec![
        Op::new(
            format!("udu{n}_u_d"),
            "UDU::new(a).unwrap(): a = u * diag(d) * u^T, u unit UPPER triangular (upstream layout)",
        )
        .input(ispd(n))
        .out(fm("u", n, n))
        .out(fv("d", n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            let udu = UDU::new(dm(x, n))?;
            let mut out = flat(&udu.u);
            out.extend(flat(&udu.d));
            Some(out)
        }),
        Op::new(
            format!("ldlt{n}_l_d"),
            "a = l * diag(d) * l^T, l unit LOWER triangular (derived from upstream Cholesky: \
             l = L * diag(L)^-1, d = diag(L)^2)",
        )
        .input(ispd(n))
        .out(fm("l", n, n))
        .out(fv("d", n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            let (l, d) = ldlt(dm(x, n))?;
            let mut out = flat(&l);
            out.extend(flat(&d));
            Some(out)
        }),
        Op::new(
            format!("udu{n}_solve"),
            "x = u^-T * diag(d)^-1 * u^-1 * b from UDU::new(a) (same x for an LDLT solve)",
        )
        .input(ispd(n))
        .input(iv("b", n))
        .out(fv("x", n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            let udu = UDU::new(dm(x, n))?;
            let y = udu.u.solve_upper_triangular(&rhs(x, n))?;
            let z = y.component_div(&udu.d);
            udu.u.tr_solve_upper_triangular(&z).map(|x| flat(&x))
        }),
        Op::new(
            format!("udu{n}_inverse"),
            "a^-1 = u^-T * diag(d)^-1 * u^-1 from UDU::new(a) (same inverse for LDLT)",
        )
        .input(ispd(n))
        .out(fm("inverse", n, n))
        .dists(&Dist::NO_LARGE)
        .tol(tol(n))
        .eval(move |x| {
            let udu = UDU::new(dm(x, n))?;
            let mut y = udu.u.solve_upper_triangular(&DMatrix::identity(n, n))?;
            for i in 0..n {
                for j in 0..n {
                    y[(i, j)] /= udu.d[i];
                }
            }
            udu.u.tr_solve_upper_triangular(&y).map(|inv| flat(&inv))
        }),
    ]
}

fn lu_ops(n: usize) -> Vec<Op> {
    let mut ops = vec![
        Op::new(format!("lu{n}_solve"), "a.lu().solve(&b).unwrap()")
            .input(iwell(n))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::NO_LARGE)
            .tol(tol(n))
            .eval(move |x| dm(x, n).lu().solve(&rhs(x, n)).map(|x| flat(&x))),
        Op::new(format!("lu{n}_inverse"), "a.lu().try_inverse().unwrap()")
            .input(iwell(n))
            .out(fm("inverse", n, n))
            .dists(&Dist::NO_LARGE)
            .tol(tol(n))
            .eval(move |x| dm(x, n).lu().try_inverse().map(|inv| flat(&inv))),
        Op::new(format!("lu{n}_determinant"), "a.lu().determinant()")
            .input(iwell(n))
            .out(fs("determinant"))
            .dists(&Dist::NO_LARGE)
            .tol(tol(n))
            .eval(move |x| Some(vec![dm(x, n).lu().determinant()])),
    ];
    if n == 3 || n == 6 {
        ops.push(
            Op::new(
                format!("lu{n}_solve_near_singular"),
                "a.lu().solve(&b).unwrap(), condition number 1e2..1e4 (FLAGGED: loose tolerance)",
            )
            .input(with(fm("a", n, n), Gen::NearSingular(n)))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::UNIT)
            .cap(NEAR_SINGULAR_CAP)
            .tol(tol(n))
            .eval(move |x| dm(x, n).lu().solve(&rhs(x, n)).map(|x| flat(&x))),
        );
    }
    ops
}

fn sorted(mut values: Vec<f64>, descending: bool) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    if descending {
        values.reverse();
    }
    values
}

fn spectral_ops(n: usize) -> Vec<Op> {
    vec![
        Op::new(
            format!("symmetric_eigen{n}_eigenvalues"),
            "a.symmetric_eigen().eigenvalues sorted ASCENDING (eigenvectors are sign / order \
             ambiguous: check them by reconstruction V * diag(lambda) * V^T = a)",
        )
        .input(with(fm("a", n, n), Gen::Sym(n)))
        .out(fv("eigenvalues", n))
        .dists(&Dist::NO_LARGE)
        .tol(SPECTRAL)
        .eval(move |x| Some(sorted(flat(&dm(x, n).symmetric_eigen().eigenvalues), false))),
        Op::new(
            format!("symmetric_eigen{n}_eigenvalues_spd"),
            "same on symmetric positive-definite matrices (inertia tensors)",
        )
        .input(with(fm("a", n, n), Gen::Spd(n)))
        .out(fv("eigenvalues", n))
        .dists(&Dist::NO_LARGE)
        .tol(SPECTRAL)
        .eval(move |x| Some(sorted(flat(&dm(x, n).symmetric_eigen().eigenvalues), false))),
        Op::new(
            format!("svd{n}_singular_values"),
            "a.singular_values() sorted DESCENDING (well-conditioned a; check U, V by \
             reconstruction U * diag(sigma) * V^T = a)",
        )
        .input(iwell(n))
        .out(fv("singular_values", n))
        .dists(&Dist::NO_LARGE)
        .tol(SPECTRAL)
        .eval(move |x| Some(sorted(flat(&dm(x, n).singular_values()), true))),
    ]
}

fn qr_ops(n: usize) -> Vec<Op> {
    vec![
        Op::new(
            format!("qr{n}_q_r"),
            "a.qr(): (q, r) with upstream's Householder signs (diag(r) may be negative; to compare \
             with another convention flip the sign of row i of r and column i of q)",
        )
        .input(iwell(n))
        .out(fm("q", n, n))
        .out(fm("r", n, n))
        .dists(&Dist::NO_LARGE)
        .tol(Tol::SensMag {
            k: 2.0 * n as f64,
            base: 4.0,
            mag: 2.0 * n as f64,
        })
        .eval(move |x| {
            let qr = dm(x, n).qr();
            let mut out = flat(&qr.q());
            out.extend(flat(&qr.r()));
            Some(out)
        }),
        Op::new(format!("qr{n}_solve"), "a.qr().solve(&b).unwrap()")
            .input(iwell(n))
            .input(iv("b", n))
            .out(fv("x", n))
            .dists(&Dist::NO_LARGE)
            .tol(tol(n))
            .eval(move |x| dm(x, n).qr().solve(&rhs(x, n)).map(|x| flat(&x))),
    ]
}

pub fn suites() -> Vec<Suite> {
    vec![
        Suite {
            name: "cholesky",
            description: "Cholesky 2..6 on SPD matrices: factor l, solve, inverse",
            ops: (2..=6).flat_map(cholesky_ops).collect(),
        },
        Suite {
            name: "udu",
            description:
                "UDU (upstream) and LDLT (DESIGN D6) 2..6 on SPD matrices: factors, solve, \
                          inverse",
            ops: (2..=6).flat_map(udu_ops).collect(),
        },
        Suite {
            name: "lu",
            description:
                "LU with partial pivoting 2..6 on well-conditioned matrices: solve, inverse, \
                          determinant (+ flagged near-singular solves)",
            ops: (2..=6).flat_map(lu_ops).collect(),
        },
        Suite {
            name: "symmetric_eigen_svd",
            description:
                "SymmetricEigen eigenvalues (ascending) and SVD singular values (descending), \
                          2x2 and 3x3",
            ops: (2..=3).flat_map(spectral_ops).collect(),
        },
        Suite {
            name: "qr",
            description: "QR 2..4 on well-conditioned matrices: factors (upstream signs), solve",
            ops: (2..=4).flat_map(qr_ops).collect(),
        },
    ]
}
