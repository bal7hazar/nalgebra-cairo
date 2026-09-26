//! WP 8.5-P16, suite `schur`: the reductions and the non-symmetric eigenvalue problem of upstream
//! `nalgebra::linalg` — `Hessenberg` (squares 1..6), `Bidiagonal` (every shape 1..6 x 1..6),
//! `SymmetricTridiagonal` (1..6), the real `Schur` decomposition through its eigenvalues (real
//! and complex, on general matrices and on structured spectra), and Parlett-Reinsch balancing.
//!
//! Upstream runs the same generic code on static and dynamic matrices; `DMatrix` is used so that
//! one definition covers every size. The factors of the reductions follow upstream's Householder
//! sign conventions and are compared entry by entry. The Schur factors `Q`, `T` are not emitted
//! (non-unique: the order of the diagonal blocks and the 2x2 blocks' form depend on the shifts):
//! the Cairo tests check them by reconstruction and compare the EIGENVALUES, emitted as `re` /
//! `im` sorted by (re ascending, then im ascending) — or `values` sorted ascending when the
//! spectrum is real.
//!
//! The structured spectra are built in f64 then quantised: the oracle evaluates the QUANTISED
//! input, so its spectrum is the one of the raws, not exactly the construction's. Cases whose
//! eigenvalues are not a smooth function of the input (ordering flips, real pairs turning complex,
//! double eigenvalues) have no measurable sensitivity and are resampled. The clustered and the
//! defective ops are FLAGGED (sensitive by construction: raised caps, loose tolerances).

use super::flat;
use crate::engine::{fm, fv, with, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::DMatrix;

/// Schur eigenvalues: an iterative algorithm with a data-dependent number of shifted QR sweeps,
/// i.e. of rounded unit-scale reflections applied to the input in Q32.32 (the error grows with
/// the iteration count, not only with the size). The Q32.32 iteration normalises the input by its
/// largest entry, so its backward error is relative to `max |input|` (`SensScaled`).
const SCHUR: Tol = Tol::SensScaled {
    k: 32.0,
    base: 64.0,
    mag: 256.0,
};

/// Schur eigenvalues of ILL-CONDITIONED spectra (non-normal, defective, near-triangular with a
/// non-normal upper triangle): the Q32.32 Schur iteration normalises the input by its largest
/// entry, so its backward error (rounding of the reflections, deflation at the Q32.32 noise
/// floor) is relative to `max |input|`, and the eigenvalue sensitivity `A` amplifies it (WP
/// 8.5-P16: measured `k` up to 95 on defective, 37 on near-triangular inputs).
const SCHUR_ILL: Tol = Tol::SensScaled {
    k: 128.0,
    base: 64.0,
    mag: 256.0,
};

/// `SCHUR_ILL` for the non-normal spectra (measured `k` up to 6: their sensitivity is large
/// already, a larger `k` would put most cases above the default cap).
const SCHUR_NONNORMAL: Tol = Tol::SensScaled {
    k: 16.0,
    base: 64.0,
    mag: 256.0,
};

/// FLAGGED clustered spectra (relative gaps 1e-3).
const CLUSTERED_CAP: u64 = 1 << 40;
/// FLAGGED defective spectra (a Jordan block of size 2: square-root sensitivity).
const DEFECTIVE_CAP: u64 = 1 << 44;

/// Iteration cap of upstream `try_schur` in the eigenvalue ops (threshold `f64::EPSILON`).
const MAX_NITER: usize = 10_000;

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

/// Householder reflections: rounded unit-scale factors applied to the input (`SensMag`), as the
/// `pivot` suite's QR.
fn qr_tol(r: usize, c: usize) -> Tol {
    let k = r.min(c) as f64;
    Tol::SensMag {
        k: 4.0 * k,
        base: 8.0,
        mag: 4.0 * k,
    }
}

/// `(re..., im...)` of the complex eigenvalues of `a.try_schur(f64::EPSILON, 10_000)`, sorted by
/// (re ascending, then im ascending). `None` when the iteration does not converge.
fn complex_eigenvalues(a: DMatrix<f64>) -> Option<Vec<f64>> {
    let schur = a.try_schur(f64::EPSILON, MAX_NITER)?;
    let mut values: Vec<(f64, f64)> = schur
        .complex_eigenvalues()
        .iter()
        .map(|c| (c.re, c.im))
        .collect();
    if values
        .iter()
        .any(|(re, im)| !re.is_finite() || !im.is_finite())
    {
        return None;
    }
    values.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .expect("finite")
            .then(a.1.partial_cmp(&b.1).expect("finite"))
    });
    let mut out: Vec<f64> = values.iter().map(|v| v.0).collect();
    out.extend(values.iter().map(|v| v.1));
    Some(out)
}

fn hessenberg_op(n: usize) -> Op {
    Op::new(
        format!("hessenberg{n}"),
        format!(
            "a.hessenberg().unpack() = (q {n}x{n} orthogonal, h {n}x{n} upper Hessenberg, zeros \
             below the subdiagonal), a = q h q^T, upstream Householder signs, general a"
        ),
    )
    .input(with(fm("a", n, n), Gen::M(n, n)))
    .out(fm("q", n, n))
    .out(fm("h", n, n))
    .dists(&Dist::NO_LARGE)
    .tol(qr_tol(n, n))
    .eval(move |x| {
        let (q, h) = mat_at(x, 0, n, n).hessenberg().unpack();
        let mut out = flat(&q);
        out.extend(flat(&h));
        Some(out)
    })
}

fn bidiagonal_op(r: usize, c: usize) -> Op {
    let k = r.min(c);
    let band = if r >= c {
        "upper bidiagonal"
    } else {
        "lower bidiagonal"
    };
    Op::new(
        format!("bidiagonal{}", name(r, c)),
        format!(
            "a.bidiagonalize().unpack() = (u {r}x{k}, d {k}x{k} {band}, v_t {k}x{c}), \
             a = u d v_t, upstream Householder signs, well-conditioned a"
        ),
    )
    .input(with(fm("a", r, c), Gen::WellCondRect(r, c)))
    .out(fm("u", r, k))
    .out(fm("d", k, k))
    .out(fm("v_t", k, c))
    .dists(&Dist::NO_LARGE)
    .tol(qr_tol(r, c))
    .eval(move |x| {
        let (u, d, v_t) = mat_at(x, 0, r, c).bidiagonalize().unpack();
        let mut out = flat(&u);
        out.extend(flat(&d));
        out.extend(flat(&v_t));
        Some(out)
    })
}

fn symmetric_tridiagonal_op(n: usize) -> Op {
    let doc = if n >= 2 {
        format!(
            "a.symmetric_tridiagonalize().unpack() = (q {n}x{n} orthogonal, diagonal, \
             off_diagonal = |subdiagonal| (upstream takes the modulus)), a = q t q^T, upstream \
             Householder signs, symmetric a"
        )
    } else {
        "a.symmetric_tridiagonalize().unpack() = (q = [[1]], diagonal = a; no off-diagonal), \
         symmetric a"
            .to_string()
    };
    let mut op = Op::new(format!("symmetric_tridiagonal{n}"), doc)
        .input(with(fm("a", n, n), Gen::Sym(n)))
        .out(fm("q", n, n))
        .out(fv("diagonal", n));
    if n >= 2 {
        op = op.out(fv("off_diagonal", n - 1));
    }
    op.dists(&Dist::NO_LARGE).tol(qr_tol(n, n)).eval(move |x| {
        let (q, diagonal, off_diagonal) = mat_at(x, 0, n, n).symmetric_tridiagonalize().unpack();
        let mut out = flat(&q);
        out.extend(flat(&diagonal));
        out.extend(flat(&off_diagonal));
        Some(out)
    })
}

/// Eigenvalue op emitting `(re, im)`: complex eigenvalues of `try_schur`, sorted.
fn complex_eigenvalue_op(n: usize, suffix: &str, doc: &str, gen: Gen, tol: Tol) -> Op {
    Op::new(
        format!("schur{n}_eigenvalues{suffix}"),
        format!(
            "a.try_schur(f64::EPSILON, {MAX_NITER}).unwrap().complex_eigenvalues() sorted by \
             (re ascending, then im ascending), emitted as (re, im), {doc} (check the Schur \
             factors by reconstruction)"
        ),
    )
    .input(with(fm("a", n, n), gen))
    .out(fv("re", n))
    .out(fv("im", n))
    .dists(&Dist::NO_LARGE)
    .tol(tol)
    .eval(move |x| complex_eigenvalues(mat_at(x, 0, n, n)))
}

fn schur_ops(n: usize) -> Vec<Op> {
    let mut ops = vec![complex_eigenvalue_op(
        n,
        "",
        "general a (independent entries)",
        Gen::M(n, n),
        SCHUR,
    )];
    ops.push(
        Op::new(
            format!("schur{n}_eigenvalues_real"),
            "a.eigenvalues().unwrap() sorted ASCENDING, a = P diag(l) P^-1 (P well-conditioned) \
             with a real, well-separated spectrum (pairwise gaps >= 0.1 max |l|)",
        )
        .input(with(fm("a", n, n), Gen::SpectrumReal(n)))
        .out(fv("values", n))
        .dists(&Dist::NO_LARGE)
        .tol(SCHUR)
        .eval(move |x| {
            let mut values = flat(&mat_at(x, 0, n, n).eigenvalues()?);
            if values.iter().any(|v| !v.is_finite()) {
                return None;
            }
            values.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
            Some(values)
        }),
    );
    if n < 2 {
        return ops;
    }
    ops.push(complex_eigenvalue_op(
        n,
        "_complex",
        "a = P B P^-1 (P well-conditioned), B block-diagonal with 2x2 blocks [[a, b], [-b, a]] \
         (|b| >= 0.25 scale: complex-conjugate pairs a +- ib) plus a 1x1 real block when n is odd",
        Gen::SpectrumComplex(n),
        SCHUR,
    ));
    ops.push(complex_eigenvalue_op(
        n,
        "_nonnormal",
        "NON-NORMAL a = Q T Q^T (Q orthogonal), T upper triangular with a real separated \
         diagonal and strictly-upper entries up to 10x the diagonal spread",
        Gen::NonNormal(n),
        SCHUR_NONNORMAL,
    ));
    ops.push(
        complex_eigenvalue_op(
            n,
            "_clustered",
            "a = P diag(l) P^-1 with CLUSTERED real eigenvalues (relative gaps 1e-3) (FLAGGED: \
             sensitive, loose tolerance)",
            Gen::SpectrumClustered(n),
            SCHUR,
        )
        .cap(CLUSTERED_CAP),
    );
    ops.push(
        complex_eigenvalue_op(
            n,
            "_defective",
            "DEFECTIVE a = P J P^-1, J with a Jordan block [[l, s], [0, l]] of size 2 (the \
             quantised input splits the double eigenvalue by about sqrt(ulp), real or complex) \
             (FLAGGED: square-root sensitivity, loose tolerance)",
            Gen::Defective(n),
            SCHUR_ILL,
        )
        .cap(DEFECTIVE_CAP),
    );
    ops.push(complex_eigenvalue_op(
        n,
        "_near_triangular",
        "upper quasi-triangular a (entries of the magnitude class, separated real diagonal) with \
         every subdiagonal entry tiny (|x| in [1e-7, 1e-5]): the near-convergence case",
        Gen::NearTriangular(n),
        SCHUR_ILL,
    ));
    ops
}

fn balance_op(n: usize) -> Op {
    Op::new(
        format!("balance{n}"),
        "let d = balance_parlett_reinsch(&mut m) on m = a: (balanced m = D^-1 a D, d = the \
         diagonal of D, powers of two), a = S M S^-1 badly scaled (S = diag(2^k), k in -6..6)",
    )
    .input(with(fm("a", n, n), Gen::BadlyScaled(n)))
    .out(fm("balanced", n, n))
    .out(fv("d", n))
    .dists(&Dist::NO_LARGE)
    .tol(Tol::Sens { k: 2.0, base: 4.0 })
    .eval(move |x| {
        let mut m = mat_at(x, 0, n, n);
        let d = nalgebra::linalg::balancing::balance_parlett_reinsch(&mut m);
        let mut out = flat(&m);
        out.extend(flat(&d));
        Some(out)
    })
}

pub fn suites() -> Vec<Suite> {
    let mut ops: Vec<Op> = (1..=6).map(hessenberg_op).collect();
    ops.extend(
        (1..=6)
            .flat_map(|r| (1..=6).map(move |c| (r, c)))
            .map(|(r, c)| bidiagonal_op(r, c)),
    );
    ops.extend((1..=6).map(symmetric_tridiagonal_op));
    ops.extend((1..=6).flat_map(schur_ops));
    ops.extend((1..=6).map(balance_op));
    vec![Suite {
        name: "schur",
        description: "Hessenberg 1..6, Bidiagonal of every shape, SymmetricTridiagonal 1..6, \
                      Schur eigenvalues (real / complex, general and structured spectra) 1..6, \
                      Parlett-Reinsch balancing 1..6 (WP 8.5-P16)",
        ops,
    }]
}
