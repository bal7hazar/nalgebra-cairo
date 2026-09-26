"""WP 8.5-P17 part of the `linalg` generator (imported by `generate.py`, which owns the driver and
the `--check` mode): the matrix exponential `exp` (`linalg/exp.cairo`) and the integer power
`pow` / `pow_mut` (`linalg/pow.cairo`) of the static squares 1..6, plus their test package
`crates/tests_linalg_exp`.

Design measurements (bit-faithful Q32.32 Python model of the emitted `exp`, WP 8.5-P17 report):
the Padé degrees 3 / 5 / 7 with the backward-error thresholds of Higham (2005) recomputed for the
Q32.32 unit roundoff `u = 2^-32` (`THETA`), degree 7 with scaling and squaring beyond. Every
emitted decision is restated in the doc comments.
"""

from __future__ import annotations

from generate import (BOUNDS, HEADER, TEST_MANIFEST, fld, int_rows, render_builders, struct_lit,
                      tname)
from p15 import chain, let_div, perm_lit, swaps, v

DIMS = range(1, 7)

# Padé [m/m] coefficients of exp (upstream's `b` arrays), for the degrees used here.
PADE = {
    3: [120, 60, 12, 1],
    5: [30240, 15120, 3360, 420, 30, 1],
    7: [17297280, 8648640, 1995840, 277200, 25200, 1512, 56, 1],
}
# theta_m for u = 2^-32 (Higham 2005, backward error `|ΔA| <= u |A|`), computed with mpmath from
# the series of `log(e^-x r_m(x))`, ROUNDED DOWN to the fraction below (a smaller threshold is
# always safe). For u = 2^-53 the same computation gives upstream's 1.4955852e-2, 2.5393983e-1,
# 9.5041790e-1 (a check of the method).
THETA = {3: (16917, 100000), 5: (10858, 10000), 7: (26681, 10000)}


def mt(n: int) -> str:
    return tname(n, n)


def mod_path(n: int) -> str:
    return f"crate::base::{tname(n, n).lower()}"


def uses(n: int) -> str:
    return f"use {mod_path(n)}::{{{mt(n)}, {mt(n)}Trait}};"


def lit(n: int, value) -> str:
    return struct_lit(n, n, value)


def comb(n: int, terms, diag: str | None) -> str:
    """`Σ c_k M_k (+ diag on the diagonal)` entry by entry: integer-valued coefficients, so every
    product is exact (one fused kernel per entry, no rounding)."""
    def entry(i, j):
        prods = [(c, f"{m}.{fld(n, n, i, j)}") for c, m in terms]
        if not prods:
            e = None
        elif len(prods) == 1:
            c, x = prods[0]
            e = x if c is None else f"{c} * {x}"
        else:
            if any(c is None for c, _ in prods):
                plain = [x for c, x in prods if c is None]
                rest = [(c, x) for c, x in prods if c is not None]
                e = " + ".join(plain) + " + " + (
                    f"{rest[0][0]} * {rest[0][1]}" if len(rest) == 1 else
                    f"R::sum_prod{len(rest)}({', '.join(f'{c}, {x}' for c, x in rest)})")
            else:
                e = f"R::sum_prod{len(prods)}({', '.join(f'{c}, {x}' for c, x in prods)})"
        if i == j and diag is not None:
            return f"{e} + {diag}" if e else diag
        return e if e else "R::zero()"
    return lit(n, entry)


def pade_fn(n: int, m: int) -> str:
    b = PADE[m]
    M = mt(n)
    lets = ["let a2 = a.mul_mat(a);"]
    if m >= 5:
        lets.append("let a4 = a2.mul_mat(a2);")
    if m >= 7:
        lets.append("let a6 = a4.mul_mat(a2);")
    pw = {2: "a2", 4: "a4", 6: "a6"}
    consts = [f"let c{k} = R::from_int({b[k]});" for k in range(m) if b[k] != 1]
    # W = b_m A^{m-1} + ... + b_3 A^2 + b_1 I  (b_m = 1), V = b_{m-1} A^{m-1} + ... + b_0 I
    wt = [(None if b[k] == 1 else f"c{k}", pw[k - 1]) for k in range(m, 2, -2)]
    vt = [(f"c{k}", pw[k]) for k in range(m - 1, 1, -2)]
    w = comb(n, wt, "c1")
    vv = comb(n, vt, "c0")
    degree_doc = {3: "A²", 5: "A², A⁴", 7: "A², A⁴, A⁶"}[m]
    nprod = {3: 2, 5: 3, 7: 4}[m]
    return f"""    /// The Padé [{m}/{m}] approximant of `exp(a)` as `(P, Q) = (V + U, V - U)`, upstream's
    /// `pade{m}`: {degree_doc} ({nprod - 1} rounded product{'s' if nprod > 2 else ''}), `V = Σ b_2k A^2k`, `W = Σ b_(2k+1) A^2k`
    /// (integer coefficients: every term exact, one fused kernel per entry), `U = a W` (one more
    /// rounded product). The magnitude of `P` and `Q` is `b_0 = {b[0]}`, so the absolute
    /// rounding of the products is `1 / b_0` relative.
    fn pade{m}(a: {M}<T>) -> ({M}<T>, {M}<T>) {{
        {' '.join(lets)}
        {' '.join(consts)}
        let w = {w};
        let v = {vv};
        let u = a.mul_mat(w);
        (v + u, v - u)
    }}
"""


def solve_fn(n: int) -> str:
    M = mt(n)
    head = f"""    /// `Q⁻¹ P`, upstream's `solve_p_q` (`q.lu().solve(&p).unwrap()`): the LU factorisation of
    /// `Q` with partial pivoting and the solve of the {n} columns of `P`. `Q` is the Padé
    /// denominator of a matrix within the threshold of its degree, well conditioned; panics with
    /// `nalgebra: singular Pade denom` (upstream's `unwrap` panic) if a pivot is exactly zero."""
    if n != 5:
        return head + f"""
    fn solve_pq(p: {M}<T>, q: {M}<T>) -> {M}<T> {{
        let mut x = p;
        assert!(Lu{n}Trait::new(q).solve_mut(ref x), "{{}}", SINGULAR_PADE);
        x
    }}
"""
    # No `Lu5` in the crate (DESIGN D4 sizes): the partial-pivot factorisation of `Lu3` / `Lu6`,
    # unrolled here, then the shared permutation and triangular kernels.
    st = [f"let mut {v(i, j)} = q.{fld(n, n, i, j)};" for j in range(n) for i in range(n)]
    for i in range(n - 1):
        st.append(f"// step {i + 1}: the first largest |a_k{i + 1}| of rows {i + 1}..{n}")
        st.append(f"let mut piv = R::abs({v(i, i)});")
        st.append(f"let mut rp{i} = {i + 1}_u8;")
        for t in range(i + 1, n):
            st.append(f"let x = R::abs({v(t, i)}); if x > piv {{ piv = x; rp{i} = {t + 1}; }}")
        body = [chain(f"rp{i}", range(i + 1, n),
                      lambda bb, i=i: swaps([(v(i, j), v(bb, j)) for j in range(n)]))]
        ls = [f"l{t}" for t in range(i + 1, n)]
        body.append(let_div(ls, [v(t, i) for t in range(i + 1, n)], v(i, i)))
        for t in range(i + 1, n):
            body.append(f"let nl = -l{t};")
            for k in range(i + 1, n):
                body.append(f"{v(t, k)} = R::mul_add(nl, {v(i, k)}, {v(t, k)});")
            body.append(f"{v(t, i)} = l{t};")
        st.append(f"if piv != R::zero() {{\n{chr(10).join(body)}\n}}")
    lu = lit(n, lambda i, j: v(i, j))
    perm = perm_lit(n, lambda k: f"rp{k}" if k < n - 1 else None)
    return head + f""" `Matrix5` has no `Lu5` (DESIGN D4 sizes): the
    /// factorisation of `Lu3` / `Lu6` (first largest pivot, one correctly rounded division per
    /// multiplier, one `mul_add` per update) is unrolled here, then `P` is permuted and solved by
    /// the crate's triangular kernels, as `LU::solve_mut` does.
    fn solve_pq(p: {M}<T>, q: {M}<T>) -> {M}<T> {{
        revoke_ap_tracking();
        {chr(10).join(st)}
        let lu = {lu};
        assert!(SolveKernel::<{M}<T>, {M}<T>>::nonzero_diagonal(lu), "{{}}", SINGULAR_PADE);
        let mut x = p;
        PermuteRows::permute_rows({perm}, ref x);
        SolveKernel::<{M}<T>, {M}<T>>::upper(lu, SolveKernel::<{M}<T>, {M}<T>>::lower_unit(lu, x))
    }}
"""


def exp_doc(n: int) -> str:
    return f"""    /// The matrix exponential `exp(a) = Σ a^k / k!`. Upstream: `OMatrix::exp`.
    ///
    /// Padé approximant with scaling and squaring, upstream's family (Higham 2005 / Al-Mohy and
    /// Higham 2009, scipy's `expm`), with the degree chosen by the 1-norm thresholds:
    ///
    /// ```text
    /// |a|₁ <= θ₃ = 0.16917        Padé [3/3]    2 rounded products + LU solve
    /// |a|₁ <= θ₅ = 1.0858         Padé [5/5]    3 rounded products + LU solve
    /// otherwise, s = the smallest integer with |a|₁ / 2^s <= θ₇ = 2.6681:
    ///                             Padé [7/7] of a / 2^s, then s squarings
    ///                             4 + s rounded products + LU solve
    /// ```
    ///
    /// The thresholds `θ_m` are Higham's backward-error bounds (`exp(a + Δa)` with `|Δa| <= u |a|`)
    /// recomputed for the Q32.32 unit roundoff `u = 2^-32` (upstream's, for `f64`'s `2^-53`, are
    /// 0.01496 / 0.2539 / 0.9504, plus degrees 9 and 13), rounded down. Deviations from upstream,
    /// all cheaper (steps criterion) and within the oracle tolerance:
    /// - the degree is chosen on `|a|₁` alone, without upstream's exact norms of `a⁴` .. `a¹⁰` and
    ///   its `ell` backward-error refinement (matrix-vector products and a base-2 logarithm);
    /// - no degree 9 or 13: at the Q32.32 thresholds (θ₉ = 4.65 < 2 θ₇, θ₁₃ = 9.22 < 4 θ₇) they
    ///   cost as many rounded products as degree 7 with one / two more squarings, and their
    ///   coefficients (up to 6.5e16) do not fit Q32.32 as integers;
    /// - the coefficients are upstream's integers (`R::from_int`), so `V`, `W` are exact
    ///   combinations of the powers and `P`, `Q` have the magnitude `b_0` (1.7e7 for degree 7).
    ///
    /// Accuracy (model, then the oracle suite `matrix_functions` in `tests_linalg_exp`): within a
    /// few ulp per unit of `max |exp(a)|` at the thresholds; each squaring roughly doubles the
    /// relative error already committed (14 ulp per unit at `|a|₁ = 15` in 6x6, 3 squarings).
    /// `a / 2^s` is the exact floor `a_ij >> s`. The scaling loop is data-dependent (`s` grows
    /// with `log2 |a|₁`, like upstream's), everything else is unrolled.
    ///
    /// Panics with the scalar's overflow error when a column sum of `|a_ij|`, a Padé term or a
    /// squaring does not fit (in particular whenever `exp(a)` itself does not), and with
    /// `nalgebra: singular Pade denom` if the LU solve meets an exactly zero pivot (upstream's
    /// `unwrap`). Sierra gas charges the degree branches at their costliest path; Cairo steps
    /// follow the executed branch.
    fn exp(self: {mt(n)}<T>) -> {mt(n)}<T> {{
        let norm = self.one_norm();
        if norm <= R::from_ratio({THETA[3][0]}, {THETA[3][1]}) {{
            let (p, q) = Exp{n}InternalTrait::pade3(self);
            return Exp{n}InternalTrait::solve_pq(p, q);
        }}
        if norm <= R::from_ratio({THETA[5][0]}, {THETA[5][1]}) {{
            let (p, q) = Exp{n}InternalTrait::pade5(self);
            return Exp{n}InternalTrait::solve_pq(p, q);
        }}
        let theta7 = R::from_ratio({THETA[7][0]}, {THETA[7][1]});
        let mut norm = norm;
        let mut factor = R::one();
        let mut s: u32 = 0;
        while norm > theta7 {{
            norm = norm * R::HALF;
            factor = factor * R::HALF;
            s += 1;
        }}
        let b = if s == 0 {{
            self
        }} else {{
            self.scale(factor)
        }};
        let (p, q) = Exp{n}InternalTrait::pade7(b);
        let mut x = Exp{n}InternalTrait::solve_pq(p, q);
        while s != 0 {{
            x = x.mul_mat(x);
            s -= 1;
        }}
        x
    }}
"""


def render_exp() -> str:
    parts = [f"""{HEADER}//! The matrix exponential of the static squares (upstream `nalgebra::linalg::exp`,
//! `OMatrix::exp`, WP 8.5-P17): `Matrix1ExpTrait` .. `Matrix6ExpTrait`. `Matrix1` is the scalar
//! exponential (upstream's `nrows() == 1` case); the others are the Padé approximant with scaling
//! and squaring documented on `Matrix2ExpTrait::exp`. Upstream's generic impl covers `DMatrix`
//! too: not ported (the D5 dynamic types have no LU; `SquareMatrix` only requires the static
//! squares, `docs/API_PARITY.md`).

use core::internal::revoke_ap_tracking;
use simba::scalar::{{Real, Transcendental}};
use crate::base::MatrixMul;
use crate::base::solve::SolveKernel;
use crate::linalg::permutation_sequence::PermuteRows;
use crate::linalg::lu::perm1_5::Perm5;
{chr(10).join(f'use crate::linalg::lu::lu{n}::Lu{n}Trait;' for n in (2, 3, 4, 6))}
use crate::base::matrix1::Matrix1;
{chr(10).join(uses(n) for n in range(2, 7))}

/// Panic message of the LU solve of a Padé denominator with an exactly zero pivot (upstream
/// unwraps `LU::solve`).
const SINGULAR_PADE: felt252 = 'nalgebra: singular Pade denom';

/// `Matrix1::exp`: the scalar exponential. Import `Matrix1ExpTrait` to use it.
#[generate_trait]
pub impl Matrix1ExpImpl<
{BOUNDS}
> of Matrix1ExpTrait<T> {{
    /// `exp(x)` of the single entry, upstream's `nrows() == 1` case (`self.map(|v| v.exp())`):
    /// `Transcendental::exp`, whose error and domain are the scalar's (fixed-cairo `ExpTrait`).
    /// Upstream: `OMatrix::exp`.
    fn exp<+Transcendental<T>>(self: Matrix1<T>) -> Matrix1<T> {{
        Matrix1 {{ x: Transcendental::exp(self.x) }}
    }}
}}
"""]
    for n in range(2, 7):
        parts.append(f"""/// `Matrix{n}::exp`. Import `Matrix{n}ExpTrait` to use it.
#[generate_trait]
pub impl Matrix{n}ExpImpl<
{BOUNDS}
> of Matrix{n}ExpTrait<T> {{
{exp_doc(n)}}}

/// Crate-internal kernels of `Matrix{n}ExpTrait::exp`: the Padé approximants and the LU solve.
#[generate_trait]
pub(crate) impl Exp{n}InternalImpl<
{BOUNDS}
> of Exp{n}InternalTrait<T> {{
{pade_fn(n, 3)}
{pade_fn(n, 5)}
{pade_fn(n, 7)}
{solve_fn(n)}}}
""")
    return "\n".join(parts)


def render_pow() -> str:
    parts = [f"""{HEADER}//! The integer power of the static squares (upstream `nalgebra::linalg::pow`, `Matrix::pow` /
//! `Matrix::pow_mut`, WP 8.5-P17): `Matrix1PowTrait` .. `Matrix6PowTrait`. Upstream's generic impl
//! covers `DMatrix` too: not ported (`SquareMatrix` only requires the static squares,
//! `docs/API_PARITY.md`).

use crate::base::MatrixMul;
use simba::scalar::Real;
{chr(10).join(uses(n) for n in DIMS)}
"""]
    for n in DIMS:
        M = mt(n)
        parts.append(f"""/// `Matrix{n}::pow` / `pow_mut`. Import `Matrix{n}PowTrait` to use them.
#[generate_trait]
pub impl Matrix{n}PowImpl<
{BOUNDS}
> of Matrix{n}PowTrait<T> {{
    /// Raises `self` to the integral power `exp` in place: the identity for `exp = 0`,
    /// exponentiation by squaring otherwise, upstream's loop: for each bit of `exp` from the
    /// lowest, `self = self * x` when the bit is set, then `x = x * x` while bits remain
    /// (`MatrixMul::mul_mat`, one fused sum of {n} products floored once per entry). Upstream
    /// starts from the identity when `exp` is even and multiplies it by the first power: here
    /// that product is a move (`I x = x` exactly in fixed point too), bit-identical and one
    /// product cheaper. At most `2 log2(exp)` products; the loop runs once per bit of `exp`.
    /// Panics with the scalar's overflow error when a product does not fit. Upstream:
    /// `Matrix::pow_mut`.
    fn pow_mut(ref self: {M}<T>, exp: u32) {{
        if exp == 0 {{
            self = {M}Trait::identity();
            return;
        }}
        let mut x = self;
        let mut e = exp;
        // `self` still stands for the identity until the first set bit.
        let mut first = true;
        loop {{
            let (q, r) = DivRem::div_rem(e, 2);
            if r == 1 {{
                if first {{
                    self = x;
                    first = false;
                }} else {{
                    self = self.mul_mat(x);
                }}
            }}
            e = q;
            if e == 0 {{
                break;
            }}
            x = x.mul_mat(x);
        }}
    }}

    /// `self` raised to the integral power `exp`, see `pow_mut`. Upstream: `Matrix::pow`.
    fn pow(self: {M}<T>, exp: u32) -> {M}<T> {{
        let mut result = self;
        Self::pow_mut(ref result, exp);
        result
    }}
}}
""")
    return "\n".join(parts)


# --- test package ------------------------------------------------------------------------------

PKG = "tests_linalg_exp"
MAX_PER_DIST = 3


def exp_tests(n: int) -> str:
    M = mt(n)
    cmp = " ".join(
        f"ex = max(ex, excess(ulp_diff(got.{fld(n, n, i, j)}, e.{fld(n, n, i, j)}), oracle_tol(abs_raw(e.{fld(n, n, i, j)}), tol)));"
        for i in range(n) for j in range(n))
    ops = [""] + (["_skew"] if n > 1 else []) + ["_scaled"]
    out = []
    for sfx in ops:
        out.append(f"""/// `exp{n}{sfx}` (oracle): every entry within the oracle tolerance.
#[test]
fn test_oracle_exp{n}{sfx}() {{
    let mut cases = oracle::exp{n}{sfx}_cases();
    let mut ex = 0;
    while let Some(case) = cases.pop_front() {{
        let (a, e, tol) = *case;
        let got = black_box(mat{n}x{n}(a)).exp();
        let e = mat{n}x{n}(e);
        {cmp}
    }}
    assert!(ex == 0, "oracle tolerance exceeded by {{}}", ex);
}}""")
    zero = int_rows([[0] * n for _ in range(n)])
    ident = [[(1 << 32) if i == j else 0 for j in range(n)] for i in range(n)]
    if n >= 2:
        # nilpotent: the first superdiagonal of ones; exp(N) = Σ N^k / k! (exact dyadic when
        # n <= 3: 1, 1, 1/2)
        nil = [[(1 << 32) if j == i + 1 else 0 for j in range(n)] for i in range(n)]
        out.append(f"""/// Exact cases: `exp(0) = I` (degree 3, `Q = P = 120 I`), and the nilpotent `N` of ones on
/// the superdiagonal (`|N|₁ = 1`: degree 5), whose Padé [5/5] is `exp(N)` exactly (`N⁷ = 0`), up
/// to the rounding of the `1 / k!` terms.
#[test]
fn test_exp{n}_exact_cases() {{
    let z = black_box(mat{n}x{n}({zero}));
    assert!(z.exp() == mat{n}x{n}({int_rows(ident)}));
    let n = black_box(mat{n}x{n}({int_rows(nil)}));
    let e = n.exp();
    let mut expected = mat{n}x{n}({int_rows(ident)});
    {exact_nilpotent(n)}
    assert!(max_ulp_{n}x{n}(e, expected) <= 2, "nilpotent {{}}", max_ulp_{n}x{n}(e, expected));
}}

/// `exp(a)` overflows: the scalar's overflow panic (a squaring of `exp(diag(30))`).
#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_exp{n}_overflow_panics() {{
    let a = black_box(mat{n}x{n}({int_rows([[30 * (1 << 32) if i == j else 0 for j in range(n)] for i in range(n)])}));
    let _ = a.exp();
}}""")
    else:
        out.append(f"""/// `exp(0) = 1` exactly.
#[test]
fn test_exp1_exact_cases() {{
    assert!(black_box(mat1x1({zero})).exp() == mat1x1({int_rows(ident)}));
}}""")
    # benches: one per degree branch
    pick = {3: "pade3", 5: "pade5", 7: "pade7_scaled"}
    out.append(f"""#[test]
#[inline(never)]
fn bench_exp{n}__baseline() {{
    let (a, _, _) = *oracle::exp{n}_cases().at(0);
    let _a = black_box(mat{n}x{n}(a));
    let e = black_box(true);
    assert!(e == e);
}}""")
    if n == 1:
        out.append(f"""/// The scalar exponential.
#[test]
#[inline(never)]
fn bench_exp1__scalar() {{
    let (a, _, _) = *oracle::exp1_cases().at(0);
    let a = black_box(mat1x1(a));
    let e = black_box(true);
    let x = a.exp();
    assert!((x.x == x.x) == e);
}}""")
    else:
        for deg, (desc, rows) in bench_inputs(n).items():
            out.append(f"""/// {desc}
#[test]
#[inline(never)]
fn bench_exp{n}__{deg}() {{
    let a = black_box(mat{n}x{n}({int_rows(rows)}));
    let e = black_box(true);
    let x = a.exp();
    assert!((x.m11 == x.m11) == e);
}}""")
    return "\n\n".join(out)


def exact_nilpotent(n: int) -> str:
    """`expected += N^k / k!` for the nilpotent superdiagonal `N` (entries (i, i + k))."""
    fact = 1
    lines = []
    for k in range(1, n):
        fact *= k
        for i in range(n - k):
            lines.append(f"expected.{fld(n, n, i, i + k)} = fx({(1 << 32) // fact});")
    return " ".join(lines)


def bench_inputs(n: int) -> dict:
    """Deterministic inputs hitting each branch: `|a|₁` = 0.1 (degree 3), 0.8 (degree 5), 2 (degree
    7, no squaring) and 8 (degree 7, two squarings)."""
    def scaled(norm):
        base = [[((i * 7 + j * 3) % 5 - 2) / 2 for j in range(n)] for i in range(n)]
        for i in range(n):
            base[i][i] += 0.25
        col = max(sum(abs(base[i][j]) for i in range(n)) for j in range(n))
        return [[int(x / col * norm * (1 << 32)) for x in row] for row in base]
    return {
        "pade3": ("Degree 3 (`|a|₁ = 0.1`).", scaled(0.1)),
        "pade5": ("Degree 5 (`|a|₁ = 0.8`).", scaled(0.8)),
        "pade7": ("Degree 7, no squaring (`|a|₁ = 2`).", scaled(2.0)),
        "pade7_squared2": ("Degree 7 of `a / 4`, two squarings (`|a|₁ = 8`).", scaled(8.0)),
    }


def pow_tests(n: int) -> str:
    M = mt(n)
    cmp = " ".join(
        f"ex = max(ex, excess(ulp_diff(got.{fld(n, n, i, j)}, e.{fld(n, n, i, j)}), oracle_tol(abs_raw(e.{fld(n, n, i, j)}), tol)));"
        for i in range(n) for j in range(n))
    ident = [[(1 << 32) if i == j else 0 for j in range(n)] for i in range(n)]
    two = [[(2 << 32) if i == j else 0 for j in range(n)] for i in range(n)]
    two7 = [[(128 << 32) if i == j else 0 for j in range(n)] for i in range(n)]
    return f"""/// `pow{n}` (oracle): every entry within the oracle tolerance; `pow_mut` agrees bit for bit.
#[test]
fn test_oracle_pow{n}() {{
    let mut cases = oracle::pow{n}_cases();
    let mut ex = 0;
    while let Some(case) = cases.pop_front() {{
        let (a, k, e, tol) = *case;
        let a = black_box(mat{n}x{n}(a));
        let k: u32 = (k / 0x100000000).try_into().unwrap();
        let got = a.pow(k);
        let e = mat{n}x{n}(e);
        {cmp}
        let mut b = a;
        b.pow_mut(k);
        assert!(b == got);
    }}
    assert!(ex == 0, "oracle tolerance exceeded by {{}}", ex);
}}

/// Exact cases: `a⁰ = I`, `a¹ = a`, `(2 I)⁷ = 128 I`.
#[test]
fn test_pow{n}_exact_cases() {{
    let (a, _, _, _) = *oracle::pow{n}_cases().at(0);
    let a = black_box(mat{n}x{n}(a));
    assert!(a.pow(0) == mat{n}x{n}({int_rows(ident)}));
    assert!(a.pow(1) == a);
    let t = black_box(mat{n}x{n}({int_rows(two)}));
    assert!(t.pow(7) == mat{n}x{n}({int_rows(two7)}));
}}

#[test]
#[inline(never)]
fn bench_pow{n}__baseline() {{
    let (a, _, _, _) = *oracle::pow{n}_cases().at(0);
    let _a = black_box(mat{n}x{n}(a));
    let _k = black_box(5_u32);
    let e = black_box(true);
    assert!(e == e);
}}

/// `a⁵`: three products (bits 101: `x`, `x²`, `x⁴`, `self = x · x⁴`).
#[test]
#[inline(never)]
fn bench_pow{n}__by_squaring() {{
    let (a, _, _, _) = *oracle::pow{n}_cases().at(0);
    let a = black_box(mat{n}x{n}(a));
    let k = black_box(5_u32);
    let e = black_box(true);
    let p = a.pow(k);
    assert!((p.{fld(n, n, 0, 0)} == p.{fld(n, n, 0, 0)}) == e);
}}"""


def test_file(what: str, n: int, body: str, traits: list[str]) -> str:
    utils = [u for u in ("abs_raw", "excess", "fx", "oracle_tol", "ulp_diff") if f"{u}(" in body]
    builders = [b for b in (f"mat{n}x{n}", f"max_ulp_{n}x{n}") if f"{b}(" in body]
    return f"""{HEADER}//! `Matrix{n}::{what}` through the public API (WP 8.5-P17): oracle vectors (`tools/oracle` suite
//! `matrix_functions`), exact cases, gas benchmarks.

use core::cmp::max;
use nalgebra::linalg::{{{', '.join(traits)}}};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{{{', '.join(utils)}}};
use crate::builders::{{{', '.join(builders)}}};
use crate::oracle_matrix_functions as oracle;

{body}
"""


def outputs() -> dict[str, str]:
    lin = "crates/nalgebra/src/linalg/"
    out = {lin + "exp.cairo": render_exp(), lin + "pow.cairo": render_pow()}
    base = f"crates/{PKG}/"
    out[base + "Scarb.toml"] = TEST_MANIFEST.format(
        name=PKG, features='"exp"',
        description="Tests and gas benchmarks of the matrix exponential and integer power "
                    "(WP 8.5-P17; not published).")
    out[base + "src/builders.cairo"] = render_builders({(n, n) for n in DIMS}, set())
    mods = ["builders", "oracle_matrix_functions"]
    for n in DIMS:
        mods.append(f"exp{n}")
        out[base + f"src/exp{n}.cairo"] = test_file("exp", n, exp_tests(n), [f"Matrix{n}ExpTrait"])
        mods.append(f"pow{n}")
        out[base + f"src/pow{n}.cairo"] = test_file("pow / pow_mut", n, pow_tests(n),
                                                     [f"Matrix{n}PowTrait"])
    ops = ",".join([f"exp{n}{s}" for n in DIMS for s in ([""] + (["_skew"] if n > 1 else [])
                                                         + ["_scaled"])]
                   + [f"pow{n}" for n in DIMS])
    out[base + "src/lib.cairo"] = (
        HEADER + f"//! Package `nalgebra_{PKG}` (WP 8.5-P17): tests and gas benchmarks of the matrix\n"
        "//! exponential and the integer power of the static squares, through the public API.\n"
        "//! Oracle vectors: `tools/oracle` suite `matrix_functions` (`oracle emit-cairo\n"
        f"//! matrix_functions --from vectors --max-per-dist {MAX_PER_DIST} --ops {ops}\n"
        "//! --out src/oracle_matrix_functions.cairo`).\n\n"
        + "".join(f"#[cfg(test)]\nmod {m};\n" for m in sorted(mods)))
    return out
