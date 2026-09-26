"""WP 8.5-P15 part of the `linalg` generator (imported by `generate.py`, which owns the driver and
the `--check` mode): the permutation sequences `Perm1` / `Perm5`, `FullPivLU` (`FullPivLuRxC`) and
`ColPivQR` (`ColPivQrRxC`) on the 36 static shapes, `LBLT` (`Lblt1..6`), their test packages.

Everything numeric is documented in the emitted doc comments.
"""

from __future__ import annotations

from generate import (BOUNDS, HEADER, divn, fld, fused, struct_lit, tmod, tname, use_shape,
                      vec_lit)

DIMS = range(1, 7)
# Bunch-Kaufman's `(1 + sqrt(17)) / 8` in Q32.32: `floor(alpha * 2^32)` (exactly representable
# as the ratio below, so `Real::from_ratio` returns it without rounding).
ALPHA_RAW = 2750446389


def shp(r: int, c: int) -> str:
    return f"{r}" if r == c else f"{r}x{c}"


def fname(r: int, c: int) -> str:
    return f"FullPivLu{shp(r, c)}"


def cname(r: int, c: int) -> str:
    return f"ColPivQr{shp(r, c)}"


def v(i: int, j: int) -> str:
    """Working variable of the (0-based) entry (i, j)."""
    return f"a{i}{j}"


def fsum(terms) -> str:
    """`fused` with the degenerate cases written out: `[]` is zero, a lone addend is itself, a lone
    product is one floored multiplication (the negation, exact, goes on its first factor)."""
    if not terms:
        return "R::zero()"
    if len(terms) == 1:
        s, a, b = terms[0]
        if b is None:
            return a if s > 0 else f"-{a}"
        return f"{a} * {b}" if s > 0 else f"(-{a}) * {b}"
    return fused(terms)


def let_div(names: list[str], xs: list[str], d: str) -> str:
    call = divn(xs, d)
    if len(names) == 1:
        return f"let {names[0]} = {call};"
    return f"let ({', '.join(names)}) = {call};"


def swaps(pairs) -> str:
    return " ".join(f"let t = {x}; {x} = {y}; {y} = t;" for x, y in pairs)


def chain(var: str, cands, body) -> str:
    """`if var == c1 { body(c1) } else if ...` over the 0-based candidates (compared 1-based)."""
    arms = [f"if {var} == {c + 1} {{ {body(c)} }}" for c in cands]
    return " else ".join(arms)


def bounds_impl(name: str, trait: str) -> str:
    return f"#[generate_trait]\npub impl {name}<\n{BOUNDS}\n> of {trait}<T> {{"


# --- Perm1 / Perm5 -----------------------------------------------------------------------------


def permute_body(n: int, r: int, c: int, rows: bool, inverse: bool) -> str:
    """`tools/shapegen/linalg_p14.py`'s `permute_body`, for `Perm1` / `Perm5`."""
    lines = [f"let mut a{i}{j} = rhs.{fld(r, c, i, j)};" for j in range(c) for i in range(r)]
    steps = list(range(n - 1))
    if inverse:
        steps.reverse()
    for k in steps:
        arms = []
        for t in range(k + 1, n):
            sw = []
            for q in range(c if rows else r):
                a, b = (f"a{k}{q}", f"a{t}{q}") if rows else (f"a{q}{k}", f"a{q}{t}")
                sw.append(f"let tmp = {a};\n{a} = {b};\n{b} = tmp;")
            arms.append(f"if self.p{k + 1} == {t + 1} {{\n" + "\n".join(sw) + "\n}")
        lines.append(" else ".join(arms))
    if n == 1:
        return "let _ = self;\nlet _ = rhs;"
    lines.append("rhs = " + struct_lit(r, c, lambda i, j: f"a{i}{j}") + ";")
    return "\n".join(lines)


def perm_struct(n: int) -> str:
    if n == 1:
        return """/// The row permutation of a 1x1 factorisation: no transposition at all (upstream
/// `PermutationSequence<U1>`, whose only possible transposition is the identity). The sequences of
/// the non-square decompositions use the permutation of their rows / columns (see
/// `linalg::full_piv_lu`).
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm1 {}

/// Test-only equality (upstream `PermutationSequence` has no `PartialEq`).
#[cfg(test)]
impl Perm1PartialEq of PartialEq<Perm1> {
    fn eq(lhs: @Perm1, rhs: @Perm1) -> bool {
        true
    }
}
"""
    fields = "\n".join(f"    /// Row swapped with row {k} at step {k}, in `{k}..={n}`.\n    pub p{k}: u8,"
                       for k in range(1, n))
    cond = " && ".join(f"lhs.p{k} == rhs.p{k}" for k in range(1, n))
    return f"""/// The row permutation of a {n}x{n} factorisation: the transpositions of steps 1 to {n - 1}.
///
/// `pK` is the 1-based index of the row swapped with row `K` at step `K` (`pK == K` = no swap), so
/// `P = T{n - 1} * .. * T1`. Upstream: a `PermutationSequence` (see `Perm2`).
#[derive(Copy, Drop, Serde, Debug)]
pub struct Perm{n} {{
{fields}
}}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`).
#[cfg(test)]
impl Perm{n}PartialEq of PartialEq<Perm{n}> {{
    fn eq(lhs: @Perm{n}, rhs: @Perm{n}) -> bool {{
        {cond}
    }}
}}
"""


def perm_methods(n: int) -> str:
    P = f"Perm{n}"
    ident = f"{P} {{}}" if n == 1 else f"{P} {{ {', '.join(f'p{k}: {k}' for k in range(1, n))} }}"
    if n == 1:
        append = """        if i != i2 {
            // Any transposition of a 1x1 sequence has an index out of bounds.
            core::panic_with_felt252(INDEX_OUT_OF_BOUNDS);
        }
        let _ = self;"""
        length = "        let _ = self;\n        0"
        empty = "        let _ = self;\n        true"
        det = "        let _ = self;\n        R::one()"
    else:
        last = " else ".join(f"if self.p{k} != {k} {{\n                {k}\n            }}"
                             for k in reversed(range(1, n))) + " else {\n                0\n            }"
        arms = "\n".join(f"                {k - 1} => self.p{k} = v," for k in range(1, n))
        append = f"""        if i != i2 {{
            let (lo, hi) = if i < i2 {{
                (i, i2)
            }} else {{
                (i2, i)
            }};
            assert(hi < {n}, INDEX_OUT_OF_BOUNDS);
            let last: usize = {last};
            assert(lo >= last, PERMUTATION_ORDER);
            let v: u8 = (hi + 1).try_into().unwrap();
            match lo {{
{arms}
                _ => {{}},
            }}
        }}"""
        length = ("        let mut n: usize = 0;\n" + "\n".join(
            f"        if self.p{k} != {k} {{\n            n += 1;\n        }}" for k in range(1, n))
            + "\n        n")
        empty = "        " + " && ".join(f"self.p{k} == {k}" for k in range(1, n))
        det = ("        let mut odd = false;\n" + "\n".join(
            f"        if self.p{k} != {k} {{\n            odd = !odd;\n        }}" for k in range(1, n))
            + "\n        if odd {\n            -R::one()\n        } else {\n            R::one()\n        }")
    return f"""/// Methods of `{P}` (upstream `PermutationSequence<U{n}>`). The row / column permutations
/// are the generic `PermuteRows` / `PermuteColumns` (`linalg/permutation_sequence.cairo`).
#[generate_trait]
pub impl {P}Impl of {P}Trait {{
    /// The identity permutation (no swap). Upstream: `PermutationSequence::identity`.
    #[inline(always)]
    fn identity() -> {P} {{
        {ident}
    }}

    /// Records the transposition of the rows (or columns) `i` and `i2` (0-based) after those
    /// already recorded; `i == i2` records nothing. Same contract as `Perm2Trait::append_permutation`
    /// (one transposition per step, in step order): panics with `nalgebra: permutation order`
    /// otherwise, and with `nalgebra: index out of bounds` when an index is `>= {n}`. Upstream:
    /// `PermutationSequence::append_permutation`.
    fn append_permutation(ref self: {P}, i: usize, i2: usize) {{
{append}
    }}

    /// The number of transpositions actually recorded. Upstream: `PermutationSequence::len`.
    fn len(self: {P}) -> usize {{
{length}
    }}

    /// Whether no transposition is recorded. Upstream: `PermutationSequence::is_empty`.
    #[inline(always)]
    fn is_empty(self: {P}) -> bool {{
{empty}
    }}

    /// `1` for an even number of transpositions, `-1` for an odd one. Exact. Upstream:
    /// `PermutationSequence::determinant`.
    fn determinant<T, impl R: Real<T>, +Neg<T>, +Drop<T>>(self: {P}) -> T {{
{det}
    }}
}}
"""


def render_perm15() -> str:
    out = [HEADER + """//! The permutation sequences of the sizes 1 and 5 (WP 8.5-P15): `Perm1` and `Perm5`, the
//! companions of `Perm2/3/4/6` (`linalg::lu`) that the full-pivot LU and column-pivot QR of every
//! shape need (upstream `nalgebra::linalg::PermutationSequence<U1 / U5>`), with their
//! `PermuteRows` / `PermuteColumns` impls on every shape with 1 / 5 rows or columns. Moves only:
//! exact.
"""]
    uses = {"use simba::scalar::Real;", "use crate::base::errors::INDEX_OUT_OF_BOUNDS;",
            "use crate::base::errors::PERMUTATION_ORDER;",
            "use super::super::permutation_sequence::{PermuteColumns, PermuteRows};"}
    impls = []
    for n in (1, 5):
        for c in DIMS:
            uses.add(use_shape(n, c))
            M = tname(n, c)
            impls.append(
                f"pub impl Perm{n}PermuteRows{M}<T, +Copy<T>, +Drop<T>> of "
                f"PermuteRows<Perm{n}, {M}<T>> {{\n"
                f"fn permute_rows(self: Perm{n}, ref rhs: {M}<T>) {{\n"
                f"{permute_body(n, n, c, True, False)}\n}}\n\n"
                f"fn inv_permute_rows(self: Perm{n}, ref rhs: {M}<T>) {{\n"
                f"{permute_body(n, n, c, True, True)}\n}}\n}}")
        for r in DIMS:
            uses.add(use_shape(r, n))
            M = tname(r, n)
            impls.append(
                f"pub impl Perm{n}PermuteColumns{M}<T, +Copy<T>, +Drop<T>> of "
                f"PermuteColumns<Perm{n}, {M}<T>> {{\n"
                f"fn permute_columns(self: Perm{n}, ref rhs: {M}<T>) {{\n"
                f"{permute_body(n, r, n, False, False)}\n}}\n\n"
                f"fn inv_permute_columns(self: Perm{n}, ref rhs: {M}<T>) {{\n"
                f"{permute_body(n, r, n, False, True)}\n}}\n}}")
    out.append("\n".join(sorted(uses)) + "\n")
    for n in (1, 5):
        out.append(perm_struct(n))
        out.append(perm_methods(n))
    out.extend(impls)
    return "\n".join(out) + "\n"


# --- FullPivLU ---------------------------------------------------------------------------------


def fplu_new(r: int, c: int) -> str:
    m = min(r, c)
    st = [f"let mut {v(i, j)} = matrix.{fld(r, c, i, j)};" for j in range(c) for i in range(r)]
    for i in range(m):
        nr, nc = r - i, c - i
        if nr * nc == 1:
            continue
        st.append(f"// step {i + 1}: the first largest |a_ij| of rows {i + 1}..{r}, columns "
                  f"{i + 1}..{c} (column-major scan)")
        st.append(f"let mut piv = R::abs({v(i, i)});")
        if nr > 1:
            st.append(f"let mut rp{i} = {i + 1}_u8;")
        if nc > 1:
            st.append(f"let mut cp{i} = {i + 1}_u8;")
        for j in range(i, c):
            for t in range(i, r):
                if (t, j) == (i, i):
                    continue
                upd = ["piv = x;"] + ([f"rp{i} = {t + 1};"] if nr > 1 else []) + (
                    [f"cp{i} = {j + 1};"] if nc > 1 else [])
                st.append(f"let x = R::abs({v(t, j)}); if x > piv {{ {' '.join(upd)} }}")
        body = []
        if nc > 1:
            body.append(chain(f"cp{i}", range(i + 1, c),
                              lambda b: swaps([(v(t, i), v(t, b)) for t in range(r)])))
        if nr > 1:
            body.append(chain(f"rp{i}", range(i + 1, r),
                              lambda b: swaps([(v(i, j), v(b, j)) for j in range(c)])))
            ls = [f"l{t}" for t in range(i + 1, r)]
            body.append(let_div(ls, [v(t, i) for t in range(i + 1, r)], v(i, i)))
            for t in range(i + 1, r):
                if c - i > 1:
                    body.append(f"let nl = -l{t};")
                for k in range(i + 1, c):
                    body.append(f"{v(t, k)} = R::mul_add(nl, {v(i, k)}, {v(t, k)});")
                body.append(f"{v(t, i)} = l{t};")
        st.append(f"if piv != R::zero() {{\n{chr(10).join(body)}\n}}")
    lu = struct_lit(r, c, lambda i, j: v(i, j))
    p = perm_lit(r, lambda k: f"rp{k}" if k < m and r - k > 1 else None)
    q = perm_lit(c, lambda k: f"cp{k}" if k < m and c - k > 1 else None)
    return "\n".join(st) + f"\n{fname(r, c)} {{ lu: {lu}, p: {p}, q: {q} }}"


def perm_lit(n: int, var) -> str:
    """`PermN` literal: field `p{k+1}` = `var(k)` (or the identity `k+1` when `None`)."""
    if n == 1:
        return "Perm1 {}"
    fs = []
    for k in range(n - 1):
        x = var(k)
        fs.append(f"p{k + 1}: {x if x is not None else k + 1}")
    return f"Perm{n} {{ {', '.join(fs)} }}"


def lu_inverse(n: int, src) -> tuple[list[str], dict]:
    """Statements computing `(L U)⁻¹` of a packed unit-lower / upper factorisation, `src(i, j)`
    the packed entry: forward substitution on the STATIC unit columns (their zeros vanish at
    generation time), then back substitution row by row with one prepared divisor per row."""
    st = []
    for j in range(n):
        for i in range(j + 1, n):
            terms = [(-1, src(i, j), None)] + [(-1, src(i, k), f"y{k}{j}") for k in range(j + 1, i)]
            st.append(f"let y{i}{j} = {fsum(terms)};")
    for i in reversed(range(n)):
        nums = []
        for j in range(n):
            if i < j:
                terms = []
            elif i == j:
                terms = [(1, "R::one()", None)]
            else:
                terms = [(1, f"y{i}{j}", None)]
            terms += [(-1, src(i, k), f"x{k}{j}") for k in range(i + 1, n)]
            nums.append(fsum(terms))
        st.append(let_div([f"x{i}{j}" for j in range(n)], nums, src(i, i)))
    return st, {(i, j): f"x{i}{j}" for i in range(n) for j in range(n)}


def render_full_piv_lu(r: int, c: int) -> str:
    F, M, m = fname(r, c), tname(r, c), min(r, c)
    L, U = tname(r, m), tname(m, c)
    uses = {"use simba::scalar::Real;", "use core::internal::revoke_ap_tracking;", use_shape(r, c),
            use_shape(r, m), use_shape(m, c), f"use crate::linalg::lu::{{Perm{r}, Perm{c}}};"
            if r != c else f"use crate::linalg::lu::Perm{r};"}
    l_lit = struct_lit(r, m, lambda i, j: "R::one()" if i == j else (
        "R::zero()" if i < j else f"self.lu.{fld(r, c, i, j)}"))
    u_lit = struct_lit(m, c, lambda i, j: f"self.lu.{fld(r, c, i, j)}" if j >= i else "R::zero()")
    square = ""
    if r == c:
        n = r
        uses |= {"use crate::base::solve::SolveKernel;",
                 "use crate::linalg::permutation_sequence::{PermuteColumns, PermuteRows};"}
        last = f"self.lu.{fld(n, n, n - 1, n - 1)}"
        flips = "\n".join(f"if self.{s}.p{k} != {k} {{ neg = !neg; }}"
                          for s in ("p", "q") for k in range(1, n))
        chain_d = "\n".join(f"let d = d * self.lu.{fld(n, n, i, i)};" for i in range(n - 1))
        inv_st, X = lu_inverse(n, lambda i, j: f"self.lu.{fld(n, n, i, j)}")
        x_lit = struct_lit(n, n, lambda i, j: X[(i, j)])
        square = f"""
    /// Whether the factored matrix is invertible: the LAST pivot is exactly nonzero, like upstream's
    /// `FullPivLU::is_invertible` (full pivoting makes the pivots non-increasing in magnitude: an
    /// exactly zero trailing block leaves zeros from its first pivot to the last).
    #[inline(always)]
    fn is_invertible(self: {F}<T>) -> bool {{
        {last} != R::zero()
    }}

    /// The determinant: `sign(P) sign(Q)` times the product of the {n} pivots, or exactly zero when
    /// the last pivot is (upstream's early return). The sign goes on the last pivot (an exact
    /// negation), then the chain is upstream's order — the last pivot times the others from the
    /// first — one floor per product (`Real` has no exact product of three scalars). Panics on
    /// overflow of a partial product. Upstream: `FullPivLU::determinant`.
    fn determinant(self: {F}<T>) -> T {{
        if {last} == R::zero() {{
            return R::zero();
        }}
        let mut neg = false;
        {flips}
        let d = if neg {{
            -{last}
        }} else {{
            {last}
        }};
        {chain_d}
        d
    }}

    /// Overwrites `b` (any shape with {n} rows: a vector or a matrix) with the solution `x` of
    /// `A x = b` and returns `true`, or returns `false` and leaves `b` unchanged when the matrix
    /// is not invertible (`is_invertible`; upstream may leave `b` overwritten). Upstream's steps
    /// on every column at once: `P b` (moves), `L y = P b` (unit diagonal: one floor per
    /// component), `U z = y` (one fused numerator and one correctly rounded division per
    /// component), `x = Qᵀ z` (moves). Panics on overflow. Upstream: `FullPivLU::solve_mut`.
    fn solve_mut<B, impl P: PermuteRows<Perm{n}, B>, impl K: SolveKernel<{M}<T>, B>, +Drop<B>>(
        self: {F}<T>, ref b: B,
    ) -> bool {{
        if !Self::is_invertible(self) {{
            return false;
        }}
        P::permute_rows(self.p, ref b);
        b = K::upper(self.lu, K::lower_unit(self.lu, b));
        P::inv_permute_rows(self.q, ref b);
        true
    }}

    /// The solution of `A x = b` for any `b` with {n} rows, or `None` when the matrix is not
    /// invertible: `solve_mut` on a copy. Upstream: `FullPivLU::solve`.
    fn solve<B, impl P: PermuteRows<Perm{n}, B>, impl K: SolveKernel<{M}<T>, B>, +Drop<B>>(
        self: {F}<T>, b: B,
    ) -> Option<B> {{
        let mut x = b;
        if Self::solve_mut(self, ref x) {{
            Some(x)
        }} else {{
            None
        }}
    }}

    /// The inverse, or `None` when the matrix is not invertible. Upstream solves against the
    /// identity (`FullPivLU::try_inverse`); here `(L U)⁻¹` is computed on the STATIC unit columns
    /// (their zeros vanish at generation time: bit-identical to the general solve, fewer products),
    /// one prepared divisor per row of the back substitution, then the columns are permuted by `P`
    /// and the rows by `Qᵀ` (moves). Two roundings per entry of the back substitution. Panics on
    /// overflow. Upstream: `FullPivLU::try_inverse`.
    fn try_inverse(self: {F}<T>) -> Option<{M}<T>> {{
        revoke_ap_tracking();
        if !Self::is_invertible(self) {{
            return None;
        }}
        {chr(10).join(inv_st)}
        let mut x = {x_lit};
        PermuteColumns::inv_permute_columns(self.p, ref x);
        PermuteRows::inv_permute_rows(self.q, ref x);
        Some(x)
    }}
"""
    shape_note = "" if r == c else (
        f"\n///\n/// Non-square: `m = min({r}, {c}) = {m}` elimination steps; `L` is {r}x{m}, `U` "
        f"{m}x{c}.")
    return f"""{HEADER}//! `{F}`: the LU factorisation with full pivoting of a `{M}` (upstream
//! `nalgebra::linalg::FullPivLU<T, U{r}, U{c}>`), fully unrolled (WP 8.5-P15).

{chr(10).join(sorted(uses))}

/// The LU factorisation with full (row and column) pivoting of a `{M}<T>`: `P A Q = L U`.
///
/// `lu` packs both factors like upstream (strict lower triangle = `L` without its unit diagonal,
/// upper triangle = `U`); `p` holds the row transpositions and `q` the column transpositions, as
/// the compact sequences `Perm{r}` / `Perm{c}` — the permutations of the rows / columns of the
/// matrix, where upstream's `PermutationSequence<DimMinimum<R, C>>` holds the same `min(R, C)`
/// transpositions (a step that swaps nothing records the identity, as upstream's `append` of
/// `(i, i)`).{shape_note} Built by `{F}Trait::new` or `{M}FullPivLuTrait::full_piv_lu`.
/// Upstream: `nalgebra::linalg::FullPivLU`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {F}<T> {{
    /// `L` (strict lower triangle) and `U` (upper triangle) packed in one matrix.
    pub lu: {M}<T>,
    /// The row transpositions (`P`).
    pub p: Perm{r},
    /// The column transpositions (`Q`).
    pub q: Perm{c},
}}

/// Methods of `{F}<T>` for any `Real` scalar.
{bounds_impl(F + "Impl", F + "Trait")}
    /// The LU factorisation of `matrix` with full pivoting, fully unrolled. Upstream:
    /// `matrix.full_piv_lu()` / `FullPivLU::new(matrix)`.
    ///
    /// Always succeeds, like upstream. At step `k` the pivot is the FIRST entry of largest `|a_ij|`
    /// of the trailing block in column-major order (upstream's `icamax_full`: strict `>`, so ties go
    /// to the lowest column, then the lowest row); its column is swapped with column `k` and its
    /// row with row `k` (whole rows and columns: upstream's `swap_columns` + `gauss_step_swap`).
    /// A trailing block that is exactly zero ends the elimination (upstream's `break`): nothing is
    /// swapped nor recorded from there on. The multipliers `l_ik = a_ik / a_kk` are correctly
    /// rounded quotients by one prepared divisor (upstream multiplies by a rounded `1 / a_kk`, as in
    /// `gauss_step`), and every update is ONE `Real::mul_add` (one floor per entry per step).
    /// Panics on overflow of an update.
    fn new(matrix: {M}<T>) -> {F}<T> {{
        revoke_ap_tracking();
        {fplu_new(r, c)}
    }}

    /// The packed factors as stored (strict lower triangle = `L`, upper triangle = `U`). Exact.
    /// Upstream: `FullPivLU::lu_internal` (`#[doc(hidden)]`).
    #[inline(always)]
    fn lu_internal(self: {F}<T>) -> {M}<T> {{
        self.lu
    }}

    /// The unit lower triangular factor `L` ({r}x{m}). Exact: moves. Upstream: `FullPivLU::l`.
    #[inline(always)]
    fn l(self: {F}<T>) -> {L}<T> {{
        {l_lit}
    }}

    /// The upper triangular factor `U` ({m}x{c}). Exact: moves. Upstream: `FullPivLU::u`.
    #[inline(always)]
    fn u(self: {F}<T>) -> {U}<T> {{
        {u_lit}
    }}

    /// The row permutation `P`. Upstream: `FullPivLU::p`.
    #[inline(always)]
    fn p(self: {F}<T>) -> Perm{r} {{
        self.p
    }}

    /// The column permutation `Q`. Upstream: `FullPivLU::q`.
    #[inline(always)]
    fn q(self: {F}<T>) -> Perm{c} {{
        self.q
    }}

    /// `(P, L, U, Q)`: `(p(), l(), u(), q())`, exact. Upstream: `FullPivLU::unpack`.
    #[inline(always)]
    fn unpack(self: {F}<T>) -> (Perm{r}, {L}<T>, {U}<T>, Perm{c}) {{
        (self.p, Self::l(self), Self::u(self), self.q)
    }}
{square}}}

/// `{M}` methods that go through the full-pivot LU factorisation. Import `{M}FullPivLuTrait`.
{bounds_impl(M + "FullPivLuImpl", M + "FullPivLuTrait")}
    /// The LU factorisation with full pivoting. Upstream: `Matrix::full_piv_lu`.
    #[inline(always)]
    fn full_piv_lu(self: {M}<T>) -> {F}<T> {{
        {F}Trait::new(self)
    }}
}}
"""


# --- ColPivQR ----------------------------------------------------------------------------------


def cpqr_new(r: int, c: int) -> str:
    m = min(r, c)
    st = [f"let mut {v(i, j)} = matrix.{fld(r, c, i, j)};" for j in range(c) for i in range(r)]
    for i in range(m):
        nr, nc = r - i, c - i
        st.append(f"// step {i + 1}")
        if nc > 1:
            st.append(f"let mut piv = R::abs({v(i, i)});")
            st.append(f"let mut cp{i} = {i + 1}_u8;")
            for j in range(i, c):
                for t in range(i, r):
                    if (t, j) == (i, i):
                        continue
                    if j == i:
                        # a larger entry of the pivot column keeps the column: only `piv` moves
                        st.append(f"let x = R::abs({v(t, j)}); if x > piv {{ piv = x; }}")
                    else:
                        st.append(f"let x = R::abs({v(t, j)}); if x > piv {{ piv = x; cp{i} = {j + 1}; }}")
            st.append(chain(f"cp{i}", range(i + 1, c),
                            lambda b: swaps([(v(t, i), v(t, b)) for t in range(r)])))
        if nr == 1:
            # One component: the axis is `sign(x0)`, the reflection is `x -> -x` on this row,
            # times upstream's sign `signum(-x0)`: the row is negated iff `x0 < 0`.
            neg_right = " ".join(f"{v(i, j)} = -{v(i, j)};" for j in range(i + 1, c))
            st.append(f"let d{i} = -{v(i, i)};")
            st.append(f"if {v(i, i)} < R::zero() {{ {v(i, i)} = -R::one(); {neg_right} }} "
                      f"else if {v(i, i)} > R::zero() {{ {v(i, i)} = R::one(); }}")
            continue
        col = [v(t, i) for t in range(i, r)]
        nrm = norm_expr(col)
        body = [f"let neg = {v(i, i)} < R::zero();",
                "let signed = if neg { -nrm } else { nrm };",
                f"let y0 = {v(i, i)} + signed;",
                f"let dd = {norm_expr(['y0'] + col[1:])};",
                let_div([f"u{t}" for t in range(i, r)], ["y0"] + col[1:], "dd")]
        body += [f"{v(t, i)} = u{t};" for t in range(i, r)]
        body.append(f"d{i} = -signed;")
        for j in range(i + 1, c):
            h = fsum([(1, v(t, i), v(t, j)) for t in range(i, r)])
            body.append(f"let h = {h}; let w = h + h; let nw = -w;")
            for t in range(i, r):
                body.append(f"{v(t, j)} = if neg {{ R::mul_add(nw, {v(t, i)}, {v(t, j)}) }} "
                            f"else {{ R::mul_add(w, {v(t, i)}, -{v(t, j)}) }};")
        st.append(f"let nrm = {nrm};")
        st.append(f"let mut d{i} = R::zero();")
        st.append(f"if nrm != R::zero() {{\n{chr(10).join(body)}\n}}")
    qr = struct_lit(r, c, lambda i, j: v(i, j))
    p = perm_lit(c, lambda k: f"cp{k}" if k < m and c - k > 1 else None)
    diag = vec_lit(m, lambda i: f"d{i}")
    return "\n".join(st) + f"\n{cname(r, c)} {{ col_piv_qr: {qr}, p: {p}, diag: {diag} }}"


def norm_expr(xs: list[str]) -> str:
    n = len(xs)
    if n == 1:
        return f"R::abs({xs[0]})"
    if n <= 4:
        return f"R::norm{n}({', '.join(xs)})"
    return fused([(1, a, a) for a in xs], rescale="R::wide_sqrt")


def q_code(r: int, c: int, ncols: int) -> tuple[list[str], dict]:
    """The Householder product applied to the first `ncols` columns of the identity (upstream's
    `ColPivQR::q` loop, reflections in REVERSE order), on symbolic entries: the static zeros and
    ones of the identity vanish at generation time."""
    m = min(r, c)
    E = {(t, j): ("1" if t == j else "0") for t in range(r) for j in range(ncols)}
    st = []
    for i in reversed(range(m)):
        u = {t: f"self.col_piv_qr.{fld(r, c, t, i)}" for t in range(i, r)}
        st.append(f"let s{i} = self.diag.{fld(m, 1, i, 0)} < R::zero();")
        for j in range(i, ncols):
            terms = []
            for t in range(i, r):
                e = E[(t, j)]
                if e == "0":
                    continue
                terms.append((1, u[t], None) if e == "1" else (1, u[t], e))
            if not terms:
                continue
            st.append(f"let h = {fsum(terms)}; let w = h + h; let nw = -w;")
            for t in range(i, r):
                e = E[(t, j)]
                name = f"q{t}{j}_{i}"
                if e == "0":
                    neg, pos = f"w * {u[t]}", f"nw * {u[t]}"
                elif e == "1":
                    neg, pos = (f"R::mul_add(w, {u[t]}, -R::one())",
                                f"R::mul_add(nw, {u[t]}, R::one())")
                else:
                    neg, pos = f"R::mul_add(w, {u[t]}, -{e})", f"R::mul_add(nw, {u[t]}, {e})"
                st.append(f"let {name} = if s{i} {{ {neg} }} else {{ {pos} }};")
                E[(t, j)] = name
    return st, E


def sym(e: str) -> str:
    return {"0": "R::zero()", "1": "R::one()"}.get(e, e)


def render_col_piv_qr(r: int, c: int) -> str:
    C, M, m = cname(r, c), tname(r, c), min(r, c)
    Qt, Rt, D = tname(r, m), tname(m, c), tname(m, 1)
    uses = {"use simba::scalar::Real;", "use core::internal::revoke_ap_tracking;", use_shape(r, c),
            use_shape(r, m), use_shape(m, c), use_shape(m, 1), use_shape(r, r),
            f"use crate::linalg::lu::Perm{c};", "use crate::base::solve::SolveKernel;"}
    q_st, QE = q_code(r, c, m)
    q_lit = struct_lit(r, m, lambda i, j: sym(QE[(i, j)]))
    r_lit = struct_lit(m, c, lambda i, j: f"self.col_piv_qr.{fld(r, c, i, j)}" if j > i else (
        f"R::abs(self.diag.{fld(m, 1, i, 0)})" if i == j else "R::zero()"))
    internal = ""
    if r > c:
        qf_st, QF = q_code(r, c, r)
        qf_lit = struct_lit(r, r, lambda i, j: sym(QF[(i, j)]))
        internal = f"""
/// Crate-internal kernel of `{C}<T>`: the SQUARE orthogonal factor.
#[generate_trait]
pub(crate) impl {C}InternalImpl<
{BOUNDS}
> of {C}InternalTrait<T> {{
    /// The full {r}x{r} Householder product (upstream applies it reflection by reflection in
    /// `q_tr_mul`); its first {m} columns are `q()`.
    fn q_full(self: {C}<T>) -> {tname(r, r)}<T> {{
        revoke_ap_tracking();
        {chr(10).join(qf_st)}
        {qf_lit}
    }}
}}
"""
        qfull = f"{C}InternalTrait::q_full(self)"
    else:
        qfull = "Self::q(self)"
    square = ""
    if r == c:
        n = r
        uses.add("use crate::linalg::permutation_sequence::PermuteRows;")
        nz = " && ".join(f"self.diag.{fld(n, 1, i, 0)} != R::zero()" for i in range(n))
        flips = "\n".join(f"if self.p.p{k} != {k} {{ neg = !neg; }}" for k in range(1, n))
        dchain = "\n".join(f"let d = d * self.diag.{fld(n, 1, i, 0)};" for i in range(1, n))
        qt_lit = struct_lit(n, n, lambda i, j: f"q.{fld(n, n, j, i)}")
        square = f"""
    /// Whether every Householder norm (`diag`, the signed diagonal of `R`) is EXACTLY nonzero,
    /// like upstream's `ColPivQR::is_invertible`.
    #[inline(always)]
    fn is_invertible(self: {C}<T>) -> bool {{
        {nz}
    }}

    /// The determinant: `sign(P)` times the product of the SIGNED Householder norms (upstream's
    /// `diag`, not `|r_ii|`), the sign on the first factor (exact), then one floor per product.
    /// Panics on overflow of a partial product. Upstream: `ColPivQR::determinant`.
    fn determinant(self: {C}<T>) -> T {{
        let mut neg = false;
        {flips}
        let d = if neg {{
            -self.diag.{fld(n, 1, 0, 0)}
        }} else {{
            self.diag.{fld(n, 1, 0, 0)}
        }};
        {dchain}
        d
    }}

    /// Overwrites `b` (any shape with {n} rows) with the solution of `A x = b` and returns
    /// `true`, or returns `false` and leaves `b` unchanged when `R` has an exactly zero diagonal
    /// entry (upstream leaves `Qᵀ b` behind). `x = P (R⁻¹ (Qᵀ b))`: `Qᵀ b` as one fused sum per
    /// entry (upstream: the reflections one after the other), back substitution on `R` (upstream's
    /// `|diag|` on the diagonal: one fused numerator and one correctly rounded division per
    /// component), then the inverse column permutation (moves). Panics on overflow. Upstream:
    /// `ColPivQR::solve_mut`.
    fn solve_mut<B, impl P: PermuteRows<Perm{n}, B>, impl K: SolveKernel<{M}<T>, B>, +Drop<B>>(
        self: {C}<T>, ref b: B,
    ) -> bool {{
        if !Self::is_invertible(self) {{
            return false;
        }}
        b = K::upper(Self::r(self), K::tr_mul_rhs(Self::q(self), b));
        P::inv_permute_rows(self.p, ref b);
        true
    }}

    /// The solution of `A x = b` for any `b` with {n} rows, or `None` when `R` has an exactly
    /// zero diagonal entry: `solve_mut` on a copy. Upstream: `ColPivQR::solve`.
    fn solve<B, impl P: PermuteRows<Perm{n}, B>, impl K: SolveKernel<{M}<T>, B>, +Drop<B>>(
        self: {C}<T>, b: B,
    ) -> Option<B> {{
        let mut x = b;
        if Self::solve_mut(self, ref x) {{
            Some(x)
        }} else {{
            None
        }}
    }}

    /// The inverse `P R⁻¹ Qᵀ`, or `None` when `R` has an exactly zero diagonal entry: the back
    /// substitution of `Qᵀ` (moves: upstream multiplies the identity by `Qᵀ`), then the inverse
    /// row permutation. Panics on overflow. Upstream: `ColPivQR::try_inverse`.
    fn try_inverse(self: {C}<T>) -> Option<{M}<T>> {{
        if !Self::is_invertible(self) {{
            return None;
        }}
        let q = Self::q(self);
        let mut x = SolveKernel::<{M}<T>, {M}<T>>::upper(Self::r(self), {qt_lit});
        PermuteRows::inv_permute_rows(self.p, ref x);
        Some(x)
    }}
"""
    return f"""{HEADER}//! `{C}`: the QR factorisation with column pivoting of a `{M}` (upstream
//! `nalgebra::linalg::ColPivQR<T, U{r}, U{c}>`) by Householder reflections, fully unrolled
//! (WP 8.5-P15).

{chr(10).join(sorted(uses))}

/// The QR factorisation with column pivoting of a `{M}<T>`: `A P = Q R` (`P` = `p`, applied to the
/// columns).
///
/// Upstream's storage: `col_piv_qr` holds the Householder axes on and below the diagonal (unit
/// vectors, one per column) and the strict upper triangle of `R` above it, `diag` the SIGNED
/// Householder norms (`r_ii = |diag_i|`), `p` the column transpositions as the compact `Perm{c}`
/// (upstream: a `PermutationSequence<DimMinimum<R, C>>` of the same {m} transpositions). Built by
/// `{C}Trait::new` or `{M}ColPivQrTrait::col_piv_qr`. Upstream: `nalgebra::linalg::ColPivQR`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {C}<T> {{
    /// Householder axes (lower part) and strict upper triangle of `R`, packed.
    pub col_piv_qr: {M}<T>,
    /// The column transpositions.
    pub p: Perm{c},
    /// The signed Householder norms (`|diag_i|` is the diagonal of `R`).
    pub diag: {D}<T>,
}}

/// Methods of `{C}<T>` for any `Real` scalar.
{bounds_impl(C + "Impl", C + "Trait")}
    /// The QR factorisation with column pivoting of `matrix` by Householder reflections, fully
    /// unrolled. Upstream: `matrix.col_piv_qr()` / `ColPivQR::new(matrix)`.
    ///
    /// At step `k` the column holding the FIRST largest `|a_ij|` of the trailing block
    /// (column-major scan, upstream's `icamax_full`: ties to the lowest column) is swapped with
    /// column `k`; then upstream's `clear_column_unchecked`: the axis `u` of the reflection mapping
    /// the trailing part of column `k` onto `-sign(a_kk) |a| e_k` (`reflection_axis_mut`, zero
    /// column: no reflection and a zero norm), and the later columns become `s (a - 2 (u·a) u)`
    /// with `s = signum(diag_k)`.
    ///
    /// Rounding: `|a|` and the axis norm are floored square roots of exact sums of squares; the axis
    /// is normalised ONCE by one prepared divisor (upstream divides, then normalises again: the
    /// second pass only repeats the first, see `reflection_axis_mut`); `u·a` is one fused sum,
    /// doubled exactly, and each reflected entry one `Real::mul_add`. A one-component axis is
    /// `sign(a_kk)` exactly (no square root). Panics on overflow.
    fn new(matrix: {M}<T>) -> {C}<T> {{
        revoke_ap_tracking();
        {cpqr_new(r, c)}
    }}

    /// The upper trapezoidal factor `R` ({m}x{c}): the strict upper triangle as stored and
    /// `|diag|` on the diagonal. Exact: moves and absolute values. Upstream: `ColPivQR::r`.
    #[inline(always)]
    fn r(self: {C}<T>) -> {Rt}<T> {{
        {r_lit}
    }}

    /// `R`, consuming the factorisation: `r()`. Upstream: `ColPivQR::unpack_r`.
    #[inline(always)]
    fn unpack_r(self: {C}<T>) -> {Rt}<T> {{
        Self::r(self)
    }}

    /// The orthonormal factor `Q` ({r}x{m}): upstream's loop, the {m} signed reflections applied in
    /// reverse order to the identity, on symbolic entries (the static zeros and ones of the
    /// identity cost nothing): one fused dot product (doubled exactly) per column and reflection,
    /// one `Real::mul_add` per entry. Upstream: `ColPivQR::q`.
    fn q(self: {C}<T>) -> {Qt}<T> {{
        revoke_ap_tracking();
        {chr(10).join(q_st)}
        {q_lit}
    }}

    /// The column permutation. Upstream: `ColPivQR::p`.
    #[inline(always)]
    fn p(self: {C}<T>) -> Perm{c} {{
        self.p
    }}

    /// `(Q, R, P)`: `(q(), r(), p())`. Upstream: `ColPivQR::unpack`.
    #[inline(always)]
    fn unpack(self: {C}<T>) -> ({Qt}<T>, {Rt}<T>, Perm{c}) {{
        (Self::q(self), Self::r(self), self.p)
    }}

    /// The packed storage (axes and strict upper triangle of `R`). Exact. Upstream:
    /// `ColPivQR::col_piv_qr_internal` (`#[doc(hidden)]`).
    #[inline(always)]
    fn col_piv_qr_internal(self: {C}<T>) -> {M}<T> {{
        self.col_piv_qr
    }}

    /// `rhs = Qᵀ rhs` in place, `Q` the FULL {r}x{r} product of the reflections (upstream applies
    /// them one after the other to `rhs`), for any `rhs` with {r} rows: the product is formed
    /// once, then ONE fused sum per entry. Upstream: `ColPivQR::q_tr_mul`.
    fn q_tr_mul<B, impl K: SolveKernel<{tname(r, r)}<T>, B>, +Drop<B>>(self: {C}<T>, ref rhs: B) {{
        rhs = K::tr_mul_rhs({qfull}, rhs);
    }}
{square}}}
{internal}
/// `{M}` methods that go through the column-pivot QR factorisation. Import `{M}ColPivQrTrait`.
{bounds_impl(M + "ColPivQrImpl", M + "ColPivQrTrait")}
    /// The QR factorisation with column pivoting. Upstream: `Matrix::col_piv_qr`.
    #[inline(always)]
    fn col_piv_qr(self: {M}<T>) -> {C}<T> {{
        {C}Trait::new(self)
    }}
}}
"""


# --- LBLT --------------------------------------------------------------------------------------


def lo(i: int, j: int) -> str:
    """The stored (lower-triangle) variable of the symmetric entry (i, j)."""
    return v(max(i, j), min(i, j))


def interchange(n: int, k: int, target: int, piv: int, two: bool) -> str:
    """Upstream's two-sided interchange of `target` and `piv` in the lower triangle."""
    pairs = [(v(i, target), v(i, piv)) for i in range(piv + 1, n)]
    pairs += [(v(j, target), v(piv, j)) for j in range(target + 1, piv)]
    pairs.append((v(target, target), v(piv, piv)))
    if two:
        pairs.append((v(k + 1, k), v(piv, k)))
    return swaps(pairs)


def lblt_step(n: int, k: int) -> str:
    """The body of the factorisation at position `k` (a 1x1 or a 2x2 block starts here)."""
    if k == n - 1:
        return (f"if {v(k, k)} == R::zero() && zero_pivot.is_none() {{ zero_pivot = Some({k}); }}")
    st = [f"let dabs = R::abs({v(k, k)});",
          f"let mut imax = {k + 1}_u8;",
          f"let mut colmax = R::abs({v(k + 1, k)});"]
    for i in range(k + 2, n):
        st.append(f"let x = R::abs({v(i, k)}); if x > colmax {{ colmax = x; imax = {i}; }}")
    # rowmax and |a(imax, imax)| for each candidate imax
    arms = []
    for im in range(k + 1, n):
        xs = [f"R::abs({lo(im, j)})" for j in range(k, im)] + [
            f"R::abs({lo(j, im)})" for j in range(im + 1, n)]
        e = xs[0]
        for x in xs[1:]:
            e = f"R::max({e}, {x})"
        arms.append((im, f"({e}, R::abs({v(im, im)}))"))
    if len(arms) == 1:
        rm = arms[0][1]
    else:
        rm = " else ".join(f"if imax == {im} {{ {e} }}" for im, e in arms[:-1]) + (
            f" else {{ {arms[-1][1]} }}")
    decide = f"""let ac = alpha * colmax;
        if dabs < ac {{
            let (rowmax, dimax) = {rm};
            let bound = R::wide_mul_scalar(R::wide_add_prod(R::wide_zero(), alpha, colmax), R::div(colmax, rowmax));
            if dabs < bound {{
                piv = imax;
                if dimax < alpha * rowmax {{
                    two = true;
                }}
            }}
        }}"""
    one_ic = chain_u8("piv", range(k + 1, n), lambda p: interchange(n, k, k, p, False))
    upd1 = []
    ls = [f"l{i}" for i in range(k + 1, n)]
    upd1.append(let_div(ls, [v(i, k) for i in range(k + 1, n)], v(k, k)))
    for j in range(k + 1, n):
        for i in range(j, n):
            upd1.append(f"{v(i, j)} = R::mul_add(-l{i}, {v(j, k)}, {v(i, j)});")
    upd1 += [f"{v(i, k)} = l{i};" for i in range(k + 1, n)]
    one = f"""{one_ic}
            {chr(10).join(upd1)}
            pi{k} = piv; ps{k} = 1;"""
    if k + 2 <= n - 1 + 1 and k + 1 <= n - 1:
        two_ic = chain_u8("piv", range(k + 2, n), lambda p: interchange(n, k, k + 1, p, True))
        upd2 = []
        if k + 2 < n:
            upd2 += [f"let sneg = {v(k + 1, k)} < R::zero();",
                     f"let d = R::abs({v(k + 1, k)});",
                     f"let d11 = R::div({v(k + 1, k + 1)}, d);",
                     f"let d22 = R::div({v(k, k)}, d);",
                     "let scale = R::recip(d * R::mul_add(d11, d22, -R::one()));"]
            for j in range(k + 2, n):
                x, y = v(j, k), v(j, k + 1)
                upd2.append(f"let (tx, ty) = if sneg {{ (-{x}, -{y}) }} else {{ ({x}, {y}) }};")
                upd2.append(f"let w1_{j} = R::wide_mul_scalar(R::wide_sub(R::wide_add_prod(R::wide_zero(), {x}, d11), ty), scale);")
                upd2.append(f"let w2_{j} = R::wide_mul_scalar(R::wide_sub(R::wide_add_prod(R::wide_zero(), {y}, d22), tx), scale);")
            for j in range(k + 2, n):
                for i in range(j, n):
                    upd2.append(f"{v(i, j)} = {fused([(1, v(i, j), None), (-1, v(i, k), f'w1_{j}'), (-1, v(i, k + 1), f'w2_{j}')])};")
            for j in range(k + 2, n):
                upd2.append(f"{v(j, k)} = w1_{j}; {v(j, k + 1)} = w2_{j};")
        twob = f"""{two_ic}
            {chr(10).join(upd2)}
            pi{k} = piv; ps{k} = 2;
            pi{k + 1} = piv; ps{k + 1} = 2;
            skip = true;"""
        branch = f"if two {{ {twob} }} else {{ {one} }}"
    else:
        branch = one
    body = "\n".join(st) + f"""
    if dabs == R::zero() && colmax == R::zero() {{
        if zero_pivot.is_none() {{ zero_pivot = Some({k}); }}
    }} else {{
        let mut piv = {k}_u8;
        let mut two = false;
        {decide}
        {branch}
    }}"""
    return body


def chain_u8(var: str, cands, body) -> str:
    """`if var == c { body(c) } else if ...` over 0-based candidates compared as 0-based u8."""
    cands = list(cands)
    if not cands:
        return ""
    return " else ".join(f"if {var} == {c} {{ {body(c)} }}" for c in cands) + ";"


def lblt_starts(n: int, idx: bool, last: bool) -> list[str]:
    """`two{k}`: the block starting at position `k` is 2x2 (`k < n - 1`); `st{k}`: position `k`
    starts a block (runtime, from the stored pivots; `st{n-1}` only when `last`); `i{k}` the
    recorded interchange (when `idx`)."""
    st = []
    for k in range(n - 1):
        st.append(f"let ({f'i{k}' if idx else '_'}, s{k}) = self.p{k + 1}; let two{k} = s{k} == 2;")
    if n > 1 or last:
        st.append("let st0 = true;")
    for k in range(1, n if last else n - 1):
        st.append(f"let st{k} = !(st{k - 1} && two{k - 1});")
    return st


def render_lblt(n: int) -> str:
    B = f"Lblt{n}"
    M = tname(n, n)
    uses = {"use simba::scalar::Real;", "use core::internal::revoke_ap_tracking;", use_shape(n, n),
            f"use crate::linalg::lu::Perm{n};", "use crate::base::solve::SolveKernel;",
            "use crate::linalg::permutation_sequence::PermuteRows;"}
    init = [f"let mut {v(i, j)} = matrix.{fld(n, n, i, j)};" for j in range(n) for i in range(j, n)]
    pivs = [f"let mut pi{k} = {k}_u8; let mut ps{k} = 1_u8;" for k in range(n)]
    steps = []
    for k in range(n):
        body = lblt_step(n, k)
        if k == 0:
            steps.append(f"// position 1\n{body}")
        else:
            steps.append(f"// position {k + 1}\nif skip {{ skip = false; }} else {{ {body} }}")
    stored = struct_lit(n, n, lambda i, j: v(i, j) if i >= j else f"matrix.{fld(n, n, i, j)}")
    pv = ", ".join(f"p{k + 1}: (pi{k}, ps{k})" for k in range(n))
    fields = "\n".join(f"    /// The pivot of position {k + 1}: upstream's `pivots[{k}]`, `(index, block size)`.\n"
                       f"    pub p{k + 1}: (u8, u8)," for k in range(n))
    # d(): diagonal always, off-diagonal of the 2x2 blocks
    d_off = "\n".join(
        f"let o{k} = if st{k} && two{k} {{ self.matrix.{fld(n, n, k + 1, k)} }} else {{ R::zero() }};"
        for k in range(n - 1))
    d_lit = struct_lit(n, n, lambda i, j: f"self.matrix.{fld(n, n, i, i)}" if i == j else (
        f"o{min(i, j)}" if abs(i - j) == 1 else "R::zero()"))
    # determinant
    det = [f"let mut det = if two0 {{ {bdet(n, 0)} }} else {{ self.matrix.{fld(n, n, 0, 0)} }};"
           if n > 1 else f"let det = self.matrix.{fld(n, n, 0, 0)};"]
    for k in range(1, n):
        f1 = f"self.matrix.{fld(n, n, k, k)}"
        fac = f"if two{k} {{ {bdet(n, k)} }} else {{ {f1} }}" if k < n - 1 else f1
        det.append(f"if st{k} {{ det = det * ({fac}); }}")
    # factors(): the permutation P and L_std (unit lower, multipliers permuted by later swaps)
    fac_st = [f"let mut l{i}{j} = R::zero();" for j in range(n) for i in range(j + 1, n)]
    fac_st += [f"let mut q{k + 1} = {k + 1}_u8;" for k in range(n - 1)]
    for k in range(n):
        one_sw = chain_u8(f"i{k}", range(k + 1, n), lambda p, k=k: swaps(
            [(f"l{k}{j}", f"l{p}{j}") for j in range(k)])) if k < n - 1 else ""
        one_set = " ".join(f"l{i}{k} = self.matrix.{fld(n, n, i, k)};" for i in range(k + 1, n))
        one_q = f"q{k + 1} = i{k} + 1;" if k < n - 1 else ""
        one = f"{one_sw} {one_q} {one_set}"
        if k < n - 1:
            t = k + 1
            two_sw = chain_u8(f"i{k}", range(t + 1, n), lambda p, k=k, t=t: swaps(
                [(f"l{t}{j}", f"l{p}{j}") for j in range(k)]))
            two_q = f"q{t + 1} = i{k} + 1;" if t < n - 1 else ""
            two_set = " ".join(f"l{i}{k} = self.matrix.{fld(n, n, i, k)}; "
                               f"l{i}{k + 1} = self.matrix.{fld(n, n, i, k + 1)};"
                               for i in range(k + 2, n))
            body = f"if two{k} {{ {two_sw} {two_q} {two_set} }} else {{ {one} }}"
        else:
            body = one
        fac_st.append(f"if st{k} {{ {body} }}")
    l_lit = struct_lit(n, n, lambda i, j: f"l{i}{j}" if i > j else (
        "R::one()" if i == j else "R::zero()"))
    perm = perm_lit(n, lambda k: f"q{k + 1}")
    # D^-1 (block diagonal)
    dinv = [f"let mut e{i}{j} = R::zero();" for j in range(n) for i in range(j, n)]
    for k in range(n):
        one = f"e{k}{k} = R::recip(self.matrix.{fld(n, n, k, k)});"
        if k < n - 1:
            a, b, cc = (f"self.matrix.{fld(n, n, k, k)}", f"self.matrix.{fld(n, n, k + 1, k)}",
                        f"self.matrix.{fld(n, n, k + 1, k + 1)}")
            two = (f"let d = R::abs({b}); let d11 = R::div({cc}, d); let d22 = R::div({a}, d); "
                   f"let e = d * R::mul_add(d11, d22, -R::one()); "
                   f"let ms = if {b} < R::zero() {{ R::one() }} else {{ -R::one() }}; "
                   f"let (x0, x1, x2) = R::div3(d11, d22, ms, e); "
                   f"e{k}{k} = x0; e{k + 1}{k + 1} = x1; e{k + 1}{k} = x2;")
            fac = f"if two{k} {{ {two} }} else {{ {one} }}"
        else:
            fac = one
        dinv.append(f"if st{k} {{ {fac} }}")
    e_lit = struct_lit(n, n, lambda i, j: f"e{max(i, j)}{min(i, j)}")
    jlj = struct_lit(n, n, lambda i, j: f"l.{fld(n, n, n - 1 - j, n - 1 - i)}" if i > j else (
        "R::one()" if i == j else "R::zero()"))
    jperm = perm_lit(n, lambda k: str(n - k) if k < n // 2 else None)
    return f"""{HEADER}//! `{B}`: the Bunch-Kaufman `LBLᵀ` factorisation of a symmetric `{M}` (upstream
//! `nalgebra::linalg::LBLT<T, U{n}>`), fully unrolled (WP 8.5-P15).

{chr(10).join(sorted(uses))}

/// The Bunch-Kaufman factorisation `P A Pᵀ = L B Lᵀ` of a symmetric `{M}<T>` (only the LOWER
/// triangle is read): `L` unit lower triangular in the permuted basis, `B` block diagonal with 1x1
/// and 2x2 blocks. Upstream's storage: `matrix` (multipliers below the diagonal, the blocks of `B`
/// on the diagonal and the first subdiagonal, the upper triangle of the input untouched), the
/// pivots of the {n} positions (`p1` = upstream's `pivots[0]`...) and the first position of a zero
/// 1x1 block. Built by `{B}Trait::new` or `{M}LbltTrait::lblt`. Upstream: `nalgebra::linalg::LBLT`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {B}<T> {{
    /// Multipliers (strict lower part), blocks of `B` (diagonal and first subdiagonal).
    pub matrix: {M}<T>,
{fields}
    /// The first position (0-based) whose column was exactly zero, if any.
    pub zero_pivot: Option<usize>,
}}

/// Methods of `{B}<T>` for any `Real` scalar.
{bounds_impl(B + "Impl", B + "Trait")}
    /// The factorisation of the symmetric `matrix` (lower triangle read), fully unrolled: upstream's
    /// partial-pivoting Bunch-Kaufman (Algorithm A, LAPACK `?sytrf`), `alpha = (1 + √17) / 8`.
    ///
    /// At position `k`: `colmax` / `imax` = the first largest `|a_ik|` below the diagonal; a zero
    /// column records `zero_pivot`; `|a_kk| >= alpha colmax` keeps the 1x1 pivot `k`; otherwise
    /// `rowmax` (largest off-diagonal `|a|` of row / column `imax`) decides between the 1x1 pivot `k`
    /// (`|a_kk| >= alpha colmax (colmax / rowmax)`), the 1x1 pivot `imax` (`|a_imax,imax| >= alpha
    /// rowmax`) and the 2x2 pivot `(k, imax)`; the pivot is moved by upstream's two-sided
    /// interchange (moves). Decisions compare floored products (`alpha colmax (colmax / rowmax)` is
    /// one floor of the exact product of `alpha colmax` and the correctly rounded quotient), so a
    /// tie at the last bit may choose differently from upstream's `f64`.
    ///
    /// Updates: a 1x1 block divides its column (correctly rounded quotients, one prepared divisor;
    /// upstream multiplies by a rounded `1 / a_kk`) and updates the trailing entries with ONE
    /// `Real::mul_add` each; a 2x2 block uses upstream's scaled inverse (`d = |b|`, `d11`, `d22`
    /// correctly rounded, `scale = 1 / (d (d11 d22 - 1))`), its two coefficients per row are one
    /// floor each of an exact `(x d11 ∓ y) scale`, and each trailing entry is one fused sum of
    /// two products. Panics on overflow (and divides by zero only on a zero pivot that rounding
    /// made exactly zero).
    fn new(matrix: {M}<T>) -> {B}<T> {{
        revoke_ap_tracking();
        let alpha = R::from_ratio({ALPHA_RAW}, 0x100000000);
        {chr(10).join(init)}
        {chr(10).join(pivs)}
        let mut zero_pivot: Option<usize> = None;
        let mut skip = false;
        {chr(10).join(steps)}
        let _ = skip;
        let _ = alpha;
        {B} {{ matrix: {stored}, {pv}, zero_pivot }}
    }}

    /// The block diagonal matrix `B`. Exact: moves. Upstream: `LBLT::d`.
    fn d(self: {B}<T>) -> {M}<T> {{
        {chr(10).join(lblt_starts(n, False, False))}
        {d_off}
        {d_lit}
    }}

    /// The permutation-aware factor `Pᵀ L` (`A = (Pᵀ L) B (Pᵀ L)ᵀ`; not lower triangular in
    /// general). Exact: every entry is a stored multiplier, `1` or `0` (upstream's products are
    /// by `0` and `1` too), placed by the recorded interchanges. Upstream: `LBLT::l_permuted`.
    fn l_permuted(self: {B}<T>) -> {M}<T> {{
        let (p, l) = {B}InternalTrait::factors(self);
        let mut m = l;
        PermuteRows::inv_permute_rows(p, ref m);
        m
    }}

    /// Overwrites `b` (any shape with {n} rows) with the solution of `A x = b` and returns `true`,
    /// or returns `false` (and leaves `b` unchanged) when a column was exactly zero
    /// (`zero_pivot`). `x = Pᵀ L⁻ᵀ B⁻¹ L⁻¹ P b`: permutation (moves), unit lower solve, `B⁻¹ b` as
    /// one fused sum per entry (`B⁻¹` from the 1x1 reciprocals and upstream's scaled 2x2
    /// inverses), the unit upper solve as a unit lower one on the reversed order (moves), the
    /// inverse permutation. Upstream interleaves the same steps (`LBLT::solve_mut`); the rounding
    /// differs (fused sums here). Panics on overflow.
    fn solve_mut<B, impl P: PermuteRows<Perm{n}, B>, impl K: SolveKernel<{M}<T>, B>, +Drop<B>>(
        self: {B}<T>, ref b: B,
    ) -> bool {{
        if self.zero_pivot.is_some() {{
            return false;
        }}
        let (p, l) = {B}InternalTrait::factors(self);
        let j = {jperm};
        P::permute_rows(p, ref b);
        b = K::lower_unit(l, b);
        b = K::tr_mul_rhs({B}InternalTrait::d_inv(self), b);
        P::permute_rows(j, ref b);
        b = K::lower_unit({jlj}, b);
        P::permute_rows(j, ref b);
        P::inv_permute_rows(p, ref b);
        true
    }}

    /// The solution of `A x = b` for any `b` with {n} rows, or `None` when a column was exactly
    /// zero: `solve_mut` on a copy. Upstream: `LBLT::solve`.
    fn solve<B, impl P: PermuteRows<Perm{n}, B>, impl K: SolveKernel<{M}<T>, B>, +Drop<B>>(
        self: {B}<T>, b: B,
    ) -> Option<B> {{
        let mut x = b;
        if Self::solve_mut(self, ref x) {{
            Some(x)
        }} else {{
            None
        }}
    }}

    /// The determinant: the product of the 1x1 pivots and of the 2x2 block determinants (`a c -
    /// b²`, one fused floor each), in position order, one floor per product. Panics on overflow of
    /// a partial product. Upstream: `LBLT::determinant`.
    fn determinant(self: {B}<T>) -> T {{
        {chr(10).join(lblt_starts(n, False, n > 1))}
        {chr(10).join(det)}
        det
    }}
}}

/// Crate-internal kernels of `{B}<T>`: the factors in the permuted basis.
#[generate_trait]
pub(crate) impl {B}InternalImpl<
{BOUNDS}
> of {B}InternalTrait<T> {{
    /// `(P, L)`: the recorded interchanges as one `Perm{n}` (a 1x1 block at `k` swaps `k`, a 2x2
    /// block swaps `k + 1`, in increasing order) and the unit lower `L` of `P A Pᵀ = L B Lᵀ`
    /// (each column's multipliers, permuted by the later interchanges). Moves only.
    fn factors(self: {B}<T>) -> (Perm{n}, {M}<T>) {{
        revoke_ap_tracking();
        {chr(10).join(lblt_starts(n, True, True))}
        {chr(10).join(fac_st)}
        ({perm}, {l_lit})
    }}

    /// `B⁻¹`: `1 / b_kk` on the 1x1 blocks, upstream's scaled inverse `[[d11, -s], [-s, d22]] / e`
    /// on the 2x2 ones (`d = |b|`, `s = sign(b)`, `e = d (d11 d22 - 1)`: no product of two entries
    /// is ever formed). Correctly rounded quotients.
    fn d_inv(self: {B}<T>) -> {M}<T> {{
        {chr(10).join(lblt_starts(n, False, True))}
        {chr(10).join(dinv)}
        {e_lit}
    }}
}}

/// `{M}` methods that go through the `LBLᵀ` factorisation. Import `{M}LbltTrait`.
{bounds_impl(M + "LbltImpl", M + "LbltTrait")}
    /// The Bunch-Kaufman `LBLᵀ` factorisation (lower triangle read). Upstream:
    /// `SquareMatrix::lblt`.
    #[inline(always)]
    fn lblt(self: {M}<T>) -> {B}<T> {{
        {B}Trait::new(self)
    }}
}}
"""


def bdet(n: int, k: int) -> str:
    a, b, c = (f"self.matrix.{fld(n, n, k, k)}", f"self.matrix.{fld(n, n, k + 1, k)}",
               f"self.matrix.{fld(n, n, k + 1, k + 1)}")
    return f"R::diff_prod({a}, {c}, {b}, {b})"


LBLT_ROOT = HEADER + """//! The Bunch-Kaufman `LBLᵀ` factorisation of the symmetric static squares (upstream
//! `nalgebra::linalg::LBLT`, WP 8.5-P15): `Lblt1` .. `Lblt6`, one unrolled module per size,
//! behind the feature `lblt`.
//!
//! The pivot sequence is data-dependent (1x1 or 2x2 blocks): each position `k` is unrolled once
//! with both block forms, and a position that is the second of a 2x2 block is skipped at run time
//! (a flag), so the code is linear in the positions.

pub mod lblt1;
pub mod lblt2;
pub mod lblt3;
pub mod lblt4;
pub mod lblt5;
pub mod lblt6;

pub use lblt1::{Lblt1, Lblt1Trait, Matrix1LbltTrait};
pub use lblt2::{Lblt2, Lblt2Trait, Matrix2LbltTrait};
pub use lblt3::{Lblt3, Lblt3Trait, Matrix3LbltTrait};
pub use lblt4::{Lblt4, Lblt4Trait, Matrix4LbltTrait};
pub use lblt5::{Lblt5, Lblt5Trait, Matrix5LbltTrait};
pub use lblt6::{Lblt6, Lblt6Trait, Matrix6LbltTrait};
"""


def root(mod: str, names, what: str, feature: str) -> str:
    lines = [HEADER + what + f"//! Behind the feature `{feature}`.\n"]
    for r in DIMS:
        for c in DIMS:
            lines.append(f"pub mod {mod}{shp(r, c)};")
    lines.append("")
    for r in DIMS:
        for c in DIMS:
            M = tname(r, c)
            t = names(r, c)
            lines.append(f"pub use {mod}{shp(r, c)}::{{{M}{t[1]}, {t[0]}, {t[0]}Trait}};")
    return "\n".join(lines) + "\n"


def outputs() -> dict[str, str]:
    lin = "crates/nalgebra/src/linalg/"
    out = {lin + "lu/perm1_5.cairo": render_perm15()}
    out[lin + "full_piv_lu.cairo"] = root(
        "full_piv_lu", lambda r, c: (fname(r, c), "FullPivLuTrait"),
        "//! LU factorisation with full pivoting of every static shape (upstream\n"
        "//! `nalgebra::linalg::FullPivLU`, WP 8.5-P15): `FullPivLu1` .. `FullPivLu6x5`.\n",
        "full_piv_lu")
    out[lin + "col_piv_qr.cairo"] = root(
        "col_piv_qr", lambda r, c: (cname(r, c), "ColPivQrTrait"),
        "//! QR factorisation with column pivoting of every static shape (upstream\n"
        "//! `nalgebra::linalg::ColPivQR`, WP 8.5-P15): `ColPivQr1` .. `ColPivQr6x5`.\n",
        "col_piv_qr")
    out[lin + "lblt.cairo"] = LBLT_ROOT
    for r in DIMS:
        for c in DIMS:
            out[lin + f"full_piv_lu/full_piv_lu{shp(r, c)}.cairo"] = render_full_piv_lu(r, c)
            out[lin + f"col_piv_qr/col_piv_qr{shp(r, c)}.cairo"] = render_col_piv_qr(r, c)
    for n in DIMS:
        out[lin + f"lblt/lblt{n}.cairo"] = render_lblt(n)
    return out
