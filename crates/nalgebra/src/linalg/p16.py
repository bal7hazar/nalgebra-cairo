"""WP 8.5-P16 part of the `linalg` generator (imported by `generate.py`, which owns the driver and
the `--check` mode): the Householder reductions `Hessenberg` (1..6), `SymmetricTridiagonal` (1..6)
and `Bidiagonal` (every shape), the real Schur decomposition `Schur` (1..6, Francis double-shift
QR with upstream's convergence test), the general eigen decomposition `Eigen` (1..6), the
balancing functions and the Householder building blocks `clear_column_unchecked`,
`clear_row_unchecked` and `assemble_q`, plus their test packages.

Everything numeric is documented in the emitted doc comments; the design measurements (bit-faithful
Q32.32 model of the Schur iteration) are summarised in the doc of `SchurN::try_new` and in the WP
8.5-P16 report.
"""

from __future__ import annotations

from generate import (BOUNDS, HEADER, fld, fused, struct_lit, tmod, tname, use_shape,
                      vec_lit)

DIMS = range(1, 7)

# Absolute deflation threshold of the Schur iteration, in raw units of the matrix normalised by
# its largest entry (see `schur_doc`): the rounding noise floor of a Francis step in Q32.32.
SCHUR_NOISE_LOG2 = 6


def shp(r: int, c: int) -> str:
    return f"{r}" if r == c else f"{r}x{c}"


def v(i: int, j: int) -> str:
    return f"a{i}{j}"


def fsum(terms) -> str:
    """`fused` with the degenerate cases written out (see `p15.fsum`)."""
    if not terms:
        return "R::zero()"
    if len(terms) == 1:
        s, a, b = terms[0]
        if b is None:
            return a if s > 0 else f"-{a}"
        return f"{a} * {b}" if s > 0 else f"(-{a}) * {b}"
    return fused(terms)


def bounds_impl(name: str, trait: str, vis: str = "pub") -> str:
    return f"#[generate_trait]\n{vis} impl {name}<\n{BOUNDS}\n> of {trait}<T> {{"


def sym(e: str) -> str:
    return {"0": "R::zero()", "1": "R::one()"}.get(e, e)


def vtype(n: int) -> str:
    """The Cairo type of a length-`n` column vector (`()` for the empty vector)."""
    return "()" if n == 0 else f"{tname(n, 1)}<T>"


def vlit(n: int, value) -> str:
    return "()" if n == 0 else vec_lit(n, value)


# --- Householder axis kernels ------------------------------------------------------------------


def norm_expr(xs: list[str]) -> str:
    n = len(xs)
    if n == 1:
        return f"R::abs({xs[0]})"
    if n <= 4:
        return f"R::norm{n}({', '.join(xs)})"
    return fused([(1, a, a) for a in xs], rescale="R::wide_sqrt")


def div_expr(xs: list[str], d: str) -> str:
    n = len(xs)
    if n == 1:
        return f"R::div({xs[0]}, {d})"
    if n == 2:
        return f"(R::div({xs[0]}, {d}), R::div({xs[1]}, {d}))"
    return f"R::div{n}({', '.join(xs)}, {d})"


def axis_fn(n: int) -> str:
    xs = [f"x{i}" for i in range(n)]
    ys = ["y0"] + xs[1:]
    vs = [f"v{i}" for i in range(n)]
    us = [f"u{i}" for i in range(n)]
    ret_t = ", ".join(["T", "bool"] + ["T"] * n)
    params = ", ".join(f"{x}: T" for x in xs)
    if n == 1:
        body = """if x0 == R::zero() {
            return (R::zero(), false, x0);
        }
        // `y0 = 2 x0`: the axis is `sign(x0)`, exactly (both normalisations are exact).
        if x0 < R::zero() {
            (-x0, true, -R::one())
        } else {
            (-x0, true, R::one())
        }"""
    else:
        body = f"""let nrm = {norm_expr(xs)};
        if nrm == R::zero() {{
            return (R::zero(), false, {', '.join(xs)});
        }}
        let signed = if x0 < R::zero() {{
            -nrm
        }} else {{
            nrm
        }};
        let y0 = x0 + signed;
        let d = {norm_expr(ys)};
        let ({', '.join(vs)}) = {div_expr(ys, 'd')};
        let d2 = {norm_expr(vs)};
        let ({', '.join(us)}) = {div_expr(vs, 'd2')};
        (-signed, true, {', '.join(us)})"""
    return f"""    /// The Householder axis of the {n}-vector `x` (upstream `reflection_axis_mut`, see the
    /// module doc): `(norm, true, u)`, or `(0, false, x)` when `x` is exactly zero.
    fn axis{n}({params}) -> ({ret_t}) {{
        {body}
    }}
"""


def render_kernels() -> str:
    fns = "\n".join(axis_fn(n) for n in DIMS)
    return f"""{HEADER}//! Crate-private kernels of the WP 8.5-P16 Householder reductions (`Hessenberg`,
//! `SymmetricTridiagonal`, `Bidiagonal`, `Schur` and the building blocks of
//! `linalg::householder_steps`): the Householder axis of a vector of length 1..6, upstream's
//! `nalgebra::linalg::householder::reflection_axis_mut` with its TWO normalisations.
//!
//! For `x != 0`: `norm = -sign(x0) |x|` (`sign(0) = 1`), `y = x + sign(x0) |x| e0`, then `v = y /
//! |y|` and `u = v / |v|`. Upstream divides by `sqrt(2 (|x|² + |x0| |x|))` (= `|y|` exactly) and
//! then normalises again "to make sure the vector is unit-sized": in Q32.32 that second pass is
//! not a formality. `|y|` is a floored square root with an ABSOLUTE error of one ulp, so a small
//! `y` (a converged bulge of the Schur iteration: a few thousand ulp) gives a first quotient
//! whose norm is off by a relative `1 / |y|`, and a reflection `I - 2 u uᵀ` built on it is not
//! orthogonal: measured on the fixed-point model of `Schur`, the backward error then DOUBLES at
//! every iteration. The second pass works on a vector of norm ~1, where one ulp is relative: the
//! reflections stay orthogonal within a few ulp (the P14a / P15 `reflection_axis_mut` /
//! `ColPivQR` keep one pass: their axes are never small relative to the matrix).
//!
//! Rounding: every norm is ONE floored square root of an exact sum of squares (`Real::normN` /
//! the wide accumulator), every quotient correctly rounded with one prepared divisor per pass. A
//! one-component axis is `sign(x0)`, exactly. Panics on overflow (`|x0| + |x|` must fit).

use simba::scalar::Real;

/// The Householder axis kernels (methods of a generic impl, not free functions: AGENTS.md).
{bounds_impl("HouseholderKernelImpl", "HouseholderKernelTrait", "pub(crate)")}
{fns}}}
"""


# --- symbolic reflections (static indices) -----------------------------------------------------


def axis_call(n: int, xs: list[str], norm: str, nz: str, us: list[str]) -> str:
    return (f"let ({norm}, {nz}, {', '.join(us)}) = "
            f"HouseholderKernelTrait::<T>::axis{n}({', '.join(xs)});")


def reflect_cols(A, us: list[str], rows: list[int], cols, neg: str | None) -> list[str]:
    """Left reflection `x -> s (x - 2 (u·x) u)` of the columns `cols` restricted to `rows`
    (upstream `Reflection::reflect_with_sign`; `neg` = the name of the `sign < 0` flag, or `None`
    for the unsigned `reflect`). `A[(i, j)]` = the variable of entry (i, j); in place."""
    st = []
    for j in cols:
        h = fsum([(1, us[k], A[(r, j)]) for k, r in enumerate(rows)])
        st.append(f"let h = {h}; let w = h + h; let nw = -w;")
        for k, r in enumerate(rows):
            x = A[(r, j)]
            if neg is None:
                st.append(f"{x} = R::mul_add(nw, {us[k]}, {x});")
            else:
                st.append(f"{x} = if {neg} {{ R::mul_add(w, {us[k]}, -{x}) }} "
                          f"else {{ R::mul_add(nw, {us[k]}, {x}) }};")
    return st


def reflect_rows(A, us: list[str], cols: list[int], rows, neg: str | None) -> list[str]:
    """Right reflection of the rows `rows` restricted to `cols` (upstream
    `Reflection::reflect_rows_with_sign` / `reflect_rows`): `h = row · u` per row."""
    st = []
    for i in rows:
        h = fsum([(1, A[(i, c)], us[k]) for k, c in enumerate(cols)])
        st.append(f"let h = {h}; let w = h + h; let nw = -w;")
        for k, c in enumerate(cols):
            x = A[(i, c)]
            if neg is None:
                st.append(f"{x} = R::mul_add(nw, {us[k]}, {x});")
            else:
                st.append(f"{x} = if {neg} {{ R::mul_add(w, {us[k]}, -{x}) }} "
                          f"else {{ R::mul_add(nw, {us[k]}, {x}) }};")
    return st


def sym_reflect_cols(E, us: list[str], rows: list[int], cols, sign_var: str, tag: str):
    """`reflect_with_sign` of the columns `cols` (restricted to `rows`) of a SYMBOLIC matrix `E`
    (entries "0", "1" or variable names): the static zeros and ones of an identity vanish at
    generation time. New entries get fresh names `{tag}{i}{j}`."""
    st = []
    for j in cols:
        terms = []
        for k, r in enumerate(rows):
            e = E[(r, j)]
            if e == "0":
                continue
            terms.append((1, us[k], None) if e == "1" else (1, us[k], e))
        if not terms:
            continue
        st.append(f"let h = {fsum(terms)}; let w = h + h; let nw = -w;")
        for k, r in enumerate(rows):
            e = E[(r, j)]
            name = f"{tag}{r}{j}"
            if e == "0":
                neg, pos = f"w * {us[k]}", f"nw * {us[k]}"
            elif e == "1":
                neg, pos = (f"R::mul_add(w, {us[k]}, -R::one())",
                            f"R::mul_add(nw, {us[k]}, R::one())")
            else:
                neg, pos = f"R::mul_add(w, {us[k]}, -{e})", f"R::mul_add(nw, {us[k]}, {e})"
            st.append(f"let {name} = if {sign_var} {{ {neg} }} else {{ {pos} }};")
            E[(r, j)] = name
    return st


def sym_reflect_rows(E, us: list[str], cols: list[int], rows, sign_var: str, tag: str):
    """`reflect_rows_with_sign` of the rows `rows` (restricted to `cols`) of a symbolic matrix."""
    st = []
    for i in rows:
        terms = []
        for k, c in enumerate(cols):
            e = E[(i, c)]
            if e == "0":
                continue
            terms.append((1, us[k], None) if e == "1" else (1, e, us[k]))
        if not terms:
            continue
        st.append(f"let h = {fsum(terms)}; let w = h + h; let nw = -w;")
        for k, c in enumerate(cols):
            e = E[(i, c)]
            name = f"{tag}{i}{c}"
            if e == "0":
                neg, pos = f"w * {us[k]}", f"nw * {us[k]}"
            elif e == "1":
                neg, pos = (f"R::mul_add(w, {us[k]}, -R::one())",
                            f"R::mul_add(nw, {us[k]}, R::one())")
            else:
                neg, pos = f"R::mul_add(w, {us[k]}, -{e})", f"R::mul_add(nw, {us[k]}, {e})"
            st.append(f"let {name} = if {sign_var} {{ {neg} }} else {{ {pos} }};")
            E[(i, c)] = name
    return st


def identity_sym(r: int, c: int):
    return {(i, j): ("1" if i == j else "0") for i in range(r) for j in range(c)}


def assemble_q_code(n: int, axis_of, sign_of, tag: str = "q") -> tuple[list[str], dict]:
    """Upstream `householder::assemble_q` on symbolic identity entries: the reflections `i = n-2
    .. 0` (axis `axis_of(k)` = stored entry (i + 1 + k, i), sign flag `sign_of(i)`) applied to the
    rows `i + 1..` of the columns `i..`."""
    E = identity_sym(n, n)
    st = []
    for i in reversed(range(n - 1)):
        rows = list(range(i + 1, n))
        us = [axis_of(i, k) for k in range(len(rows))]
        sv = f"sq{i}"
        st.append(f"let {sv} = {sign_of(i)} < R::zero();")
        st += sym_reflect_cols(E, us, rows, range(i, n), sv, f"{tag}{i}_")
    return st, E


# --- Hessenberg --------------------------------------------------------------------------------


def hname(n: int) -> str:
    return f"Hessenberg{n}"


def render_hessenberg(n: int) -> str:
    H, M, V = hname(n), tname(n, n), tname(n, 1)
    uses = {"use simba::scalar::Real;", use_shape(n, n), use_shape(n, 1),
            "use core::internal::revoke_ap_tracking;"}
    if n >= 2:
        uses.add(use_shape(n - 1, 1))
        uses.add("use crate::linalg::householder_kernels::HouseholderKernelTrait;")
    A = {(i, j): v(i, j) for i in range(n) for j in range(n)}
    st = [f"let mut {v(i, j)} = hess.{fld(n, n, i, j)};" for j in range(n) for i in range(n)]
    for i in range(n - 1):
        rows = list(range(i + 1, n))
        L = len(rows)
        us = [f"u{i}_{k}" for k in range(L)]
        st.append(f"// column {i}: `clear_column_unchecked(hess, {i}, 1, Some(work))`")
        st.append(axis_call(L, [A[(r, i)] for r in rows], f"s{i}", f"nz{i}", us))
        body = [f"let neg = s{i} < R::zero();"]
        body += reflect_rows(A, us, rows, range(n), "neg")
        body += reflect_cols(A, us, rows, range(i + 1, n), "neg")
        body += [f"{A[(r, i)]} = {us[k]};" for k, r in enumerate(rows)]
        st.append(f"if nz{i} {{\n" + "\n".join(body) + "\n}")
    hess_lit = struct_lit(n, n, lambda i, j: v(i, j))
    if n >= 2:
        sub = vec_lit(n - 1, lambda i: f"s{i}")
        lit = f"{H} {{ hess: {hess_lit}, subdiag: {sub} }}"
        sub_field = (f"    /// The SIGNED subdiagonal: `|subdiag_i|` is the entry (i + 1, i) of `H`.\n"
                     f"    pub subdiag: {vtype(n - 1)},\n")
    else:
        lit = f"{H} {{ hess: {hess_lit} }}"
        sub_field = ""
    q_st, QE = assemble_q_code(
        n, lambda i, k: f"self.hess.{fld(n, n, i + 1 + k, i)}",
        lambda i: f"self.subdiag.{fld(n - 1, 1, i, 0)}")
    q_lit = struct_lit(n, n, lambda i, j: sym(QE[(i, j)]))

    def h_entry(i, j):
        if i > j + 1:
            return "R::zero()"
        if i == j + 1:
            return f"R::abs(self.subdiag.{fld(n - 1, 1, j, 0)})"
        return f"self.hess.{fld(n, n, i, j)}"
    h_lit = struct_lit(n, n, h_entry)
    work_doc = ("the workspace is not read, and left unchanged (upstream leaves scratch values in it: "
                "its content is unspecified there)")
    return f"""{HEADER}//! `{H}`: the Hessenberg decomposition of a `{M}` (upstream
//! `nalgebra::linalg::Hessenberg<T, U{n}>`) by Householder reflections, fully unrolled
//! (WP 8.5-P16).

{chr(10).join(sorted(uses))}

/// The Hessenberg decomposition `A = Q H Qᵀ` of a `{M}<T>`: `H` upper Hessenberg (zero below the
/// first subdiagonal), `Q` orthogonal.
///
/// Upstream's storage: `hess` holds `H` on and above the diagonal and the Householder axes (unit
/// vectors) below the first subdiagonal, `subdiag` the SIGNED norms of the reflections (`|subdiag|`
/// is the first subdiagonal of `H`). Built by `{H}Trait::new` or `{M}HessenbergTrait::hessenberg`.
/// Upstream: `nalgebra::linalg::Hessenberg`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {H}<T> {{
    /// `H` on and above the diagonal, the Householder axes below the first subdiagonal.
    pub hess: {M}<T>,
{sub_field}}}

/// Methods of `{H}<T>` for any `Real` scalar.
{bounds_impl(H + "Impl", H + "Trait")}
    /// The Hessenberg decomposition of `hess` by Householder reflections: `new_with_workspace`
    /// on a zero workspace. Upstream: `Hessenberg::new` / `hess.hessenberg()`.
    #[inline(always)]
    fn new(hess: {M}<T>) -> {H}<T> {{
        let mut work = {vec_lit(n, lambda i: "R::zero()")};
        Self::new_with_workspace(hess, ref work)
    }}

    /// The Hessenberg decomposition of `hess` ({work_doc}). Upstream:
    /// `Hessenberg::new_with_workspace`.
    ///
    /// Column `i` = 0..{n - 2} is upstream's `clear_column_unchecked(hess, i, 1, Some(work))`: the
    /// axis `u` of the reflection that maps the entries below the diagonal onto `norm e_(i+1)`
    /// (see `linalg::householder_kernels`, two normalisations), then with `s = sign(norm)` the
    /// columns `i + 1..` of EVERY row become `s (x - 2 (x·u) uᵀ)` (the bilateral part) and the
    /// rows `i + 1..` of those columns `s (x - 2 (u·x) u)`; the axis is stored in place of the
    /// cleared entries. Rounding: one fused sum per dot product (doubled exactly), one
    /// `Real::mul_add` per updated entry. A zero column is left as it is (norm 0). Panics on
    /// overflow.
    fn new_with_workspace(hess: {M}<T>, ref work: {V}<T>) -> {H}<T> {{
        revoke_ap_tracking();
        {chr(10).join(st)}
        {lit}
    }}

    /// `(Q, H)`: `(q(), unpack_h())`. Upstream: `Hessenberg::unpack`.
    #[inline(always)]
    fn unpack(self: {H}<T>) -> ({M}<T>, {M}<T>) {{
        (Self::q(self), Self::unpack_h(self))
    }}

    /// The upper Hessenberg matrix `H`: `hess` with zeros below the first subdiagonal and
    /// `|subdiag|` on it. Exact (moves and absolute values). Upstream: `Hessenberg::unpack_h`.
    #[inline(always)]
    fn unpack_h(self: {H}<T>) -> {M}<T> {{
        {h_lit}
    }}

    /// `H`, like `unpack_h` (upstream's `h` borrows, `unpack_h` consumes: Cairo copies). Upstream:
    /// `Hessenberg::h`.
    #[inline(always)]
    fn h(self: {H}<T>) -> {M}<T> {{
        Self::unpack_h(self)
    }}

    /// The orthogonal factor `Q`: upstream's `householder::assemble_q`, the {n - 1} signed
    /// reflections applied in reverse order to the identity on symbolic entries (its static zeros
    /// and ones cost nothing): one fused dot product per column and reflection (doubled exactly),
    /// one `Real::mul_add` per entry. Upstream: `Hessenberg::q`.
    fn q(self: {H}<T>) -> {M}<T> {{
        revoke_ap_tracking();
        {chr(10).join(q_st)}
        {q_lit}
    }}

    /// The packed storage (`H` and the Householder axes). Exact. Upstream:
    /// `Hessenberg::hess_internal` (`#[doc(hidden)]`).
    #[inline(always)]
    fn hess_internal(self: {H}<T>) -> {M}<T> {{
        self.hess
    }}
}}

/// `{M}` methods that go through the Hessenberg decomposition. Import `{M}HessenbergTrait`.
{bounds_impl(M + "HessenbergImpl", M + "HessenbergTrait")}
    /// The Hessenberg decomposition. Upstream: `SquareMatrix::hessenberg`.
    #[inline(always)]
    fn hessenberg(self: {M}<T>) -> {H}<T> {{
        {H}Trait::new(self)
    }}
}}
"""


# --- SymmetricTridiagonal ----------------------------------------------------------------------


def stname(n: int) -> str:
    return f"SymmetricTridiagonal{n}"


def render_symmetric_tridiagonal(n: int) -> str:
    S, M = stname(n), tname(n, n)
    uses = {"use simba::scalar::Real;", use_shape(n, n), use_shape(n, 1),
            "use core::internal::revoke_ap_tracking;"}
    if n >= 2:
        uses.add(use_shape(n - 1, 1))
        uses.add("use crate::linalg::householder_kernels::HouseholderKernelTrait;")
    st = [f"let mut {v(i, j)} = m.{fld(n, n, i, j)};" for j in range(n) for i in range(n)]

    def low(i, j):
        return v(i, j) if i >= j else v(j, i)
    for i in range(n - 1):
        rows = list(range(i + 1, n))
        L = len(rows)
        us = [f"u{i}_{k}" for k in range(L)]
        st.append(f"// column {i}")
        st.append(axis_call(L, [v(r, i) for r in rows], f"s{i}", f"nz{i}", us))
        body = []
        # p = 2 M u, M = the trailing block, read from its lower triangle (upstream `hegemv`)
        for a, ra in enumerate(rows):
            h = fsum([(1, low(ra, rb), us[b]) for b, rb in enumerate(rows)])
            body.append(f"let h = {h}; let p{a} = h + h;")
        body.append(f"let dot = {fsum([(1, us[a], f'p{a}') for a in range(L)])};")
        body.append("let ndot = -dot;")
        for a in range(L):
            body.append(f"let r{a} = R::mul_add(ndot, {us[a]}, p{a});")
        for b in range(L):
            for a in range(b, L):
                x = v(rows[a], rows[b])
                body.append(f"{x} = {fused([(1, x, None), (-1, f'r{a}', us[b]), (-1, us[a], f'r{b}')])};")
        body += [f"{v(r, i)} = {us[k]};" for k, r in enumerate(rows)]
        st.append(f"if nz{i} {{\n" + "\n".join(body) + "\n}")
    tri_lit = struct_lit(n, n, lambda i, j: v(i, j))
    if n >= 2:
        lit = f"{S} {{ tri: {tri_lit}, off_diagonal: {vec_lit(n - 1, lambda i: f's{i}')} }}"
        off_field = (f"    /// The SIGNED off-diagonal (`|off_diagonal|` is the off-diagonal of the\n"
                     f"    /// tridiagonal matrix).\n    pub off_diagonal: {vtype(n - 1)},\n")
        off_abs = vec_lit(n - 1, lambda i: f"R::abs(self.off_diagonal.{fld(n - 1, 1, i, 0)})")
    else:
        lit = f"{S} {{ tri: {tri_lit} }}"
        off_field = ""
        off_abs = "()"
    diag = vec_lit(n, lambda i: f"self.tri.{fld(n, n, i, i)}")
    q_st, QE = assemble_q_code(
        n, lambda i, k: f"self.tri.{fld(n, n, i + 1 + k, i)}",
        lambda i: f"self.off_diagonal.{fld(n - 1, 1, i, 0)}")
    q_lit = struct_lit(n, n, lambda i, j: sym(QE[(i, j)]))
    # recompose: B = Q T (T tridiagonal: diag d, off-diagonal |e|), then B Qᵀ
    rec = ["let q = Self::q(self);"]
    rec.append(f"let d = {diag};")
    if n >= 2:
        rec.append(f"let e = {off_abs};")

    def tk(k, l):
        if k == l:
            return f"d.{fld(n, 1, k, 0)}"
        if abs(k - l) == 1:
            return f"e.{fld(n - 1, 1, min(k, l), 0)}"
        return None
    for i in range(n):
        for l in range(n):
            terms = [(1, f"q.{fld(n, n, i, k)}", tk(k, l)) for k in range(n) if tk(k, l)]
            rec.append(f"let b{i}{l} = {fsum(terms)};")
    rec.append(struct_lit(n, n, lambda i, j: fsum(
        [(1, f"b{i}{l}", f"q.{fld(n, n, j, l)}") for l in range(n)])))
    return f"""{HEADER}//! `{S}`: the tridiagonalisation of a symmetric `{M}` (upstream
//! `nalgebra::linalg::SymmetricTridiagonal<T, U{n}>`) by Householder reflections, fully unrolled
//! (WP 8.5-P16).

{chr(10).join(sorted(uses))}

/// The tridiagonalisation `A = Q T Qᵀ` of a symmetric `{M}<T>`: `T` symmetric tridiagonal, `Q`
/// orthogonal.
///
/// Upstream's storage: `tri` holds the diagonal of `T` and the Householder axes below the first
/// subdiagonal of its LOWER triangle (its strict upper triangle is the input's, never read),
/// `off_diagonal` the SIGNED norms of the reflections (`|off_diagonal|` is the off-diagonal of
/// `T`). Built by `{S}Trait::new` or `{M}SymmetricTridiagonalTrait::symmetric_tridiagonalize`.
/// Upstream: `nalgebra::linalg::SymmetricTridiagonal`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {S}<T> {{
    /// The diagonal of `T` and the Householder axes (lower part); the strict upper triangle of
    /// the input.
    pub tri: {M}<T>,
{off_field}}}

/// Methods of `{S}<T>` for any `Real` scalar.
{bounds_impl(S + "Impl", S + "Trait")}
    /// The tridiagonalisation of the symmetric matrix `m`: only its LOWER triangle (diagonal
    /// included) is read, like upstream. Upstream: `SymmetricTridiagonal::new`.
    ///
    /// Column `i` = 0..{n - 2}: the axis `u` of the reflection that maps the entries below the
    /// diagonal onto `norm e_(i+1)` (`reflection_axis_mut`, see `linalg::householder_kernels`),
    /// then the trailing block `M` becomes `(I - 2 u uᵀ) M (I - 2 u uᵀ)` on its lower triangle:
    /// `p = 2 M u` (upstream `hegemv`), `dot = u·p`, and each entry `m_ab - r_a u_b - u_a r_b`
    /// with `r = p - dot u` (LAPACK's `dsytd2` form: two products per entry where upstream's three
    /// `hegerc` rank-one updates `- p uᵀ - u pᵀ + 2 dot u uᵀ` take three; equal in exact
    /// arithmetic). Rounding: one fused sum per entry of `p` (doubled exactly), per `dot`, per
    /// updated entry; one `Real::mul_add` per `r_a`. Panics on overflow.
    fn new(m: {M}<T>) -> {S}<T> {{
        revoke_ap_tracking();
        {chr(10).join(st)}
        {lit}
    }}

    /// The packed storage. Exact. Upstream: `SymmetricTridiagonal::internal_tri`
    /// (`#[doc(hidden)]`).
    #[inline(always)]
    fn internal_tri(self: {S}<T>) -> {M}<T> {{
        self.tri
    }}

    /// `(Q, diagonal, off-diagonal)`: `(q(), diagonal(), off_diagonal())`{" (the off-diagonal of a 1x1 matrix is the empty `()`)" if n == 1 else ""}.
    /// Upstream: `SymmetricTridiagonal::unpack`.
    #[inline(always)]
    fn unpack(self: {S}<T>) -> ({M}<T>, {vtype(n)}, {vtype(n - 1)}) {{
        (Self::q(self), Self::diagonal(self), Self::off_diagonal(self))
    }}

    /// `(diagonal, off-diagonal)`. Exact. Upstream: `SymmetricTridiagonal::unpack_tridiagonal`.
    #[inline(always)]
    fn unpack_tridiagonal(self: {S}<T>) -> ({vtype(n)}, {vtype(n - 1)}) {{
        (Self::diagonal(self), Self::off_diagonal(self))
    }}

    /// The diagonal of `T`. Exact. Upstream: `SymmetricTridiagonal::diagonal`.
    #[inline(always)]
    fn diagonal(self: {S}<T>) -> {vtype(n)} {{
        {diag}
    }}

    /// The off-diagonal of `T` (`|off_diagonal|`){", the empty `()` for a 1x1 matrix" if n == 1 else ""}. Exact. Upstream:
    /// `SymmetricTridiagonal::off_diagonal`.
    #[inline(always)]
    fn off_diagonal(self: {S}<T>) -> {vtype(n - 1)} {{
        {"let _ = self;" if n == 1 else ""}
        {off_abs}
    }}

    /// The orthogonal factor `Q`: upstream's `householder::assemble_q` on symbolic identity
    /// entries (see `Hessenberg{n}Trait::q`). Upstream: `SymmetricTridiagonal::q`.
    fn q(self: {S}<T>) -> {M}<T> {{
        revoke_ap_tracking();
        {chr(10).join(q_st)}
        {q_lit}
    }}

    /// The symmetric matrix `Q T Qᵀ` (`T` the tridiagonal matrix of `diagonal()` and
    /// `off_diagonal()`), like upstream's `&q * tri * q.adjoint()`: `B = Q T` (one fused sum of at
    /// most three products per entry), then `B Qᵀ` (one fused sum per entry). Upstream:
    /// `SymmetricTridiagonal::recompose`.
    fn recompose(self: {S}<T>) -> {M}<T> {{
        {chr(10).join(rec)}
    }}
}}

/// `{M}` methods that go through the symmetric tridiagonalisation. Import
/// `{M}SymmetricTridiagonalTrait`.
{bounds_impl(M + "SymmetricTridiagonalImpl", M + "SymmetricTridiagonalTrait")}
    /// The tridiagonalisation of this symmetric matrix (its lower triangle is read). Upstream:
    /// `SquareMatrix::symmetric_tridiagonalize`.
    #[inline(always)]
    fn symmetric_tridiagonalize(self: {M}<T>) -> {S}<T> {{
        {S}Trait::new(self)
    }}
}}
"""


# --- Bidiagonal --------------------------------------------------------------------------------


def bname(r: int, c: int) -> str:
    return f"Bidiagonal{shp(r, c)}"


def bid_new(r: int, c: int) -> list[str]:
    k = min(r, c)
    A = {(i, j): v(i, j) for i in range(r) for j in range(c)}
    st = [f"let mut {v(i, j)} = matrix.{fld(r, c, i, j)};" for j in range(c) for i in range(r)]

    def clear_column(icol, shift, out):
        rows = list(range(icol + shift, r))
        us = [f"uc{icol}_{t}" for t in range(len(rows))]
        st.append(f"// `{out} = clear_column_unchecked(matrix, {icol}, {shift}, None)`")
        st.append(axis_call(len(rows), [A[(t, icol)] for t in rows], out, f"nz_{out}", us))
        refl = reflect_cols(A, us, rows, range(icol + 1, c), "neg")
        body = ([f"let neg = {out} < R::zero();"] if refl else []) + refl
        body += [f"{A[(t, icol)]} = {us[q]};" for q, t in enumerate(rows)]
        st.append(f"if nz_{out} {{\n" + "\n".join(body) + "\n}")

    def clear_row(irow, shift, out):
        cols = list(range(irow + shift, c))
        us = [f"ur{irow}_{t}" for t in range(len(cols))]
        st.append(f"// `{out} = clear_row_unchecked(matrix, axis_packed, work, {irow}, {shift})`")
        st.append(axis_call(len(cols), [A[(irow, t)] for t in cols], out, f"nz_{out}", us))
        refl = reflect_rows(A, us, cols, range(irow + 1, r), "neg")
        body = ([f"let neg = {out} < R::zero();"] if refl else []) + refl
        body += [f"{A[(irow, t)]} = {us[q]};" for q, t in enumerate(cols)]
        st.append(f"if nz_{out} {{\n" + "\n".join(body) + "\n}")
    if r >= c:
        for ite in range(k - 1):
            clear_column(ite, 0, f"d{ite}")
            clear_row(ite, 1, f"e{ite}")
        clear_column(k - 1, 0, f"d{k - 1}")
    else:
        for ite in range(k - 1):
            clear_row(ite, 0, f"d{ite}")
            clear_column(ite, 1, f"e{ite}")
        clear_row(k - 1, 0, f"d{k - 1}")
    return st


def render_bidiagonal(r: int, c: int) -> str:
    B, M = bname(r, c), tname(r, c)
    k = min(r, c)
    upper = r >= c
    U, D, Vt = tname(r, k), tname(k, k), tname(k, c)
    uses = {"use simba::scalar::Real;", use_shape(r, c), use_shape(r, k), use_shape(k, k),
            use_shape(k, c), use_shape(k, 1), "use core::internal::revoke_ap_tracking;",
            "use crate::linalg::householder_kernels::HouseholderKernelTrait;"}
    if k >= 2:
        uses.add(use_shape(k - 1, 1))
    st = bid_new(r, c)
    uv = struct_lit(r, c, lambda i, j: v(i, j))
    diag = vec_lit(k, lambda i: f"d{i}")
    if k >= 2:
        lit = (f"{B} {{ uv: {uv}, diagonal: {diag}, off_diagonal: "
               f"{vec_lit(k - 1, lambda i: f'e{i}')}, upper_diagonal: {'true' if upper else 'false'} }}")
        off_field = (f"    /// The SIGNED off-diagonal (`|off_diagonal|` is the off-diagonal of `D`).\n"
                     f"    pub off_diagonal: {vtype(k - 1)},\n")
        off_abs = vec_lit(k - 1, lambda i: f"R::abs(self.off_diagonal.{fld(k - 1, 1, i, 0)})")
    else:
        lit = f"{B} {{ uv: {uv}, diagonal: {diag}, upper_diagonal: {'true' if upper else 'false'} }}"
        off_field = ""
        off_abs = "()"

    def dsig(i):
        return f"self.diagonal.{fld(k, 1, i, 0)}"

    def esig(i):
        return f"self.off_diagonal.{fld(k - 1, 1, i, 0)}"
    # u(): identity R x k, reflections i = k - shift0 - 1 .. 0 on rows i + shift0.., cols i..
    shift0, shift1 = (0, 1) if upper else (1, 0)
    E = identity_sym(r, k)
    u_st = []
    for i in reversed(range(k - shift0)):
        rows = list(range(i + shift0, r))
        us = [f"self.uv.{fld(r, c, t, i)}" for t in rows]
        sv = f"su{i}"
        u_st.append(f"let {sv} = {dsig(i) if upper else esig(i)} < R::zero();")
        u_st += sym_reflect_cols(E, us, rows, range(i, k), sv, f"u{i}_")
    u_lit = struct_lit(r, k, lambda i, j: sym(E[(i, j)]))
    F = identity_sym(k, c)
    v_st = []
    for i in reversed(range(k - shift1)):
        cols = list(range(i + shift1, c))
        us = [f"self.uv.{fld(r, c, i, t)}" for t in cols]
        sv = f"sv{i}"
        v_st.append(f"let {sv} = {esig(i) if upper else dsig(i)} < R::zero();")
        v_st += sym_reflect_rows(F, us, cols, range(i, k), sv, f"v{i}_")
    v_lit = struct_lit(k, c, lambda i, j: sym(F[(i, j)]))

    def d_entry(i, j):
        if i == j:
            return f"R::abs({dsig(i)})"
        if upper and j == i + 1:
            return f"R::abs({esig(i)})"
        if not upper and i == j + 1:
            return f"R::abs({esig(j)})"
        return "R::zero()"
    d_lit = struct_lit(k, k, d_entry)
    kind = "UPPER" if upper else "LOWER"
    if upper:
        steps_doc = (f"for `i` = 0..{k - 2}: `diagonal_i = clear_column_unchecked(m, i, 0)` (a "
                     f"reflection from the left clears column `i` below the diagonal) then "
                     f"`off_diagonal_i = clear_row_unchecked(m, i, 1)` (one from the right "
                     f"clears row `i` past the superdiagonal); last `clear_column_unchecked(m, "
                     f"{k - 1}, 0)`")
    else:
        steps_doc = (f"for `i` = 0..{k - 2}: `diagonal_i = clear_row_unchecked(m, i, 0)` then "
                     f"`off_diagonal_i = clear_column_unchecked(m, i, 1)`; last "
                     f"`clear_row_unchecked(m, {k - 1}, 0)`")
    if k == 1:
        steps_doc = ("a single " + ("`clear_column_unchecked(m, 0, 0)`" if upper
                                    else "`clear_row_unchecked(m, 0, 0)`"))
    return f"""{HEADER}//! `{B}`: the bidiagonalisation of a `{M}` (upstream
//! `nalgebra::linalg::Bidiagonal<T, U{r}, U{c}>`) by Householder reflections, fully unrolled
//! (WP 8.5-P16).

{chr(10).join(sorted(uses))}

/// The bidiagonalisation `A = U D Vᵀ` of a `{M}<T>`: `D` {k}x{k} {kind} bidiagonal (`{r} {'>=' if upper else '<'} {c}`),
/// `U` ({r}x{k}) and `Vᵀ` ({k}x{c}) with orthonormal columns / rows.
///
/// Upstream's storage: `uv` holds the Householder axes of both sides (unit vectors: the left ones
/// below the diagonal{'' if upper else ' band'}, the right ones past the {'super' if upper else ''}diagonal), `diagonal` /
/// `off_diagonal` the SIGNED norms of the reflections (their absolute values are the diagonal /
/// off-diagonal of `D`){"; there is no off-diagonal in a " + str(k) + "x" + str(k) + " `D` (upstream: an empty vector)" if k == 1 else ""}.
/// Built by `{B}Trait::new` or `{M}BidiagonalTrait::bidiagonalize`. Upstream:
/// `nalgebra::linalg::Bidiagonal`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {B}<T> {{
    /// The Householder axes of both sides, packed.
    pub uv: {M}<T>,
    /// The SIGNED diagonal (`|diagonal|` is the diagonal of `D`).
    pub diagonal: {vtype(k)},
{off_field}    /// Whether `D` is upper bidiagonal (`{str(upper).lower()}` for this shape: rows >= columns).
    pub upper_diagonal: bool,
}}

/// Methods of `{B}<T>` for any `Real` scalar.
{bounds_impl(B + "Impl", B + "Trait")}
    /// The bidiagonalisation of `matrix` by Householder reflections, fully unrolled: {steps_doc}.
    /// Upstream: `Bidiagonal::new` / `matrix.bidiagonalize()`.
    ///
    /// Each step computes the axis `u` of the reflection (`reflection_axis_mut`, see
    /// `linalg::householder_kernels`), then applies `s (x - 2 (u·x) u)` with `s = sign(norm)` to the
    /// trailing columns (left reflection) or `s (x - 2 (x·u) uᵀ)` to the trailing rows (right
    /// reflection) and stores the axis in place of the cleared entries. Rounding: one fused sum
    /// per dot product (doubled exactly), one `Real::mul_add` per updated entry. A zero column /
    /// row is left as it is (norm 0). Panics on overflow.
    fn new(matrix: {M}<T>) -> {B}<T> {{
        revoke_ap_tracking();
        {chr(10).join(st)}
        {lit}
    }}

    /// Whether `D` is upper bidiagonal (rows >= columns). Upstream:
    /// `Bidiagonal::is_upper_diagonal`.
    #[inline(always)]
    fn is_upper_diagonal(self: {B}<T>) -> bool {{
        self.upper_diagonal
    }}

    /// `(U, D, Vᵀ)`: `(u(), d(), v_t())`, with `A = U D Vᵀ`. Upstream: `Bidiagonal::unpack`.
    #[inline(always)]
    fn unpack(self: {B}<T>) -> ({U}<T>, {D}<T>, {Vt}<T>) {{
        (Self::u(self), Self::d(self), Self::v_t(self))
    }}

    /// The bidiagonal factor `D` ({k}x{k}): `|diagonal|` on the diagonal, `|off_diagonal|` on the
    /// {'super' if upper else 'sub'}diagonal. Exact. Upstream: `Bidiagonal::d`.
    #[inline(always)]
    fn d(self: {B}<T>) -> {D}<T> {{
        {d_lit}
    }}

    /// The left factor `U` ({r}x{k}): upstream's loop, the signed left reflections applied in
    /// reverse order to the first {k} columns of the identity, on symbolic entries (its static zeros
    /// and ones cost nothing): one fused dot product per column and reflection (doubled exactly),
    /// one `Real::mul_add` per entry. A zero axis is a no-op, as upstream's `continue`. Upstream:
    /// `Bidiagonal::u`.
    fn u(self: {B}<T>) -> {U}<T> {{
        revoke_ap_tracking();
        {chr(10).join(u_st)}
        {u_lit}
    }}

    /// The right factor `Vᵀ` ({k}x{c}): the signed right reflections applied in reverse order to
    /// the first {k} rows of the identity (same rounding as `u`). Upstream: `Bidiagonal::v_t`.
    fn v_t(self: {B}<T>) -> {Vt}<T> {{
        revoke_ap_tracking();
        {chr(10).join(v_st)}
        {v_lit}
    }}

    /// The diagonal of `D` (`|diagonal|`). Exact. Upstream: `Bidiagonal::diagonal`.
    #[inline(always)]
    fn diagonal(self: {B}<T>) -> {vtype(k)} {{
        {vec_lit(k, lambda i: f"R::abs({dsig(i)})")}
    }}

    /// The off-diagonal of `D` (`|off_diagonal|`){", the empty `()` here" if k == 1 else ""}. Exact. Upstream:
    /// `Bidiagonal::off_diagonal`.
    #[inline(always)]
    fn off_diagonal(self: {B}<T>) -> {vtype(k - 1)} {{
        {"let _ = self;" if k == 1 else ""}
        {off_abs}
    }}

    /// The packed storage (the Householder axes). Exact. Upstream: `Bidiagonal::uv_internal`
    /// (`#[doc(hidden)]`).
    #[inline(always)]
    fn uv_internal(self: {B}<T>) -> {M}<T> {{
        self.uv
    }}
}}

/// `{M}` methods that go through the bidiagonalisation. Import `{M}BidiagonalTrait`.
{bounds_impl(M + "BidiagonalImpl", M + "BidiagonalTrait")}
    /// The bidiagonalisation. Upstream: `Matrix::bidiagonalize`.
    #[inline(always)]
    fn bidiagonalize(self: {M}<T>) -> {B}<T> {{
        {B}Trait::new(self)
    }}
}}
"""


def root(mod: str, items, what: str, feature: str) -> str:
    """A family root: `pub mod` lines and the `pub use` of every module's items."""
    lines = [HEADER + what + f"//! Behind the feature `{feature}`.\n"]
    for m, _ in items:
        lines.append(f"pub mod {m};")
    lines.append("")
    for m, names in items:
        lines.append(f"pub use {m}::{{{', '.join(sorted(names))}}};")
    return "\n".join(lines) + "\n"


def outputs() -> dict[str, str]:
    lin = "crates/nalgebra/src/linalg/"
    out = {lin + "householder_kernels.cairo": render_kernels()}
    items = []
    for n in DIMS:
        out[lin + f"hessenberg/hessenberg{n}.cairo"] = render_hessenberg(n)
        items.append((f"hessenberg{n}", [hname(n), hname(n) + "Trait",
                                         tname(n, n) + "HessenbergTrait"]))
    out[lin + "hessenberg.cairo"] = root(
        "hessenberg", items,
        "//! The Hessenberg decomposition of the static squares (upstream\n"
        "//! `nalgebra::linalg::Hessenberg`, WP 8.5-P16): `Hessenberg1` .. `Hessenberg6`.\n",
        "hessenberg")
    items = []
    for n in DIMS:
        out[lin + f"symmetric_tridiagonal/symmetric_tridiagonal{n}.cairo"] = \
            render_symmetric_tridiagonal(n)
        items.append((f"symmetric_tridiagonal{n}", [stname(n), stname(n) + "Trait",
                                                    tname(n, n) + "SymmetricTridiagonalTrait"]))
    out[lin + "symmetric_tridiagonal.cairo"] = root(
        "symmetric_tridiagonal", items,
        "//! The tridiagonalisation of the symmetric static squares (upstream\n"
        "//! `nalgebra::linalg::SymmetricTridiagonal`, WP 8.5-P16): `SymmetricTridiagonal1` ..\n"
        "//! `SymmetricTridiagonal6`.\n",
        "symmetric_tridiagonal")
    items = []
    for r in DIMS:
        for c in DIMS:
            out[lin + f"bidiagonal/bidiagonal{shp(r, c)}.cairo"] = render_bidiagonal(r, c)
            items.append((f"bidiagonal{shp(r, c)}", [bname(r, c), bname(r, c) + "Trait",
                                                     tname(r, c) + "BidiagonalTrait"]))
    out[lin + "bidiagonal.cairo"] = root(
        "bidiagonal", items,
        "//! The bidiagonalisation of every static shape (upstream `nalgebra::linalg::Bidiagonal`,\n"
        "//! WP 8.5-P16): `Bidiagonal1` .. `Bidiagonal6x5`.\n",
        "bidiagonal")
    items = []
    for n in DIMS:
        out[lin + f"schur/schur{n}.cairo"] = render_schur(n)
        items.append((f"schur{n}", [scname(n), scname(n) + "Trait", tname(n, n) + "SchurTrait"]))
    out[lin + "schur.cairo"] = root(
        "schur", items,
        "//! The real Schur decomposition of the static squares (upstream\n"
        "//! `nalgebra::linalg::Schur`, WP 8.5-P16): `Schur1` .. `Schur6`, and the eigenvalues of\n"
        "//! the squares (`eigenvalues`, `complex_eigenvalues`).\n",
        "schur")
    items = []
    for n in DIMS:
        out[lin + f"eigen/eigen{n}.cairo"] = render_eigen(n)
        items.append((f"eigen{n}", [egname(n), egname(n) + "Trait"]))
    out[lin + "eigen.cairo"] = root(
        "eigen", items,
        "//! The eigen decomposition of the static squares with real eigenvalues (upstream\n"
        "//! `nalgebra::linalg::Eigen`, WP 8.5-P16): `Eigen1` .. `Eigen6`, on `Schur1` .. `Schur6`.\n",
        "schur")
    out[lin + "balancing.cairo"] = render_balancing()
    out[lin + "householder_steps.cairo"] = render_householder_steps()
    return out


# --- Schur -------------------------------------------------------------------------------------


def scname(n: int) -> str:
    return f"Schur{n}"


def tv(i: int, j: int) -> str:
    return f"t{i}{j}"


def qv(i: int, j: int) -> str:
    return f"q{i}{j}"


def load(n: int, var: str, pre) -> list[str]:
    return [f"let mut {pre(i, j)} = {var}.{fld(n, n, i, j)};" for j in range(n) for i in range(n)]


def eig2_code(h00: str, h01: str, h10: str, h11: str) -> list[str]:
    """`d4 = 4 discr` of the 2x2 block, exact then floored once (its sign is exact)."""
    return [f"let dd = {h00} - {h11};",
            f"let d4 = R::wide_rescale(R::wide_add_prod(R::wide_add_prod(R::wide_add_prod("
            f"R::wide_add_prod(R::wide_add_prod(R::wide_zero(), dd, dd), {h10}, {h01}), {h10}, "
            f"{h01}), {h10}, {h01}), {h10}, {h01}));"]


def block_fn(n: int, s: int) -> str:
    """The 2x2 branch of upstream's loop on the rows / columns `s, s + 1` (`end = s + 1`)."""
    e = s + 1
    st = load(n, "t", tv) + load(n, "q", qv)
    st += [f"let h00 = {tv(s, s)}; let h01 = {tv(s, e)}; let h10 = {tv(e, s)}; "
           f"let h11 = {tv(e, e)};"]
    body = eig2_code("h00", "h01", "h10", "h11")
    body.append("if d4 >= R::zero() {")
    body.append("let sq = R::sqrt(d4);")
    body.append("let x1 = (dd + sq) * half; let x2 = (dd - sq) * half;")
    body.append("let x = if R::abs(x1) > R::abs(x2) { x1 } else { x2 };")
    body.append("let (rot, _) = GivensRotationTrait::new(x, h10);")
    body.append("let c = rot.c(); let sn = rot.s();")
    rot = []
    # inv_rot.rotate(t[s..s+2, s..n]): a' = a c + s b, b' = -s a + c b
    for j in range(s, n):
        a, b = tv(s, j), tv(e, j)
        rot.append(f"let a = {a}; let b = {b}; {a} = R::sum_prod2(a, c, sn, b); "
                   f"{b} = R::diff_prod(c, b, sn, a);")
    # rot.rotate_rows(t[0..e+1, s..s+2]): a' = a c + s b, b' = -s a + c b
    for i in range(0, e + 1):
        a, b = tv(i, s), tv(i, e)
        rot.append(f"let a = {a}; let b = {b}; {a} = R::sum_prod2(a, c, sn, b); "
                   f"{b} = R::diff_prod(c, b, sn, a);")
    rot.append(f"{tv(e, s)} = R::zero();")
    qrot = []
    for i in range(n):
        a, b = qv(i, s), qv(i, e)
        qrot.append(f"let a = {a}; let b = {b}; {a} = R::sum_prod2(a, c, sn, b); "
                    f"{b} = R::diff_prod(c, b, sn, a);")
    body += rot
    body.append("if compute_q {\n" + "\n".join(qrot) + "\n}")
    body.append("}")
    st.append(f"if h10 != R::zero() {{\nlet half = R::from_ratio(1, 2);\n"
              + "\n".join(body) + "\n}")
    st.append(f"t = {struct_lit(n, n, tv)};")
    st.append(f"q = {struct_lit(n, n, qv)};")
    return f"""    /// The 2x2 block at rows / columns {s}, {e} (upstream's `compute_2x2_basis` branch): when
    /// its eigenvalues are real, the Givens rotation that upper-triangulates it is applied to the
    /// rows {s}, {e} of `t` (columns {s}..), to its columns {s}, {e} (rows ..={e}) and to the
    /// columns of `q`, and the entry ({e}, {s}) is set to zero.
    fn block{s}(ref t: {tname(n, n)}<T>, ref q: {tname(n, n)}<T>, compute_q: bool) {{
        {chr(10).join(st)}
    }}
"""


def francis_fn(n: int, s: int, e: int) -> str:
    """One implicit double-shift QR step on the active window `s..=e` (`e - s >= 2`)."""
    st = ["revoke_ap_tracking();"] + load(n, "t", tv) + load(n, "q", qv)
    m = e - 1
    T = {(i, j): tv(i, j) for i in range(n) for j in range(n)}
    Q = {(i, j): qv(i, j) for i in range(n) for j in range(n)}
    st.append(f"let h11 = {tv(s, s)}; let h12 = {tv(s, s + 1)}; let h21 = {tv(s + 1, s)}; "
              f"let h22 = {tv(s + 1, s + 1)}; let h32 = {tv(s + 2, s + 1)};")
    st.append(f"let hnn = {tv(e, e)}; let hmm = {tv(m, m)}; let hnm = {tv(e, m)}; "
              f"let hmn = {tv(m, e)};")
    st.append("let tra = hnn + hmm;")
    # Scaled first column of (H - σ1)(H - σ2): see the doc of `try_new`.
    st.append("let d1 = h11 - hmm; let d2 = h11 - hnn;")
    st.append("let sc = R::abs(h21) + R::abs(d1) + R::abs(d2) + R::abs(hnm);")
    st.append("let (p, r, g) = R::div3(d1, hnm, h21, sc);")
    st.append(f"let mut ax = {fused([(1, 'p', 'd2'), (-1, 'hmn', 'r'), (1, 'h12', 'g')])};")
    st.append("let mut ay = g * (h11 + h22 - tra);")
    st.append("let mut az = g * h32;")
    for k in range(s, e - 1):
        us = ["u0", "u1", "u2"]
        rows = [k, k + 1, k + 2]
        st.append(f"// bulge at column {k}")
        st.append(axis_call(3, ["ax", "ay", "az"], "nrm", "nz", us))
        body = []
        if k > s:
            body.append(f"{tv(k, k - 1)} = nrm; {tv(k + 1, k - 1)} = R::zero(); "
                        f"{tv(k + 2, k - 1)} = R::zero();")
        else:
            body.append("let _ = nrm;")
        body += reflect_cols(T, us, rows, range(k, n), None)
        body += reflect_rows(T, us, rows, range(0, min(k + 4, e + 1)), None)
        qb = reflect_rows(Q, us, rows, range(n), None)
        body.append("if compute_q {\n" + "\n".join(qb) + "\n}")
        st.append("if nz {\n" + "\n".join(body) + "\n}")
        st.append(f"ax = {tv(k + 1, k)}; ay = {tv(k + 2, k)};")
        if k < e - 2:
            st.append(f"az = {tv(k + 3, k)};")
    st.append("let _ = az;")
    us = ["u0", "u1"]
    rows = [m, e]
    st.append(axis_call(2, ["ax", "ay"], "nrm", "nz", us))
    body = [f"{tv(m, m - 1)} = nrm; {tv(e, m - 1)} = R::zero();"]
    body += reflect_cols(T, us, rows, range(m, n), None)
    body += reflect_rows(T, us, rows, range(0, e + 1), None)
    qb = reflect_rows(Q, us, rows, range(n), None)
    body.append("if compute_q {\n" + "\n".join(qb) + "\n}")
    st.append("if nz {\n" + "\n".join(body) + "\n}")
    st.append(f"t = {struct_lit(n, n, tv)};")
    st.append(f"q = {struct_lit(n, n, qv)};")
    return f"""    /// One implicit double-shift (Francis) QR step on the active window {s}..={e}, upstream's
    /// `subdim > 2` branch: the bulge made by the first column of `(H - σ1)(H - σ2)` (`σ` the
    /// eigenvalues of the trailing 2x2 block) is chased down by 3-reflections, then a 2-reflection
    /// on the rows {m}, {e}; each reflection is applied to `t` from both sides (the whole rows /
    /// columns of the quasi-triangular form) and to the columns of `q`.
    fn francis{s}_{e}(ref t: {tname(n, n)}<T>, ref q: {tname(n, n)}<T>, compute_q: bool) {{
        {chr(10).join(st)}
    }}
"""


def small_expr(x: str, a: str, b: str) -> str:
    return f"(R::abs({x}) <= thr || R::abs({x}) <= eps * (R::abs({a}) + R::abs({b})))"


def delimit_fn(n: int, end: int) -> str:
    """Upstream's `delimit_subproblem(t, eps, end)` for a static `end`."""
    st = [f"let mut nn: usize = {end};"]
    # the bottom loop: zero the negligible subdiagonal entries from `end` up
    code = "nn = 0;"
    for k in range(1, end + 1):
        x, a, b = f"t.{fld(n, n, k, k - 1)}", f"t.{fld(n, n, k, k)}", f"t.{fld(n, n, k - 1, k - 1)}"
        code = (f"if {small_expr(x, a, b)} {{\n{x} = R::zero();\n{code}\n}} else {{\nnn = {k};\n}}")
    st.append(code)
    st.append("if nn == 0 {\nreturn (0, 0);\n}")
    # the top loop from `nn - 1` down, for each possible `nn`
    arms = []
    for nn in range(1, end + 1):
        code = "start = 0;"
        for ns in range(1, nn):
            x = f"t.{fld(n, n, ns, ns - 1)}"
            a, b = f"t.{fld(n, n, ns, ns)}", f"t.{fld(n, n, ns - 1, ns - 1)}"
            code = (f"if {small_expr(x, a, b)} {{\n{x} = R::zero();\nstart = {ns};\n}} "
                    f"else {{\n{code}\n}}")
        arms.append((nn, code))
    chain = " else ".join(f"if nn == {nn} {{\n{code}\n}}" for nn, code in arms)
    st.append("let mut start: usize = 0;")
    st.append(chain)
    st.append("(start, nn)")
    return f"""    /// Upstream's `delimit_subproblem(t, eps, {end})`: zeroes the negligible subdiagonal entries
    /// from row {end} up, and returns the active window `(start, end)` (`(0, 0)`: converged).
    fn delimit{end}(ref t: {tname(n, n)}<T>, eps: T, thr: T) -> (usize, usize) {{
        {chr(10).join(st)}
    }}
"""


def schur_doc(n: int) -> str:
    return f"""    /// The real Schur decomposition `A = Q T Qᵀ` (`T` upper quasi-triangular), or `None` when
    /// `max_niter` iterations (`0`: no limit) do not converge. Upstream: `Schur::try_new(m, eps,
    /// max_niter)` / `m.try_schur(eps, max_niter)`.
    ///
    /// Upstream's algorithm, with its data-dependent loop: `m` divided by its largest `|m_ij|`
    /// (one prepared divisor, skipped for the zero matrix), the Hessenberg decomposition
    /// (`Hessenberg{n}`), then the implicit double-shift QR iteration on the active window
    /// `start..=end` delimited by upstream's `delimit_subproblem`: a Francis step when the window
    /// has three rows or more, the 2x2 standardisation otherwise (a 2x2 block with real
    /// eigenvalues is rotated to upper triangular, a complex pair is left as a 2x2 block); one
    /// iteration per pass, `None` when the count reaches `max_niter`. `T` is scaled back by the
    /// largest entry (one floor per entry).
    ///
    /// Deflation: an entry `t_k,k-1` is negligible when `|t| <= eps (|t_kk| + |t_k-1,k-1|)`
    /// (upstream's relative test, `eps` a `Real` in raw units: `default_epsilon()` is one ulp) or
    /// `|t| <= max(eps², 2^-{SCHUR_NOISE_LOG2 * -1 + 32})` in the normalised matrix. Upstream's absolute threshold is `eps²`, "the
    /// equivalent of LAPACK's SMLNUM"; in Q32.32 `eps²` is zero (only exact zeros would deflate),
    /// and a Francis step leaves a rounding noise of a few ulp to a few thousand ulp in the
    /// entries it chases through (the angle of a reflection built on a small bulge is only known to
    /// `1 / |bulge|`): measured on a bit-faithful Q32.32 model of this code (random, symmetric,
    /// integer, non-normal, clustered and defective matrices of sizes 3 to 6), the threshold 0
    /// leaves 10 % to 60 % of the inputs iterating for more than 500 passes, 2^-26 (64 ulp)
    /// converges on all of them (median 4 / 6 / 10 / 16 passes for n = 3 / 4 / 5 / 6, a tail of a
    /// few percent above 100 where a subdiagonal entry sits in the noise), for an eigenvalue error of
    /// a few hundred ulp per unit of the largest entry. Larger thresholds (256, 1024, 4096 ulp) cut
    /// the tail but grow the error of well-conditioned inputs; smaller ones lengthen the tail.
    ///
    /// The shift vector (the first column of `(H - σ1)(H - σ2)`) is upstream's `(h11² + h12 h21
    /// - tra h11 + det, h21 (h11 + h22 - tra), h21 h32)` DIVIDED by `sc = |h21| + |h11 - hmm| +
    /// |h11 - hnn| + |hnm|` before its products (LAPACK `dlahqr`'s scaling): it is the product of
    /// two first-order small quantities near convergence, which Q32.32 floors to exactly zero
    /// (the step then does nothing, forever: measured on clustered spectra). Same direction, hence
    /// the same reflection, in exact arithmetic: `(p d2 - hmn r + h12 g, g (h11 + h22 - tra), g h32)`
    /// with `(p, r, g) = (h11 - hmm, hnm, h21) / sc` and `d2 = h11 - hnn`, one fused sum.
    ///
    /// Reflections: axes by `linalg::householder_kernels` (upstream's two normalisations), one
    /// fused dot product (doubled exactly) and one `Real::mul_add` per updated entry; the 2x2
    /// branch computes `4 discr = 4 h10 h01 + (h00 - h11)²` exactly (its sign is exact: upstream's
    /// `0.5 (h00 - h11)` then `discr` rounds twice) and the rotation from `x = (h00 - h11 ± √(4
    /// discr)) / 2` (the one of larger magnitude, as upstream) by `GivensRotation::new`.
    ///
    /// Like upstream, `max_niter = 0` never gives up: a matrix on which the unshifted-exception-free
    /// Francis iteration cycles (e.g. the cyclic permutation of size 6, on which upstream's `f64`
    /// code loops forever too) never returns. Panics on overflow."""


def render_schur(n: int) -> str:
    S, M, V = scname(n), tname(n, n), tname(n, 1)
    uses = {"use simba::scalar::Real;", use_shape(n, n), use_shape(n, 1)}
    kernel = ""
    if n == 1:
        decompose = f"""        let _ = (eps, max_niter);
        Option::Some({S} {{ q: Matrix1 {{ x: R::one() }}, t: m }})"""
        decompose_raw = ""
    elif n == 2:
        uses.add("use crate::linalg::givens::GivensRotationTrait;")
        body = eig2_code("m.m11", "m.m12", "m.m21", "m.m22")
        decompose = f"""        let _ = (eps, max_niter);
        let (q, t) = {S}KernelTrait::decompose(m);
        Option::Some({S} {{ q, t }})"""
        kernel = f"""
/// Crate-internal kernel of `{S}<T>`.
{bounds_impl(S + "KernelImpl", S + "KernelTrait", "pub(crate)")}
    /// Upstream's `decompose_2x2`: when the eigenvalues of `m` are real (`4 discr >= 0`, exact
    /// sign) and `m21 != 0`, the Givens rotation `G` that upper-triangulates it: `(G, Gᵀ m G)` with
    /// the entry (2, 1) set to zero; otherwise `(I, m)`. `G = [[c, -s], [s, c]]` is built from
    /// `GivensRotation::new(x, m21)`, `x = (m11 - m22 ± √(4 discr)) / 2` (the one of larger
    /// magnitude).
    fn decompose(m: Matrix2<T>) -> (Matrix2<T>, Matrix2<T>) {{
        if m.m21 == R::zero() {{
            return (Matrix2 {{ m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() }}, m);
        }}
        {chr(10).join(body)}
        if d4 < R::zero() {{
            return (Matrix2 {{ m11: R::one(), m21: R::zero(), m12: R::zero(), m22: R::one() }}, m);
        }}
        let half = R::from_ratio(1, 2);
        let sq = R::sqrt(d4);
        let x1 = (dd + sq) * half;
        let x2 = (dd - sq) * half;
        let x = if R::abs(x1) > R::abs(x2) {{
            x1
        }} else {{
            x2
        }};
        let (rot, _) = GivensRotationTrait::new(x, m.m21);
        let c = rot.c();
        let sn = rot.s();
        // inv_rot.rotate(m), then rot.rotate_rows(m)
        let a11 = R::sum_prod2(m.m11, c, sn, m.m21);
        let a21 = R::diff_prod(c, m.m21, sn, m.m11);
        let a12 = R::sum_prod2(m.m12, c, sn, m.m22);
        let a22 = R::diff_prod(c, m.m22, sn, m.m12);
        let b11 = R::sum_prod2(a11, c, sn, a12);
        let b12 = R::diff_prod(c, a12, sn, a11);
        let b22 = R::diff_prod(c, a22, sn, a21);
        let _ = R::sum_prod2(a21, c, sn, a22);
        (
            Matrix2 {{ m11: c, m21: sn, m12: -sn, m22: c }},
            Matrix2 {{ m11: b11, m21: R::zero(), m12: b12, m22: b22 }},
        )
    }}
}}
"""
    else:
        uses |= {"use crate::linalg::givens::GivensRotationTrait;",
                 f"use crate::linalg::hessenberg::hessenberg{n}::{hname(n)}Trait;",
                 "use crate::linalg::householder_kernels::HouseholderKernelTrait;",
                 "use core::internal::revoke_ap_tracking;"}
        fns = [delimit_fn(n, e) for e in range(1, n)]
        fns += [block_fn(n, s) for s in range(0, n - 1)]
        fns += [francis_fn(n, s, e) for e in range(2, n) for s in range(0, e - 1)]
        dl = " else ".join(f"if end == {e} {{\nSelf::delimit{e}(ref t, eps, thr)\n}}"
                           for e in range(1, n - 1)) + (
            f" else {{\nSelf::delimit{n - 1}(ref t, eps, thr)\n}}" if n > 2 else "")
        fr_arms = []
        for e in range(2, n):
            inner = " else ".join(f"if start == {s} {{\nSelf::francis{s}_{e}(ref t, ref q, compute_q);\n}}"
                                  for s in range(0, e - 1))
            fr_arms.append(f"if end == {e} {{\n{inner}\n}}")
        fr = " else ".join(fr_arms)
        bl = " else ".join(f"if start == {s} {{\nSelf::block{s}(ref t, ref q, compute_q);\n}}"
                           for s in range(0, n - 1))
        amax = "let mut amax = R::abs(m.m11);\n" + "\n".join(
            f"let x = R::abs(m.{fld(n, n, i, j)}); if x > amax {{ amax = x; }}"
            for j in range(n) for i in range(n) if (i, j) != (0, 0))
        names = [f"n{i}{j}" for j in range(n) for i in range(n)]
        srcs = [f"m.{fld(n, n, i, j)}" for j in range(n) for i in range(n)]
        divs = []
        k = 0
        while k < len(srcs):
            size = next(z for z in (16, 9, 6, 5, 4, 3, 1) if z <= len(srcs) - k)
            if size == 1:
                divs.append(f"let {names[k]} = R::div({srcs[k]}, amax);")
            else:
                divs.append(f"let ({', '.join(names[k:k + size])}) = "
                            f"R::div{size}({', '.join(srcs[k:k + size])}, amax);")
            k += size
        norm_lit = struct_lit(n, n, lambda i, j: f"n{i}{j}")
        scale_lit = struct_lit(n, n, lambda i, j: f"t.{fld(n, n, i, j)} * amax")
        kernel = f"""
/// Crate-internal kernels of `{S}<T>`: upstream's `do_decompose` and its steps, one function per
/// active window (static indices).
{bounds_impl(S + "KernelImpl", S + "KernelTrait", "pub(crate)")}
    /// Upstream's `do_decompose(m, work, eps, max_niter, compute_q)` (see `{S}Trait::try_new`);
    /// `q` is the identity when `compute_q` is false.
    fn decompose(
        m: {M}<T>, eps: T, max_niter: usize, compute_q: bool,
    ) -> Option<({M}<T>, {M}<T>)> {{
        {amax}
        let a = if amax != R::zero() {{
            {chr(10).join(divs)}
            {norm_lit}
        }} else {{
            m
        }};
        let hess = {hname(n)}Trait::new(a);
        let (mut q, mut t) = if compute_q {{
            {hname(n)}Trait::unpack(hess)
        }} else {{
            ({struct_lit(n, n, lambda i, j: "R::one()" if i == j else "R::zero()")}, {hname(n)}Trait::unpack_h(hess))
        }};
        let noise = R::from_ratio(1, {1 << (32 - SCHUR_NOISE_LOG2)});
        let e2 = eps * eps;
        let thr = if e2 > noise {{
            e2
        }} else {{
            noise
        }};
        let (mut start, mut end) = Self::delimit{n - 1}(ref t, eps, thr);
        let mut niter: usize = 0;
        let mut failed = false;
        while end != start {{
            if end - start >= 2 {{
                {fr}
            }} else {{
                {bl}
                if end > 2 {{
                    end -= 2;
                }} else {{
                    break;
                }}
            }}
            let (s, e) = {dl};
            start = s;
            end = e;
            niter += 1;
            if niter == max_niter {{
                failed = true;
                break;
            }}
        }}
        if failed {{
            return Option::None;
        }}
        Option::Some((q, {scale_lit}))
    }}

{chr(10).join(fns)}}}
"""
        decompose = f"""        match {S}KernelTrait::decompose(m, eps, max_niter, true) {{
            Option::Some((q, t)) => Option::Some({S} {{ q, t }}),
            Option::None => Option::None,
        }}"""
    # eigenvalues from t
    real_ok = " && ".join(f"self.t.{fld(n, n, i + 1, i)} == R::zero()" for i in range(n - 1))
    diag = vec_lit(n, lambda i: f"self.t.{fld(n, n, i, i)}")
    if n == 1:
        eigvals = f"""        Option::Some({diag})"""
    else:
        eigvals = f"""        if {real_ok} {{
            Option::Some({diag})
        }} else {{
            Option::None
        }}"""
    # complex eigenvalues
    ce = [f"let mut re{i} = self.t.{fld(n, n, i, i)}; let mut im{i} = R::zero();" for i in range(n)]
    if n >= 2:
        ce.append("let half = R::from_ratio(1, 2);")
        ce.append("let mut second = false;")
    for mm in range(n - 1):
        nn = mm + 1
        hmm, hnm, hmn, hnn = (f"self.t.{fld(n, n, mm, mm)}", f"self.t.{fld(n, n, nn, mm)}",
                              f"self.t.{fld(n, n, mm, nn)}", f"self.t.{fld(n, n, nn, nn)}")
        blk = eig2_code(hmm, hmn, hnm, hnn)
        blk += ["let im = if d4 < R::zero() { R::sqrt(-d4) * half } else { R::zero() };",
                f"let re = ({hmm} + {hnn}) * half;",
                f"re{mm} = re; im{mm} = im; re{nn} = re; im{nn} = -im;",
                "second = true;"]
        ce.append(f"if !second && {hnm} != R::zero() {{\n" + "\n".join(blk)
                  + "\n} else {\nsecond = false;\n}")
    if n >= 2:
        ce.append("let _ = second;")
    ce.append(f"({vec_lit(n, lambda i: f're{i}')}, {vec_lit(n, lambda i: f'im{i}')})")
    # matrix methods
    if n == 1:
        m_eig = f"Option::Some(Matrix1 {{ x: self.x }})"
        m_cplx = "(Matrix1 { x: self.x }, Matrix1 { x: R::zero() })"
    elif n == 2:
        m_eig = """let half = R::from_ratio(1, 2);
        let dd = self.m11 - self.m22;
        let d4 = R::wide_rescale(R::wide_add_prod(R::wide_add_prod(R::wide_add_prod(
            R::wide_add_prod(R::wide_add_prod(R::wide_zero(), dd, dd), self.m21, self.m12),
            self.m21, self.m12), self.m21, self.m12), self.m21, self.m12));
        if d4 < R::zero() {
            return Option::None;
        }
        let sq = R::sqrt(d4);
        let tra = self.m11 + self.m22;
        Option::Some(Vector2 { x: (tra + sq) * half, y: (tra - sq) * half })"""
        m_cplx = f"""let (q, t) = {S}KernelTrait::decompose(self);
        {S}Trait::complex_eigenvalues({S} {{ q, t }})"""
    else:
        m_eig = f"""let (q, t) = {S}KernelTrait::decompose(self, R::default_epsilon(), 0, false).unwrap();
        {S}Trait::eigenvalues({S} {{ q, t }})"""
        m_cplx = f"""let (q, t) = {S}KernelTrait::decompose(self, R::default_epsilon(), 0, false).unwrap();
        {S}Trait::complex_eigenvalues({S} {{ q, t }})"""
    m2_doc = ("upstream's special case: the closed form `(tra ± √(4 discr)) / 2` of the 2x2 "
              "eigenvalues (`4 discr = 4 m21 m12 + (m11 - m22)²` exact, sign exact), `None` when "
              "negative" if n == 2 else
              "the Schur iteration without `Q` (`compute_q` false: the `q` updates are skipped), "
              "`None` when `T` has a 2x2 block" if n >= 3 else "`Some(m)`")
    cplx_doc = ("of the 2x2 decomposition (`decompose_2x2`)" if n == 2 else
                "of the Schur iteration without `Q`" if n >= 3 else "of `m`")
    return f"""{HEADER}//! `{S}`: the real Schur decomposition of a `{M}` (upstream
//! `nalgebra::linalg::Schur<T, U{n}>`), WP 8.5-P16.

{chr(10).join(sorted(uses))}

/// The real Schur decomposition `A = Q T Qᵀ` of a `{M}<T>`: `Q` orthogonal, `T` upper
/// quasi-triangular (1x1 blocks for the real eigenvalues, 2x2 blocks with complex-conjugate
/// eigenvalues). Built by `{S}Trait::new` / `try_new` or `{M}SchurTrait::schur` / `try_schur`.
/// Upstream: `nalgebra::linalg::Schur`.
#[derive(Copy, Drop, Serde, Debug)]
pub struct {S}<T> {{
    /// The orthogonal factor.
    pub q: {M}<T>,
    /// The quasi-triangular factor.
    pub t: {M}<T>,
}}

/// Methods of `{S}<T>` for any `Real` scalar.
{bounds_impl(S + "Impl", S + "Trait")}
    /// `try_new(m, default_epsilon(), 0)`, unwrapped (`max_niter = 0`: no limit, see `try_new`).
    /// Upstream: `Schur::new` / `m.schur()`.
    #[inline(always)]
    fn new(m: {M}<T>) -> {S}<T> {{
        Self::try_new(m, R::default_epsilon(), 0).unwrap()
    }}

{schur_doc(n) if n >= 3 else f'''    /// The real Schur decomposition of `m`: {"`(Q, T) = (1, m)`, always `Some`" if n == 1 else "upstream'" + "s closed form `decompose_2x2` (see `" + S + "KernelTrait::decompose`), always `Some`"} (`eps` and
    /// `max_niter` are not used, as upstream). Upstream: `Schur::try_new`.'''}
    fn try_new(m: {M}<T>, eps: T, max_niter: usize) -> Option<{S}<T>> {{
{decompose}
    }}

    /// `(Q, T)`. Upstream: `Schur::unpack`.
    #[inline(always)]
    fn unpack(self: {S}<T>) -> ({M}<T>, {M}<T>) {{
        (self.q, self.t)
    }}

    /// The eigenvalues (the diagonal of `T`) when they are all real, i.e. every subdiagonal entry
    /// of `T` is exactly zero, `None` otherwise. Exact. Upstream: `Schur::eigenvalues`.
    fn eigenvalues(self: {S}<T>) -> Option<{V}<T>> {{
{eigvals}
    }}

    /// The complex eigenvalues as `(re, im)`: a 1x1 block `t_kk` gives `(t_kk, 0)`, a 2x2 block
    /// (nonzero subdiagonal entry) the pair `(tra / 2, ±√(-discr))` (`4 discr = 4 hnm hmn + (hmm -
    /// hnn)²` exact then floored, one floored square root, halvings floored; `0` if the block's
    /// discriminant is not negative, which a decomposition built here never gives). The Cairo form
    /// of upstream's `OVector<Complex<T>, D>` (`num-complex`'s `Complex` is not a scalar type here):
    /// two vectors, real parts and imaginary parts, in upstream's order. Upstream:
    /// `Schur::complex_eigenvalues`.
    fn complex_eigenvalues(self: {S}<T>) -> ({V}<T>, {V}<T>) {{
        {chr(10).join(ce)}
    }}
}}
{kernel}
/// `{M}` methods that go through the Schur decomposition. Import `{M}SchurTrait`.
{bounds_impl(M + "SchurImpl", M + "SchurTrait")}
    /// The real Schur decomposition. Upstream: `SquareMatrix::schur`.
    #[inline(always)]
    fn schur(self: {M}<T>) -> {S}<T> {{
        {S}Trait::new(self)
    }}

    /// The real Schur decomposition, or `None` when `max_niter` iterations (`0`: no limit) do not
    /// converge (see `{S}Trait::try_new`). Upstream: `SquareMatrix::try_schur`.
    #[inline(always)]
    fn try_schur(self: {M}<T>, eps: T, max_niter: usize) -> Option<{S}<T>> {{
        {S}Trait::try_new(self, eps, max_niter)
    }}

    /// The eigenvalues when they are all real, `None` otherwise: {m2_doc}. Upstream:
    /// `SquareMatrix::eigenvalues`.
    fn eigenvalues(self: {M}<T>) -> Option<{V}<T>> {{
        {m_eig}
    }}

    /// The complex eigenvalues `(re, im)` (see `{S}Trait::complex_eigenvalues`) {cplx_doc}.
    /// Upstream: `SquareMatrix::complex_eigenvalues`.
    fn complex_eigenvalues(self: {M}<T>) -> ({V}<T>, {V}<T>) {{
        {m_cplx}
    }}
}}
"""


# --- Eigen -------------------------------------------------------------------------------------


def egname(n: int) -> str:
    return f"Eigen{n}"


def render_eigen(n: int) -> str:
    E, M, V = egname(n), tname(n, n), tname(n, 1)
    uses = {"use simba::scalar::Real;", use_shape(n, n), use_shape(n, 1),
            f"use crate::linalg::schur::schur{n}::{scname(n)}Trait;"}
    st = ["let (q, t) = " + scname(n) + "Trait::unpack(" + scname(n) + "Trait::new(m));"]
    if n >= 2:
        st.append("if " + " || ".join(f"t.{fld(n, n, i + 1, i)} != R::zero()" for i in range(n - 1))
                  + " {\nreturn Option::None;\n}")
    st += [f"let mut t{i}{j} = t.{fld(n, n, i, j)};" for i in range(n) for j in range(i + 1, n)]
    st += [f"let mut v{i}{j} = q.{fld(n, n, i, j)};" for j in range(n) for i in range(n)]
    for j in range(1, n):
        for i in range(j):
            body = [f"let z = R::div(-t{i}{j}, diff);"]
            if j + 1 < n:
                body.append("let nz = -z;")
            body += [f"t{i}{k} = R::mul_add(nz, t{j}{k}, t{i}{k});" for k in range(j + 1, n)]
            body += [f"v{k}{j} = R::mul_add(z, v{k}{i}, v{k}{j});" for k in range(n)]
            st.append(f"let diff = t.{fld(n, n, i, i)} - t.{fld(n, n, j, j)};")
            st.append(f"if diff == R::zero() {{\nif t{i}{j} != R::zero() {{\nreturn Option::None;\n}}\n}} "
                      f"else {{\n" + "\n".join(body) + "\n}")
    if n >= 2:
        st.append("let _ = (" + ", ".join(f"t{i}{j}" for i in range(n) for j in range(i + 1, n)) + ");")
    for j in range(n):
        xs = [f"v{i}{j}" for i in range(n)]
        st.append(f"let nrm = {norm_expr(xs)};")
        if n == 1:
            st.append(f"if nrm != R::zero() {{\nv{0}{j} = R::div(v{0}{j}, nrm);\n}}")
        else:
            st.append(f"if nrm != R::zero() {{\nlet ({', '.join(f'w{i}' for i in range(n))}) = "
                      f"{div_expr(xs, 'nrm')};\n"
                      + " ".join(f"v{i}{j} = w{i};" for i in range(n)) + "\n}")
    st.append(f"Option::Some({E} {{ eigenvectors: {struct_lit(n, n, lambda i, j: f'v{i}{j}')}, "
              f"eigenvalues: {vec_lit(n, lambda i: f't.{fld(n, n, i, i)}')} }})")
    return f"""{HEADER}//! `{E}`: the eigen decomposition of a `{M}` with real eigenvalues (upstream
//! `nalgebra::linalg::Eigen<T, U{n}>`), WP 8.5-P16.

{chr(10).join(sorted(uses))}

/// The eigen decomposition `A V = V diag(eigenvalues)` of a `{M}<T>` whose eigenvalues are real
/// and simple: `eigenvectors` holds unit eigenvectors in its columns, in the order of
/// `eigenvalues`. Built by `{E}Trait::new`. Upstream: `nalgebra::linalg::Eigen` (whose module
/// upstream keeps out of its build, "not complete enough for publishing": it handles only
/// eigenvalues of multiplicity one; its `new` is ported as written).
#[derive(Copy, Drop, Serde, Debug)]
pub struct {E}<T> {{
    /// The unit eigenvectors, one per column.
    pub eigenvectors: {M}<T>,
    /// The eigenvalues, in the order of the columns of `eigenvectors`.
    pub eigenvalues: {V}<T>,
}}

/// Methods of `{E}<T>` for any `Real` scalar.
{bounds_impl(E + "Impl", E + "Trait")}
    /// The eigen decomposition of `m`, or `None` when its real Schur form `Q T Qᵀ`
    /// (`Schur{n}Trait::new`) has a 2x2 block (complex eigenvalues) or two exactly equal
    /// eigenvalues `t_ii = t_jj` with a nonzero coupling `t_ij`. Upstream: `Eigen::new`.
    ///
    /// Upstream's algorithm: for `j` = 1..{n - 1} and `i` < `j`, `z = -t_ij / (t_ii - t_jj)`
    /// (correctly rounded), the rows `t_i,k -= z t_j,k` (`k > j`) and the eigenvector columns `v_j
    /// += z v_i` (one `Real::mul_add` each), then every column normalised (its norm one floored
    /// square root of the exact sum of squares, one prepared divisor). Deviation: when `t_ii = t_jj`
    /// and `t_ij = 0`, upstream divides `0 / 0` (NaN); here the step is skipped (`z = 0`). Panics
    /// on overflow (a tiny `t_ii - t_jj` makes `z` large).
    fn new(m: {M}<T>) -> Option<{E}<T>> {{
        {chr(10).join(st)}
    }}
}}
"""


# --- balancing ---------------------------------------------------------------------------------


def balance_impl(n: int) -> str:
    M, V = tname(n, n), tname(n, 1)
    st = [f"let mut {v(i, j)} = matrix.{fld(n, n, i, j)};" for j in range(n) for i in range(n)]
    st += [f"let mut d{i} = R::one();" for i in range(n)]
    st.append("let two = R::from_int(2);")
    st.append("let half = R::from_ratio(1, 2);")
    st.append("let tol = R::from_ratio(95, 100);")
    st.append("let mut converged = false;")
    loop = ["converged = true;"]
    for i in range(n):
        col = [v(r, i) for r in range(n)]
        row = [v(i, c) for c in range(n)]
        body = [f"let c2 = {fsum([(1, x, x) for x in col])};",
                f"let r2 = {fsum([(1, x, x) for x in row])};",
                "let s = c2 + r2;",
                f"let mut n_col = {norm_expr(col) if n > 1 else f'R::abs({col[0]})'};",
                f"let mut n_row = {norm_expr(row) if n > 1 else f'R::abs({row[0]})'};",
                "if n_col != R::zero() && n_row != R::zero() {",
                "let mut f = R::one();",
                "let mut finv = R::one();",
                "while n_col < n_row * half {",
                "n_col = n_col * two; n_row = n_row * half; f = f * two; finv = finv * half;",
                "}",
                "while n_col >= n_row * two {",
                "n_col = n_col * half; n_row = n_row * two; f = f * half; finv = finv * two;",
                "}",
                f"if {fsum([(1, 'n_col', 'n_col'), (1, 'n_row', 'n_row')])} < tol * s {{",
                "converged = false;",
                f"d{i} = d{i} * f;"]
        body += [f"{v(r, i)} = {v(r, i)} * f;" for r in range(n)]
        body += [f"{v(i, c)} = {v(i, c)} * finv;" for c in range(n)]
        body += ["}", "}"]
        loop.append("{\n" + "\n".join(body) + "\n}")
    st.append("while !converged {\n" + "\n".join(loop) + "\n}")
    st.append(f"matrix = {struct_lit(n, n, lambda i, j: v(i, j))};")
    st.append(vec_lit(n, lambda i: f"d{i}"))
    ub = [f"let dinv{j} = R::recip(d.{fld(n, 1, j, 0)});" for j in range(n)]
    ub.append(f"m = {struct_lit(n, n, lambda i, j: f'm.{fld(n, n, i, j)} * (d.{fld(n, 1, i, 0)} * dinv{j})')};")
    return f"""/// `balance_parlett_reinsch` / `unbalance` on `{M}` (see the free functions).
pub impl {M}Balancing<
{BOUNDS}
> of Balancing<{M}<T>, {V}<T>> {{
    fn balance_parlett_reinsch(ref matrix: {M}<T>) -> {V}<T> {{
        {chr(10).join(st)}
    }}

    fn unbalance(ref m: {M}<T>, d: {V}<T>) {{
        {chr(10).join(ub)}
    }}
}}
"""


def render_balancing() -> str:
    uses = {"use simba::scalar::Real;"}
    for n in DIMS:
        uses.add(use_shape(n, n))
        uses.add(use_shape(n, 1))
    impls = "\n".join(balance_impl(n) for n in DIMS)
    return f"""{HEADER}//! Matrix balancing (upstream `nalgebra::linalg::balancing`, WP 8.5-P16):
//! `balance_parlett_reinsch` and `unbalance` on the static squares `Matrix1` .. `Matrix6`.

{chr(10).join(sorted(uses))}

/// The balancing of one square shape `M` with its diagonal vector `V` (the free functions below
/// are upstream's interface).
pub trait Balancing<M, V> {{
    fn balance_parlett_reinsch(ref matrix: M) -> V;
    fn unbalance(ref m: M, d: V);
}}

/// Applies in place a modified Parlett and Reinsch balancing with 2-norm to `matrix` (`matrix <-
/// D⁻¹ matrix D`) and returns the diagonal of `D`. Upstream:
/// `nalgebra::linalg::balancing::balance_parlett_reinsch` (<https://arxiv.org/pdf/1401.5766.pdf>).
///
/// Upstream's data-dependent loop: until a sweep changes nothing, for each index `i` with a
/// nonzero column and row, the column norm `c` and row norm `r` (floored square roots of the exact
/// sums of squares) are moved by factors of 2 until `r / 2 <= c < 2 r` (the factor `f` a power of
/// two), and when `c² + r² < 0.95 (|col|² + |row|²)` (`0.95` = `from_ratio(95, 100)`, the sums
/// of squares floored once each) the column `i` is multiplied by `f`, the row `i` by `1 / f`
/// (tracked as a power of two too: upstream divides by `f`) and `d_i` by `f`. Products by powers
/// of two are exact while no bit falls below one ulp (`f < 1`: floored); `r / 2` is a floored
/// product by `1 / 2`. Panics on overflow.
pub fn balance_parlett_reinsch<M, V, impl B: Balancing<M, V>>(ref matrix: M) -> V {{
    B::balance_parlett_reinsch(ref matrix)
}}

/// Computes in place `D m D⁻¹` with `D = diag(d)`: `m_ij * (d_i * (1 / d_j))`, upstream's
/// formula (`1 / d_j` correctly rounded, then two floored products: exact for the powers of two
/// `balance_parlett_reinsch` returns, while no bit falls below one ulp). Upstream:
/// `nalgebra::linalg::balancing::unbalance`.
pub fn unbalance<M, V, impl B: Balancing<M, V>>(ref m: M, d: V) {{
    B::unbalance(ref m, d)
}}

{impls}"""


# --- Householder building blocks with run-time indices -----------------------------------------


def column_major_impl(r: int, c: int) -> str:
    M = tname(r, c)
    comps = [fld(r, c, i, j) for j in range(c) for i in range(r)]
    names = [f"v{k}" for k in range(len(comps))]
    lit = struct_lit(r, c, lambda i, j: names[j * r + i])
    return f"""impl {M}ColumnMajor<T, +Copy<T>, +Drop<T>> of ColumnMajor<{M}<T>, T> {{
    #[inline(always)]
    fn nrows() -> usize {{
        {r}
    }}

    #[inline(always)]
    fn ncols() -> usize {{
        {c}
    }}

    fn to_column_major(self: {M}<T>) -> Array<T> {{
        array![{', '.join(f'self.{f}' for f in comps)}]
    }}

    fn from_column_major(data: Span<T>) -> {M}<T> {{
        let boxed: @Box<[T; {len(comps)}]> = data.try_into().expect(SLICE_LENGTH);
        let [{', '.join(names)}] = boxed.unbox();
        {lit}
    }}
}}
"""


STEPS_BODY = """
/// Run-time column-major access to a static shape: the interface of the building blocks below
/// (their indices are run-time values, like upstream's).
pub trait ColumnMajor<M, T> {
    /// The number of rows.
    fn nrows() -> usize;
    /// The number of columns.
    fn ncols() -> usize;
    /// The components in column-major order.
    fn to_column_major(self: M) -> Array<T>;
    /// The matrix of the column-major components `data` (panics with `nalgebra: wrong slice
    /// length` unless there are exactly `nrows * ncols`).
    fn from_column_major(data: Span<T>) -> M;
}

/// The loops of the building blocks (methods of a generic impl: AGENTS.md).
#[generate_trait]
impl HouseholderStepsImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of HouseholderStepsTrait<T> {
    /// `linalg::householder_kernels`'s axis of the vector `x` (any length), bit for bit: `(norm,
    /// true, u)`, or `(0, false, x)` when `x` is exactly zero.
    fn axis(x: Span<T>) -> (T, bool, Span<T>) {
        let len = x.len();
        let mut w = R::wide_zero();
        for xi in x {
            w = R::wide_add_prod(w, *xi, *xi);
        }
        let nrm = R::wide_sqrt(w);
        if nrm == R::zero() {
            return (R::zero(), false, x);
        }
        let x0 = *x[0];
        if len == 1 {
            let u = if x0 < R::zero() {
                -R::one()
            } else {
                R::one()
            };
            return (-x0, true, array![u].span());
        }
        let signed = if x0 < R::zero() {
            -nrm
        } else {
            nrm
        };
        let y0 = x0 + signed;
        let mut w = R::wide_add_prod(R::wide_zero(), y0, y0);
        for xi in x.slice(1, len - 1) {
            w = R::wide_add_prod(w, *xi, *xi);
        }
        let d = R::wide_sqrt(w);
        let mut v: Array<T> = array![R::div(y0, d)];
        for xi in x.slice(1, len - 1) {
            v.append(R::div(*xi, d));
        }
        let mut w = R::wide_zero();
        for vi in v.span() {
            w = R::wide_add_prod(w, *vi, *vi);
        }
        let d2 = R::wide_sqrt(w);
        let mut u: Array<T> = array![];
        for vi in v.span() {
            u.append(R::div(*vi, d2));
        }
        (-signed, true, u.span())
    }

    /// `u · x` over `len` entries of `x` starting at `start` with stride `step`, one floor.
    fn dot(u: Span<T>, x: Span<T>, start: usize, step: usize) -> T {
        let mut w = R::wide_zero();
        let mut k = 0;
        for ui in u {
            w = R::wide_add_prod(w, *ui, *x[start + k * step]);
            k += 1;
        }
        R::wide_rescale(w)
    }

    /// `s (x - 2 h u_k)`: `Real::mul_add(-2h, u_k, x)`, negated exactly when `neg`.
    #[inline(always)]
    fn update(x: T, h: T, uk: T, neg: bool) -> T {
        let w = h + h;
        if neg {
            R::mul_add(w, uk, -x)
        } else {
            R::mul_add(-w, uk, x)
        }
    }

    /// The column-major `a` (`nr` rows, `nc` columns) with its rows `r0..` of the columns `c0..`
    /// reflected from the LEFT by `s (I - 2 u uᵀ)` (`u` of length `nr - r0`).
    fn reflect_left(a: Span<T>, nr: usize, nc: usize, r0: usize, c0: usize, u: Span<T>, neg: bool) -> Array<T> {
        let mut out: Array<T> = array![];
        let mut j = 0;
        while j < nc {
            if j < c0 {
                let mut i = 0;
                while i < nr {
                    out.append(*a[j * nr + i]);
                    i += 1;
                }
            } else {
                let h = Self::dot(u, a, j * nr + r0, 1);
                let mut i = 0;
                while i < nr {
                    let x = *a[j * nr + i];
                    out.append(if i < r0 {
                        x
                    } else {
                        Self::update(x, h, *u[i - r0], neg)
                    });
                    i += 1;
                }
            }
            j += 1;
        }
        out
    }

    /// The column-major `a` with its columns `c0..` of the rows `r0..` reflected from the RIGHT
    /// by `s (I - 2 u uᵀ)` (`u` of length `nc - c0`), and the dot products `row · u` of those
    /// rows (upstream's `work`).
    fn reflect_right(
        a: Span<T>, nr: usize, nc: usize, r0: usize, c0: usize, u: Span<T>, neg: bool,
    ) -> (Array<T>, Array<T>) {
        let mut hs: Array<T> = array![];
        let mut i = r0;
        while i < nr {
            hs.append(Self::dot(u, a, c0 * nr + i, nr));
            i += 1;
        }
        let hs_span = hs.span();
        let mut out: Array<T> = array![];
        let mut j = 0;
        while j < nc {
            let mut i = 0;
            while i < nr {
                let x = *a[j * nr + i];
                out.append(if j < c0 || i < r0 {
                    x
                } else {
                    Self::update(x, *hs_span[i - r0], *u[j - c0], neg)
                });
                i += 1;
            }
            j += 1;
        }
        (out, hs)
    }

    /// `a` with the column `col` (rows `r0..`) replaced by `u`.
    fn store_column(a: Span<T>, nr: usize, col: usize, r0: usize, u: Span<T>) -> Array<T> {
        let mut out: Array<T> = array![];
        let mut k = 0;
        for x in a {
            let (j, i) = DivRem::div_rem(k, nr.try_into().unwrap());
            out.append(if j == col && i >= r0 {
                *u[i - r0]
            } else {
                *x
            });
            k += 1;
        }
        out
    }

    /// `a` with the row `row` (columns `c0..`) replaced by `u`.
    fn store_row(a: Span<T>, nr: usize, row: usize, c0: usize, u: Span<T>) -> Array<T> {
        let mut out: Array<T> = array![];
        let mut k = 0;
        for x in a {
            let (j, i) = DivRem::div_rem(k, nr.try_into().unwrap());
            out.append(if i == row && j >= c0 {
                *u[j - c0]
            } else {
                *x
            });
            k += 1;
        }
        out
    }

    /// `v` (length `len`) with its entries `r0..` replaced by `x`.
    fn store_tail(v: Span<T>, r0: usize, x: Span<T>) -> Array<T> {
        let mut out: Array<T> = array![];
        let mut i = 0;
        for vi in v {
            out.append(if i < r0 {
                *vi
            } else {
                *x[i - r0]
            });
            i += 1;
        }
        out
    }
}

/// Uses a Householder reflection to zero out the column `icol` below its entry `icol + shift`,
/// and returns the signed norm of that part of the column. Upstream:
/// `nalgebra::linalg::householder::clear_column_unchecked` (`#[doc(hidden)]`), every static shape.
///
/// The axis `u` of the entries `icol + shift..` of the column (`reflection_axis_mut`, upstream's
/// two normalisations, bit for bit the kernels of the unrolled decompositions); when the column
/// part is not zero, with `s = sign(norm)`: if `bilateral` is `Some(work)`, the columns `icol +
/// 1..` of EVERY row become `s (x - 2 (x·u) uᵀ)` and `work` receives the dot products `x·u` of the
/// rows (upstream's scratch `work`: the reflection axis must then have as many entries as there
/// are columns after `icol`, panics with `nalgebra: dimension mismatch` otherwise); then the rows
/// `icol + shift..` of the columns `icol + 1..` become `s (x - 2 (u·x) u)`, and the axis is
/// stored in place of the cleared entries. Same rounding as the unrolled `Hessenberg` /
/// `Bidiagonal` (one fused dot product, doubled exactly, one `Real::mul_add` per entry): the
/// Hessenberg decomposition built from this function is bit-identical to `HessenbergN::new`.
///
/// Run-time indices (upstream's interface): loops over column-major arrays, where the
/// decompositions of this crate unroll every step (a static port would be one kernel per shape,
/// `icol` and `shift`). Panics with `nalgebra: index out of bounds` unless `icol < ncols` and
/// `icol + shift < nrows` (upstream: out-of-bounds access), and on overflow.
pub fn clear_column_unchecked<
    M,
    T,
    V,
    impl R: Real<T>,
    impl C: ColumnMajor<M, T>,
    impl W: ColumnMajor<V, T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<M>,
    +Drop<M>,
    +Copy<V>,
    +Drop<V>,
>(
    ref matrix: M, icol: usize, shift: usize, ref bilateral: Option<V>,
) -> T {
    let nr = C::nrows();
    let nc = C::ncols();
    assert(icol < nc && icol + shift < nr, INDEX_OUT_OF_BOUNDS);
    let r0 = icol + shift;
    let a = C::to_column_major(matrix).span();
    let (norm, nz, u) = HouseholderStepsTrait::axis(a.slice(icol * nr + r0, nr - r0));
    if !nz {
        return norm;
    }
    let neg = norm < R::zero();
    let mut a = a;
    if let Option::Some(_) = bilateral {
        assert(nr - r0 == nc - icol - 1, DIMENSION_MISMATCH);
        let (b, hs) = HouseholderStepsTrait::reflect_right(a, nr, nc, 0, icol + 1, u, neg);
        bilateral = Option::Some(W::from_column_major(hs.span()));
        a = b.span();
    }
    let b = HouseholderStepsTrait::reflect_left(a, nr, nc, r0, icol + 1, u, neg);
    let b = HouseholderStepsTrait::store_column(b.span(), nr, icol, r0, u);
    matrix = C::from_column_major(b.span());
    norm
}

/// Uses a Householder reflection to zero out the row `irow` past its entry `irow + shift`, and
/// returns the signed norm of that part of the row. Upstream:
/// `nalgebra::linalg::householder::clear_row_unchecked` (`#[doc(hidden)]`), every static shape.
///
/// The axis `u` of the entries `irow + shift..` of the row (`reflection_axis_mut`), written to the
/// entries `irow + shift..` of `axis_packed` (upstream's scratch); when the row part is not zero,
/// with `s = sign(norm)`, the columns `irow + shift..` of the rows `irow + 1..` become `s (x - 2
/// (x·u) uᵀ)` and the entries `irow + 1..` of `work` receive the dot products `x·u` (upstream's
/// scratch); the axis is stored in place of the cleared entries of the row. Same rounding as the
/// unrolled `Bidiagonal`. Run-time indices (see `clear_column_unchecked`). Panics with
/// `nalgebra: index out of bounds` unless `irow < nrows` and `irow + shift < ncols`, and on
/// overflow.
pub fn clear_row_unchecked<
    M,
    T,
    VC,
    VR,
    impl R: Real<T>,
    impl C: ColumnMajor<M, T>,
    impl AC: ColumnMajor<VC, T>,
    impl WR: ColumnMajor<VR, T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<M>,
    +Drop<M>,
    +Copy<VC>,
    +Drop<VC>,
    +Copy<VR>,
    +Drop<VR>,
>(
    ref matrix: M, ref axis_packed: VC, ref work: VR, irow: usize, shift: usize,
) -> T {
    let nr = C::nrows();
    let nc = C::ncols();
    assert(AC::nrows() == nc && WR::nrows() == nr, DIMENSION_MISMATCH);
    assert(irow < nr && irow + shift < nc, INDEX_OUT_OF_BOUNDS);
    let c0 = irow + shift;
    let a = C::to_column_major(matrix).span();
    let mut row: Array<T> = array![];
    let mut j = c0;
    while j < nc {
        row.append(*a[j * nr + irow]);
        j += 1;
    }
    let (norm, nz, u) = HouseholderStepsTrait::axis(row.span());
    let packed = HouseholderStepsTrait::store_tail(AC::to_column_major(axis_packed).span(), c0, u);
    axis_packed = AC::from_column_major(packed.span());
    if !nz {
        return norm;
    }
    let neg = norm < R::zero();
    let (b, hs) = HouseholderStepsTrait::reflect_right(a, nr, nc, irow + 1, c0, u, neg);
    let w = HouseholderStepsTrait::store_tail(WR::to_column_major(work).span(), irow + 1, hs.span());
    work = WR::from_column_major(w.span());
    let b = HouseholderStepsTrait::store_row(b.span(), nr, irow, c0, u);
    matrix = C::from_column_major(b.span());
    norm
}

/// The orthogonal matrix of the Householder axes stored below the diagonal of the square `m`
/// (column `i` holds the axis of reflection `i` in its rows `i + 1..`), `signs[i]` the signed norm
/// of reflection `i`: the reflections `i = n - 2 .. 0`, each `sign(signs[i]) (I - 2 u uᵀ)`
/// (`sign(0) = 1`), applied to the rows `i + 1..` of the columns `i..` of the identity. Upstream:
/// `nalgebra::linalg::householder::assemble_q` (`#[doc(hidden)]`), the squares 1..6. Same
/// rounding as the unrolled `HessenbergN::q` (bit-identical on the columns the identity's
/// structural zeros leave untouched). Run-time loops (see `clear_column_unchecked`). Panics with
/// `nalgebra: dimension mismatch` when `m` is not square or `signs` has fewer than `n - 1`
/// entries (upstream: an assertion / an out-of-bounds index).
pub fn assemble_q<
    M,
    T,
    impl R: Real<T>,
    impl C: ColumnMajor<M, T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<M>,
    +Drop<M>,
>(
    m: M, signs: Span<T>,
) -> M {
    let n = C::nrows();
    assert(C::ncols() == n, DIMENSION_MISMATCH);
    assert(n == 0 || signs.len() + 1 >= n, DIMENSION_MISMATCH);
    let a = C::to_column_major(m).span();
    let mut res: Array<T> = array![];
    let mut k = 0;
    while k < n * n {
        let (j, i) = DivRem::div_rem(k, n.try_into().unwrap());
        res.append(if i == j {
            R::one()
        } else {
            R::zero()
        });
        k += 1;
    }
    let mut res = res.span();
    let mut i = n;
    while i > 1 {
        i -= 1;
        let col = i - 1;
        let u = a.slice(col * n + col + 1, n - col - 1);
        let neg = *signs[col] < R::zero();
        res = HouseholderStepsTrait::reflect_left(res, n, n, col + 1, col, u, neg).span();
    }
    C::from_column_major(res)
}
"""


def render_householder_steps() -> str:
    uses = {"use simba::scalar::Real;",
            "use crate::base::errors::{DIMENSION_MISMATCH, INDEX_OUT_OF_BOUNDS, SLICE_LENGTH};"}
    for r in DIMS:
        for c in DIMS:
            uses.add(use_shape(r, c))
    impls = "\n".join(column_major_impl(r, c) for r in DIMS for c in DIMS)
    return f"""{HEADER}//! The Householder building blocks of upstream `nalgebra::linalg::householder` that work on
//! run-time indices (WP 8.5-P16): `clear_column_unchecked`, `clear_row_unchecked` and
//! `assemble_q`, on every static shape through `ColumnMajor`. The decompositions of this crate do
//! not call them: they unroll the same steps with static indices (`Hessenberg`, `Bidiagonal`,
//! `SymmetricTridiagonal`); these functions reproduce them bit for bit. `reflection_axis_mut` is
//! `linalg::householder`'s.

{chr(10).join(sorted(uses))}
{STEPS_BODY}
{impls}"""
