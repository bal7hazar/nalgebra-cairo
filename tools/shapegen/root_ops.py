"""WP 8.6-P21: the per-shape impls behind upstream's `Sum` / `Product` and the crate-root
`nalgebra::{inf, sup, inf_sup}`, and the construction macros (`crates/nalgebra/src/macros.cairo`).

Per shape (in the shape's own module: Cairo finds an impl in the module of its trait or of one of
its types, so `iter.sum()` needs no import):

* `core::iter::Sum<S<T>>` (upstream `Sum for OMatrix`, which folds from `zero()`): folds from the
  FIRST item instead, one addition fewer and bit-identical (`0 + x == x` exactly); the zero
  matrix when the iterator is empty, like upstream;
* `core::iter::Sum<@S<T>>`: upstream's `Sum<&'a OMatrix>` (an iterator of references, here of
  snapshots: `span.into_iter().sum()`), a snapshot of the sum (`*span.into_iter().sum()`);
* on the squares, `core::iter::Product<@S<T>>` (upstream `Product<&'a OMatrix>`): folds from the
  first item (one matrix product fewer than upstream's fold from `one()`, bit-identical: every
  output of `I * x` is the exact sum floored once, i.e. `x`); the identity when empty. The owned
  `Product<S<T>>` is corelib's blanket `ProductMultiplicativeTypesImpl` (`One` + `Mul`, both
  upstream impls of the squares): an explicit impl would be ambiguous with it (E2313, measured);
* `crate::root::MatrixInfSup<S<T>>`, the kernel of the free functions `nalgebra::inf` / `sup` /
  `inf_sup` (delegates to the shape's `inf` / `sup` / `inf_sup`).

The macros (`render_macros`) are Cairo declarative macros, one arm per shape (`matrix!`,
`vector!`, `point!`), per row count (`dmatrix!`: the macro grammar of Cairo 2.19 has no nested
repetition with a `;` separator) and per block grid (`stack!`, through the `HStack` / `VStack`
block traits whose impls are generated here, one per conformable pair).
"""

import library as L
from model import ALL_SHAPES, COORDS, Shape

# `dmatrix!` literals: one arm per row count, up to this many rows (any number of columns).
DMATRIX_MAX_ROWS = 16
# `count!` returns a literal up to this many expressions (`dmatrix!` column counts).
COUNT_LITERALS = 16


# The accumulation loop of the iterator impls, unrolled twice: one loop iteration (~1 100 gas of
# overhead, AGENTS.md rule 1) per two items. Measured on 4 `Matrix3` (`bench_sum_matrix3_4`):
# 34 320 gas net against 37 660 for one item per iteration, 34 050 unrolled 4 times (twice the
# code for 1 %) and 47 750 for `Iterator::fold` with a closure (candidates kept as benchmarks).
UNROLLED2 = ("loop {{\nlet Option::Some(x) = iter.next() else {{\nbreak;\n}};\nacc = acc {op} {deref}x;\n"
             "let Option::Some(x) = iter.next() else {{\nbreak;\n}};\nacc = acc {op} {deref}x;\n}}")


def _bounds(extra: list[str]) -> str:
    return ", ".join(["T", "impl R: Real<T>"] + extra)


def items(s: Shape) -> list[str]:
    S, T = s.name, f"{s.name}<T>"
    zero = L.lit(S, [(k, "R::zero()") for k in s.fields])
    out = [L.section("iterator sums and products, crate-root functions (WP 8.6-P21)", 100)]
    add_bounds = _bounds(["+Add<T>", "+Copy<T>", "+Drop<T>"])
    out.append(
        f"/// `iter.sum()` of an iterator of `{S}`s: the first item plus the others, in order "
        f"(exact;\n/// panics on overflow); the zero {s.kind} when empty. Upstream: `Sum for "
        f"Matrix` (a fold\n/// from `zero()`: the same result, one addition more).\n"
        f"pub impl {S}Sum<{add_bounds}> of core::iter::Sum<{T}> {{\n"
        f"fn sum<I, +Iterator<I>[Item: {T}], +Destruct<I>, +Destruct<{T}>>(mut iter: I) -> {T} {{\n"
        f"let Option::Some(mut acc) = iter.next() else {{\nreturn {zero};\n}};\n"
        f"{UNROLLED2.format(op='+', deref='')}\nacc\n}}\n}}")
    out.append(
        f"/// `*iter.sum()` of an iterator of snapshots `@{S}` (`span.into_iter()`): a snapshot "
        f"of the\n/// sum of the items, like `Sum<{S}>`. Upstream: `Sum<&Matrix> for Matrix` "
        f"(references).\n"
        f"pub impl {S}SumSnapshot<{add_bounds}> of core::iter::Sum<@{T}> {{\n"
        f"fn sum<I, +Iterator<I>[Item: @{T}], +Destruct<I>, +Destruct<@{T}>>(mut iter: I) -> @{T} "
        f"{{\nlet Option::Some(first) = iter.next() else {{\nreturn @{zero};\n}};\n"
        f"let mut acc = *first;\n"
        f"{UNROLLED2.format(op='+', deref='*')}\n@acc\n}}\n}}")
    if s.is_square:
        ident = L.lit(S, [(s.f(i, j), "R::one()" if i == j else "R::zero()")
                          for j in range(s.c) for i in range(s.r)])
        mul_bounds = _bounds(["+Mul<T>", "+Copy<T>", "+Drop<T>"] +
                             (["+Drop<R::Wide>"] if s.r >= 5 else []))
        out.append(
            f"/// `*iter.product()` of an iterator of snapshots `@{S}`: the ordered matrix "
            f"product of the\n/// items (`a * b * ..`, each product floored once per "
            f"component), the identity when empty.\n/// Folds from the first item: "
            f"bit-identical to upstream's fold from `one()` (`I * x == x`\n/// exactly), one "
            f"matrix product fewer. The owned form `Product<{S}>` is corelib's blanket\n/// impl "
            f"over `One` + `Mul`. Upstream: `Product<&Matrix> for SquareMatrix` (references).\n"
            f"pub impl {S}ProductSnapshot<{mul_bounds}> of core::iter::Product<@{T}> {{\n"
            f"fn product<I, +Iterator<I>[Item: @{T}], +Destruct<I>, +Destruct<@{T}>>(\n"
            f"mut iter: I,\n) -> @{T} {{\nlet Option::Some(first) = iter.next() else {{\n"
            f"return @{ident};\n}};\nlet mut acc = *first;\n"
            f"{UNROLLED2.format(op='*', deref='*')}\n@acc\n}}\n}}")
    out.append(
        f"/// The kernel of the crate-root `nalgebra::inf` / `sup` / `inf_sup` on `{S}`: the "
        f"shape's\n/// `inf` / `sup` / `inf_sup`.\n"
        f"pub impl {S}InfSup<\n{L.bounds(L.MATRIX_IMPL_BOUNDS)}\n> of "
        f"crate::root::MatrixInfSup<{T}> {{\n"
        f"{L.INLINE}\nfn inf(a: {T}, b: {T}) -> {T} {{\n{S}Trait::inf(a, b)\n}}\n"
        f"{L.INLINE}\nfn sup(a: {T}, b: {T}) -> {T} {{\n{S}Trait::sup(a, b)\n}}\n"
        f"{L.INLINE}\nfn inf_sup(a: {T}, b: {T}) -> ({T}, {T}) {{\n"
        f"{S}Trait::inf_sup(a, b)\n}}\n}}")
    return out


# --------------------------------------------------------------------------------------------
# crates/nalgebra/src/macros.cairo
# --------------------------------------------------------------------------------------------

HEADER = ("// Generated by tools/shapegen/shapegen.py: do not edit by hand. Template:\n"
          "// tools/shapegen/root_ops.py.\n")

POINTS = {1: "geometry", 2: "base", 3: "base", 4: "geometry", 5: "geometry", 6: "geometry"}


def _args(prefix: str, n: int) -> list[str]:
    return [f"${prefix}{k}" for k in range(n)]


def _matrix_arm(s: Shape) -> str:
    rows = [[f"$m{i}{j}" for j in range(s.c)] for i in range(s.r)]
    pattern = "; ".join(", ".join(f"{v}:expr" for v in row) for row in rows)
    args = ", ".join(v for row in rows for v in row)
    return f"[{pattern} $(;)?] => {{ $defsite::super::{s.name}Trait::new({args}) }};"


def _list_arm(n: int, target: str) -> str:
    xs = _args("x", n)
    return (f"[{', '.join(f'{x}:expr' for x in xs)} $(,)?] => {{ $defsite::super::{target}::new("
            f"{', '.join(xs)}) }};")


def _dmatrix_arm(nrows: int) -> str:
    """The row-length check is felt252 arithmetic on the literal counts (`count!`), which the
    compiler folds: zero gas when the rows agree (a `usize` `!=` chain costs 300 gas on 3 rows,
    `bench_dmatrix_macro3__alt_usize_check`). Squares of small differences cannot cancel."""
    rows = [f"$r{i}" for i in range(nrows)]
    pattern = " ; ".join(f"$({r}:expr),+" for r in rows)
    count = lambda r: f"$defsite::count![$({r}),+]"  # noqa: E731
    data = ", ".join(f"$({r}),+" for r in rows)
    lines = []
    if nrows > 1:
        diffs = [f"({count(r)} - {count(rows[0])})" for r in rows[1:]]
        lines.append("if " + " + ".join(f"{d} * {d}" for d in diffs) + " != 0 {")
        lines.append("    $defsite::dimension_mismatch();")
        lines.append("}")
    lines.append(f"$defsite::super::DMatrixTrait::from_row_slice({nrows}, {count(rows[0])}, "
                 f"array![{data}].span())")
    body = "\n".join(" " * 12 + line for line in lines)
    return f"[{pattern} $(;)?] => {{\n        {{\n{body}\n        }}\n    }};"


def _stack_arm(br: int, bc: int) -> str:
    blocks = [[f"$b{i}{j}" for j in range(bc)] for i in range(br)]
    pattern = "; ".join(", ".join(f"{b}:expr" for b in row) for row in blocks)

    def fold(trait: str, fn: str, xs: list[str]) -> str:
        acc = xs[0]
        for x in xs[1:]:
            acc = f"$defsite::{trait}::{fn}({acc}, {x})"
        return acc
    rows = [fold("HStack", "hstack", row) for row in blocks]
    return f"[{pattern} $(;)?] => {{ {fold('VStack', 'vstack', rows)} }};"


def _block_impl(trait: str, fn: str, a: Shape, b: Shape) -> str:
    """`[a, b]` (HStack: same rows) or `[a; b]` (VStack: same columns)."""
    o = Shape(a.r, a.c + b.c) if trait == "HStack" else Shape(a.r + b.r, a.c)
    values = []
    for j in range(o.c):
        for i in range(o.r):
            if trait == "HStack":
                src, si, sj = ("l", i, j) if j < a.c else ("r", i, j - a.c)
            else:
                src, si, sj = ("l", i, j) if i < a.r else ("r", i - a.r, j)
            values.append((o.f(i, j), f"{src}.{(a if src == 'l' else b).f(si, sj)}"))
    A, B, O = f"{a.name}<T>", f"{b.name}<T>", f"{o.name}<T>"
    return (f"pub impl {trait}{a.name}{b.name}<T, +Drop<T>> of {trait}<{A}, {B}, {O}> {{\n"
            f"{L.INLINE}\nfn {fn}(l: {A}, r: {B}) -> {O} {{\n{L.lit(o.name, values)}\n}}\n}}")


def render_macros() -> str:
    shapes = sorted(ALL_SHAPES, key=lambda s: (s.r, s.c))
    uses = ["use crate::base::{" + ", ".join(sorted(s.name for s in shapes)) + "};"]
    doc = """//! The construction macros of upstream's `nalgebra-macros` crate (`matrix!`, `vector!`,
//! `point!`, `dmatrix!`, `dvector!`, `stack!`) as Cairo declarative macros (DESIGN D10), behind the
//! feature `macros` (in `default`, like upstream's). A dependent calls them as `nalgebra::matrix![..]`
//! (or after `use nalgebra::matrix;`) WITHOUT enabling the experimental feature
//! `user_defined_inline_macros` itself (measured, WP 8.6-P21).
//!
//! Upstream syntax (MATLAB-like: `,` between the columns, `;` between the rows), with these
//! Cairo forms:
//! - `matrix!` / `vector!` / `point!` expand to the constructor of the static shape (`Matrix2x3Trait::
//!   new(..)`, row-major arguments like upstream's): the same code, the same gas (measured). Sizes 1
//!   to 6, like the static shapes; no empty form (upstream's `matrix![]` is a 0x0 matrix).
//! - `dmatrix!` expands to `DMatrixTrait::from_row_slice(nrows, ncols, array![..].span())`; the
//!   column count is a compile-time constant, and a row of another length panics with `nalgebra:
//!   dimension mismatch` (a compile error upstream). Up to 16 rows (one arm per row count: Cairo's
//!   macro grammar has no nested repetition separated by `;`), any number of columns. `dvector!`
//!   expands to `DVectorTrait::from_vec(array![..])`. Both need the feature `dynamic` too.
//! - `stack!` concatenates STATIC blocks through the block traits `HStack` / `VStack` (one impl
//!   per conformable pair, moves only: no arithmetic), grids of 1x1 to 6x6 blocks. Upstream's
//!   implicit zero blocks (`0`) and dynamic blocks are not supported: write the zero block
//!   (`Matrix2x3Trait::zeros()`), and use the dynamic edition methods for dynamic blocks.
//!
//! A trailing `;` (`matrix!`, `dmatrix!`, `stack!`) or `,` (`vector!`, `point!`, `dvector!`) is
//! accepted like upstream. Upstream's 0-sized forms (`matrix![]`, `vector![]`, `point![]`) have
//! no Cairo shape, and the macros are not `const` (they call the constructors).
//!
//! Every macro argument is evaluated exactly once, in order."""
    out = [HEADER + doc, "\n".join(uses)]
    out.append(
        "/// Panics with `nalgebra: dimension mismatch` (a `dmatrix!` row whose length is not the "
        "first's).\n/// A function of this module: a macro body resolves names at its definition "
        "site only through\n/// `$defsite::` (the prelude's `core::` / `usize` are not in "
        "scope there).\n#[cfg(feature: 'dynamic')]\nfn dimension_mismatch() {\n"
        "core::panic_with_felt252(crate::base::errors::DIMENSION_MISMATCH)\n}")
    out.append(
        "/// `[a, b]`: the blocks `a` (left) and `b` (right) side by side, `stack!`'s kernel. Moves "
        "only.\npub trait HStack<L, R, Out> {\n    fn hstack(l: L, r: R) -> Out;\n}")
    out.append(
        "/// `[a; b]`: the block `a` above the block `b`, `stack!`'s kernel. Moves only.\n"
        "pub trait VStack<L, R, Out> {\n    fn vstack(l: L, r: R) -> Out;\n}")
    arms = [f"[{', '.join(f'$x{k}:expr' for k in range(n))}] => {{ {n} }};"
            for n in range(1, COUNT_LITERALS + 1)]
    out.append(
        "/// The number of the given expressions (`dmatrix!`'s column count): a literal up to "
        f"{COUNT_LITERALS}\n/// expressions, then `N + count![rest]` (literal arithmetic, folded "
        "at compile time).\n"
        "macro count {\n" + "\n".join(arms) + "\n"
        f"[{', '.join(f'$x{k}:expr' for k in range(COUNT_LITERALS))}, $($rest:expr),+] => "
        f"{{ {COUNT_LITERALS} + $defsite::count![$($rest),+] }};\n}}")
    out.append(
        "/// A statically sized matrix of the given components, `,` between the columns and `;` "
        "between\n/// the rows (row-major, like upstream): `matrix![1, 2, 3; 4, 5, 6]` is a "
        "`Matrix2x3`, `matrix![1, 2]`\n/// a `RowVector2`, `matrix![1; 2]` a `Vector2`. Upstream: "
        "`nalgebra::matrix!`.\npub macro matrix {\n"
        + "\n".join(_matrix_arm(s) for s in shapes) + "\n}")
    out.append(
        "/// A statically sized column vector of the given components: `vector![1, 2, 3]` is a "
        "`Vector3`\n/// (`vector![x]` a `Matrix1`, upstream's `Vector1`). Upstream: "
        "`nalgebra::vector!`.\npub macro vector {\n"
        + "\n".join(_list_arm(n, f"{Shape(n, 1).name}Trait") for n in range(1, 7)) + "\n}")
    out.append(
        "/// A point of the given coordinates: `point![1, 2, 3]` is a `Point3`. Upstream: "
        "`nalgebra::point!`.\npub macro point {\n"
        + "\n".join(_list_arm(n, f"base::point{n}::Point{n}Trait" if POINTS[n] == "base"
                               else f"Point{n}Trait") for n in range(1, 7)) + "\n}")
    out.append(
        "/// A `DMatrix` of the given components, `,` between the columns and `;` between the "
        "rows\n/// (`dmatrix![1, 2, 3; 4, 5, 6]` is 2x3); `dmatrix![]` is the 0x0 matrix. Panics "
        "with `nalgebra:\n/// dimension mismatch` when a row has another length than the first. "
        "Upstream: `nalgebra::dmatrix!`.\n#[cfg(feature: 'dynamic')]\npub macro dmatrix {\n"
        "[] => { $defsite::super::DMatrixTrait::from_row_slice(0, 0, array![].span()) };\n"
        + "\n".join(_dmatrix_arm(n) for n in range(1, DMATRIX_MAX_ROWS + 1)) + "\n}")
    out.append(
        "/// A `DVector` of the given components (`dvector![1, 2, 3]`; `dvector![]` is empty). "
        "Upstream:\n/// `nalgebra::dvector!`.\n#[cfg(feature: 'dynamic')]\npub macro dvector {\n"
        "[$($x:expr),*] => { $defsite::super::DVectorTrait::from_vec(array![$($x),*]) };\n}")
    out.append(
        "/// The block matrix of the given STATIC blocks, `,` between the block columns and `;` "
        "between\n/// the block rows: `stack![a, b; c, d]`. The blocks of a block row have the "
        "same number of\n/// rows, those of a block column the same number of columns, and the "
        "result is at most 6x6\n/// (otherwise no `HStack` / `VStack` impl matches: a compile "
        "error, like upstream's). Upstream:\n/// `nalgebra::stack!` (whose `0` zero blocks and "
        "dynamic blocks are not supported).\npub macro stack {\n"
        + "\n".join(_stack_arm(br, bc) for br in range(1, 7) for bc in range(1, 7)) + "\n}")
    for trait, fn in (("HStack", "hstack"), ("VStack", "vstack")):
        out.append(f"// --- {trait} " + "-" * (100 - 8 - len(trait)))
        for a in shapes:
            for b in shapes:
                if trait == "HStack" and a.r == b.r and a.c + b.c <= 6:
                    out.append(_block_impl(trait, fn, a, b))
                if trait == "VStack" and a.c == b.c and a.r + b.r <= 6:
                    out.append(_block_impl(trait, fn, a, b))
    return "\n\n".join(out) + "\n"
