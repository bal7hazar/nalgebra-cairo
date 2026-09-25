"""Templates of the swizzles, rows, columns, blocks and edition (WP 8.2c, API_PARITY packages P04
and P05), for the 36 shapes and the six points.

Applied like `completion.py` / `functional.py`: a shape that already has a method of that name
keeps it (`missing`), so no existing kernel changes.

Cairo has no borrowed views and no const generics, which fixes the two forms of this module:

* **owned copies**: `row(i)` / `column(j)` return a `RowVectorC` / `VectorR` value, and every
  view of upstream `matrix_view.rs` (`fixed_rows`, `rows`, `view`, `row_part`...) a matrix of the
  smaller shape; nothing aliases `self` (upstream's `_mut` views are excluded, `borrow`);
* **the output type selects the size**: upstream's const generic (`m.fixed_rows::<2>(i)`,
  `m.fixed_view::<2, 3>(i, j)`, `m.fixed_resize::<3, 4>(v)`) becomes the output type of a generic
  trait, inferred like `Into` (`let b: Matrix2x3<Fixed> = m.fixed_view(i, j);`). The traits
  (`base/matrix_view.cairo`) have one impl per (shape, output shape) pair, in the module of the
  source shape, so an impossible size is a compile error ("no impl"), as upstream's. Upstream's
  runtime-sized views (`rows(i, n)`, `view(start, shape)`, `rows_range(r)`, `row_part(i, n)`,
  `select_rows(idx)`, `resize(r, c, v)`) are methods of the same traits: the requested size must
  equal the output type's, else they panic with `nalgebra: dimension mismatch`.

Families (upstream `src/base/*.rs`):

* methods of `<S>Trait` (`matrix.rs`, `properties.rs`, `edition.rs`, `construction.rs`,
  `swizzle.rs`): `nrows`, `ncols`, `shape`, `is_square`, `vector_to_matrix_index`, `row`,
  `column`, `upper_triangle`, `lower_triangle`, `from_rows`, `from_columns`, `is_orthogonal`,
  `insert_row` / `remove_row` / `insert_column` / `remove_column` (the neighbouring shape, when
  it exists), on the squares with a determinant `is_invertible` / `is_special_orthogonal`, on
  the column vectors the swizzles `xx() .. zzz()`;
* per-pair items: `FixedRows` (`fixed_rows`, `rows`, `rows_range`, `select_rows`),
  `FixedColumns` (`fixed_columns`, `columns`, `columns_range`, `select_columns`), `FixedView`
  (`fixed_view`, `view`, the deprecated `fixed_slice` / `slice`), `RowPart` (`row_part`),
  `ColumnPart` (`column_part`), `MatrixKronecker` (`kronecker`, `type Output`), and the
  crate-private `PadTo6` / `CropFrom6` behind the one blanket impl of `FixedResize`
  (`fixed_resize`, `resize`);
* the point swizzles (`geometry/swizzle.rs`): `base/point_swizzle.cairo`, one
  `<Point>SwizzleTrait` per point.

Kernels: a runtime position selects ONE struct literal through a `match` (nested for two
positions): measured cheaper than composing two copies (`bench_matrix4_fixed_view__alt_*`).
`fixed_resize` builds the zero-cost `Matrix6` canvas and crops it: struct moves are free once
inlined (`bench_matrix2x4_fixed_resize__alt_direct`, equal gas), which keeps 72 small impls
instead of 1,296 literals. `kronecker` is one floored product per component.
"""

import library as L
from completion import INDEX_ERROR, fn, item, vec
from model import ALL_SHAPES, COORDS, Shape

DIM_ERROR = "errors::DIMENSION_MISMATCH"
BOUNDS = "T, +Copy<T>, +Drop<T>"
KRON_BOUNDS = "T, +Mul<T>, +Copy<T>, +Drop<T>"
# Squares with a determinant / inverse in the library (`Matrix1` / `Matrix5` come with P14).
DET_SQUARES = {2, 3, 4, 6}
CANVAS = Shape(6, 6)


def use_of(s: Shape) -> str:
    return f"super::{s.module}::{s.name}"


def match_arms(var: str, arms: list[str], default: str) -> str:
    return f"match {var} {{\n" + "".join(f"{k} => {a},\n" for k, a in enumerate(arms)) + \
        f"_ => {default},\n}}"


def panic(err: str = INDEX_ERROR) -> str:
    return f"core::panic_with_felt252({err})"


def small(out: Shape) -> bool:
    """`#[inline(always)]` for literals of at most 16 components (`functional.py`'s rule)."""
    return out.n <= 16


def block(s: Shape, out: Shape, di: int, dj: int, src: str = "self") -> str:
    """The `out`-shaped block of `s` whose top-left component is `(di, dj)`."""
    return L.lit(out.name, [(out.f(i, j), f"{src}.{s.f(i + di, j + dj)}")
                            for j in range(out.c) for i in range(out.r)])


# --------------------------------------------------------------------------------------------
# Swizzles (P04)
# --------------------------------------------------------------------------------------------

# Upstream's `impl_swizzle!` set (`base/swizzle.rs`, `geometry/swizzle.rs`): name -> indices.
SWIZZLES: dict[str, tuple[int, ...]] = {}
for _k in (2, 3):
    for _idx in __import__("itertools").product(range(3), repeat=_k):
        SWIZZLES["".join(COORDS[i] for i in _idx)] = _idx


def swizzle_names(dim: int) -> list[str]:
    """The swizzles defined for a dimension (upstream: every index below the dimension), in
    upstream's declaration order (by largest index, then as listed)."""
    order = sorted(SWIZZLES, key=lambda n: (max(SWIZZLES[n]), len(n) != 2, n))
    return [n for n in order if max(SWIZZLES[n]) < dim]


def swizzle_fn(s: Shape, name: str) -> L.Fn:
    idx = SWIZZLES[name]
    out = vec(len(idx))
    body = L.lit(out.name, [(out.f(k, 0), f"self.{s.f(i, 0)}") for k, i in enumerate(idx)])
    return fn(name, f"The `{out.name}` `({', '.join(COORDS[i] for i in idx)})` of components of "
                    f"`self`. Exact. Upstream: the `{name}` swizzle.",
              f"fn {name}(self: {s.name}<T>) -> {out.name}<T>", body)


# --------------------------------------------------------------------------------------------
# Methods of `<S>Trait`
# --------------------------------------------------------------------------------------------


def methods(s: Shape) -> list[L.Fn]:
    S, T, F = s.name, f"{s.name}<T>", s.fields
    ij = {k: (i, j) for j in range(s.c) for i in range(s.r) for k in [s.f(i, j)]}
    rw, cl = Shape(1, s.c), Shape(s.r, 1)
    out: list[L.Fn] = []
    out.append(fn("nrows", f"The number of rows, {s.r}. Upstream: `nrows`.",
                  f"fn nrows(self: {T}) -> usize", f"{s.r}"))
    out.append(fn("ncols", f"The number of columns, {s.c}. Upstream: `ncols`.",
                  f"fn ncols(self: {T}) -> usize", f"{s.c}"))
    out.append(fn("shape", f"`(nrows, ncols)`, `({s.r}, {s.c})`. Upstream: `shape`.",
                  f"fn shape(self: {T}) -> (usize, usize)", f"({s.r}, {s.c})"))
    out.append(fn("is_square", f"Whether the shape is square: `{str(s.is_square).lower()}`. "
                               f"Upstream: `is_square`.",
                  f"fn is_square(self: {T}) -> bool", str(s.is_square).lower()))
    if s.r == 1:
        v2m = "(0, i)"
    elif s.c == 1:
        v2m = "(i, 0)"
    else:
        v2m = f"let (j, i) = DivRem::div_rem(i, {s.r});\n(i, j)"
    out.append(fn("vector_to_matrix_index",
                  f"The `(row, column)` of the `i`-th component in column-major order: `(i % "
                  f"{s.r}, i / {s.r})`, one `DivRem` (no bounds check, like upstream). Upstream: "
                  f"`vector_to_matrix_index`.",
                  f"fn vector_to_matrix_index(self: {T}, i: usize) -> (usize, usize)", v2m))
    out.append(fn(
        "row", f"Row `i`, a `{rw.name}` (an owned copy: Cairo has no borrowed views). Panics with "
               f"`nalgebra: index out of bounds` for `i >= {s.r}`. ONE `match` on `i` selects the "
               f"literal (the private `row_at` of `swap_rows`). Upstream: `row` (a view).",
        f"fn row(self: {T}, i: usize) -> {rw.name}<T>", f"{S}EditTrait::row_at(self, i)"))
    out.append(fn(
        "column", f"Column `j`, a `{cl.name}` (an owned copy: Cairo has no borrowed views). Panics "
                  f"with `nalgebra: index out of bounds` for `j >= {s.c}`. ONE `match` on `j` "
                  f"selects the literal (the private `column_at` of `swap_columns`). Upstream: `column` (a "
                  f"view).",
        f"fn column(self: {T}, j: usize) -> {cl.name}<T>", f"{S}EditTrait::column_at(self, j)"))
    lit = lambda g: L.lit(S, [(k, g(k)) for k in F])  # noqa: E731
    out.append(fn("upper_triangle", "The upper triangle of `self` (the diagonal included), the "
                                    "components below the diagonal set to zero. Exact. Upstream: "
                                    "`upper_triangle`.",
                  f"fn upper_triangle(self: {T}) -> {T}",
                  lit(lambda k: "R::zero()" if ij[k][0] > ij[k][1] else f"self.{k}")))
    out.append(fn("lower_triangle", "The lower triangle of `self` (the diagonal included), the "
                                    "components above the diagonal set to zero. Exact. Upstream: "
                                    "`lower_triangle`.",
                  f"fn lower_triangle(self: {T}) -> {T}",
                  lit(lambda k: "R::zero()" if ij[k][0] < ij[k][1] else f"self.{k}")))
    rv = vec(s.c)
    out.append(fn(
        "from_rows",
        f"The {s.kind} whose {s.r} rows are the given vectors, each a `{rv.name}` of the row's "
        f"{s.c} components (the argument form of the former `Matrix2/3/4::from_rows`, used by "
        f"`linalg`). Exact. Upstream: `from_rows` (a slice of row vectors).",
        f"fn from_rows({', '.join(f'r{i + 1}: {rv.name}<T>' for i in range(s.r))}) -> {T}",
        lit(lambda k: f"r{ij[k][0] + 1}.{rv.f(ij[k][1], 0)}")))
    cv = vec(s.r)
    out.append(fn(
        "from_columns", f"The {s.kind} whose {s.c} columns are the given `{cv.name}`s. Exact. "
                        f"Upstream: `from_columns` (a slice of column vectors).",
        f"fn from_columns({', '.join(f'c{j + 1}: {cv.name}<T>' for j in range(s.c))}) -> {T}",
        lit(lambda k: f"c{ij[k][1] + 1}.{cv.f(ij[k][0], 0)}")))
    sq = Shape(s.c, s.c)
    if s.is_column:
        body, how = ("R::abs_diff_eq(Self::norm_squared(self), R::one(), ulps)",
                     "`|self|² = 1` within `ulps` (`selfᵀ * self` is the 1x1 `norm_squared`)")
    else:
        prod = "MatrixTrMul::tr_mul(self, self)"
        body = (f"Self::is_identity({prod}, ulps)" if s.is_square else
                f"{sq.name}Trait::is_identity({prod}, ulps)")
        how = (f"`selfᵀ * self` (fused `tr_mul`, one rounding per component) is the {s.c}x{s.c} "
               f"identity within `ulps`")
    out.append(fn("is_orthogonal", f"Whether the columns of `self` are orthonormal: {how}. "
                                   f"Upstream: `is_orthogonal` (`eps`: here a tolerance in ulp, "
                                   f"DESIGN D3).",
                  f"fn is_orthogonal(self: {T}, ulps: u64) -> bool", body))
    if s.is_square and s.r in DET_SQUARES:
        via = "Matrix6LuTrait::" if s.r == 6 else "Self::"
        out.append(fn("is_invertible", f"Whether `self` is invertible: `try_inverse()` is `Some`"
                                       f"{' (through `Lu6`, like `Matrix6::try_inverse`)' if s.r == 6 else ''}. "
                                       f"Upstream: `is_invertible`.",
                      f"fn is_invertible(self: {T}) -> bool",
                      f"{via}try_inverse(self).is_some()", s.r != 6))
        out.append(fn("is_special_orthogonal",
                      "Whether `self` is a rotation: orthogonal within `ulps` (`is_orthogonal`) "
                      "with a positive determinant. Upstream: `is_special_orthogonal`.",
                      f"fn is_special_orthogonal(self: {T}, ulps: u64) -> bool",
                      f"Self::is_orthogonal(self, ulps) && {via}determinant(self) > R::zero()",
                      s.r != 6))
    # Edition into the neighbouring shape.
    if s.r < 6:
        o = Shape(s.r + 1, s.c)
        arms = [L.lit(o.name, [(o.f(i, j), "val" if i == p else
                                f"self.{s.f(i if i < p else i - 1, j)}")
                               for j in range(o.c) for i in range(o.r)]) for p in range(s.r + 1)]
        out.append(fn("insert_row", f"`self` with a row of `val` inserted at index `i` (`i <= "
                                    f"{s.r}`), a `{o.name}`. Panics with `nalgebra: index out of "
                                    f"bounds` for `i > {s.r}`. Upstream: `insert_row`.",
                      f"fn insert_row(self: {T}, i: usize, val: T) -> {o.name}<T>",
                      match_arms("i", arms, panic()), small(o)))
    if s.c < 6:
        o = Shape(s.r, s.c + 1)
        arms = [L.lit(o.name, [(o.f(i, j), "val" if j == p else
                                f"self.{s.f(i, j if j < p else j - 1)}")
                               for j in range(o.c) for i in range(o.r)]) for p in range(s.c + 1)]
        out.append(fn("insert_column", f"`self` with a column of `val` inserted at index `i` (`i "
                                       f"<= {s.c}`), a `{o.name}`. Panics with `nalgebra: index "
                                       f"out of bounds` for `i > {s.c}`. Upstream: "
                                       f"`insert_column`.",
                      f"fn insert_column(self: {T}, i: usize, val: T) -> {o.name}<T>",
                      match_arms("i", arms, panic()), small(o)))
    if s.r > 1:
        o = Shape(s.r - 1, s.c)
        arms = [L.lit(o.name, [(o.f(i, j), f"self.{s.f(i if i < p else i + 1, j)}")
                               for j in range(o.c) for i in range(o.r)]) for p in range(s.r)]
        out.append(fn("remove_row", f"`self` without its row `i`, a `{o.name}`. Panics with "
                                    f"`nalgebra: index out of bounds` for `i >= {s.r}`. Upstream: "
                                    f"`remove_row`.",
                      f"fn remove_row(self: {T}, i: usize) -> {o.name}<T>",
                      match_arms("i", arms, panic()), small(o)))
    if s.c > 1:
        o = Shape(s.r, s.c - 1)
        arms = [L.lit(o.name, [(o.f(i, j), f"self.{s.f(i, j if j < p else j + 1)}")
                               for j in range(o.c) for i in range(o.r)]) for p in range(s.c)]
        out.append(fn("remove_column", f"`self` without its column `i`, a `{o.name}`. Panics with "
                                       f"`nalgebra: index out of bounds` for `i >= {s.c}`. "
                                       f"Upstream: `remove_column`.",
                      f"fn remove_column(self: {T}, i: usize) -> {o.name}<T>",
                      match_arms("i", arms, panic()), small(o)))
    if s.is_column:
        out += [swizzle_fn(s, n) for n in swizzle_names(s.r)]
    return out


def missing(s: Shape, have: set[str]) -> list[L.Fn]:
    return [f for f in methods(s) if f.name not in have]


# --------------------------------------------------------------------------------------------
# Top-level items (in the module of the source shape)
# --------------------------------------------------------------------------------------------


def impl(name: str, trait: str, bounds: str, fns: list[str]) -> str:
    return f"pub impl {name}<{bounds}> of {trait} {{\n" + "\n\n".join(fns) + "\n}"


def method(sig: str, body: str, inline: bool = True, hint: bool = False) -> str:
    attr = "#[inline(always)]\n" if inline else ("#[inline]\n" if hint else "")
    return f"{attr}{sig} {{\n{body}\n}}"


def check(cond: str) -> str:
    return f"if {cond} {{\n{panic(DIM_ERROR)}\n}}\n"


def fixed_rows_impl(s: Shape, d: int) -> str:
    S, o = s.name, Shape(d, s.c)
    O = o.name
    arms = [block(s, o, di, 0) for di in range(s.r - d + 1)]
    rows = "\n".join(f"let r{k} = {S}EditTrait::row_at(self, *irows[{k}]);" for k in range(d))
    rv = Shape(1, s.c)
    sel = L.lit(O, [(o.f(i, j), f"r{i}.{rv.f(0, j)}") for j in range(o.c) for i in range(o.r)])
    fns = [
        method(f"fn fixed_rows(self: {S}<T>, i: usize) -> {O}<T>",
               match_arms("i", arms, panic()), small(o)),
        method(f"fn select_rows(self: {S}<T>, irows: Span<usize>) -> {O}<T>",
               check(f"irows.len() != {d}") + rows + "\n" + sel, small(o)),
    ]
    return item(f"The {d} consecutive rows of a `{S}` as a `{O}` (`rows` / `rows_range`: default "
                f"methods). Upstream: `fixed_rows::<{d}>`, `select_rows`.",
                impl(f"{S}FixedRows{O}", f"FixedRows<{S}<T>, {O}<T>>", BOUNDS, fns))


def fixed_columns_impl(s: Shape, d: int) -> str:
    S, o = s.name, Shape(s.r, d)
    O = o.name
    arms = [block(s, o, 0, dj) for dj in range(s.c - d + 1)]
    cols = "\n".join(f"let c{k} = {S}EditTrait::column_at(self, *icols[{k}]);" for k in range(d))
    cv = Shape(s.r, 1)
    sel = L.lit(O, [(o.f(i, j), f"c{j}.{cv.f(i, 0)}") for j in range(o.c) for i in range(o.r)])
    fns = [
        method(f"fn fixed_columns(self: {S}<T>, i: usize) -> {O}<T>",
               match_arms("i", arms, panic()), small(o)),
        method(f"fn select_columns(self: {S}<T>, icols: Span<usize>) -> {O}<T>",
               check(f"icols.len() != {d}") + cols + "\n" + sel, small(o)),
    ]
    return item(f"The {d} consecutive columns of a `{S}` as a `{O}` (`columns` / `columns_range`: "
                f"default methods). Upstream: `fixed_columns::<{d}>`, `select_columns`.",
                impl(f"{S}FixedColumns{O}", f"FixedColumns<{S}<T>, {O}<T>>", BOUNDS, fns))


# `nested`: ONE nested `match` selects the block literal (measured cheapest); `composed`:
# `fixed_columns` then `fixed_rows` (less code, dearer).
VIEW_KERNEL = "nested"


def fixed_view_impl(s: Shape, o: Shape) -> str:
    S, O = s.name, o.name
    if VIEW_KERNEL == "nested":
        outer = []
        for dj in range(s.c - o.c + 1):
            inner = [block(s, o, di, dj) for di in range(s.r - o.r + 1)]
            outer.append(match_arms("irow", inner, panic()))
        body, inline = match_arms("icol", outer, panic()), small(o)
    else:
        mid = Shape(s.r, o.c)
        body = (f"let c: {mid.name}<T> = FixedColumns::fixed_columns(self, icol);\n"
                f"FixedRows::fixed_rows(c, irow)")
        inline = True
    fns = [method(f"fn fixed_view(self: {S}<T>, irow: usize, icol: usize) -> {O}<T>", body,
                  inline)]
    return item(f"The {o.r}x{o.c} blocks of a `{S}` as a `{O}` (`view`, `fixed_slice`, `slice`: "
                f"default methods). Upstream: `fixed_view::<{o.r}, {o.c}>`.",
                impl(f"{S}FixedView{O}", f"FixedView<{S}<T>, {O}<T>>", BOUNDS, fns))


def kronecker_impl(a: Shape, b: Shape) -> str:
    o = Shape(a.r * b.r, a.c * b.c)
    A, B, O = a.name, b.name, o.name
    vals = [(o.f(i, j), f"self.{a.f(i // b.r, j // b.c)} * rhs.{b.f(i % b.r, j % b.c)}")
            for j in range(o.c) for i in range(o.r)]
    return item(f"The Kronecker product of a `{A}` and a `{B}`, a `{O}`: one floored product per "
                f"component. Panics on overflow. Upstream: `kronecker`.",
                f"pub impl {A}Kronecker{B}<{KRON_BOUNDS}> of MatrixKronecker<{A}<T>, {B}<T>> {{\n"
                f"type Output = {O}<T>;\n"
                + method(f"fn kronecker(self: {A}<T>, rhs: {B}<T>) -> {O}<T>", L.lit(O, vals),
                         small(o)) + "\n}")


def kron_rhs(a: Shape) -> list[Shape]:
    return [b for b in ALL_SHAPES if a.r * b.r <= 6 and a.c * b.c <= 6]


def resize_impls(s: Shape) -> list[str]:
    S, C = s.name, CANVAS
    pad = L.lit(C.name, [(C.f(i, j), f"self.{s.f(i, j)}" if i < s.r and j < s.c else "val")
                         for j in range(C.c) for i in range(C.r)])
    crop = block(C, s, 0, 0, "m")
    return [
        f"/// `self` in the top-left corner of a `Matrix6` filled with `val` (`FixedResize`).\n"
        f"pub(crate) impl {S}PadTo6<{BOUNDS}> of PadTo6<{S}<T>, T> {{\n"
        + method(f"fn pad(self: {S}<T>, val: T) -> Matrix6<T>", pad) + "\n}",
        f"/// The top-left `{S}` of a `Matrix6` (`FixedResize`).\n"
        f"pub(crate) impl {S}CropFrom6<{BOUNDS}> of CropFrom6<{S}<T>, T> {{\n"
        + method(f"fn crop(m: Matrix6<T>) -> {S}<T>", crop) + "\n}",
        f"/// `({s.r}, {s.c})`: the size checks of the runtime-sized views.\n"
        f"pub(crate) impl {S}ShapeDims<T> of ShapeDims<{S}<T>> {{\n"
        + method("fn dims() -> (usize, usize)", f"({s.r}, {s.c})") + "\n}",
    ]


def items(s: Shape) -> list[str]:
    out = [L.section("rows, columns, blocks and edition (WP 8.2c)", 100)]
    out += [fixed_rows_impl(s, d) for d in range(1, s.r + 1)]
    out += [fixed_columns_impl(s, d) for d in range(1, s.c + 1)]
    out += [fixed_view_impl(s, Shape(r, c)) for r in range(1, s.r + 1) for c in range(1, s.c + 1)]
    out += [kronecker_impl(s, b) for b in kron_rhs(s)]
    out += resize_impls(s)
    return out


def uses(s: Shape) -> list[str]:
    out = ["super::errors",
           "super::matrix_view::{CropFrom6, FixedColumns, FixedRows, FixedView, PadTo6, "
           "ShapeDims}", "super::matrix_kronecker::MatrixKronecker",
           "super::matrix_tr_mul::MatrixTrMul"]
    shapes = {Shape(1, s.c), Shape(s.r, 1), CANVAS, vec(s.c), vec(s.r)}
    shapes |= {Shape(d, s.c) for d in range(1, s.r + 1)}
    shapes |= {Shape(s.r, d) for d in range(1, s.c + 1)}
    shapes |= {Shape(r, c) for r in range(1, s.r + 1) for c in range(1, s.c + 1)}
    shapes |= {Shape(s.r * b.r, s.c * b.c) for b in kron_rhs(s)} | set(kron_rhs(s))
    shapes |= {Shape(s.r + 1, s.c)} if s.r < 6 else set()
    shapes |= {Shape(s.r, s.c + 1)} if s.c < 6 else set()
    shapes |= {Shape(s.r - 1, s.c)} if s.r > 1 else set()
    shapes |= {Shape(s.r, s.c - 1)} if s.c > 1 else set()
    if s.is_column:
        shapes |= {vec(2), vec(3)}
    out += [use_of(t) for t in sorted(shapes) if t != s]
    if not s.is_column and not s.is_square:
        out.append(f"super::{Shape(s.c, s.c).module}::{Shape(s.c, s.c).name}Trait")
    if s.is_square and s.r == 6:
        out.append("crate::linalg::Matrix6LuTrait")
    return out


# --------------------------------------------------------------------------------------------
# Shared modules
# --------------------------------------------------------------------------------------------

HEADER = ("// Generated by tools/shapegen/shapegen.py: do not edit by hand. Template:\n"
          "// tools/shapegen/views.py.\n")


def render_matrix_view() -> str:
    vec_uses = "\n".join(f"use super::{v.module}::{v.name};"
                         for v in sorted({Shape(1, n) for n in range(1, 7)}
                                         | {Shape(n, 1) for n in range(1, 7)} | {CANVAS}))
    lens = []
    for n in range(1, 7):
        for trait, v in (("RowVectorLen", Shape(1, n)), ("ColumnVectorLen", Shape(n, 1))):
            if n == 1 and trait == "ColumnVectorLen":
                v = Shape(1, 1)
            lens.append(f"impl {v.name}{trait}<T> of {trait}<{v.name}<T>> {{\n"
                        f"#[inline(always)]\nfn len() -> usize {{\n{n}\n}}\n}}")
    return HEADER + """//! Owned rows, columns and blocks of the static shapes (upstream `base/matrix_view.rs`,
//! `base/edition.rs`).
//!
//! Upstream returns borrowed views sized by const generics (`m.fixed_rows::<2>(i)`,
//! `m.fixed_view::<2, 3>(i, j)`) or at run time (`m.rows(i, n)`, `m.view(start, shape)`). Cairo
//! has neither borrowed views nor const generics: each block is an OWNED copy whose size is the
//! OUTPUT TYPE, inferred like `Into`'s:
//!
//! ```cairo
//! let b: Matrix2x3<Fixed> = m.fixed_view(1, 0); // upstream m.fixed_view::<2, 3>(1, 0)
//! let r: Matrix2x4<Fixed> = m.rows(1, 2); // upstream m.rows(1, 2)
//! ```
//!
//! `FixedRows`, `FixedColumns` and `FixedView` have one impl per (shape, output shape) pair, in
//! the module of the source shape: a size that does not fit is a compile error, like upstream's.
//! Their runtime-sized forms (`rows`, `rows_range`, `view`...) are default methods, and `RowPart`
//! / `ColumnPart` / `FixedResize` blanket impls: they panic with `nalgebra: dimension mismatch`
//! when the requested size is not the output type's. Every position out of the shape panics with
//! `nalgebra: index out of bounds`. A runtime position selects ONE struct literal through a
//! `match` (no loop, no per-component test).

use super::errors;
""" + vec_uses + """

/// `D` consecutive rows (`Out` has `D` rows and the columns of `M`). Upstream: `fixed_rows`,
/// `rows`, `rows_range`, `select_rows`.
pub trait FixedRows<M, Out> {
    /// The rows `i .. i + D`. Upstream: `fixed_rows::<D>(i)`.
    fn fixed_rows(self: M, i: usize) -> Out;
    /// The rows at the `D` indices of `irows` (any order, repeats allowed). Upstream:
    /// `select_rows` (an iterator of indices and a dynamic result; a `Span` and a static result
    /// here).
    fn select_rows(self: M, irows: Span<usize>) -> Out;
    /// The rows `first_row .. first_row + nrows`; `nrows` must be `D`. Upstream: `rows`.
    #[inline]
    fn rows<+ShapeDims<Out>, +Drop<M>>(self: M, first_row: usize, nrows: usize) -> Out {
        let (d, _) = ShapeDims::<Out>::dims();
        if nrows != d {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        Self::fixed_rows(self, first_row)
    }
    /// The rows `start .. end` of `rows` (`end - start = D`). Upstream: `rows_range` (a
    /// `Range<usize>` here; upstream also takes a single index and the other range forms).
    #[inline]
    fn rows_range<+ShapeDims<Out>, +Drop<M>>(self: M, rows: core::ops::Range<usize>) -> Out {
        let (d, _) = ShapeDims::<Out>::dims();
        let core::ops::Range { start, end } = rows;
        if end != start + d {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        Self::fixed_rows(self, start)
    }
}

/// `D` consecutive columns (`Out` has the rows of `M` and `D` columns). Upstream:
/// `fixed_columns`, `columns`, `columns_range`, `select_columns`.
pub trait FixedColumns<M, Out> {
    /// The columns `i .. i + D`. Upstream: `fixed_columns::<D>(i)`.
    fn fixed_columns(self: M, i: usize) -> Out;
    /// The columns at the `D` indices of `icols` (any order, repeats allowed). Upstream:
    /// `select_columns` (an iterator of indices and a dynamic result; a `Span` and a static
    /// result here).
    fn select_columns(self: M, icols: Span<usize>) -> Out;
    /// The columns `first_col .. first_col + ncols`; `ncols` must be `D`. Upstream: `columns`.
    #[inline]
    fn columns<+ShapeDims<Out>, +Drop<M>>(self: M, first_col: usize, ncols: usize) -> Out {
        let (_, d) = ShapeDims::<Out>::dims();
        if ncols != d {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        Self::fixed_columns(self, first_col)
    }
    /// The columns `start .. end` of `cols` (`end - start = D`). Upstream: `columns_range` (a
    /// `Range<usize>` here).
    #[inline]
    fn columns_range<+ShapeDims<Out>, +Drop<M>>(self: M, cols: core::ops::Range<usize>) -> Out {
        let (_, d) = ShapeDims::<Out>::dims();
        let core::ops::Range { start, end } = cols;
        if end != start + d {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        Self::fixed_columns(self, start)
    }
}

/// An `R2 x C2` block (`Out`). Upstream: `fixed_view`, `view`, `fixed_slice`, `slice`.
pub trait FixedView<M, Out> {
    /// The block whose top-left component is `(irow, icol)`. Upstream:
    /// `fixed_view::<R2, C2>(irow, icol)`.
    fn fixed_view(self: M, irow: usize, icol: usize) -> Out;
    /// The block at `start` (row, column) of size `shape`, which must be `(R2, C2)`. Upstream:
    /// `view`.
    #[inline]
    fn view<+ShapeDims<Out>, +Drop<M>>(self: M, start: (usize, usize), shape: (usize, usize)) -> Out {
        if shape != ShapeDims::<Out>::dims() {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        let (irow, icol) = start;
        Self::fixed_view(self, irow, icol)
    }
    /// `fixed_view` (deprecated upstream alias). Upstream: `fixed_slice`.
    #[inline(always)]
    fn fixed_slice(self: M, irow: usize, icol: usize) -> Out {
        Self::fixed_view(self, irow, icol)
    }
    /// `view` (deprecated upstream alias). Upstream: `slice`.
    #[inline]
    fn slice<+ShapeDims<Out>, +Drop<M>>(self: M, start: (usize, usize), shape: (usize, usize)) -> Out {
        Self::view(self, start, shape)
    }
}

/// The first `n` components of a row, as a row vector (`Out = RowVectorN`, `Matrix1` for one).
/// Upstream: `row_part`.
pub trait RowPart<M, Out> {
    /// The first `n` components of row `i`; `n` must be `Out`'s length. Upstream: `row_part`.
    fn row_part(self: M, i: usize, n: usize) -> Out;
}

/// The first `n` components of a column, as a column vector (`Out = VectorN`, `Matrix1` for
/// one). Upstream: `column_part`.
pub trait ColumnPart<M, Out> {
    /// The first `n` components of column `i`; `n` must be `Out`'s length. Upstream:
    /// `column_part`.
    fn column_part(self: M, i: usize, n: usize) -> Out;
}

/// `self` resized to the shape `Out`: the common top-left block is kept, the new components are
/// `val`. Upstream: `fixed_resize`, `resize`.
pub trait FixedResize<M, Out, T> {
    /// `self` resized to `Out`, new components `val`. Upstream: `fixed_resize::<R2, C2>(val)`.
    fn fixed_resize(self: M, val: T) -> Out;
    /// `self` resized to `(new_nrows, new_ncols)`, which must be `Out`'s shape. Upstream:
    /// `resize` (a dynamic result upstream).
    fn resize(self: M, new_nrows: usize, new_ncols: usize, val: T) -> Out;
}

/// The shape `(nrows, ncols)` of `S` (the size checks of the runtime-sized forms).
pub(crate) trait ShapeDims<S> {
    fn dims() -> (usize, usize);
}

/// The length of a row vector `S` (`RowPart`).
pub(crate) trait RowVectorLen<S> {
    fn len() -> usize;
}

/// The length of a column vector `S` (`ColumnPart`).
pub(crate) trait ColumnVectorLen<S> {
    fn len() -> usize;
}

/// `self` in the top-left corner of a `Matrix6` filled with `val` (the canvas of `FixedResize`).
pub(crate) trait PadTo6<M, T> {
    fn pad(self: M, val: T) -> Matrix6<T>;
}

/// The top-left block of a `Matrix6` of the shape `Out` (the canvas of `FixedResize`).
pub(crate) trait CropFrom6<Out, T> {
    fn crop(m: Matrix6<T>) -> Out;
}

""" + "\n\n".join(lens) + """

/// `row_part` for every row-vector output: `fixed_view(i, 0)` after the length check.
pub impl RowPartImpl<
    M, Out, impl V: FixedView<M, Out>, impl L: RowVectorLen<Out>, +Drop<M>,
> of RowPart<M, Out> {
    #[inline(always)]
    fn row_part(self: M, i: usize, n: usize) -> Out {
        if n != L::len() {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        V::fixed_view(self, i, 0)
    }
}

/// `column_part` for every column-vector output: `fixed_view(0, i)` after the length check.
pub impl ColumnPartImpl<
    M, Out, impl V: FixedView<M, Out>, impl L: ColumnVectorLen<Out>, +Drop<M>,
> of ColumnPart<M, Out> {
    #[inline(always)]
    fn column_part(self: M, i: usize, n: usize) -> Out {
        if n != L::len() {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        V::fixed_view(self, 0, i)
    }
}

/// `FixedResize` for every pair of shapes: pad to the `Matrix6` canvas, crop to `Out`. Struct
/// moves are free once inlined, so this costs exactly one literal of `Out` (measured equal to the
/// direct literal, `bench_matrix2x4_fixed_resize__alt_direct`) for 72 small impls instead of
/// 1,296 literals.
pub impl FixedResizeImpl<
    M,
    Out,
    T,
    impl P: PadTo6<M, T>,
    impl C: CropFrom6<Out, T>,
    impl D: ShapeDims<Out>,
    +Drop<M>,
    +Drop<T>,
> of FixedResize<M, Out, T> {
    #[inline(always)]
    fn fixed_resize(self: M, val: T) -> Out {
        C::crop(P::pad(self, val))
    }

    #[inline(always)]
    fn resize(self: M, new_nrows: usize, new_ncols: usize, val: T) -> Out {
        if (new_nrows, new_ncols) != D::dims() {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH)
        }
        C::crop(P::pad(self, val))
    }
}
"""


def render_matrix_kronecker() -> str:
    return HEADER + """//! `MatrixKronecker`: the Kronecker product of two static shapes (upstream `base/ops.rs`).
//!
//! Upstream's `kronecker` takes any right-hand matrix and returns an `(R1 R2) x (C1 C2)` matrix;
//! Cairo has no overloading, so it is the method of a trait generic over both operands, the
//! output shape being the impl's associated `Output`. The impls (one per pair whose product
//! still has at most 6 rows and 6 columns, 196) live in the module of the left operand.

/// `self ⊗ rhs`: the block matrix `[self[(i, j)] * rhs]`, one floored product per component.
/// Upstream: `Matrix::kronecker`.
pub trait MatrixKronecker<Lhs, Rhs> {
    /// The shape of `lhs ⊗ rhs`.
    type Output;
    /// `self ⊗ rhs`.
    fn kronecker(self: Lhs, rhs: Rhs) -> Self::Output;
}
"""


POINTS = {1: "geometry::point1::Point1", 2: "base::point2::Point2", 3: "base::point3::Point3",
          4: "geometry::point4::Point4", 5: "geometry::point5::Point5",
          6: "geometry::point6::Point6"}
# Swizzles the points already have as methods of their own trait.
POINT_HAVE = {3: {"xy"}}


def render_point_swizzle() -> str:
    blocks = [HEADER + """//! The swizzles of the points (upstream `geometry/swizzle.rs`): `p.xy()`, `p.zyx()`... build a
//! `Point2` / `Point3` from components of `p`, on every point that has them (upstream: every
//! index below the dimension). One `<Point>SwizzleTrait` per point (import it to call them);
//! `Point3::xy` predates them and stays in `Point3Trait`.""",
              "\n".join(f"use crate::{POINTS[d]};" for d in sorted(POINTS) if d != 1)
              + "\nuse crate::geometry::point1::Point1;"]
    for d in range(1, 7):
        P = f"Point{d}"
        fns = []
        for name in swizzle_names(d):
            if name in POINT_HAVE.get(d, set()):
                continue
            idx = SWIZZLES[name]
            out = f"Point{len(idx)}"
            body = f"{out} {{ " + ", ".join(f"{COORDS[k]}: self.{COORDS[i]}"
                                             for k, i in enumerate(idx)) + " }"
            fns.append(f"/// The `{out}` `({', '.join(COORDS[i] for i in idx)})` of coordinates of "
                       f"`self`. Upstream: the `{name}` swizzle.\n#[inline(always)]\n"
                       f"fn {name}(self: {P}<T>) -> {out}<T> {{\n{body}\n}}")
        blocks.append(f"/// The swizzles of `{P}`.\n#[generate_trait]\n"
                      f"pub impl {P}SwizzleImpl<{BOUNDS}> of {P}SwizzleTrait<T> {{\n"
                      + "\n\n".join(fns) + "\n}")
    return "\n\n".join(blocks) + "\n"


SHARED_MODULES = {"matrix_view": render_matrix_view, "matrix_kronecker": render_matrix_kronecker,
                  "point_swizzle": render_point_swizzle}
SHARED_EXPORTS = {"matrix_view": ["ColumnPart", "FixedColumns", "FixedResize", "FixedRows",
                                  "FixedView", "RowPart"],
                  "matrix_kronecker": ["MatrixKronecker"],
                  "point_swizzle": [f"Point{d}SwizzleTrait" for d in range(1, 7)]}
