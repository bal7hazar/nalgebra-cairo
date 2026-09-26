"""Templates of the functional and in-place variants (WP 8.2b, API_PARITY package P03), for the
36 shapes.

One template per method family, applied to every shape it concerns (`Shape`), like
`completion.py`; a shape that already has a method of that name keeps it (`missing`).

Families (upstream `src/base/*.rs`):

* functional (`matrix.rs`): `map`, `map_with_location`, `zip_map`, `zip_zip_map`, `fold`,
  `fold_with`, `zip_fold`, `apply`, `apply_into`, `zip_apply`, `zip_zip_apply`, `fill_with`
  (`edition.rs`), and on squares `map_diagonal`. The closures are `core::ops::Fn` values (like
  WP 8.2a's `from_fn`): a Cairo closure receives its arguments BY VALUE and cannot mutate what it
  captures, so upstream's `FnMut(&mut T)` of `apply` / `zip_apply` / `zip_zip_apply` becomes a
  closure RETURNING the new component (`Fn(T) -> T`), and `fold_with`'s `&T` arguments are values;
  the calls run in column-major order like upstream's iterators.
* edition (`edition.rs`, `matrix.rs`): `fill`, `fill_diagonal`, `fill_with_identity`, `fill_row`,
  `fill_column`, `fill_lower_triangle`, `fill_upper_triangle`, `set_row`, `set_column`,
  `set_diagonal`, `set_partial_diagonal`, `copy_from`, `copy_from_slice`, `tr_copy_from`, `swap`,
  `swap_rows`, `swap_columns`, and on squares `fill_lower_triangle_with_upper_triangle` /
  `fill_upper_triangle_with_lower_triangle`. A runtime index or shift selects ONE struct literal
  through a `match` (no loop, no per-component comparison).
* in place (`ref self`, Cairo's `&mut self`): `neg_mut`, `scale_mut`, `unscale_mut`,
  `normalize_mut`, `try_normalize_mut`, `set_magnitude`, `add_scalar_mut`, `component_mul_mut` /
  `component_div_mut` (deprecated upstream), `conjugate_mut`, on squares `transpose_mut`,
  `adjoint_mut`, `conjugate_transform_mut`; and the `*_to` forms writing into `ref out`:
  `transpose_to`, `adjoint_to`, `conjugate_transpose_to`, `add_to`, `sub_to` (`mul_to`,
  `tr_mul_to`, `ad_mul_to` are default methods of `MatrixMul` / `MatrixTrMul`, `shapes.py`). Every
  in-place form delegates to its by-value kernel (`self = self.op()`): bit-identical, and the
  by-value kernels are the measured ones.
* norms (`norm.rs`): `apply_norm` / `apply_metric_distance` through the `Norm<N, M, T>` trait of
  `base/norm.cairo`, implemented in each shape's module for the four markers (`EuclideanNorm` ->
  `norm` / `metric_distance`, `LpNorm` -> `lp_norm`, `OneNorm` -> `one_norm`, `UniformNorm` ->
  `amax`).
"""

import library as L
from completion import INDEX_ERROR, fn, item, vec
from model import Shape

CLOSURE = "F, +Drop<F>, impl Func: core::ops::Fn<F, ({args})>"
# A closure whose outputs are kept while the next calls run (they may panic): `Drop` needed.
MAPPING = CLOSURE + ", +Drop<Func::Output>"
CLOSURE_NOTE = ("`f` is any closure or `Fn` value; Cairo closures take their arguments by value and "
                "cannot mutate their captures.")


class Hinted(L.Fn):
    """A method with the `#[inline]` HINT: `#[inline(always)]` is refused on functions with impl
    generic parameters (the closures, `Norm`), and the hint is measured to remove their call
    (`bench_matrix3_apply_norm`: equal to a direct `norm()`)."""

    def attrs(self) -> list[str]:
        return ["#[inline]"]


def hinted(f: L.Fn) -> L.Fn:
    return Hinted(f.name, f.doc, f.sig, f.body, False)


def use_of(s: Shape) -> str:
    return f"super::{s.module}::{s.name}"


def row_of(s: Shape) -> Shape:
    return Shape(1, s.c)


def col_of(s: Shape) -> Shape:
    return Shape(s.r, 1)


def diag_of(s: Shape) -> Shape:
    return vec(min(s.r, s.c))


def chain(first: str, calls: list[str], ty: str) -> str:
    """`let acc: ty = f(first, x0).into(); ... f(acc, xn).into()`: one closure call per component,
    in the order of `calls` (argument tails)."""
    lines, acc = [], first
    for k, tail in enumerate(calls):
        call = f"f({acc}, {tail}).into()"
        if k == len(calls) - 1:
            lines.append(call)
        else:
            lines.append(f"let acc: {ty} = {call};")
            acc = "acc"
    return "\n".join(lines)


def match_arms(var: str, arms: list[str], default: str) -> str:
    return f"match {var} {{\n" + "".join(f"{k} => {a},\n" for k, a in enumerate(arms)) + \
        f"_ => {default},\n}}"


def panic() -> str:
    return f"core::panic_with_felt252({INDEX_ERROR})"


# --------------------------------------------------------------------------------------------
# Methods of `<S>Trait`
# --------------------------------------------------------------------------------------------


def methods(s: Shape) -> list[L.Fn]:
    S, T, F, n = s.name, f"{s.name}<T>", s.fields, s.n
    small = n <= 16
    lit = lambda g: L.lit(S, [(k, g(k)) for k in F])  # noqa: E731
    ij = {k: (i, j) for j in range(s.c) for i in range(s.r) for k in [s.f(i, j)]}
    out: list[L.Fn] = []

    # --- functional ----------------------------------------------------------------------------
    out.append(fn(
        "map", f"The {s.kind} of `f(x)` for every component `x` (called in column-major order). "
               f"{CLOSURE_NOTE} Upstream: `map`.",
        f"fn map<{MAPPING.format(args='T,')}>(self: {T}, f: F) -> {S}<Func::Output>",
        lit(lambda k: f"f(self.{k})"), False))
    out.append(fn(
        "map_with_location",
        f"The {s.kind} of `f(i, j, x)` for every component `x` at row `i`, column `j` (0-based, "
        f"column-major order). {CLOSURE_NOTE} Upstream: `map_with_location`.",
        f"fn map_with_location<{MAPPING.format(args='usize, usize, T')}>(self: {T}, f: F) -> "
        f"{S}<Func::Output>",
        lit(lambda k: f"f({ij[k][0]}, {ij[k][1]}, self.{k})"), False))
    out.append(fn(
        "zip_map", f"The {s.kind} of `f(a, b)` for the components `a` of `self` and `b` of `rhs` "
                   f"at the same position. {CLOSURE_NOTE} Upstream: `zip_map`.",
        f"fn zip_map<T2, +Copy<T2>, +Drop<T2>, {MAPPING.format(args='T, T2')}>(self: {T}, rhs: "
        f"{S}<T2>, f: F) -> {S}<Func::Output>",
        lit(lambda k: f"f(self.{k}, rhs.{k})"), False))
    out.append(fn(
        "zip_zip_map", f"The {s.kind} of `f(a, b, c)` for the components of `self`, `b` and `c` "
                       f"at the same position. {CLOSURE_NOTE} Upstream: `zip_zip_map`.",
        f"fn zip_zip_map<T2, T3, +Copy<T2>, +Drop<T2>, +Copy<T3>, +Drop<T3>, "
        f"{MAPPING.format(args='T, T2, T3')}>(self: {T}, b: {S}<T2>, c: {S}<T3>, f: F) -> "
        f"{S}<Func::Output>",
        lit(lambda k: f"f(self.{k}, b.{k}, c.{k})"), False))
    out.append(fn(
        "fold", f"`f(.. f(f(init, x0), x1) .., x{n - 1})` over the components in column-major "
                f"order. The closure's output converts `Into<Acc>` (the identity included): Cairo "
                f"cannot state `Output = Acc` without the `associated_item_constraints` "
                f"experimental feature. {CLOSURE_NOTE} Upstream: `fold`.",
        f"fn fold<Acc, {CLOSURE.format(args='Acc, T')}, +Into<Func::Output, Acc>>(self: {T}, "
        f"init: Acc, f: F) -> Acc",
        chain("init", [f"self.{k}" for k in F], "Acc"), False))
    out.append(fn(
        "fold_with",
        f"`init_f(Some(x0))`, then `f(acc, x)` over the other components in column-major order: "
        f"upstream's `fold_with` (`init_f` receives the first component, `None` only for an empty "
        f"matrix, which a static shape never is). The accumulator has the type of `init_f`'s "
        f"output; `f`'s output converts `Into` it. The closures receive values instead of "
        f"upstream's `&T`. Upstream: `fold_with`.",
        f"fn fold_with<G, +Drop<G>, impl Init: core::ops::Fn<G, (Option<T>,)>, "
        f"+Drop<Init::Output>, {CLOSURE.format(args='Init::Output, T')}, "
        f"+Into<Func::Output, Init::Output>>(self: {T}, init_f: G, f: F) -> Init::Output",
        (f"let acc: Init::Output = init_f(Option::Some(self.{F[0]}));\n" +
         (chain("acc", [f"self.{k}" for k in F[1:]], "Init::Output") if n > 1 else "acc")),
        False))
    out.append(fn(
        "zip_fold", f"`fold` over the pairs of components of `self` and `rhs` at the same "
                    f"position: `f(acc, a, b)`, column-major. {CLOSURE_NOTE} Upstream: `zip_fold`.",
        f"fn zip_fold<T2, +Copy<T2>, +Drop<T2>, Acc, {CLOSURE.format(args='Acc, T, T2')}, "
        f"+Into<Func::Output, Acc>>(self: {T}, rhs: {S}<T2>, init: Acc, f: F) -> Acc",
        chain("init", [f"self.{k}, rhs.{k}" for k in F], "Acc"), False))
    apply_note = ("Upstream's closure is `FnMut(&mut T)`, writing through the reference; a Cairo "
                  "closure cannot, so it RETURNS the new component (its output converts "
                  "`Into<T>`).")
    out.append(fn(
        "apply", f"Replaces every component `x` by `f(x)` (column-major order). {apply_note} "
                 f"Upstream: `apply`.",
        f"fn apply<{CLOSURE.format(args='T,')}, +Into<Func::Output, T>>(ref self: {T}, f: F)",
        f"self = {lit(lambda k: f'f(self.{k}).into()')};", False))
    out.append(fn(
        "apply_into", f"`self` with every component `x` replaced by `f(x)`: `apply` by value. "
                      f"{apply_note} Upstream: `apply_into`.",
        f"fn apply_into<{CLOSURE.format(args='T,')}, +Into<Func::Output, T>>(self: {T}, f: F) -> {T}",
        lit(lambda k: f"f(self.{k}).into()"), False))
    out.append(fn(
        "zip_apply", f"Replaces every component `a` by `f(a, b)`, `b` the component of `rhs` at "
                     f"the same position. {apply_note} Upstream: `zip_apply`.",
        f"fn zip_apply<T2, +Copy<T2>, +Drop<T2>, {CLOSURE.format(args='T, T2')}, "
        f"+Into<Func::Output, T>>(ref self: {T}, rhs: {S}<T2>, f: F)",
        f"self = {lit(lambda k: f'f(self.{k}, rhs.{k}).into()')};", False))
    out.append(fn(
        "zip_zip_apply", f"Replaces every component `a` by `f(a, b, c)`, `b` and `c` the "
                         f"components of `b` and `c` at the same position. {apply_note} "
                         f"Upstream: `zip_zip_apply`.",
        f"fn zip_zip_apply<T2, T3, +Copy<T2>, +Drop<T2>, +Copy<T3>, +Drop<T3>, "
        f"{CLOSURE.format(args='T, T2, T3')}, +Into<Func::Output, T>>(ref self: {T}, b: {S}<T2>, "
        f"c: {S}<T3>, f: F)",
        f"self = {lit(lambda k: f'f(self.{k}, b.{k}, c.{k}).into()')};", False))
    out.append(fn(
        "fill_with", f"Sets every component to `f()` (one call per component, column-major). The "
                     f"closure's output converts `Into<T>`. Upstream: `fill_with` (`impl Fn() -> "
                     f"T`).",
        f"fn fill_with<{CLOSURE.format(args='')}, +Into<Func::Output, T>>(ref self: {T}, f: F)",
        f"self = {lit(lambda k: 'f().into()')};", False))
    if s.is_square:
        v = col_of(s)
        out.append(fn(
            "map_diagonal", f"The `{v.name}` of `f(d)` for every diagonal component `d` (top to "
                            f"bottom): `self.diagonal().map(f)`. {CLOSURE_NOTE} Upstream: "
                            f"`map_diagonal`.",
            f"fn map_diagonal<{MAPPING.format(args='T,')}>(self: {T}, f: F) -> "
            f"{v.name}<Func::Output>",
            L.lit(v.name, [(v.f(i, 0), f"f(self.{s.f(i, i)})") for i in range(s.r)]), False))

    # --- edition -------------------------------------------------------------------------------
    def put(sel, val) -> str:
        """The literal of `self` with the components `sel(k)` replaced by `val(k)`."""
        return lit(lambda k: val(k) if sel(k) else f"self.{k}")

    out.append(fn("fill", "Sets every component to `val`. Upstream: `fill`.",
                  f"fn fill(ref self: {T}, val: T)", f"self = {lit(lambda k: 'val')};", small))
    out.append(fn("fill_diagonal", f"Sets the {min(s.r, s.c)} diagonal components to `val`, "
                                   f"the others unchanged. Upstream: `fill_diagonal`.",
                  f"fn fill_diagonal(ref self: {T}, val: T)",
                  f"self = {put(lambda k: ij[k][0] == ij[k][1], lambda k: 'val')};", small))
    out.append(fn("fill_with_identity", "Sets `self` to the identity: ones on the diagonal, zeros "
                                        "elsewhere (`fill(0)` then `fill_diagonal(1)`). Upstream: "
                                        "`fill_with_identity`.",
                  f"fn fill_with_identity(ref self: {T})",
                  f"self = {lit(lambda k: 'R::one()' if ij[k][0] == ij[k][1] else 'R::zero()')};",
                  small))
    out.append(fn(
        "fill_row", f"Sets the {s.c} components of row `i` to `val`. Panics with `nalgebra: "
                    f"index out of bounds` for `i >= {s.r}`. Upstream: `fill_row`.\n\nONE `match` on `i` "
                    f"selects the literal: measured 2.1 times cheaper than one comparison per "
                    f"component (`bench_matrix4_fill_row__alt_per_component`).",
        f"fn fill_row(ref self: {T}, i: usize, val: T)",
        "self = " + match_arms("i", [put(lambda k, r=r: ij[k][0] == r, lambda k: "val")
                                     for r in range(s.r)], panic()) + ";", small))
    out.append(fn(
        "fill_column", f"Sets the {s.r} components of column `j` to `val`. Panics with `nalgebra: "
                       f"index out of bounds` for `j >= {s.c}`. Upstream: `fill_column`.",
        f"fn fill_column(ref self: {T}, j: usize, val: T)",
        "self = " + match_arms("j", [put(lambda k, c=c: ij[k][1] == c, lambda k: "val")
                                     for c in range(s.c)], panic()) + ";", small))
    out.append(fn(
        "fill_lower_triangle",
        f"Sets every component `(i, j)` with `i >= j + shift` to `val`: the lower triangle with "
        f"the diagonal for `shift = 0`, without it for `shift = 1`, leaving `shift - 1` "
        f"subdiagonals as well above; nothing changes for `shift >= {s.r}`. ONE `match` on "
        f"`shift` selects the literal: measured 2.7 times cheaper than one threshold test per "
        f"component (`bench_matrix4_fill_lower_triangle__alt_per_component`). Upstream: "
        f"`fill_lower_triangle`.",
        f"fn fill_lower_triangle(ref self: {T}, val: T, shift: usize)",
        "self = " + match_arms("shift", [put(lambda k, h=h: ij[k][0] >= ij[k][1] + h,
                                             lambda k: "val") for h in range(s.r)], "self")
        + ";", small))
    out.append(fn(
        "fill_upper_triangle",
        f"Sets every component `(i, j)` with `j >= i + shift` to `val`: the upper triangle with "
        f"the diagonal for `shift = 0`, without it for `shift = 1`, leaving `shift - 1` "
        f"superdiagonals as well below; nothing changes for `shift >= {s.c}`. ONE `match` on "
        f"`shift` selects the literal. Upstream: `fill_upper_triangle`.",
        f"fn fill_upper_triangle(ref self: {T}, val: T, shift: usize)",
        "self = " + match_arms("shift", [put(lambda k, h=h: ij[k][1] >= ij[k][0] + h,
                                             lambda k: "val") for h in range(s.c)], "self")
        + ";", small))
    if s.is_square:
        out.append(fn(
            "fill_lower_triangle_with_upper_triangle",
            "Copies the strict upper triangle onto the strict lower one (`m[(i, j)] = m[(j, i)]` "
            "for `i > j`), making `self` symmetric. Exact. Upstream: "
            "`fill_lower_triangle_with_upper_triangle`.",
            f"fn fill_lower_triangle_with_upper_triangle(ref self: {T})",
            f"self = {put(lambda k: ij[k][0] > ij[k][1], lambda k: 'self.' + s.f(ij[k][1], ij[k][0]))};",
            small))
        out.append(fn(
            "fill_upper_triangle_with_lower_triangle",
            "Copies the strict lower triangle onto the strict upper one (`m[(i, j)] = m[(j, i)]` "
            "for `i < j`), making `self` symmetric. Exact. Upstream: "
            "`fill_upper_triangle_with_lower_triangle`.",
            f"fn fill_upper_triangle_with_lower_triangle(ref self: {T})",
            f"self = {put(lambda k: ij[k][0] < ij[k][1], lambda k: 'self.' + s.f(ij[k][1], ij[k][0]))};",
            small))
    rw, cl, dg = row_of(s), col_of(s), diag_of(s)
    out.append(fn(
        "set_row", f"Replaces row `i` by `row`, a `{rw.name}`. Panics with `nalgebra: index out "
                   f"of bounds` for `i >= {s.r}`. Upstream: `set_row`.",
        f"fn set_row(ref self: {T}, i: usize, row: {rw.name}<T>)",
        "self = " + match_arms("i", [put(lambda k, r=r: ij[k][0] == r,
                                         lambda k: f"row.{rw.f(0, ij[k][1])}")
                                     for r in range(s.r)], panic()) + ";", small))
    out.append(fn(
        "set_column", f"Replaces column `j` by `column`, a `{cl.name}`. Panics with `nalgebra: "
                      f"index out of bounds` for `j >= {s.c}`. Upstream: `set_column`.",
        f"fn set_column(ref self: {T}, j: usize, column: {cl.name}<T>)",
        "self = " + match_arms("j", [put(lambda k, c=c: ij[k][1] == c,
                                         lambda k: f"column.{cl.f(ij[k][0], 0)}")
                                     for c in range(s.c)], panic()) + ";", small))
    out.append(fn(
        "set_diagonal", f"Replaces the diagonal by `diag`, a `{dg.name}` (upstream requires the "
                        f"length `min(R, C)` = {dg.r}). Upstream: `set_diagonal`.",
        f"fn set_diagonal(ref self: {T}, diag: {dg.name}<T>)",
        f"self = {put(lambda k: ij[k][0] == ij[k][1], lambda k: f'diag.{dg.f(ij[k][0], 0)}')};",
        small))
    part = {k: f"if len > {ij[k][0]} {{\n*diag[{ij[k][0]}]\n}} else {{\nself.{k}\n}}"
            for k in F if ij[k][0] == ij[k][1]}
    out.append(fn(
        "set_partial_diagonal",
        f"Replaces the first `min(diag.len(), {dg.r})` diagonal components by the values of "
        f"`diag` (the extra values are ignored, like upstream's `take`), the others unchanged. "
        f"Upstream: `set_partial_diagonal` (an iterator; a `Span` here).",
        f"fn set_partial_diagonal(ref self: {T}, diag: Span<T>)",
        f"let len = diag.len();\nself = {put(lambda k: k in part, lambda k: part[k])};", small))
    out.append(fn("copy_from", f"Sets `self` to `other` (a `{S}`: upstream requires the same "
                               f"shape). Upstream: `copy_from`.",
                  f"fn copy_from(ref self: {T}, other: {T})", "self = other;"))
    out.append(fn(
        "copy_from_slice",
        f"Sets `self` to the {n} values of `slice` in column-major order (`from_column_slice`). "
        f"Panics with `nalgebra: wrong slice length` unless `slice.len() == {n}`. Upstream: "
        f"`copy_from_slice` (`&[T]`).",
        f"fn copy_from_slice(ref self: {T}, slice: Span<T>)",
        "self = Self::from_column_slice(slice);"))
    t = s.transposed()
    out.append(fn(
        "tr_copy_from", f"Sets `self` to the transpose of `other`, a `{t.name}`. Exact. Upstream: "
                        f"`tr_copy_from`.",
        f"fn tr_copy_from(ref self: {T}, other: {t.name}<T>)",
        f"self = {lit(lambda k: f'other.{t.f(ij[k][1], ij[k][0])}')};", small))
    out.append(fn(
        "swap", "Exchanges the components at `row_cols1` and `row_cols2` (`(row, column)`). "
                "Panics with `nalgebra: index out of bounds` when either is out of the shape. "
                "Upstream: `swap`.",
        f"fn swap(ref self: {T}, row_cols1: (usize, usize), row_cols2: (usize, usize))",
        f"let a = MatrixIndex::index(self, row_cols1);\nlet b = MatrixIndex::index(self, row_cols2);"
        f"\nself = {S}EditTrait::replace({S}EditTrait::replace(self, row_cols1, b), row_cols2, a);",
        False))
    out.append(fn(
        "swap_rows", f"Exchanges rows `irow1` and `irow2`. Panics with `nalgebra: index out of "
                     f"bounds` when either is `>= {s.r}`. Upstream: `swap_rows`.\n\nTwo reads and two "
                     f"writes of a row (one `match` each). ONE nested `match` on both rows is 940 gas "
                     f"(15%) cheaper on `Matrix3` (`bench_matrix3_swap_rows__alt_pair_match`) but "
                     f"generates R² whole-shape literals (about 40 000 lines over the 36 shapes, "
                     f"per method): kept as a benchmark.",
        f"fn swap_rows(ref self: {T}, irow1: usize, irow2: usize)",
        f"let a = {S}EditTrait::row_at(self, irow1);\nlet b = {S}EditTrait::row_at(self, irow2);\n"
        f"Self::set_row(ref self, irow1, b);\nSelf::set_row(ref self, irow2, a);", False))
    out.append(fn(
        "swap_columns", f"Exchanges columns `icol1` and `icol2`. Panics with `nalgebra: index out "
                        f"of bounds` when either is `>= {s.c}`. Upstream: `swap_columns`.",
        f"fn swap_columns(ref self: {T}, icol1: usize, icol2: usize)",
        f"let a = {S}EditTrait::column_at(self, icol1);\n"
        f"let b = {S}EditTrait::column_at(self, icol2);\n"
        f"Self::set_column(ref self, icol1, b);\nSelf::set_column(ref self, icol2, a);", False))

    # --- in place ------------------------------------------------------------------------------
    out.append(fn("neg_mut", "`self = -self`. Exact; panics on overflow (`-MIN`). Upstream: "
                             "`neg_mut`.", f"fn neg_mut(ref self: {T})", "self = -self;"))
    out.append(fn("scale_mut", "`self = self.scale(k)`, each component floored once. Panics on "
                               "overflow. Upstream: `scale_mut`.",
                  f"fn scale_mut(ref self: {T}, k: T)", "self = Self::scale(self, k);"))
    out.append(fn("unscale_mut", "`self = self.unscale(k)`, each component correctly rounded. "
                                 "Panics on a zero `k` and on overflow. Upstream: `unscale_mut`.",
                  f"fn unscale_mut(ref self: {T}, k: T)", "self = Self::unscale(self, k);"))
    out.append(fn("normalize_mut", "Normalizes `self` in place (`self = self.normalize()`, "
                                   "bit-identical) and returns the norm it had. Panics with a "
                                   "division by zero when the norm is zero. Upstream: "
                                   "`normalize_mut`.",
                  f"fn normalize_mut(ref self: {T}) -> T",
                  "let n = Self::norm(self);\nself = Self::unscale(self, n);\nn"))
    out.append(fn("try_normalize_mut", "Normalizes `self` in place and returns `Some` of the "
                                       "norm it had, or leaves it unchanged and returns `None` "
                                       "when that norm is `<= min_norm`. Upstream: "
                                       "`try_normalize_mut`.",
                  f"fn try_normalize_mut(ref self: {T}, min_norm: T) -> Option<T>",
                  "let n = Self::norm(self);\nif n <= min_norm {\nreturn None;\n}\n"
                  "self = Self::unscale(self, n);\nSome(n)"))
    out.append(fn("set_magnitude", "Scales `self` to the norm `magnitude`: `self.scale(magnitude "
                                   "/ norm)`, the ratio rounded to nearest (like "
                                   "`try_set_magnitude`). Panics with a division by zero when the "
                                   "norm is zero (upstream's floats give NaN). Upstream: "
                                   "`set_magnitude`.",
                  f"fn set_magnitude(ref self: {T}, magnitude: T)",
                  "let n = Self::norm(self);\nself = Self::scale(self, R::div(magnitude, n));"))
    out.append(fn("add_scalar_mut", "`self = self.add_scalar(k)`. Exact; panics on overflow. "
                                    "Upstream: `add_scalar_mut`.",
                  f"fn add_scalar_mut(ref self: {T}, k: T)", "self = Self::add_scalar(self, k);"))
    out.append(fn("component_mul_mut", "Alias of `component_mul_assign` (deprecated upstream). "
                                       "Upstream: `component_mul_mut`.",
                  f"fn component_mul_mut(ref self: {T}, rhs: {T})",
                  "Self::component_mul_assign(ref self, rhs);"))
    out.append(fn("component_div_mut", "Alias of `component_div_assign` (deprecated upstream). "
                                       "Upstream: `component_div_mut`.",
                  f"fn component_div_mut(ref self: {T}, rhs: {T})",
                  "Self::component_div_assign(ref self, rhs);"))
    out.append(fn("conjugate_mut", "Conjugates every component in place: nothing changes for a "
                                   "real scalar. Upstream: `conjugate_mut`.",
                  f"fn conjugate_mut(ref self: {T})", "self = Self::conjugate(self);"))
    if s.is_square:
        for name, what in (("transpose_mut", "`self = self.transpose()`. Exact."),
                           ("adjoint_mut", "`self = self.adjoint()`: the transpose for a real "
                                           "scalar. Exact."),
                           ("conjugate_transform_mut", "`adjoint_mut` (upstream's alias, kept for "
                                                       "complex matrices). Exact.")):
            out.append(fn(name, f"{what} Upstream: `{name}`.", f"fn {name}(ref self: {T})",
                          "self = Self::transpose(self);"))
    for name, what in (("transpose_to", "the transpose of `self`"),
                       ("adjoint_to", "the adjoint of `self` (its transpose for a real scalar)"),
                       ("conjugate_transpose_to", "the adjoint of `self` (deprecated upstream "
                                                  "alias of `adjoint_to`)")):
        out.append(fn(name, f"Writes {what} into `out`, a `{t.name}`. Exact. Upstream: `{name}`.",
                      f"fn {name}(self: {T}, ref out: {t.name}<T>)",
                      "out = Self::transpose(self);"))
    for name, op, what in (("add_to", "+", "sum"), ("sub_to", "-", "difference")):
        out.append(fn(name, f"Writes the {what} `self {op} rhs` into `out`. Exact; panics on "
                            f"overflow. Upstream: `{name}`.",
                      f"fn {name}(self: {T}, rhs: {T}, ref out: {T})", f"out = self {op} rhs;"))

    # --- norms ---------------------------------------------------------------------------------
    out.append(fn(
        "apply_norm", "The norm `norm` of `self`: `EuclideanNorm {}` (`norm`), `LpNorm { p }` "
                      "(`lp_norm(p)`), `OneNorm {}` (`one_norm`) or `UniformNorm {}` (`amax`), "
                      "through their `Norm` impls (static dispatch). Upstream: `apply_norm` "
                      "(`&impl Norm<T>`; the markers are `Copy` values here).",
        f"fn apply_norm<N, +Drop<N>, impl Nm: Norm<N, {T}, T>>(self: {T}, norm: N) -> T",
        "Nm::norm(@norm, self)", False))
    out.append(fn(
        "apply_metric_distance",
        "The distance between `self` and `rhs` in the norm `norm` (see `apply_norm`; the "
        "Euclidean one is the fused `metric_distance`, the others the norm of the exact "
        "difference). Upstream: `apply_metric_distance`.",
        f"fn apply_metric_distance<N, +Drop<N>, impl Nm: Norm<N, {T}, T>>(self: {T}, rhs: {T}, "
        f"norm: N) -> T",
        "Nm::metric_distance(@norm, self, rhs)", False))
    return [hinted(f) if not f.inline and f.name in HINTED else f for f in out]


# Methods with impl generic parameters or runtime positions, measured cheaper with the hint.
HINTED = {"map", "map_with_location", "zip_map", "zip_zip_map", "fold", "fold_with", "zip_fold",
          "apply", "apply_into", "zip_apply", "zip_zip_apply", "fill_with", "map_diagonal",
          "swap", "swap_rows", "swap_columns", "apply_norm", "apply_metric_distance"}


# The Scarb feature of the methods taking a closure (`map`, `fold`, `apply`, `zip_*`, `fill_with`,
# `map_diagonal`...): the costliest code of the library per line (WP 8.1d: -0.6 GB on every
# compilation unit for 12k lines), used by nothing else in the crate. In `default`.
FEATURE = "closures"


def takes_closure(f: L.Fn) -> bool:
    return "core::ops::Fn<" in f.sig


def missing(s: Shape, have: set[str]) -> list[L.Fn]:
    out = [f for f in methods(s) if f.name not in have]
    for f in out:
        if takes_closure(f):
            f.feature = FEATURE
    return out


# --------------------------------------------------------------------------------------------
# Top-level items
# --------------------------------------------------------------------------------------------


def items(s: Shape) -> list[str]:
    S, T, F = s.name, f"{s.name}<T>", s.fields
    rw, cl = row_of(s), col_of(s)
    small = s.n <= 16
    out = [L.section("functional and in-place variants", 100)]
    # Private helpers of `swap`, `swap_rows`, `swap_columns`.
    rows = [L.lit(rw.name, [(rw.f(0, j), f"self.{s.f(i, j)}") for j in range(s.c)])
            for i in range(s.r)]
    cols = [L.lit(cl.name, [(cl.f(i, 0), f"self.{s.f(i, j)}") for i in range(s.r)])
            for j in range(s.c)]
    arms = []
    for j in range(s.c):
        inner = [L.lit(S, [(k, "v" if k == s.f(i, j) else f"self.{k}") for k in F])
                 for i in range(s.r)]
        arms.append(match_arms("i", inner, panic()))
    helpers = [
        L.Fn("replace", "/// `self` with the component at `index` (`(row, column)`) replaced by "
                        "`v`; panics out of bounds.", f"fn replace(self: {T}, index: (usize, "
                        f"usize), v: T) -> {T}",
             "let (i, j) = index;\n" + match_arms("j", arms, panic()), small),
        L.Fn("row_at", f"/// Row `i`, a `{rw.name}`; panics out of bounds.",
             f"fn row_at(self: {T}, i: usize) -> {rw.name}<T>", match_arms("i", rows, panic()),
             small),
        L.Fn("column_at", f"/// Column `j`, a `{cl.name}`; panics out of bounds.",
             f"fn column_at(self: {T}, j: usize) -> {cl.name}<T>",
             match_arms("j", cols, panic()), small),
    ]
    for h in helpers:
        h.raw_doc = [h.doc]
    out.append("/// Private helpers of the `swap*` methods (runtime positions: one `match` each).\n"
               f"#[generate_trait]\nimpl {S}EditImpl<T, +Copy<T>, +Drop<T>> of {S}EditTrait<T> {{\n"
               + "\n\n".join(h.definition() for h in helpers) + "\n}")
    real = L.bounds(L.MATRIX_IMPL_BOUNDS)
    angle = L.bounds(L.EXTENDED_ANGLE_BOUNDS)
    for marker, bounds, norm, dist, what in (
            ("EuclideanNorm", real, f"{S}Trait::norm(m)", f"{S}Trait::metric_distance(m1, m2)",
             "`m.norm()` and the fused `metric_distance`"),
            ("LpNorm", angle, f"{S}AngleTrait::lp_norm(m, *self.p)",
             f"{S}AngleTrait::lp_norm(m1 - m2, *self.p)", "`m.lp_norm(p)`, of `m1 - m2` for the "
                                                          "distance"),
            ("OneNorm", real, f"{S}Trait::one_norm(m)", f"{S}Trait::one_norm(m1 - m2)",
             "`m.one_norm()`, of `m1 - m2` for the distance"),
            ("UniformNorm", real, f"{S}Trait::amax(m)", f"{S}Trait::amax(m1 - m2)",
             "`m.amax()`, of `m1 - m2` for the distance")):
        out.append(item(
            f"`{marker}` on `{S}`: {what}. Upstream: `Norm<T> for {marker}`.",
            f"pub impl {S}{marker}<\n{bounds}\n> of Norm<{marker}, {T}, T> {{\n{L.INLINE}\n"
            f"fn norm(self: @{marker}, m: {T}) -> T {{\n{norm}\n}}\n{L.INLINE}\n"
            f"fn metric_distance(self: @{marker}, m1: {T}, m2: {T}) -> T {{\n{dist}\n}}\n}}"))
    return out


def uses(s: Shape) -> list[str]:
    out = ["super::norm::{EuclideanNorm, LpNorm, Norm, OneNorm, UniformNorm}",
           "super::matrix_index::MatrixIndex"]
    for u in (row_of(s), col_of(s), diag_of(s), s.transposed()):
        if u != s:
            out.append(use_of(u))
    if s.is_square:
        out.append(use_of(col_of(s)))
    return [u for u in L.dedup(out) if u != use_of(s)]
