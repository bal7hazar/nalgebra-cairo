"""Templates of the static base completion (WP 8.2a, API_PARITY package P02), for the 36 shapes.

One template per method family, applied to every shape it concerns (`Shape`); a shape that already
has a method of the same name (the hand-written kernels of `library.py` and their
specialisations, `shapes.py`'s surface of the new shapes) keeps it, bit- and gas-identical, and
only receives the methods it lacks (`missing`).

Families (upstream `src/base/*.rs`):

* norms (`norm.rs`): `norm_squared`, `norm`, `magnitude(_squared)`, `metric_distance`, `dot`,
  `unscale` (`Div<T>`), `normalize`, `try_normalize`, `cap_magnitude`, `try_set_magnitude`,
  `one_norm`, and, with `Transcendental`, `lp_norm` and `angle` (`<S>AngleTrait`);
* component-wise (`componentwise.rs`): `component_mul/div` and their `_assign` forms, `inf`,
  `sup`, `inf_sup`, `add_scalar`, `cmpy`, `cdpy`;
* extrema (`min_max.rs`): `min`, `max`, `amin`, `amax`, `camin`, `camax`, `iamax_full`,
  `icamax_full`; on column vectors `imin`, `imax`, `iamin`, `iamax`, `icamax`, `argmin`,
  `argmax`;
* construction (`construction.rs`): `repeat`, `from_element`, `from_fn` (a closure),
  `from_row_slice`, `from_column_slice`, `from_partial_diagonal` (`Span`), `is_zero`; on column
  vectors the axes `x() .. b()`, `ith`, `ith_axis`;
* structure (`matrix.rs`): `adjoint`, `conjugate`, `conjugate_transpose`, `cast`, `try_cast`,
  `relative_eq`, `ulps_eq`; on squares `symmetric_part`, `hermitian_part`, `to_homogeneous`; on
  column vectors `push`, `to_homogeneous`, `from_homogeneous`, and with `Transcendental` `slerp`;
* items: `IndexView<usize>` / `IndexView<(usize, usize)>` and the generic `MatrixIndex`
  (`get`, `index`), `PartialOrd`, `Bounded`, `One` on squares, `[[T; R]; C]` conversions, and
  on column vectors the `Unit<VectorN>` traits (`cast`, axes, `slerp`, `try_slerp`).

Kernels (AGENTS.md rule 4): every sum of products or of squares is ONE fused kernel per output
scalar (`sum_prodK` / `normK` / `norm_squaredK` up to 4 terms, one `Real::Wide` chain above);
divisions by a common divisor go through the prepared-divisor `Real::divN` (bit-identical to `N`
`Real::div`), chunked to the fewest calls.
"""

import textwrap

import library as L
from model import COORDS, Shape

INDEX_ERROR = "errors::INDEX_OUT_OF_BOUNDS"
DIV_SIZES = (16, 9, 6, 5, 4, 3, 1)  # `Real::divN` (1 = `Real::div`)

ANGLE_BOUNDS = ["T", "impl R: Real<T>", "impl Tr: Transcendental<T>", "+Copy<T>", "+Drop<T>",
                "+Drop<R::Wide>", "+Add<T>", "+Sub<T>", "+Mul<T>", "+Neg<T>", "+PartialEq<T>",
                "+PartialOrd<T>"]


def wrap(text: str, indent: int = 4) -> str:
    width = 100 - indent - 4
    paras = [textwrap.fill(p, width, break_long_words=False, break_on_hyphens=False)
             for p in text.strip().split("\n\n")]
    return "\n\n".join(paras)


def fn(name: str, doc: str, sig: str, body: str, inline: bool = True) -> L.Fn:
    return L.Fn(name, wrap(doc, 4), sig, body, inline)


def item(doc: str, code: str) -> str:
    return "\n".join(f"/// {line}" if line else "///" for line in wrap(doc, 0).split("\n")) + \
        "\n" + code


def vec(n: int) -> Shape:
    return Shape(n, 1)


# --------------------------------------------------------------------------------------------
# Kernels
# --------------------------------------------------------------------------------------------


def squares(terms: list[str], final: str) -> str:
    """The sum of the squares of `terms`, `final` = `rescale` (floored) or `sqrt` (square root of
    the unscaled exact sum, floored once)."""
    k = len(terms)
    if k == 1:
        return f"{terms[0]} * {terms[0]}" if final == "rescale" else f"R::abs({terms[0]})"
    if k <= 4:
        return f"R::{'norm_squared' if final == 'rescale' else 'norm'}{k}({', '.join(terms)})"
    return L.wide_chain(terms, f"wide_{final}")


def products(pairs: list[tuple[str, str]]) -> str:
    """ONE fused sum of products (floored once)."""
    k = len(pairs)
    if k == 1:
        return f"{pairs[0][0]} * {pairs[0][1]}"
    if k <= 4:
        return f"R::sum_prod{k}({', '.join(f'{a}, {b}' for a, b in pairs)})"
    return L.wide_chain(pairs, "wide_rescale")


def div_chunks(n: int) -> list[int]:
    """The fewest `Real::divN` calls covering `n` quotients (larger chunks first)."""
    best: dict[int, list[int]] = {0: []}
    for m in range(1, n + 1):
        options = [best[m - d] + [d] for d in DIV_SIZES if d <= m]
        best[m] = min(options, key=lambda c: (len(c), [-x for x in sorted(c, reverse=True)]))
    return sorted(best[n], reverse=True)


def quotients(s: Shape, src: str, divisor: str) -> str:
    """`let` statements binding each field name to `src.<field> / divisor`, correctly rounded."""
    lines, fields, i = [], s.fields, 0
    for d in div_chunks(s.n):
        chunk = fields[i:i + d]
        i += d
        if d == 1:
            lines.append(f"let {chunk[0]} = R::div({src}.{chunk[0]}, {divisor});")
        else:
            args = ", ".join(f"{src}.{f}" for f in chunk)
            lines.append(f"let ({', '.join(chunk)}) = R::div{d}({args}, {divisor});")
    return "\n".join(lines)


def shorthand(s: Shape) -> str:
    return L.lit(s.name, [(f, f) for f in s.fields])


def fold(op: str, xs: list[str]) -> str:
    return L.fold(op, xs) if len(xs) > 1 else xs[0]


def scan(s: Shape, value, cmp: str, pair: bool) -> str:
    """Index of the first extremum of `value(field)` over the components in storage order
    (column-major), a strict comparison `cmp` keeping the first one on ties (upstream's loops)."""
    idx = (lambda i, j: f"({i}, {j})") if pair else (lambda i, j: f"{i + s.r * j}")
    order = [(i, j) for j in range(s.c) for i in range(s.r)]
    if len(order) == 1:
        return idx(0, 0)
    first = s.f(*order[0])
    lines = [f"let mut best: {'(usize, usize)' if pair else 'usize'} = {idx(0, 0)};",
             f"let {'mut ' if len(order) > 2 else ''}m = {value(first)};"]
    for k, (i, j) in enumerate(order[1:], 1):
        lines.append(f"let v = {value(s.f(i, j))};")
        update = "" if k == len(order) - 1 else "m = v;\n"
        lines.append(f"if v {cmp} m {{\n{update}best = {idx(i, j)};\n}}")
    lines.append("best")
    return "\n".join(lines)


def argscan(s: Shape, cmp: str) -> str:
    """`(index, value)` of the first extremum of a column vector (upstream `argmin` / `argmax`)."""
    xs = s.fields
    if len(xs) == 1:
        return f"(0, self.{xs[0]})"
    lines = ["let mut i: usize = 0;", f"let mut m = self.{xs[0]};"]
    for k, x in enumerate(xs[1:], 1):
        lines.append(f"if self.{x} {cmp} m {{\nm = self.{x};\ni = {k};\n}}")
    lines.append("(i, m)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Methods of `<S>Trait` (Real scalar)
# --------------------------------------------------------------------------------------------


def methods(s: Shape) -> list[L.Fn]:
    S, T, n, F = s.name, f"{s.name}<T>", s.n, s.fields
    small = n <= 16  # component-wise bodies: inlined up to 16 components, like `Matrix4`
    lit = lambda g: L.lit(S, [(k, g(k)) for k in F])  # noqa: E731
    wide = n > 4
    norm = "Self::norm(self)" if wide else squares([f"self.{k}" for k in F], "sqrt")
    out: list[L.Fn] = []

    # --- construction --------------------------------------------------------------------------
    out.append(fn("repeat", f"The {s.kind} whose components all equal `elem`. Upstream: "
                            f"`{S}::repeat`.", f"fn repeat(elem: T) -> {T}", lit(lambda k: "elem")))
    out.append(fn("from_element", f"Alias of `repeat`. Upstream: `{S}::from_element`.",
                  f"fn from_element(elem: T) -> {T}", lit(lambda k: "elem")))
    calls = {s.f(i, j): f"f({i}, {j}).into()" for i in range(s.r) for j in range(s.c)}
    out.append(fn(
        "from_fn",
        f"The {s.kind} whose component `(i, j)` (row, column, 0-based) is `f(i, j)`, `f` being "
        f"called in column-major order like upstream. `f` is any closure or `Fn` value of `(usize, "
        f"usize)` whose output converts `Into<T>` (the identity included): Cairo cannot state "
        f"`Output = T` on the closure without the `associated_item_constraints` experimental "
        f"feature. Upstream: `{S}::from_fn`.",
        f"fn from_fn<F, +Drop<F>, impl Func: core::ops::Fn<F, (usize, usize)>, "
        f"+Into<Func::Output, T>, +Drop<Func::Output>>(f: F) -> {T}",
        L.lit(S, [(k, calls[k]) for k in F]), False))
    for name, order, what in (("from_row_slice", s.row_major, "row-major"),
                              ("from_column_slice", F, "column-major")):
        pos = {k: order.index(k) for k in F}
        out.append(fn(
            name,
            f"The {s.kind} of the {n} values of `data`, in {what} order. Panics with "
            f"`nalgebra: wrong slice length` unless `data.len() == {n}`. Upstream: "
            f"`{S}::{name}` (`&[T]`).",
            f"fn {name}(data: Span<T>) -> {T}",
            f"if data.len() != {n} {{\ncore::panic_with_felt252(errors::SLICE_LENGTH);\n}}\n"
            + L.lit(S, [(k, f"*data[{pos[k]}]") for k in F]), small))
    m = min(s.r, s.c)
    diag = {s.f(i, i): (f"if len > {i} {{\n*data[{i}]\n}} else {{\nR::zero()\n}}") for i in range(m)}
    out.append(fn(
        "from_partial_diagonal",
        f"The {s.kind} whose first `data.len()` diagonal components are `data`, every other "
        f"component zero. Panics with `nalgebra: diagonal too long` when `data.len() > "
        f"{m}`. Upstream: `{S}::from_partial_diagonal` (`&[T]`).",
        f"fn from_partial_diagonal(data: Span<T>) -> {T}",
        f"let len = data.len();\nif len > {m} {{\ncore::panic_with_felt252(errors::TOO_MANY_DIAGONAL);"
        f"\n}}\n" + L.lit(S, [(k, diag.get(k, "R::zero()")) for k in F]), small))
    if s.is_column:
        for i, a in enumerate(COORDS[:s.r]):
            unit = tuple("1" if j == i else "0" for j in range(s.r))
            out.append(fn(a, f"The unit axis `{L.tup(unit)}`. Upstream: `{S}::{a}` (`{a}_axis` "
                             f"is the `Unit` form).", f"fn {a}() -> {T}",
                          lit(lambda k, a=a: "R::one()" if k == a else "R::zero()")))
        out.append(fn(
            "ith", f"The vector whose component `i` is `val`, every other one zero. Panics with "
                   f"`nalgebra: index out of bounds` for `i >= {s.r}`. Upstream: `{S}::ith`.",
            f"fn ith(i: usize, val: T) -> {T}",
            "match i {\n" + "".join(f"{k} => {lit(lambda f, k=k: 'val' if f == F[k] else 'R::zero()')},\n"
                                    for k in range(s.r))
            + f"_ => core::panic_with_felt252({INDEX_ERROR}),\n}}", small))
        out.append(fn(
            "ith_axis", f"The unit vector along axis `i` (`ith(i, 1)`). Panics with `nalgebra: "
                        f"index out of bounds` for `i >= {s.r}`. Upstream: `{S}::ith_axis`.",
            f"fn ith_axis(i: usize) -> Unit<{T}>", "Unit { value: Self::ith(i, R::one()) }"))
    out.append(fn("is_zero", "`true` when every component is zero. Upstream: `Zero::is_zero`.",
                  f"fn is_zero(self: {T}) -> bool", " && ".join(f"self.{k} == R::zero()" for k in F),
                  small))

    # --- component-wise ------------------------------------------------------------------------
    out.append(fn("component_mul", "Component-wise (Hadamard) product, each component floored "
                                   "once. Panics on overflow. Upstream: `component_mul`.",
                  f"fn component_mul(self: {T}, rhs: {T}) -> {T}",
                  lit(lambda k: f"self.{k} * rhs.{k}"), small))
    out.append(fn("component_mul_assign", "`self = self.component_mul(rhs)`. Upstream: "
                                          "`component_mul_assign`.",
                  f"fn component_mul_assign(ref self: {T}, rhs: {T})",
                  f"self = {lit(lambda k: f'self.{k} * rhs.{k}')};", small))
    out.append(fn("component_div", "Component-wise quotient, each component rounded to nearest "
                                   "(ties to even). Panics on a zero component of `rhs` and on "
                                   "overflow. Upstream: `component_div`.",
                  f"fn component_div(self: {T}, rhs: {T}) -> {T}",
                  lit(lambda k: f"R::div(self.{k}, rhs.{k})"), small))
    out.append(fn("component_div_assign", "`self = self.component_div(rhs)`. Upstream: "
                                          "`component_div_assign`.",
                  f"fn component_div_assign(ref self: {T}, rhs: {T})",
                  f"self = {lit(lambda k: f'R::div(self.{k}, rhs.{k})')};", small))
    for name, op, what in (("inf", "min", "minimum (infimum)"), ("sup", "max", "maximum (supremum)")):
        out.append(fn(name, f"Component-wise {what}. Exact. Upstream: `{name}`.",
                      f"fn {name}(self: {T}, other: {T}) -> {T}",
                      lit(lambda k, op=op: f"R::{op}(self.{k}, other.{k})"), small))
    out.append(fn("inf_sup", "`(self.inf(other), self.sup(other))`. Exact. Upstream: `inf_sup`.",
                  f"fn inf_sup(self: {T}, other: {T}) -> ({T}, {T})",
                  "(Self::inf(self, other), Self::sup(self, other))", small))
    out.append(fn("add_scalar", "`self + k` added to every component. Exact; panics on overflow. "
                                "Upstream: `add_scalar`.", f"fn add_scalar(self: {T}, k: T) -> {T}",
                  lit(lambda k: f"self.{k} + k"), small))
    out.append(fn(
        "cmpy",
        "`self = alpha * a ∘ b + beta * self` (component-wise product): per component `alpha * a` "
        "is floored, then the two products are ONE fused `sum_prod2` (floored once). Panics on "
        "overflow. Upstream: `cmpy` (which skips reading `self` when `beta` is zero: a "
        "difference only for NaN, which fixed point has not).",
        f"fn cmpy(ref self: {T}, alpha: T, a: {T}, b: {T}, beta: T)",
        f"self = {lit(lambda k: f'R::sum_prod2(alpha * a.{k}, b.{k}, beta, self.{k})')};", small))
    out.append(fn(
        "cdpy",
        "`self = alpha * a / b + beta * self` (component-wise quotient): per component `alpha * "
        "a` is floored, divided by `b` (rounded to nearest), then `beta * self + quotient` is ONE "
        "`mul_add` (floored once). Panics on a zero component of `b` and on overflow. Upstream: "
        "`cdpy`.",
        f"fn cdpy(ref self: {T}, alpha: T, a: {T}, b: {T}, beta: T)",
        f"self = {lit(lambda k: f'R::mul_add(beta, self.{k}, R::div(alpha * a.{k}, b.{k}))')};",
        small))

    # --- extrema -------------------------------------------------------------------------------
    xs = [f"self.{k}" for k in F]
    axs = [f"R::abs(self.{k})" for k in F]
    out.append(fn("min", "The smallest component. Exact. Upstream: `min`.",
                  f"fn min(self: {T}) -> T", fold("R::min", xs), small))
    out.append(fn("max", "The largest component. Exact. Upstream: `max`.",
                  f"fn max(self: {T}) -> T", fold("R::max", xs), small))
    out.append(fn("amin", "The smallest absolute value of a component. Panics on the scalar's "
                          "`MIN`. Upstream: `amin`.", f"fn amin(self: {T}) -> T",
                  fold("R::min", axs), small))
    out.append(fn("amax", "The largest absolute value of a component (the uniform norm). Panics "
                          "on the scalar's `MIN`. Upstream: `amax`.", f"fn amax(self: {T}) -> T",
                  fold("R::max", axs), small))
    out.append(fn("camin", "`amin`: the modulus of a real scalar is its absolute value. Upstream: "
                           "`camin`.", f"fn camin(self: {T}) -> T", fold("R::min", axs), small))
    out.append(fn("camax", "`amax`: the modulus of a real scalar is its absolute value. Upstream: "
                           "`camax`.", f"fn camax(self: {T}) -> T", fold("R::max", axs), small))
    out.append(fn("iamax_full", "`(row, column)` of the component with the largest absolute "
                                "value, the first one in column-major order on ties. Panics on "
                                "the scalar's `MIN`. Upstream: `iamax_full`.",
                  f"fn iamax_full(self: {T}) -> (usize, usize)",
                  scan(s, lambda k: f"R::abs(self.{k})", ">", True), n <= 6))
    out.append(fn("icamax_full", "`iamax_full`: the modulus of a real scalar is its absolute "
                                 "value. Upstream: `icamax_full`.",
                  f"fn icamax_full(self: {T}) -> (usize, usize)", "Self::iamax_full(self)"))
    if s.is_column:
        out.append(fn("argmin", "`(index, value)` of the smallest component, the first one on "
                                "ties. Exact. Upstream: `argmin`.",
                      f"fn argmin(self: {T}) -> (usize, T)", argscan(s, "<")))
        out.append(fn("argmax", "`(index, value)` of the largest component, the first one on "
                                "ties. Exact. Upstream: `argmax`.",
                      f"fn argmax(self: {T}) -> (usize, T)", argscan(s, ">")))
        out.append(fn("imin", "Index of the smallest component, the first one on ties. Upstream: "
                              "`imin`.", f"fn imin(self: {T}) -> usize",
                      "let (i, _) = Self::argmin(self);\ni"))
        out.append(fn("imax", "Index of the largest component, the first one on ties. Upstream: "
                              "`imax`.", f"fn imax(self: {T}) -> usize",
                      "let (i, _) = Self::argmax(self);\ni"))
        out.append(fn("iamin", "Index of the component with the smallest absolute value, the "
                               "first one on ties. Panics on the scalar's `MIN`. Upstream: `iamin`.",
                      f"fn iamin(self: {T}) -> usize",
                      scan(s, lambda k: f"R::abs(self.{k})", "<", False)))
        out.append(fn("iamax", "Index of the component with the largest absolute value, the "
                               "first one on ties. Panics on the scalar's `MIN`. Upstream: `iamax`.",
                      f"fn iamax(self: {T}) -> usize",
                      scan(s, lambda k: f"R::abs(self.{k})", ">", False)))
        out.append(fn("icamax", "`iamax`: the modulus of a real scalar is its absolute value. "
                                "Upstream: `icamax`.", f"fn icamax(self: {T}) -> usize",
                      "Self::iamax(self)"))

    # --- norms ---------------------------------------------------------------------------------
    out.append(fn("dot", "Dot product (the sum of the component-wise products, upstream's "
                         "Frobenius inner product for matrices): the exact sum is floored ONCE, "
                         "only the result must fit. Upstream: `dot`.",
                  f"fn dot(self: {T}, rhs: {T}) -> T",
                  products([(f"self.{k}", f"rhs.{k}") for k in F]), n <= 6))
    sq = squares(xs, "rescale")
    out.append(fn("norm_squared", "Squared Euclidean (Frobenius) norm: the exact sum of squares "
                                  "floored once. Panics on overflow (above a norm of about 46 340 "
                                  "in Q32.32 only `norm` works). Upstream: `norm_squared`.",
                  f"fn norm_squared(self: {T}) -> T", sq, n <= 6))
    out.append(fn("norm", "Euclidean (Frobenius) norm: square root of the UNSCALED exact sum of "
                          "squares, floored once. No intermediate overflow: only the result must "
                          "fit. Upstream: `norm`.",
                  f"fn norm(self: {T}) -> T", squares(xs, "sqrt"), n <= 6))
    out.append(fn("magnitude_squared", "Alias of `norm_squared`. Upstream: `magnitude_squared`.",
                  f"fn magnitude_squared(self: {T}) -> T", "Self::norm_squared(self)"))
    out.append(fn("magnitude", "Alias of `norm`. Upstream: `magnitude`.",
                  f"fn magnitude(self: {T}) -> T", "Self::norm(self)"))
    out.append(fn("metric_distance", "`(self - rhs).norm()`: the differences are exact, then one "
                                     "fused norm. Panics when a difference or the result "
                                     "overflows. Upstream: `metric_distance`.",
                  f"fn metric_distance(self: {T}, rhs: {T}) -> T",
                  squares([f"self.{k} - rhs.{k}" for k in F], "sqrt"), n <= 6))
    out.append(fn(
        "unscale", f"`self / k`, each component the correctly rounded quotient (nearest, ties to "
                   f"even), through {len(div_chunks(n))} prepared-divisor `Real::divN` call(s), "
                   f"bit-identical to one `Real::div` per component. Panics on a zero `k` and on "
                   f"overflow. Upstream: `unscale` (`self / k`).",
        f"fn unscale(self: {T}, k: T) -> {T}", quotients(s, "self", "k") + "\n" + shorthand(s),
        n <= 6))
    out.append(fn("normalize", "`self / self.norm()`: the floored norm, then `unscale`. Panics "
                               "with a division by zero when the norm is zero, and on overflow "
                               "when the norm does not fit. Upstream: `normalize`.",
                  f"fn normalize(self: {T}) -> {T}", f"Self::unscale(self, {norm})", n <= 6))
    out.append(fn("try_normalize", "`Some(self.normalize())`, or `None` when the norm is `<= "
                                   "min_norm` (never divides by zero for `min_norm >= 0`). "
                                   "Upstream: `try_normalize`.",
                  f"fn try_normalize(self: {T}, min_norm: T) -> Option<{T}>",
                  f"let n = {norm};\nif n <= min_norm {{\nNone\n}} else {{\n"
                  f"Some(Self::unscale(self, n))\n}}", n <= 6))
    out.append(fn("cap_magnitude", "`self` when its norm is `<= max`, otherwise `self.scale(max / "
                                   "norm)` (the ratio rounded to nearest, like upstream's `max / "
                                   "n`). Panics only when the norm does not fit. Upstream: "
                                   "`cap_magnitude`.",
                  f"fn cap_magnitude(self: {T}, max: T) -> {T}",
                  f"let n = {norm};\nif n <= max {{\nself\n}} else {{\n"
                  f"Self::scale(self, R::div(max, n))\n}}", n <= 6))
    out.append(fn("try_set_magnitude", "Scales `self` to the norm `magnitude` (`self.scale(magnitude "
                                       "/ norm)`, the ratio rounded to nearest) when its norm is "
                                       "`> min_magnitude`, leaves it unchanged otherwise. Upstream: "
                                       "`try_set_magnitude` (`&mut self`).",
                  f"fn try_set_magnitude(ref self: {T}, magnitude: T, min_magnitude: T)",
                  f"let n = Self::norm(self);\nif n > min_magnitude {{\n"
                  f"self = Self::scale(self, R::div(magnitude, n));\n}}", n <= 6))
    cols = [" + ".join(f"R::abs(self.{s.f(i, j)})" for i in range(s.r)) for j in range(s.c)]
    out.append(fn("one_norm", "The induced 1-norm: the largest absolute column sum (the L1 norm "
                              "of a column vector, the largest absolute value of a row vector). "
                              "Exact; panics on overflow. Upstream: `one_norm`.",
                  f"fn one_norm(self: {T}) -> T", fold("R::max", cols), small))

    # --- structure -----------------------------------------------------------------------------
    t = s.transposed()
    out.append(fn("adjoint", f"The conjugate transpose, a `{t.name}`: the transpose for a real "
                             f"scalar. Exact. Upstream: `adjoint`.",
                  f"fn adjoint(self: {T}) -> {t.name}<T>", "Self::transpose(self)"))
    out.append(fn("conjugate_transpose", f"Alias of `adjoint` (deprecated upstream). Upstream: "
                                         f"`conjugate_transpose`.",
                  f"fn conjugate_transpose(self: {T}) -> {t.name}<T>", "Self::transpose(self)"))
    out.append(fn("conjugate", "The component-wise conjugate: `self` for a real scalar. Upstream: "
                               "`conjugate`.", f"fn conjugate(self: {T}) -> {T}", "self"))
    if s.is_square:
        half = {}
        for j in range(s.c):
            for i in range(s.r):
                if i == j:
                    half[s.f(i, j)] = f"self.{s.f(i, j)}"
                elif i > j:
                    half[s.f(i, j)] = f"a{i + 1}{j + 1}"
                else:
                    half[s.f(i, j)] = f"a{j + 1}{i + 1}"
        lets = "".join(f"let a{i + 1}{j + 1} = R::sum_prod2(self.{s.f(i, j)}, R::HALF, "
                       f"self.{s.f(j, i)}, R::HALF);\n"
                       for j in range(s.c) for i in range(s.r) if i > j)
        body = lets + L.lit(S, [(k, half[k]) for k in F])
        out.append(fn("symmetric_part", "`(self + selfᵀ) / 2`: each off-diagonal pair is ONE "
                                        "`sum_prod2` by `1/2` (floored once, no intermediate "
                                        "overflow), the diagonal is copied exactly. Upstream: "
                                        "`symmetric_part`.",
                      f"fn symmetric_part(self: {T}) -> {T}", body, small))
        out.append(fn("hermitian_part", "`symmetric_part`: the adjoint of a real matrix is its "
                                        "transpose. Upstream: `hermitian_part`.",
                      f"fn hermitian_part(self: {T}) -> {T}", "Self::symmetric_part(self)"))
        if 2 <= s.r <= 5:
            u = Shape(s.r + 1, s.r + 1)
            vals = {u.f(i, j): (f"self.{s.f(i, j)}" if i < s.r and j < s.r else
                                "R::one()" if i == j else "R::zero()")
                    for i in range(u.r) for j in range(u.c)}
            out.append(fn("to_homogeneous", f"The `{u.name}` `[[self, 0], [0, 1]]`: `self` as the "
                                            f"linear part of a homogeneous transformation. Exact. "
                                            f"Upstream: `to_homogeneous`.",
                          f"fn to_homogeneous(self: {T}) -> {u.name}<T>",
                          L.lit(u.name, [(k, vals[k]) for k in u.fields]), small))
    if s.is_column and s.r <= 5:
        u = vec(s.r + 1)
        last = u.fields[-1]
        out.append(fn("push", f"The `{u.name}` of the components of `self` followed by `val`. "
                              f"Upstream: `push`.", f"fn push(self: {T}, val: T) -> {u.name}<T>",
                      L.lit(u.name, [(k, "val" if k == last else f"self.{k}") for k in u.fields])))
        if s.r >= 2:
            out.append(fn("to_homogeneous", f"`self` followed by `0`, a `{u.name}`: the "
                                            f"homogeneous coordinates of a vector (as opposed to a "
                                            f"point). Upstream: `to_homogeneous`.",
                          f"fn to_homogeneous(self: {T}) -> {u.name}<T>",
                          L.lit(u.name, [(k, "R::zero()" if k == last else f"self.{k}")
                                         for k in u.fields])))
        out.append(fn("from_homogeneous", f"The first {s.r} components of `v` when its last one "
                                          f"is zero (a homogeneous VECTOR), `None` otherwise. "
                                          f"Exact. Upstream: `{S}::from_homogeneous`.",
                      f"fn from_homogeneous(v: {u.name}<T>) -> Option<{T}>",
                      f"if v.{last} == R::zero() {{\nSome({L.lit(S, [(k, f'v.{k}') for k in F])})\n}}"
                      f" else {{\nNone\n}}"))
    out.append(fn("cast", "The same shape with every component converted by `Into<T, U>`. With "
                          "the single scalar of this library (`Fixed`) it is the identity; it "
                          "exists for scalar-generic code. Upstream: `cast` (and "
                          "`SubsetOf<Matrix<U>>`, the `nalgebra::convert` it goes through).",
                  f"fn cast<U, +Into<T, U>, +Drop<U>>(self: {T}) -> {S}<U>",
                  lit(lambda k: f"self.{k}.into()"), False))
    tries = "\n".join(f"let {k}: U = match self.{k}.try_into() {{\nOption::Some(v) => v,\n"
                      f"Option::None => {{\nreturn Option::None;\n}},\n}};" for k in F)
    out.append(fn("try_cast", "`Some` of the shape with every component converted by "
                              "`TryInto<T, U>`, `None` as soon as one conversion fails. Upstream: "
                              "`try_cast`.",
                  f"fn try_cast<U, +TryInto<T, U>, +Drop<U>>(self: {T}) -> Option<{S}<U>>",
                  f"{tries}\nOption::Some({shorthand(s)})", False))
    out.append(fn(
        "relative_eq",
        "`true` when every component is within `epsilon` ulp of `other`'s, or has the same sign "
        "and lies within `max_relative` times the larger magnitude of the two (`|a - b| <= "
        "max(|a|, |b|) · max_relative`). Panics on a component equal to the scalar's `MIN`, and "
        "on overflow of that product (only possible with `max_relative > 1`). Upstream: "
        "`approx::RelativeEq::relative_eq`, `epsilon` counted in ulp instead of a float epsilon "
        "(DESIGN D3).",
        f"fn relative_eq(self: {T}, other: {T}, epsilon: u64, max_relative: T) -> bool",
        "\n&& ".join(f"ApproxEqTrait::relative_eq(self.{k}, other.{k}, epsilon, max_relative)"
                     for k in F), small))
    out.append(fn(
        "ulps_eq",
        "`true` when every component is within `epsilon` ulp of `other`'s, or has the same sign "
        "and lies within `max_ulps` ulp (in fixed point the distance in ulp IS the raw "
        "difference; the `max_ulps` budget does not cross zero, like upstream's float `ulps_eq`). "
        "Cannot overflow. Upstream: `approx::UlpsEq::ulps_eq`.",
        f"fn ulps_eq(self: {T}, other: {T}, epsilon: u64, max_ulps: u32) -> bool",
        "\n&& ".join(f"ApproxEqTrait::ulps_eq(self.{k}, other.{k}, epsilon, max_ulps)" for k in F),
        small))
    return out


def missing(s: Shape, have: set[str]) -> list[L.Fn]:
    """The methods of `methods(s)` that the shape does not have yet."""
    return [f for f in methods(s) if f.name not in have]


# --------------------------------------------------------------------------------------------
# `<S>AngleTrait`: the methods that need `Transcendental`
# --------------------------------------------------------------------------------------------


def angle_methods(s: Shape) -> list[L.Fn]:
    S, T, F = s.name, f"{s.name}<T>", s.fields
    Tr = f"{S}Trait"
    out = []
    out.append(fn(
        "angle",
        "The angle between `self` and `other` seen as vectors of the Frobenius inner product, in "
        "`[0, π]` (up to the rounding of `atan2`); `0` when one of them is zero. Computed as `2 * "
        "atan2(|u - v|, |u + v|)` on the normalized `u`, `v` (Kahan): unlike upstream's `acos(dot "
        "/ (|a| * |b|))` it cannot overflow on long inputs and stays accurate for nearly parallel "
        "ones. Panics when a norm does not fit. Upstream: `angle`.",
        f"fn angle(self: {T}, other: {T}) -> T",
        f"let n1 = {Tr}::norm(self);\nlet n2 = {Tr}::norm(other);\n"
        f"if n1 == R::zero() || n2 == R::zero() {{\nreturn R::zero();\n}}\n"
        f"let u = {Tr}::unscale(self, n1);\nlet v = {Tr}::unscale(other, n2);\n"
        f"let half = Tr::atan2({Tr}::metric_distance(u, v), {Tr}::norm(u + v));\nhalf + half",
        False))
    abs_sum = " + ".join(f"R::abs(self.{k})" for k in F)
    powers = " + ".join(f"Powi::powi(r.{k}, q)" for k in F)
    out.append(fn(
        "lp_norm",
        "The entrywise Lp norm `(Σ |a|^p)^(1/p)`. `p = 1` is the exact sum of the absolute values "
        "and `p = 2` the fused `norm` (both exact up to their one rounding); above, the "
        "components are first divided by the largest absolute value `m` (so no power can "
        "overflow), `Σ (|a| / m)^p` is summed (`p` floored products each), and the root is `m * "
        "exp(ln(Σ) / p)`: a few ulp relative to the result, the rounding of `exp` / `ln`. Panics "
        "with `nalgebra: lp_norm needs p >= 1` for `p < 1` (upstream returns meaningless values: "
        "an infinite root for `p = 0`). Upstream: `lp_norm`.",
        f"fn lp_norm(self: {T}, p: i32) -> T",
        f"if p < 1 {{\ncore::panic_with_felt252(errors::LP_NORM_P);\n}}\n"
        f"if p == 1 {{\nreturn {abs_sum};\n}}\n"
        f"if p == 2 {{\nreturn {Tr}::norm(self);\n}}\n"
        f"let m = {Tr}::amax(self);\nif m == R::zero() {{\nreturn R::zero();\n}}\n"
        f"let r = {Tr}::unscale({Tr}::abs(self), m);\nlet q: u32 = p.try_into().unwrap();\n"
        f"let s = {powers};\n"
        f"m * Tr::exp(R::div(Tr::ln(s), R::from_int(p)))", False))
    if s.is_column:
        out.append(fn(
            "slerp",
            "Spherical interpolation of the DIRECTIONS of `self` and `rhs`: both are normalized, "
            "then `Unit::slerp` (the unit result along the great arc, with constant angular "
            "velocity; `t` is not clamped). Returns the normalized `self` when the directions are "
            "opposite (the arc is not defined), like upstream. Panics when a norm is zero or does "
            "not fit. Upstream: `slerp`.",
            f"fn slerp(self: {T}, rhs: {T}, t: T) -> {T}",
            f"let me = {Tr}::normalize(self);\nlet other = {Tr}::normalize(rhs);\n"
            f"match slerp_unit(me, other, t, R::default_epsilon()) {{\nOption::Some(v) => v,\n"
            f"Option::None => me,\n}}", False))
    return out


# --------------------------------------------------------------------------------------------
# Top-level items
# --------------------------------------------------------------------------------------------


def index_linear(s: Shape, src: str, some: bool) -> str:
    wrapf = (lambda x: f"Option::Some({x})") if some else (lambda x: x)
    miss = "Option::None" if some else f"core::panic_with_felt252({INDEX_ERROR})"
    arms = "".join(f"{k} => {wrapf(f'{src}{f}')},\n" for k, f in enumerate(s.fields))
    return f"match index {{\n{arms}_ => {miss},\n}}"


def index_pair(s: Shape, src: str, some: bool) -> str:
    wrapf = (lambda x: f"Option::Some({x})") if some else (lambda x: x)
    miss = "Option::None" if some else f"core::panic_with_felt252({INDEX_ERROR})"
    cols = []
    for j in range(s.c):
        rows = "".join(f"{i} => {wrapf(f'{src}{s.f(i, j)}')},\n" for i in range(s.r))
        cols.append(f"{j} => match i {{\n{rows}_ => {miss},\n}},\n")
    return f"let (i, j) = index;\nmatch j {{\n{''.join(cols)}_ => {miss},\n}}"


def items(s: Shape) -> list[str]:
    S, T, F, n = s.name, f"{s.name}<T>", s.fields, s.n
    out = [L.section("indexing, comparisons, conversions", 100)]
    for idx, body in (("usize", index_linear), ("(usize, usize)", index_pair)):
        suffix = "Linear" if idx == "usize" else "Pair"
        what = ("the component `index` in column-major (storage) order" if idx == "usize" else
                "the component at `(row, column)`")
        out.append(item(
            f"`m.get({'i' if idx == 'usize' else '(i, j)'})` / `m.index(..)`: {what}. `get` is "
            f"`None` out of bounds, `index` panics with `nalgebra: index out of bounds`. "
            f"Upstream: `Matrix::get` / `Matrix::index` (their `MatrixIndex` argument).",
            f"pub impl {S}MatrixIndex{suffix}<T, +Copy<T>, +Drop<T>> of MatrixIndex<{T}, {idx}> {{\n"
            f"type Output = T;\n{L.INLINE}\nfn get(self: {T}, index: {idx}) -> Option<T> {{\n"
            f"{body(s, 'self.', True)}\n}}\n{L.INLINE}\nfn index(self: {T}, index: {idx}) -> T {{\n"
            f"{body(s, 'self.', False)}\n}}\n}}"))
        out.append(item(
            f"`m[{'i' if idx == 'usize' else '(i, j)'}]`: {what}. Panics with `nalgebra: index out "
            f"of bounds`. Upstream: `Index<{idx}>`.",
            f"pub impl {S}Index{suffix}<T, +Copy<T>, +Drop<T>> of IndexView<{T}, {idx}> {{\n"
            f"type Target = T;\n{L.INLINE}\nfn index(self: @{T}, index: {idx}) -> T {{\n"
            f"{body(s, '*self.', False)}\n}}\n}}"))
    cmp = lambda op: "\n&& ".join(f"lhs.{k} {op} rhs.{k}" for k in F)  # noqa: E731
    out.append(item(
        "The component-wise partial order: `a < b` when EVERY component of `a` is smaller than "
        "`b`'s (likewise `<=`, `>`, `>=`), so two matrices may be unordered (`!(a < b) && !(a >= "
        "b)`). Upstream: `PartialOrd for Matrix`.",
        f"pub impl {S}PartialOrd<T, +PartialOrd<T>, +Copy<T>, +Drop<T>> of PartialOrd<{T}> {{\n"
        + "".join(f"{L.INLINE}\nfn {name}(lhs: {T}, rhs: {T}) -> bool {{\n{cmp(op)}\n}}\n"
                  for name, op in (("lt", "<"), ("le", "<="), ("gt", ">"), ("ge", ">=")))
        + "}"))
    out.append(item(
        "The component-wise bounds: every component `Bounded::<T>::MIN` / `MAX`. Upstream: "
        "`num::Bounded for Matrix`.",
        f"pub impl {S}Bounded<T, +Bounded<T>, +Drop<T>> of Bounded<{T}> {{\n"
        f"const MIN: {T} = {L.lit(S, [(k, 'Bounded::<T>::MIN') for k in F])};\n"
        f"const MAX: {T} = {L.lit(S, [(k, 'Bounded::<T>::MAX') for k in F])};\n}}"))
    if s.is_square:
        ident = " && ".join(f"*self.{s.f(i, j)} == {'R::one()' if i == j else 'R::zero()'}"
                            for j in range(s.c) for i in range(s.r))
        out.append(item(
            "The multiplicative identity (`identity()`), exactly. Upstream: `num::One for "
            "SquareMatrix`.",
            f"pub impl {S}One<T, impl R: Real<T>, +PartialEq<T>, +Copy<T>, +Drop<T>> of One<{T}> {{\n"
            f"{L.INLINE}\nfn one() -> {T} {{\n"
            f"{L.lit(S, [(s.f(i, j), 'R::one()' if i == j else 'R::zero()') for j in range(s.c) for i in range(s.r)])}"
            f"\n}}\n{L.INLINE}\nfn is_one(self: @{T}) -> bool {{\n{ident}\n}}\n{L.INLINE}\n"
            f"fn is_non_one(self: @{T}) -> bool {{\n!Self::is_one(self)\n}}\n}}"))
    colnames = [[s.f(i, j) for i in range(s.r)] for j in range(s.c)]
    arr_t = f"[[T; {s.r}]; {s.c}]"
    cvars = [f"c{j}" for j in range(s.c)]
    out.append(item(
        f"The {s.kind} of the given COLUMNS (`[[m11, m21, ..], [m12, ..], ..]`). Upstream: "
        f"`From<[[T; R]; C]>`.",
        f"pub impl {S}FromColumnArrays<T, +Drop<T>> of Into<{arr_t}, {T}> {{\n{L.INLINE}\n"
        f"fn into(self: {arr_t}) -> {T} {{\nlet [{', '.join(cvars)}] = self;\n"
        + "".join(f"let [{', '.join(colnames[j])}] = c{j};\n" for j in range(s.c))
        + f"{shorthand(s)}\n}}\n}}"))
    out.append(item(
        f"The columns of the {s.kind} as nested arrays (`[[m11, m21, ..], [m12, ..], ..]`). "
        f"Upstream: `Into<[[T; R]; C]>`.",
        f"pub impl {S}IntoColumnArrays<T, +Drop<T>> of Into<{T}, {arr_t}> {{\n{L.INLINE}\n"
        f"fn into(self: {T}) -> {arr_t} {{\nlet {shorthand(s)} = self;\n"
        f"[{', '.join('[' + ', '.join(c) + ']' for c in colnames)}]\n}}\n}}"))
    if s.r in (2, 3) and s.c in (2, 3):
        # `m * p`: the points of the crate are `Point2` / `Point3`.
        pr, pc = f"Point{s.r}", f"Point{s.c}"
        comps = COORDS[:s.r]
        vals = [(comps[i], products([(f"self.{s.f(i, q)}", f"rhs.{COORDS[q]}")
                                    for q in range(s.c)])) for i in range(s.r)]
        out.append(item(
            f"`self * p`, a `{pr}`: the linear map applied to the coordinates of `p`, each "
            f"coordinate one fused `sum_prod{s.c}` (floored once). Panics on overflow. Upstream: "
            f"`Mul<Point> for Matrix` (`m * p`).",
            f"pub impl {S}MulPoint<T, impl R: Real<T>, +Mul<T>, +Copy<T>, +Drop<T>> of "
            f"MatrixMul<{T}, {pc}<T>> {{\ntype Output = {pr}<T>;\n{L.INLINE}\n"
            f"fn mul_mat(self: {T}, rhs: {pc}<T>) -> {pr}<T> {{\n{L.lit(pr, vals)}\n}}\n}}"))
    if s.is_column:
        out += unit_items(s)
    return out


def unit_items(s: Shape) -> list[str]:
    """`UnitVector{n}Trait` (Real) and `UnitVector{n}AngleTrait` (Transcendental)."""
    S, n, F = s.name, s.r, s.fields
    U = f"Unit<{S}<T>>"
    alias = s.unit_alias
    fns = [fn("cast", f"The unit vector with every component converted by `Into<T, U>` (the "
                      f"identity for `Fixed`). Upstream: `Unit::cast`.",
              f"fn cast<U, +Into<T, U>, +Drop<U>>(self: {U}) -> Unit<{S}<U>>",
              f"Unit {{ value: {L.lit(S, [(k, f'self.value.{k}.into()') for k in F])} }}", False)]
    for name, sig, what, call in (
            ("relative_eq", "epsilon: u64, max_relative: T", "`relative_eq`",
             "relative_eq(self.value, other.value, epsilon, max_relative)"),
            ("ulps_eq", "epsilon: u64, max_ulps: u32", "`ulps_eq`",
             "ulps_eq(self.value, other.value, epsilon, max_ulps)")):
        fns.append(fn(name, f"{what} of the two vectors (see `{S}Trait::{name}`). Upstream: "
                            f"`approx::{'RelativeEq' if name == 'relative_eq' else 'UlpsEq'}` for "
                            f"`Unit`.",
                      f"fn {name}(self: {U}, other: {U}, {sig}) -> bool", f"{S}Trait::{call}"))
    if n in (1, 5, 6):  # `Unit2/3/4Trait` (`unit.cairo`) have the axes of the former shapes
        for a in F:
            fns.append(fn(f"{a}_axis", f"The unit vector along `{a}`. Upstream: "
                                       f"`Unit::<{S}>::{a}_axis`.", f"fn {a}_axis() -> {U}",
                          f"Unit {{ value: {L.lit(S, [(k, 'R::one()' if k == a else 'R::zero()') for k in F])} }}"))
    real = "\n\n".join(f.definition() for f in fns)
    out = [item(f"Methods of `{alias}<T>` (`Unit<{S}<T>>`) specific to the shape.",
                f"#[generate_trait]\npub impl {alias}Impl<\n{L.bounds(L.MATRIX_IMPL_BOUNDS)}\n> of "
                f"{alias}Trait<T> {{\n{real}\n}}")]
    slerp = [
        fn("slerp", "Spherical linear interpolation between two unit vectors along the great arc, "
                    "with constant angular velocity (`t` is not clamped). Returns `self` when the "
                    "vectors are opposite (the arc is not defined), like upstream. Upstream: "
                    "`Unit::slerp`.",
           f"fn slerp(self: {U}, rhs: {U}, t: T) -> {U}",
           "match slerp_unit(self.value, rhs.value, t, R::default_epsilon()) {\n"
           "Option::Some(v) => Unit { value: v },\nOption::None => self,\n}", False),
        fn("try_slerp", "`slerp`, or `None` when `sin` of the angle between the vectors is `<= "
                        "epsilon` (nearly parallel or opposite vectors: the interpolation plane is "
                        "ill-conditioned; `self` is returned as is for exactly equal ones). "
                        "`epsilon` is in scalar units. Upstream: `Unit::try_slerp`.",
           f"fn try_slerp(self: {U}, rhs: {U}, t: T, epsilon: T) -> Option<{U}>",
           "match slerp_unit(self.value, rhs.value, t, epsilon) {\n"
           "Option::Some(v) => Option::Some(Unit { value: v }),\nOption::None => Option::None,\n}",
           False),
    ]
    out.append(item(f"The interpolations of `{alias}<T>`, which need `Transcendental`.",
                    f"#[generate_trait]\npub impl {alias}AngleImpl<\n{L.bounds(ANGLE_BOUNDS)}\n> of "
                    f"{alias}AngleTrait<T> {{\n" + "\n\n".join(f.definition() for f in slerp) + "\n}"))
    # The shared kernel of `Vector::slerp`, `Unit::slerp` and `Unit::try_slerp`.
    blk = lambda e: e if "\n" not in e else "{\n" + e + "\n}"  # noqa: E731
    d = squares([f"a.{k} - b.{k}" for k in F], "sqrt")
    sm = squares([f"a.{k} + b.{k}" for k in F], "sqrt")
    body = (f"let d = {blk(d)};\nif d == R::zero() {{\nreturn Option::Some(a);\n}}\n"
            f"let s = {blk(sm)};\n"
            "let half = Tr::atan2(d, s);\nlet hang = half + half;\n"
            "let shang = (d * s) * R::HALF;\n"
            "if shang <= epsilon {\nreturn Option::None;\n}\n"
            "let ta = R::div(Tr::sin((R::one() - t) * hang), shang);\n"
            "let tb = R::div(Tr::sin(t * hang), shang);\n"
            f"Option::Some({L.lit(S, [(k, f'R::sum_prod2(a.{k}, ta, b.{k}, tb)') for k in F])})")
    out.append(
        "/// `Unit::try_slerp` on the values `a`, `b` of two unit vectors (upstream "
        "`interpolation.rs`): `a` when\n/// they are equal, `None` when `sin θ <= epsilon`, "
        "otherwise each component is ONE `sum_prod2` of\n/// the weights `sin((1 - t) θ) / sin "
        "θ` and `sin(t θ) / sin θ`.\n///\n/// The angle comes from the Kahan half-angle form of "
        "`angle`: `θ = 2 atan2(|a - b|, |a + b|)` and\n/// `sin θ = |a - b| |a + b| / 2` (unit "
        "vectors), two fused norms of exact differences / sums. Upstream's\n/// `acos(a · b)` and "
        "`sqrt(1 - (a · b)²)` lose the last bit of `1 - c²` near `c = 1` (a unit\n/// vector "
        "interpolated with itself came out √2 too long), and is kept as a benchmark\n/// "
        "(`bench_vector4_slerp__alt_acos`).\n"
        f"fn slerp_unit<\n{L.bounds(ANGLE_BOUNDS)}\n>(a: {S}<T>, b: {S}<T>, t: T, epsilon: T) -> "
        f"Option<{S}<T>> {{\n{body}\n}}")
    return out


def uses(s: Shape) -> list[str]:
    out = ["core::num::traits::{Bounded, One}" if s.is_square else "core::num::traits::Bounded",
           "core::ops::IndexView", "super::matrix_index::MatrixIndex", "super::errors",
           "super::kernels::Powi", "crate::geometry::quaternion::ApproxEqTrait"]
    t = s.transposed()
    if t != s:
        out.append(f"super::{t.module}::{t.name}")
    if s.is_column or s.unit_alias:
        out.append("super::unit::Unit")
    if s.is_column and s.r <= 5:
        u = vec(s.r + 1)
        out.append(f"super::{u.module}::{u.name}")
    if s.is_square and 2 <= s.r <= 5:
        u = Shape(s.r + 1, s.r + 1)
        out.append(f"super::{u.module}::{u.name}")
    if s.r in (2, 3) and s.c in (2, 3):
        out += ["super::matrix_mul::MatrixMul"] + [f"super::point{d}::Point{d}"
                                                   for d in sorted({s.r, s.c})]
    return out


def exported(s: Shape) -> list[str]:
    """The traits of this module that the crate root re-exports for the shape."""
    names = [f"{s.name}AngleTrait"]
    if s.unit_alias:
        names += [f"{s.unit_alias}Trait", f"{s.unit_alias}AngleTrait"]
    return names


def angle_impl(s: Shape, fns: list[L.Fn]) -> str:
    """A `#[generate_trait]` `<S>AngleImpl` of `fns` (the new shapes; `Vector6`, `Matrix2/3/4/6`)."""
    return (f"/// The operations of `{s.name}<T>` that need `Transcendental` (inverse "
            f"trigonometry, `exp`, `ln`):\n/// a scalar may implement `Real` only.\n"
            f"#[generate_trait]\npub impl {s.name}AngleImpl<\n{L.bounds(ANGLE_BOUNDS)}\n> of "
            f"{s.name}AngleTrait<T> {{\n" + "\n\n".join(f.definition() for f in fns) + "\n}")
