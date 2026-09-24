"""Templates of the library shapes of `crates/nalgebra` (WP 8.1b-1, `DESIGN.md` §2.2).

`shapegen.py` writes `crates/nalgebra/src/base/{vector2,vector3,vector4,vector6,matrix2,matrix3,
matrix4,matrix6}.cairo` from the templates of this module and the verbatim kernels of
`specialisations/<module>.cairo`. `Vector6` / `Matrix6` (WP 8.1b-2) are flat like upstream
(`x y z w a b`, `m11 .. m66`) and keep the surface and the kernels of the hand-written block types
they replaced: the 6-term sums of products are ONE `Real::Wide` chain each (`wide_add_prod` then
`wide_rescale` / `wide_sqrt`), `simba::Real` stopping at `sum_prod4`.
The templates reproduce the hand-written files they replaced EXACTLY (same items, same bodies,
same attributes, same doc comments), so that the migration changes no bit and no gas figure: the
`Vector2/3/4/6` family keeps its explicit `pub trait VectorNTrait` + `pub impl VectorNImpl`, the
`Matrix2/3/4/6` family its `#[generate_trait]` impls, `R::zero()` / `R::one()`, and the kernel of
each method as it was measured (the 6D layout changed, so only the bits and the gas are kept
there: `DESIGN.md` §2.4). Unifying the two families (and the prototype's templates) is
8.1b-3's work, under the gas gate.

Doc comments are data: they are written with their line breaks (the hand-written ones are not
greedily wrapped, and `scarb fmt` never re-wraps comments).

Specialisation files (`specialisations/<module>.cairo`) are split in sections:

* `// @use <path>`: an extra `use <path>;` line of the generated module;
* `// @method <name>`: a method of the public `<Struct>Trait`, replacing the template of that name
  or filling the slot of that name in the shape's method order (`VECTOR_ORDER`, `MATRIX_ORDER`);
* `// @internal <name>`: the same for the crate-internal `<Struct>InternalTrait` of the matrices;
* `// @doc <block>`: the doc comment of a generated block (`internal`: the internal impl;
  `module`: `//!` lines appended to the module doc; `<method>`: replaces the doc comment of that
  generated method; `<Trait>` (`Mul`, ...): replaces the doc comment of that operator impl);
* `// @item <name> <anchor>`: a verbatim top-level item, placed at `<anchor>` (`struct`: after
  the struct; `impl`: after the main impl; `end`: at the end of the file). Items keep their
  relative order within an anchor.

Text before the first section is a comment for the reader and is ignored. A `@method` /
`@internal` body is a doc comment, attributes and ONE `fn`; the generator re-renders it in the
family's form (declaration in the trait, definition in the impl, for the vectors).
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

COORDS = "xyzwab"
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "crates" / "nalgebra" / "src" / "base"
SPECS = Path(__file__).resolve().parent / "specialisations"
INLINE = "#[inline(always)]"

# The shapes this module generates: (rows, columns) -> module.
LIBRARY_SHAPES = {
    (2, 1): "vector2",
    (3, 1): "vector3",
    (4, 1): "vector4",
    (2, 2): "matrix2",
    (3, 3): "matrix3",
    (4, 4): "matrix4",
    (6, 1): "vector6",
    (6, 6): "matrix6",
}

NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
ORDINALS = {1: "First", 2: "Second"}


# --------------------------------------------------------------------------------------------
# Code model
# --------------------------------------------------------------------------------------------


@dataclass
class Fn:
    """One method: doc comment, attributes, signature (`fn name(..) -> ..`), body."""

    name: str
    doc: str  # the doc comment WITHOUT `/// `, lines separated by `\n`; '' for none
    sig: str
    body: str
    inline: bool = True
    raw_doc: list[str] | None = None  # verbatim `///` lines of a specialisation

    def doc_lines(self) -> list[str]:
        if self.raw_doc is not None:
            return self.raw_doc
        if not self.doc:
            return []
        return [f"/// {line}" if line else "///" for line in self.doc.split("\n")]

    def attrs(self) -> list[str]:
        return [INLINE] if self.inline else []

    def definition(self, with_doc: bool = True) -> str:
        head = (self.doc_lines() if with_doc else []) + self.attrs()
        return "\n".join(head + [f"{self.sig} {{", self.body, "}"])

    def declaration(self) -> str:
        return "\n".join(self.doc_lines() + [f"{self.sig};"])


def parse_fn(name: str, text: str) -> Fn:
    """A specialisation section (doc, attributes, one `fn`) as an `Fn`, verbatim."""
    lines = text.strip("\n").split("\n")
    doc = []
    while lines and lines[0].lstrip().startswith("///"):
        doc.append(lines.pop(0).strip())
    inline = False
    while lines and lines[0].lstrip().startswith("#["):
        attr = lines.pop(0).strip()
        if attr != INLINE:
            raise ValueError(f"unsupported attribute {attr} in specialisation {name}")
        inline = True
    rest = "\n".join(lines).strip()
    if not rest.startswith("fn "):
        raise ValueError(f"specialisation {name}: expected `fn`, got {rest[:40]!r}")
    brace = rest.index("{")
    if not rest.endswith("}"):
        raise ValueError(f"specialisation {name}: the `fn` must end the section")
    sig = " ".join(rest[:brace].split())
    body = rest[brace + 1:-1].strip("\n")
    return Fn(name, "", sig, body, inline, doc)


@dataclass
class Extra:
    """What the generic shape module (`shapes.py`) adds to a shape file of this module: `use`
    lines, items after the struct (the upstream aliases) and items at the end (the products)."""

    uses: list[str] = field(default_factory=list)
    struct: list[str] = field(default_factory=list)
    end: list[str] = field(default_factory=list)
    methods: list["Fn"] = field(default_factory=list)  # appended to the shape's main trait
    angle: list["Fn"] = field(default_factory=list)  # the `Transcendental` trait (`angle`...)


def dedup(xs: list[str]) -> list[str]:
    return list(dict.fromkeys(xs))


@dataclass
class Specs:
    uses: list[str] = field(default_factory=list)
    methods: dict[str, Fn] = field(default_factory=dict)
    internal: dict[str, Fn] = field(default_factory=dict)
    docs: dict[str, str] = field(default_factory=dict)
    items: list[tuple[str, str, str]] = field(default_factory=list)  # (name, anchor, text)


def read_specs(module: str) -> Specs:
    specs = Specs()
    path = SPECS / f"{module}.cairo"
    if not path.is_file():
        return specs
    parts = re.split(r"^// @(use|method|internal|doc|item) ?(.*)\n", path.read_text(), flags=re.M)
    for i in range(1, len(parts), 3):
        kind, arg, text = parts[i], parts[i + 1].strip(), parts[i + 2]
        if kind == "use":
            specs.uses.append(arg)
            if text.strip():
                raise ValueError(f"{path}: `// @use` takes no body")
        elif kind == "method":
            specs.methods[arg] = parse_fn(arg, text)
        elif kind == "internal":
            specs.internal[arg] = parse_fn(arg, text)
        elif kind == "doc":
            specs.docs[arg] = text.strip("\n")
        else:
            name, anchor = arg.split()
            specs.items.append((name, anchor, text.strip("\n")))
    return specs


def lit(struct: str, values: list[tuple[str, str]]) -> str:
    """A struct literal, with the field init shorthand where the value is the field's name."""
    body = ", ".join(k if k == v else f"{k}: {v}" for k, v in values)
    return f"{struct} {{ {body} }}"


def tup(xs) -> str:
    return f"({', '.join(xs)})"


def bounds(names: list[str]) -> str:
    return ",\n".join(names) + ","


# --------------------------------------------------------------------------------------------
# Column vectors: Vector2, Vector3, Vector4, Vector6
# --------------------------------------------------------------------------------------------

VECTOR_ORDER = [
    "new", "zeros", "repeat", "from_element", "axes", "swizzles", "push", "to_homogeneous",
    "transpose", "scale", "unscale", "component_mul", "component_div", "abs", "inf", "sup", "inf_sup", "min",
    "max", "amin", "amax", "imin", "imax", "iamin", "iamax", "sum", "is_zero", "abs_diff_eq",
    "dot", "perp", "cross", "norm_squared", "norm", "magnitude_squared", "magnitude",
    "metric_distance", "normalize", "try_normalize", "cap_magnitude", "lerp",
]

# `Vector{N}AngleTrait::angle` doc: the measured gas of the robust form and of upstream's form.
ANGLE_GAS = {2: ("61 200", "38 790"), 3: ("71 740", "39 120"), 4: ("81 980", "39 990")}

# The largest vector of the `Vector2/3/4` family: `push` / `to_homogeneous` need `Vector{n + 1}`.
MAX_VECTOR = 4

# The surface of the vectors that do not carry all of `VECTOR_ORDER` (no `angle` either): `Vector6`
# keeps the methods of the hand-written block type it replaced (WP 8.1b-2); 8.2 widens it.
VECTOR_SURFACE = {
    6: ["new", "zeros", "transpose", "scale", "unscale", "component_mul", "abs", "inf", "sup", "sum",
        "abs_diff_eq", "dot", "norm_squared", "norm", "normalize", "lerp"],
}


class Vec:
    def __init__(self, n: int):
        self.n = n
        self.S = f"Vector{n}"
        self.c = list(COORDS[:n])

    def ty(self) -> str:
        return f"{self.S}<T>"

    def fields(self, f) -> list[tuple[str, str]]:
        return [(c, f(c)) for c in self.c]

    def lit(self, f) -> str:
        return lit(self.S, self.fields(f))

    def args(self, fmt: str) -> str:
        return ", ".join(fmt.format(c=c) for c in self.c)


def fold(op: str, xs: list[str]) -> str:
    """`op(op(x0, x1), x2)`: left fold."""
    acc = xs[0]
    for x in xs[1:]:
        acc = f"{op}({acc}, {x})"
    return acc


def vector_index(v: Vec, cmp: str) -> str:
    """Body of `imin` (`cmp = '<='`) / `imax` (`'>='`): pairwise tournament, first on ties."""
    c = v.c
    if v.n == 2:
        return f"if self.x {cmp} self.y {{\n0\n}} else {{\n1\n}}"
    first = f"let (i, m): (usize, T) = if self.x {cmp} self.y {{\n(0, self.x)\n}} else {{\n(1, self.y)\n}};"
    if v.n == 3:
        return f"{first}\nif m {cmp} self.z {{\ni\n}} else {{\n2\n}}"
    if v.n == 4:
        second = (f"let (j, p): (usize, T) = if self.z {cmp} self.w {{\n(2, self.z)\n}} else "
                  f"{{\n(3, self.w)\n}};")
        return f"{first}\n{second}\nif m {cmp} p {{\ni\n}} else {{\nj\n}}"
    raise ValueError(f"imin / imax of {v.S}: no template for {len(c)} components")


def vector_unscale(v: Vec, divisor: str, target: str) -> str:
    """`target` = the divided vector, one correctly rounded division per component:
    `Real::div3` / `div4` / `div6` (one prepared divisor) when it exists, `R::div` per component otherwise."""
    if v.n in (3, 4, 6):
        names = tup(v.c)
        return (f"let {names} = R::div{v.n}({v.args('self.{c}')}, {divisor});\n"
                f"{target}{lit(v.S, [(c, c) for c in v.c])}")
    return f"{target}{v.lit(lambda c: f'R::div(self.{c}, {divisor})')}"


def vector_fns(v: Vec) -> dict[str, Fn | list[Fn]]:
    S, T, n = v.S, v.ty(), v.n
    tuple_c = tup(v.c)
    ones = lambda i: tup("1" if j == i else "0" for j in range(n))  # noqa: E731
    # Up to 4 components `Real` has the fused kernels; above, ONE `Real::Wide` chain (`wide_chain`).
    wide = n > 4
    norm = "Self::norm(self)" if wide else f"R::norm{n}({v.args('self.{c}')})"
    squares = [f"self.{c}" for c in v.c]
    out: dict[str, Fn | list[Fn]] = {}
    out["new"] = Fn("new", f"The vector `{tuple_c}`. Upstream: `{S}::new`.",
                    f"fn new({v.args('{c}: T')}) -> {T}", lit(S, [(c, c) for c in v.c]))
    out["zeros"] = Fn("zeros", f"The zero vector. Upstream: `{S}::zeros`.", f"fn zeros() -> {T}",
                      v.lit(lambda c: "R::zero()"))
    out["repeat"] = Fn("repeat",
                       f"The vector whose components all equal `elem`. Upstream: `{S}::repeat`.",
                       f"fn repeat(elem: T) -> {T}", v.lit(lambda c: "elem"))
    out["from_element"] = Fn("from_element", f"Alias of `repeat`. Upstream: `{S}::from_element`.",
                             f"fn from_element(elem: T) -> {T}", v.lit(lambda c: "elem"))
    axes = []
    for i, a in enumerate(v.c):
        head = f"The unit axis `{ones(i)}`. Upstream: `{S}::{a}` (`{a}_axis` returns a `Unit`, which is"
        doc = (f"{head} not\nported yet)." if len(head) + 8 <= 96 else f"{head}\nnot ported yet).")
        axes.append(Fn(a, doc, f"fn {a}() -> {T}",
                       v.lit(lambda c, a=a: "R::one()" if c == a else "R::zero()")))
    out["axes"] = axes
    swz = []
    for k in range(2, n):
        name = "".join(v.c[:k])
        sub = Vec(k)
        swz.append(Fn(name, f"The first {NUMBER_WORDS[k]} components. Upstream: the `{name}` "
                            f"swizzle (`fixed_rows::<{k}>(0)`).",
                      f"fn {name}(self: {T}) -> {sub.ty()}", sub.lit(lambda c: f"self.{c}")))
    out["swizzles"] = swz
    if n < MAX_VECTOR:
        up = Vec(n + 1)
        new = COORDS[n]
        out["push"] = Fn("push", f"`{tup(up.c)}`: appends a component. Upstream: `push`.",
                         f"fn push(self: {T}, {new}: T) -> {up.ty()}",
                         up.lit(lambda c: c if c == new else f"self.{c}"))
        out["to_homogeneous"] = Fn(
            "to_homogeneous",
            f"`{tup(v.c + ['0'])}`: homogeneous coordinates of a vector (as opposed to a point). "
            f"Upstream:\n`to_homogeneous`.",
            f"fn to_homogeneous(self: {T}) -> {up.ty()}",
            up.lit(lambda c: "R::zero()" if c == new else f"self.{c}"))
    out["transpose"] = Fn("transpose", f"The transpose, a `RowVector{n}` with the same components. "
                                       f"Exact. Upstream: `transpose`.",
                          f"fn transpose(self: {T}) -> RowVector{n}<T>",
                          lit(f"RowVector{n}", [(c, f"self.{c}") for c in v.c]))
    out["scale"] = Fn("scale", "`self * k`, each component floored once. Panics on overflow. "
                               "Upstream: `scale`\n(`self * k`).",
                      f"fn scale(self: {T}, k: T) -> {T}", v.lit(lambda c: f"self.{c} * k"))
    out["unscale"] = Fn(
        "unscale",
        "`self / k`, each component being the correctly rounded quotient. Panics on a zero `k` and\n"
        "on overflow. Upstream: `unscale` (`self / k`).\n\n"
        "One division per component on purpose: `scale(k.recip())` is cheaper but rounds `1 / k`\n"
        "first, which costs up to `|self|` ulp instead of 1 (measured by\n"
        f"`bench_vector{n}_unscale__alt_recip` and `test_unscale_alt_recip_is_less_accurate`).\n"
        "Callers dividing many vectors by the same value should store its reciprocal and `scale`.",
        f"fn unscale(self: {T}, k: T) -> {T}", vector_unscale(v, "k", ""))
    out["component_mul"] = Fn(
        "component_mul", "Component-wise product, each component floored once. Panics on "
                         "overflow. Upstream:\n`component_mul`.",
        f"fn component_mul(self: {T}, rhs: {T}) -> {T}", v.lit(lambda c: f"self.{c} * rhs.{c}"))
    out["component_div"] = Fn(
        "component_div", "Component-wise quotient, each component rounded to nearest. Panics on a "
                         "zero component of\n`rhs` and on overflow. Upstream: `component_div`.",
        f"fn component_div(self: {T}, rhs: {T}) -> {T}",
        v.lit(lambda c: f"R::div(self.{c}, rhs.{c})"))
    out["abs"] = Fn("abs", "Component-wise absolute value. Exact; panics on overflow (`|MIN|`). "
                           "Upstream: `abs`.",
                    f"fn abs(self: {T}) -> {T}", v.lit(lambda c: f"R::abs(self.{c})"))
    for name, op, what in (("inf", "min", "minimum (infimum)"), ("sup", "max", "maximum (supremum)")):
        out[name] = Fn(name, f"Component-wise {what}. Exact. Upstream: `{name}`.",
                       f"fn {name}(self: {T}, other: {T}) -> {T}",
                       v.lit(lambda c, op=op: f"R::{op}(self.{c}, other.{c})"))
    out["inf_sup"] = Fn("inf_sup", "`(self.inf(other), self.sup(other))`. Exact. Upstream: `inf_sup`.",
                        f"fn inf_sup(self: {T}, other: {T}) -> ({T}, {T})",
                        "(Self::inf(self, other), Self::sup(self, other))")
    out["min"] = Fn("min", "The smallest component. Exact. Upstream: `min`.",
                    f"fn min(self: {T}) -> T", fold("R::min", [f"self.{c}" for c in v.c]))
    out["max"] = Fn("max", "The largest component. Exact. Upstream: `max`.",
                    f"fn max(self: {T}) -> T", fold("R::max", [f"self.{c}" for c in v.c]))
    out["amin"] = Fn("amin", "The smallest absolute value of a component. Panics on overflow "
                             "(`|MIN|`). Upstream:\n`amin`.",
                     f"fn amin(self: {T}) -> T", fold("R::min", [f"R::abs(self.{c})" for c in v.c]))
    out["amax"] = Fn("amax", "The largest absolute value of a component (infinity norm). Panics "
                             "on overflow (`|MIN|`).\nUpstream: `amax`.",
                     f"fn amax(self: {T}) -> T", fold("R::max", [f"R::abs(self.{c})" for c in v.c]))
    idx = tup(str(i) for i in range(n))
    if n <= 4:  # no tournament template above 4 components (not in `VECTOR_SURFACE[6]`)
        out["imin"] = Fn("imin", f"Index {idx} of the smallest component, the first one on ties. "
                                 f"Upstream: `imin`.", f"fn imin(self: {T}) -> usize",
                         vector_index(v, "<="))
        out["imax"] = Fn("imax", f"Index {idx} of the largest component, the first one on ties. "
                                 f"Upstream: `imax`.", f"fn imax(self: {T}) -> usize",
                         vector_index(v, ">="))
    out["iamin"] = Fn("iamin", "Index of the component with the smallest absolute value, the "
                               "first one on ties. Panics on\noverflow (`|MIN|`). Upstream: `iamin`.",
                      f"fn iamin(self: {T}) -> usize", "Self::imin(Self::abs(self))")
    out["iamax"] = Fn("iamax", "Index of the component with the largest absolute value, the first "
                               "one on ties. Panics on\noverflow (`|MIN|`). Upstream: `iamax`.",
                      f"fn iamax(self: {T}) -> usize", "Self::imax(Self::abs(self))")
    out["sum"] = Fn("sum", "Sum of the components. Exact; panics on overflow. Upstream: `sum`.",
                    f"fn sum(self: {T}) -> T", " + ".join(f"self.{c}" for c in v.c))
    out["is_zero"] = Fn("is_zero", "`true` when every component is zero. Upstream: `Zero::is_zero`.",
                        f"fn is_zero(self: {T}) -> bool",
                        " && ".join(f"self.{c} == R::zero()" for c in v.c))
    out["abs_diff_eq"] = Fn(
        "abs_diff_eq",
        "`true` when every component is within `ulps` smallest units (raw units for fixed point) "
        "of\nthe matching component of `other`; cannot overflow. Upstream:\n"
        "`approx::AbsDiffEq::abs_diff_eq`, the tolerance being counted in ulp instead of a float\n"
        "epsilon (DESIGN D3).",
        f"fn abs_diff_eq(self: {T}, other: {T}, ulps: u64) -> bool",
        " && ".join(f"R::abs_diff_eq(self.{c}, other.{c}, ulps)" for c in v.c))
    out["dot"] = Fn("dot", f"Dot product, fused (`Real::sum_prod{n}`): the exact sum of products "
                           f"is floored once. Only\nthe result must fit: panics on overflow. "
                           f"Upstream: `dot`.",
                    f"fn dot(self: {T}, rhs: {T}) -> T",
                    f"R::sum_prod{n}({v.args('self.{c}, rhs.{c}')})")
    out["norm_squared"] = Fn(
        "norm_squared", "Squared Euclidean norm, fused (floored once). Panics on overflow: above "
                        "a norm of about\n46 340 (Q32.32) only `norm` works. Upstream: `norm_squared`.",
        f"fn norm_squared(self: {T}) -> T", f"R::norm_squared{n}({v.args('self.{c}')})")
    out["norm"] = Fn(
        "norm", f"Euclidean norm (`Real::norm{n}`): square root of the UNSCALED exact sum of "
                f"squares, floored\nonce. No intermediate overflow: only the result must fit, so "
                f"the norm of\n`{tup(['1e6'] * n)}` is fine. Upstream: `norm`.",
        f"fn norm(self: {T}) -> T", norm)
    out["magnitude_squared"] = Fn(
        "magnitude_squared", "Alias of `norm_squared`. Upstream: `magnitude_squared`.",
        f"fn magnitude_squared(self: {T}) -> T", f"R::norm_squared{n}({v.args('self.{c}')})")
    out["magnitude"] = Fn("magnitude", "Alias of `norm`. Upstream: `magnitude`.",
                          f"fn magnitude(self: {T}) -> T", norm)
    out["metric_distance"] = Fn(
        "metric_distance", "`(self - rhs).norm()`. Panics when a component difference or the "
                           "result overflows.\nUpstream: `metric_distance`.",
        f"fn metric_distance(self: {T}, rhs: {T}) -> T",
        f"R::norm{n}({v.args('self.{c} - rhs.{c}')})")
    out["normalize"] = Fn(
        "normalize",
        "`self / self.norm()`: the floored norm, then one correctly rounded division per component\n"
        "(`unscale`). The error is about `1 + 1 / norm` ulp per component whatever the magnitude of\n"
        "`self`, from a few ulp up to the longest vector whose norm fits. Panics with a division by\n"
        "zero when the norm is zero, and on overflow when the norm does not fit. Upstream:\n"
        "`normalize`.\n\n"
        f"The cheaper candidates are kept as benchmarks (`bench_vector{n}_normalize__alt_*`): one\n"
        "reciprocal of the norm then one product per component is off by about `norm` ulp and\n"
        "overflows for norms up to `2^-31`; `recip(sqrt(norm_squared))` also overflows for norms\n"
        "above 46 340 and has no precision left for short vectors.",
        f"fn normalize(self: {T}) -> {T}", f"Self::unscale(self, {norm})")
    out["try_normalize"] = Fn(
        "try_normalize", "`Some(self.normalize())`, or `None` when the norm is `<= min_norm`. With "
                         "`min_norm >= 0`\nit never divides by zero. Upstream: `try_normalize`.",
        f"fn try_normalize(self: {T}, min_norm: T) -> Option<{T}>",
        f"let n = {norm};\nif n <= min_norm {{\nNone\n}} else {{\nSome(Self::unscale(self, n))\n}}")
    out["cap_magnitude"] = Fn(
        "cap_magnitude",
        "`self` when its norm is `<= max`, otherwise `self.scale(max / norm)` like upstream. The\n"
        "ratio is rounded to nearest (like Rust's `max / n` in f64), so the capped norm is within\n"
        "about `norm / max` ulp of `max` and, as in Rust, may exceed it by the rounding. `max` is\n"
        "expected to be `>= 0`. Panics only when the norm does not fit. Upstream: `cap_magnitude`.\n\n"
        "The more accurate and dearer `normalize().scale(max)` is kept as a benchmark\n"
        f"(`bench_vector{n}_cap_magnitude__alt_normalize`).",
        f"fn cap_magnitude(self: {T}, max: T) -> {T}",
        f"let n = {norm};\nif n <= max {{\nself\n}} else {{\nSelf::scale(self, R::div(max, n))\n}}")
    out["lerp"] = Fn(
        "lerp",
        "`self + (rhs - self) * t` per component (`Real::lerp`: exact difference and product, one\n"
        "floor rounding). `t` is not clamped; `t = 0` gives `self` and `t = 1` gives `rhs` exactly.\n"
        "Panics on overflow of the result. Upstream: `lerp` (`self * (1 - t) + rhs * t`).",
        f"fn lerp(self: {T}, rhs: {T}, t: T) -> {T}", v.lit(lambda c: f"R::lerp(self.{c}, rhs.{c}, t)"))
    if wide:
        words = NUMBER_WORDS[n]
        out["dot"] = Fn(
            "dot", f"Dot product of the {words} components: the products are accumulated EXACTLY "
                   f"in the `Real::Wide`\naccumulator and rescaled ONCE (one floor, one overflow "
                   f"check), so the result is the exact floor\nof the mathematical dot product. "
                   f"Only the result must fit. Upstream: `dot`.",
            f"fn dot(self: {T}, rhs: {T}) -> T",
            wide_chain([(f"self.{c}", f"rhs.{c}") for c in v.c], "wide_rescale"))
        out["norm_squared"] = Fn(
            "norm_squared", f"Squared Euclidean norm: the {words} squares accumulated exactly, "
                            f"floored once. Panics on\noverflow — above a norm of about 46 340 "
                            f"(Q32.32) only `norm` works. Upstream:\n`norm_squared`.",
            f"fn norm_squared(self: {T}) -> T", wide_chain(squares, "wide_rescale"))
        out["norm"] = Fn(
            "norm", "Euclidean norm: square root of the UNSCALED exact sum of squares "
                    "(`Real::wide_sqrt`),\nfloored once. No intermediate overflow: only the result "
                    f"must fit, so the norm of\n`{tup(['1e6', '..', '1e6'])}` is fine. Upstream: "
                    f"`norm`.",
            f"fn norm(self: {T}) -> T", wide_chain(squares, "wide_sqrt"))
    return out


def vector_angle(v: Vec) -> Fn:
    S, T, n = v.S, v.ty(), v.n
    robust, upstream = ANGLE_GAS[n]
    doc = (
        "The smallest angle between two vectors, in `[0, π]` (up to the rounding of `atan2`); `0`\n"
        "when one of them is zero.\n\n"
        "Computed as `2 * atan2(|u - v|, |u + v|)` on the normalized vectors `u`, `v` (Kahan):\n"
        "unlike upstream's `acos(dot / (|a| * |b|))`, it cannot overflow on long vectors and stays\n"
        "accurate for nearly parallel ones. Panics when a norm does not fit. Upstream: `angle`.\n\n"
        f"The robustness costs {robust} gas against {upstream} for upstream's form, kept as\n"
        f"`bench_vector{n}_angle__alt_acos`: that one returns exactly 0 for two directions 2^-20 rad\n"
        "apart (its cosine floors to 1) and panics on vectors whose norms multiply out of range.")
    lines = [f"let n1 = R::norm{n}({v.args('self.{c}')});",
             f"let n2 = R::norm{n}({v.args('other.{c}')});",
             "if n1 == R::zero() || n2 == R::zero() {\nreturn R::zero();\n}"]
    if n in (3, 4):
        us, vs = [f"u{c}" for c in v.c], [f"v{c}" for c in v.c]
        lines += [f"let {tup(us)} = R::div{n}({v.args('self.{c}')}, n1);",
                  f"let {tup(vs)} = R::div{n}({v.args('other.{c}')}, n2);",
                  f"let u = {v.lit(lambda c: f'u{c}')};", f"let v = {v.lit(lambda c: f'v{c}')};"]
    else:
        lines += [f"let u = {v.lit(lambda c: f'R::div(self.{c}, n1)')};",
                  f"let v = {v.lit(lambda c: f'R::div(other.{c}, n2)')};"]
    lines += [f"let d = R::norm{n}({v.args('u.{c} - v.{c}')});",
              f"let s = R::norm{n}({v.args('u.{c} + v.{c}')});",
              "let half = Tr::atan2(d, s);", "half + half"]
    return Fn("angle", doc, f"fn angle(self: {T}, other: {T}) -> T", "\n".join(lines), False)


def vector_operators(v: Vec) -> list[str]:
    S, T, n = v.S, v.ty(), v.n
    out = []
    for trait, op, verb in (("Add", "+", "add"), ("Sub", "-", "sub")):
        out.append(f"/// `lhs {op} rhs`, component-wise. Exact; panics on overflow. Upstream: `{trait}`.\n"
                   f"pub impl {S}{trait}<T, +{trait}<T>, +Copy<T>, +Drop<T>> of {trait}<{T}> {{\n"
                   f"{INLINE}\nfn {verb}(lhs: {T}, rhs: {T}) -> {T} {{\n"
                   f"{v.lit(lambda c: f'lhs.{c} {op} rhs.{c}')}\n}}\n}}")
    out.append(f"/// `-a`, component-wise. Exact; panics on overflow (`-MIN`). Upstream: `Neg`.\n"
               f"pub impl {S}Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<{T}> {{\n"
               f"{INLINE}\nfn neg(a: {T}) -> {T} {{\n{v.lit(lambda c: f'-a.{c}')}\n}}\n}}")
    for trait, op, verb in (("Add", "+", "add"), ("Sub", "-", "sub")):
        out.append(f"/// `self {op}= rhs`. Exact; panics on overflow. Upstream: `{trait}Assign`.\n"
                   f"pub impl {S}{trait}Assign<T, +{trait}<T>, +Copy<T>, +Drop<T>> of "
                   f"{trait}Assign<{T}, {T}> {{\n{INLINE}\n"
                   f"fn {verb}_assign(ref self: {T}, rhs: {T}) {{\n"
                   f"self = {v.lit(lambda c: f'self.{c} {op} rhs.{c}')};\n}}\n}}")
    out.append("/// `self *= k` for a scalar `k`: `scale` in place (corelib's binary `*` is homogeneous, "
               "so `v * k`\n/// is the named method `scale`). Upstream: `MulAssign<T>`.\n"
               f"pub impl {S}MulAssign<T, +Mul<T>, +Copy<T>, +Drop<T>> of MulAssign<{T}, T> {{\n"
               f"{INLINE}\nfn mul_assign(ref self: {T}, rhs: T) {{\n"
               f"self = {v.lit(lambda c: f'self.{c} * rhs')};\n}}\n}}")
    out.append("/// `self /= k` for a scalar `k`: `unscale` in place. Upstream: `DivAssign<T>`.\n"
               f"pub impl {S}DivAssign<T, impl R: Real<T>, +Copy<T>, +Drop<T>> of DivAssign<{T}, T> {{\n"
               f"{INLINE}\nfn div_assign(ref self: {T}, rhs: T) {{\n"
               f"{vector_unscale(v, 'rhs', 'self = ')};\n}}\n}}")
    arr = f"[{', '.join(v.c)}]"
    out.append(f"/// `{arr}.into()`. Upstream: `From<[T; {n}]>`.\n"
               f"pub impl {S}FromArray<T> of Into<[T; {n}], {T}> {{\n{INLINE}\n"
               f"fn into(self: [T; {n}]) -> {T} {{\nlet {arr} = self;\n"
               f"{lit(S, [(c, c) for c in v.c])}\n}}\n}}")
    out.append(f"/// The components as a fixed-size array `{arr}`. Upstream: `Into<[T; {n}]>`.\n"
               f"pub impl {S}IntoArray<T> of Into<{T}, [T; {n}]> {{\n{INLINE}\n"
               f"fn into(self: {T}) -> [T; {n}] {{\nlet {lit(S, [(c, c) for c in v.c])} = self;\n"
               f"{arr}\n}}\n}}")
    return out


VECTOR_IMPL_BOUNDS = ["T", "impl R: Real<T>", "+Add<T>", "+Sub<T>", "+Mul<T>", "+Neg<T>",
                      "+PartialEq<T>", "+PartialOrd<T>", "+Copy<T>", "+Drop<T>"]
VECTOR_ANGLE_BOUNDS = ["T", "impl R: Real<T>", "impl Tr: Transcendental<T>", "+Add<T>", "+Sub<T>",
                       "+PartialEq<T>", "+Copy<T>", "+Drop<T>"]
# The bounds of an `AngleImpl` that also carries the base completion's `lp_norm` / `slerp`.
EXTENDED_ANGLE_BOUNDS = ["T", "impl R: Real<T>", "impl Tr: Transcendental<T>", "+Copy<T>",
                         "+Drop<T>", "+Drop<R::Wide>", "+Add<T>", "+Sub<T>", "+Mul<T>", "+Neg<T>",
                         "+PartialEq<T>", "+PartialOrd<T>"]


def slot_fns(order: list[str], templates: dict, specs: dict[str, Fn]) -> list[Fn]:
    """The methods of a shape in `order`: a specialisation fills its slot (replacing the template
    of that name), a template slot may expand to several methods (`axes`, `swizzles`)."""
    known = set(order)
    for name in specs:
        if name not in known and name not in templates:
            raise ValueError(f"specialisation `{name}` has no slot in the method order")
    out = []
    for name in order:
        if name in specs:
            out.append(specs[name])
            continue
        t = templates.get(name)
        if t is None:
            continue
        out.extend(t if isinstance(t, list) else [t])
    return out


def items_at(specs: Specs, anchor: str) -> list[str]:
    return [text for _, a, text in specs.items if a == anchor]


def with_docs(fns: list[Fn], specs: Specs) -> list[Fn]:
    """`fns`, the doc comment of each method named in a `// @doc <method>` section replaced."""
    for f in fns:
        if f.name in specs.docs:
            f.raw_doc = specs.docs[f.name].split("\n")
    return fns


def vector_surface(n: int, module: str) -> tuple[list[Fn], Fn | None]:
    """`(methods of Vector{n}Trait, Vector{n}AngleTrait::angle or None)`."""
    specs = read_specs(module)
    order = VECTOR_SURFACE.get(n, VECTOR_ORDER)
    templates = {k: t for k, t in vector_fns(Vec(n)).items() if k in order}
    fns = with_docs(slot_fns([k for k in VECTOR_ORDER if k in order], templates, specs.methods),
                    specs)
    if n in VECTOR_SURFACE:
        return fns, None
    return fns, specs.methods.get("angle") or vector_angle(Vec(n))


def render_vector(n: int, module: str, test_modules: list[str], extra: Extra) -> str:
    v = Vec(n)
    S, T = v.S, v.ty()
    specs = read_specs(module)
    fns, angle = vector_surface(n, module)
    fns = fns + extra.methods
    angles = ([angle] if angle else []) + extra.angle
    if n in VECTOR_SURFACE:
        others = []
    else:
        others = sorted({f"super::vector{m}::Vector{m}" for m in range(2, n + 2 if n < MAX_VECTOR
                                                                      else n) if m != n})
    scalar = "simba::scalar::{Real, Transcendental}" if angles else "simba::scalar::Real"
    uses = dedup(["core::ops::{AddAssign, DivAssign, MulAssign, SubAssign}", scalar] + others
                 + [f"super::row_vector{n}::RowVector{n}"] + specs.uses + extra.uses)
    mods = "\n".join(f"#[cfg(test)]\nmod {m};" for m in test_modules)
    decls = "\n".join(f.declaration() for f in fns)
    defs = "\n\n".join(f.definition(with_doc=False) for f in fns)
    blocks = [
        f"//! `{S}`: a statically sized {n}-dimensional column vector (upstream `nalgebra::{S}`).\n"
        f"//!\n"
        f"//! - `{S}Trait` / `{S}Impl`: constructors, component-wise operations, reductions, "
        f"products,\n//!   norms and interpolation, generic over a `simba::scalar::Real` scalar;\n"
        + (f"//! - `{S}AngleTrait` / `{S}AngleImpl`: `angle`, which additionally needs\n"
           f"//!   `simba::scalar::Transcendental`;\n" if angle else "")
        + (f"//! - the base completion (WP 8.2a): the rest of upstream's `Matrix` / `Vector` "
           f"API, in `{S}Trait`\n//!   and, for the operations that need `Transcendental`, "
           f"`{S}AngleTrait`;\n" if extra.methods else "")
        + f"//! - operators `+`, `-`, unary `-`, `+=`, `-=` between vectors, `*=` and `/=` by a "
        f"scalar, and\n//!   conversions from / to `[T; {n}]`: their impls live in this module, "
        f"where the\n//!   compiler finds them without any import;\n"
        f"//! - products with every conformable shape: `MatrixMul::mul_mat` (`self * rhs`) and\n"
        f"//!   `MatrixTrMul::tr_mul` (`selfᵀ * rhs`), impls of this module.\n//!\n"
        f"//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` "
        f"kernel (one\n//! floor rounding and one overflow check per output scalar); nothing "
        f"wraps silently." + ("\n" + specs.docs["module"] if "module" in specs.docs else ""),
        "\n".join(f"use {u};" for u in uses),
    ]
    if mods:
        blocks.append(mods)
    blocks.append(f"/// A {n}-dimensional column vector.\n"
                  f"#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]\n"
                  f"pub struct {S}<T> {{\n" + "\n".join(f"pub {c}: T," for c in v.c) + "\n}")
    blocks += items_at(specs, "struct") + extra.struct
    blocks.append(f"/// Operations of `{T}` over a `Real` scalar. By value, unrolled, no loop.\n"
                  f"pub trait {S}Trait<T> {{\n{decls}\n}}")
    if angles:
        blocks.append("/// `angle` needs inverse trigonometry, hence its own trait: scalars may "
                      "implement `Real` only.\n"
                      f"pub trait {S}AngleTrait<T> {{\n"
                      + "\n".join(a.declaration() for a in angles) + "\n}")
    impl_bounds = VECTOR_IMPL_BOUNDS + (["+Drop<R::Wide>"] if n > 4 else [])
    blocks.append(f"pub impl {S}Impl<\n{bounds(impl_bounds)}\n> of {S}Trait<T> {{\n{defs}\n}}")
    blocks += items_at(specs, "impl")
    if angles:
        angle_bounds = VECTOR_ANGLE_BOUNDS if not extra.angle else EXTENDED_ANGLE_BOUNDS
        blocks.append(f"pub impl {S}AngleImpl<\n{bounds(angle_bounds)}\n> of "
                      f"{S}AngleTrait<T> {{\n"
                      + "\n\n".join(a.definition(with_doc=False) for a in angles) + "\n}")
    blocks += vector_operators(v)
    blocks += items_at(specs, "end") + extra.end
    return "\n\n".join(blocks) + "\n"


# --------------------------------------------------------------------------------------------
# Square matrices: Matrix2, Matrix3, Matrix4, Matrix6
# --------------------------------------------------------------------------------------------

# `(section, methods)`: the order of `Matrix{n}Trait`. A section header is printed before the
# first method of the section.
MATRIX_ORDER = [
    ("constructors", ["new", "zeros", "identity", "from_diagonal", "from_diagonal_element",
                      "from_columns", "from_rows", "cross_matrix"]),
    ("accessors", ["diagonal"]),
    ("exact operations", ["transpose", "trace", "abs"]),
    ("products", ["scale", "component_mul"]),
    ("norms", ["norm_squared", "norm"]),
    ("determinant and inverse", ["determinant", "try_inverse"]),
    ("approximate equality", ["is_identity", "abs_diff_eq"]),
]

# The order of the crate-internal `Matrix{n}InternalTrait` (the hand-written order of each shape).
MATRIX_INTERNAL_ORDER = {
    2: ["from_outer", "cross_matrix_mul", "columns", "rows", "mul_transpose", "adjugate"],
    3: ["from_outer", "cross_matrix_mul", "columns", "rows", "mul_transpose", "adjugate"],
    4: ["columns", "rows", "from_outer", "cross_matrix_mul", "mul_transpose", "adjugate"],
    6: [],
}

# The surface of the matrices that do not carry all of `MATRIX_ORDER`: `Matrix6` keeps the methods
# of the hand-written block type it replaced (WP 8.1b-2; its determinant, inverse and
# decompositions are the LU / Cholesky kernels of `linalg`, DESIGN D6); 8.2 widens it.
MATRIX_SURFACE = {
    6: ["new", "zeros", "identity", "from_diagonal", "from_diagonal_element", "diagonal",
        "transpose", "trace", "abs", "scale", "is_identity", "abs_diff_eq"],
}

MATRIX_IMPL_BOUNDS = ["T", "impl R: Real<T>", "+Copy<T>", "+Drop<T>", "+Drop<R::Wide>", "+Add<T>",
                      "+Sub<T>", "+Mul<T>", "+Neg<T>", "+PartialEq<T>", "+PartialOrd<T>"]


def section(name: str, width: int) -> str:
    head = f"// --- {name} "
    return head + "-" * (width - len(head))


class Mat:
    def __init__(self, n: int):
        self.n = n
        self.S = f"Matrix{n}"
        self.V = Vec(n)
        self.col_major = [(i, j) for j in range(n) for i in range(n)]
        self.row_major = [(i, j) for i in range(n) for j in range(n)]

    @staticmethod
    def f(i: int, j: int) -> str:
        return f"m{i + 1}{j + 1}"

    def ty(self) -> str:
        return f"{self.S}<T>"

    def lit(self, g) -> str:
        """Struct literal, column-major, `g(i, j)` the value of component (i, j)."""
        return lit(self.S, [(self.f(i, j), g(i, j)) for i, j in self.col_major])


def matrix_fns(m: Mat) -> dict[str, Fn | list[Fn]]:
    S, T, n, f = m.S, m.ty(), m.n, m.f
    V, VT, c = m.V.S, m.V.ty(), COORDS
    nn = n * n
    diag = lambda on, off="R::zero()": m.lit(lambda i, j: on(i) if i == j else off)  # noqa: E731
    out: dict[str, Fn | list[Fn]] = {}
    params = ", ".join(f"{f(i, j)}: T" for i, j in m.row_major)
    out["new"] = Fn("new", f"The matrix with the given components, in ROW-major order. Upstream: "
                           f"`{S}::new`.", f"fn new({params}) -> {T}",
                    m.lit(lambda i, j: f(i, j)))
    out["zeros"] = Fn("zeros", f"The zero matrix. Upstream: `{S}::zeros`.", f"fn zeros() -> {T}",
                      m.lit(lambda i, j: "R::zero()"))
    out["identity"] = Fn("identity", f"The identity matrix. Upstream: `{S}::identity`.",
                         f"fn identity() -> {T}", diag(lambda i: "R::one()"))
    what = "`diag(d.x, d.y)`" if n == 2 else "`diag(d)`"
    out["from_diagonal"] = Fn("from_diagonal", f"The diagonal matrix {what}. Upstream: "
                                               f"`{S}::from_diagonal`.",
                              f"fn from_diagonal(d: {VT}) -> {T}", diag(lambda i: f"d.{c[i]}"))
    out["from_diagonal_element"] = Fn(
        "from_diagonal_element", f"The matrix `e * I`. Upstream: `{S}::from_diagonal_element`.",
        f"fn from_diagonal_element(e: T) -> {T}", diag(lambda i: "e"))
    for name, arg, noun, pick in (("from_columns", "c", "columns", lambda i, j: (j, i)),
                                  ("from_rows", "r", "rows", lambda i, j: (i, j))):
        which = "`c1`, `c2`" if arg == "c" else "`r1`, `r2`"
        what = which if n == 2 else "the given vectors"
        args = ", ".join(f"{arg}{k + 1}: {VT}" for k in range(n))
        out[name] = Fn(name, f"The matrix whose {noun} are {what}. Upstream: `{S}::{name}`.",
                       f"fn {name}({args}) -> {T}",
                       m.lit(lambda i, j, arg=arg, pick=pick:
                             f"{arg}{pick(i, j)[0] + 1}.{c[pick(i, j)[1]]}"))
    what = "The diagonal `(m11, m22)`." if n == 2 else "The diagonal."
    out["diagonal"] = Fn("diagonal", f"{what} Upstream: `diagonal`.",
                         f"fn diagonal(self: {T}) -> {VT}",
                         m.V.lit(lambda k: f"self.{f(c.index(k), c.index(k))}"))
    out["transpose"] = Fn("transpose", "The transpose. Exact. Upstream: `transpose`.",
                          f"fn transpose(self: {T}) -> {T}", m.lit(lambda i, j: f"self.{f(j, i)}"))
    what = "`m11 + m22`." if n == 2 else "Sum of the diagonal."
    out["trace"] = Fn("trace", f"{what} Exact; panics on overflow. Upstream: `trace`.",
                      f"fn trace(self: {T}) -> T", " + ".join(f"self.{f(i, i)}" for i in range(n)))
    out["abs"] = Fn("abs", "Component-wise absolute value. Panics on the scalar's `MIN`. Upstream: "
                           "`abs`.", f"fn abs(self: {T}) -> {T}",
                    m.lit(lambda i, j: f"R::abs(self.{f(i, j)})"))
    out["scale"] = Fn("scale", "`self * k`: each component is one floored product. Panics on "
                               "overflow.\nUpstream: `self * k`.", f"fn scale(self: {T}, k: T) -> {T}",
                      m.lit(lambda i, j: f"self.{f(i, j)} * k"))
    out["component_mul"] = Fn(
        "component_mul", "Component-wise (Hadamard) product: each component is one floored "
                         "product. Panics on\noverflow. Upstream: `component_mul`.",
        f"fn component_mul(self: {T}, rhs: {T}) -> {T}",
        m.lit(lambda i, j: f"self.{f(i, j)} * rhs.{f(i, j)}"))
    wide = n > 4
    cm = [f"self.{f(i, j)}" for i, j in m.col_major]
    if nn <= 4:
        out["norm_squared"] = Fn(
            "norm_squared", f"Squared Frobenius norm, one rounding (`norm_squared{nn}`). Panics on "
                            f"overflow.\nUpstream: `norm_squared`.",
            f"fn norm_squared(self: {T}) -> T", f"R::norm_squared{nn}({', '.join(cm)})")
        norm_body = f"R::norm{nn}({', '.join(cm)})"
    else:
        out["norm_squared"] = Fn(
            "norm_squared", f"Squared Frobenius norm: the {nn} squares are accumulated exactly "
                            f"(wide accumulator) and\nrounded once. Panics on overflow. Upstream: "
                            f"`norm_squared`.",
            f"fn norm_squared(self: {T}) -> T", wide_chain(cm, "wide_rescale"))
        norm_body = wide_chain(cm, "wide_sqrt")
    out["norm"] = Fn("norm", "Frobenius norm: square root of the exact, unscaled sum of squares "
                             "(one rounding, no\nintermediate overflow: only the result must fit). "
                             "Upstream: `norm`.", f"fn norm(self: {T}) -> T", norm_body)
    out["is_identity"] = Fn(
        "is_identity", "Whether every component is within `ulps` smallest units of the "
                       "identity's.\nUpstream: `is_identity(eps)`, with the tolerance in raw units "
                       "instead of a float epsilon.", f"fn is_identity(self: {T}, ulps: u64) -> bool",
        "\n&& ".join(f"R::abs_diff_eq(self.{f(i, j)}, {'R::one()' if i == j else 'R::zero()'}, ulps)"
                     for i, j in m.col_major), False)
    out["abs_diff_eq"] = Fn(
        "abs_diff_eq", "Whether every component of `self` is within `ulps` smallest units of "
                       "`other`'s.\nUpstream: `abs_diff_eq`, with the tolerance in raw units "
                       "instead of a float epsilon.",
        f"fn abs_diff_eq(self: {T}, other: {T}, ulps: u64) -> bool",
        "\n&& ".join(f"R::abs_diff_eq(self.{f(i, j)}, other.{f(i, j)}, ulps)"
                     for i, j in m.col_major), False)
    if wide:
        # The comparisons are inlined above 4x4: on `Matrix6` the 36-term chain as a call costs
        # about 103 000 gas whatever the input, inlined 78 000 (all compared) to 32 000.
        out["is_identity"].inline = out["abs_diff_eq"].inline = True
    return out


def wide_chain(xs: list, final: str) -> str:
    """The exact sum of the products `a * b` of `xs` (pairs, or single terms for their squares) in
    `Real::Wide`, then `R::<final>` (one rounding), as `let w = ..` statements."""
    ps = [x if isinstance(x, tuple) else (x, x) for x in xs]
    lines = [f"let w = R::wide_add_prod(R::wide_zero(), {ps[0][0]}, {ps[0][1]});"]
    lines += [f"let w = R::wide_add_prod(w, {a}, {b});" for a, b in ps[1:-1]]
    lines.append(f"R::{final}(R::wide_add_prod(w, {ps[-1][0]}, {ps[-1][1]}))")
    return "\n".join(lines)


def wide_expr(pairs: list[tuple[str, str]]) -> str:
    """The exact sum of products of `pairs` in `Real::Wide`, rescaled once, as ONE nested
    expression (the form of the hand-written `Matrix6` products)."""
    acc = "R::wide_zero()"
    for a, b in pairs:
        acc = f"R::wide_add_prod({acc}, {a}, {b})"
    return f"R::wide_rescale({acc})"


def matrix_internal_fns(m: Mat) -> dict[str, Fn | list[Fn]]:
    S, T, n, f = m.S, m.ty(), m.n, m.f
    VT, c = m.V.ty(), COORDS
    out: dict[str, Fn | list[Fn]] = {}
    out["from_outer"] = Fn(
        "from_outer", "The outer product `a * bᵀ`: each component is one floored product. Panics "
                      "with the\nscalar's overflow error. Upstream: `a * b.transpose()`.",
        f"fn from_outer(a: {VT}, b: {VT}) -> {T}", m.lit(lambda i, j: f"a.{c[i]} * b.{c[j]}"))
    cols = []
    for j in range(n):
        doc = (f"{ORDINALS[j + 1]} column." if n == 2 else f"Column {j + 1}.")
        cols.append(Fn(f"column{j + 1}", f"{doc} Upstream: `column({j})`.",
                       f"fn column{j + 1}(self: {T}) -> {VT}",
                       m.V.lit(lambda k, j=j: f"self.{f(c.index(k), j)}")))
    out["columns"] = cols
    if n > 2:
        out["rows"] = [Fn(f"row{i + 1}", f"Row {i + 1}, as a (column) vector. Upstream: "
                                         f"`row({i}).transpose()`.",
                          f"fn row{i + 1}(self: {T}) -> {VT}",
                          m.V.lit(lambda k, i=i: f"self.{f(i, c.index(k))}")) for i in range(n)]
    return out


def matrix_operators(m: Mat, docs: dict[str, str]) -> list[str]:
    S, T, n, f = m.S, m.ty(), m.n, m.f
    wide = n > 4
    out = [section("operators", 100)]
    for trait, op, verb in (("Add", "+", "add"), ("Sub", "-", "sub")):
        out.append(f"/// `a {op} b`, component-wise. Exact; panics on overflow.\n"
                   f"pub impl {S}{trait}<T, +{trait}<T>, +Copy<T>, +Drop<T>> of {trait}<{T}> {{\n"
                   f"{INLINE}\nfn {verb}(lhs: {T}, rhs: {T}) -> {T} {{\n"
                   f"{m.lit(lambda i, j: f'lhs.{f(i, j)} {op} rhs.{f(i, j)}')}\n}}\n}}")
    out.append(f"/// `-a`, component-wise. Exact; panics on the scalar's `MIN`.\n"
               f"pub impl {S}Neg<T, +Neg<T>, +Copy<T>, +Drop<T>> of Neg<{T}> {{\n"
               f"{INLINE}\nfn neg(a: {T}) -> {T} {{\n{m.lit(lambda i, j: f'-a.{f(i, j)}')}\n}}\n}}")
    if wide:
        product = m.lit(lambda i, j: wide_expr([(f"lhs.{f(i, k)}", f"rhs.{f(k, j)}")
                                                for k in range(n)]))
        doc = (f"/// `a * b` (matrix product): {n * n} {n}-term `Real::Wide` accumulations, ONE "
               f"rounding and one overflow\n/// check per output scalar. Panics on overflow.")
    else:
        product = m.lit(lambda i, j: f"R::sum_prod{n}" + tup(f"lhs.{f(i, k)}, rhs.{f(k, j)}"
                                                               for k in range(n)))
        doc = (f"/// `a * b` (matrix product): {n * n} `sum_prod{n}`, one rounding per component. "
               f"Panics on overflow.")
    real = "impl R: Real<T>, +Copy<T>, +Drop<T>" + (", +Drop<R::Wide>" if wide else "")
    out.append(f"{docs.get('Mul', doc)}\n"
               f"pub impl {S}Mul<T, {real}> of Mul<{T}> {{\n"
               f"fn mul(lhs: {T}, rhs: {T}) -> {T} {{\n{product}\n}}\n}}")
    for trait, op, verb, bound in (("Add", "+", "add", "+Add<T>"), ("Sub", "-", "sub", "+Sub<T>")):
        out.append(f"/// `a {op}= b`.\n"
                   f"pub impl {S}{trait}Assign<T, {bound}, +Copy<T>, +Drop<T>> of "
                   f"{trait}Assign<{T}, {T}> {{\n{INLINE}\n"
                   f"fn {verb}_assign(ref self: {T}, rhs: {T}) {{\nself = self {op} rhs;\n}}\n}}")
    out.append(f"/// `a *= b` (matrix product, `a = a * b`).\n"
               f"pub impl {S}MulAssign<T, {real}> of "
               f"MulAssign<{T}, {T}> {{\n{INLINE}\n"
               f"fn mul_assign(ref self: {T}, rhs: {T}) {{\nself = self * rhs;\n}}\n}}")
    return out


def matrix_sections(n: int, module: str) -> list[tuple[str, list[Fn]]]:
    """The methods of `Matrix{n}Trait`, by section."""
    specs = read_specs(module)
    templates = matrix_fns(Mat(n))
    keep = MATRIX_SURFACE.get(n)
    if keep is not None:
        templates = {k: t for k, t in templates.items() if k in keep}
    order = [name for _, names in MATRIX_ORDER for name in names]
    slot_fns(order, templates, specs.methods)  # validates the specialisation names
    out = []
    for title, names in MATRIX_ORDER:
        fns = slot_fns(names, {k: templates[k] for k in names if k in templates},
                       {k: specs.methods[k] for k in names if k in specs.methods})
        with_docs(fns, specs)
        if fns:
            out.append((title, fns))
    return out


def surface(shape: tuple[int, int]) -> list[Fn]:
    """Every public method of the shape's traits (the comparison's subject)."""
    module = LIBRARY_SHAPES[shape]
    if shape[1] == 1:
        fns, angle = vector_surface(shape[0], module)
        return fns + ([angle] if angle else [])
    return [f for _, fns in matrix_sections(shape[0], module) for f in fns]


def render_matrix(n: int, module: str, test_modules: list[str], extra: Extra) -> str:
    m = Mat(n)
    S, T = m.S, m.ty()
    specs = read_specs(module)
    sections = ["\n\n".join([section(title, 94)] + [x.definition() for x in fns])
                for title, fns in matrix_sections(n, module)]
    if extra.methods:
        sections.append("\n\n".join([section("base completion (WP 8.2a)", 94)]
                                      + [x.definition() for x in extra.methods]))
    internal = slot_fns(MATRIX_INTERNAL_ORDER[n], matrix_internal_fns(m), specs.internal)
    scalar = "simba::scalar::{Real, Transcendental}" if extra.angle else "simba::scalar::Real"
    uses = dedup(["core::ops::{AddAssign, MulAssign, SubAssign}", scalar,
                  f"super::vector{n}::Vector{n}"] + specs.uses + extra.uses)
    blocks = [
        f"//! `{S}`: a statically sized {n}x{n} matrix (upstream `nalgebra::{S}`).\n//!\n"
        f"//! Every sum of products goes through a fused `Real` kernel: one rounding (floor) and "
        f"one overflow\n//! check per output scalar. Operators (`+`, `-`, unary `-`, `*` between "
        f"matrices and their\n//! assigning forms) are implemented in this module, so they need "
        f"no import; the other operations\n//! are methods of `{S}Trait`, the products with every "
        f"conformable shape `MatrixMul::mul_mat`\n//! (`self * rhs`) and `MatrixTrMul::tr_mul` "
        f"(`selfᵀ * rhs`)."
        + ("\n" + specs.docs["module"] if "module" in specs.docs else ""),
        "\n".join(f"use {u};" for u in uses),
    ]
    if test_modules:
        blocks.append("\n".join(f"#[cfg(test)]\nmod {t};" for t in test_modules))
    blocks += [
        f"/// A {n}x{n} matrix. `mRC` is the component at row `R`, column `C`.\n///\n"
        f"/// Fields are declared in column-major order, so `Serde` matches upstream's storage "
        f"order, while\n/// `new` takes its arguments in row-major order like upstream.\n"
        f"#[derive(Copy, Drop, PartialEq, Serde, Default, Debug, Hash)]\n"
        f"pub struct {S}<T> {{\n" + "\n".join(f"pub {m.f(i, j)}: T," for i, j in m.col_major)
        + "\n}",
    ]
    blocks += items_at(specs, "struct") + extra.struct
    blocks.append(f"/// Methods of `{T}` for any `Real` scalar.\n#[generate_trait]\n"
                  f"pub impl {S}Impl<\n{bounds(MATRIX_IMPL_BOUNDS)}\n> of {S}Trait<T> {{\n"
                  + "\n\n".join(sections) + "\n}")
    blocks += items_at(specs, "impl")
    if extra.angle:
        blocks.append(f"/// The operations of `{T}` that need `Transcendental` (inverse "
                      f"trigonometry, `exp`, `ln`):\n/// a scalar may implement `Real` only.\n"
                      f"#[generate_trait]\npub impl {S}AngleImpl<\n{bounds(EXTENDED_ANGLE_BOUNDS)}"
                      f"\n> of {S}AngleTrait<T> {{\n"
                      + "\n\n".join(x.definition() for x in extra.angle) + "\n}")
    if internal:
        doc = specs.docs.get("internal")
        if doc is None:
            raise ValueError(f"{module}: the internal impl needs a `// @doc internal` section")
        blocks.append(f"{doc}\n#[generate_trait]\npub(crate) impl {S}InternalImpl<\n"
                      f"{bounds(MATRIX_IMPL_BOUNDS)}\n> of {S}InternalTrait<T> {{\n"
                      + "\n\n".join(x.definition() for x in internal) + "\n}")
    blocks += matrix_operators(m, specs.docs)
    blocks += items_at(specs, "end") + extra.end
    return "\n\n".join(blocks) + "\n"


def render(shape: tuple[int, int], tests: bool = True, extra: Extra | None = None) -> str:
    """The library file of `shape`; `tests = False` omits its test modules (staging package);
    `extra`: the aliases and products of `shapes.py` (none in the staging package)."""
    module = LIBRARY_SHAPES[shape]
    r, c = shape
    mods = sorted(p.stem for p in (BASE / module).glob("*.cairo")) if tests else []
    extra = extra or Extra()
    if c == 1:
        body = render_vector(r, module, mods, extra)
    else:
        body = render_matrix(r, module, mods, extra)
    return HEADER.format(module=module) + body


HEADER = """// Generated by tools/shapegen/shapegen.py: do not edit by hand. Templates:
// tools/shapegen/library.py, shapes.py; kernels: tools/shapegen/specialisations/{module}.cairo.
"""
