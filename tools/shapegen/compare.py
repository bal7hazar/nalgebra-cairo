"""Staging comparison of the generated library shapes against `crates/nalgebra` (DESIGN.md §2.2).

`shapegen.py --compare DIR` writes a throw-away Scarb package that holds the GENERATED
`Vector2/3/4/6`, `Matrix2/3/4/6` (library templates + specialisations, test modules omitted, with
copies of the crate-internal `SymMatrix2/3` they return) next to a dependency on the CURRENT
`crates/nalgebra`, and one comparison module:

* `test_<module>_<op>_bit_identical`: the `Serde` images of both results are equal on three
  inputs (raw magnitudes below 2^4, 2^0 and 2^-2: the second and third reach the pre-scaled branch
  of `try_inverse`);
* `bench_<module>_<op>__generated` / `__handwritten`: the same black-boxed inputs, the result
  black-boxed; `compare.sh` reports the raw `l2_gas` of both and whether they are equal.

`<op>` is every public method of `<Struct>Trait` / `<Struct>AngleTrait` and every operator /
conversion impl of the shape. The crate-internal traits (`Matrix3InternalTrait`, ...) cannot be
called from another crate; they are spliced verbatim and covered by the crate's own tests and
benches under the gas snapshot.

WP 8.1b-2: the hand-written `Vector6` / `Matrix6` were block types (`v.a.x`, `m.m21.m32`); their
inputs are written as block literals holding the same components, and a hand-written `Matrix6`
result is compared through its flat column-major image (`himage6`), the order of the generated
flat `Serde`. The hand-written `Vector6` already serialised as `x y z w a b`.

Run it BEFORE the generated files replace the hand-written ones (afterwards it compares the
generator with itself): `tools/shapegen/compare.sh`.
"""

import random
import re
from pathlib import Path

import library as L

ROOT = L.ROOT
SEEDS = (("large", 36), ("unit", 32), ("small", 30))

MANIFEST = """[package]
name = "shapegen_compare"
version = "0.1.0"
edition = "2024_07"
cairo-version = "2.19.4"
publish = false

[dependencies]
simba = "0.1.0"

[dev-dependencies]
fixed = "0.3.0"
nalgebra = {{ path = "{root}/crates/nalgebra" }}
nalgebra_testing = {{ path = "{root}/crates/testing" }}
snforge_std = "0.61.0"
assert_macros = "2.19.4"

[tool.fmt]
sort-module-level-items = true
max-line-length = 100

[tool.scarb]
allow-prebuilt-plugins = ["snforge_std"]
"""

STRUCT = re.compile(r"\b(Vector[2346]|Matrix[2346])\b")

# The hand-written block layout of the 6D shapes (WP 8.1b-2): flat field -> block path.
H6_VECTOR = {c: f"{'a' if k < 3 else 'b'}.{'xyz'[k % 3]}" for k, c in enumerate(L.COORDS)}


def h6_matrix(i: int, j: int) -> str:
    """The hand-written path of the scalar at row `i`, column `j` (0-based) of a `Matrix6`."""
    return f"m{i // 3 + 1}{j // 3 + 1}.m{i % 3 + 1}{j % 3 + 1}"


def field_path(side: "Side", struct: str, field: str) -> str:
    """`field` of a `struct` value on `side` (`x` → `a.x` on a hand-written `Vector6`)."""
    if side.tag == "h" and struct == "Vector6":
        return H6_VECTOR[field]
    return field


def strip_tests(text: str) -> str:
    """The module without its `#[cfg(test)]` declarations and inline test module."""
    text = re.sub(r"#\[cfg\(test\)\]\nmod \w+;\n", "", text)
    cut = text.find("#[cfg(test)]\nmod tests {")
    return text if cut < 0 else text[:cut].rstrip("\n") + "\n"


def params(sig: str) -> tuple[str, list[tuple[str, str]], str]:
    """`fn name(a: A, b: B) -> R` as `(name, [(a, A), (b, B)], R)` (`R` = '' for none)."""
    m = re.fullmatch(r"fn (\w+)\(([^)]*)\)(?: -> (.+))?", sig)
    if not m:
        raise ValueError(sig)
    out = []
    if m.group(2).strip():
        for p in m.group(2).split(", "):
            name, ty = p.split(": ", 1)
            out.append((name.replace("ref ", ""), ty))
    return m.group(1), out, m.group(3) or ""


class Side:
    """Type and trait names of one side: generated (`g`, this package) or hand-written (`h`)."""

    def __init__(self, tag: str):
        self.tag = tag

    def ty(self, ty: str) -> str:
        ty = ty.replace("<T>", "<Fixed>").replace("[T;", "[Fixed;")
        ty = re.sub(r"\bT\b", "Fixed", ty)
        return STRUCT.sub(r"H\1", ty) if self.tag == "h" else ty

    def name(self, name: str) -> str:
        return f"H{name}" if self.tag == "h" else name


def value(side: Side, ty: str, rng: random.Random, bits: int) -> str:
    """A literal of type `ty` (`T`, `u64`, a shape) with random raw components."""
    raw = lambda: rng.randrange(-(1 << bits), 1 << bits)  # noqa: E731
    if ty == "T":
        return f"fx({raw()})"
    if ty == "u64":
        return f"{rng.randrange(0, 1 << 16)}"
    m = re.fullmatch(r"(Vector|Matrix)(\d)<T>", ty)
    if not m:
        raise ValueError(f"no input generator for `{ty}`")
    n = int(m.group(2))
    if m.group(1) == "Vector":
        comps = {(k, 0): raw() for k in range(n)}
    else:
        comps = {(i, j): raw() for j in range(n) for i in range(n)}
    if side.tag == "h" and n == 6:
        return h6_literal(m.group(1), comps)
    fields = ([(c, (k, 0)) for k, c in enumerate(L.COORDS[:n])] if m.group(1) == "Vector"
              else [(L.Mat.f(i, j), (i, j)) for j in range(n) for i in range(n)])
    body = ", ".join(f"{f}: fx({comps[ij]})" for f, ij in fields)
    return f"{side.name(m.group(1) + m.group(2))} {{ {body} }}"


def h6_literal(kind: str, comps: dict[tuple[int, int], int]) -> str:
    """A hand-written block `Vector6` / `Matrix6` literal holding the components `comps`."""
    if kind == "Vector":
        blocks = [", ".join(f"{c}: fx({comps[(3 * b + k, 0)]})" for k, c in enumerate("xyz"))
                  for b in range(2)]
        return f"HVector6 {{ a: HVector3 {{ {blocks[0]} }}, b: HVector3 {{ {blocks[1]} }} }}"
    out = []
    for bi, bj in ((0, 0), (1, 0), (0, 1), (1, 1)):
        body = ", ".join(f"m{i + 1}{j + 1}: fx({comps[(3 * bi + i, 3 * bj + j)]})"
                         for j in range(3) for i in range(3))
        out.append(f"m{bi + 1}{bj + 1}: HMatrix3 {{ {body} }}")
    return f"HMatrix6 {{ {', '.join(out)} }}"


class Op:
    """One comparable operation: named inputs, a result type, an expression per side."""

    def __init__(self, name: str, inputs: list[tuple[str, str]], ret: str, expr, mutate=None):
        self.name, self.inputs, self.ret, self.expr, self.mutate = name, inputs, ret, expr, mutate

    def lines(self, side: Side, values: dict[str, str] | None, bench: bool) -> list[str]:
        """Bindings of the inputs (black-boxed in a bench), then `let r_<side>: R = ..;`."""
        out = []
        for name, ty in self.inputs:
            v = values[f"{side.tag}:{name}"]
            out.append(f"let {'mut ' if name == self.mutate else ''}{name}_{side.tag}: "
                       f"{side.ty(ty)} = {f'black_box({v})' if bench else v};")
        args = {name: f"{name}_{side.tag}" for name, _ in self.inputs}
        if self.mutate:
            out.append(self.expr(side, args) + ";")
            out.append(f"let r_{side.tag}: {side.ty(self.ret)} = {args[self.mutate]};")
        else:
            out.append(f"let r_{side.tag}: {side.ty(self.ret)} = {self.expr(side, args)};")
        return out


def method_op(shape: tuple[int, int], fn: L.Fn, trait: str) -> Op:
    name, ps, ret = params(fn.sig)
    if ps and ps[0][0] == "self":
        def expr(side, a, name=name, ps=ps):
            rest = ", ".join(a[p] for p, _ in ps[1:])
            return f"{a['self']}.{name}({rest})"
    else:
        def expr(side, a, name=name, ps=ps):
            return f"{side.name(trait)}::{name}({', '.join(a[p] for p, _ in ps)})"
    return Op(name, [(p, t) for p, t in ps], ret, expr)


def operator_ops(shape: tuple[int, int]) -> list[Op]:
    r, c = shape
    S = f"Vector{r}" if c == 1 else f"Matrix{r}"
    T = f"{S}<T>"
    ops = [
        Op("add", [("a", T), ("b", T)], T, lambda s, a: f"{a['a']} + {a['b']}"),
        Op("sub", [("a", T), ("b", T)], T, lambda s, a: f"{a['a']} - {a['b']}"),
        Op("neg", [("a", T)], T, lambda s, a: f"-{a['a']}"),
        Op("add_assign", [("a", T), ("b", T)], T, lambda s, a: f"{a['a']} += {a['b']}", "a"),
        Op("sub_assign", [("a", T), ("b", T)], T, lambda s, a: f"{a['a']} -= {a['b']}", "a"),
    ]
    if c == 1:
        arr = f"[T; {r}]"
        ops += [
            Op("mul_assign", [("a", T), ("k", "T")], T, lambda s, a: f"{a['a']} *= {a['k']}", "a"),
            Op("div_assign", [("a", T), ("k", "T")], T, lambda s, a: f"{a['a']} /= {a['k']}", "a"),
            Op("into_array", [("a", T)], arr, lambda s, a: f"{a['a']}.into()"),
            Op("from_array", [("a", T)], T,
               lambda s, a: "[" + ", ".join(f"{a['a']}.{field_path(s, S, x)}"
                                            for x in L.COORDS[:r]) + "].into()"),
        ]
    else:
        ops += [
            Op("mul", [("a", T), ("b", T)], T, lambda s, a: f"{a['a']} * {a['b']}"),
            Op("mul_assign", [("a", T), ("b", T)], T, lambda s, a: f"{a['a']} *= {a['b']}", "a"),
        ]
    return ops


def shape_ops(shape: tuple[int, int]) -> list[Op]:
    r, c = shape
    S = f"Vector{r}" if c == 1 else f"Matrix{r}"
    ops = []
    for fn in L.surface(shape):
        trait = f"{S}AngleTrait" if fn.name == "angle" else f"{S}Trait"
        ops.append(method_op(shape, fn, trait))
    return ops + operator_ops(shape)


def render_compare() -> str:
    uses = {"fixed::Fixed", "nalgebra_testing::black_box"}
    fns = []
    for shape, module in L.LIBRARY_SHAPES.items():
        r, c = shape
        S = f"Vector{r}" if c == 1 else f"Matrix{r}"
        traits = [S, f"{S}Trait"] + ([f"{S}AngleTrait"] if c == 1 and r not in L.VECTOR_SURFACE
                                     else [])
        uses.add(f"crate::{module}::{{{', '.join(traits)}}}")
        uses.add(f"nalgebra::base::{module}::{{{', '.join(f'{t} as H{t}' for t in traits)}}}")
        for op in shape_ops(shape):
            group = f"{module}_{op.name}"
            rng = random.Random(f"compare/{group}")
            cases = []
            for _, bits in SEEDS:
                vals = {}
                for name, ty in op.inputs:
                    state = rng.getstate()
                    vals[f"g:{name}"] = value(Side("g"), ty, rng, bits)
                    rng.setstate(state)
                    vals[f"h:{name}"] = value(Side("h"), ty, rng, bits)
                cases.append(vals)
            body = []
            for vals in cases:
                body.append("{")
                body += op.lines(Side("g"), vals, False) + op.lines(Side("h"), vals, False)
                h = "himage6" if op.ret == "Matrix6<T>" else "image"
                body.append(f"assert!(image(r_g) == {h}(r_h));")
                body.append("}")
            fns.append(f"#[test]\nfn test_{group}_bit_identical() {{\n" + "\n".join(body) + "\n}")
            for tag, variant in (("g", "generated"), ("h", "handwritten")):
                lines = op.lines(Side(tag), cases[0], True)
                lines.append(f"black_box(r_{tag});")
                fns.append(f"#[test]\n#[inline(never)]\nfn bench_{group}__{variant}() {{\n"
                           + "\n".join(lines) + "\n}")
    imports = "\n".join(f"use {u};" for u in sorted(uses))
    return (f"//! Generated by tools/shapegen/shapegen.py --compare: the generated shapes against "
            f"the hand-written ones\n//! of `crates/nalgebra` (see `tools/shapegen/compare.py`)."
            f"\n\n{imports}\n\n"
            "fn fx(raw: i64) -> Fixed {\n    Fixed { raw }\n}\n\n"
            "/// The `Serde` image of a value: equal images are equal bits.\n"
            "fn image<S, +Serde<S>, +Drop<S>>(value: S) -> Span<felt252> {\n"
            "    let mut out: Array<felt252> = array![];\n    value.serialize(ref out);\n"
            "    out.span()\n}\n\n"
            "/// The flat column-major image of a hand-written block `Matrix6` (the generated\n"
            "/// `Serde` order).\n"
            "fn himage6(m: HMatrix6<Fixed>) -> Span<felt252> {\n"
            "    let mut out: Array<felt252> = array![];\n"
            + "".join(f"    m.{h6_matrix(i, j)}.serialize(ref out);\n"
                      for j in range(6) for i in range(6))
            + "    out.span()\n}\n\n" + "\n\n".join(fns) + "\n")


def generate(pkg: Path, root: str) -> None:
    src = pkg / "src"
    src.mkdir(parents=True, exist_ok=True)
    (pkg / "Scarb.toml").write_text(MANIFEST.format(root=root))
    mods = []
    for shape, module in L.LIBRARY_SHAPES.items():
        (src / f"{module}.cairo").write_text(strip_tests(L.render(shape, tests=False)))
        mods.append(module)
    for sym in ("sym_matrix2", "sym_matrix3"):
        (src / f"{sym}.cairo").write_text(strip_tests((L.BASE / f"{sym}.cairo").read_text()))
        mods.append(sym)
    lib = "".join(f"pub mod {m};\n" for m in sorted(mods)) + "#[cfg(test)]\nmod compare;\n"
    (src / "lib.cairo").write_text("//! Staging comparison (tools/shapegen/compare.py).\n\n" + lib)
    (src / "compare.cairo").write_text(render_compare())
