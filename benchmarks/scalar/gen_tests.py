#!/usr/bin/env python3
"""Generate the representation benchmarks (`src/tests/rep_*.cairo`).

Every representation gets the exact same battery of operations with the same *values*; expected
results are computed here with exact rational arithmetic and the rounding mode of the
representation. Groups are `<op>_<storage>`: the baseline of a group uses the same storage type
(same `black_box` inputs, same equality assertion on the same result type), so that
net = raw - baseline is the cost of the operation only.

Usage: python3 gen_tests.py   (from benchmarks/scalar)
"""

from fractions import Fraction as F
from math import floor
from pathlib import Path

OUT = Path(__file__).parent / "src" / "tests"


class Rep:
    def __init__(self, key, mod, ty, store, frac, rounding, int_ty, signmag=False, extra=None):
        self.key, self.mod, self.ty, self.store = key, mod, ty, store
        self.frac, self.rounding, self.int_ty, self.signmag = frac, rounding, int_ty, signmag
        self.extra = extra or {}

    def raw(self, v):
        r = F(v) * (1 << self.frac)
        assert r.denominator == 1, (self.ty, v)
        return int(r)

    def round(self, v):
        """Value -> raw with the rounding mode of the representation."""
        r = F(v) * (1 << self.frac)
        if self.rounding == "floor":
            return floor(r)
        return int(r)  # truncation toward zero

    def lit_raw(self, r):
        if self.signmag:
            return f"{self.ty} {{ mag: {hex(abs(r))}, sign: {'true' if r < 0 else 'false'} }}"
        return f"{self.ty} {{ raw: {'-' if r < 0 else ''}{hex(abs(r))} }}"

    def lit(self, v):
        return self.lit_raw(self.raw(v))


BOUNDED_EXTRA = {
    "mul": [("bounded_precheck", "bi64::mul_precheck(a, b)"), ("bounded_u128", "bi64::mul_via_u128(a, b)"),
            ("bounded_trunc", "bi64::mul_trunc(a, b)")],
    "add": [("bounded_bi", "bi64::add_bounded(a, b)")],
    "sub": [("bounded_bi", "bi64::sub_bounded(a, b)")],
    "neg": [("bounded_bi", "bi64::neg_bounded(a)")],
    "abs": [("bounded_bi", "bi64::abs_bounded(a)")],
    "dot3": [("bounded_fused", "bi64::dot3(ax, ay, az, bx, by, bz)")],
    "cross3": [("bounded_fused", "bi64::cross3(ax, ay, az, bx, by, bz)")],
    "lerp": [("bounded_fused", "bi64::lerp(a, b, c)")],
    "mul_add": [("bounded_fused", "bi64::mul_add(a, b, c)")],
    "length3": [("bounded_fused", "bi64::length3(a, b, c)")],
}
FELT_EXTRA = {
    "mul": [("felt_soft", "felt_fixed::mul_soft(a, b)"), ("felt_plain", "felt_fixed::mul_plain(a, b)")],
    "dot3": [("felt_lazy", "felt_fixed::dot3(ax, ay, az, bx, by, bz)")],
    "cross3": [("felt_lazy", "felt_fixed::cross3(ax, ay, az, bx, by, bz)")],
    "lerp": [("felt_lazy", "felt_fixed::lerp(a, b, c)")],
    "mul_add": [("felt_lazy", "felt_fixed::mul_add(a, b, c)")],
}

REPS = [
    Rep("signmag", "signmag64", "SignMag64", "sm64", 32, "trunc", "i32", signmag=True),
    Rep("native", "i64q32", "I64Q32", "i64", 32, "trunc", "i32"),
    Rep("bounded", "bi64", "BI64", "i64", 32, "floor", "i32", extra=BOUNDED_EXTRA),
    Rep("native", "i128q64", "I128Q64", "i128", 64, "trunc", "i64", extra={
        "mul": [("bounded", "i128q64::mul_bounded(a, b)")],
        "dot3": [("bounded_lazy", "i128q64::dot3_bounded(ax, ay, az, bx, by, bz)")],
    }),
    Rep("felt", "felt_fixed", "FeltQ32", "felt", 32, "floor", "i32", extra=FELT_EXTRA),
    Rep("native", "i32q16", "I32Q16", "i32", 16, "trunc", "i16"),
    Rep("signmag", "signmag32", "SignMag32", "sm32", 16, "trunc", "i16", signmag=True),
]

A, B, T = F(7, 2), F(9, 4), F(1, 4)
V1 = (F(3, 2), F(-9, 4), F(3))
V2 = (F(-1, 2), F(4), F(5, 2))
SIGNS = {"pp": (1, 1), "pn": (1, -1), "np": (-1, 1), "nn": (-1, -1)}


def scalar_cases(rep, op):
    """Yield (case, inputs {name: literal}, expected literal, default expr, kind)."""
    m = rep.mod
    if op in ("add", "sub", "mul", "div"):
        sym = {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op]
        cases = ["pp", "pn", "nn"] if op != "sub" else ["pp", "pn"]
        if op == "div":
            cases = ["pp", "pn"]
        for case in cases:
            sa, sb = SIGNS[case]
            a, b = sa * A, sb * B
            if op == "add":
                e = rep.lit(a + b)
            elif op == "sub":
                e = rep.lit(a - b)
            elif op == "mul":
                e = rep.lit(a * b)
            else:
                e = rep.lit_raw(rep.round(a / b))
            yield case, {"a": rep.lit(a), "b": rep.lit(b)}, e, f"a {sym} b", "value"
    elif op in ("add4", "mul4"):
        # Same op chained 4 times: (op4 - op1) / 3 is the marginal cost, free of first-use effects.
        sym = "+" if op == "add4" else "*"
        e = A + 4 * -B if op == "add4" else A * (-B) ** 4
        ins = {"a": rep.lit(A), "b": rep.lit(-B)}
        yield "pn", ins, rep.lit(e), f"a {sym} b {sym} b {sym} b {sym} b", "value"
    elif op == "neg":
        yield "p", {"a": rep.lit(A)}, rep.lit(-A), "-a", "value"
    elif op == "abs":
        yield "p", {"a": rep.lit(A)}, rep.lit(A), f"{m}::abs(a)", "value"
        yield "n", {"a": rep.lit(-A)}, rep.lit(A), f"{m}::abs(a)", "value"
    elif op == "lt":
        yield "pp", {"a": rep.lit(B), "b": rep.lit(A)}, "true", "a < b", "value"
        yield "np", {"a": rep.lit(-B), "b": rep.lit(A)}, "true", "a < b", "value"
        yield "nn", {"a": rep.lit(-A), "b": rep.lit(-B)}, "true", "a < b", "value"
    elif op == "eq":
        yield "p", {"a": rep.lit(A), "b": rep.lit(A)}, "true", "a == b", "value"
    elif op == "from_int":
        yield "n", {"a": f"-7_{rep.int_ty}"}, rep.lit(-7), f"{m}::from_int(a)", "value"
    elif op == "to_int":
        yield "p", {"a": rep.lit(A)}, f"3_{rep.int_ty}", f"{m}::to_int(a)", "value"
        yield "n", {"a": rep.lit(-B)}, f"-3_{rep.int_ty}", f"{m}::to_int(a)", "value"
    elif op == "sqrt":
        yield "p", {"a": rep.lit(F(9, 4))}, rep.lit(F(3, 2)), f"{m}::sqrt(a)", "value"
    elif op in ("lerp", "mul_add"):
        e = A + (B - A) * T if op == "lerp" else A * -B + T
        ins = {"a": rep.lit(A), "b": rep.lit(B if op == "lerp" else -B), "c": rep.lit(T)}
        yield "mix", ins, rep.lit(e), f"generic::{op}(a, b, c)", "value"
    elif op == "length3":
        ins = {"a": rep.lit(2), "b": rep.lit(-3), "c": rep.lit(6)}
        yield "mix", ins, rep.lit(7), f"{m}::sqrt(generic::dot3(a, b, c, a, b, c))", "value"
    elif op in ("dot3", "cross3"):
        names = ["ax", "ay", "az", "bx", "by", "bz"]
        ins = {n: rep.lit(v) for n, v in zip(names, V1 + V2)}
        (x1, y1, z1), (x2, y2, z2) = V1, V2
        if op == "dot3":
            e = rep.lit(x1 * x2 + y1 * y2 + z1 * z2)
        else:
            c = (y1 * z2 - z1 * y2, z1 * x2 - x1 * z2, x1 * y2 - y1 * x2)
            e = "(" + ", ".join(rep.lit(v) for v in c) + ")"
        yield "mix", ins, e, f"generic::{op}(ax, ay, az, bx, by, bz)", "value"


OPS = [
    "add", "sub", "neg", "abs", "mul", "div", "lt", "eq", "from_int", "to_int", "sqrt", "dot3",
    "cross3", "lerp", "mul_add", "length3", "add4", "mul4",
]


def test_fn(name, inputs, expected, expr, unused):
    lines = ["#[test]", "#[inline(never)]", f"fn {name}() {{"]
    for var, lit in inputs.items():
        lines.append(f"    let {'_' if unused else ''}{var} = black_box({lit});")
    lines.append(f"    let e = black_box({expected});")
    lines.append(f"    assert!({'e' if unused else '(' + expr + ')'} == e);")
    lines.append("}")
    return "\n".join(lines)


def main():
    OUT.mkdir(exist_ok=True)
    stores = {}
    for rep in REPS:
        stores.setdefault(rep.store, []).append(rep)
    mods = []
    for store, reps in stores.items():
        out = [
            "// GENERATED by gen_tests.py - do not edit.",
            "use harness::black_box;",
            "#[allow(unused_imports)]",
            "use crate::generic;",
        ]
        for rep in reps:
            out.append(f"use crate::{rep.mod}::{{self, {rep.ty}}};")
            if rep.mod == "felt_fixed":
                pass
        out.append("")
        for op in OPS:
            group = f"{op}_{store}"
            first = True
            for rep in reps:
                for case, ins, e, expr, _ in scalar_cases(rep, op):
                    if first:
                        out += [test_fn(f"bench_{group}__baseline", ins, e, expr, True), ""]
                        first = False
                    variants = [(rep.key, expr)] + rep.extra.get(op, [])
                    for vname, vexpr in variants:
                        name = f"bench_{group}__{vname}_{case}"
                        out += [test_fn(name, ins, e, vexpr, False), ""]
        (OUT / f"rep_{store}.cairo").write_text("\n".join(out))
        mods.append(f"rep_{store}")
    return mods


if __name__ == "__main__":
    main()
