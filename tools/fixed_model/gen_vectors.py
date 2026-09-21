#!/usr/bin/env python3
"""Generate `crates/simba/src/fixed/tests_generated.cairo` from the bit-exact model.

For every kernel of `simba::fixed`, draws pseudo-random operands (fixed seed, so the output is
reproducible), computes the expected raw result with `fixed_model`, and emits snforge tests that
loop over const arrays of `[inputs.., expected]` rows (`CASES_PER_TEST` rows per test function,
to keep the number of tests and the CI time low). Operand tuples on which the model raises
(overflow, division by zero, sqrt of a negative) are re-drawn: panics are covered by the
hand-written `#[should_panic]` tests next to each function.

Usage (from the repository root):

    python3 tools/fixed_model/gen_vectors.py          # writes the file, then runs `scarb fmt`
    python3 tools/fixed_model/gen_vectors.py --check  # exits 1 if the committed file is stale
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import fixed_model as m  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "crates" / "simba" / "src" / "fixed" / "tests_generated.cairo"
SEED = 0x51_4D_42_41  # "SIMBA"
CASES_PER_TEST = 64
TESTS_PER_KERNEL = 4

ONE, HALF, MIN, MAX = m.ONE, m.HALF, m.MIN, m.MAX
BOUNDARY = [
    MIN, MIN + 1, MIN + ONE, -MAX // 2, -ONE - 1, -ONE, -ONE + 1, -HALF, -2, -1, 0, 1, 2, HALF,
    ONE - 1, ONE, ONE + 1, MAX // 2, MAX - ONE + 1, MAX - 1, MAX,
]  # fmt: skip


def operand(rng):
    """One raw operand: a mix of physics-scale values, all magnitudes, tiny and boundary values."""
    kind = rng.random()
    if kind < 0.35:  # physics scale: |x| in [2^-8, 2^12), full fractional precision
        mag = rng.getrandbits(rng.randrange(24, 45))
        return -mag if rng.random() < 0.5 else mag
    if kind < 0.60:  # any magnitude, uniform in bit length
        bits = rng.randrange(0, 64)
        mag = rng.getrandbits(bits) if bits else 0
        return max(MIN, -mag) if rng.random() < 0.5 else min(MAX, mag)
    if kind < 0.75:  # a few ulps
        return rng.randrange(-16, 17)
    if kind < 0.85:  # integers and half-integers
        return rng.randrange(-(1 << 20), 1 << 20) * HALF
    return rng.choice(BOUNDARY)


def small_operand(rng):
    """Operand for kernels whose result overflows easily (squares, norms, long sums)."""
    if rng.random() < 0.8:
        mag = rng.getrandbits(rng.randrange(0, 46))
        return -mag if rng.random() < 0.5 else mag
    return operand(rng)


def non_negative_operand(rng):
    return min(MAX, abs(operand(rng)))


def angle_operand(rng):
    """Operand for sin / cos / tan / atan: physics-scale angles, huge angles and edge cases."""
    if rng.random() < 0.55:  # |x| < 8 rad, full fractional precision
        mag = rng.getrandbits(rng.randrange(0, 36))
        return -mag if rng.random() < 0.5 else mag
    return operand(rng)


def unit_operand(rng):
    """Operand for asin / acos: `|x| <= 1`, with the endpoints and the 1/2 branch boundary."""
    if rng.random() < 0.9:
        v = rng.randrange(0, ONE + 1)
    else:
        v = rng.choice([0, 1, HALF - 1, HALF, HALF + 1, ONE - 1, ONE])
    return -v if rng.random() < 0.5 else v


def exp_operand(rng):
    """Operand for `exp`: mostly the representable window `[-22.9, 21.49]`, plus the tails
    (which underflow to zero or overflow and are re-drawn by `rows_for`)."""
    if rng.random() < 0.9:
        return rng.randrange(-23 * ONE, 22 * ONE)
    return operand(rng)


def positive_operand(rng):
    """Operand for `ln`: all magnitudes, log-uniform in bit length, never zero."""
    return max(1, min(MAX, abs(operand(rng))))


def ratio_operand(rng):
    """Integer numerators / denominators for `from_ratio`: small integers or anything."""
    return rng.randrange(-1000, 1001) if rng.random() < 0.5 else operand(rng)


def wide_mixed(a, b, c, d, k, l):
    return m.Wide().add_prod(a, b).sub_prod(c, d).add(k).sub(l).rescale()


def wide_sum8(*v):
    w = m.Wide()
    for i in range(0, 16, 2):
        w = w.add_prod(v[i], v[i + 1])
    return w.rescale()


def wide_norm6(*v):
    w = m.Wide()
    for x in v:
        w = w.add_prod(x, x)
    return w.sqrt()


def sin_cos_packed(x):
    """`sin_cos` folded into one expectation: `sin - cos`, exact (both are in [-1, 1])."""
    s, c = m.sin_cos(x)
    return s - c


def ordering(a, b):
    """lt | le << 1 | gt << 2 | ge << 3 | eq << 4."""
    return (a < b) | (a <= b) << 1 | (a > b) << 2 | (a >= b) << 3 | (a == b) << 4


def ulps(rng):
    return rng.choice([0, 1, 2, 3, rng.getrandbits(8), rng.getrandbits(40), rng.getrandbits(63)])


def near(rng):
    """A pair of close operands (for abs_diff_eq)."""
    a = operand(rng)
    return a, max(MIN, min(MAX, a + rng.randrange(-300, 301)))


# name, arity, model function, Cairo expression over `f(x)`-wrapped inputs a, b, c.. and the raw
# expected value `e`, operand sampler.
KERNELS = [
    ("add", 2, m.add, "f(a) + f(b) == f(e)", operand),
    ("sub", 2, m.sub, "f(a) - f(b) == f(e)", operand),
    ("neg", 1, m.neg, "-f(a) == f(e)", operand),
    ("mul", 2, m.mul, "f(a) * f(b) == f(e)", operand),
    ("div", 2, m.div, "f(a) / f(b) == f(e)", operand),
    ("rem", 2, m.rem, "f(a) % f(b) == f(e)", operand),
    ("ordering", 2, ordering, "ordering(f(a), f(b)) == e", operand),
    ("abs", 1, m.abs_, "math::abs(f(a)) == f(e)", operand),
    ("signum", 1, m.signum, "math::signum(f(a)) == f(e)", operand),
    ("min", 2, m.min_, "math::min(f(a), f(b)) == f(e)", operand),
    ("max", 2, m.max_, "math::max(f(a), f(b)) == f(e)", operand),
    ("clamp", 3, m.clamp, "math::clamp(f(a), f(b), f(c)) == f(e)", operand),
    ("floor", 1, m.floor, "math::floor(f(a)) == f(e)", operand),
    ("ceil", 1, m.ceil, "math::ceil(f(a)) == f(e)", operand),
    ("round", 1, m.round_, "math::round(f(a)) == f(e)", operand),
    ("trunc", 1, m.trunc, "math::trunc(f(a)) == f(e)", operand),
    ("fract", 1, m.fract, "math::fract(f(a)) == f(e)", operand),
    ("recip", 1, m.recip, "math::recip(f(a)) == f(e)", operand),
    ("sqrt", 1, m.sqrt, "math::sqrt(f(a)) == f(e)", non_negative_operand),
    ("inv_sqrt", 1, m.inv_sqrt, "math::inv_sqrt(f(a)) == f(e)", non_negative_operand),
    ("to_int", 1, m.to_int, "convert::to_int(f(a)).into() == e", operand),
    ("from_ratio", 2, m.from_ratio, "convert::from_ratio(a, b) == f(e)", ratio_operand),
    ("sqr", 1, m.sqr, "fused::sqr(f(a)) == f(e)", small_operand),
    ("sum_prod2", 4, m.sum_prod2, "fused::sum_prod2(f(a), f(b), f(c), f(d)) == f(e)", operand),
    ("sum_prod3", 6, m.sum_prod3,
     "fused::sum_prod3(f(a), f(b), f(c), f(d), f(g), f(h)) == f(e)", small_operand),
    ("sum_prod4", 8, m.sum_prod4,
     "fused::sum_prod4(f(a), f(b), f(c), f(d), f(g), f(h), f(i), f(j)) == f(e)", small_operand),
    ("diff_prod", 4, m.diff_prod, "fused::diff_prod(f(a), f(b), f(c), f(d)) == f(e)", operand),
    ("mul_add", 3, m.mul_add, "fused::mul_add(f(a), f(b), f(c)) == f(e)", operand),
    ("mul_sub", 3, m.mul_sub, "fused::mul_sub(f(a), f(b), f(c)) == f(e)", operand),
    ("lerp", 3, m.lerp, "fused::lerp(f(a), f(b), f(c)) == f(e)", operand),
    ("norm_squared2", 2, m.norm_squared2, "fused::norm_squared2(f(a), f(b)) == f(e)",
     small_operand),
    ("norm_squared3", 3, m.norm_squared3, "fused::norm_squared3(f(a), f(b), f(c)) == f(e)",
     small_operand),
    ("norm_squared4", 4, m.norm_squared4,
     "fused::norm_squared4(f(a), f(b), f(c), f(d)) == f(e)", small_operand),
    ("norm2", 2, m.norm2, "fused::norm2(f(a), f(b)) == f(e)", operand),
    ("norm3", 3, m.norm3, "fused::norm3(f(a), f(b), f(c)) == f(e)", operand),
    ("norm4", 4, m.norm4, "fused::norm4(f(a), f(b), f(c), f(d)) == f(e)", operand),
    ("wide_mixed", 6, wide_mixed,
     "WideTrait::from_prod(f(a), f(b)).sub_prod(f(c), f(d)).add(f(g)).sub(f(h)).rescale() == f(e)",
     operand),
    ("wide_sum_prod8", 16, wide_sum8, "sum8(row) == f(e)", small_operand),
    ("wide_norm6", 6, wide_norm6, "norm6(row) == f(e)", operand),
    ("sin", 1, m.sin, "tr::sin(f(a)) == f(e)", angle_operand),
    ("cos", 1, m.cos, "tr::cos(f(a)) == f(e)", angle_operand),
    ("tan", 1, m.tan, "tr::tan(f(a)) == f(e)", angle_operand),
    ("sin_cos", 1, sin_cos_packed, "sin_cos_packed(f(a)) == f(e)", angle_operand),
    ("atan", 1, m.atan, "tr::atan(f(a)) == f(e)", angle_operand),
    ("atan2", 2, m.atan2, "tr::atan2(f(a), f(b)) == f(e)", operand),
    ("asin", 1, m.asin, "tr::asin(f(a)) == f(e)", unit_operand),
    ("acos", 1, m.acos, "tr::acos(f(a)) == f(e)", unit_operand),
    ("exp", 1, m.exp, "tr::exp(f(a)) == f(e)", exp_operand),
    ("ln", 1, m.ln, "tr::ln(f(a)) == f(e)", positive_operand),
]  # fmt: skip

NAMES = "abcdghijklmnopqr"  # `e` is the expected value, `f` the wrapper

HEADER = '''//! GENERATED by `tools/fixed_model/gen_vectors.py` (seed {seed:#x}): do not edit by hand.
//!
//! {total} pseudo-random vectors: {per_kernel} per kernel, {per_test} per test. Every row is
//! `[inputs.., expected]` in raw units; expectations come from the bit-exact Python integer model
//! `tools/fixed_model/fixed_model.py` (floor rounding, once per output).

use super::types::Fixed;
use super::wide::WideTrait;
use super::{{convert, fused, math, transcendental as tr, types}};

#[inline(always)]
fn f(raw: i64) -> Fixed {{
    Fixed {{ raw }}
}}

/// `sin(x) - cos(x)`: both halves of `sin_cos` in one expectation (the difference is exact).
#[inline(always)]
fn sin_cos_packed(x: Fixed) -> Fixed {{
    let (s, c) = tr::sin_cos(x);
    s - c
}}

/// lt | le << 1 | gt << 2 | ge << 3 | eq << 4
fn ordering(a: Fixed, b: Fixed) -> i64 {{
    let mut bits = 0;
    if a < b {{
        bits += 1;
    }}
    if a <= b {{
        bits += 2;
    }}
    if a > b {{
        bits += 4;
    }}
    if a >= b {{
        bits += 8;
    }}
    if a == b {{
        bits += 16;
    }}
    bits
}}

/// Dot product of 8 pairs through the wide accumulator; the row is `[a0, b0, .., a7, b7, e]`.
fn sum8(row: [i64; 17]) -> Fixed {{
    let mut w = WideTrait::zero();
    let mut pairs = row.span().slice(0, 16);
    while let Some(pair) = pairs.multi_pop_front::<2>() {{
        let [a, b] = (*pair).unbox();
        w = w.add_prod(f(a), f(b));
    }}
    w.rescale()
}}

/// Norm of a 6-vector through the wide accumulator; the row is `[x0, .., x5, e]`.
fn norm6(row: [i64; 7]) -> Fixed {{
    let mut w = WideTrait::zero();
    for x in row.span().slice(0, 6) {{
        w = w.add_prod(f(*x), f(*x));
    }}
    w.sqrt()
}}
'''


def lit(v):
    return str(v)


def rows_for(rng, arity, fn, sampler, count):
    rows = []
    while len(rows) < count:
        args = [sampler(rng) for _ in range(arity)]
        try:
            expected = fn(*args)
        except m.FixedError:
            continue
        rows.append(args + [int(expected)])
    return rows


def emit_kernel(out, name, arity, expr, rows, index):
    width = arity + 1
    out.append("")
    out.append("#[test]")
    out.append(f"fn test_{name}_vectors_{index}() {{")
    out.append(f"    let rows: [[i64; {width}]; {len(rows)}] = [")
    for row in rows:
        out.append("        [" + ", ".join(lit(v) for v in row) + "],")
    out.append("    ];")
    out.append("    for row in rows.span() {")
    if expr.startswith(("sum8", "norm6")):
        out.append("        let row = *row;")
        out.append(f"        let e = *row.span().at({arity});")
        out.append(f'        assert!({expr}, "{name} {{:?}}", row.span());')
    else:
        names = ", ".join(list(NAMES[:arity]) + ["e"])
        out.append(f"        let [{names}] = *row;")
        out.append(f'        assert!({expr}, "{name} {{:?}}", row.span());')
    out.append("    }")
    out.append("}")


def abs_diff_eq_rows(rng, count):
    rows = []
    while len(rows) < count:
        a, b = near(rng) if rng.random() < 0.7 else (operand(rng), operand(rng))
        u = abs(a - b) + rng.choice([-1, 0, 1]) if rng.random() < 0.5 else ulps(rng)
        u = max(0, min(u, MAX))
        rows.append([a, b, u, int(m.abs_diff_eq(a, b, u))])
    return rows


def generate():
    rng = random.Random(SEED)
    per_kernel = CASES_PER_TEST * TESTS_PER_KERNEL
    total = per_kernel * (len(KERNELS) + 1)
    out = [
        HEADER.format(seed=SEED, total=total, per_kernel=per_kernel, per_test=CASES_PER_TEST)
    ]

    out.append("#[test]")
    out.append("fn test_constants_match_model() {")
    for name, raw in m.constants().items():
        out.append(f"    assert!(types::{name}.raw == {raw});")
    out.append("}")

    for name, arity, fn, expr, sampler in KERNELS:
        for index in range(TESTS_PER_KERNEL):
            rows = rows_for(rng, arity, fn, sampler, CASES_PER_TEST)
            emit_kernel(out, name, arity, expr, rows, index)

    for index in range(TESTS_PER_KERNEL):
        rows = abs_diff_eq_rows(rng, CASES_PER_TEST)
        out.append("")
        out.append("#[test]")
        out.append(f"fn test_abs_diff_eq_vectors_{index}() {{")
        out.append(f"    let rows: [[i64; 4]; {len(rows)}] = [")
        for row in rows:
            out.append("        [" + ", ".join(lit(v) for v in row) + "],")
        out.append("    ];")
        out.append("    for row in rows.span() {")
        out.append("        let [a, b, ulps, e] = *row;")
        out.append("        let ulps: u64 = ulps.try_into().unwrap();")
        out.append(
            '        assert!(math::abs_diff_eq(f(a), f(b), ulps) == (e == 1), '
            '"abs_diff_eq {:?}", row.span());'
        )
        out.append("    }")
        out.append("}")
    emit_wide_mul_scalar(out)
    return "\n".join(out) + "\n"


# --- wide_mul_scalar: appended last, on its own RNG, so that no earlier vector moves -----------

WMS_SEED = 0x57_4D_53  # "WMS"
WMS_EXPR = (
    "WideTrait::from_prod(f(a), f(b)).add_prod(f(c), f(d)).add_prod(f(g), f(h)).mul_scalar(f(i))"
    " == f(e)"
)
WMS_RANDOM_TESTS = 1
WMS_RANDOM_ROWS = 24


def wms(a, b, c, d, g, h, s):
    return m.wide_mul_scalar(a * b + c * d + g * h, s)


def wms_decompose(w):
    """`[a, b, c, d, g, h]` in i64 with `a*b + c*d + g*h == w`, for `|w| <= 2^127 + 2^63`: two
    near-extreme products and an exact remainder (`h` in 1..7)."""
    clamp = lambda v: max(MIN, min(MAX, v))  # noqa: E731
    for b, d in ((MIN, MIN), (MAX, MAX), (MIN, MAX)):
        a0 = clamp(w // 2 // b)
        c0 = clamp((w - a0 * b) // d)
        for da in range(-2, 3):  # near the extremes, leave room for the remainder
            for dc in range(-2, 3):
                a, c = a0 + da, c0 + dc
                rest = w - a * b - c * d
                for h in range(1, 8):
                    if MIN <= a <= MAX and MIN <= c <= MAX and rest % h == 0 \
                            and MIN <= rest // h <= MAX:
                        return [a, b, c, d, rest // h, h]
    raise ValueError(w)


def wms_scalar(rng, w):
    """A scalar that mostly brings `w * s / 2^64` back into range (re-drawn when it does not)."""
    if rng.random() < 0.15:
        return operand(rng)
    room = max(0, 127 - abs(w).bit_length())
    mag = rng.getrandbits(rng.randrange(0, min(room, 63) + 1) or 1)
    return -mag if rng.random() < 0.5 else mag


def wms_random_rows(rng, count):
    rows = []
    while len(rows) < count:
        # Mixed magnitudes: the sum of 3 products reaches ~2^127.6, far beyond the Fixed range.
        args = [operand(rng) if rng.random() < 0.5 else small_operand(rng) for _ in range(6)]
        w = args[0] * args[1] + args[2] * args[3] + args[4] * args[5]
        s = wms_scalar(rng, w)
        try:
            rows.append(args + [s, wms(*args, s)])
        except m.FixedError:
            continue
    return rows


def wms_edge_rows(rng):
    """Boundaries (result MIN / MAX, their neighbours MIN - 1 / MAX + 1 are the should_panic
    tests), negative inexact products (floor != trunc), zero scalar, zero accumulator, huge
    accumulators brought back by tiny scalars, and `s = 1.0` (== `rescale`)."""
    rows = []

    def add(w, s):
        args = wms_decompose(w)
        rows.append(args + [s, wms(*args, s)])

    for s in [1, -1, 3, HALF + 1, -(5 * ONE + 7), ONE, MIN]:
        for t in (MIN, MAX):
            # the smallest and largest accumulators whose product floors exactly to t: one step
            # further is the overflow of the should_panic tests
            ends = {f(x, s) for x in (t << 64, ((t + 1) << 64) - 1) for f in (_floordiv, _ceildiv)}
            ok = sorted(w for w in ends if _ok(w, s) and m.wide_mul_scalar(w, s) == t)
            add(ok[0], s)
            if ok[-1] != ok[0]:
                add(ok[-1], s)
    for s in [-1, 1, 3, -(ONE + 1)]:  # negative, inexact: floor is one below truncation
        w = -(rng.getrandbits(90) | 1) if s > 0 else rng.getrandbits(90) | 1
        assert (w * s) % (1 << 64) != 0 and w * s < 0
        add(w, s)
    for w in [0, 1, -1, (1 << 127), -(1 << 127) - (1 << 62)]:  # zero scalar
        add(w, 0)
    for s in [MIN, MAX, -1, 1]:  # zero accumulator
        add(0, s)
    for w, s in [((1 << 127) - 1, 1), (-(1 << 127), 1), ((1 << 126) - 12345, 2), (-(3 << 125), -1)]:
        add(w, s)  # far beyond the Fixed range (2^95), back in range after the product
    for _ in range(2):  # s = 1.0: identical to `rescale`
        w = rng.randrange(-(1 << 95), 1 << 95)
        add(w, ONE)
        assert rows[-1][-1] == m.rescale(w)
    return rows


def _floordiv(x, s):
    return x // s


def _ceildiv(x, s):
    return -(-x // s)


def _ok(w, s):
    try:
        m.wide_mul_scalar(w, s)
        return True
    except m.FixedError:
        return False


def emit_wide_mul_scalar(out):
    rng = random.Random(WMS_SEED)
    out.append("")
    out.append("// `Wide::mul_scalar`: the accumulator is `a*b + c*d + g*h` (up to ~2^127.6, far beyond")
    out.append("// the Fixed range), the scalar `i`. Generated after the other kernels, on its own seed.")
    for index in range(WMS_RANDOM_TESTS):
        emit_kernel(out, "wide_mul_scalar", 7, WMS_EXPR, wms_random_rows(rng, WMS_RANDOM_ROWS), index)
    emit_kernel(out, "wide_mul_scalar_edge", 7, WMS_EXPR, wms_edge_rows(rng), 0)


def scarb_fmt(path):
    """Format one file in place with the workspace settings (`scarb fmt` works per package)."""
    subprocess.run(["scarb", "fmt", "-p", "simba"], cwd=ROOT, check=True, capture_output=True)
    return path.read_text()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--check", action="store_true", help="fail if the committed file is stale")
    parser.add_argument("--no-fmt", action="store_true", help="do not run `scarb fmt` afterwards")
    args = parser.parse_args()

    text = generate()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        # Compare modulo formatting: strip all whitespace.
        squash = lambda s: "".join(s.split()).replace(",]", "]").replace(",)", ")")  # noqa: E731
        if squash(current) != squash(text):
            sys.exit(f"{OUT.relative_to(ROOT)} is stale: run tools/fixed_model/gen_vectors.py")
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    OUT.write_text(text)
    if not args.no_fmt:
        scarb_fmt(OUT)
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
