#!/usr/bin/env python3
"""Generate `crates/simba_fixed/src/vectors.cairo`: operands for the conformance suite.

The suite (`crate::conformance`) asserts, case by case, that glam.cairo's `fixed::Fixed` and
simba's `simba::fixed::Fixed` agree bit for bit on the same raw operands — and records the few
places where they do not. The operands are drawn here, once, from a fixed seed, and filtered so
that NEITHER implementation panics on them: a conformance case must compare two values, not two
panics (panic parity is covered by the hand-written `#[should_panic]` tests).

Both scalars are modelled exactly, on plain Python integers (`simba_*` / `glam_*` below); the
model is only used to reject operands, never to produce an expected value, so it cannot silently
become the oracle of the thing it selects.

Usage (from the repository root):

    python3 crates/simba_fixed/tools/gen_vectors.py          # write the file, then `scarb fmt`
    python3 crates/simba_fixed/tools/gen_vectors.py --check  # exit 1 if the committed file is stale
"""

import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "crates" / "simba_fixed" / "src" / "vectors.cairo"
SEED = 0x474C_414D  # "GLAM"

ONE = 1 << 32
HALF = ONE >> 1
MIN = -(1 << 63)
MAX = (1 << 63) - 1


def fits(x):
    return MIN <= x <= MAX


def trunc_div(a, b):
    """Quotient of the integer division rounded toward zero (Python's `//` floors)."""
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


# --- exact models of the two scalars ------------------------------------------------------------
# Only the operations the conformance suite exercises, on raw values. `None` means "panics".


def simba_mul(a, b):
    r = (a * b) // ONE
    return r if fits(r) else None


def simba_div(a, b):
    if b == 0:
        return None
    r = (a * ONE) // b
    return r if fits(r) else None


def glam_div(a, b):
    if b == 0:
        return None
    r = trunc_div(a * ONE, b)
    return r if fits(r) else None


def simba_ceil(a):
    r = -((-a // ONE) * ONE)
    return r if fits(r) else None


def simba_round(a):
    r = ((a + HALF) // ONE) * ONE if a >= 0 else -(((-a + HALF) // ONE) * ONE)
    return r if fits(r) else None


def simba_recip(a):
    if a == 0:
        return None
    r = (ONE * ONE) // a
    return r if fits(r) else None


def glam_recip(a):
    if a == 0:
        return None
    r = trunc_div(ONE * ONE, a)
    return r if fits(r) else None


def rescale(wide):
    """One `Wide::rescale`: floor of an unscaled accumulator, or `None` on overflow."""
    r = wide // ONE
    return r if fits(r) else None


# --- operand draws ------------------------------------------------------------------------------

BOUNDARY = [MIN + 1, -MAX // 2, -ONE - 1, -ONE, -HALF, -3, -1, 0, 1, 3, HALF, ONE, ONE + 1, MAX]


def magnitude(rng, lo, hi):
    """A raw magnitude log-uniform in `[lo, hi)`, with a random sign."""
    m = int(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    return -m if rng.random() < 0.5 else m


def operand(rng):
    """One raw operand covering both signs and every magnitude class of DESIGN D7."""
    kind = rng.random()
    if kind < 0.15:  # small: |x| < 1
        return magnitude(rng, 1, ONE)
    if kind < 0.40:  # unit: |x| in [0.5, 2)
        return magnitude(rng, HALF, 2 * ONE)
    if kind < 0.70:  # medium: |x| in [2, 1e3)
        return magnitude(rng, 2 * ONE, 1000 * ONE)
    if kind < 0.90:  # large: |x| in [1e3, 4e4)
        return magnitude(rng, 1000 * ONE, 40000 * ONE)
    return rng.choice(BOUNDARY)  # edges of the representable range


def draw(rng, count, make, keep):
    """`count` distinct tuples from `make`, keeping only those `keep` accepts."""
    out, seen = [], set()
    while len(out) < count:
        case = make()
        if case in seen or not keep(case):
            continue
        seen.add(case)
        out.append(case)
    return out


# --- case sets ----------------------------------------------------------------------------------


def binary_ok(case):
    """`a + b`, `a - b`, `a * b`, `a / b`, `a % b` all defined and in range, for both scalars."""
    a, b = case
    return (
        b != 0
        and fits(a + b)
        and fits(a - b)
        and simba_mul(a, b) is not None
        and simba_div(a, b) is not None
        and glam_div(a, b) is not None
    )


def unary_ok(a):
    """`-a`, `|a|`, `floor`, `ceil`, `round`, `trunc`, `fract` all defined and in range."""
    return a != MIN and simba_ceil(a) is not None and simba_round(a) is not None


def mul_add_ok(case):
    """`a * b + c` and `a * b - c` in range after the single rescale."""
    a, b, c = case
    return rescale(a * b + c * ONE) is not None and rescale(a * b - c * ONE) is not None


def vec3_ok(case):
    """Both vectors have an in-range squared norm, dot product and cross product."""
    ax, ay, az, bx, by, bz = case
    sums = [
        ax * ax + ay * ay + az * az,
        bx * bx + by * by + bz * bz,
        ax * bx + ay * by + az * bz,
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx,
    ]
    return all(rescale(s) is not None for s in sums)


def cases(rng):
    """Every generated table: `(name, cairo tuple type, doc, rows)`."""
    small = lambda: magnitude(rng, 1, 4 * ONE)  # noqa: E731
    return [
        (
            "binary",
            "(i64, i64)",
            "Pairs `(a, b)` on which `+ - * / %` and the comparisons are defined and in range\n"
            "/// for both scalars.",
            draw(rng, 64, lambda: (operand(rng), operand(rng)), binary_ok),
        ),
        (
            "unary",
            "i64",
            "Values on which `-a`, `abs`, `floor`, `ceil`, `round`, `trunc`, `fract` and `signum`\n"
            "/// are defined and in range for both scalars.",
            draw(rng, 40, lambda: (operand(rng),), lambda c: unary_ok(c[0])),
        ),
        (
            "sqrt",
            "i64",
            "Non-negative values, for `sqrt` (both scalars take the exact integer square root of\n"
            "/// `raw * 2^32`, so both floor).",
            draw(rng, 32, lambda: (abs(operand(rng)),), lambda c: True),
        ),
        (
            "recip",
            "i64",
            "Values whose reciprocal is defined and in range for both scalars (`|raw| >= 3`).",
            draw(
                rng,
                32,
                lambda: (operand(rng),),
                lambda c: simba_recip(c[0]) is not None and glam_recip(c[0]) is not None,
            ),
        ),
        (
            "mul_add",
            "(i64, i64, i64)",
            "Triples `(a, b, c)` on which the fused `a * b + c` and `a * b - c` are in range.",
            draw(rng, 40, lambda: (small(), small(), operand(rng)), mul_add_ok),
        ),
        (
            "vec3",
            "(i64, i64, i64, i64, i64, i64)",
            "Vector pairs `(ax, ay, az, bx, by, bz)` whose dot, cross and squared norms are in\n"
            "/// range: `sum_prod3` vs `fixed::wide::dot3`, `norm3`, `norm_squared3`.",
            draw(rng, 32, lambda: tuple(small() for _ in range(6)), vec3_ok),
        ),
        (
            "angle",
            "i64",
            "Angles in radians for `sin`, `cos`, `sin_cos`, `tan` and `atan`, from a fraction of\n"
            "/// an ulp to many turns.",
            draw(rng, 40, lambda: (operand(rng),), lambda c: True),
        ),
        (
            "unit",
            "i64",
            "Values in `[-1, 1]`, the common domain of `asin` and `acos`.",
            draw(rng, 32, lambda: (magnitude(rng, 1, ONE + 1),), lambda c: abs(c[0]) <= ONE),
        ),
        (
            "atan2",
            "(i64, i64)",
            "Pairs `(y, x)` covering the four quadrants and both axes.",
            draw(
                rng,
                32,
                lambda: (operand(rng), operand(rng)),
                lambda c: not (c[0] == 0 and c[1] == 0),
            ),
        ),
    ]


HEADER = '''//! Operands of the conformance suite (`crate::conformance`), covering both signs and the small /
//! unit / medium / large / edge magnitude classes of DESIGN D7.
//!
//! GENERATED by `crates/simba_fixed/tools/gen_vectors.py` (seed 0x474C414D). Do not edit.
//! `python3 crates/simba_fixed/tools/gen_vectors.py --check` fails if this file is stale.
//!
//! Every row is a raw Q32.32 `i64` (`value = raw / 2^32`), filtered so that neither
//! `fixed::Fixed` nor `simba::fixed::Fixed` panics on it for the operations of its table. There is
//! no expected value here: the conformance suite compares the two implementations against each
//! other, not against a model.
'''


def render():
    rng = random.Random(SEED)
    out = [HEADER]
    for name, kind, doc, rows in cases(rng):
        rows = [row[0] if len(row) == 1 else "(" + ", ".join(str(v) for v in row) + ")" for row in rows]
        out.append(f"/// {doc}")
        out.append("#[cairofmt::skip]")
        out.append(f"pub fn {name}_cases() -> Span<{kind}> {{")
        out.append(f"    const CASES: [{kind}; {len(rows)}] = [")
        out += [f"        {row}," for row in rows]
        out.append("    ];")
        out.append("    CASES.span()")
        out.append("}")
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed file is stale")
    args = parser.parse_args()

    content = render()
    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != content:
            sys.exit(f"{OUT.relative_to(ROOT)} is stale: rerun crates/simba_fixed/tools/gen_vectors.py")
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return
    OUT.write_text(content)
    subprocess.run(["scarb", "fmt"], cwd=ROOT, check=True)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(content.splitlines())} lines)")


if __name__ == "__main__":
    main()
