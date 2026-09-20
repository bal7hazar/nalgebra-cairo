#!/usr/bin/env python3
"""Generator of the typed polynomial kernels of `simba::fixed` (DESIGN D6).

Outputs (committed, reproducible byte for byte, never edited by hand):

* `crates/simba/src/fixed/kernels/poly.cairo`: the exported kernels (range reductions, typed
  Horner polynomials, final roundings) as `BoundedInt` straight-line code whose bounds are proven
  here by interval arithmetic (`typed.py`);
* `crates/simba/src/fixed/kernels/poly_alternatives.cairo`: the candidates that lost the
  accuracy / gas comparison, compiled under `#[cfg(test)]` only (evidence, AGENTS.md rule 8);
* `tools/fixed_model/poly_ops.py`: the recorded programs, so that the bit-exact Python model of
  `tools/fixed_model` runs *the same* integer operations as the Cairo code.

Coefficients are near-minimax (Chebyshev) fits computed with mpmath at 200 bits, rounded to the
scale of each Horner step. Every fitted function `f` with `f(0)` known exactly (`sin(r)/r`,
`cos(r)`, `atan(t)/t`, `asin(s)/s` -> 1) is fitted in the form `f = 1 + z * G(z)` with the leading
1 exact, so the kernels are exact at 0 (`sin(0) = 0`, `cos(0) = 1`, `atan(0) = 0`, `asin(0) = 0`)
and reduce to the identity for tiny arguments.

Usage (from the repository root):

    python3 tools/polygen/polygen.py            # write the three files (runs `scarb fmt`)
    python3 tools/polygen/polygen.py --check    # exit 1 if a committed file is stale
    python3 tools/polygen/polygen.py --report   # accuracy of every candidate, in ulp

`--check` and the default mode also replay every generated kernel on Python integers through
`tools/fixed_model`, so a divergence between the model and the generated code is a hard error.
"""

import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

from mpmath import chebyfit, mp, mpf

sys.path.insert(0, str(Path(__file__).parent))
from typed import Kernel, Module, hx  # noqa: E402

mp.prec = 240

ROOT = Path(__file__).resolve().parents[2]
KERNELS_DIR = ROOT / "crates" / "simba" / "src" / "fixed" / "kernels"
OUT_POLY = KERNELS_DIR / "poly.cairo"
OUT_ALT = KERNELS_DIR / "poly_alternatives.cairo"
OUT_OPS = ROOT / "tools" / "fixed_model" / "poly_ops.py"

I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
ONE = 1 << 32


def rnd(v):
    return int(mp.nint(v))


# --- scales and constants -----------------------------------------------------------------------

# Scale of reduced angles and ratios (61 fractional bits). pi/4 * 2^61 is within 0.13 of an
# integer, so reducing the largest angle (|x| < 2^31 rad, 2.7e9 octants) drifts by less than
# 2.9 raw Q32.32 units; every divisor stays below 2^123, the limit of the cheapest `div_rem`.
A = 61
# Scale of the squared argument fed to the polynomials (`z = floor(r^2 / 2^(2A - Z))`): a small
# `z` lets a Horner accumulator absorb up to 3 multiplications between two rescales.
Z = 36
# Scale of the Horner accumulators after a rescale.
ACC = 64
# Largest shift of the cheap `div_rem` algorithm (`rhs * 2^128 < P`).
MAX_SHIFT = 122

QUARTER_PI = rnd(mp.pi / 4 * 2**A)
HALF_PI = 2 * QUARTER_PI

# Common intervals exported by the kernels. Interval arithmetic gives `[0, 1]` for a sine or a
# cosine on the octant; the angles are bounded a little above pi/6 and pi/3. The `upcast` that
# widens a kernel result to these types fails at generation time if a candidate exceeds them.
MAG32_HI = 0x100000000  # 1.0, Q32.32
MAG48_HI = 0x1000000000000  # 1.0, scale 2^48
ANG_QP_HI = 0x100000000  # 1.0 rad > pi/4
ASIN_HI = 0x87000000  # > pi/6
ASIN2_HI = 0x10E000000  # > pi/3


def offset(period):
    """Smallest multiple of `period` that is `>= 2^(A+31)`: `x * 2^(A-32) + offset >= 0`."""
    return -(-(1 << (A + 31)) // period) * period


OFF_TRIG = offset(8 * QUARTER_PI)

# --- fits ---------------------------------------------------------------------------------------

MASTER = 200  # coefficients are kept at scale 2^MASTER and rounded to the scale of each step


def tail(f, one):
    """`g(z) = (f(z) - one) / z`: the part of `f` that a forced exact leading `one` leaves."""
    return lambda z: (f(z) - one) / z


def odd_part(f):
    """`g(z) = f(sqrt z) / sqrt z` for an odd `f` with `f'(0) = 1`."""
    return lambda z: f(mp.sqrt(z)) / mp.sqrt(z)


# name: (function of z, lower bound, upper bound, number of FITTED coefficients, scale of z)
# The fitted polynomial is `1 + z * G(z)`, i.e. `n + 1` coefficients in the Horner program.
SPECS = {
    # sin(r) = r * S(r^2), r in [0, pi/4] (octant) or [0, pi/2] (quarter turn)
    "SIN_O3": (tail(odd_part(mp.sin), 1), 0, (mp.pi / 4) ** 2, 3, Z),
    "SIN_O4": (tail(odd_part(mp.sin), 1), 0, (mp.pi / 4) ** 2, 4, Z),
    "SIN_O5": (tail(odd_part(mp.sin), 1), 0, (mp.pi / 4) ** 2, 5, Z),
    "SIN_Q5": (tail(odd_part(mp.sin), 1), 0, (mp.pi / 2) ** 2, 5, Z),
    "SIN_Q6": (tail(odd_part(mp.sin), 1), 0, (mp.pi / 2) ** 2, 6, Z),
    # cos(r) = C(r^2), r in [0, pi/4]
    "COS_O3": (tail(lambda z: mp.cos(mp.sqrt(z)), 1), 0, (mp.pi / 4) ** 2, 3, Z),
    "COS_O4": (tail(lambda z: mp.cos(mp.sqrt(z)), 1), 0, (mp.pi / 4) ** 2, 4, Z),
    "COS_O5": (tail(lambda z: mp.cos(mp.sqrt(z)), 1), 0, (mp.pi / 4) ** 2, 5, Z),
    # atan(t) = t * T(t^2), t in [0, 1]
    "ATAN_8": (tail(odd_part(mp.atan), 1), 0, 1, 8, Z),
    "ATAN_10": (tail(odd_part(mp.atan), 1), 0, 1, 10, Z),
    "ATAN_12": (tail(odd_part(mp.atan), 1), 0, 1, 12, Z),
    "ATAN_14": (tail(odd_part(mp.atan), 1), 0, 1, 14, Z),
    # asin(s) = s * R(s^2), s in [0, 1/2]
    "ASIN_5": (tail(odd_part(mp.asin), 1), 0, mpf(1) / 4, 5, Z),
    "ASIN_7": (tail(odd_part(mp.asin), 1), 0, mpf(1) / 4, 7, Z),
    "ASIN_9": (tail(odd_part(mp.asin), 1), 0, mpf(1) / 4, 9, Z),
}

_FITS = {}


def fit(name):
    """Master coefficients (lowest degree first, scale 2^MASTER): `[1, g0, g1, ..]`."""
    if name not in _FITS:
        g, lo, hi, n, _ = SPECS[name]
        poly, err = chebyfit(g, [lo, hi], n, error=True)
        coefs = [1 << MASTER] + [rnd(c * 2**MASTER) for c in reversed(poly)]
        _FITS[name] = (coefs, err)
    return _FITS[name]


def scaled(c, scale):
    """Master coefficient rounded to nearest at `scale` (exact for the leading 1)."""
    shift = MASTER - scale
    return (c + (1 << (shift - 1))) >> shift


def program(name, final=ACC, lead=ACC):
    """Lazy-rescale Horner program of a fit: `[("lead", c), ("step", c) | ("shr", bits), ..]`.

    The accumulator starts at scale `lead`; every step multiplies by the variable (scale `v`) and
    adds the next coefficient at the grown scale; the accumulator is brought back to `ACC` when
    one more step would need a shift larger than `MAX_SHIFT`. `final` is the scale of the result
    (`None`: whatever the last step leaves). Returns `(program, scale)`.
    """
    master, _ = fit(name)
    v = SPECS[name][4]
    scale = lead
    ops = [("lead", scaled(master[-1], scale))]
    for c in reversed(master[:-1]):
        if scale + v - ACC > MAX_SHIFT:
            ops.append(("shr", scale - ACC))
            scale = ACC
        scale += v
        ops.append(("step", scaled(c, scale)))
    if final is not None and scale != final:
        ops.append(("shr", scale - final))
        scale = final
    return ops, scale


# --- kernels -------------------------------------------------------------------------------------
# Every kernel returns a magnitude rounded ONCE to nearest (`shr_round`, ties up); signs,
# reflections, domain checks and panics are handled by the hand-written
# `kernels/transcendental.cairo`.

ANGLE_DOC = "Reduced angle in [0, pi/4], scale 2^61."
OCT_DOC = "Octant index (`x` reduced modulo 2*pi, in eighths of a turn)."
REM_DOC = "Remainder of the octant reduction, scale 2^61."


def k_reduce(m, name, shift, doc, cfg=None):
    """`x -> (n, r)`: `x * 2^(A-32) + OFF = q * (pi/4) + r`, `n = (q + shift) mod 8`."""
    k = Kernel(m, name, "Red", doc, cfg=cfg)
    x = k.input("x", I64_MIN, I64_MAX, ty="i64")
    q, r = k.div_rem(k.add(k.mul(x, 1 << (A - 32)), OFF_TRIG), QUARTER_PI)
    if shift:
        q = k.add(q, shift)
    _, n = k.div_rem(q, 8)
    return k.finish((n, r), ("Octant", "OctRem"), (OCT_DOC, REM_DOC))


def k_complement(m, doc):
    """`r -> pi/4 - r`, in the argument type of the polynomial kernels."""
    k = Kernel(m, "octant_complement", "Comp", doc)
    r = k.input("r", 0, QUARTER_PI - 1, alias="OctRem", alias_doc=REM_DOC)
    return k.finish(k.upcast(k.sub(QUARTER_PI, r), 0, QUARTER_PI), ("OctArg",), (ANGLE_DOC,))


def cos_round(k, acc, scale, bits):
    """Rounds a lazy Horner accumulator to `bits` fractional bits, in one `div_rem` when the
    shift fits the cheap algorithm and in two otherwise."""
    if scale - bits > MAX_SHIFT:
        acc, scale = k.shr_floor(acc, scale - ACC), ACC
    return k.shr_round(acc, scale - bits)


def square(k, r, a=A):
    """`z = floor(r^2 / 2^(2a - Z))`: the squared argument at scale 2^Z."""
    return k.shr_floor(k.mul(r, r), 2 * a - Z)


def k_sin(m, name, prefix, unit, fit_name, bits, doc, cfg=None):
    """`sin(r) = r * S(r^2)` for `r` in `[0, unit]`, rounded to `bits` fractional bits."""
    k = Kernel(m, name, prefix, doc, cfg=cfg)
    r = k.input("r", 0, unit, alias="OctArg", alias_doc=ANGLE_DOC)
    acc = k.horner(square(k, r), program(fit_name)[0])
    out = k.shr_round(k.mul(r, acc), A + ACC - bits)
    k.raw = {"Mag32" if bits == 32 else "Mag48": out.hi}
    hi, alias = (MAG32_HI, "Mag32") if bits == 32 else (MAG48_HI, "Mag48")
    return k.finish(k.upcast(out, 0, hi), (alias,), (f"Magnitude in [0, 1], scale 2^{bits}.",))


def k_cos(m, name, prefix, fit_name, bits, doc, cfg=None):
    """`cos(r) = C(r^2)` for `r` in `[0, pi/4]`, rounded to `bits` fractional bits."""
    k = Kernel(m, name, prefix, doc, cfg=cfg)
    r = k.input("r", 0, QUARTER_PI, alias="OctArg", alias_doc=ANGLE_DOC)
    ops, scale = program(fit_name, final=None)
    out = cos_round(k, k.horner(square(k, r), ops), scale, bits)
    k.raw = {"Mag32" if bits == 32 else "Mag48": out.hi}
    hi, alias = (MAG32_HI, "Mag32") if bits == 32 else (MAG48_HI, "Mag48")
    return k.finish(k.upcast(out, 0, hi), (alias,), (f"Magnitude in [0, 1], scale 2^{bits}.",))


def k_sin_cos(m, name, prefix, sin_fit, cos_fit, bits, doc, cfg=None):
    """`(sin r, cos r)` sharing `r^2`: the same operations as `k_sin` / `k_cos`, bit for bit."""
    k = Kernel(m, name, prefix, doc, cfg=cfg)
    r = k.input("r", 0, QUARTER_PI, alias="OctArg", alias_doc=ANGLE_DOC)
    z = square(k, r)
    s = k.shr_round(k.mul(r, k.horner(z, program(sin_fit)[0])), A + ACC - bits)
    ops, scale = program(cos_fit, final=None)
    c = cos_round(k, k.horner(z, ops), scale, bits)
    k.raw = {"Mag32" if bits == 32 else "Mag48": max(s.hi, c.hi)}
    hi, alias = (MAG32_HI, "Mag32") if bits == 32 else (MAG48_HI, "Mag48")
    doc_out = f"Magnitude in [0, 1], scale 2^{bits}."
    return k.finish(
        (k.upcast(s, 0, hi), k.upcast(c, 0, hi)), (alias, alias), (doc_out, doc_out)
    )


def k_atan(m, name, prefix, fit_name, doc, cfg=None):
    """`atan(t) = t * T(t^2)` for `t` in `[0, 1]` at scale 2^A, result Q32.32."""
    k = Kernel(m, name, prefix, doc, cfg=cfg)
    t = k.input("t", 0, 1 << A, alias="Ratio", alias_doc="Ratio in [0, 1], scale 2^61.")
    acc = k.horner(square(k, t), program(fit_name)[0])
    out = k.shr_round(k.mul(t, acc), A + ACC - 32)
    k.raw = {"AngQp": out.hi}
    return k.finish(k.upcast(out, 0, ANG_QP_HI), ("AngQp",), ("Angle in [0, pi/4], Q32.32.",))


def k_asin(m, name, prefix, fit_name, double, doc, cfg=None):
    """`asin(s)` (or `2 asin(s)`) for `s` in `[0, 1/2]` at scale 2^A, result Q32.32."""
    k = Kernel(m, name, prefix, doc, cfg=cfg)
    s = k.input("s", 0, 1 << (A - 1), alias="AsinArg", alias_doc="`s` in [0, 1/2], scale 2^61.")
    acc = k.horner(square(k, s), program(fit_name)[0])
    out = k.shr_round(k.mul(s, acc), A + ACC - 32 - (1 if double else 0))
    k.raw = {"AngPi3" if double else "AngPi6": out.hi}
    hi, alias, doc_out = (
        (ASIN2_HI, "AngPi3", "Angle in [0, pi/3], Q32.32.")
        if double
        else (ASIN_HI, "AngPi6", "Angle in [0, pi/6], Q32.32.")
    )
    return k.finish(k.upcast(out, 0, hi), (alias,), (doc_out,))


def k_tan_ratio(m, doc):
    """`round(s * 2^32 / c)` for the wide sine and cosine (scale 2^48), `c != 0`.

    The quotient is taken at scale 2^33 and halved with the usual round-to-nearest shift, so the
    non-zero divisor is the caller's `NonZero<Mag48>` and no second product is needed.
    """
    k = Kernel(m, "tan_ratio", "Tan", doc)
    s = k.input("s", 0, MAG48_HI, alias="Mag48")
    c = k.input("c", 0, MAG48_HI, alias="Mag48", nz=True)
    q, _ = k.div_rem(k.mul(s, 1 << 33), c)
    return k.finish(k.shr_round(q, 1), ("TanMag",),
                    ("`|tan|` at Q32.32, before the overflow check.",))


# --- module assembly -------------------------------------------------------------------------------

HEADER = """//! GENERATED by `tools/polygen/polygen.py`: do not edit by hand.
//!
//! Typed polynomial kernels of the transcendental functions (DESIGN D6): range reductions, Horner
//! evaluations and final roundings, entirely in the `BoundedInt` domain.
//!
//! Every value carries the *exact* interval of its operation, computed by the generator with
//! Python integers and re-checked by the Cairo compiler, so these kernels contain **no overflow
//! check, no sign branch and no rounding but the documented one**. Intermediates keep 64
//! fractional bits (2^-64), far below the 2^-32 resolution of `Fixed`, and each kernel rounds its
//! result once, to nearest, ties up.
//!
//! Signs, reflections, domain checks and panics live in the hand-written sibling module
//! `kernels::transcendental`. Coefficients are near-minimax (Chebyshev) fits of
//! `f = 1 + z * G(z)` with the leading 1 exact, so `sin(0) = 0`, `cos(0) = 1`, `atan(0) = 0` and
//! `asin(0) = 0` hold bit for bit.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, upcast,
};

/// `pi/4` at scale 2^61: the unit of the octant reduction.
pub const QUARTER_PI: felt252 = $QUARTER_PI;
/// `pi/2` at scale 2^61.
pub const HALF_PI: felt252 = $HALF_PI;
"""

ALT_HEADER = """//! GENERATED by `tools/polygen/polygen.py`: do not edit by hand.
//!
//! Polynomial kernels that LOST the accuracy / gas comparison against `kernels::poly`, kept as
//! evidence (AGENTS.md rule 8) and compiled under `#[cfg(test)]` only. `polygen.py --report`
//! prints the measured error of every candidate, and `kernels::transcendental` benchmarks the
//! variants side by side.

#[feature("bounded-int-utils")]
use core::internal::bounded_int::{
    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, SubHelper, UnitInt, upcast,
};
use super::poly::{$IMPORTS};
"""


def deg(name):
    """Degree of the polynomial in the natural variable (`r`, `t`, `s`)."""
    n = SPECS[name][3] + 1
    return 2 * n - 2 if name.startswith("COS") else 2 * n - 1


# Chosen variants (see `--report` and the `bench_*` groups of `fixed::transcendental`). One extra
# coefficient costs ~100 gas out of ~12,800 (the lazy rescale absorbs three multiplications between
# two `div_rem`s), so the sine and the cosine are taken at the degree where the polynomial error
# disappears below the final rounding (0.53 ulp, measured); `atan` and `asin` stop one degree
# earlier, where the extra step costs 400 gas and buys less than 0.3 ulp.
SIN_FIT = "SIN_O5"
COS_FIT = "COS_O5"
ATAN_FIT = "ATAN_12"
ASIN_FIT = "ASIN_7"

FAMILIES = {
    "SIN_O": ("sin_octant", "SinO", lambda m, n, d, c: k_sin(
        m, n, f"SinO{d}", QUARTER_PI, c, 32, f"`sin(r)` on the octant, degree {d}.", cfg="test")),
    "COS_O": ("cos_octant", "CosO", lambda m, n, d, c: k_cos(
        m, n, f"CosO{d}", c, 32, f"`cos(r)` on the octant, degree {d}.", cfg="test")),
    "ATAN_": ("atan_unit", "Atan", lambda m, n, d, c: k_atan(
        m, n, f"Atan{d}", c, f"`atan(t)` on [0, 1], degree {d}.", cfg="test")),
    "ASIN_": ("asin_unit", "Asin", lambda m, n, d, c: k_asin(
        m, n, f"Asin{d}", c, False, f"`asin(s)` on [0, 1/2], degree {d}.", cfg="test")),
}
CHOSEN = {"SIN_O": SIN_FIT, "COS_O": COS_FIT, "ATAN_": ATAN_FIT, "ASIN_": ASIN_FIT}


def build_main():
    """The shipped kernels."""
    m = Module("poly")
    k_reduce(m, "reduce_octant", 0, "`sin`: octant index and remainder of `x` (Q32.32 radians).")
    k_reduce(m, "reduce_octant_cos", 2, "`cos(x) = sin(x + pi/2)`: the octant shifted by two.")
    k_complement(m, "`pi/4 - r`: the reflection used by the odd octants.")
    k_sin(m, "sin_octant", "SinO", QUARTER_PI, SIN_FIT, 32,
          f"`sin(r)` for `r` in [0, pi/4], degree {deg(SIN_FIT)}.")
    k_cos(m, "cos_octant", "CosO", COS_FIT, 32,
          f"`cos(r)` for `r` in [0, pi/4], degree {deg(COS_FIT)}.")
    k_sin_cos(m, "sin_cos_octant", "ScO", SIN_FIT, COS_FIT, 32,
              "`(sin r, cos r)` sharing `r^2`: bit for bit `(sin_octant(r), cos_octant(r))`.")
    k_sin_cos(m, "sin_cos_octant_wide", "ScW", SIN_FIT, COS_FIT, 48,
              "`(sin r, cos r)` at scale 2^48: the internal precision of `tan`.")
    k_tan_ratio(m, "`|tan|` from a wide sine and cosine: one rounding to nearest, ties up.")
    k_atan(m, "atan_unit", "Atan", ATAN_FIT,
           f"`atan(t)` for `t` in [0, 1], degree {deg(ATAN_FIT)}.")
    k_asin(m, "asin_unit", "Asin", ASIN_FIT, False,
           f"`asin(s)` for `s` in [0, 1/2], degree {deg(ASIN_FIT)}.")
    k_asin(m, "asin_double", "AsinD", ASIN_FIT, True,
           f"`2 * asin(s)` for `s` in [0, 1/2], degree {deg(ASIN_FIT)} (`acos` of a big argument).")
    return m


def build_alternatives(main, keep_chosen=False):
    """The candidates kept as `#[cfg(test)]` evidence (`keep_chosen`: all of them, for `--report`)."""
    m = Module("poly_alternatives")
    # Type aliases are imported from `poly`, not re-declared; trait impls are module-local.
    m.seen.update({k: v for k, v in main.seen.items() if "<" not in k})
    for prefix, (_, _, make) in FAMILIES.items():
        for name in SPECS:
            if name.startswith(prefix) and (keep_chosen or name != CHOSEN[prefix]):
                d = deg(name)
                make(m, f"{FAMILIES[prefix][0]}_{d}", d, name)
                if prefix == "ASIN_":
                    k_asin(m, f"asin_double_{d}", f"AsinD{d}", name, True,
                           f"`2 * asin(s)` on [0, 1/2], degree {d}.", cfg="test")
    return m


def render(module, main=None):
    if module.name == "poly":
        text = HEADER.replace("$QUARTER_PI", hx(QUARTER_PI)).replace("$HALF_PI", hx(HALF_PI))
    else:
        body = module.render()
        names = sorted(k for k in main.seen if "<" not in k and f"{k}" in body)
        text = ALT_HEADER.replace("$IMPORTS", ", ".join(names))
    if module.needs_is_zero:
        text += (
            "\n// `core::zeroable::IsZeroResult` is crate-private in the corelib: the libfunc is\n"
            "// re-declared with a local result enum.\nenum IsZero<T> {\n    Zero,\n"
            "    NonZero: NonZero<T>,\n}\n"
            "extern fn bounded_int_is_zero<T>(value: T) -> IsZero<T> implicits() nopanic;\n"
        )
    return text + "\n" + module.render()


def render_ops(modules):
    """`tools/fixed_model/poly_ops.py`: the recorded programs, for the bit-exact model."""
    out = [
        '"""GENERATED by `tools/polygen/polygen.py`: do not edit by hand.',
        "",
        "The straight-line integer programs of `simba::fixed::kernels::poly`, recorded operation by",
        "operation. `run(name, *args)` executes exactly what the Cairo kernel executes, so the",
        "Python model of `tools/fixed_model` cannot drift from the generated code.",
        '"""',
        "",
        "# Scales and constants of the generator (`tools/polygen/polygen.py`).",
        f"A = {A}",
        f"Z = {Z}",
        f"ACC = {ACC}",
        f"QUARTER_PI = {QUARTER_PI}",
        f"HALF_PI = {HALF_PI}",
        f"OFF_TRIG = {OFF_TRIG}",
        "",
        "KERNELS = {",
    ]
    for module in modules:
        for k in module.kernels:
            rec = k.record()
            out.append(f'    "{k.name}": {{')
            out.append(f'        "params": {rec["params"]!r},')
            out.append(f'        "outs": {rec["outs"]!r},')
            out.append('        "ops": [')
            for op in rec["ops"]:
                out.append(f"            {op!r},")
            out.append("        ],")
            out.append("    },")
    out += [
        "}",
        "",
        "OPS = {",
        '    "mul": lambda x, y: x * y,',
        '    "add": lambda x, y: x + y,',
        '    "sub": lambda x, y: x - y,',
        "}",
        "",
        "",
        "def run(name, *args):",
        '    """Runs a recorded kernel on Python integers, checking the input intervals."""',
        "    rec = KERNELS[name]",
        "    env = {}",
        '    for (pname, lo, hi), a in zip(rec["params"], args):',
        "        assert lo <= a <= hi, (name, pname, a, lo, hi)",
        "        env[pname] = a",
        "",
        "    def get(e):",
        "        return env[e] if e in env else int(e, 16)",
        "",
        '    for op, out, a, b in rec["ops"]:',
        '        if op == "upcast":',
        "            env[out] = get(a)",
        '        elif op == "div_rem":',
        "            env[out[0]], env[out[1]] = divmod(get(a), get(b))",
        "        else:",
        "            env[out] = OPS[op](get(a), get(b))",
        '    res = [env[o] for o in rec["outs"]]',
        "    return res[0] if len(res) == 1 else tuple(res)",
        "",
    ]
    return "\n".join(out)


# --- accuracy report ---------------------------------------------------------------------------


# Q32.32 reflection constants, FLOORED, identical to `simba::fixed::types` (so that
# `atan2(0, x < 0) == Real::PI` and `atan2(y > 0, 0) == Real::FRAC_PI_2` exactly).
HALF_PI_Q32 = 6746518852
PI_Q32 = 2 * HALF_PI_Q32


def _reduce(x, shift=0):
    q, r = divmod(x * (1 << (A - 32)) + OFF_TRIG, QUARTER_PI)
    return (q + shift) % 8, r


def sin_model(run, sin_name="sin_octant", cos_name="cos_octant"):
    """`sin` through the candidate kernels (mirrors `kernels::transcendental::sin`)."""

    def f(x, shift=0):
        n, r = _reduce(x, shift)
        high, m = divmod(n, 4)
        q, low = divmod(m, 2)
        arg = QUARTER_PI - r if low else r
        mag = run(cos_name, arg) if q != low else run(sin_name, arg)
        return -mag if high else mag

    return f


def atan2_model(run, name="atan_unit"):
    """`atan2` through a candidate `atan` kernel (mirrors `kernels::transcendental::atan2`)."""

    def f(y, x):
        ax, ay = abs(x), abs(y)
        swap = ay > ax
        num, den = (ax, ay) if swap else (ay, ax)
        if den == 0:
            return 0
        a = run(name, (num << A) // den)
        if swap:
            a = HALF_PI_Q32 - a
        if x < 0:
            a = PI_Q32 - a
        return -a if y < 0 else a

    return f


def asin_model(run, name="asin_unit", acos=False):
    """`asin` / `acos` (mirrors `kernels::transcendental::asin_acos`)."""

    def f(x):
        ax = abs(x)
        assert ax <= ONE
        if ax <= ONE // 2:
            a = run(name, ax << (A - 32))
            value = HALF_PI_Q32 - a if acos else a
        else:
            s = math.isqrt((ONE - ax) << (2 * A - 33))
            d = run(name.replace("asin_unit", "asin_double"), s)
            value = d if acos else HALF_PI_Q32 - d
        if x < 0:
            return PI_Q32 - value if acos else -value
        return value

    return f


def variants(prefix):
    """Names of the generated kernels of a family, lowest degree first."""
    return [f"{FAMILIES[prefix][0]}_{deg(n)}" for n in SPECS if n.startswith(prefix)]


def report():
    main = build_main()
    alt = build_alternatives(main, keep_chosen=True)
    kernels = {k.name: k for mod in (main, alt) for k in mod.kernels}
    run = lambda name, *a: kernels[name].run(*a)  # noqa: E731
    rng = random.Random(7)

    print("Uniform fit error of every candidate, in Q32.32 ulp:\n")
    print("| fit | degree | coefficients | uniform error (ulp) |")
    print("|---|---:|---:|---:|")
    for name in SPECS:
        _, err = fit(name)
        print(f"| `{name}` | {deg(name)} | {SPECS[name][3] + 1} | {float(err) * ONE:.3f} |")

    def show(fn, name, variant, points, ref):
        worst, arg = 0, None
        for p in points:
            d = abs(fn(*p) - ref(*[mpf(v) / ONE for v in p]) * ONE)
            if d > worst:
                worst, arg = d, p
        print(f"| `{name}` | `{variant}` | {float(worst):.2f} | {arg} |")

    print("\nEnd-to-end error against mpmath, in Q32.32 ulp (dense + random sweeps):\n")
    print("| function | kernel | max ulp | worst input (raw) |")
    print("|---|---|---:|---|")

    angles = [rnd(mpf(-7) + mpf(14) * i / 30000) for i in range(30001)]
    angles += [rng.randrange(-(1 << 40), 1 << 40) for _ in range(10000)]
    pts = [(x,) for x in angles]
    for sn, cn in zip(variants("SIN_O"), variants("COS_O")):
        show(sin_model(run, sn, cn), "sin", f"{sn} / {cn}", pts, mp.sin)
        show(lambda x, s=sn, c=cn: sin_model(run, s, c)(x, 2), "cos", f"{sn} / {cn}", pts, mp.cos)

    circle = [
        (rnd(rad * mp.sin(t)), rnd(rad * mp.cos(t)))
        for rad in (mpf("0.001"), mpf(1), mpf(1000), mpf(100000))
        for t in (-mp.pi + 2 * mp.pi * (i + mpf(1) / 2) / 4000 for i in range(4000))
    ]
    for name in variants("ATAN_"):
        show(atan2_model(run, name), "atan2", name, circle, mp.atan2)

    print("\nRange-reduction drift of `sin` / `cos` for large arguments (chosen kernels):\n")
    print("| |x| up to | max ulp |")
    print("|---|---:|")
    f = sin_model(run)
    for bits in (34, 40, 46, 52, 58, 63):
        pts = [rng.randrange(-(1 << bits), 1 << bits) for _ in range(20000)]
        worst = max(abs(f(x) - mp.sin(mpf(x) / ONE) * ONE) for x in pts)
        print(f"| 2^{bits - 32} rad | {float(worst):.2f} |")

    unit = [rnd(mpf(-1) + 2 * mpf(i) / 40000) for i in range(40001)]
    unit += [rng.randrange(-ONE, ONE + 1) for _ in range(10000)]
    pts = [(x,) for x in unit]
    for name in variants("ASIN_"):
        show(asin_model(run, name), "asin", name, pts, mp.asin)
    show(asin_model(run, acos=True), "acos", ASIN_FIT, pts, mp.acos)


# --- writing --------------------------------------------------------------------------------------


def squash(s):
    return "".join(s.split()).replace(",]", "]").replace(",)", ")").replace(",>", ">")


def scarb_fmt():
    subprocess.run(["scarb", "fmt", "-p", "simba"], cwd=ROOT, check=True, capture_output=True)


def outputs():
    main = build_main()
    alt = build_alternatives(main)
    return {
        OUT_POLY: render(main),
        OUT_ALT: render(alt, main),
        OUT_OPS: render_ops([main, alt]),
    }


def cross_check():
    """Replays the generated kernels through `tools/fixed_model` (the model cannot drift)."""
    sys.path.insert(0, str(ROOT / "tools" / "fixed_model"))
    for mod in ("poly_ops", "fixed_model"):
        sys.modules.pop(mod, None)
    import fixed_model as fm  # noqa: E402

    fm.self_check_transcendental()
    print("polygen: model cross-check passed")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--check", action="store_true", help="fail if a committed file is stale")
    parser.add_argument("--report", action="store_true", help="accuracy of every candidate")
    parser.add_argument("--no-fmt", action="store_true", help="do not run `scarb fmt` afterwards")
    args = parser.parse_args()

    if args.report:
        report()
        return

    files = outputs()
    if args.check:
        for path, text in files.items():
            current = path.read_text() if path.exists() else ""
            if squash(current) != squash(text):
                sys.exit(f"{path.relative_to(ROOT)} is stale: run tools/polygen/polygen.py")
        print("polygen: generated files are up to date")
    else:
        for path, text in files.items():
            path.write_text(text)
        if not args.no_fmt:
            scarb_fmt()
        for path in files:
            print(f"wrote {path.relative_to(ROOT)}")
    cross_check()


if __name__ == "__main__":
    main()
