#!/usr/bin/env python3
"""Algorithm benchmarks for the BI64 representation: tables, unrolled code, tests, accuracy.

Generates (do not edit by hand):
  src/algo/seeds.cairo   power-of-two Newton seeds as compare trees (no bitwise op)
  src/algo/lut.cairo     sin tables (const arrays + `match`)
  src/algo/cordic.cairo  unrolled CORDIC
  src/tests/algo.cairo   benchmarks; expected values are BIT-EXACT outputs of the integer models
                         below, which mirror the Cairo code operation by operation.
and prints the accuracy table (max abs error of each model against double precision math).

Usage: python3 gen_algos.py   (from benchmarks/scalar)
"""

import math
from math import isqrt
from pathlib import Path

ROOT = Path(__file__).parent / "src"
ONE = 1 << 32
HALF_PI = round(math.pi / 2 * ONE)
PI = round(math.pi * ONE)
QUARTER_PI = round(math.pi / 4 * ONE)
TAN_PI_8 = 0x6A09E667
OFFSET = HALF_PI * 1367130552
INV_HALF_PI = (1 << 64) // HALF_PI
assert HALF_PI == 0x1921FB544 and PI == 0x3243F6A89 and QUARTER_PI == 0xC90FDAA2
assert OFFSET == 0x800000013F628CE0 and INV_HALF_PI == 0xA2F9836E
assert HALF_PI * INV_HALF_PI == 0xFFFFFFFF5A85AF38 and (HALF_PI * INV_HALF_PI) >> 32 < 1 << 32

SIN7 = [4294964019, -715785355, 35704885, -795537]
SIN9 = [4294967278, -715827508, 35790178, -850791, 11189]
SIN11 = [4294967296, -715827881, 35791384, -852158, 11822, -103]
ATAN15 = [4294966789, -1431590453, 857574481, -601436734, 423864544, -252671866, 102136550, -19584157]
ATAN9 = [4294967216, -1431632248, 857880307, -594788126, 342579125]
ACOS7 = [6746518730, -921692060, 382118315, -215239293, 131950511, -72331769, 27880872, -5204374]
CORDIC_N = 20
CORDIC_ATAN = [round(math.atan(2.0 ** -i) * ONE) for i in range(CORDIC_N)]
CORDIC_K = round(ONE * math.prod(1 / math.sqrt(1 + 4.0 ** -i) for i in range(CORDIC_N)))


def raw(v):
    return round(v * ONE)


# ------------------------------------------------------------------------------------------------
# Integer models (mirror the Cairo code exactly; `>>` floors like the BI64 rescale)
# ------------------------------------------------------------------------------------------------

def mul(a, b):
    return (a * b) >> 32


def mul_add(a, b, c):
    return (a * b + (c << 32)) >> 32


def div(a, b):
    return (a << 32) // b  # floor


def tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def sqrt_fix(a):
    return isqrt(a << 32)


def reduce(x, shift=0):
    q, r = divmod(x + OFFSET, HALF_PI)
    n = (q + shift) % 4
    return (HALF_PI - r if n % 2 else r), n >= 2


def horner_odd(t, coefs):
    z = mul(t, t)
    acc = mul_add(z, coefs[-1], coefs[-2])
    for k in reversed(coefs[:-2]):
        acc = mul_add(z, acc, k)
    return mul(t, acc)


def sin_poly(coefs, shift=0):
    def f(x):
        r, neg = reduce(x, shift)
        s = horner_odd(r, coefs)
        return -s if neg else s
    return f


def sin_poly9_unfused(x):
    r, neg = reduce(x)
    z = mul(r, r)
    acc = mul(z, SIN9[4]) + SIN9[3]
    for k in (SIN9[2], SIN9[1], SIN9[0]):
        acc = mul(z, acc) + k
    s = mul(r, acc)
    return -s if neg else s


def sin_taylor_loop(x):
    r, neg = reduce(x)
    z = mul(r, r)
    acc = ONE
    for i in range(7, 0, -1):
        acc = ONE - tdiv(mul(z, acc), (2 * i) * (2 * i + 1))
    s = mul(r, acc)
    return -s if neg else s


def sin_table(n):
    return [round(math.sin(i * math.pi / 2 / n) * ONE) for i in range(n + 1)]


def sin_lut(bits):
    table = sin_table(1 << bits)
    shift = 32 - bits

    def f(x):
        r, neg = reduce(x)
        v = (r * INV_HALF_PI) >> 32
        idx, frac = v >> shift, v & ((1 << shift) - 1)
        s = table[idx] + (table[idx + 1] - table[idx]) * frac // (1 << shift)
        return -s if neg else s
    return f


def cordic_q(r):
    x, y, z = CORDIC_K, 0, r
    for i in range(CORDIC_N):
        d = 1 << i
        if z >= 0:
            x, y, z = x - tdiv(y, d), y + tdiv(x, d), z - CORDIC_ATAN[i]
        else:
            x, y, z = x + tdiv(y, d), y - tdiv(x, d), z + CORDIC_ATAN[i]
    return y


def sin_cordic(x):
    r, neg = reduce(x)
    s = cordic_q(r)
    return -s if neg else s


def atan_unit_reduced(t):
    if t <= TAN_PI_8:
        return horner_odd(t, ATAN9)
    return QUARTER_PI - horner_odd(div(ONE - t, ONE + t), ATAN9)


def atan2(unit):
    def f(y, x):
        ax, ay = abs(x), abs(y)
        if ay <= ax:
            if ax == 0:
                return 0
            a = unit(div(ay, ax))
        else:
            a = HALF_PI - unit(div(ax, ay))
        if x < 0:
            a = PI - a
        return -a if y < 0 else a
    return f


atan2_poly15 = atan2(lambda t: horner_odd(t, ATAN15))
atan2_reduced9 = atan2(atan_unit_reduced)


def acos_poly7(x):
    ax = abs(x)
    s = sqrt_fix(ONE - ax)
    acc = mul_add(ax, ACOS7[-1], ACOS7[-2])
    for k in reversed(ACOS7[:-2]):
        acc = mul_add(ax, acc, k)
    r = mul(s, acc)
    return PI - r if x < 0 else r


def acos_via_atan2(x):
    return atan2_reduced9(sqrt_fix(mul_add(-x, x, ONE)), x)


def rsqrt_seed(m):
    j = (m.bit_length() + 1) // 2
    return 1 << (48 - j)


def sqrt_seed(m):
    j = (m.bit_length() + 1) // 2
    return 1 << (16 + j)


def inv_sqrt_newton(a):
    half_a = mul(a, ONE // 2)
    y = rsqrt_seed(a)
    for _ in range(6):
        y = mul(y, ONE + ONE // 2 - mul(mul(half_a, y), y))
    return y


def sqrt_newton_seeded(a):
    n = a << 32
    x = sqrt_seed(a)
    for _ in range(6):
        x = (x + n // x) // 2
    return x - 1 if x * x > n else x


# ------------------------------------------------------------------------------------------------
# Typed Horner: everything stays in the BoundedInt domain with 64 fractional bits. Bounds are
# tracked at compile time by interval arithmetic, so there is NO overflow check at all and the
# intermediate precision is 2^-64 instead of 2^-32.
# ------------------------------------------------------------------------------------------------

def fit64(g, zmax, n):
    """Near-minimax (Chebyshev) fit of g on [0, zmax], n coefficients, scaled by 2^64."""
    from mpmath import chebyfit, mp, nint
    mp.prec = 200
    return [int(nint(k * 2 ** 64)) for k in reversed(chebyfit(g, [0, zmax], n))]


def hx(v):
    return ("-" if v < 0 else "") + hex(abs(v))


def bi(lo, hi):
    return f"BoundedInt<{hx(lo)}, {hx(hi)}>"


def imul(a, b):
    c = [x * y for x in a for y in b]
    return min(c), max(c)


SEEN_IMPLS = set()


def gen_typed_poly(name, fn_name, coefs, doc, in_ty, in_rng, square):
    """Cairo for m * P(v) with coefficients at scale 2^64, bounds tracked by interval arithmetic.

    square=True : fn(r) = r * P(r^2)      (v = r * r, exact, scale 2^64; m = r)
    square=False: fn(v, m) = m * P(v)     (v and m share the same type, scale 2^32)
    """
    impls, body = [], []
    if square:
        v_rng, v_bits, v_name = imul(in_rng, in_rng), 64, "z"
        v_ty = bi(*v_rng)
        impls.append(f"impl {name}Sq of MulHelper<{in_ty}, {in_ty}> {{\n    type Result = {v_ty};\n}}")
        body.append(f"    let z: {v_ty} = bounded_int::mul::<{in_ty}, {in_ty}>(r, r);")
        sig, m_name = f"r: {in_ty}", "r"
    else:
        v_rng, v_bits, v_name, v_ty = in_rng, 32, "v", in_ty
        sig, m_name = f"v: {in_ty}, m: {in_ty}", "m"
    acc_rng, acc_ty, acc_val = (coefs[-1], coefs[-1]), f"UnitInt<{hx(coefs[-1])}>", hx(coefs[-1])
    steps = list(reversed(coefs[:-1])) + [None]
    for i, k in enumerate(steps):
        last = k is None
        lhs_ty, lhs_rng, lhs, bits = (in_ty, in_rng, m_name, 64) if last else (v_ty, v_rng, v_name, v_bits)
        shift = 1 << bits
        p_rng = imul(lhs_rng, acc_rng)
        p_ty = bi(*p_rng)
        impls.append(f"impl {name}Mul{i} of MulHelper<{lhs_ty}, {acc_ty}> {{\n    type Result = {p_ty};\n}}")
        body.append(f"    let p: {p_ty} = bounded_int::mul::<{lhs_ty}, {acc_ty}>({lhs}, {acc_val});")
        koff = max(0, -(p_rng[0] // shift))  # offset (in quotient units) making the sum >= 0
        const = koff * shift
        s_rng = (p_rng[0] + const, p_rng[1] + const)
        assert s_rng[0] >= 0
        if const:
            s_ty = bi(*s_rng)
            impls.append(f"impl {name}Off{i} of AddHelper<{p_ty}, UnitInt<{hx(const)}>> {{\n    type Result = {s_ty};\n}}")
            body.append(f"    let p: {s_ty} = bounded_int::add::<{p_ty}, UnitInt<{hx(const)}>>(p, {hx(const)});")
        else:
            s_ty = p_ty
        q_rng = (s_rng[0] // shift, s_rng[1] // shift)
        q_ty = bi(*q_rng)
        impls.append(
            f"impl {name}Div{i} of DivRemHelper<{s_ty}, UnitInt<{hx(shift)}>> {{\n"
            f"    type DivT = {q_ty};\n    type RemT = {bi(0, shift - 1)};\n}}"
        )
        body.append(f"    let (q, _) = bounded_int::div_rem::<{s_ty}, UnitInt<{hx(shift)}>>(p, {hx(shift)});")
        # next accumulator = q - koff + k  (one constant, may be negative or positive)
        delta = (0 if last else k) - koff
        acc_rng = (q_rng[0] + delta, q_rng[1] + delta)
        acc_ty = bi(*acc_rng)
        if delta:
            impls.append(f"impl {name}Add{i} of AddHelper<{q_ty}, UnitInt<{hx(delta)}>> {{\n    type Result = {acc_ty};\n}}")
            body.append(f"    let acc: {acc_ty} = bounded_int::add::<{q_ty}, UnitInt<{hx(delta)}>>(q, {hx(delta)});")
        else:
            body.append("    let acc = q;")
        acc_val = "acc"
    assert -(1 << 40) <= acc_rng[0] and acc_rng[1] < 1 << 40, acc_rng
    impls = [i for i in impls if i.split(" of ", 1)[1].split(" {")[0] not in SEEN_IMPLS
             and not SEEN_IMPLS.add(i.split(" of ", 1)[1].split(" {")[0])]
    out = "\n".join(impls) + f"\n\npub type {name}Out = {acc_ty};\n\n/// {doc}\n#[inline(always)]\n"
    out += f"pub fn {fn_name}({sig}) -> {name}Out {{\n" + "\n".join(body) + "\n    acc\n}\n\n"
    return out


def typed_coefs():
    from mpmath import mp, mpf, sin, atan, acos, sqrt, pi
    mp.prec = 200
    g_sin = lambda z: sin(sqrt(z)) / sqrt(z) if z != 0 else mpf(1)
    g_atan = lambda z: atan(sqrt(z)) / sqrt(z) if z != 0 else mpf(1)
    g_acos = lambda x: acos(x) / sqrt(1 - x) if x != 1 else sqrt(mpf(2))
    return {
        "sin9": fit64(g_sin, (pi / 2) ** 2, 5),
        "sin11": fit64(g_sin, (pi / 2) ** 2, 6),
        "atan15": fit64(g_atan, 1, 8),
        "atan19": fit64(g_atan, 1, 10),
        "acos9": fit64(g_acos, 1, 10),
    }


def gen_typed():
    coefs = typed_coefs()
    out = "// GENERATED by gen_algos.py - do not edit.\n"
    out += "//! Typed Horner kernels: BoundedInt end to end, 64 fractional bits inside, bounds proven at\n"
    out += "//! compile time (interval arithmetic) => no overflow check, no sign branch.\n\n"
    out += '#[feature("bounded-int-utils")]\nuse core::internal::bounded_int::{\n'
    out += "    self, AddHelper, BoundedInt, DivRemHelper, MulHelper, UnitInt,\n};\n\n"
    out += f"/// Reduced angle, [0, pi/2] in Q32.32.\npub type RFull = {bi(0, HALF_PI)};\n"
    out += f"/// [0, 1] in Q32.32.\npub type TUnit = {bi(0, ONE)};\n\n"
    rf, tu = (bi(0, HALF_PI), (0, HALF_PI)), (bi(0, ONE), (0, ONE))
    out += gen_typed_poly("SinQ9", "sin_q9", coefs["sin9"], "sin(r), r in [0, pi/2], degree 9.", *rf, True)
    out += gen_typed_poly("SinQ11", "sin_q11", coefs["sin11"], "sin(r), r in [0, pi/2], degree 11.", *rf, True)
    out += gen_typed_poly("Atan15", "atan_unit15", coefs["atan15"], "atan(r), r in [0, 1], degree 15.", *tu, True)
    out += gen_typed_poly("Atan19", "atan_unit19", coefs["atan19"], "atan(r), r in [0, 1], degree 19.", *tu, True)
    out += gen_typed_poly(
        "Acos9", "acos_kernel9", coefs["acos9"],
        "m * P9(v): acos(v) = sqrt(1 - v) * P9(v) on [0, 1] with m = sqrt(1 - v).", *tu, False,
    )
    (ROOT / "algo" / "poly_typed.cairo").write_text(out)
    return coefs


def typed_horner(coefs, v, bits, m):
    acc = coefs[-1]
    for k in reversed(coefs[:-1]):
        acc = ((v * acc) >> bits) + k
    return (m * acc) >> 64


def typed_model(coefs):
    def f(x, shift=0):
        r, neg = reduce(x, shift)
        s = typed_horner(coefs, r * r, 64, r)
        return -s if neg else s
    return f


def typed_atan2(coefs):
    return atan2(lambda t: typed_horner(coefs, t * t, 64, t))


def typed_acos(coefs):
    def f(x):
        ax = abs(x)
        r = typed_horner(coefs, ax, 32, isqrt((ONE - ax) << 32))
        return PI - r if x < 0 else r
    return f


# ------------------------------------------------------------------------------------------------
# Accuracy
# ------------------------------------------------------------------------------------------------

def sweep(model, ref, points):
    return max(abs(model(*p) / ONE - ref(*(v / ONE for v in p))) for p in points)


def accuracy():
    angles = [(raw(-7 + 14 * i / 20000),) for i in range(20001)]
    unit = [(raw(-1 + 2 * i / 20000),) for i in range(20001)]
    circle = [
        (raw(rad * math.sin(t)), raw(rad * math.cos(t)))
        for rad in (0.001, 1.0, 1000.0)
        for t in (-math.pi + 2 * math.pi * (i + 0.5) / 4000 for i in range(4000))
    ]
    pos = [(raw(10 ** (-3 + 9 * i / 4000)),) for i in range(4001)]
    rows = [
        ("sin_poly7", sweep(sin_poly(SIN7), math.sin, angles)),
        ("sin_poly9 / cos_poly9", sweep(sin_poly(SIN9), math.sin, angles)),
        ("cos_poly9", sweep(sin_poly(SIN9, 1), math.cos, angles)),
        ("sin_poly9_unfused", sweep(sin_poly9_unfused, math.sin, angles)),
        ("sin_poly11", sweep(sin_poly(SIN11), math.sin, angles)),
        ("sin_poly9_typed", sweep(typed_model(TYPED["sin9"]), math.sin, angles)),
        ("sin_poly11_typed / cos_poly11_typed", sweep(typed_model(TYPED["sin11"]), math.sin, angles)),
        ("sin_taylor_loop", sweep(sin_taylor_loop, math.sin, angles)),
        ("sin_lut64 (match/array)", sweep(sin_lut(6), math.sin, angles)),
        ("sin_lut256_array", sweep(sin_lut(8), math.sin, angles)),
        ("sin_cordic20", sweep(sin_cordic, math.sin, angles)),
        ("atan2_poly15", sweep(atan2_poly15, math.atan2, circle)),
        ("atan2_reduced9", sweep(atan2_reduced9, math.atan2, circle)),
        ("atan2_poly15_typed", sweep(typed_atan2(TYPED["atan15"]), math.atan2, circle)),
        ("atan2_poly19_typed", sweep(typed_atan2(TYPED["atan19"]), math.atan2, circle)),
        ("acos_poly7", sweep(acos_poly7, math.acos, unit)),
        ("acos_via_atan2", sweep(acos_via_atan2, math.acos, unit)),
        ("acos_poly9_typed", sweep(typed_acos(TYPED["acos9"]), math.acos, unit)),
        ("sqrt_corelib / newton (exact floor)", sweep(sqrt_fix, math.sqrt, pos)),
        ("inv_sqrt_direct / inv_sqrt_div, a in [1e-3, 1e6]",
         sweep(lambda a: (1 << 64) // sqrt_fix(a), lambda a: 1 / math.sqrt(a), pos)),
        ("inv_sqrt_newton, a in [1e-3, 1e6]", sweep(inv_sqrt_newton, lambda a: 1 / math.sqrt(a), pos)),
        ("sqrt_via_inv_sqrt, a in [1e-3, 1e6]",
         sweep(lambda a: mul(a, inv_sqrt_newton(a)), math.sqrt, pos)),
    ]
    print("| algorithm | max abs error |")
    print("|---|---:|")
    for name, err in rows:
        print(f"| `{name}` | {err:.2e} |")


# ------------------------------------------------------------------------------------------------
# Code generation
# ------------------------------------------------------------------------------------------------

def tree(lo, hi, leaf, ind):
    pad = "    " * ind
    if lo == hi:
        return f"{pad}{leaf(lo)}\n"
    mid = (lo + hi) // 2
    return (
        f"{pad}if m < {hex(1 << (2 * mid))} {{\n" + tree(lo, mid, leaf, ind + 1)
        + f"{pad}}} else {{\n" + tree(mid + 1, hi, leaf, ind + 1) + f"{pad}}}\n"
    )


def gen_seeds():
    out = "// GENERATED by gen_algos.py - do not edit.\n"
    out += "//! Power-of-two Newton seeds from a 5-level compare tree on the magnitude (no bitwise op).\n\n"
    out += "/// 2^(16 + j) >= sqrt(m << 32), with j = ceil(bit_length(m) / 2).\n"
    out += "pub fn sqrt_seed(m: u64) -> u64 {\n" + tree(1, 32, lambda j: hex(1 << (16 + j)), 1) + "}\n\n"
    out += "/// 2^(48 - j) <= 2^48 / sqrt(m): safe (convergent) seed for the inverse sqrt Newton.\n"
    out += "pub fn rsqrt_seed(m: u64) -> u64 {\n" + tree(1, 32, lambda j: hex(1 << (48 - j)), 1) + "}\n"
    (ROOT / "algo" / "seeds.cairo").write_text(out)


def gen_lut():
    out = "// GENERATED by gen_algos.py - do not edit.\n"
    out += "//! sin(i * pi / 2 / N) in Q32.32, i in 0..=N.\n\n"
    for n in (64, 256):
        t = sin_table(n)
        out += f"pub const SIN{n}: [u64; {n + 1}] = [\n"
        out += "".join(f"    {hex(v)},\n" for v in t) + "];\n\n"
    out += "pub fn sin64_match(idx: felt252) -> u64 {\n    match idx {\n"
    t = sin_table(64)
    out += "".join(f"        {i} => {hex(v)},\n" for i, v in enumerate(t[:-1]))
    out += f"        _ => {hex(t[-1])},\n    }}\n}}\n"
    (ROOT / "algo" / "lut.cairo").write_text(out)


def gen_cordic():
    out = "// GENERATED by gen_algos.py - do not edit.\n"
    out += f"//! Unrolled CORDIC (rotation mode), {CORDIC_N} iterations. Shifts are signed divisions.\n\n"
    out += "/// sin(r) for r in [0, pi/2], Q32.32 raw.\n"
    out += f"pub fn sin_q(r: i64) -> i64 {{\n    let x: i64 = {hex(CORDIC_K)};\n    let y: i64 = 0;\n    let z = r;\n"
    for i in range(CORDIC_N):
        a = hex(CORDIC_ATAN[i])
        ys, xs = ("y", "x") if i == 0 else (f"y / {hex(1 << i)}", f"x / {hex(1 << i)}")
        out += "    let (x, y, z) = if z >= 0 {\n"
        out += f"        (x - {ys}, y + {xs}, z - {a})\n    }} else {{\n"
        out += f"        (x + {ys}, y - {xs}, z + {a})\n    }};\n"
    out += "    let _ = x;\n    let _ = z;\n    y\n}\n"
    (ROOT / "algo" / "cordic.cairo").write_text(out)


def lit(v):
    return f"BI64 {{ raw: {'-' if v < 0 else ''}{hex(abs(v))} }}"


def bench(group, variant, inputs, expected, expr):
    lines = ["#[test]", "#[inline(never)]", f"fn bench_{group}__{variant}() {{"]
    base = variant == "baseline"
    for name, v in inputs.items():
        lines.append(f"    let {'_' if base else ''}{name} = black_box({lit(v)});")
    lines.append(f"    let e = black_box({expected});")
    lines.append(f"    assert!({'e' if base else '(' + expr + ')'} == e);")
    lines.append("}\n")
    return "\n".join(lines)


def gen_tests():
    out = [
        "// GENERATED by gen_algos.py - do not edit.",
        "// Expected values are the bit-exact outputs of the Python integer models.",
        "use harness::black_box;",
        "use crate::algo::{sqrt, trig};",
        "use crate::bi64::BI64;",
        "",
    ]

    def group(name, cases, variants):
        """cases: {case: inputs}; variants: [(variant, expr, model)]."""
        first = True
        for case, inputs in cases.items():
            args = list(inputs.values())
            for variant, expr, model in variants:
                res = model(*args)
                e = "(" + ", ".join(lit(v) for v in res) + ")" if isinstance(res, tuple) else lit(res)
                if first:
                    out.append(bench(name, "baseline", inputs, e, ""))
                    first = False
                out.append(bench(name, f"{variant}_{case}", inputs, e, expr))

    a = raw(1234.5678)
    group("sqrt_algo", {"1234": {"a": a}, "0p02": {"a": raw(0.02)}}, [
        ("corelib", "sqrt::sqrt_corelib(a)", sqrt_fix),
        ("newton_loop", "sqrt::sqrt_newton_loop(a)", sqrt_fix),
        ("newton_seeded_unrolled", "sqrt::sqrt_newton_seeded(a)", sqrt_newton_seeded),
        ("via_inv_sqrt_newton", "sqrt::sqrt_via_inv_sqrt(a)", lambda a: mul(a, inv_sqrt_newton(a))),
    ])
    group("inv_sqrt_algo", {"1234": {"a": a}}, [
        ("sqrt_then_signed_div", "sqrt::inv_sqrt_div(a)", lambda a: div(ONE, sqrt_fix(a))),
        ("sqrt_then_const_udiv", "sqrt::inv_sqrt_direct(a)", lambda a: (1 << 64) // sqrt_fix(a)),
        ("newton_mul_only", "sqrt::inv_sqrt_newton(a)", inv_sqrt_newton),
    ])

    def length(x, y, z):
        return isqrt(x * x + y * y + z * z)

    v = {"x": raw(12.5), "y": raw(-340.25), "z": raw(7.125)}
    group("normalize3_algo", {"mix": v}, [
        ("length_3div", "sqrt::normalize3_div(x, y, z)",
         lambda x, y, z: tuple(div(c, length(x, y, z)) for c in (x, y, z))),
        ("length_inv32_3mul", "sqrt::normalize3_inv(x, y, z)",
         lambda x, y, z: tuple(mul(c, (1 << 64) // length(x, y, z)) for c in (x, y, z))),
        ("length_inv96_3mul", "sqrt::normalize3_wide_inv(x, y, z)",
         lambda x, y, z: tuple((c * ((1 << 96) // length(x, y, z))) >> 64 for c in (x, y, z))),
    ])
    group("sin_algo", {"0p5": {"x": raw(0.5)}, "m2p5": {"x": raw(-2.5)}}, [
        ("poly7_fused", "trig::sin_poly7(x)", sin_poly(SIN7)),
        ("poly9_fused", "trig::sin_poly9(x)", sin_poly(SIN9)),
        ("poly11_fused", "trig::sin_poly11(x)", sin_poly(SIN11)),
        ("poly9_unfused", "trig::sin_poly9_unfused(x)", sin_poly9_unfused),
        ("poly9_typed", "trig::sin_poly9_typed(x)", typed_model(TYPED["sin9"])),
        ("poly11_typed", "trig::sin_poly11_typed(x)", typed_model(TYPED["sin11"])),
        ("taylor_loop_cubit_style", "trig::sin_taylor_loop(x)", sin_taylor_loop),
        ("lut64_match", "trig::sin_lut64_match(x)", sin_lut(6)),
        ("lut64_array", "trig::sin_lut64_array(x)", sin_lut(6)),
        ("lut256_array", "trig::sin_lut256_array(x)", sin_lut(8)),
        ("cordic20_unrolled", "trig::sin_cordic20(x)", sin_cordic),
    ])
    group("cos_algo", {"m2p5": {"x": raw(-2.5)}}, [
        ("poly9_fused", "trig::cos_poly9(x)", sin_poly(SIN9, 1)),
        ("poly11_typed", "trig::cos_poly11_typed(x)", lambda x: typed_model(TYPED["sin11"])(x, 1)),
    ])
    group("atan2_algo", {
        "small": {"y": raw(0.25), "x": raw(1.75)},
        "swap_neg": {"y": raw(-1.25), "x": raw(-0.75)},
    }, [
        ("poly15", "trig::atan2_poly15(y, x)", atan2_poly15),
        ("reduced_poly9", "trig::atan2_reduced9(y, x)", atan2_reduced9),
        ("poly15_typed", "trig::atan2_poly15_typed(y, x)", typed_atan2(TYPED["atan15"])),
        ("poly19_typed", "trig::atan2_poly19_typed(y, x)", typed_atan2(TYPED["atan19"])),
    ])
    group("acos_algo", {"m0p3": {"x": raw(-0.3)}}, [
        ("sqrt_poly7", "trig::acos_poly7(x)", acos_poly7),
        ("via_atan2", "trig::acos_via_atan2(x)", acos_via_atan2),
        ("sqrt_poly9_typed", "trig::acos_poly9_typed(x)", typed_acos(TYPED["acos9"])),
    ])

    # Sanity: the models are close to the true functions at the benchmark points.
    assert abs(sin_poly(SIN9)(raw(-2.5)) / ONE - math.sin(-2.5)) < 1e-8
    assert abs(atan2_reduced9(raw(-1.25), raw(-0.75)) / ONE - math.atan2(-1.25, -0.75)) < 1e-8
    assert abs(acos_poly7(raw(-0.3)) / ONE - math.acos(-0.3)) < 1e-7

    # Accuracy tests against the TRUE values (not the models), with tolerance.
    out.append("fn close(got: BI64, want: i64, tol: i64) -> bool {")
    out.append("    let d = got.raw - want;")
    out.append("    d <= tol && d >= -tol")
    out.append("}\n")
    checks = []
    for x in (-6.0, -2.5, -0.001, 0.0, 0.5, 1.5707, 3.0, 100.0):
        checks.append(f"trig::sin_poly9(BI64 {{ raw: {raw(x)} }}), {raw(math.sin(x))}, 60")
        checks.append(f"trig::cos_poly9(BI64 {{ raw: {raw(x)} }}), {raw(math.cos(x))}, 60")
        checks.append(f"trig::sin_poly11_typed(BI64 {{ raw: {raw(x)} }}), {raw(math.sin(x))}, {4 if abs(x) < 10 else 60}")
        checks.append(f"trig::cos_poly11_typed(BI64 {{ raw: {raw(x)} }}), {raw(math.cos(x))}, {4 if abs(x) < 10 else 60}")
        checks.append(f"trig::sin_poly7(BI64 {{ raw: {raw(x)} }}), {raw(math.sin(x))}, 6000")
        checks.append(f"trig::sin_lut256_array(BI64 {{ raw: {raw(x)} }}), {raw(math.sin(x))}, 25000")
        checks.append(f"trig::sin_cordic20(BI64 {{ raw: {raw(x)} }}), {raw(math.sin(x))}, 25000")
    for y, x in ((0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (-3.0, 0.5), (0.002, -7.0), (-5.0, -5.0)):
        checks.append(
            f"trig::atan2_reduced9(BI64 {{ raw: {raw(y)} }}, BI64 {{ raw: {raw(x)} }}), {raw(math.atan2(y, x))}, 60"
        )
        checks.append(
            f"trig::atan2_poly19_typed(BI64 {{ raw: {raw(y)} }}, BI64 {{ raw: {raw(x)} }}), {raw(math.atan2(y, x))}, 12"
        )
        checks.append(
            f"trig::atan2_poly15(BI64 {{ raw: {raw(y)} }}, BI64 {{ raw: {raw(x)} }}), {raw(math.atan2(y, x))}, 400"
        )
    for x in (-1.0, -0.7, 0.0, 0.3, 0.999, 1.0):
        checks.append(f"trig::acos_poly9_typed(BI64 {{ raw: {raw(x)} }}), {raw(math.acos(x))}, 8")
        checks.append(f"trig::acos_poly7(BI64 {{ raw: {raw(x)} }}), {raw(math.acos(x))}, 200")
    for x in (2.0, 0.0001, 1234.5678, 1e6):
        checks.append(f"sqrt::sqrt_corelib(BI64 {{ raw: {raw(x)} }}), {raw(math.sqrt(raw(x) / ONE))}, 1")
        checks.append(f"sqrt::sqrt_newton_loop(BI64 {{ raw: {raw(x)} }}), {raw(math.sqrt(raw(x) / ONE))}, 1")
        checks.append(f"sqrt::sqrt_newton_seeded(BI64 {{ raw: {raw(x)} }}), {raw(math.sqrt(raw(x) / ONE))}, 1")
    out.append("#[test]\nfn test_accuracy_against_true_values() {")
    for chk in checks:
        out.append(f"    assert!(close({chk}));")
    out.append("}\n")
    (ROOT / "tests" / "algo.cairo").write_text("\n".join(out))


if __name__ == "__main__":
    TYPED = gen_typed()
    gen_seeds()
    gen_lut()
    gen_cordic()
    gen_tests()
    accuracy()
