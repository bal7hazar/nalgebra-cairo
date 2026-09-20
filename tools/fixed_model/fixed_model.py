#!/usr/bin/env python3
"""Bit-exact integer model of `simba::fixed` (Q32.32 on a signed 64-bit integer).

Every function takes and returns *raw* integers (`value * 2^32`) and mirrors the numeric
specification of the Cairo implementation:

* rounding is floor (toward -inf), applied once per output scalar;
* results that do not fit `i64` raise `Overflow` (Cairo panics with `'simba: overflow'`);
* `DivisionByZero` <-> `'simba: division by zero'`, `SqrtNegative` <-> `'simba: sqrt of negative'`;
* intermediates are unbounded Python integers, like the wide accumulators of the Cairo kernels.

The model is the reference used by `gen_vectors.py` to emit the expectations of
`crates/simba/src/fixed/tests_generated.cairo`. Run this file to self-check the model.
"""

from math import isqrt

FRAC_BITS = 32
ONE = 1 << FRAC_BITS
HALF = ONE >> 1
MIN = -(1 << 63)
MAX = (1 << 63) - 1
I32_MIN = -(1 << 31)
I32_MAX = (1 << 31) - 1


class FixedError(ArithmeticError):
    """Base class: the Cairo implementation panics."""

    message = "simba: error"


class Overflow(FixedError):
    message = "simba: overflow"


class DivisionByZero(FixedError):
    message = "simba: division by zero"


class SqrtNegative(FixedError):
    message = "simba: sqrt of negative"


class Domain(FixedError):
    message = "simba: out of domain"


def check(raw):
    """The single overflow check performed at the end of every kernel."""
    if not MIN <= raw <= MAX:
        raise Overflow(raw)
    return raw


def rescale(wide):
    """floor(wide / 2^32), checked. `wide` is a sum of unscaled products (scale 2^64)."""
    return check(wide >> FRAC_BITS)  # Python's >> is an arithmetic shift: floor.


# --- operators ------------------------------------------------------------------------------------


def add(a, b):
    return check(a + b)


def sub(a, b):
    return check(a - b)


def neg(a):
    return check(-a)


def mul(a, b):
    return rescale(a * b)


def div(a, b):
    """Floor division: floor(a * 2^32 / b) in exact arithmetic."""
    if b == 0:
        raise DivisionByZero()
    return check((a << FRAC_BITS) // b)  # Python's // is floor division.


def rem(a, b):
    """Floored modulo: a - b * floor(a / b); zero or the sign of the divisor. Never overflows."""
    if b == 0:
        raise DivisionByZero()
    return a % b  # Python's % is the floored modulo.


# --- conversions ----------------------------------------------------------------------------------


def from_int(v):
    if not I32_MIN <= v <= I32_MAX:
        raise Overflow(v)
    return v << FRAC_BITS


def to_int(a):
    """Floor to integer (always fits i32)."""
    return a >> FRAC_BITS


def from_ratio(num, den):
    """floor(num * 2^32 / den) for integers num, den (both i64)."""
    return div(num, den)


# --- helpers --------------------------------------------------------------------------------------


def abs_(a):
    return check(-a) if a < 0 else a


def signum(a):
    return ONE if a > 0 else (-ONE if a < 0 else 0)


def min_(a, b):
    return a if a < b else b


def max_(a, b):
    return a if a > b else b


def clamp(x, lo, hi):
    """simba's `clamp`: no check that lo <= hi."""
    if x > lo:
        return x if x < hi else hi
    return lo


def floor(a):
    return (a >> FRAC_BITS) << FRAC_BITS


def ceil(a):
    return check(-((-a >> FRAC_BITS) << FRAC_BITS))


def trunc(a):
    return floor(a) if a >= 0 else -floor(-a)


def fract(a):
    """a - trunc(a): same sign as `a` (Rust semantics)."""
    return a - trunc(a)


def round_(a):
    """Round half away from zero (Rust semantics)."""
    if a >= 0:
        return check(floor(a + HALF))
    return -floor(-a + HALF)


def round_half_up(a):
    """Alternative (not exported): floor(a + 1/2)."""
    return check(floor(a + HALF))


def recip(a):
    """floor(2^64 / a)."""
    if a == 0:
        raise DivisionByZero()
    return check((1 << (2 * FRAC_BITS)) // a)


def is_negative(a):
    return a < 0


def is_positive(a):
    return a > 0


def abs_diff_eq(a, b, ulps):
    return abs(a - b) <= ulps


# --- fused kernels (one floor per output) -----------------------------------------------------------


def sqr(a):
    return rescale(a * a)


def sum_prod2(a0, b0, a1, b1):
    return rescale(a0 * b0 + a1 * b1)


def sum_prod3(a0, b0, a1, b1, a2, b2):
    return rescale(a0 * b0 + a1 * b1 + a2 * b2)


def sum_prod4(a0, b0, a1, b1, a2, b2, a3, b3):
    return rescale(a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3)


def diff_prod(a, b, c, d):
    return rescale(a * b - c * d)


def mul_add(a, b, c):
    return rescale(a * b + (c << FRAC_BITS))


def mul_sub(a, b, c):
    return rescale(a * b - (c << FRAC_BITS))


def lerp(a, b, t):
    """a + (b - a) * t: exact difference and product, one floor."""
    return rescale((b - a) * t + (a << FRAC_BITS))


def norm_squared2(x, y):
    return rescale(x * x + y * y)


def norm_squared3(x, y, z):
    return rescale(x * x + y * y + z * z)


def norm_squared4(x, y, z, w):
    return rescale(x * x + y * y + z * z + w * w)


def wide_sqrt(wide):
    """floor(sqrt(wide)) for an unscaled (2^64 scale) accumulator: the result has scale 2^32."""
    if wide < 0:
        raise SqrtNegative()
    return check(isqrt(wide))


def norm2(x, y):
    return wide_sqrt(x * x + y * y)


def norm3(x, y, z):
    return wide_sqrt(x * x + y * y + z * z)


def norm4(x, y, z, w):
    return wide_sqrt(x * x + y * y + z * z + w * w)


def sqrt(a):
    """floor(sqrt(a * 2^32)): exactly floored."""
    if a < 0:
        raise SqrtNegative()
    return isqrt(a << FRAC_BITS)


def inv_sqrt(a):
    """floor(2^48 / sqrt(a)) = isqrt(floor(2^96 / a)): exactly floored."""
    if a < 0:
        raise SqrtNegative()
    if a == 0:
        raise DivisionByZero()
    return isqrt((1 << (3 * FRAC_BITS)) // a)


def inv_sqrt_two_step(a):
    """Alternative (not exported): floor(2^64 / floor(sqrt(a * 2^32))), two roundings."""
    if a < 0:
        raise SqrtNegative()
    s = isqrt(a << FRAC_BITS)
    if s == 0:
        raise DivisionByZero()
    return check((1 << (2 * FRAC_BITS)) // s)


# --- wide accumulator -------------------------------------------------------------------------------


class Wide:
    """Unscaled accumulator (scale 2^64). Grows by at most 2^126 per operation."""

    def __init__(self, value=0):
        self.value = value

    def add_prod(self, a, b):
        return Wide(self.value + a * b)

    def sub_prod(self, a, b):
        return Wide(self.value - a * b)

    def add(self, c):
        return Wide(self.value + (c << FRAC_BITS))

    def sub(self, c):
        return Wide(self.value - (c << FRAC_BITS))

    def rescale(self):
        return rescale(self.value)

    def sqrt(self):
        return wide_sqrt(self.value)


def sum_prod_n(pairs):
    """Reference for any accumulation of products."""
    return rescale(sum(a * b for a, b in pairs))


# --- transcendental functions -------------------------------------------------------------------
# The polynomial halves are the GENERATED programs of `simba::fixed::kernels::poly`, replayed
# operation by operation (`poly_ops.run`), so the model cannot drift from the Cairo code. What is
# written here is the hand-written half, `simba::fixed::kernels::transcendental`: octant folding,
# signs, reflections and domain checks.

from poly_ops import A, run as _poly  # noqa: E402

FRAC_PI_2 = 6746518852  # floor(pi/2 * 2^32), the constant of `fixed::types`
PI_RAW = 2 * FRAC_PI_2  # floor(pi * 2^32), exactly twice FRAC_PI_2


def _split_octant(n):
    """`(high, q, low)` = the three symmetry bits of an octant index."""
    high, m = divmod(n, 4)
    q, low = divmod(m, 2)
    return bool(high), bool(q), bool(low)


def _octant_arg(low, r):
    return _poly("octant_complement", r) if low else r


def _sin_of_octant(n, r):
    high, q, low = _split_octant(n)
    arg = _octant_arg(low, r)
    mag = _poly("cos_octant", arg) if q != low else _poly("sin_octant", arg)
    return -mag if high else mag


def sin(x):
    """sin(x), x in Q32.32 radians. Total."""
    return _sin_of_octant(*_poly("reduce_octant", x))


def cos(x):
    """cos(x) = sin(x + pi/2): the same reduction, octant shifted by two. Total."""
    return _sin_of_octant(*_poly("reduce_octant_cos", x))


def _fold(kernel, x):
    n, r = _poly("reduce_octant", x)
    high, q, low = _split_octant(n)
    a, b = _poly(kernel, _octant_arg(low, r))
    s0, c0 = (b, a) if low else (a, b)
    sm, cm = (c0, s0) if q else (s0, c0)
    return sm, cm, high, q


def sin_cos(x):
    """(sin x, cos x): one reduction, one shared r^2. Bit for bit (sin(x), cos(x))."""
    sm, cm, high, q = _fold("sin_cos_octant", x)
    return (-sm if high else sm), (-cm if q != high else cm)


def tan(x):
    """tan(x) from a sine and a cosine with 48 fractional bits."""
    sm, cm, _high, q = _fold("sin_cos_octant_wide", x)
    if cm == 0:
        raise DivisionByZero()
    m = check(_poly("tan_ratio", sm, cm))
    return -m if q else m


def atan2(y, x):
    """atan2(y, x) in (-pi, pi]. atan2(0, 0) = 0. Total."""
    ax, ay = abs(x), abs(y)
    swap = ay > ax
    num, den = (ax, ay) if swap else (ay, ax)
    if den == 0:
        return 0
    a = _poly("atan_unit", (num << A) // den)
    if swap:
        a = FRAC_PI_2 - a
    if x < 0:
        a = PI_RAW - a
    return -a if y < 0 else a


def atan(x):
    """atan(x) in (-pi/2, pi/2). Bit for bit atan2(x, ONE). Total."""
    ax = abs(x)
    swap = ax > ONE
    t = (1 << (A + FRAC_BITS)) // ax if swap else ax << (A - FRAC_BITS)
    a = _poly("atan_unit", t)
    if swap:
        a = FRAC_PI_2 - a
    return -a if x < 0 else a


def _asin_halves(ax):
    """(asin|x|, acos|x|) for 0 <= ax <= 1, whichever branch applies."""
    if ax <= HALF:
        a = _poly("asin_unit", ax << (A - FRAC_BITS))
        return a, FRAC_PI_2 - a
    # s = sqrt((1 - x) / 2) at scale 2^A: sqrt(d * 2^(2A - 33)) with d = ONE - ax.
    d = _poly("asin_double", isqrt((ONE - ax) << (2 * A - FRAC_BITS - 1)))
    return FRAC_PI_2 - d, d


def asin(x):
    """asin(x) in [-pi/2, pi/2]. Raises `Domain` for |x| > 1."""
    ax = abs(x)
    if ax > ONE:
        raise Domain(x)
    m, _ = _asin_halves(ax)
    return -m if x < 0 else m


def acos(x):
    """acos(x) in [0, pi]. Raises `Domain` for |x| > 1."""
    ax = abs(x)
    if ax > ONE:
        raise Domain(x)
    _, m = _asin_halves(ax)
    return PI_RAW - m if x < 0 else m


# --- constants --------------------------------------------------------------------------------------


def constants():
    """Floor-rounded Q32.32 constants (name -> raw), computed with 80 significant digits."""
    import mpmath

    mpmath.mp.dps = 80
    real = {
        "PI": mpmath.pi,
        "TAU": 2 * mpmath.pi,
        "FRAC_PI_2": mpmath.pi / 2,
        "FRAC_PI_3": mpmath.pi / 3,
        "FRAC_PI_4": mpmath.pi / 4,
        "FRAC_PI_6": mpmath.pi / 6,
        "FRAC_1_PI": 1 / mpmath.pi,
        "E": mpmath.e,
        "LN_2": mpmath.log(2),
        "LN_10": mpmath.log(10),
        "SQRT_2": mpmath.sqrt(2),
        "FRAC_1_SQRT_2": 1 / mpmath.sqrt(2),
    }
    out = {
        "ZERO": 0,
        "ONE": ONE,
        "NEG_ONE": -ONE,
        "TWO": 2 * ONE,
        "HALF": HALF,
        "EPSILON": 1,
        "MIN": MIN,
        "MAX": MAX,
    }
    for name, value in real.items():
        out[name] = int(mpmath.floor(value * ONE))
    return out


def self_check_transcendental(dense=4000):
    """Checks the transcendental model against mpmath and pins the exact values.

    Called by `tools/polygen/polygen.py` (default mode and `--check`): a divergence between the
    generated Cairo kernels and this model is a hard error.
    """
    import mpmath as mpm
    import random

    mpm.mp.prec = 200
    rng = random.Random(0x51_4D_42_41)

    # Exact values that the documentation guarantees.
    assert sin(0) == 0 and cos(0) == ONE and tan(0) == 0
    assert sin_cos(0) == (0, ONE)
    assert asin(0) == 0 and atan(0) == 0 and atan2(0, 0) == 0
    assert acos(0) == FRAC_PI_2 and acos(ONE) == 0 and acos(-ONE) == PI_RAW
    assert asin(ONE) == FRAC_PI_2 and asin(-ONE) == -FRAC_PI_2
    assert atan2(0, ONE) == 0 and atan2(0, -ONE) == PI_RAW
    assert atan2(ONE, 0) == FRAC_PI_2 and atan2(-ONE, 0) == -FRAC_PI_2
    assert atan2(-1, -ONE) == -PI_RAW + 1 or atan2(-1, -ONE) < 0

    worst = {}

    def note(name, got, want):
        worst[name] = max(worst.get(name, 0), abs(got - want * ONE))

    for i in range(dense + 1):
        x = int(mpm.nint((mpm.mpf(-7) + mpm.mpf(14) * i / dense) * ONE))
        note("sin", sin(x), mpm.sin(mpm.mpf(x) / ONE))
        note("cos", cos(x), mpm.cos(mpm.mpf(x) / ONE))
        s, c = sin_cos(x)
        assert (s, c) == (sin(x), cos(x)), x
        note("atan", atan(x), mpm.atan(mpm.mpf(x) / ONE))
        assert atan(x) == atan2(x, ONE), x
        u = int(mpm.nint((mpm.mpf(-1) + 2 * mpm.mpf(i) / dense) * ONE))
        note("asin", asin(u), mpm.asin(mpm.mpf(u) / ONE))
        note("acos", acos(u), mpm.acos(mpm.mpf(u) / ONE))
        assert sin(-x) == -sin(x) and cos(-x) == cos(x), x
        assert asin(-u) == -asin(u) and atan(-x) == -atan(x), (x, u)

    for _ in range(dense):
        y, x = rng.randrange(MIN, MAX), rng.randrange(MIN, MAX)
        note("atan2", atan2(y, x), mpm.atan2(mpm.mpf(y) / ONE, mpm.mpf(x) / ONE))
        x = rng.randrange(-(1 << 63), 1 << 63)
        note("sin_big", sin(x), mpm.sin(mpm.mpf(x) / ONE))

    limits = {"sin": 3, "cos": 3, "atan": 3, "asin": 3, "acos": 3, "atan2": 3, "sin_big": 4}
    for name, limit in limits.items():
        assert worst[name] <= limit, (name, float(worst[name]), limit)
    return {k: float(v) for k, v in worst.items()}


# --- self-check -------------------------------------------------------------------------------------


def _self_check():
    from fractions import Fraction as F
    import math
    import random

    rng = random.Random(1)

    def rnd():
        bits = rng.randrange(0, 64)
        v = rng.getrandbits(bits) if bits else 0
        return max(MIN, min(MAX, -v if rng.random() < 0.5 else v))

    def fl(x):
        return math.floor(x)

    for _ in range(20000):
        a, b, c, d = rnd(), rnd(), rnd(), rnd()
        fa, fb, fc, fd = F(a, ONE), F(b, ONE), F(c, ONE), F(d, ONE)
        for got, want in (
            (lambda: mul(a, b), fl(fa * fb * ONE)),
            (lambda: div(a, b), fl(fa / fb * ONE) if b else None),
            (lambda: rem(a, b), (fa - fb * fl(fa / fb)) * ONE if b else None),
            (lambda: recip(a), fl(1 / fa * ONE) if a else None),
            (lambda: sum_prod2(a, b, c, d), fl((fa * fb + fc * fd) * ONE)),
            (lambda: diff_prod(a, b, c, d), fl((fa * fb - fc * fd) * ONE)),
            (lambda: mul_add(a, b, c), fl((fa * fb + fc) * ONE)),
            (lambda: mul_sub(a, b, c), fl((fa * fb - fc) * ONE)),
            (lambda: lerp(a, b, c), fl((fa + (fb - fa) * fc) * ONE)),
            (lambda: floor(a), fl(fa) * ONE),
            (lambda: ceil(a), math.ceil(fa) * ONE),
            (lambda: trunc(a), math.trunc(fa) * ONE),
            (lambda: fract(a), (fa - math.trunc(fa)) * ONE),
            (lambda: round_(a), int(math.copysign(fl(abs(fa) + F(1, 2)), a)) * ONE),
            (lambda: to_int(a), fl(fa)),
        ):
            if want is None:
                continue
            try:
                value = got()
            except Overflow:
                assert not MIN <= want <= MAX, (a, b, c, d, want)
                continue
            assert value == want, (a, b, c, d, value, want)
        if a > 0:
            s = sqrt(a)  # s^2 <= a * 2^32 < (s + 1)^2
            assert s * s <= a * ONE < (s + 1) * (s + 1)
            r = inv_sqrt(a)  # r <= 2^48 / sqrt(a) < r + 1  <=>  r^2 * a <= 2^96 < (r + 1)^2 * a
            assert r * r * a <= 1 << 96 < (r + 1) * (r + 1) * a
        try:
            n = norm2(a, b)
            assert n * n <= a * a + b * b < (n + 1) * (n + 1)
        except Overflow:
            assert isqrt(a * a + b * b) > MAX
        # Identities used by the documentation.
        try:
            assert mul_add(a, b, c) == add(mul(a, b), c)
        except Overflow:
            pass
        if b:
            try:
                q = div(a, b)
                # a/b - 1ulp < q <= a/b
                assert F(q, ONE) <= fa / fb < F(q + 1, ONE)
            except Overflow:
                pass
            assert rem(a, ONE) == a - floor(a)
    print("fixed_model: self-check passed")
    worst = self_check_transcendental()
    print("  transcendental worst error (ulp): " + ", ".join(
        f"{k} {v:.2f}" for k, v in sorted(worst.items())
    ))
    for name, raw in constants().items():
        print(f"  {name:<14} {raw:>22}  {raw / ONE!r}")


if __name__ == "__main__":
    _self_check()
