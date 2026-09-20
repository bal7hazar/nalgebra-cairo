"""Interval-typed straight-line integer programs, emitted as Cairo `BoundedInt` code.

A `Kernel` records a sequence of exact integer operations (`mul`, `add`, `sub`, `div_rem`,
`upcast`). Every value carries the exact interval `[lo, hi]` of the operation over the operand
intervals, which is precisely what the Sierra `bounded_int_*` libfuncs require as result types:
the generator proves the bounds, the Cairo compiler re-checks them (a wrong bound is a compile
error), and no overflow check is left at run time.

The same recorded program can be evaluated on Python integers (`Kernel.run`) and serialised
(`Kernel.record`), which is how the bit-exact model of `tools/fixed_model` reproduces the
generated Cairo bit for bit without duplicating a single constant.
"""

PRIME = 2**251 + 17 * 2**192 + 1

# Constraints of the Sierra `bounded_int` libfuncs, checked as the program is recorded.
MAX_VALUE = PRIME // 2  # every intermediate must fit a felt252 without wrapping
MAX_DIV_OPERAND = 1 << 128


def hx(v):
    return ("-" if v < 0 else "") + hex(abs(v))


OPS = {
    "mul": lambda x, y: x * y,
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
}


class Val:
    """A typed value: a Cairo expression (variable name or literal) and its interval."""

    def __init__(self, expr, lo, hi, ty=None, nz=False):
        assert lo <= hi, (expr, lo, hi)
        self.expr, self.lo, self.hi = expr, lo, hi
        self._ty = ty
        self.nz = nz

    @property
    def ty(self):
        if self._ty is not None:
            return self._ty
        if self.lo == self.hi:
            return f"UnitInt<{hx(self.lo)}>"
        return f"BoundedInt<{hx(self.lo)}, {hx(self.hi)}>"


def const(c):
    return Val(hx(c), c, c)


class Module:
    """A generated Cairo file: helper impls and type aliases are de-duplicated across kernels."""

    def __init__(self, name):
        self.name = name
        self.seen = {}
        self.items = []
        self.kernels = []
        self.needs_is_zero = False

    def helper(self, prefix, trait, body):
        """Declares `impl .. of <trait> { <body> }` once per trait signature."""
        if trait in self.seen:
            assert self.seen[trait][1] == body, trait
            return None
        name = f"{prefix}H{len(self.seen)}"
        self.seen[trait] = (name, body)
        return f"impl {name} of {trait} {{\n{body}}}\n"

    def add(self, text):
        self.items.append(text)

    def render(self):
        return "\n".join(self.items)

    def record(self):
        return {k.name: k.record() for k in self.kernels}


class Kernel:
    def __init__(self, module, name, prefix, doc, inline="always", cfg=None):
        self.module, self.name, self.prefix, self.doc = module, name, prefix, doc
        self.inline, self.cfg = inline, cfg
        self.params, self.impls, self.lines, self.ops, self.aliases = [], [], [], [], []
        self.count = 0
        module.kernels.append(self)

    # --- inputs ------------------------------------------------------------------------------

    def input(self, name, lo, hi, ty=None, alias=None, alias_doc=None, nz=False):
        """A parameter. `ty` names a corelib type (`i64`); `alias` declares `pub type alias`.

        `nz=True` takes the value as a `NonZero<..>` (the caller proves it): the zero check and
        its panic message stay in hand-written code.
        """
        v = Val(name, lo, hi, ty, nz=nz)
        if alias:
            if alias not in self.module.seen:
                self.module.seen[alias] = (alias, v.ty)
                self.aliases.append(f"/// {alias_doc}\npub type {alias} = {v.ty};\n")
            assert self.module.seen[alias][1] == v.ty, (alias, self.module.seen[alias][1], v.ty)
        self.params.append((v, alias))
        return v

    def _fresh(self, lo, hi):
        self.count += 1
        return Val(f"v{self.count}", lo, hi)

    def _helper(self, trait, body):
        text = self.module.helper(self.prefix, trait, body)
        if text:
            self.impls.append(text)

    def _bin(self, op, trait, a, b, lo, hi):
        out = self._fresh(lo, hi)
        assert max(abs(lo), abs(hi)) < MAX_VALUE, (self.name, op, lo, hi)
        self._helper(f"{trait}<{a.ty}, {b.ty}>", f"    type Result = {out.ty};\n")
        self.lines.append(
            f"    let {out.expr}: {out.ty} = bounded_int::{op}::<{a.ty}, {b.ty}>({a.expr}, {b.expr});"
        )
        self.ops.append((op, out.expr, a.expr, b.expr))
        return out

    # --- operations --------------------------------------------------------------------------

    @staticmethod
    def _v(x):
        return x if isinstance(x, Val) else const(x)

    def mul(self, a, b):
        a, b = self._v(a), self._v(b)
        c = [a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi]
        return self._bin("mul", "MulHelper", a, b, min(c), max(c))

    def add(self, a, b):
        a, b = self._v(a), self._v(b)
        return self._bin("add", "AddHelper", a, b, a.lo + b.lo, a.hi + b.hi)

    def sub(self, a, b):
        a, b = self._v(a), self._v(b)
        return self._bin("sub", "SubHelper", a, b, a.lo - b.hi, a.hi - b.lo)

    def div_rem(self, a, b):
        """Unsigned `div_rem` by a positive constant or a `NonZero` value: `a >= 0`."""
        a, b = self._v(a), self._v(b)
        # A `NonZero` divisor keeps the interval of its type (which may contain 0): the value
        # itself cannot be 0, so the quotient bound uses 1 as the smallest divisor.
        b_lo = max(b.lo, 1) if b.nz else b.lo
        assert a.lo >= 0 and b_lo > 0, (self.name, a.lo, b.lo)
        q = self._fresh(a.lo // b.hi, a.hi // b_lo)
        r = self._fresh(0, b.hi - 1)
        # Constraints of the Sierra libfunc (`BoundedIntDivRemAlgorithm`).
        assert q.hi < MAX_DIV_OPERAND and b.hi <= MAX_DIV_OPERAND, (self.name, q.hi, b.hi)
        assert (
            b.hi << 128 < PRIME or (q.hi + 1) << 128 < PRIME or a.hi < MAX_DIV_OPERAND
        ), self.name
        self._helper(
            f"DivRemHelper<{a.ty}, {b.ty}>",
            f"    type DivT = {q.ty};\n    type RemT = {r.ty};\n",
        )
        self.lines.append(
            f"    let (<{q.expr}>, <{r.expr}>) = "
            f"bounded_int::div_rem::<{a.ty}, {b.ty}>({a.expr}, {b.expr});"
        )
        self.ops.append(("div_rem", (q.expr, r.expr), a.expr, b.expr))
        return q, r

    def nonzero(self, a):
        """`NonZero<T>` view of a value whose interval excludes zero.

        `bounded_int_is_zero` only accepts a type that *contains* zero, so the value is first
        widened to `[0, hi]`; the zero branch is then unreachable by construction.
        """
        assert a.lo > 0, (self.name, a.lo)
        self.module.needs_is_zero = True
        wide = self.upcast(a, 0, a.hi)
        out = self._fresh(0, a.hi)
        out.nz = True
        self.lines.append(
            f"    let {out.expr}: NonZero<{wide.ty}> = match bounded_int_is_zero({wide.expr}) {{\n"
            f"        IsZero::Zero => core::panic_with_felt252('simba: unreachable'),\n"
            f"        IsZero::NonZero(v) => v,\n    }};"
        )
        self.ops.append(("upcast", out.expr, wide.expr, None))
        return out

    def shr_round(self, a, bits):
        """`floor((a + 2^(bits-1)) / 2^bits)`: round to nearest, ties up. `a + 2^(bits-1) >= 0`."""
        q, _ = self.div_rem(self.add(a, 1 << (bits - 1)), 1 << bits)
        return q

    def shr_floor(self, a, bits):
        """`floor(a / 2^bits)` for an `a` of any sign (biased unsigned `div_rem`)."""
        if a.lo >= 0:
            return self.div_rem(a, 1 << bits)[0]
        k = -(a.lo >> bits)  # smallest offset (in quotient units) making the numerator >= 0
        q, _ = self.div_rem(self.add(a, k << bits), 1 << bits)
        return self.sub(q, k)

    def narrow(self, a, lo, hi):
        """Range check down to `[lo, hi]`, for a bound the generator knows but the libfunc does
        not (a `div_rem` by a `NonZero` type whose interval still contains 0 must declare the
        widest quotient). The check cannot fire; it costs one range check."""
        assert a.lo <= lo and hi <= a.hi and hi - lo < 1 << 128, (self.name, lo, hi)
        assert a.hi - a.lo < 1 << 128, (self.name, "downcast source too wide")
        out = self._fresh(lo, hi)
        self.lines.append(
            f"    let {out.expr}: {out.ty} = downcast({a.expr}).expect('simba: unreachable');"
        )
        self.ops.append(("upcast", out.expr, a.expr, None))
        return out

    def upcast(self, a, lo, hi, ty=None):
        assert lo <= a.lo and a.hi <= hi
        out = self._fresh(lo, hi)
        out._ty = ty
        self.lines.append(f"    let {out.expr}: {out.ty} = upcast({a.expr});")
        self.ops.append(("upcast", out.expr, a.expr, None))
        return out

    def horner(self, v, program):
        """Lazy-rescale Horner evaluation of a `program` (see `polygen.program`).

        `("lead", c)` loads the leading coefficient, `("step", c)` computes `acc = v * acc + c`
        exactly (the scale of `acc` grows by the scale of `v`, `c` is given at the new scale) and
        `("shr", bits)` is `acc = floor(acc / 2^bits)`: one `div_rem` for several steps.
        """
        acc = None
        for op, arg in program:
            if op == "lead":
                acc = const(arg)
            elif op == "step":
                acc = self.add(self.mul(v, acc), arg) if arg else self.mul(v, acc)
            else:
                acc = self.shr_floor(acc, arg)
        return acc

    # --- output --------------------------------------------------------------------------------

    def finish(self, outs, out_names=None, out_docs=None):
        """Emits the function returning `outs` (a Val or a tuple); `out_names` are type aliases."""
        outs = outs if isinstance(outs, (tuple, list)) else (outs,)
        self.outs = [o.expr for o in outs]
        names = out_names or [None] * len(outs)
        docs = out_docs or [None] * len(outs)
        tys = []
        for o, n, d in zip(outs, names, docs):
            if n and n not in self.module.seen:
                self.module.seen[n] = (n, o.ty)
                self.aliases.append(f"/// {d or n}\npub type {n} = {o.ty};\n")
            if n:
                assert self.module.seen[n][1] == o.ty, (n, self.module.seen[n][1], o.ty)
            tys.append(n or o.ty)
        ret = tys[0] if len(tys) == 1 else "(" + ", ".join(tys) + ")"
        val = self.outs[0] if len(tys) == 1 else "(" + ", ".join(self.outs) + ")"
        sig = ", ".join(
            f"{p.expr}: " + (f"NonZero<{alias or p.ty}>" if p.nz else (alias or p.ty))
            for p, alias in self.params
        )
        # Unused halves of a `div_rem` are bound to `_`.
        used = set(self.outs)
        for _op, _out, a, b in self.ops:
            used.update((a, b))
        for op, out, _a, _b in self.ops:
            if op == "div_rem":
                for name in out:
                    self.lines = [
                        ln.replace(f"<{name}>", name if name in used else "_") for ln in self.lines
                    ]
        text = "".join(self.impls) + "".join(self.aliases)
        text += "".join(f"/// {line}".rstrip() + "\n" for line in self.doc.split("\n"))
        if self.cfg:
            text += f"#[cfg({self.cfg})]\n"
        text += f"#[inline({self.inline})]\n" if self.inline else ""
        text += (
            f"pub fn {self.name}({sig}) -> {ret} {{\n" + "\n".join(self.lines) + f"\n    {val}\n}}\n"
        )
        self.module.add(text)
        return self

    # --- evaluation and serialisation ----------------------------------------------------------

    def record(self):
        """The program as plain data, for the Python model (`fixed_model.poly_ops`)."""
        return {
            "params": [[p.expr, p.lo, p.hi] for p, _ in self.params],
            "ops": [list(op) if not isinstance(op[1], tuple) else [op[0], list(op[1]), op[2], op[3]]
                    for op in self.ops],
            "outs": list(self.outs),
        }

    def run(self, *args):
        """Evaluates the recorded program on Python integers, checking the input intervals."""
        return run_record(self.record(), *args)


def run_record(rec, *args):
    """Runs a serialised kernel (see `Kernel.record`) on Python integers."""
    env = {}
    for (name, lo, hi), a in zip(rec["params"], args):
        assert lo <= a <= hi, (name, a, lo, hi)
        env[name] = a

    def get(e):
        return env[e] if e in env else int(e, 16)

    for op, out, a, b in rec["ops"]:
        if op == "upcast":
            env[out] = get(a)
        elif op == "div_rem":
            env[out[0]], env[out[1]] = divmod(get(a), get(b))
        else:
            env[out] = OPS[op](get(a), get(b))
    res = [env[o] for o in rec["outs"]]
    return res[0] if len(res) == 1 else tuple(res)
