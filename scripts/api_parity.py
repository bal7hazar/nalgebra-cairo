#!/usr/bin/env python3
"""Generate the nalgebra-rs 0.35.0 versus nalgebra-cairo public API inventory (docs/API_PARITY.md).

Dependency free, like glam-cairo's scripts/api_parity.py which it mirrors.  It is not a Rust or
Cairo parser: it masks comments and strings, finds balanced blocks, expands the handful of
`macro_rules!` shapes nalgebra-rs uses to declare its API (positional binding of `$ident`
fragments, geometry operator macros, swizzles) and recognizes declarations.  The Rust inventory
is embedded in the generated Markdown, so the normal run and `--check` need no Rust checkout:

    python3 scripts/api_parity.py                  # regenerate docs/API_PARITY.md
    python3 scripts/api_parity.py --check          # fail if the committed doc is stale
    python3 scripts/api_parity.py --refresh --nalgebra-rs ../refs/nalgebra-0.35.0 \
        --simba ~/.cargo/registry/src/<index>/simba-0.10.2

Model.  An upstream item is `(owner, kind, name)`:
- owner: the upstream type family the item is reachable from (`Matrix` for any `Matrix<T, R, C, S>`,
  `SquareMatrix`, `Vector`, `Matrix3` for `Matrix<T, U3, U3, S>`-only impls, `Isometry`,
  `UnitQuaternion`, `LU`, ...), or a module for free functions (`nalgebra`, `nalgebra::linalg`...);
- kind: `type` (struct / enum / type alias users name), `trait`, `method` (inherent `pub fn`,
  associated functions included), `const`, `impl` (operator / conversion / comparison trait impl,
  normalized to type families: `Mul<Matrix>`, `From<[T; N]>`, `Into<[T; N]>`), `function`.
Reference (`&a * &b`) and owned operator variants are folded: Cairo values are `Copy`.

Statuses: `ported` (same name, or a documented rename), `partial` (ported on some but not all of
the Cairo types of one family, e.g. on `Vector3` but not `Vector2`), `missing`, `excluded` (one
reason of the closed list `EXCLUSIONS`, applied by the explicit `EXCLUDE` rules).  An unmatched
item without a rule is `missing`: when in doubt, missing (the owner decides exclusions).
"""

from __future__ import annotations

import argparse
import difflib
import functools
import itertools
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

VERSION = "0.35.0"
SIMBA_VERSION = "0.10.2"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "API_PARITY.md"
INVENTORY_START = "<!-- api-parity-rust-inventory\n"
INVENTORY_END = "\napi-parity-rust-inventory -->"

# --------------------------------------------------------------------------------------------
# Text utilities
# --------------------------------------------------------------------------------------------


def mask_comments(text: str) -> str:
    """Blank out comments, string and char literals while preserving offsets and newlines."""
    out = list(text)
    i, n = 0, len(text)
    state = "code"
    depth = 0
    while i < n:
        pair = text[i:i + 2]
        if state == "code":
            if pair == "//":
                state = "line"
                continue
            if pair == "/*":
                state, depth = "block", 1
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if text[i] == '"':
                state = "string"
                out[i] = " "
                i += 1
                continue
            # Char literals ('a', '\n', '\''); lifetimes ('a) have no closing quote.
            m = re.match(r"'(?:\\.|[^\\'\n])'", text[i:i + 4]) if text[i] == "'" else None
            if m:
                for k in range(i, i + len(m.group(0))):
                    out[k] = " "
                i += len(m.group(0))
                continue
            i += 1
            continue
        if state == "line":
            if text[i] == "\n":
                state = "code"
            else:
                out[i] = " "
            i += 1
            continue
        if state == "block":
            if pair == "/*":
                depth += 1
                out[i] = out[i + 1] = " "
                i += 2
                continue
            if pair == "*/":
                depth -= 1
                out[i] = out[i + 1] = " "
                i += 2
                if depth == 0:
                    state = "code"
                continue
            if text[i] != "\n":
                out[i] = " "
            i += 1
            continue
        # string
        if text[i] == "\\" and i + 1 < n:
            out[i] = " "
            if text[i + 1] != "\n":
                out[i + 1] = " "
            i += 2
            continue
        if text[i] == '"':
            state = "code"
        if text[i] != "\n":
            out[i] = " "
        i += 1
    return "".join(out)


OPEN = {"{": "}", "(": ")", "[": "]"}


def closing(text: str, opening: int) -> int:
    """Index of the bracket closing the one at `opening` ({, ( or [)."""
    stack = []
    for i in range(opening, len(text)):
        c = text[i]
        if c in OPEN:
            stack.append(OPEN[c])
        elif c in ")}]":
            if not stack or stack[-1] != c:
                raise ValueError(f"unbalanced {c!r} at byte {i}")
            stack.pop()
            if not stack:
                return i
    raise ValueError(f"unclosed bracket at byte {opening}")


def split_top(value: str, seps: str = ",") -> list[str]:
    """Split on separators outside <>, (), [] and {}."""
    parts, start, depth = [], 0, 0
    i = 0
    while i < len(value):
        c = value[i]
        if c in "<([{":
            depth += 1
        elif c in ">)]}":
            if not (c == ">" and i > 0 and value[i - 1] in "-="):
                depth -= 1
        elif depth == 0 and c in seps:
            parts.append(value[start:i].strip())
            start = i + 1
        i += 1
    parts.append(value[start:].strip())
    return parts


def skip_generics(value: str, start: int) -> int:
    """If value[start] is '<', return the index after the matching '>' (arrows ignored)."""
    if start >= len(value) or value[start] != "<":
        return start
    depth = 0
    for i in range(start, len(value)):
        c = value[i]
        if c == "<":
            depth += 1
        elif c == ">" and value[i - 1] not in "-=":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(value)


def squash(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


# --------------------------------------------------------------------------------------------
# Items
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True, order=True)
class Item:
    owner: str
    kind: str
    name: str
    module: str = ""
    source: str = ""
    flags: str = ""

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.owner, self.kind, self.name)

    def as_json(self) -> dict[str, str]:
        data = {"owner": self.owner, "kind": self.kind, "name": self.name,
                "module": self.module, "source": self.source}
        if self.flags:
            data["flags"] = self.flags
        return data


def unique_items(items) -> list[Item]:
    """Deduplicate by API identity, keeping the lexicographically first module/source."""
    result: dict[tuple[str, str, str], Item] = {}
    for item in sorted(items, key=lambda it: (it.key, it.module, it.source)):
        result.setdefault(item.key, item)
    return sorted(result.values())


# --------------------------------------------------------------------------------------------
# Rust side: type families
# --------------------------------------------------------------------------------------------

MATRIX_BASES = {"Matrix", "OMatrix", "SMatrix", "MatrixN", "MatrixMN", "SquareMatrix",
                "UninitMatrix"}
VECTOR_BASES = {"Vector", "OVector", "SVector", "VectorN"}
ONE = {"U1", "Const<1>", "1"}
DYN = {"Dyn", "Dynamic"}
FIXED_DIM = re.compile(r"^(?:U([1-6])|Const<([1-6])>|([1-6]))$")


def dim_value(value: str) -> str:
    """'3' for U3 / Const<3> / 3, 'X' for Dyn, '' for a generic dimension."""
    value = value.strip()
    if value in DYN:
        return "X"
    m = FIXED_DIM.match(value)
    if m:
        return next(g for g in m.groups() if g)
    return ""


def type_head(value: str) -> tuple[str, list[str]]:
    """('Matrix', ['T', 'R', 'C', 'S']) for 'Matrix<T, R, C, S>'."""
    value = value.strip().lstrip("&").strip()
    value = re.sub(r"^(?:mut\s+|dyn\s+)", "", value)
    value = re.sub(r"'\s*[A-Za-z_]\w*\s*,?\s*", "", value)
    value = value.strip().lstrip("&").strip()
    value = re.sub(r"\b(?:crate|super|self|base|geometry|linalg|alias|core|std)::", "", value)
    value = re.sub(r"\b(?:na|nalgebra)::", "", value)
    m = re.match(r"([A-Za-z_$][\w$]*)\s*(<.*>)?\s*$", value, re.S)
    if not m:
        return value, []
    args = split_top(m.group(2)[1:-1]) if m.group(2) else []
    return m.group(1), [a for a in args if a]


def matrix_owner(base: str, args: list[str]) -> str:
    """Upstream family of a Matrix-like self type."""
    if base in ("DMatrix", "DVector", "RowDVector"):
        return base
    if base in ("Matrix1", "Matrix2", "Matrix3", "Matrix4", "Matrix5", "Matrix6",
                "Vector1", "Vector2", "Vector3", "Vector4", "Vector5", "Vector6"):
        return base
    dims = args[1:3] if base in MATRIX_BASES and base != "SquareMatrix" else args[1:2]
    if base == "SquareMatrix":
        d = dim_value(args[1]) if len(args) > 1 else ""
        return f"Matrix{d}" if d and d != "X" else ("DMatrix" if d == "X" else "SquareMatrix")
    if base in VECTOR_BASES:
        d = dim_value(args[1]) if len(args) > 1 else ""
        return "DVector" if d == "X" else (f"Vector{d}" if d else "Vector")
    if len(dims) < 2:
        return "Matrix"
    r, c = (d.strip() for d in dims)
    dr, dc = dim_value(r), dim_value(c)
    if "X" in (dr, dc):
        if dc == "1":
            return "DVector"
        if dr == "1":
            return "RowDVector"
        return "DMatrix"
    if dc == "1" and dr:
        return f"Vector{dr}"
    if dc == "1" or c in ONE:
        return "Vector"
    if dr and dc:
        return f"Matrix{dr}" if dr == dc else f"Matrix{dr}x{dc}"
    if r == c:
        return "SquareMatrix"
    return "Matrix"


def rust_owner(self_type: str) -> str:
    """Normalize an impl self type to its upstream owner family."""
    base, args = type_head(self_type)
    base = base.lstrip("$")
    if base in MATRIX_BASES or base in VECTOR_BASES or base in (
            "DMatrix", "DVector", "RowDVector") or re.fullmatch(r"(?:Matrix|Vector)[1-6]", base):
        return matrix_owner(base, args)
    if base == "Unit" and args:
        inner, inner_args = type_head(args[0])
        if inner == "Quaternion":
            return "UnitQuaternion"
        if inner == "Complex":
            return "UnitComplex"
        if inner == "DualQuaternion":
            return "UnitDualQuaternion"
        if inner in VECTOR_BASES or inner in MATRIX_BASES:
            return "Unit<Vector>"
        return "Unit"
    if base in ("Point", "OPoint") or re.fullmatch(r"Point[1-6]?", base):
        d = dim_value(args[1]) if len(args) > 1 else ""
        m = re.fullmatch(r"Point([1-6])", base)
        return f"Point{m.group(1)}" if m else (f"Point{d}" if d else "Point")
    if base in ("Rotation", "Translation", "Scale"):
        d = dim_value(args[1]) if len(args) > 1 else ""
        return f"{base}{d}" if d and d != "X" else base
    if base in ("Isometry", "Similarity"):
        if len(args) >= 3:
            rot = type_head(args[1])[0].lstrip("$")
            d = dim_value(args[2])
            if rot in ("UnitComplex",) and d == "2":
                return f"{base}2"
            if rot in ("Rotation2",) and d == "2":
                return f"{base}Matrix2"
            if rot in ("UnitQuaternion",) and d == "3":
                return f"{base}3"
            if rot in ("Rotation3",) and d == "3":
                return f"{base}Matrix3"
            if rot in ("Rot",) and d == "3":
                return f"{base}3"
        return base
    if base == "Transform":
        return "Transform"
    if base in ("Reflection",):
        return "Reflection"
    if base in ("TAffine", "TGeneral", "TProjective", "TCategory", "TCategoryMul",
                "SuperTCategoryOf", "SubTCategoryOf"):
        return "Transform"
    if base in ("T", "Name", "ViewStorage", "ViewStorageMut") or re.search(r"View|Slice", base):
        return "MatrixView"
    return base


def impl_rhs_family(value: str) -> str:
    """Normalize a type appearing inside an impl name (operator rhs, conversion source)."""
    value = value.strip()
    if not value:
        return ""
    if value.startswith("["):
        inner = value[1:-1]
        parts = split_top(inner, ";")
        elem = impl_rhs_family(parts[0])
        return f"[{elem}; N]"
    if value.startswith("("):
        return "(" + ", ".join(impl_rhs_family(p) for p in split_top(value[1:-1])) + ")"
    base, args = type_head(value)
    base = base.lstrip("$")
    if base in ("T", "N", "Self::Element", "T::Element") or re.fullmatch(
            r"(?:f32|f64|u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize)", base):
        return "T"
    owner = rust_owner(value)
    if re.fullmatch(r"(?:Matrix|Vector|RowVector|DMatrix|DVector|RowDVector|SquareMatrix)\w*",
                    owner):
        return "Matrix"
    if owner == "Unit<Vector>":
        return "Unit<Matrix>"
    for fam in ("Point", "Rotation", "Translation", "Scale", "Isometry", "Similarity"):
        if owner.startswith(fam):
            return fam
    if owner in ("MatrixView",):
        return "Matrix"
    return owner


# Rust traits whose impls are part of the inventory.  Other impls (storage, allocator, typenum,
# simba subset glue) are machinery; their families are covered by the EXCLUDE rules below when
# they are listed.
RUST_IMPL_TRAITS = {
    "Add", "AddAssign", "Sub", "SubAssign", "Mul", "MulAssign", "Div", "DivAssign", "Neg",
    "Index", "IndexMut", "From", "Default", "PartialEq", "Eq", "PartialOrd", "Hash",
    "AbsDiffEq", "RelativeEq", "UlpsEq", "Sum", "Product", "Display", "Debug", "LowerExp",
    "Copy", "Clone", "Serialize", "Deserialize", "Zero", "One", "Bounded", "Deref", "DerefMut",
    "Distribution", "Arbitrary", "Pod", "Zeroable", "SubsetOf", "Borrow", "BorrowMut",
    "AsRef", "AsMut", "IntoIterator", "Extend", "FromIterator", "TryFrom", "Archive",
    "ShaderType",
}
GLAM_TYPE = re.compile(r"(?:[DIUB]|I64|U64|I16|U16|I8|U8)?Vec[234]A?|D?Mat[234]A?|D?Quat|"
                       r"D?Affine[23]A?")
BINARY_OPS = {"Add", "AddAssign", "Sub", "SubAssign", "Mul", "MulAssign", "Div", "DivAssign"}


def rust_impl_item(trait_expr: str, self_type: str) -> tuple[str, str] | None:
    """(owner, impl name) of `impl trait_expr for self_type`, or None if not tracked."""
    trait_expr = squash(trait_expr)
    trait_expr = re.sub(r"\b(?:ops|cmp|fmt|hash|iter|convert|borrow|approx|num|simba|serde|"
                        r"rand|distr|distributions|bytemuck|core|std|marker|"
                        r"simba::scalar)::", "", trait_expr)
    base, args = type_head(trait_expr)
    base = base.split("::")[-1]
    if base not in RUST_IMPL_TRAITS:
        return None
    target = squash(self_type)
    target_owner = rust_owner(target)
    target_fam = impl_rhs_family(target)
    foreign_target = target.startswith(("[", "(")) or target_fam == "T" or target_owner in (
        "Vec", "Box") or re.match(r"(?:mint|glam|Complex)\b", target) is not None or \
        GLAM_TYPE.fullmatch(type_head(target)[0]) is not None
    args = [re.sub(r"\bSelf\b", target, a) for a in args]
    if base == "From" and args:
        src = args[0]
        if foreign_target:
            return rust_owner(src), f"Into<{impl_rhs_family(target)}>"
        return target_owner, f"From<{impl_rhs_family(src)}>"
    if base in ("SubsetOf",) and args:
        # `impl SubsetOf<Sup> for Sub`: `nalgebra::convert` from Sub into Sup.
        return target_owner, f"SubsetOf<{impl_rhs_family(args[0])}>"
    if base in BINARY_OPS:
        rhs = impl_rhs_family(args[0]) if args else impl_rhs_family(target)
        if target_fam == "T" and args:
            # `impl Mul<Matrix> for f32`: scalar on the left.
            return rust_owner(args[0]), f"{base}<{rhs}> for T"
        return target_owner, f"{base}<{rhs}>"
    if base in ("Index", "IndexMut") and args:
        idx = squash(args[0])
        idx = "usize" if idx == "usize" else ("(usize, usize)" if "usize" in idx else "range")
        return target_owner, f"{base}<{idx}>"
    if base in ("Sum", "Product") and args:
        return target_owner, f"{base}<{impl_rhs_family(args[0])}>"
    if base in ("Deref", "DerefMut", "AsRef", "AsMut", "Borrow", "BorrowMut"):
        return target_owner, base
    if base == "Distribution" and args:
        return rust_owner(args[0]), "Distribution"
    if base == "TryFrom" and args:
        return target_owner, f"TryFrom<{impl_rhs_family(args[0])}>"
    if base in ("Extend", "FromIterator", "IntoIterator"):
        return target_owner, base
    return target_owner, base


# --------------------------------------------------------------------------------------------
# Rust side: macro expansion (positional `$ident` binding)
# --------------------------------------------------------------------------------------------


@dataclass
class Macro:
    name: str
    params: list[str]
    matcher: str
    body: str
    file: str


MACRO_RE = re.compile(r"\bmacro_rules!\s*([A-Za-z_]\w*)\s*[({]")


def parse_macros(text: str, rel: str) -> tuple[list[Macro], list[tuple[int, int]]]:
    """macro_rules definitions (first arm only) and their spans."""
    macros, spans = [], []
    for m in MACRO_RE.finditer(text):
        opening = m.end() - 1
        end = closing(text, opening)
        spans.append((m.start(), end))
        inner = text[opening + 1:end]
        arm = re.search(r"[(\[{]", inner)
        if not arm:
            continue
        a0 = arm.start()
        a1 = closing(inner, a0)
        matcher = inner[a0 + 1:a1]
        arrow = inner.find("=>", a1)
        bopen = re.search(r"[{(\[]", inner[arrow + 2:])
        if arrow < 0 or not bopen:
            continue
        b0 = arrow + 2 + bopen.start()
        b1 = closing(inner, b0)
        params = re.findall(r"\$([A-Za-z_]\w*)\s*:", matcher)
        macros.append(Macro(m.group(1), params, matcher, inner[b0 + 1:b1], rel))
    return macros, spans


TOKEN_RE = re.compile(r"\s*(::|=>|->|[A-Za-z_$][\w$]*|\d[\w.]*|.)", re.S)


def token_trees(text: str) -> list[str]:
    """Rust token trees: bracketed groups are one token, `::` / `=>` / `->` are one token."""
    out: list[str] = []
    i = 0
    while i < len(text):
        m = TOKEN_RE.match(text, i)
        if not m or not m.group(1).strip():
            break
        tok = m.group(1)
        start = m.start(1)
        if tok in "([{":
            end = closing(text, start)
            out.append(text[start:end + 1])
            i = end + 1
            continue
        out.append(tok)
        i = m.end()
    return out


def matcher_elements(matcher: str) -> list:
    """Matcher as a list of ('param', name, frag) / ('lit', token) / ('rep', elements, sep)."""
    elements: list = []
    toks = token_trees(matcher)
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == "$" and i + 1 < len(toks) and toks[i + 1].startswith("("):
            inner = matcher_elements(toks[i + 1][1:-1])
            i += 2
            sep = ""
            if i < len(toks) and toks[i] not in ("*", "+", "?"):
                sep = toks[i]
                i += 1
            i += 1  # the repetition operator
            elements.append(("rep", inner, sep))
            continue
        if t.startswith("$") and len(t) > 1 and i + 2 < len(toks) and toks[i + 1] == ":":
            elements.append(("param", t[1:], toks[i + 2]))
            i += 3
            continue
        elements.append(("lit", t))
        i += 1
    return elements


def match_elements(elements: list, toks: list[str], pos: int, binding: dict[str, str],
                   out: list[dict[str, str]], top: bool) -> int:
    """Greedy, backtracking-free match of `elements` against `toks` from `pos`; returns the new
    position or -1.  Linear in the number of tokens."""
    for k, el in enumerate(elements):
        nxt = elements[k + 1] if k + 1 < len(elements) else None
        stop = nxt[1] if nxt and nxt[0] == "lit" else None
        if el[0] == "lit":
            if pos < len(toks) and toks[pos] == el[1]:
                pos += 1
                continue
            return -1
        if el[0] == "param":
            name, frag = el[1], el[2]
            if frag in ("ident", "tt", "lifetime", "literal"):
                if pos >= len(toks):
                    return -1
                binding[name] = toks[pos]
                pos += 1
                continue
            start, depth = pos, 0
            while pos < len(toks):
                t = toks[pos]
                if depth == 0 and (t == stop or (stop is None and t in (",", ";"))):
                    break
                if t == "<":
                    depth += 1
                elif t == ">":
                    depth = max(0, depth - 1)
                pos += 1
            if pos == start:
                return -1
            binding[name] = " ".join(toks[start:pos])
            continue
        inner, sep = el[1], el[2]
        while pos < len(toks) and (stop is None or toks[pos] != stop):
            local = {} if top else binding
            new = match_elements(inner, toks, pos, local, out, False)
            if new < 0 or new == pos:
                break
            pos = new
            if top:
                out.append(local)
            if sep and pos < len(toks) and toks[pos] == sep:
                pos += 1
            elif sep:
                break
    return pos


def bind(macro: Macro, args: str) -> list[dict[str, str]]:
    """Bindings of the macro's parameters: one dict for a plain matcher, one per repetition of a
    top-level `$( .. ),*` group."""
    if not macro.params:
        return [{}]
    reps: list[dict[str, str]] = []
    binding: dict[str, str] = {}
    pos = match_elements(matcher_elements(macro.matcher), token_trees(args), 0, binding, reps,
                         True)
    if pos < 0:
        return [{}]
    return [{**binding, **r} for r in reps] if reps else [binding]


def substitute(text: str, binding: dict[str, str]) -> str:
    return re.sub(r"\$([A-Za-z_]\w*)", lambda m: binding.get(m.group(1), m.group(0)), text)


# --------------------------------------------------------------------------------------------
# Rust side: declarations
# --------------------------------------------------------------------------------------------

IMPL_RE = re.compile(r"(?<![\w$])impl\b")
PUB_FN_RE = re.compile(r"\bpub\s+(?:const\s+)?(unsafe\s+)?fn\s+(\$?[A-Za-z_]\w*)")
PUB_CONST_RE = re.compile(r"\bpub\s+const\s+([A-Z][A-Z0-9_]*)\s*:")


def parse_impl_header(text: str, at: int) -> tuple[str | None, str, int] | None:
    """(trait or None, self type, index of the opening brace) of the impl starting at `at`."""
    i = at + len("impl")
    while i < len(text) and text[i].isspace():
        i += 1
    i = skip_generics(text, i)
    depth = 0
    j = i
    while j < len(text):
        c = text[j]
        if c in "<([":
            depth += 1
        elif c in ">)]" and not (c == ">" and text[j - 1] in "-="):
            depth -= 1
        elif c == "{" and depth <= 0:
            break
        elif c == ";" and depth <= 0:
            return None
        j += 1
    else:
        return None
    header = text[i:j]
    wm = re.search(r"(?<![\w$])where\b", header)
    if wm:
        header = header[:wm.start()]
    header = squash(header)
    parts = [p for p in re.split(r"(?<![\w$])for(?![\w$])", header)]
    # `for` inside generics (HRTB) is rare in the public API; take the last top-level split.
    if len(parts) >= 2:
        return squash(" for ".join(parts[:-1])), squash(parts[-1]), j
    return None, header, j


def module_of(rel: str) -> str:
    first = rel.split("/")[0]
    return "root" if first.endswith(".rs") else first


def is_deprecated(text: str, pos: int) -> bool:
    """`#[deprecated]` among the attributes right before the item at `pos`."""
    start = max(text.rfind("}", 0, pos), text.rfind(";", 0, pos))
    return "#[deprecated" in text[start + 1:pos]


def pub_fns(body: str) -> list[tuple[str, str, str]]:
    """(name, kind, flags) of the `pub fn`s of a body; `unsafe fn`s get the kind
    `unsafe-method`, `#[deprecated]` ones the flag `deprecated`."""
    return [(m.group(2), "unsafe-method" if m.group(1) else "method",
             "deprecated" if is_deprecated(body, m.start()) else "")
            for m in PUB_FN_RE.finditer(body)]


def impl_blocks(text: str) -> list[tuple[str | None, str, int, int, int]]:
    """(trait, self type, impl start, body start, body end) of every impl in `text`."""
    found = []
    for m in IMPL_RE.finditer(text):
        # Skip `impl Trait` in argument/return position.
        if re.search(r"(?:->|[,:<&]|(?<!\$)\()\s*$", text[max(0, m.start() - 12):m.start()]):
            continue
        header = parse_impl_header(text, m.start())
        if header is None:
            continue
        trait, self_type, opening = header
        try:
            end = closing(text, opening)
        except ValueError:
            continue
        found.append((trait, self_type, m.start(), opening, end))
    return found


def inside(pos: int, spans) -> bool:
    return any(s < pos < e for s, e in spans)


@dataclass
class RustFile:
    rel: str
    text: str
    macros: list[Macro]
    macro_spans: list[tuple[int, int]]


def add_impl_items(items: set[Item], trait: str | None, self_type: str, body: str,
                   module: str, rel: str) -> None:
    if trait is None:
        owner = rust_owner(self_type)
        for name, kind, flags in pub_fns(body):
            if not name.startswith("$"):
                items.add(Item(owner, kind, name, module, rel, flags))
        for m in PUB_CONST_RE.finditer(body):
            items.add(Item(owner, "const", m.group(1), module, rel))
        return
    tracked = rust_impl_item(trait, self_type)
    if tracked:
        owner, name = tracked
        items.add(Item(impl_owner(owner, module), "impl", name, module, rel))


def impl_owner(owner: str, module: str) -> str:
    """Trait impls stated per dimension by macros (coordinates `Deref`, `From<[T; D]>`...) belong
    to the generic family; only the glam conversions are genuinely per dimension."""
    if module == "third_party":
        return owner
    if re.fullmatch(r"Matrix[1-6]x[1-6]", owner) or re.fullmatch(r"Matrix[156]", owner):
        return "Matrix"
    if re.fullmatch(r"Vector[1-6]", owner):
        return "Vector"
    m = re.fullmatch(r"(Point|Translation|Scale)[1-6]", owner)
    if m:
        return m.group(1)
    return owner


SELF_RHS_RE = re.compile(r"\bself\s*:\s*")


def op_macro_items(args: str) -> list[tuple[str, str, str]]:
    """(trait, self type, rhs type) triples of a geometry operator macro invocation."""
    head = re.match(r"\s*([A-Z]\w*)\s*,", args)
    if not head:
        return []
    trait = head.group(1)
    out = []
    for m in SELF_RHS_RE.finditer(args):
        rest = args[m.end():]
        fields = split_top(rest, ",;")
        if len(fields) < 2:
            continue
        self_type = fields[0]
        rhs = re.sub(r"^\s*\w+\s*:\s*", "", fields[1])
        out.append((trait, self_type, rhs))
    return out


SWIZZLE_RE = re.compile(r"\b([xyzw]{2,4})\s*\(\s*\)\s*->")


def parse_rust_file(rel: str, raw: str, all_macros: dict[str, Macro], items: set[Item]) -> None:
    text = mask_comments(raw)
    module = module_of(rel)
    macros, macro_spans = parse_macros(text, rel)
    # Several files define a macro of the same name: the local definition wins.
    all_macros = {**all_macros, **{m.name: m for m in macros}}

    # cfg(test) modules are private test code.
    test_spans = []
    for m in re.finditer(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{", text):
        test_spans.append((m.start(), closing(text, m.end() - 1)))

    blocks = impl_blocks(text)
    impl_spans = [(b, e) for _, _, _, b, e in blocks]

    # 1. Plain impls (outside macro definitions).
    for trait, self_type, start, opening, end in blocks:
        if inside(start, macro_spans) or inside(start, test_spans):
            continue
        body = text[opening + 1:end]
        if "$" in self_type or (trait and "$" in trait):
            continue
        add_impl_items(items, trait, self_type, body, module, rel)
        # Macro invocations inside an inherent impl body: `component_binop_impl!(..)`.
        if trait is None:
            owner = rust_owner(self_type)
            for inv in re.finditer(r"\b([a-z_]\w*)!\s*\(", body):
                macro = all_macros.get(inv.group(1))
                if not macro:
                    continue
                args = body[inv.end():closing(body, inv.end() - 1)]
                if inv.group(1) == "impl_swizzle":
                    for sw in SWIZZLE_RE.finditer(args):
                        items.add(Item(owner, "method", sw.group(1), module, rel))
                    continue
                for binding in bind(macro, args):
                    for name, kind, flags in pub_fns(macro.body):
                        name = substitute(name, binding)
                        if not name.startswith("$"):
                            items.add(Item(owner, kind, name, module, rel, flags))

    # 2. Macro invocations at file level: expand the impls of the macro body.
    for inv in re.finditer(r"(?<![\w$])([a-z_]\w*)!\s*[({]", text):
        if inside(inv.start(), macro_spans) or inside(inv.start(), test_spans):
            continue
        if inside(inv.start(), impl_spans):
            continue
        name = inv.group(1)
        opening = inv.end() - 1
        try:
            args = text[opening + 1:closing(text, opening)]
        except ValueError:
            continue
        if "self:" in args and "Output" in args:
            for trait, self_type, rhs in op_macro_items(args):
                tracked = rust_impl_item(f"{trait}<{rhs}>", self_type)
                if tracked:
                    items.add(Item(impl_owner(tracked[0], module), "impl", tracked[1], module,
                                   rel))
            continue
        macro = all_macros.get(name)
        if not macro:
            continue
        for binding in bind(macro, args):
            body = macro.body
            for trait, self_type, start, b0, b1 in impl_blocks(body):
                trait_s = substitute(trait, binding) if trait else None
                self_s = substitute(self_type, binding)
                add_impl_items(items, trait_s, self_s, substitute(body[b0 + 1:b1], binding),
                               module, rel)

    # 3. Types, traits, aliases and free functions.
    fn_spans = impl_spans + macro_spans + test_spans
    trait_spans = []
    for m in re.finditer(r"\bpub\s+trait\s+([A-Za-z_]\w*)[^{;]*\{", text):
        end = closing(text, m.end() - 1)
        trait_spans.append((m.start(), end))
        items.add(Item(m.group(1), "trait", m.group(1), module, rel))
    for m in re.finditer(r"\bpub\s+(struct|enum)\s+([A-Za-z_]\w*)", text):
        if inside(m.start(), macro_spans + test_spans):
            continue
        items.add(Item(rust_owner(m.group(2)), "type", m.group(2), module, rel))
        for trait in derived_traits(text, m.start()):
            if trait in RUST_IMPL_TRAITS:
                items.add(Item(rust_owner(m.group(2)), "impl", trait, module, rel))
    for m in re.finditer(r"\bpub\s+type\s+([A-Za-z_]\w*)\s*(<[^=]*>)?\s*=\s*([^;]+);", text):
        if inside(m.start(), macro_spans + test_spans + trait_spans):
            continue
        items.add(Item(alias_owner(m.group(1), m.group(3)), "type", m.group(1), module, rel))
    for m in PUB_FN_RE.finditer(text):
        if inside(m.start(), fn_spans + trait_spans):
            continue
        owner = "nalgebra" if module == "root" else f"nalgebra::{module}"
        items.add(Item(owner, "function", m.group(2), module, rel))


def derived_traits(text: str, pos: int) -> list[str]:
    """Traits of the `#[derive(..)]` / `#[cfg_attr(.., derive(..))]` attributes of the item at
    `pos` (attributes only: the segment since the previous item's end)."""
    start = max(text.rfind("}", 0, pos), text.rfind(";", 0, pos))
    segment = text[start + 1:pos]
    traits = []
    for m in re.finditer(r"\bderive\s*\(", segment):
        args = segment[m.end():closing(segment, m.end() - 1)]
        traits += [t.split("::")[-1] for t in re.findall(r"[A-Za-z_][\w:]*", args)]
    return traits


def alias_owner(name: str, target: str) -> str:
    """Owner family of a `pub type` alias."""
    for prefix, owner in (("Matrix", "Matrix"), ("SMatrix", "Matrix"), ("OMatrix", "Matrix"),
                          ("DMatrix", "DMatrix"), ("RowDVector", "RowDVector"),
                          ("DVector", "DVector"), ("RowVector", "Matrix"),
                          ("RowSVector", "Matrix"), ("RowOVector", "Matrix"),
                          ("SVector", "Vector"), ("OVector", "Vector"), ("Vector", "Vector"),
                          ("UnitVector", "Unit<Vector>"), ("SquareMatrix", "SquareMatrix"),
                          ("UnitDualQuaternion", "UnitDualQuaternion"),
                          ("UnitQuaternion", "UnitQuaternion"), ("UnitComplex", "UnitComplex"),
                          ("Point", "Point"), ("Rotation", "Rotation"),
                          ("Translation", "Translation"), ("Scale", "Scale"),
                          ("IsometryMatrix", "Isometry"), ("Isometry", "Isometry"),
                          ("SimilarityMatrix", "Similarity"), ("Similarity", "Similarity"),
                          ("Affine", "Transform"), ("Projective", "Transform"),
                          ("Transform", "Transform"), ("Reflection", "Reflection"),
                          ("MatrixView", "MatrixView"), ("MatrixSlice", "MatrixView"),
                          ("VectorView", "MatrixView"), ("VectorSlice", "MatrixView"),
                          ("DMatrixView", "MatrixView"), ("DVectorView", "MatrixView"),
                          ("DMatrixSlice", "MatrixView"), ("DVectorSlice", "MatrixView"),
                          ("Owned", "Allocator"), ("Cs", "CsMatrix")):
        if name.startswith(prefix):
            if re.search(r"View|Slice", name):
                return "MatrixView"
            return owner
    base = type_head(target)[0]
    return rust_owner(target) if base else name


def parse_rust(nalgebra_root: Path) -> list[Item]:
    src = nalgebra_root / "src"
    if not (src / "lib.rs").is_file():
        raise SystemExit(f"nalgebra-rs source directory not found: {src}")
    files = []
    for path in sorted(src.rglob("*.rs")):
        rel = str(path.relative_to(src))
        files.append((rel, path.read_text()))
    all_macros: dict[str, Macro] = {}
    for rel, raw in files:
        for macro in parse_macros(mask_comments(raw), rel)[0]:
            all_macros.setdefault(macro.name, macro)
    items: set[Item] = set()
    for rel, raw in files:
        parse_rust_file(rel, raw, all_macros, items)
    # `nalgebra_macros` re-exports (lib.rs `pub use nalgebra_macros::{..}`).
    lib = mask_comments((src / "lib.rs").read_text())
    m = re.search(r"pub\s+use\s+nalgebra_macros::\{([^}]*)\}", lib)
    if m:
        for name in re.findall(r"\w+", m.group(1)):
            items.add(Item("nalgebra", "macro", f"{name}!", "root", "lib.rs"))
    return unique_items(items)


def parse_simba(simba_root: Path) -> list[str]:
    """Method names of simba's ComplexField / RealField / Field traits (annotation only)."""
    names = set()
    for rel in ("src/scalar/real.rs", "src/scalar/complex.rs", "src/scalar/field.rs"):
        path = simba_root / rel
        if not path.is_file():
            raise SystemExit(f"simba source missing: {path}")
        text = mask_comments(path.read_text())
        for m in re.finditer(r"\bpub\s+trait\s+(RealField|ComplexField|Field)\b[^{]*\{", text):
            body = text[m.end():closing(text, m.end() - 1)]
            names.update(re.findall(r"\bfn\s+([A-Za-z_]\w*)", body))
    return sorted(names)


# --------------------------------------------------------------------------------------------
# Cairo side
# --------------------------------------------------------------------------------------------

CAIRO_CRATES = ("crates/nalgebra/src",)


@functools.cache
def simba_root() -> Path:
    """Root of the `simba` dependency (registry package from simba-cairo, fetched by Scarb)."""
    meta = json.loads(subprocess.run(
        ["scarb", "--json", "metadata", "--format-version", "1"], cwd=ROOT, check=True,
        capture_output=True, text=True).stdout.splitlines()[-1])
    roots = [Path(p["root"]) for p in meta["packages"] if p["name"] == "simba"]
    if not roots:
        raise SystemExit("simba package not found in `scarb metadata`")
    return roots[0]
TEST_FILE = re.compile(r"^(?:tests|testing|benches|oracle.*|matrix_test_utils)$")
CAIRO_CORE_TRAITS = {
    "Add", "AddAssign", "Sub", "SubAssign", "Mul", "MulAssign", "Div", "DivAssign", "Neg",
    "Into", "TryInto", "Default", "PartialEq", "PartialOrd", "Hash", "Serde", "Debug", "Display",
    "IndexView", "Index", "Zero", "One", "Bounded",
}
CAIRO_DERIVES = {"Copy", "PartialEq", "Serde", "Default", "Debug", "Hash", "Display"}


def cairo_rhs_family(value: str) -> str:
    value = value.strip()
    if value in ("T", "Fixed"):
        return "T"
    return impl_rhs_family(value)


def cairo_owner(name: str, types: list[str], fallback: str) -> str:
    """Owner of a trait / impl from its name: the longest Cairo type name it starts with."""
    stem = re.sub(r"(?:Trait|Impl)$", "", name)
    best = ""
    for t in types:
        if stem.startswith(t) and len(t) > len(best):
            best = t
    return best or fallback


def cairo_impl_item(trait_expr: str, owner: str) -> tuple[str, str] | None:
    trait_expr = squash(trait_expr)
    base, args = type_head(trait_expr)
    base = base.split("::")[-1]
    if base not in CAIRO_CORE_TRAITS:
        return None
    if base in ("Into", "TryInto") and len(args) >= 2:
        src, dst = args[0], args[1]
        rust = "From" if base == "Into" else "TryFrom"
        # Like the Rust side: `From<Src>` on the target, unless the target is foreign (tuple,
        # array, scalar), then `Into<Dst>` on the source.
        if dst.strip().startswith(("(", "[")) or cairo_rhs_family(dst) == "T":
            return rust_owner(src), f"Into<{cairo_rhs_family(dst)}>"
        return rust_owner(dst), f"{rust}<{cairo_rhs_family(src)}>"
    if base.endswith("Assign") and len(args) >= 2:
        return owner, f"{base}<{cairo_rhs_family(args[1])}>"
    if base in BINARY_OPS:
        target = rust_owner(args[0]) if args else owner
        return target, f"{base}<{cairo_rhs_family(args[0]) if args else ''}>"
    if base in ("IndexView", "Index") and len(args) >= 2:
        return owner, f"Index<{squash(args[1])}>"
    if base == "Neg":
        return (rust_owner(args[0]) if args else owner), "Neg"
    return (rust_owner(args[0]) if args else owner), base


def cairo_files() -> list[Path]:
    paths = []
    for src in [ROOT / crate for crate in CAIRO_CRATES] + [simba_root() / "src"]:
        for path in sorted(src.rglob("*.cairo")):
            rel = path.relative_to(src)
            if any(TEST_FILE.match(Path(part).stem) for part in rel.parts):
                continue
            paths.append(path)
    return paths


# Generic traits implemented in other files than their own: `Norm<N, M, T>` (`base/norm.cairo`),
# implemented in each shape's module for the four norm markers (`Matrix3EuclideanNorm`), so its
# `norm` / `metric_distance` are the shapes' methods of the same names (WP 8.2b).
CROSS_FILE_TRAITS = {"Norm"}


def parse_cairo() -> list[Item]:
    files = [(p, mask_comments(p.read_text())) for p in cairo_files()]
    types = sorted({m.group(1) for _, text in files
                    for m in re.finditer(r"\bpub\s+(?:struct|enum)\s+([A-Za-z_]\w*)", text)})
    items: set[Item] = set()
    trait_re = re.compile(r"\bpub\s+trait\s+([A-Za-z_]\w*)[^{;]*\{")
    impl_re = re.compile(r"(#\[generate_trait\]\s*)?\bpub\s+impl\s+([A-Za-z_]\w*)")
    for path, text in files:
        if path.is_relative_to(ROOT / "crates/nalgebra"):
            crate, source = "nalgebra", str(path.relative_to(ROOT))
        else:
            crate, source = "simba", str(Path("simba") / path.relative_to(simba_root()))
        module = "simba" if crate == "simba" else (
            path.relative_to(ROOT / "crates/nalgebra/src").parts[0].removesuffix(".cairo"))
        test_spans = []
        for m in re.finditer(r"#\[cfg\(test\)\]\s*(?:pub(?:\(crate\))?\s+)?mod\s+\w+\s*\{", text):
            test_spans.append((m.start(), closing(text, m.end() - 1)))
        spans = []
        for m in trait_re.finditer(text):
            if inside(m.start(), test_spans):
                continue
            end = closing(text, m.end() - 1)
            spans.append((m.start(), end))
            name = m.group(1)
            owner = cairo_owner(name, types, "")
            if name in ("Real", "Transcendental"):
                owner = f"simba::{name}"
            items.add(Item(owner or name, "trait", name, module, source))
            body = text[m.end():end]
            fns = re.findall(r"\bfn\s+([A-Za-z_]\w*)", body)
            consts = re.findall(r"\bconst\s+([A-Z][A-Z0-9_]*)\s*:", body)
            if not owner:
                # Generic trait (`Normalizable<V, T>`): its methods belong to each implementor,
                # found in the trait's file, or in every file for `CROSS_FILE_TRAITS`.
                scope = "\n".join(t for _, t in files) if name in CROSS_FILE_TRAITS else text
                impls = re.findall(rf"\bpub\s+impl\s+([A-Za-z_]\w*)\s*(?:<[^{{]*?>)?\s*of\s+"
                                   rf"{name}\b", mask_comments(scope))
                owners = sorted({cairo_owner(i, types, "") for i in impls} - {""}) or [
                    {"PermTrait": "Perm"}.get(name, name)]
            else:
                owners = [owner]
            for o in owners:
                for fn in fns:
                    items.add(Item(o, "method", fn, module, source))
                for c in consts:
                    items.add(Item(o, "const", c, module, source))
        for m in impl_re.finditer(text):
            if inside(m.start(), test_spans):
                continue
            opening = text.find("{", m.end())
            header = squash(text[m.end():opening])
            end = closing(text, opening)
            spans.append((m.start(), end))
            impl_name = m.group(2)
            of = re.search(r"\bof\s+(.+)$", header)
            owner = cairo_owner(impl_name, types, "")
            body = text[opening + 1:end]
            if m.group(1):
                # #[generate_trait]: the impl declares the trait.
                trait_name = type_head(of.group(1))[0] if of else impl_name
                owner = owner or {"PermTrait": "Perm"}.get(trait_name, trait_name)
                items.add(Item(owner, "trait", trait_name, module, source))
                for fn in re.findall(r"\bfn\s+([A-Za-z_]\w*)", body):
                    items.add(Item(owner, "method", fn, module, source))
                continue
            if of:
                tracked = cairo_impl_item(of.group(1), owner)
                if tracked:
                    items.add(Item(tracked[0], "impl", tracked[1], module, source))
        for m in re.finditer(r"\bpub\s+type\s+([A-Za-z_]\w*)\s*(?:<[^=]*>)?\s*=", text):
            # `pub type Matrix3x1<T> = Vector3<T>;`: an upstream alias name, a type of its own
            # name (`tools/shapegen/DESIGN.md` §4.1).
            if not inside(m.start(), spans + test_spans):
                items.add(Item(m.group(1), "type", m.group(1), module, source))
        for m in re.finditer(r"\bpub\s+(struct|enum)\s+([A-Za-z_]\w*)", text):
            if inside(m.start(), test_spans):
                continue
            name = m.group(2)
            items.add(Item(name, "type", name, module, source))
            derive = re.search(r"#\[derive\(([^)]*)\)\]\s*(?:#\[[^\]]*\]\s*)*$",
                               text[max(0, m.start() - 400):m.start()])
            if derive:
                for trait in re.findall(r"[A-Za-z_]\w*", derive.group(1)):
                    if trait in CAIRO_DERIVES:
                        items.add(Item(name, "impl", trait, module, source))
        for m in re.finditer(r"\bpub\s+fn\s+([A-Za-z_]\w*)", text):
            if inside(m.start(), spans + test_spans):
                continue
            items.add(Item(f"{crate}::{module}", "function", m.group(1), module, source))
    return unique_items(items)


# --------------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------------

# The 36 static shapes of `tools/shapegen` (one Cairo struct per shape, the other upstream names
# being `type` aliases): `Matrix1`, `VectorN` (N x 1), `RowVectorN` (1 x N), `MatrixN`, `MatrixRxC`.
DIMS = range(1, 7)


def shape_name(r: int, c: int) -> str:
    if r == c == 1:
        return "Matrix1"
    if c == 1:
        return f"Vector{r}"
    if r == 1:
        return f"RowVector{c}"
    return f"Matrix{r}" if r == c else f"Matrix{r}x{c}"


SHAPES = [shape_name(r, c) for r in DIMS for c in DIMS]
SQUARES = [shape_name(n, n) for n in DIMS]
COLUMNS = [shape_name(n, 1) for n in DIMS]
ROWS = [shape_name(1, n) for n in DIMS]
# Upstream alias -> canonical struct (`Matrix3x1` -> `Vector3`, `Vector1` -> `Matrix1`, ...).
SHAPE_ALIASES = {"Vector1": "Matrix1", "RowVector1": "Matrix1",
                 **{f"Matrix{n}x1": f"Vector{n}" for n in DIMS if n > 1},
                 **{f"Matrix1x{n}": f"RowVector{n}" for n in DIMS if n > 1}}
# The shapes that existed before WP 8.1b-3 (the dimension-restricted items below still name them).
V = ["Vector2", "Vector3", "Vector4", "Vector6"]
M = ["Matrix2", "Matrix3", "Matrix4", "Matrix6"]

# The six aliases of upstream's `Transform<T, C, D>` (WP 8.4-P11a), one Cairo struct each.
TRANSFORMS = ["Transform2", "Transform3", "Projective2", "Projective3", "Affine2", "Affine3"]

# Upstream owner -> the Cairo types that stand for it.  Owners absent from this table have no
# Cairo counterpart yet (their items are missing unless excluded).
OWNER_CANDIDATES: dict[str, list[str]] = {
    "Matrix": SHAPES,
    "SquareMatrix": SQUARES,
    "Vector": COLUMNS,
    "RowSVector": ROWS,
    "RowVector": ROWS,
    **{t: [t] for t in SHAPES},
    **{alias: [t] for alias, t in SHAPE_ALIASES.items()},
    # WP 8.4-P12: `UnitDualQuaternion` is upstream's `Unit<DualQuaternion>`.
    "Unit": ["Unit", "UnitComplex", "UnitQuaternion", "UnitDualQuaternion"],
    "Unit<Vector>": ["Unit"],
    "Point": ["Point1", "Point2", "Point3", "Point4", "Point5", "Point6"],
    "Point1": ["Point1"],
    "Point2": ["Point2"],
    "Point3": ["Point3"],
    "Point4": ["Point4"],
    "Point5": ["Point5"],
    "Point6": ["Point6"],
    "Rotation": ["Rotation2", "Rotation3"],
    "Rotation2": ["Rotation2"],
    "Rotation3": ["Rotation3"],
    "Translation": ["Translation1", "Translation2", "Translation3", "Translation4", "Translation5", "Translation6"],
    "Translation1": ["Translation1"],
    "Translation2": ["Translation2"],
    "Translation3": ["Translation3"],
    "Translation4": ["Translation4"],
    "Translation5": ["Translation5"],
    "Translation6": ["Translation6"],
    "Scale": [f"Scale{d}" for d in range(1, 7)],
    **{f"Scale{d}": [f"Scale{d}"] for d in range(1, 7)},
    "Reflection": [f"Reflection{d}" for d in range(1, 7)],
    **{f"Reflection{d}": [f"Reflection{d}"] for d in range(1, 7)},
    "Isometry": ["Isometry2", "Isometry3", "IsometryMatrix2", "IsometryMatrix3"],
    "Isometry2": ["Isometry2"],
    "Isometry3": ["Isometry3"],
    "IsometryMatrix2": ["IsometryMatrix2"],
    "IsometryMatrix3": ["IsometryMatrix3"],
    "Similarity": ["Similarity2", "Similarity3", "SimilarityMatrix2", "SimilarityMatrix3"],
    "Similarity2": ["Similarity2"],
    "Similarity3": ["Similarity3"],
    "SimilarityMatrix2": ["SimilarityMatrix2"],
    "SimilarityMatrix3": ["SimilarityMatrix3"],
    "Quaternion": ["Quaternion"],
    "UnitQuaternion": ["UnitQuaternion"],
    "UnitComplex": ["UnitComplex"],
    "DualQuaternion": ["DualQuaternion"],
    "UnitDualQuaternion": ["UnitDualQuaternion"],
    # WP 8.4-P11b: camera projections (a `Matrix4` wrapper each).
    "Perspective3": ["Perspective3"],
    "Orthographic3": ["Orthographic3"],
    # WP 8.4-P11a: upstream's `Transform<T, C, D>` is one Cairo struct per alias (category x dim).
    "Transform": TRANSFORMS,
    "Cholesky": ["Cholesky2", "Cholesky3", "Cholesky4", "Cholesky6"],
    "UDU": ["Udu2", "Udu3", "Udu4", "Udu6"],
    "LU": ["Lu2", "Lu3", "Lu4", "Lu6"],
    "QR": ["Qr2", "Qr3", "Qr4"],
    "SVD": ["Svd2", "Svd3"],
    "SymmetricEigen": ["SymmetricEigen2", "SymmetricEigen3"],
    "PermutationSequence": ["Perm2", "Perm3", "Perm4", "Perm6"],
    # WP 8.5-P14a.
    "GivensRotation": ["GivensRotation"],
    "nalgebra::linalg": ["nalgebra::linalg"],
    # The norm markers of `base/norm.rs` (WP 8.2a).
    **{t: [t] for t in ("EuclideanNorm", "LpNorm", "OneNorm", "UniformNorm", "Normed")},
}

# Methods upstream declares on a generic family but that only make sense for some dimensions
# (they panic or do not type-check elsewhere): the partial check ignores the other Cairo types.
DIM_ONLY: dict[str, set[str]] = {
    "cross": {"Vector3"},
    "cross_matrix": {"Vector3"},
    "perp": {"Vector2"},
    # The axes of `construction.rs` exist from the dimension of their coordinate on (`x()` on
    # `Vector1..6`, `b()` on `Vector6` only).
    **{a: set(COLUMNS[k:]) for k, a in enumerate("xyzwab")},
    "x_axis": {"Unit"}, "y_axis": {"Unit"}, "z_axis": {"Unit"}, "w_axis": {"Unit"},
    "a_axis": {"Unit"}, "b_axis": {"Unit"},
    # Upstream's swizzles (`impl_swizzle!`, `base/swizzle.rs`, `geometry/swizzle.rs`) exist on the
    # vectors and points whose dimension exceeds their largest index (`xx` from 1 on, `zyx` from
    # 3 on): `Vector1` is `Matrix1`.
    **{name: set(COLUMNS[max(idx):]) | {f"Point{d}" for d in range(max(idx) + 1, 7)}
       for k in (2, 3) for idx in itertools.product(range(3), repeat=k)
       for name in ["".join("xyz"[i] for i in idx)]},
    # One dimension more (`push`, homogeneous coordinates): upstream's aliases, hence the Cairo
    # shapes, stop at 6, so `Vector6` / `Matrix6` have none (owner ruling, issue #41). `Matrix1` is both `Vector1` and a
    # 1x1 square: upstream's two `to_homogeneous` (`Vector1 -> Vector2`, `Matrix1 -> Matrix2`)
    # are ambiguous on it (a Rust call does not compile), so it has neither.
    "push": set(COLUMNS[:5]), "from_homogeneous": set(COLUMNS[:5]),
    "to_homogeneous": set(COLUMNS[1:5] + SQUARES[1:5]),
    # Upstream names isometries and similarities in 2D and 3D only (`Isometry2/3`,
    # `IsometryMatrix2/3`, `Similarity2/3`...), so their homogeneous matrices are 3x3 and 4x4.
    **{name: {"Matrix3", "Matrix4"} for name in ("From<Isometry>", "From<Similarity>")},
    # `m / r` needs as many columns as the rotation's dimension: `Rotation2` / `Rotation3`
    # (upstream's `Rotation<D>` aliases stop at 3).
    "Div<Rotation>": {shape_name(r, c) for r in DIMS for c in (2, 3)},
    # The homogeneous matrix of `Translation<D>` is `(D + 1)x(D + 1)`: `Matrix2..6` for
    # `Translation1..5` (no `Translation0`, no 7x7 matrix).
    "From<Translation>": set(SQUARES[1:]),
    # Likewise `Scale<D>`: `Matrix2..6` for `Scale1..5` (no 7x7 matrix for `Scale6`).
    "From<Scale>": set(SQUARES[1:]),
    "orthonormal_subspace_basis": {"Vector3"},
    # Square-matrix semantics, generated on the 6 squares (`Matrix1..6`).
    **{name: set(SQUARES) for name in (
        "trace", "identity", "is_identity", "from_diagonal_element", "MulAssign<Matrix>")},
    # Backed by a closed form or a decomposition that exists for 2, 3, 4, 6 only: `Matrix1` and
    # `Matrix5` come with the LU / QR / SVD completion (P14, WP 8.5); `is_invertible` /
    # `is_special_orthogonal` are `try_inverse` / `determinant` (WP 8.2c).
    **{name: set(M) for name in (
        "determinant", "try_inverse", "lu", "qr", "svd", "pseudo_inverse", "singular_values",
        "is_invertible", "is_special_orthogonal")},
    # WP 8.5-P14a: the decomposition entry points and the in-place inverse exist where the
    # decomposition / inverse does (`Cholesky2/3/4/6`, `Udu2/3/4/6`, `try_inverse`).
    **{name: set(M) for name in ("cholesky", "udu", "try_inverse_mut")},
    # WP 8.3-P06: upstream's symmetric / hermitian rank-one updates assert a square `self` at
    # run time (`xxgerx`): generated on the six squares.
    **{name: set(SQUARES) for name in ("syger", "hegerc", "ger_symm")},
    # One row / column more or less (WP 8.2c): the neighbouring static shape, when it exists
    # (upstream's aliases, hence the Cairo shapes, stop at 6; no 0-row shape).
    "insert_row": {shape_name(r, c) for r in range(1, 6) for c in DIMS},
    "remove_row": {shape_name(r, c) for r in range(2, 7) for c in DIMS},
    "insert_column": {shape_name(r, c) for r in DIMS for c in range(1, 6)},
    "remove_column": {shape_name(r, c) for r in DIMS for c in range(2, 7)},
    # 1x1 only (upstream `Matrix1` / `Vector1` impls).
    **{name: {"Matrix1"} for name in ("into_scalar", "as_scalar", "to_scalar", "as_scalar_mut")},
    # Upstream multiplies a translation by an isometry / similarity of the same dimension; the
    # `Isometry` / `Similarity` families exist in 2D and 3D only (upstream's aliases).
    **{name: {"Translation2", "Translation3", "Rotation2", "Rotation3", "UnitComplex",
              "UnitQuaternion", "Isometry2", "Isometry3", "Similarity2", "Similarity3",
              "IsometryMatrix2", "IsometryMatrix3", "SimilarityMatrix2", "SimilarityMatrix3"}
       for name in ("Mul<Isometry>", "Mul<Similarity>")},
    # WP 8.4-P09b: upstream's `Isometry * Rotation` / `Similarity / Rotation` exist for the
    # rotation-MATRIX instances only (`Isometry<T, Rotation<T, D>, D>`), `Translation * Rotation`
    # in 2D and 3D, `Matrix * Rotation` for the shapes with 2 or 3 columns.
    **{name: {"Rotation2", "Rotation3", "UnitComplex", "UnitQuaternion", "Translation2",
              "Translation3", "IsometryMatrix2", "IsometryMatrix3", "SimilarityMatrix2",
              "SimilarityMatrix3", *(shape_name(r, c) for r in DIMS for c in (2, 3))}
       for name in ("Mul<Rotation>", "Div<Rotation>")},
    # WP 8.3-P07 (`base/cg.rs`): the helpers whose argument or result is a `D - 1` vector exist
    # on `Matrix2..6` (`Vector1` is `Matrix1`; `Matrix1` would need a 0-dimensional vector); the
    # uniform `new_scaling` / `append_scaling` / `prepend_scaling` exist on `Matrix1` too. The
    # names are shared with the geometry types, which keep their own checks.
    **{name: {t for ts in OWNER_CANDIDATES.values() for t in ts} - {"Matrix1"} for name in (
        "new_nonuniform_scaling", "new_translation", "append_nonuniform_scaling",
        "append_nonuniform_scaling_mut", "prepend_nonuniform_scaling",
        "prepend_nonuniform_scaling_mut", "append_translation", "append_translation_mut",
        "prepend_translation", "prepend_translation_mut", "transform_vector")},
    # WP 8.4-P12: dual quaternions are 3D only (`Translation3 * UnitDualQuaternion`...).
    **{name: {"Translation3", "UnitQuaternion", "Isometry3", "DualQuaternion",
              "UnitDualQuaternion"}
       for name in ("Mul<UnitDualQuaternion>", "Div<UnitDualQuaternion>")},
    # WP 8.4-P11a: upstream's `inverse` family needs `C: SubTCategoryOf<TProjective>` (no
    # `Transform2/3`); the transforms by a unit complex are 2D, by a unit quaternion 3D; the
    # products and quotients by a transform exist for the 2D / 3D types (upstream's aliases).
    **{name: {t for ts in OWNER_CANDIDATES.values() for t in ts} - {"Transform2", "Transform3"}
       for name in ("inverse", "inverse_mut", "inverse_transform_point",
                    "inverse_transform_vector")},
    "Mul<UnitComplex>": {t for ts in OWNER_CANDIDATES.values() for t in ts}
    - {"Affine3", "Projective3", "Transform3"},
    **{name: {t for ts in OWNER_CANDIDATES.values() for t in ts}
       - {"Affine2", "Projective2", "Transform2"}
       for name in ("Mul<UnitQuaternion>", "Div<UnitQuaternion>")},
    **{name: {t for ts in OWNER_CANDIDATES.values() for t in ts}
       - {"Translation1", "Translation4", "Translation5", "Translation6"}
       for name in ("Mul<Transform>", "Div<Transform>")},
}
# WP 8.4-P11a: the transforms have the square-matrix and pose items named above too.
for _name in ("identity", "try_inverse", "to_homogeneous", "Mul<Rotation>", "Div<Rotation>",
              "Mul<Isometry>", "Mul<Similarity>"):
    DIM_ONLY[_name] = DIM_ONLY[_name] | set(TRANSFORMS)

EXCLUSIONS = {
    "simd": "SIMD lanes (`SimdValue`, `simd_*`, AoSoA types): Cairo has no SIMD; the scalar "
            "path is the only path.",
    "rayon": "`rayon` parallel iterators: a Cairo program is sequential.",
    "unsafe": "Raw pointers, `unsafe` functions and uninitialized-memory storage internals: "
              "Cairo memory is write-once and has no pointer arithmetic.",
    "borrow": "Borrowed references (`AsRef` / `AsMut` / `Borrow` / `Deref` to slices, "
              "`&mut` element access, mutable views): Cairo values are `Copy` and passed by value.",
    "fmt": "`Debug` / `Display` / `LowerExp` formatting (Cairo's derived `Debug` exists but "
           "prints nothing nalgebra-shaped): diagnostics, not API.",
    "glue": "Serialization / zero-copy glue other than Cairo `Serde` (`bytemuck`, `rkyv`, "
            "`encase`, the `serde` visitor types).",
    "random": "`rand` / `proptest` / `quickcheck` generators and the `debug` random-matrix "
              "helpers: a proof has no entropy source.",
    "interop": "Interop with other Rust crates (`mint`, `alga`, `num-complex`'s `Complex` "
               "scalar, and the `glam` types glam-cairo does not have: f64 `D*`, aligned `*A`, "
               "`i8`..`u64` integer vectors other than `IVec*` / `UVec*`). The glam conversions for types glam-cairo has are "
               "in scope.",
    "generic-dim": "Generic-dimension machinery subsumed by concrete types: `Dim`, `DimName`, "
                   "`Const`, `Dyn`, typenum `U*`, `Storage` / `RawStorage` / `ArrayStorage` / "
                   "`VecStorage`, `Allocator`, `DefaultAllocator`, `ShapeConstraint`, views / "
                   "slices / iterators as types, `*_generic` constructors, `into_owned` / "
                   "`clone_owned` (identity on owned types).",
}


@dataclass(frozen=True)
class Rule:
    owner: re.Pattern[str]
    item: re.Pattern[str]
    value: str
    note: str = ""


def rule(owner: str, item: str, value: str, note: str = "") -> Rule:
    return Rule(re.compile(owner), re.compile(item), value, note)


def rendered(item: Item) -> str:
    return item.name if item.kind in ("method", "function") else f"{item.kind}:{item.name}"


# Documented renames: upstream (owner, item) -> Cairo item on the owner's candidates.  The value
# is a Cairo rendered name (`method` names are bare, other kinds prefixed); `\1` refers to the
# item pattern's groups.  Owners may be redirected with `Owner::name`.
RENAMES = (
    # WP 8.5-P14a: upstream's triangular solves take any right-hand side with as many rows.
    rule(r"SquareMatrix", r"((?:tr_|ad_)?solve_(?:lower|upper)_triangular\w*)", r"MatrixSolve::\1",
         "method of the generic `MatrixSolve` (one kernel impl per square and right-hand side "
         "with as many rows, 36; `base/solve.cairo`)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d",
         r"impl:Mul<Matrix>", r"MatrixMul::mul_mat",
         "conformable products are `mul_mat` (Cairo's `Mul` is homogeneous; `*` stays on the "
         "square shapes)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"tr_mul",
         r"MatrixTrMul::tr_mul", "method of the generic `MatrixTrMul` (one impl per pair of "
         "shapes with the same number of rows)"),
    # WP 8.3-P06 (`base/blas.rs`): the forms built on `gemm` / `mul_mat` are blanket impls of
    # generic traits (`tools/shapegen/blas.py`), so their methods belong to the trait.
    rule(r"Vector", r"gemv", r"MatrixGemv::gemv", "method of the generic `MatrixGemv` (`gemm` "
         "with a one-column `self`, any matrix `a` with as many rows)"),
    rule(r"Vector", r"(gemv_tr|gemv_ad)", r"MatrixGemvTr::\1", "method of the generic "
         "`MatrixGemvTr` (`gemv` of the transpose of `a`; `gemv_ad` is `gemv_tr` for a real "
         "scalar)"),
    rule(r"Matrix", r"(gemm_tr|gemm_ad)", r"MatrixGemmTr::\1", "method of the generic "
         "`MatrixGemmTr` (`gemm` of the transpose of `a`; `gemm_ad` is `gemm_tr` for a real "
         "scalar)"),
    rule(r"SquareMatrix", r"(quadform|quadform_with_workspace)", r"MatrixQuadform::\1",
         "method of the generic `MatrixQuadform` (`mid.mul_mat(rhs)` then `gemm_tr`; the "
         "workspace is a `ref` argument)"),
    rule(r"SquareMatrix", r"(quadform_tr|quadform_tr_with_workspace)", r"MatrixQuadformTr::\1",
         "method of the generic `MatrixQuadformTr` (`lhs.mul_mat(mid)` then `gemm`; the "
         "workspace is a `ref` argument)"),
    rule(r"Matrix|Vector|SquareMatrix|Point|Quaternion|Scale", r"impl:Mul<T>", "scale",
         "heterogeneous operators are named methods (DESIGN D4)"),
    rule(r"Matrix|Vector|SquareMatrix|Point|Quaternion", r"impl:Div<T>", "unscale",
         "heterogeneous operators are named methods (DESIGN D4)"),
    rule(r".*", r"impl:(?:Serialize|Deserialize)", "impl:Serde", "Cairo `Serde`"),
    rule(r".*", r"impl:Clone", "impl:Copy", "Cairo values are `Copy`"),
    rule(r".*", r"impl:Eq", "impl:PartialEq", "Cairo has no separate `Eq`"),
    rule(r".*", r"impl:AbsDiffEq", "abs_diff_eq", "tolerance in ulp (DESIGN D3)"),
    rule(r".*", r"impl:Zero", "is_zero", "`zeros()` + `is_zero()`"),
    rule(r"Matrix|Vector|Point", r"impl:Deref", "fields",
         "coordinates are named struct fields (`v.x`, `m.m11`)"),
    rule(r"Vector3?", r"cross_matrix", r"Matrix3::cross_matrix",
         "constructor on the matrix side (`Matrix3::cross_matrix(v)`)"),
    rule(r"Vector", r"(x|y|z|w|a|b)_axis", r"Unit::\1_axis",
         "on `Unit<VectorN>` (`Unit2Trait`, `UnitVector5Trait`...)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"ad_mul",
         r"MatrixTrMul::ad_mul", "method of the generic `MatrixTrMul` (`tr_mul` for a real "
         "scalar)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"mul_to",
         r"MatrixMul::mul_to", "default method of the generic `MatrixMul` (`out = "
         "self.mul_mat(rhs)`, the output of the product's own shape)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d",
         r"(tr_mul_to|ad_mul_to)", r"MatrixTrMul::\1", "default method of the generic "
         "`MatrixTrMul` (`out = self.tr_mul(rhs)`)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d",
         r"(get|index)", r"MatrixIndex::\1", "method of the generic `MatrixIndex` (one impl per "
         "shape and index type: `usize`, `(usize, usize)`)"),
    # WP 8.2c: upstream's views, sized by const generics or at run time, are owned copies whose
    # size is the OUTPUT type of a generic trait (`base/matrix_view.cairo`), inferred like `Into`.
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"(fixed_rows|rows|rows_range|select_rows)",
         r"FixedRows::\1", "method of the generic `FixedRows<M, Out>` (owned copy; the output "
         "type is the row count, upstream's const generic or runtime size)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"(fixed_columns|columns|columns_range|select_columns)",
         r"FixedColumns::\1", "method of the generic `FixedColumns<M, Out>` (owned copy; the "
         "output type is the column count)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"(fixed_view|view|fixed_slice|slice)", r"FixedView::\1",
         "method of the generic `FixedView<M, Out>` (owned copy; the output type is the block "
         "size)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"row_part", r"RowPart::row_part", "method of the generic `RowPart<M, Out>` "
         "(owned copy; the output row vector is the length)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"column_part", r"ColumnPart::column_part", "method of the generic "
         "`ColumnPart<M, Out>` (owned copy; the output vector is the length)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"(fixed_resize|resize)", r"FixedResize::\1", "method of the generic "
         "`FixedResize<M, Out, T>` (the output type is the new shape)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector|Matrix\w+|Vector\d|RowVector\d", r"kronecker", r"MatrixKronecker::kronecker", "method of the generic "
         "`MatrixKronecker` (one impl per pair whose product fits in 6x6)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector", r"impl:Mul<Matrix> for T", "scale",
         "Cairo-imposed: heterogeneous operator (`k * m` is `m.scale(k)`)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector", r"impl:Mul<(?:Point|Rotation)>",
         r"MatrixMul::mul_mat", "`m * p` / `m * r` is `m.mul_mat(..)` (Cairo's `Mul` is "
         "homogeneous)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector", r"impl:Div<Rotation>", "div_rotation",
         "Cairo-imposed: heterogeneous operator (`m / r` is `m.div_rotation(r)`)"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector", r"impl:SubsetOf<Matrix>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"Matrix|SquareMatrix|Vector|RowS?Vector", r"eq", "impl:PartialEq",
         "the `PartialEq::eq` of the derived impl (`a == b`)"),
    rule(r"Isometry[23]?|Similarity[23]?|Rotation[23]?|UnitQuaternion|UnitComplex|Translation[23]?|"
         r"Scale",
         r"impl:Mul<Point>", "transform_point", "heterogeneous operators are named methods"),
    rule(r"Isometry[23]?|Similarity[23]?|Rotation[23]?|UnitQuaternion|UnitComplex",
         r"impl:Mul<Matrix>", "transform_vector", "heterogeneous operators are named methods"),
    rule(r"Isometry[23]?|Similarity[23]?|Rotation[23]?|UnitQuaternion|UnitComplex",
         r"impl:Mul<Unit<Matrix>>", "transform_unit_vector",
         "heterogeneous operators are named methods"),
    rule(r"Point", r"impl:Add<Matrix>", "add_vector",
         "`p + v`: Cairo's `Add` is homogeneous, the heterogeneous operator is a named method"),
    rule(r"Point", r"impl:Sub<Matrix>", "sub_vector",
         "`p - v`: Cairo's `Sub` is homogeneous, the heterogeneous operator is a named method"),
    rule(r"Point", r"impl:Sub<Point>", "sub_point",
         "`p - q` (a vector): Cairo's `Sub` is homogeneous, the heterogeneous operator is a named "
         "method"),
    rule(r"Isometry|Similarity", r"impl:Mul<Translation>", "mul_translation",
         "`iso * t`: Cairo's `Mul` is homogeneous, the heterogeneous operator is a named method"),
    rule(r"Isometry2|Similarity2", r"impl:Mul<UnitComplex>", "mul_unit_complex",
         "`iso * r`: Cairo's `Mul` is homogeneous, the heterogeneous operator is a named method"),
    rule(r"Isometry3|Similarity3", r"impl:Mul<UnitQuaternion>", "mul_unit_quaternion",
         "`iso * r`: Cairo's `Mul` is homogeneous, the heterogeneous operator is a named method"),
    rule(r".*", r"impl:RelativeEq", "relative_eq", "tolerance in ulp (DESIGN D3)"),
    rule(r".*", r"impl:UlpsEq", "ulps_eq", "tolerance in ulp (DESIGN D3)"),
    rule(r"Quaternion", r"impl:Mul<Quaternion> for T", "scale",
         "Cairo-imposed: heterogeneous operator (`k * q` is `q.scale(k)`)"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:Mul<Isometry>", "mul_isometry", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:Mul<Rotation>", "mul_rotation", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:Mul<Similarity>", "mul_similarity", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:Mul<Translation>", "mul_translation", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitQuaternion", r"impl:Div<Isometry>", "div_isometry", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:Div<Rotation>", "div_rotation", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitQuaternion", r"impl:Div<Similarity>", "div_similarity", "Cairo-imposed: heterogeneous operator"),
    rule(r"Quaternion|UnitQuaternion|UnitComplex",
         r"impl:SubsetOf<(?:Quaternion|UnitQuaternion|UnitComplex)>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:SubsetOf<Rotation>", "to_rotation_matrix",
         "Cairo-imposed: `nalgebra::convert` into a rotation matrix is `to_rotation_matrix`"),
    rule(r"UnitQuaternion|UnitComplex", r"impl:SubsetOf<Matrix>", "to_homogeneous",
         "Cairo-imposed: `nalgebra::convert` into a matrix is `to_homogeneous`"),
    rule(r"UnitQuaternion", r"impl:SubsetOf<Isometry>", "Isometry3::impl:From<UnitQuaternion>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"UnitQuaternion", r"impl:SubsetOf<Similarity>",
         "Similarity3::impl:From<UnitQuaternion>", "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"UnitComplex", r"impl:SubsetOf<Isometry>", "Isometry2::impl:From<UnitComplex>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"UnitComplex", r"impl:SubsetOf<Similarity>", "Similarity2::impl:From<UnitComplex>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    # WP 8.4-P09a: Rotation, Translation, Point completion.
    rule(r"AbstractRotation", r"trait:AbstractRotation", "AbstractRotation::trait:AbstractRotation",
         "a Cairo trait generic over the rotation type (associated `Vector` / `Point` types)"),
    rule(r"Point", r"impl:Mul<Point> for T", "scale",
         "Cairo-imposed: heterogeneous operator (`k * p` is `p.scale(k)`)"),
    rule(r"Point", r"impl:Bounded", "max_value",
         "num's `Bounded::min_value` / `max_value` as methods (Cairo's `Bounded` holds constants)"),
    rule(r"Point|Rotation|Translation|Scale", r"impl:SubsetOf<Matrix>", "to_homogeneous",
         "Cairo-imposed: `nalgebra::convert` into a matrix / homogeneous vector is `to_homogeneous`"),
    rule(r"Point", r"impl:SubsetOf<Point>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"Rotation", r"impl:SubsetOf<Rotation>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"Translation", r"impl:SubsetOf<Translation>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"Rotation", r"impl:SubsetOf<Isometry>", "Isometry3::impl:From<Rotation>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (`Isometry2` likewise)"),
    rule(r"Rotation", r"impl:SubsetOf<Similarity>", "Similarity3::impl:From<Rotation>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (`Similarity2` likewise)"),
    rule(r"Translation", r"impl:SubsetOf<Isometry>", "Isometry3::impl:From<Translation>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (`Isometry2` likewise)"),
    rule(r"Translation", r"impl:SubsetOf<Similarity>", "Similarity3::impl:From<Translation>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (`Similarity2` likewise)"),
    rule(r"Rotation2", r"impl:Mul<UnitComplex>", "mul_unit_complex",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Rotation2", r"impl:Div<UnitComplex>", "div_unit_complex",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Rotation2", r"impl:SubsetOf<UnitComplex>", "UnitComplex::impl:From<Rotation>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"Rotation3", r"impl:Mul<UnitQuaternion>", "mul_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Rotation3", r"impl:Div<UnitQuaternion>", "div_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Rotation3", r"impl:SubsetOf<UnitQuaternion>", "UnitQuaternion::impl:From<Rotation>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"Translation", r"impl:Mul<Isometry>", "mul_isometry",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Translation", r"impl:Mul<Similarity>", "mul_similarity",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Translation", r"impl:Mul<UnitComplex>", "Translation2::mul_unit_complex",
         "Cairo-imposed: heterogeneous operator (2D only)"),
    rule(r"Translation", r"impl:Mul<UnitQuaternion>", "Translation3::mul_unit_quaternion",
         "Cairo-imposed: heterogeneous operator (3D only)"),
    # WP 8.4-P09b: Isometry, Similarity completion (incl. `IsometryMatrix2/3`,
    # `SimilarityMatrix2/3`) and the rotation-matrix operators P09a deferred.
    rule(r"Isometry|Similarity|Translation", r"impl:Mul<Rotation>", "mul_rotation",
         "Cairo-imposed: heterogeneous operator (rotation-matrix instances only)"),
    rule(r"Isometry|Similarity", r"impl:Div<Rotation>", "div_rotation",
         "Cairo-imposed: heterogeneous operator (rotation-matrix instances only)"),
    rule(r"Isometry2|Similarity2", r"impl:Div<UnitComplex>", "div_unit_complex",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Isometry3|Similarity3", r"impl:Div<UnitQuaternion>", "div_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Isometry|Rotation", r"impl:Mul<Similarity>", "mul_similarity",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Isometry|Rotation", r"impl:Div<Similarity>", "div_similarity",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Similarity|Rotation", r"impl:Mul<Isometry>", "mul_isometry",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Similarity|Rotation", r"impl:Div<Isometry>", "div_isometry",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Rotation", r"impl:Mul<Translation>", "mul_translation",
         "Cairo-imposed: heterogeneous operator (output `IsometryMatrix2/3`)"),
    rule(r"Isometry|Similarity", r"impl:SubsetOf<Matrix>", "to_homogeneous",
         "Cairo-imposed: `nalgebra::convert` into a matrix is `to_homogeneous`"),
    rule(r"Isometry", r"impl:SubsetOf<Isometry>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"Similarity", r"impl:SubsetOf<Similarity>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"Isometry", r"impl:SubsetOf<Similarity>", "Similarity3::impl:From<Isometry>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (`Similarity2`, `SimilarityMatrix2/3` "
         "likewise)"),
    # WP 8.4-P10: Scale and Reflection.
    rule(r"Scale", r"impl:Mul<Matrix>", "mul_vector",
         "Cairo-imposed: heterogeneous operator (`s * v` on a column vector is `s.mul_vector(v)`)"),
    rule(r"Scale", r"impl:SubsetOf<Scale>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    # WP 8.4-P12: DualQuaternion, UnitDualQuaternion.
    rule(r"DualQuaternion", r"impl:Mul<T>|impl:Mul<DualQuaternion> for T", "scale",
         "Cairo-imposed: heterogeneous operator (`dq * k` / `k * dq` is `dq.scale(k)`)"),
    rule(r"DualQuaternion", r"impl:Div<T>", "unscale",
         "Cairo-imposed: heterogeneous operator (`dq / k` is `dq.unscale(k)`)"),
    rule(r"DualQuaternion|UnitDualQuaternion",
         r"impl:SubsetOf<(?:DualQuaternion|UnitDualQuaternion)>", "cast",
         "Cairo-imposed: the scalar conversion behind `SubsetOf` is `cast`"),
    rule(r"DualQuaternion|Isometry3|Translation|UnitQuaternion", r"impl:Mul<UnitDualQuaternion>",
         "mul_unit_dual_quaternion", "Cairo-imposed: heterogeneous operator"),
    rule(r"DualQuaternion|Isometry3|Translation|UnitQuaternion", r"impl:Div<UnitDualQuaternion>",
         "div_unit_dual_quaternion", "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Mul<DualQuaternion>", "mul_dual_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Mul<Isometry>", "mul_isometry",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Div<Isometry>", "div_isometry",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Mul<Translation>", "mul_translation",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Div<Translation>", "div_translation",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Mul<UnitQuaternion>", "mul_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Div<UnitQuaternion>", "div_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"UnitDualQuaternion", r"impl:Mul<Point>", "transform_point",
         "heterogeneous operators are named methods"),
    rule(r"UnitDualQuaternion", r"impl:Mul<Matrix>", "transform_vector",
         "heterogeneous operators are named methods"),
    rule(r"UnitDualQuaternion", r"impl:Mul<Unit<Matrix>>", "transform_unit_vector",
         "heterogeneous operators are named methods"),
    rule(r"UnitDualQuaternion", r"impl:SubsetOf<Matrix>", "to_homogeneous",
         "Cairo-imposed: `nalgebra::convert` into a matrix is `to_homogeneous`"),
    rule(r"UnitDualQuaternion", r"impl:SubsetOf<Isometry>",
         "Isometry3::impl:From<UnitDualQuaternion>", "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"UnitDualQuaternion", r"impl:SubsetOf<Similarity>",
         "Similarity3::impl:From<UnitDualQuaternion>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"Isometry3", r"impl:SubsetOf<UnitDualQuaternion>",
         "UnitDualQuaternion::impl:From<Isometry>", "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"Rotation3", r"impl:SubsetOf<UnitDualQuaternion>",
         "UnitDualQuaternion::impl:From<Rotation>", "Cairo-imposed: `nalgebra::convert` is `Into`"),
    rule(r"Translation", r"impl:SubsetOf<UnitDualQuaternion>",
         "UnitDualQuaternion::impl:From<Translation>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (3D only)"),
    rule(r"UnitQuaternion", r"impl:SubsetOf<UnitDualQuaternion>",
         "UnitDualQuaternion::impl:From<UnitQuaternion>",
         "Cairo-imposed: `nalgebra::convert` is `Into`"),
    # WP 8.4-P11a: Transform, Affine, Projective. The products whose category depends on both
    # operands are the generic `TransformMul::mul_transform` / `TransformDiv::div_transform`.
    rule(r"Transform", r"impl:Mul<Point>", "transform_point",
         "heterogeneous operators are named methods"),
    rule(r"Transform", r"impl:Mul<Matrix>", "transform_vector",
         "heterogeneous operators are named methods"),
    rule(r"Transform|Isometry|Similarity|Rotation|Translation|UnitComplex|UnitQuaternion",
         r"impl:Mul<Transform>", "mul_transform",
         "Cairo-imposed: the output category depends on both operands (`TCategoryMul`), a method "
         "of the generic `TransformMul` (`*` stays on the same-category pairs)"),
    rule(r"Transform|Rotation|Translation|UnitQuaternion", r"impl:Div<Transform>",
         "div_transform",
         "Cairo-imposed: the output category depends on both operands, a method of the generic "
         "`TransformDiv` (`/` stays on the same-category pairs)"),
    rule(r"Transform", r"impl:Mul<Rotation>", "mul_rotation", "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Div<Rotation>", "div_rotation", "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Mul<UnitComplex>", "mul_unit_complex",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Mul<UnitQuaternion>", "mul_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Div<UnitQuaternion>", "div_unit_quaternion",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Mul<Translation>", "mul_translation",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Div<Translation>", "div_translation",
         "Cairo-imposed: heterogeneous operator"),
    rule(r"Transform", r"impl:Mul<Isometry>", "mul_isometry",
         "Cairo-imposed: heterogeneous operator (`Isometry2/3`; the rotation-matrix isometries "
         "through `Affine2/3::from`)"),
    rule(r"Transform", r"impl:Mul<Similarity>", "mul_similarity",
         "Cairo-imposed: heterogeneous operator (`Similarity2/3`; the rotation-matrix similarities "
         "through `Affine2/3::from`)"),
    rule(r"Transform", r"impl:SubsetOf<Matrix>", "to_homogeneous",
         "Cairo-imposed: `nalgebra::convert` into a matrix is `to_homogeneous` (`try_convert` "
         "from a matrix is `TryInto`)"),
    rule(r"Transform", r"impl:SubsetOf<Transform>", "Projective3::impl:From<Affine3>",
         "Cairo-imposed: `nalgebra::convert` is `Into` (every widening pair; `try_convert` is "
         "`TryInto`)"),
    rule(r"SquareMatrix", r"impl:From<Transform>", "Matrix4::impl:From<Affine3>",
         "`Into` on the six transform types (`Matrix3` / `Matrix4`)"),
    *(rule(owner, r"impl:SubsetOf<Transform>", f"{target}::impl:From<{owner}>",
           "Cairo-imposed: `nalgebra::convert` is `Into` (into the three categories)")
      for owner, target in (("Rotation", "Affine3"), ("UnitComplex", "Affine2"),
                            ("UnitQuaternion", "Affine3"), ("Translation", "Affine3"),
                            ("Isometry", "Affine3"), ("Similarity", "Affine3"),
                            ("Scale", "Affine3"), ("UnitDualQuaternion", "Affine3"))),
)

# Cairo-imposed forms (WP 8.0, owner's rule of 2026-09-24): public Cairo items that spell an
# upstream operator, field or `Deref` access that has no single upstream ITEM to rename (the
# upstream item is shared with another Cairo spelling, or is a field / a method reached through
# `Deref`). They are listed in their own section and are not counted as extras.  (owner, rendered
# item, upstream spelling, reason) — fullmatch regexes on the Cairo owner and rendered item.
CAIRO_FORMS = (
    (r"Point[1-6]", r"coords", "`p.coords`",
     "an upstream public field; Cairo's points store `x, y(, z)` as fields (upstream's `Deref` "
     "view), so the vector is a method"),
    (r"Unit", r"dot|scale", "`u.dot(&w)`, `u * k`",
     "`Vector` methods reached through upstream's `Deref<Target = Vector>`"),
    (r"UnitComplex", r"re|im", "`c.re`, `c.im`",
     "the fields of upstream's `Complex`, reached through `Deref`"),
    (r"UnitQuaternion", r"dot|imag|scalar", "`q.dot(&r)`, `q.imag()`, `q.scalar()`",
     "`Quaternion` methods reached through upstream's `Deref<Target = Quaternion>`"),
    (r"Point[1-6]", r"min_value", "`<Point as Bounded>::min_value()`",
     "the second method of upstream's `Bounded` impl (`RENAMES` maps the impl to `max_value`)"),
    (r"Isometry2|Similarity2", r"impl:From<(?:Rotation|Translation)>", "`convert(r)` / `convert(t)`",
     "the 2D side of upstream's `SubsetOf<Isometry | Similarity>` (`RENAMES` points at the 3D impls)"),
    (r"Isometry[23]|IsometryMatrix[23]", r"impl:From<Isometry>", "`convert(iso)`",
     "the change of rotation representation (unit complex / quaternion <-> matrix) of upstream's "
     "`SubsetOf<Isometry>` (`RENAMES` maps its scalar side to `cast`)"),
    (r"Similarity[23]|SimilarityMatrix[23]", r"impl:From<Similarity>", "`convert(sim)`",
     "the change of rotation representation (unit complex / quaternion <-> matrix) of upstream's "
     "`SubsetOf<Similarity>` (`RENAMES` maps its scalar side to `cast`)"),
    (r"Similarity2|SimilarityMatrix[23]", r"impl:From<Isometry>", "`convert(iso)`",
     "the other instances of upstream's generic `SubsetOf<Similarity> for Isometry` (`RENAMES` "
     "points at `Similarity3`)"),
    # WP 8.4-P11a.
    (r"(?:Affine|Projective|Transform)[23]",
     r"impl:(?:Mul|Div)<(?:Affine|Projective|Transform)[23]>", "`a * b`, `a / b`",
     "the same-category instances of upstream's `Mul<Transform>` / `Div<Transform>` (`RENAMES` "
     "maps the items to `mul_transform` / `div_transform`, every pair of categories)"),
    (r"(?:Projective|Transform)[23]", r"impl:From<(?:Affine|Projective)[23]>", "`convert(t)`",
     "the widening instances of upstream's `SubsetOf<Transform> for Transform` (`RENAMES` points "
     "at `Projective3`); also `set_category`"),
    (r"(?:Affine|Projective|Transform)[23]",
     r"impl:TryFrom<(?:Matrix|Projective[23]|Transform[23])>", "`try_convert(m)`",
     "the checked side of upstream's `SubsetOf<Matrix>` / `SubsetOf<Transform>` (`is_in_subset`, "
     "`check_homogeneous_invariants`)"),
    (r"(?:Affine|Projective|Transform)[23]",
     r"impl:From<(?:Rotation|UnitComplex|UnitQuaternion|Translation|Isometry|Similarity|Scale|"
     r"UnitDualQuaternion)>", "`convert(g)`",
     "the instances of upstream's `SubsetOf<Transform>` for the geometry types, into every "
     "category (`RENAMES` points at `Affine2/3`)"),
    (r"Matrix[34]", r"impl:From<(?:Affine|Projective|Transform)[23]>", "`t.into()`",
     "the instances of upstream's `From<Transform> for OMatrix` (`RENAMES` points at "
     "`Matrix4::From<Affine3>`)"),
)


def cairo_form(item: Item) -> tuple[str, str] | None:
    r = rendered(item)
    for owner, name, spelling, reason in CAIRO_FORMS:
        if re.fullmatch(owner, item.owner) and re.fullmatch(name, r):
            return spelling, reason
    return None


# Names of `simba::Real` methods that come from simba-rs supertraits outside `RealField` /
# `ComplexField` / `Field` (`parse_simba` reads only those three): `num::Zero::zero`,
# `num::One::one`, `approx::AbsDiffEq::default_epsilon`.
SIMBA_SUPERTRAIT_NAMES = {
    "zero": "num::Zero", "one": "num::One", "default_epsilon": "approx::AbsDiffEq",
}

# The one documented exception of the scalar layer (owner ruling 2026-09-24): scalar items
# without a simba-rs name. (owner, rendered item) fullmatch regexes; the rationale is shared.
SCALAR_KERNELS_WHY = (
    "owner ruling 2026-09-24: fused scalar kernels, the numeric contract of the stack (DESIGN "
    "D2-D3); confined to the scalar trait layer, like fixed-cairo's `fixed::wide`. Constants "
    "without a simba-rs name stay associated constants because Cairo has no `f64` literal "
    "conversion such as upstream's `crate::convert(0.5)`"
)
SCALAR_KERNELS = (
    (r"simba::Real",
     r"sum_prod[234]|diff_prod|sqr|norm[234]|norm_squared[234]|lerp|abs_diff_eq|"
     r"div|div(?:3|4|5|6|9|16)|rem|from_int|from_ratio|"
     r"wide_(?:zero|add|sub|add_prod|sub_prod|rescale|sqrt|mul_scalar)|"
     r"const:(?:NEG_ONE|TWO|HALF|FRAC_1_SQRT_2)"),
)


def scalar_kernel(item: Item) -> bool:
    r = rendered(item)
    return any(re.fullmatch(o, item.owner) and re.fullmatch(n, r) for o, n in SCALAR_KERNELS)


def simba_named(item: Item, simba: list[str]) -> str | None:
    """The simba-rs trait a scalar-layer method is named after (`RealField::pi`...), if any."""
    if not item.owner.startswith("simba::") or item.kind != "method":
        return None
    if item.name in SIMBA_SUPERTRAIT_NAMES:
        return SIMBA_SUPERTRAIT_NAMES[item.name]
    return "RealField / ComplexField / Field" if item.name in simba else None


def exclude(owner: str, item: str, reason: str) -> Rule:
    assert reason in EXCLUSIONS, reason
    return rule(owner, item, reason)


# Exclusions (closed reason list).  Applied only to items without a direct Cairo match.
EXCLUDE = (
    exclude(r".*ShapeConstraint.*|MatrixIndex|MatrixIndexMut", r".*", "generic-dim"),
    exclude(r".*", r"type:MatrixVec", "generic-dim"),
    exclude(r"CustomPhantom", r".*", "glue"),
    exclude(r"(?:mut )?\[T\]", r".*", "borrow"),
    exclude(r"MatrixValueTree", r".*", "random"),
    exclude(r".*", r"unsafe-method:.*", "unsafe"),
    exclude(r".*", r"impl:(?:Debug|Display|LowerExp)", "fmt"),
    exclude(r".*", r"impl:(?:Pod|Zeroable|Archive|ShaderType)", "glue"),
    exclude(r".*", r"impl:(?:Distribution|Arbitrary)", "random"),
    exclude(r".*", r"(?:new_random|from_distribution|from_distribution_generic)", "random"),
    exclude(r"RandomOrthogonal|RandomSDP|MatrixStrategy|MatrixParameters|DimRange|"
            r"nalgebra::proptest|nalgebra::debug", r".*", "random"),
    exclude(r".*", r"(?:simd_\w+|\w+_simd|type:Simd\w*|trait:Simd\w*)", "simd"),
    exclude(r"Quaternion|UnitQuaternion|UnitComplex|Point|Rotation|Translation|Scale|Isometry|Similarity", r"impl:From<\[(?:Quaternion|UnitQuaternion|UnitComplex|Point|Rotation|Translation|Scale|Isometry|Similarity); N\]>", "simd"),
    exclude(r"Matrix|Unit<Vector>", r"impl:From<\[(?:Matrix|Unit<Matrix>); N\]>", "simd"),
    exclude(r"Matrix", r"impl:From<Matrix>", "borrow"),
    exclude(r"Matrix|Vector", r"type:(?:MatrixComponentOp|MatrixCross|MatrixSum|VectorSum)", "generic-dim"),
    exclude(r"nalgebra::base", r"reject|reject_rand", "random"),
    exclude(r"ParColumnIter|ParColumnIterMut|ColumnIntoIter", r".*", "rayon"),
    exclude(r".*", r"par_\w+", "rayon"),
    exclude(r".*", r"(?:as_ptr|as_mut_ptr|as_slice_unchecked|as_mut_slice_unchecked|"
            r"assume_init|uninit\w*|\w+_uninit|data_mut|ptr_\w+)", "unsafe"),
    exclude(r"Init|Uninit|UninitMatrix|InitStatus|RawIter", r".*", "unsafe"),
    exclude(r".*", r"impl:(?:AsRef|AsMut|Borrow|BorrowMut|DerefMut)", "borrow"),
    exclude(r".*", r"impl:IndexMut<.*>", "borrow"),
    exclude(r".*", r"(?:as_slice|as_mut_slice|get_mut|index_mut|\w+_mut_unchecked|"
            r"iter_mut|column_iter_mut|row_iter_mut|as_mut|as_mut_unchecked|"
            r"coords_mut|vector_mut|matrix_mut\w*|as_mut_\w+)", "borrow"),
    # `&mut` / `&T -> &Unit<T>` accessors (WP 8.4-R, owner-validated reason `borrow`): a Cairo
    # value is `Copy` and passed by value, so `Quaternion::as_vector_mut` (`&mut Vector4`),
    # `Vector1::as_scalar_mut` (`&mut T`) and `Unit::from_ref_unchecked` (`&T -> &Unit<T>`) have
    # no Cairo counterpart (`*_mut` in-place forms are `ref self` methods).
    exclude(r"Quaternion|Matrix1|Vector1|Unit", r"(?:as_vector_mut|as_scalar_mut|from_ref_unchecked)",
            "borrow"),
    exclude(r"Dyn|Const|U\d+|Dim|DimName|DimAdd|DimSub|DimMul|DimDiv|DimMin|DimMax|"
            r"DimNameAdd|DimNameSub|DimNameMul|DimNameDiv|ToTypenum|ToConst|IsDynamic|"
            r"IsNotStaticOne|ArrayStorage|ArrayStorageVisitor|VecStorage|Storage|StorageMut|"
            r"RawStorage|RawStorageMut|ContiguousStorage|ContiguousStorageMut|IsContiguous|"
            r"Owned|Allocator|DefaultAllocator|Reallocator|SameShapeAllocator|"
            r"SameShapeVectorAllocator|ShapeConstraint|SameNumberOfRows|SameNumberOfColumns|"
            r"SameDimension|AreMultipliable|DimEq|SameShapeStorage|SameShapeR|SameShapeC|"
            r"MatrixView|ViewStorage|ViewStorageMut|SliceStorage|SliceStorageMut|"
            r"MatrixIter|MatrixIterMut|ColumnIter|ColumnIterMut|RowIter|RowIterMut|"
            r"MatrixComponentOp|MatrixSum|MatrixCross|MatrixVec|MatrixMN|MatrixN|VectorN|"
            r"CStride|RStride|Scalar|Norm|DimRange|OwnedUninit", r".*", "generic-dim"),
    exclude(r".*", r"(?:type|trait):(?:Dim\w*|Const|Dyn|Dynamic|U\d+|\w*Storage\w*|"
            r"\w*Allocator|ShapeConstraint|SameNumberOf\w+|SameDimension|AreMultipliable|"
            r"DimEq|Owned\w*|OMatrix|SMatrix|OVector|SVector|RowOVector|RowSVector|OPoint|"
            r"MatrixN|MatrixMN|VectorN|UninitMatrix|UninitVector|ToTypenum|ToConst|"
            r"IsContiguous|Scalar|Matrix|Vector|Point|SquareMatrix|TCategory\w*|"
            r"SuperTCategoryOf|SubTCategoryOf|TAffine|TGeneral|TProjective|"
            r"Reallocator|InitStatus|RowVector|Unit|Transform|Rotation|Translation|Scale|"
            r"Isometry|Similarity|Reflection|MatrixView\w*|\w*View\w*|\w*Slice\w*)",
            "generic-dim"),
    exclude(r".*", r"(?:\w*_generic\w*|into_owned|clone_owned|clone_owned_sum|"
            r"from_data|from_array_storage|from_vec_storage|shape_generic|data|"
            r"as_view|as_view_mut|as_slice_view|reshape_generic|\w+_with_steps?|"
            r"generic_\w+|from_slice_with_strides\w*|from_slice\w*_generic\w*|"
            r"from_slices?_with_strides\w*|with_step|strides|shape_generic|"
            r"into_view|convert_ref|into_mat|into_owned_sum|legacy_\w+|"
            r"columns_range_pair\w*|rows_range_pair\w*|view_range\w*|slice_range\w*|"
            r"fixed_view\w*_mut|fixed_rows_mut|fixed_columns_mut|rows_mut|columns_mut|"
            r"row_mut|column_mut|row_part_mut|column_part_mut|view_mut|slice_mut|"
            r"fixed_slice_mut|\w+_range_mut|\w*_mut_with_step\w*|\w+_with_steps?_mut)",
            "generic-dim"),
    exclude(r".*", r"impl:(?:IntoIterator|FromIterator|Extend|Deref)", "generic-dim"),
    exclude(r".*", r"(?:iter|column_iter|row_iter|into_iter)", "generic-dim"),
    exclude(r"CsVecStorage|CsStorage|CsStorageMut|CsStorageIter|CsStorageIterMut|"
            r"ColumnEntries", r".*", "generic-dim"),
    exclude(r".*", r"(?:type|trait):(?:DVec\d|DMat\d|DQuat|DAffine\d|Vec3A|Mat3A|Affine3A|"
            r"I64Vec\d|U64Vec\d)", "interop"),
    exclude(r".*", r"impl:.*\b(?:DVec\d|DMat\d|DQuat|DAffine\d|Vec3A|Mat3A|Affine3A|BVec\dA|"
            r"I64Vec\d|U64Vec\d|I16Vec\d|U16Vec\d|I8Vec\d|U8Vec\d)\b.*", "interop"),
    exclude(r"DVec\d|DMat\d|DQuat|DAffine\d|Vec3A|Mat3A|Affine3A", r".*", "interop"),
    exclude(r".*", r"impl:.*\bmint\b.*", "interop"),
    exclude(r".*", r"(?:type|trait):(?:ComplexField|RealField|Field|ClosedAdd\w*|"
            r"ClosedSub\w*|ClosedMul\w*|ClosedDiv\w*|ClosedNeg|Complex)", "interop"),
)

STATUSES = ("ported", "partial", "missing", "excluded")


@dataclass
class Result:
    status: str
    cairo: str = ""
    detail: str = ""


def find_rule(rules, item: Item) -> Rule | None:
    r = rendered(item)
    for candidate in rules:
        if candidate.owner.fullmatch(item.owner) and candidate.item.fullmatch(r):
            return candidate
    return None


def compress(types: list[str]) -> str:
    """'Vector2/3/4' for ['Vector2', 'Vector3', 'Vector4']."""
    groups: dict[str, list[str]] = {}
    for t in types:
        m = re.fullmatch(r"(.*?)(\d+)", t)
        stem, num = (m.group(1), m.group(2)) if m else (t, "")
        groups.setdefault(stem, []).append(num)
    return ", ".join(stem + "/".join(nums) for stem, nums in groups.items())


def classify(rust: list[Item], cairo: list[Item]) -> tuple[dict[Item, Result], list[Item]]:
    by_key = {item.key: item for item in cairo}
    consumed: set[tuple[str, str, str]] = set()
    results: dict[Item, Result] = {}

    def lookup(owners: list[str], kind: str, name: str) -> list[str]:
        found = []
        for o in owners:
            if (o, kind, name) in by_key:
                found.append(o)
                consumed.add((o, kind, name))
        return found

    for item in rust:
        candidates = OWNER_CANDIDATES.get(item.owner, [])
        kind = "method" if item.kind == "unsafe-method" else item.kind
        cairo_name, cairo_kind, note, owners = item.name, kind, "", candidates
        if item.kind == "type":
            present = [item.name] if (item.name, "type", item.name) in by_key else []
            if present:
                consumed.add((item.name, "type", item.name))
                results[item] = Result("ported", item.name)
                continue
            if item.name == item.owner and candidates:
                # A generic upstream struct stands for its concrete Cairo instances.
                present = [c for c in candidates if (c, "type", c) in by_key]
                if present:
                    for c in present:
                        consumed.add((c, "type", c))
                    results[item] = Result("ported", compress(present),
                                           "generic upstream type, concrete Cairo types")
                    continue
        else:
            present = lookup(candidates, kind, item.name) if candidates else []
            allowed = DIM_ONLY.get(item.name)
            if present and [o for o in candidates if (allowed is None or o in allowed)
                            and o not in present] and find_rule(RENAMES, item):
                # A partial direct match (`Mul<Matrix>`: the square `*`) gives way to a rename
                # that covers every candidate (`MatrixMul::mul_mat`).
                present = []
            if not present:
                ren = find_rule(RENAMES, item)
                if ren:
                    target = ren.item.sub(ren.value, rendered(item))
                    if "::" in target:
                        o, target = target.split("::", 1)
                        owners = [o]
                    cairo_kind, cairo_name = ("method", target) if ":" not in target else \
                        tuple(target.split(":", 1))
                    note = f"renamed `{cairo_name}`: {ren.note}" if cairo_name != item.name \
                        else ren.note
                    if cairo_name == "fields":
                        present = [o for o in owners if (o, "type", o) in by_key]
                    else:
                        present = lookup(owners, cairo_kind, cairo_name)
            if present:
                allowed = DIM_ONLY.get(item.name)
                required = [o for o in owners if allowed is None or o in allowed]
                lacking = [o for o in required if o not in present]
                path = f"{compress(present)}::{cairo_name}" if cairo_kind == "method" else \
                    f"{compress(present)} ({cairo_kind} `{cairo_name}`)"
                if cairo_name == "fields":
                    path = f"{compress(present)} fields"
                if lacking:
                    results[item] = Result("partial", path,
                                           f"not on {compress(lacking)}" + (f"; {note}" if note else ""))
                else:
                    results[item] = Result("ported", path, note)
                continue
        ex = find_rule(EXCLUDE, item)
        if ex:
            results[item] = Result("excluded", "", ex.value)
            continue
        results[item] = Result("missing", "", "deprecated upstream" if item.flags else "")
    standing = {c for cs in OWNER_CANDIDATES.values() for c in cs}
    extras = [item for item in cairo if item.key not in consumed and item.kind != "trait"
              and not (item.kind == "type" and item.name in standing)]
    return results, extras


# --------------------------------------------------------------------------------------------
# Proposed work packages for the missing / partial items
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkPackage:
    key: str
    title: str
    tier: str  # mechanical / standard numerics / hard numerics
    depends: str
    scope: str
    match: tuple[tuple[str, str, str], ...]  # (owner, rendered item, source) fullmatch regexes


def wp(key, title, tier, depends, scope, *match) -> WorkPackage:
    return WorkPackage(key, title, tier, depends, scope, tuple(match))


STATIC = r"Matrix|SquareMatrix|Vector|Matrix[1-6](?:x[1-6])?|Vector[1-6]|Unit|Unit<Vector>"
DYNAMIC_ALIAS = r"type:(?:DMatrix|DVector|RowDVector|Matrix[1-6]xX|MatrixXx[1-6])"

# Assignment is first-match in ASSIGN_ORDER; the rendered table is ordered by `key` (dependency
# order).  Every missing / partial item must land in a package (checked by `render`).
WORK_PACKAGES = (
    wp("P01", "Rectangular and remaining static shapes", "mechanical", "—",
       "`Matrix1`, `Matrix5`, every `MatrixRxC` (1..6), `Vector1`, `Vector5`, `RowVector1..6`, "
       "`UnitVector1..6` with the base API of the square types (the item count is the types; "
       "each type carries the P02-P05 surface)",
       (r".*", r"type:(?:Matrix[1-6](?:x[1-6])?|Vector[1-6]|RowVector[1-6]|UnitVector[1-6])", r".*"),
       (r"Matrix1|Vector1", r".*", r".*")),
    wp("P13", "DMatrix / DVector core", "standard numerics", "P01-P05 (API to mirror)",
       "D5 `Span`-backed `DMatrix` / `DVector` / `RowDVector` and the `MatrixXxN` / `MatrixNxX` "
       "aliases: construction, element-wise ops, resize / insert / remove, `from_vec`, "
       "`Sum`, the `dmatrix!` / `dvector!` macros",
       (r".*", DYNAMIC_ALIAS, r".*"),
       (r"DMatrix|DVector|RowDVector", r".*", r".*"),
       (r"nalgebra", r"macro:d(?:matrix|vector)!", r".*"),
       (STATIC, r"(?:resize\w*|insert_\w+|remove_\w+|from_vec|from_iterator|from_row_iterator|"
        r"is_empty|len|compress_\w+|select_\w+|impl:From<Vec>|impl:Sum\w*)", r".*")),
    wp("P04", "Swizzles", "mechanical", "P01 (Vector2/3 results)",
       "`xx()` .. `zzz()` on vectors and points (`base/swizzle.rs`, `geometry/swizzle.rs`)",
       (r".*", r".*", r".*swizzle\.rs")),
    wp("P03", "Functional and in-place variants", "mechanical", "—",
       "`map`, `zip_*`, `fold*`, `apply*`, `fill*`, `copy_from`, `swap*`, `set_*`, and the "
       "`*_mut` / `*_assign` / `*_to` in-place forms of existing operations (Cairo `ref self`)",
       (STATIC + r"|Point\d?|Quaternion|UnitQuaternion|UnitComplex|Rotation\d?|Translation\d?|"
        r"Isometry\d?|Similarity\d?",
        r"(?:map\w*|zip_\w+|fold\w*|apply\w*|fill\w*|copy_from\w*|tr_copy_from|swap\w*|set_\w+|"
        r"\w+_mut|\w+_assign|\w*(?:mul|add|sub|adjoint|transpose)_to|neg_mut|impl:(?:Add|Sub|Mul|Div)Assign<.*>)", r"base/.*|"
        r"geometry/(?:point|quaternion|unit_complex|rotation|translation|isometry|similarity)"
        r"\w*\.rs")),
    wp("P05", "Rows, columns, blocks and edition", "mechanical", "P01",
       "`row` / `column` / `rows` / `columns` / `fixed_rows` / `fixed_columns` / `view` / "
       "`fixed_view` returning owned values, `*_range`, `set_row` / `set_column`, "
       "`upper_triangle` / `lower_triangle`, `fixed_resize`, `transpose` on vectors (row vectors)",
       (STATIC, r".*", r"base/(?:matrix_view|edition)\.rs"),
       (STATIC, r"(?:row\w*|column\w*|upper_triangle|lower_triangle|fixed_resize|transpose|"
        r"tr_mul|kronecker|ncols|nrows|shape|vector_to_matrix_index|impl:Mul<Matrix>|"
        r"impl:MulAssign<Matrix>|from_rows|from_columns|is_identity|from_diagonal_element|"
        r"is_square|is_orthogonal|is_special_orthogonal|is_invertible)", r".*")),
    wp("P02", "Static base completion (Vector / Matrix 2-6)", "mechanical", "—",
       "the operations upstream has on every `Matrix` that nalgebra-cairo only has on some types "
       "(`norm`, `normalize`, `component_*`, `inf` / `sup`, `amax`..., `scale`, `dot` on "
       "matrices, `Vector6` gaps), construction (`from_fn`, `from_row_slice`, arrays, "
       "`from_partial_diagonal`), conversions, `Index`, `RelativeEq` / `UlpsEq`, `PartialOrd`, "
       "`Sum`, scalar-on-the-left `*`, `Unit` gaps, `cast`",
       (STATIC, r".*", r"base/(?:norm|componentwise|min_max|interpolation|unit|construction|"
        r"conversion|indexing|ops|properties|matrix|helper|coordinates)\.rs"),
       (STATIC, r".*", r"geometry/.*"),
       (r"UnitVector\d|RowSVector|nalgebra::base|EuclideanNorm|LpNorm|OneNorm|UniformNorm|"
        r"Normed", r".*", r".*")),
    wp("P06", "Statistics and BLAS-like kernels", "standard numerics", "P01, P05",
       "`sum` / `product` / `mean` / `variance` per row and column, `gemm` / `gemv` / `ger` / "
       "`axpy` / `cmpy` / `cdpy` / `quadform*`, `dotc` / `tr_dot` / `ad_mul` (fused kernels, "
       "no loops)",
       (STATIC, r".*", r"base/(?:statistics|blas)\.rs")),
    wp("P07", "Homogeneous / computer-graphics helpers", "standard numerics", "P01",
       "`base/cg.rs`: `new_scaling`, `new_translation`, `new_rotation*`, `look_at_*`, "
       "`new_perspective`, `new_orthographic`, `append_*` / `prepend_*`, `transform_point` / "
       "`transform_vector` on `Matrix3` / `Matrix4`, `to_homogeneous` on square matrices",
       (STATIC, r".*", r"base/cg\.rs")),
    wp("P08", "Quaternion, UnitQuaternion, UnitComplex completion", "standard numerics", "—",
       "quaternion transcendental functions (`exp`, `ln`, `powf`, `sqrt`, trig), polar "
       "decomposition, `from_matrix` / `*_eps` constructors, `mean_of`, operator impls with "
       "rotations / isometries, `Index`, `RelativeEq`, `cast`",
       (r"Quaternion|UnitQuaternion|UnitComplex", r".*", r".*")),
    wp("P09a", "Rotation, Translation, Point completion", "mechanical", "P08",
       "cross-type operators (`Rotation * Translation`, `Translation * UnitQuaternion`...), "
       "`Point1/4/5/6`, `Translation1/4/5/6`, `Rotation3::new`, `from_matrix*`, "
       "`from_basis_unchecked`, `slerp` / `powf` on rotations, `cast`, `RelativeEq`",
       (r"Rotation\d?|Translation\d?|Point\d?|AbstractRotation", r".*", r"geometry/.*"),
       (r".*", r"type:(?:Point[1-6]|Translation[1-6]|Rotation[1-6])", r".*")),
    wp("P09b", "Isometry, Similarity completion (incl. rotation-matrix variants)", "mechanical",
       "P09a",
       "`IsometryMatrix2/3`, `SimilarityMatrix2/3`, cross-type operators (`Isometry / Rotation`, "
       "`Similarity * Translation`...), `look_at_lh`, `new_observer_frames`, "
       "`rotation_wrt_point`, `to_matrix`, `inverse_transform_unit_vector`, `cast`",
       (r"Isometry\w*|Similarity\w*", r".*", r"geometry/.*"),
       (r".*", r"type:(?:IsometryMatrix[23]|SimilarityMatrix[23])", r".*")),
    wp("P10", "Scale and Reflection", "mechanical", "P09a",
       "`Scale1..6` (non-uniform scaling, inverse, homogeneous form, operators) and "
       "`Reflection1..6` (`reflect`, `reflect_rows`...)",
       (r"Scale\d?|Reflection\d?", r".*", r".*")),
    wp("P11a", "Transform, Affine, Projective", "standard numerics", "P07, P09b",
       "`Transform<T, C, D>` with its categories (`TAffine`, `TProjective`, `TGeneral`) as "
       "concrete `Affine2/3`, `Projective2/3`, `Transform2/3`: inverse, operators with every "
       "geometry type, homogeneous conversions",
       (r"Transform", r".*", r".*"),
       (r".*", r".*", r"geometry/transform\w*\.rs")),
    wp("P11b", "Perspective3, Orthographic3", "standard numerics", "P07",
       "camera projections: construction, accessors / setters, `project_*` / `unproject_point`, "
       "conversions to `Matrix4` / `Projective3`",
       (r"Perspective3|Orthographic3", r".*", r".*"),
       (r".*", r".*", r"geometry/(?:perspective|orthographic)\.rs")),
    wp("P12", "DualQuaternion, UnitDualQuaternion", "standard numerics", "P08, P09b",
       "dual quaternions: construction, ops, `sclerp`, conversion to / from isometries",
       (r"DualQuaternion|UnitDualQuaternion", r".*", r".*"),
       (r".*", r".*", r"geometry/dual_quaternion\w*\.rs")),
    wp("P14", "Decomposition API completion and triangular solves", "standard numerics",
       "P01, P05",
       "`solve_*_triangular*` / `tr_solve_*` / `ad_solve_*` on square matrices, the missing "
       "members of `LU` / `QR` / `Cholesky` / `SVD` / `SymmetricEigen` / `UDU` / "
       "`PermutationSequence` (`unpack`, `solve_mut`, `rank_one_update`, `try_new`, "
       "`*_unordered`...), `Matrix::lu()` .. `svd()` on the missing sizes, `rank`, `polar`",
       (r".*", r".*", r"linalg/(?:solve|lu|qr|cholesky|svd|svd2|svd3|symmetric_eigen|udu|"
        r"permutation_sequence|inverse|decomposition|givens|householder)\.rs")),
    wp("P15", "Full-pivot LU, column-pivot QR, LBLᵀ", "standard numerics", "P14",
       "`FullPivLU`, `ColPivQR`, `LBLT` (pivoted factorizations, static sizes first)",
       (r".*", r".*", r"linalg/(?:full_piv_lu|col_piv_qr|lblt)\.rs"),
       (r".*", r"(?:full_piv_lu|col_piv_qr|lblt)", r"linalg/decomposition\.rs")),
    wp("P16", "Schur, Hessenberg, Bidiagonal, tridiagonal, general eigen", "hard numerics",
       "P14",
       "`Schur` (real Schur form, `eigenvalues`, `complex_eigenvalues`), `Hessenberg`, "
       "`Bidiagonal`, `SymmetricTridiagonal`, `Eigen`, balancing, Wilkinson shift: iterative "
       "algorithms upstream, need a fixed-cost formulation (no convergence loop, AGENTS.md)",
       (r".*", r".*", r"linalg/(?:schur|hessenberg|bidiagonal|symmetric_tridiagonal|eigen|"
        r"balancing|mod)\.rs"),
       (r".*", r"(?:bidiagonalize|hessenberg|schur|try_schur|symmetric_tridiagonalize|"
        r"eigenvalues|complex_eigenvalues)", r"linalg/decomposition\.rs|linalg/schur\.rs")),
    wp("P17", "Matrix exponential and power", "hard numerics", "P14, P16",
       "`exp` (Padé approximant with scaling and squaring) and `pow` / `pow_mut`",
       (r".*", r".*", r"linalg/(?:exp|pow)\.rs")),
    wp("P18", "Convolution", "mechanical", "P13",
       "`convolve_full` / `convolve_same` / `convolve_valid` on vectors",
       (r".*", r".*", r"linalg/convolution\.rs")),
    wp("P19", "glam-cairo conversions", "mechanical", "WP 6.2 (glam-cairo pin)",
       "`third_party/glam`: `From` / `Into` between nalgebra-cairo and glam-cairo "
       "(`Vec2/3/4`, `IVec*`, `UVec*`, `BVec*`, `Mat2/3/4`, `Quat`, `Affine2/3` through "
       "isometries); f64 / aligned variants are excluded (`interop`)",
       (r".*", r".*", r"third_party/glam/.*")),
    wp("P20", "Sparse matrices and Matrix Market I/O", "standard numerics", "P13",
       "legacy `nalgebra::sparse` (`CsMatrix`, `CsVector`, `CsCholesky`, triangular solves) and "
       "`nalgebra::io` Matrix Market parsing",
       (r".*", r".*", r"sparse/.*|io/.*")),
    wp("P21", "Crate-root functions and construction macros", "mechanical", "P01, P13",
       "`nalgebra::{distance, distance_squared, center, wrap, clamp, inf, sup, partial_*...}` "
       "and the `vector!` / `matrix!` / `point!` / `stack!` macros (Cairo declarative macros)",
       (r"nalgebra", r".*", r".*")),
)


# First-match order of the assignment (specific packages before the broad base ones).
ASSIGN_ORDER = ("P01", "P13", "P19", "P04", "P17", "P18", "P16", "P15", "P14", "P11a", "P11b",
                "P12", "P06", "P07", "P03", "P05", "P02", "P08", "P09a", "P09b", "P10", "P20",
                "P21")


def assign_wp(item: Item) -> str:
    r = rendered(item)
    by_key = {p.key: p for p in WORK_PACKAGES}
    assert sorted(ASSIGN_ORDER) == sorted(by_key), "ASSIGN_ORDER must list every package"
    for package in (by_key[k] for k in ASSIGN_ORDER):
        for owner, name, source in package.match:
            if re.fullmatch(owner, item.owner) and re.fullmatch(name, r) and \
                    re.fullmatch(source, item.source):
                return package.key
    return ""


# Documented rationale of the Cairo-only items ("neither more nor less": justify or remove).
EXTRA_RATIONALE = (
    (r"\w+", r"impl:(?:Copy|PartialEq|Serde|Default|Debug|Hash)",
     "Standard Cairo derive set (`Copy, Drop, PartialEq, Serde, Default, Debug, Hash`) on a "
     "type whose upstream counterpart lacks this trait."),
)


def extra_rationale(item: Item) -> str:
    r = rendered(item)
    for owner, name, text in EXTRA_RATIONALE:
        if re.fullmatch(owner, item.owner) and re.fullmatch(name, r):
            return text
    return "Undocumented: justify (doc comment + DESIGN) or remove before 0.1.0."


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

MODULE_ORDER = ("base", "geometry", "linalg", "sparse", "io", "third_party", "root",
                "proptest", "debug")


def anchor(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def md(value: str) -> str:
    return value.replace("|", "\\|")


def display(item: Item) -> str:
    return f"{item.kind} `{md(item.name)}`"


def pct(num: int, den: int) -> str:
    return "—" if den == 0 else f"{100.0 * num / den:.1f}%"


def dimension_grid(types: set[str]) -> list[str]:
    cols = ["1", "2", "3", "4", "5", "6", "X"]
    lines = ["| R \\ C | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for r in cols:
        cells = []
        for c in cols:
            if r == "X" and c == "X":
                name = "DMatrix"
            elif c == "1" and r == "X":
                name = "DVector"
            elif r == "1" and c == "X":
                name = "RowDVector"
            elif r == c:
                name = f"Matrix{r}"
            elif c == "1":
                name = f"Vector{r}"
            elif r == "1":
                name = f"RowVector{c}"
            elif r == "X":
                name = f"MatrixXx{c}"
            else:
                name = f"Matrix{r}x{c}"
            cells.append(f"**{name}**" if name in types else name.replace("Matrix", "M")
                         .replace("RowVector", "RV").replace("Vector", "V") + " ✗")
        lines.append(f"| **{r}** | " + " | ".join(cells) + " |")
    return lines


def render(rust: list[Item], cairo: list[Item], simba: list[str]) -> str:
    results, extras = classify(rust, cairo)
    forms = [i for i in extras if cairo_form(i)]
    extras = [i for i in extras if not cairo_form(i)]
    named = [i for i in extras if simba_named(i, simba)]
    extras = [i for i in extras if not simba_named(i, simba)]
    kernels = [i for i in extras if scalar_kernel(i)]
    extras = [i for i in extras if not scalar_kernel(i)]
    cairo_types = {i.name for i in cairo if i.kind == "type"}
    todo = [i for i in rust if results[i].status in ("missing", "partial")]
    wp_of = {i: assign_wp(i) for i in todo}
    unassigned = [i for i in todo if not wp_of[i]]
    if unassigned:
        raise SystemExit("items without a proposed work package: " +
                         ", ".join(f"{i.owner}::{rendered(i)} ({i.source})" for i in unassigned))

    counts = {m: {s: 0 for s in STATUSES} for m in MODULE_ORDER}
    for item in rust:
        counts[item.module][results[item].status] += 1
    total = {s: sum(counts[m][s] for m in MODULE_ORDER) for s in STATUSES}

    out = [
        f"# API parity with nalgebra-rs {VERSION}",
        "",
        "<!-- Generated by scripts/api_parity.py: do not edit by hand. -->",
        "",
        "Generated by `python3 scripts/api_parity.py` (`--check` fails when this file is stale; "
        "`--refresh --nalgebra-rs <checkout> --simba <checkout>` re-reads the Rust sources). The "
        "nalgebra-rs inventory is embedded at the end of this file, so neither the check nor the "
        "regeneration needs a Rust checkout.",
        "",
        "The coverage target of nalgebra-cairo 0.1.0 is **strictly nalgebra-rs's public API, "
        "neither more nor less**, with Rust semantics mapped onto a generic `T: Real` scalar "
        "(fixed-cairo's `fixed::Fixed`), static unrolled types and no loops in static code "
        "(AGENTS.md).",
        "",
        "How to read it:",
        "",
        "- an upstream **item** is `(owner, kind, name)`: the owner is the type family it is "
        "reachable from (`Matrix` = any `Matrix<T, R, C, S>`, `SquareMatrix`, `Vector`, "
        "`Matrix3` for `U3 x U3`-only impls, `Isometry`, `UnitQuaternion`, `LU`, ...) or the "
        "module of a free function; kinds are `type` (structs and the aliases users name), "
        "`trait`, `method` (inherent `pub fn`, associated functions included), `const`, "
        "`impl` (operator / conversion / comparison traits, types normalized to families: "
        "`Mul<Matrix>`, `From<[T; N]>`, `Into<[T; N]>`; reference and owned variants folded), "
        "`function`, `macro`;",
        "- an upstream owner maps onto the Cairo types listed in `OWNER_CANDIDATES` (e.g. "
        "`Matrix` → the 36 static shapes `Matrix1..6`, `MatrixRxC`, `Vector2..6`, `RowVector2..6`); "
        "dimension-restricted methods (`cross`, "
        "`perp`, `x_axis`...) only require the matching Cairo types (`DIM_ONLY`);",
        "- **ported**: same name on every candidate Cairo type, or a documented rename "
        "(`RENAMES`, shown in the detail column); **partial**: on some but not all candidate "
        "types (the missing ones are listed); **missing**: in the coverage target and not "
        "done — the default for anything without an explicit exclusion rule; **excluded**: "
        "one reason of the closed list below. When in doubt an item is `missing`: exclusions "
        "are the owner's decision, the rules only encode the brief's list.",
        "- Coverage is `ported / (items − excluded)`; `partial` counts as not done.",
        "",
        "## Coverage summary",
        "",
        "| Module | Ported | Partial | Missing | Excluded | Items | Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in MODULE_ORDER:
        c = counts[m]
        n = sum(c.values())
        if not n:
            continue
        out.append(f"| {m} | {c['ported']} | {c['partial']} | {c['missing']} | "
                   f"{c['excluded']} | {n} | {pct(c['ported'], n - c['excluded'])} |")
    n = sum(total.values())
    out.append(f"| **total** | **{total['ported']}** | **{total['partial']}** | "
               f"**{total['missing']}** | **{total['excluded']}** | **{n}** | "
               f"**{pct(total['ported'], n - total['excluded'])}** |")
    out += ["", f"nalgebra-cairo items with no upstream counterpart (undocumented extras): "
            f"**{len(extras)}** ([list](#items-in-nalgebra-cairo-but-not-upstream)); Cairo-imposed "
            f"forms of upstream operators, fields and `Deref` access: **{len(forms)}** "
            f"([list](#cairo-imposed-forms)); scalar layer: **{len(named)}** items named as in "
            f"simba-rs, **{len(kernels)}** documented exceptions "
            "([list](#scalar-layer-simba)).", ""]

    # Work packages.
    out += [
        "## Proposed work packages",
        "",
        "Every `missing` / `partial` item is assigned to one package (first matching rule of "
        "`WORK_PACKAGES`), each sized for one agent and one PR. Ordered by dependency; the tier is "
        "mechanical (API surface over existing kernels), standard numerics (new fused kernels, "
        "closed forms, oracle vectors) or hard numerics (iterative upstream algorithms that need "
        "a fixed-cost formulation).",
        "",
        "| WP | Title | Items | Tier | Depends on | Main upstream files |",
        "|---|---|---:|---|---|---|",
    ]
    for package in sorted(WORK_PACKAGES, key=lambda p: p.key):
        mine = [i for i in todo if wp_of[i] == package.key]
        files: dict[str, int] = {}
        for i in mine:
            files[i.source] = files.get(i.source, 0) + 1
        top = sorted(files.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
        out.append(f"| [{package.key}](#{anchor(package.key + ' ' + package.title)}) | "
                   f"{package.title} | {len(mine)} | {package.tier} | {package.depends} | "
                   + ", ".join(f"`{f}` ({k})" for f, k in top) + " |")
    out.append("")
    for package in sorted(WORK_PACKAGES, key=lambda p: p.key):
        mine = [i for i in todo if wp_of[i] == package.key]
        by_owner: dict[str, list[str]] = {}
        for i in mine:
            by_owner.setdefault(i.owner, []).append(
                f"`{md(rendered(i))}`" + ("*" if results[i].status == "partial" else ""))
        out += [f"### {package.key} {package.title}", "",
                f"{package.scope}. Tier: {package.tier}. Depends on: {package.depends}. "
                f"{len(mine)} items (`*` = partial):", ""]
        for owner in sorted(by_owner):
            out.append(f"- **{md(owner)}**: " + ", ".join(by_owner[owner]))
        out.append("")

    # Dimensions.
    out += [
        "## Concrete dimensions",
        "",
        "nalgebra-rs is generic over dimensions; its users name the aliases of "
        "`src/base/alias.rs` (and the geometry `*_alias.rs`). nalgebra-cairo provides the "
        "shapes in **bold**; `✗` = no Cairo type yet (P01 static shapes, P13 dynamic).",
        "",
    ]
    out += dimension_grid(cairo_types)
    geo = [("Point", [f"Point{d}" for d in range(1, 7)]),
           ("Translation", [f"Translation{d}" for d in range(1, 7)]),
           ("Scale", [f"Scale{d}" for d in range(1, 7)]),
           ("Reflection", [f"Reflection{d}" for d in range(1, 7)]),
           ("UnitVector", [f"UnitVector{d}" for d in range(1, 7)]),
           ("Rotation", ["Rotation2", "Rotation3", "UnitComplex", "UnitQuaternion"]),
           ("Isometry", ["Isometry2", "Isometry3", "IsometryMatrix2", "IsometryMatrix3"]),
           ("Similarity", ["Similarity2", "Similarity3", "SimilarityMatrix2",
                           "SimilarityMatrix3"]),
           ("Transform", ["Affine2", "Affine3", "Projective2", "Projective3", "Transform2",
                          "Transform3", "Perspective3", "Orthographic3"]),
           ("Quaternion", ["Quaternion", "DualQuaternion", "UnitDualQuaternion"])]
    out += ["", "| Family | Upstream aliases (**bold** = provided by nalgebra-cairo) |", "|---|---|"]
    for fam, names in geo:
        out.append(f"| {fam} | " + ", ".join(f"**{n}**" if n in cairo_types else n
                                             for n in names) + " |")
    out.append("")

    # Exclusions.
    ex_counts: dict[str, int] = {}
    for item in rust:
        if results[item].status == "excluded":
            ex_counts[results[item].detail] = ex_counts.get(results[item].detail, 0) + 1
    out += ["## Exclusion reasons (closed list)", "",
            "| Reason | Items | Justification |", "|---|---:|---|"]
    for code, text in EXCLUSIONS.items():
        out.append(f"| `{code}` | {ex_counts.get(code, 0)} | {text} |")
    out += [
        "",
        "`docs/PLAN.md` \"Out of scope\" also lists sparse, macros, complex numbers, Schur, "
        "Hessenberg, matrix exponential and convolution. They are NOT excluded here: they belong "
        "to nalgebra-rs's API, hence to the 0.1.0 target, until the owner decides otherwise "
        "(packages P16, P17, P18, P20, P21).",
        "",
    ]

    # Cairo-imposed forms.
    out += ["## Cairo-imposed forms", "",
            "Public Cairo items that spell an upstream operator, field or `Deref` access Cairo "
            "cannot express the same way, where no single upstream item can carry the rename "
            "(`CAIRO_FORMS`; the ones that can are `RENAMES`, shown in the inventory). They are "
            "the only admissible differences besides the renames (owner, 2026-09-24).", "",
            "| Owner | Items | Upstream | Reason |", "|---|---|---|---|"]
    fgrouped: dict[tuple[str, str, str], list[str]] = {}
    for item in forms:
        spelling, reason = cairo_form(item)
        fgrouped.setdefault((item.owner, spelling, reason), []).append(f"`{md(rendered(item))}`")
    for (owner, spelling, reason), names in sorted(fgrouped.items()):
        out.append(f"| {md(owner)} | {', '.join(sorted(names))} | {md(spelling)} | {md(reason)} |")
    out.append("")

    # Extras.
    out += ["## Items in nalgebra-cairo but not upstream", "",
            "\"Neither more nor less\": each item below must be justified (rationale shown when "
            "documented) or removed before 0.1.0. Traits that only carry Cairo methods "
            "(`Vector3Trait`...) and concrete instances of generic upstream types "
            "(`Cholesky3` for `Cholesky<T, U3>`) are not listed.", "",
            "| Owner | Items | Rationale |", "|---|---|---|"]
    grouped: dict[tuple[str, str], list[str]] = {}
    for item in extras:
        grouped.setdefault((item.owner, extra_rationale(item)), []).append(
            f"`{md(rendered(item))}`")
    for (owner, why), names in sorted(grouped.items()):
        out.append(f"| {md(owner)} | {', '.join(names)} | {md(why)} |")
    if not grouped:
        out.append("| none | | |")
    out.append("")

    # Scalar layer.
    out += ["## Scalar layer (simba)", "",
            "The `simba` package ([simba-cairo](https://github.com/bal7hazar/simba-cairo), a "
            f"registry dependency) is the counterpart of simba-rs {SIMBA_VERSION}'s `RealField`: every "
            "scalar item that has a simba-rs name carries it (`num::Zero::zero`, "
            "`num::One::one`, `RealField::pi`, `RealField::is_sign_negative`, "
            "`approx::AbsDiffEq::default_epsilon`, ...), and the rest is the one documented "
            "exception below.", "",
            "### Named as in simba-rs", "", "| Owner | Items | simba-rs trait |", "|---|---|---|"]
    ngrouped: dict[tuple[str, str], list[str]] = {}
    for item in named:
        ngrouped.setdefault((item.owner, simba_named(item, simba)), []).append(
            f"`{md(rendered(item))}`")
    for (owner, trait), names in sorted(ngrouped.items()):
        out.append(f"| {md(owner)} | {', '.join(sorted(names))} | {md(trait)} |")
    out += ["", "### Documented exception: fused scalar kernels and constants", "",
            "| Owner | Items | Rationale |", "|---|---|---|"]
    kgrouped: dict[str, list[str]] = {}
    for item in kernels:
        kgrouped.setdefault(item.owner, []).append(f"`{md(rendered(item))}`")
    for owner, names in sorted(kgrouped.items()):
        out.append(f"| {md(owner)} | {', '.join(sorted(names))} | {md(SCALAR_KERNELS_WHY)} |")
    out.append("")

    # Per owner.
    out += ["## Inventory by module and owner", ""]
    for m in MODULE_ORDER:
        owned = [i for i in rust if i.module == m]
        if not owned:
            continue
        out += [f"### Module `{m}`", ""]
        for owner in sorted({i.owner for i in owned}):
            items = [i for i in owned if i.owner == owner]
            c = {s: sum(results[i].status == s for i in items) for s in STATUSES}
            cands = OWNER_CANDIDATES.get(owner)
            out += [f"#### {md(owner)} ({m})", "",
                    f"Cairo: {compress(cands) if cands else 'none'} · ported {c['ported']}, "
                    f"partial {c['partial']}, missing {c['missing']}, excluded "
                    f"{c['excluded']}.", "",
                    "| Item | Status | Cairo | Detail / WP | Source |", "|---|---|---|---|---|"]
            for i in items:
                r = results[i]
                detail = r.detail
                if i in wp_of:
                    detail = (detail + "; " if detail else "") + wp_of[i]
                if i.flags:
                    detail = (detail + "; " if detail else "") + i.flags
                out.append(f"| {display(i)} | {r.status} | {md(r.cairo)} | {md(detail)} | "
                           f"`{i.source}` |")
            out.append("")

    inventory = {"nalgebra": [i.as_json() for i in rust], "simba": simba}
    body = "{\n\"nalgebra\": [\n" + ",\n".join(
        json.dumps(e, sort_keys=True) for e in inventory["nalgebra"]) + "\n],\n\"simba\": " + \
        json.dumps(simba) + "\n}"
    out += ["## Embedded nalgebra-rs inventory", "",
            f"Machine-readable nalgebra-rs {VERSION} inventory (and the simba {SIMBA_VERSION} "
            "scalar method names used to annotate the scalar layer), rewritten only by "
            "`--refresh`.", "", INVENTORY_START + body + INVENTORY_END, ""]
    return "\n".join(out)


def load_inventory(path: Path) -> tuple[list[Item], list[str]]:
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; run with --refresh first")
    text = path.read_text()
    start = text.find(INVENTORY_START)
    end = text.find(INVENTORY_END, start + len(INVENTORY_START))
    if start < 0 or end < 0:
        raise SystemExit(f"{path} has no embedded inventory; run with --refresh")
    data = json.loads(text[start + len(INVENTORY_START):end])
    return sorted(Item(**entry) for entry in data["nalgebra"]), data["simba"]


def write_or_check(generated: str, check: bool) -> int:
    current = OUTPUT.read_text() if OUTPUT.exists() else ""
    rel = OUTPUT.relative_to(ROOT)
    if check:
        if current == generated:
            print(f"{rel} is up to date")
            return 0
        print(f"{rel} is stale; run python3 scripts/api_parity.py", file=sys.stderr)
        diff = difflib.unified_diff(current.splitlines(), generated.splitlines(),
                                    fromfile=str(rel), tofile="generated", lineterm="")
        for line in list(diff)[:80]:
            print(line, file=sys.stderr)
        return 1
    OUTPUT.write_text(generated)
    print(f"wrote {rel}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--check", action="store_true", help="fail if docs/API_PARITY.md is stale")
    modes.add_argument("--refresh", action="store_true",
                       help="re-read the Rust sources and rewrite the embedded inventory")
    parser.add_argument("--nalgebra-rs", type=Path,
                        default=ROOT.parent.parent / "refs" / f"nalgebra-{VERSION}",
                        help=f"nalgebra-rs {VERSION} checkout (only with --refresh)")
    parser.add_argument("--simba", type=Path, default=None,
                        help=f"simba {SIMBA_VERSION} checkout (only with --refresh; defaults to "
                             "the cargo registry)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.refresh:
        rust = parse_rust(args.nalgebra_rs)
        simba_root = args.simba or next(iter(sorted(
            Path.home().glob(f".cargo/registry/src/*/simba-{SIMBA_VERSION}"))), None)
        if simba_root is None:
            raise SystemExit("simba checkout not found; pass --simba")
        simba = parse_simba(simba_root)
    else:
        rust, simba = load_inventory(OUTPUT)
    return write_or_check(render(rust, parse_cairo(), simba), args.check)


if __name__ == "__main__":
    raise SystemExit(main())
