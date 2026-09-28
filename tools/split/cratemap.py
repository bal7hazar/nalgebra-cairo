#!/usr/bin/env python3
"""The crate map of the package split (`tools/split/crates.toml`, docs/SPLIT.md §7; WP 9-NS2).

Library API (used by the generators, `scripts/api_parity.py`, `public_paths.py`,
`rewrite_imports.py`):

    cm = cratemap.load()                   # tools/split/crates.toml, or $NALGEBRA_CRATE_MAP
    cm.single                              # True: every crate is hosted by the facade package
    cm.packages()                          # the Cairo packages of the map, lowest first
    cm.route({"base/matrix3.cairo": text, ...})
        # -> {"crates/<dir>/src/base/matrix3.cairo": text, ...}: every generated file cut into the
        #    packages of its items (module path unchanged, impls moved to an anchor module when
        #    Cairo's lookup needs it), plus the facade's re-export files of the modules that left it.
        #    Single-crate mode: the identity (the texts come back byte for byte).
    cm.place_text(module_file, text)        # [(item, crate, package, module_file)]

Command line:

    python3 tools/split/cratemap.py --show [--map M]         # the crates, packages and rules
    python3 tools/split/cratemap.py --place [--map M]        # items per crate of today's library
    python3 tools/split/cratemap.py --compare-plan EDGES.json PLAN.json [--map M] [--generated]
        # item-by-item comparison with a plan of `tools/split/plan.py` (NS1's `final`): crate and
        # module of every item against `plan.py` + `prototype.py`'s anchor rule
    python3 tools/split/cratemap.py --split-map OUT.toml     # the map with every crate hosted by
        # its own package `nalgebra_<crate>` (the facade by `nalgebra`): NS1's final layout

Rules and format: the comments of `crates.toml`.
"""
import argparse
import collections
import fnmatch
import os
import re
import sys
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import consumer_cost as cc  # noqa: E402
from edges import IDENT, USE, expand_use, items as cut_items, raw_start  # noqa: E402

DEFAULT_MAP = os.path.join(HERE, "crates.toml")
INTERNAL = "internal/"  # the modules of the `[internal]` items of a package (SPLIT §12.3)
SHAPE_FILE = re.compile(r"base/(matrix|vector|row_vector)(\d)(?:x(\d))?\.cairo$")
SHAPE_NAME = re.compile(r"(?:Matrix|Vector|RowVector|UnitVector)(\d)(?:x(\d))?$")
CRATE_MAP_END = "// crate-map: end"
ITEMS_BEGIN = "// crate-map: generated items (tools/split/cratemap.py)"
ITEM_BLOCK = re.compile(r"^// crate-map: generated items \(tools/split/cratemap\.py\) \[(\w+)\]\n.*?"
                        r"^// crate-map: end\n?", re.S | re.M)
TEST_ONLY = re.compile(r"#\[cfg\((test|[^)]*\btest\b)")
FMT_MANIFEST = """[package]
name = "crate_map_routed"
version = "0.1.0"
edition = "2024_07"

[tool.fmt]
sort-module-level-items = true
max-line-length = 100
"""


def shape_of_file(path):
    """(rows, cols) of a base shape file (`base/matrix2x5.cairo`), None otherwise."""
    m = SHAPE_FILE.match(path)
    if not m:
        return None
    a, b = int(m.group(2)), int(m.group(3) or m.group(2))
    return {"vector": (a, 1), "row_vector": (1, a)}.get(m.group(1), (a, b))


def shape_type(path):
    """`Matrix2x5` / `Vector3` / `RowVector6` of a base shape file."""
    s = shape_of_file(path)
    if s is None:
        return None
    stem = path.rsplit("/", 1)[-1][: -len(".cairo")]
    return "".join(w.capitalize() for w in stem.split("_"))


def file_band(path):
    """Dimension of a file: the max of a base shape, else the digits ending a `linalg` file name."""
    s = shape_of_file(path)
    if s:
        return max(s)
    if path.startswith("linalg/"):
        m = re.search(r"(\d)(?:x(\d))?\.cairo$", path)
        if m and m.group(1) != "0":
            return max(int(m.group(1)), int(m.group(2) or 0))
    return None


def name_band(name):
    """Largest digit of an item name (`Matrix3x5Trait` -> 5), None without digit."""
    m = re.findall(r"(\d)(?:x(\d))?", name or "")
    if not m:
        return None
    return max(max(int(a), int(b or a)) for a, b in m)


def shape_band(name):
    m = SHAPE_NAME.match(name or "")
    if not m:
        return None
    return max(int(m.group(1)), int(m.group(2) or m.group(1)))


def package_dir(package):
    """Directory of a package: `nalgebra` -> crates/nalgebra, `nalgebra_x` -> crates/x."""
    if package == "nalgebra":
        return os.path.join("crates", "nalgebra")
    return os.path.join("crates", package[len("nalgebra_"):] if package.startswith("nalgebra_") else package)


class Item:
    """A top-level item of a module file."""

    __slots__ = ("file", "kind", "name", "gen", "of", "args", "start", "end", "public")

    def __init__(self, file, kind, name, gen, of, args, start, end, public):
        self.file, self.kind, self.name, self.gen, self.of = file, kind, name, gen, of
        self.args, self.start, self.end, self.public = args, start, end, public

    @property
    def label(self):
        return self.gen or self.name

    def __repr__(self):
        return f"{self.file}:{self.label}"


def parse(file, text, known=None):
    """The top-level items of a module file (use / mod declarations included), in order, with the
    doc comments and attributes right above each one. `known`: the library's type / trait names
    (an impl's `args` are the known names of its `of X<...>` part)."""
    masked = cc.mask(text)
    raw_lines = text.split("\n")
    starts = [0]
    for line in raw_lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)
    out = []
    local_traits = {}
    cut = cut_items(masked)
    for kind, name, gen, of, s, e in cut:
        if kind == "trait":
            local_traits[name] = local_traits.get(name, 0)
    impls_of = collections.Counter(of for kind, _n, _g, of, _s, _e in cut if kind == "impl" and of)
    for kind, name, gen, of, s, e in cut:
        head = masked[s:e].split("{", 1)[0]
        args = []
        if kind == "impl" and of:
            tail = head[head.find(" of ") + 4:] if " of " in head else ""
            args = sorted({x for x in IDENT.findall(tail) if (known is None or x in known) and x != of})
        k = kind
        if kind == "impl" and gen:
            k = "inherent"
        elif kind == "impl" and of in local_traits and impls_of[of] == 1:
            k = "inherent"
            gen = of
        elif kind == "trait" and impls_of.get(name) == 1:
            k = "inherent"
            gen = name
        public = bool(re.match(r"pub\s", masked[s:e]))
        out.append(Item(file, k, name, gen, of, args, raw_start(raw_lines, starts, s), e, public))
    return out


class Rule:
    def __init__(self, d, crates):
        self.d = d
        for key in ("file", "kind", "trait", "not_trait", "shape", "band", "args_any", "nargs"):
            if key in d and not isinstance(d[key], list):
                raise SystemExit(f"crates.toml: `{key}` must be a list in rule {d}")
        self.name = re.compile(d["name"]) if "name" in d else None
        targets = [k for k in ("crate", "bands", "follow") if k in d]
        if len(targets) != 1:
            raise SystemExit(f"crates.toml: a rule needs exactly one of crate / bands / follow: {d}")
        for c in [d.get("crate"), d.get("default"), *(d.get("bands") or {}).values()]:
            if c is not None and c not in crates:
                raise SystemExit(f"crates.toml: unknown crate {c!r} in rule {d}")

    def matches(self, it):
        d = self.d
        if "file" in d and not any(fnmatch.fnmatchcase(it.file, g) for g in d["file"]):
            return False
        if "kind" in d and it.kind not in d["kind"]:
            return False
        if self.name is not None and not self.name.fullmatch(it.label or ""):
            return False
        if "trait" in d and it.of not in d["trait"]:
            return False
        if "not_trait" in d and it.of in d["not_trait"]:
            return False
        if "shape" in d and shape_type(it.file) not in d["shape"]:
            return False
        if "band" in d and file_band(it.file) not in d["band"]:
            return False
        if "args_any" in d and not set(it.args) & set(d["args_any"]):
            return False
        if "nargs" in d and len(it.args) not in d["nargs"]:
            return False
        if "shape_file" in d and (shape_of_file(it.file) is not None) != d["shape_file"]:
            return False
        return True


class CrateMap:
    def __init__(self, path=None):
        self.path = path or os.environ.get("NALGEBRA_CRATE_MAP") or DEFAULT_MAP
        with open(self.path, "rb") as f:
            d = tomllib.load(f)
        self.crates = dict(d["crates"])  # crate -> package, lowest first
        self.order = list(self.crates)
        self.rank = {c: i for i, c in enumerate(self.order)}
        self.facade = d.get("facade", self.order[-1])
        self.facade_package = self.crates[self.facade]
        self.lift = d.get("lift", True)
        self.rules = [Rule(r, self.crates) for r in d.get("rule", [])]
        self.single = len(set(self.crates.values())) == 1
        self.internal = [re.compile(x) for x in d.get("internal", {}).get("names", [])]
        self._index = None

    # --- packages --------------------------------------------------------------------------------

    def packages(self):
        """The Cairo packages of the map, lowest first (the facade's package last)."""
        out = []
        for c in self.order:
            p = self.crates[c]
            if p not in out and p != self.facade_package:
                out.append(p)
        return out + [self.facade_package]

    def package_rank(self, package):
        return max(self.rank[c] for c, p in self.crates.items() if p == package)

    # --- placement -------------------------------------------------------------------------------

    def rule_crate(self, it, index=None):
        """(crate, rule number) of an item by the rules alone (before the lift)."""
        for k, r in enumerate(self.rules):
            if not r.matches(it):
                continue
            d = r.d
            if "crate" in d:
                return d["crate"], k
            if "follow" in d:
                st = (index or {}).get(it.args[0]) if it.args else None
                if st is None:
                    continue
                return self.crate_of(st, index), k
            by = d.get("by", "file")
            if by == "file":
                b = file_band(it.file)
            elif by == "name":
                b = name_band(it.label)
            elif by == "auto":
                b = file_band(it.file) or name_band(it.label)
            elif by == "args":
                bs = [shape_band(a) for a in it.args]
                bs = [x for x in bs if x is not None]
                b = max(bs) if bs else None
            else:
                raise SystemExit(f"crates.toml: unknown `by` {by!r}")
            if b is None:
                if "default" in d:
                    return d["default"], k
                continue
            for key in sorted(d["bands"], key=int):
                if b <= int(key):
                    return d["bands"][key], k
            if "default" in d:
                return d["default"], k
            continue
        return self.facade, None

    def crate_of(self, it, index=None):
        """The planned crate of an item: its rule, lifted above its type arguments (impls)."""
        c, _ = self.rule_crate(it, index)
        if self.lift and it.kind == "impl" and index is not None:
            for a in it.args:
                st = index.get(a)
                if st is not None and st.kind in ("struct", "enum") and st is not it:
                    ca, _ = self.rule_crate(st, index)
                    if self.rank[ca] > self.rank[c]:
                        c = ca
        return c

    def is_internal(self, it, crate):
        """An item of `[internal]`: crate-private in 0.1.0, hosted by a package below the facade
        and used above it (split mode only)."""
        # (not `it.public`: once moved, the committed item is `pub` in its `internal` module)
        return (not self.single and self.crates[crate] != self.facade_package
                and any(r.fullmatch(it.label or "") for r in self.internal))

    def module_of(self, it, crate, index):
        """Module file of an item in its package: its own, unless it is an impl with type
        arguments whose module holds neither its trait nor an argument type in the same package
        (Cairo would not find it): then the first such anchor's module (arguments in name order,
        then the trait). An `[internal]` item goes to `internal/<its module file>` (SPLIT §12.3)."""
        m = self._module_of(it, crate, index)
        return INTERNAL + m if self.is_internal(it, crate) else m

    def _module_of(self, it, crate, index):
        if not (it.kind == "impl" and it.args):
            return it.file
        pkg = self.crates[crate]
        here = []
        for a in list(it.args) + ([it.of] if it.of else []):
            an = index.get(a)
            if an is None or an.kind not in ("struct", "enum", "trait", "inherent"):
                continue
            ca = self.crate_of(an, index)
            if self.crates[ca] == pkg:
                here.append(self.module_of(an, ca, index))
        if not here or it.file in here:
            return it.file
        return here[0]

    # --- the library index -----------------------------------------------------------------------

    def library_files(self, overrides=None):
        """{module file: text} of the library: every package of the map (their `src/`; the parts
        of a module several packages host are concatenated), with `overrides` (freshly generated
        texts) replacing the committed ones."""
        out = {}
        sys.path.insert(0, HERE)
        from files import files as walk

        for pkg in self.packages():
            d = os.path.join(ROOT, package_dir(pkg))
            if not os.path.exists(os.path.join(d, "src", "lib.cairo")):
                continue
            for rel in walk(d):
                if rel in ("lib.cairo", INTERNAL[:-1] + ".cairo") and pkg != self.facade_package:
                    continue
                text = open(os.path.join(d, "src", rel), encoding="utf-8").read()
                # `internal/x.cairo` holds the `[internal]` items of `x.cairo`: placed as x's
                rel = rel[len(INTERNAL):] if rel.startswith(INTERNAL) else rel
                # a module split over several packages: the items of every part
                out[rel] = out[rel] + "\n" + text if rel in out else text
        out.update(overrides or {})
        return out

    def index(self, texts):
        """{unique struct / enum / trait / type name: Item} over the library (first file wins for
        a name defined twice, as `prototype.py` does)."""
        names = collections.defaultdict(list)
        parsed = {}
        for f in sorted(texts):
            parsed[f] = parse(f, texts[f])
            for it in parsed[f]:
                if it.kind in ("use", "mod"):
                    continue
                for nm in {it.name, it.gen} - {None, ""}:
                    names[nm].append(it)
        known = {k for k, v in names.items() if len(v) == 1}
        idx = {}
        for f in sorted(texts):
            for it in parsed[f]:
                if it.kind in ("struct", "enum", "trait", "inherent", "type"):
                    for nm in {it.name, it.gen} - {None, ""}:
                        if it.kind == "inherent" and nm != it.gen:
                            continue
                        idx.setdefault(nm, it)
        # the impl of a `#[generate_trait]` trait, imported by its own name (`use x::Powi;`)
        for f in sorted(texts):
            for it in parsed[f]:
                if it.kind == "inherent" and it.gen and it.name != it.gen and names.get(it.name) == [it]:
                    idx.setdefault(it.name, it)
        return idx, known

    def place_all(self, texts):
        """{module file: [(item, crate, module file)]} for the texts (use / mod items excluded)."""
        idx, known = self.index(texts)
        out = {}
        for f in sorted(texts):
            rows = []
            for it in parse(f, texts[f], known):
                if it.kind in ("use", "mod"):
                    continue
                c = self.crate_of(it, idx)
                rows.append((it, c, self.module_of(it, c, idx)))
            out[f] = rows
        return out

    # --- routing (the generators) ----------------------------------------------------------------

    def route(self, generated, prefix="crates/nalgebra/src/", tag="generator"):
        """Cut generated module files into the packages of their items.

        `generated`: {module file relative to `src/` (or `prefix` + it): text}. Returns
        {repository path: text}. Single-crate mode: every file comes back unchanged at
        `crates/<facade package dir>/src/<module file>`. `tag` names the generator: the blocks it
        writes into files it does not own (re-exports, module roots, moved impls) carry it, and
        the blocks of other generators are kept."""
        gen = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in generated.items()}
        facade_dir = package_dir(self.facade_package)
        if self.single:
            return {os.path.join(facade_dir, "src", k): v for k, v in gen.items()}
        texts = self.library_files(gen)
        idx, known = self.index(texts)
        out = {}
        hosted = collections.defaultdict(set)  # module file -> packages hosting part of it
        pieces = collections.defaultdict(list)  # (package, module) -> [(origin, item)]
        headers, uses_of, placed = {}, {}, {}
        bodies = {}  # path -> the text of a module this generator writes
        whole = {}  # (package, module) -> a generated file that goes there whole
        stmts = collections.defaultdict(list)  # path -> `use` / `pub mod` lines it must hold
        items = collections.defaultdict(list)  # path -> [(origins, moved impls)]
        explicit = collections.defaultdict(list)  # facade module -> explicit re-exports
        for f, text in gen.items():
            its = parse(f, text, known)
            decl = [it for it in its if it.kind in ("use", "mod")]
            body = [it for it in its if it.kind not in ("use", "mod")]
            first = min([it.start for it in its] or [len(text)])
            headers[f] = text[:first]
            uses_of[f] = [text[it.start:it.end] for it in decl if it.kind == "use"]
            rows = []
            for it in body:
                c = self.crate_of(it, idx)
                pkg = self.crates[c]
                mod = self.module_of(it, c, idx)
                rows.append((it, pkg, mod))
                placed[(f, it.label)] = (pkg, mod)
            pkgs = {pkg for _it, pkg, mod in rows if mod == f} or {self.facade_package}
            if len(pkgs) == 1 and all(mod == f for _i, _p, mod in rows):
                # the whole file goes to one package: written as it is (comments, markers)
                (pkg,) = pkgs
                whole[(pkg, f)] = text
                hosted[f].add(pkg)
                continue
            mods = [it for it in decl if it.kind == "mod"]
            for it, pkg, mod in rows:
                chunk = text[it.start:it.end]
                if mod.startswith(INTERNAL) and not it.public:
                    chunk = publish(chunk)
                pieces[(pkg, mod)].append((f, chunk))
                if mod != f and it.public:
                    # a public impl moved to an anchor module keeps its 0.1.0 path in the facade
                    root = "crate" if pkg == self.facade_package else pkg
                    explicit[f].append(f"pub use {root}::{mod[: -len('.cairo')].replace('/', '::')}::{it.name};")
            for it in mods:
                # a test module (`#[cfg(test)] mod tests;`) goes with the file's highest package,
                # which hosts the methods it tests; any other module with the lowest
                test = TEST_ONLY.search(text[it.start:it.end]) is not None
                host = (max if test else min)(pkgs, key=self.package_rank)
                pieces[(host, f)].append((f, text[it.start:it.end]))
        # a whole file keeps its text, except the imports of `[internal]` items (now at `internal::`)
        for (pkg, f), text in list(whole.items()):
            whole[(pkg, f)] = self._internal_uses(text, f, pkg, placed, idx)
        for (pkg, mod), chunks in sorted(pieces.items(), key=lambda kv: (kv[0][1], self.package_rank(kv[0][0]))):
            hosted[mod].add(pkg)
            origins = sorted({o for o, _ in chunks})
            header = headers.get(mod) or headers[origins[0]]
            joined = "\n\n".join(t.strip("\n") for _o, t in chunks)
            # the names the piece uses (a method declared `fn gauss_step` is not a use of the free
            # function `gauss_step`), minus the ones its module defines (never imported there)
            masked_joined = cc.mask(joined)
            # (nor is `Self::gauss_step`: a name after `::` is never imported)
            used = set(IDENT.findall(re.sub(r"\bfn\s+\w+|::\s*\w+", " ", masked_joined)))
            used -= _defined(masked_joined + "\n" + cc.mask(whole.get((pkg, mod), "")))
            lines = []
            for o in origins:
                for u in uses_of[o]:
                    kept = self._rewrite_use(u, o, mod, pkg, used, placed, idx)
                    if kept and kept not in lines:
                        lines.append(kept)
            # names defined in another piece of the same original file
            for o in origins:
                for (of, lab), (qp, qm) in placed.items():
                    if of != o or lab not in used or (qp, qm) == (pkg, mod):
                        continue
                    if any(re.search(rf"\b{re.escape(lab)}\b", t) is not None and
                           re.search(rf"^(pub\s+)?(impl|trait|struct|fn|const|type|enum)\s+{re.escape(lab)}\b", t, re.M)
                           for _o, t in chunks):
                        continue
                    root = "crate" if qp == pkg else qp
                    line = f"use {root}::{qm[:-len('.cairo')].replace('/', '::')}::{lab};"
                    if line not in lines:
                        lines.append(line)
            if pkg == self.facade_package:
                # the facade module re-exports the same module of the packages below
                # (`pub use nalgebra_x::<path>::*`): a private import of one of their names would
                # shadow that re-export for the rest of the crate
                here = mod[: -len(".cairo")].replace("/", "::")
                subs = {self.crates[c] for c in self.crates} - {pkg}
                lines = [x for x in (keep_uses(ln, lambda full: not any(
                    full.startswith(f"{p}::{here}::") and full.count("::") == here.count("::") + 2
                    for p in subs)) for ln in lines) if x]
                # the in-crate tests it keeps (`#[cfg(test)] mod tests;`) reach the `[internal]`
                # items of the module that moved below through a test-only import (WP 9-NS4)
                lines += self._test_internal_uses(mod, pkg, chunks, placed)
            path = os.path.join(package_dir(pkg), "src", mod)
            if (pkg, mod) in whole:
                # a whole file plus impls moved into its module: its text, their imports, them
                own = whole.pop((pkg, mod))
                have = statements(own)
                taken = local_names(own)
                extra = [ln for ln in lines if not statements(ln) <= have]
                extra = [x for x in (keep_uses(ln, lambda full: use_local(full) not in taken) for ln in extra) if x]
                bodies[path] = own.rstrip("\n") + ("\n\n" + "\n".join(extra) if extra else "") + \
                    "\n\n" + joined + "\n"
            elif mod in gen or (mod.startswith(INTERNAL) and mod[len(INTERNAL):] in gen):
                if mod not in gen:  # the `[internal]` items of a generated module: owned here too
                    header = INTERNAL_DOC.format(module=mod[len(INTERNAL): -len(".cairo")].replace("/", "::"))
                bodies[path] = header + "\n".join(lines) + ("\n\n" if lines else "") + joined + "\n"
            else:
                # impls moved into a module this generator does not write (an anchor type's)
                stmts[path].extend(lines)
                items[path].append((", ".join(origins), joined))
        for (pkg, mod), text in whole.items():
            bodies[os.path.join(package_dir(pkg), "src", mod)] = text
        # a generated module the facade no longer hosts: its module doc and the re-exports only
        for f in gen:
            path = os.path.join(facade_dir, "src", f)
            if path not in bodies:
                bodies[path] = headers[f]
        # the facade: the re-exports of every module that (partly) left it
        for mod in sorted(set(hosted) | set(explicit)):
            if mod.startswith(INTERNAL):
                continue  # never re-exported by the facade (SPLIT §12.3)
            subs = sorted((p for p in hosted.get(mod, ()) if p != self.facade_package), key=self.package_rank)
            path = mod[: -len(".cairo")].replace("/", "::")
            stmts[os.path.join(facade_dir, "src", mod)].extend(
                [f"pub use {p}::{path}::*;" for p in subs] + sorted(explicit.get(mod, [])))
        # the module roots of the sub-crates: `pub mod x;` down to every hosted module
        for mod, pkgs in hosted.items():
            for pkg in pkgs - {self.facade_package}:
                parts = mod[: -len(".cairo")].split("/")
                for i in range(len(parts)):
                    parent = "/".join(parts[:i]) + ".cairo" if i else "lib.cairo"
                    stmts[os.path.join(package_dir(pkg), "src", parent)].append(f"pub mod {parts[i]};")
        for path in sorted(set(bodies) | set(stmts) | set(items)):
            out[path] = self._assemble(path, bodies.get(path), stmts.get(path, []), items.get(path, []), tag)
        return out

    @staticmethod
    def _assemble(path, body, stmts, items, tag):
        """The text of `path`.

        `body`: the text this generator writes (a generated module), else the committed file is
        kept (a module another generator or a person writes). `stmts`: `use` / `pub use` /
        `pub mod` lines the file must hold: each one is added if the file does not have it yet
        (grouping aside), never removed (`scarb fmt` sorts these lines, so they carry no
        marker). `items`: impls moved into this module, one marked block per origin; the blocks
        of this generator are replaced, the other generators' kept (with the `use` lines they
        need when `body` replaces the file), always at the end of the file (`scarb fmt` does not
        reorder impls)."""
        full = os.path.join(ROOT, path)
        committed = open(full, encoding="utf-8").read() if os.path.exists(full) else ""
        blocks = list(ITEM_BLOCK.finditer(committed))
        foreign = [m.group(0).rstrip("\n") for m in blocks if m.group(1) != tag]
        if body is None:
            base = ITEM_BLOCK.sub("", committed)
        else:
            base = body
            # the `use` lines of the committed file that the kept blocks of other generators need
            need = set(IDENT.findall(cc.mask("\n".join(foreign))))
            have = statements(base)
            for m in USE.finditer(cc.mask(ITEM_BLOCK.sub("", committed))):
                stmt = committed[m.start():m.end()].strip()
                names = {x.split(" as ")[-1].split("::")[-1].strip() for x in expand_use(m.group(1))}
                if names & need and not statements(stmt) <= have:
                    stmts = [stmt] + list(stmts)
        text = base.rstrip("\n")
        present = statements(text)
        taken = local_names(text)
        add = []
        for ln in stmts:
            ln = keep_uses(ln, lambda full: use_local(full) not in taken) if ln.lstrip().startswith("use ") else ln
            if not ln:
                continue
            st = statements(ln)
            if st and not st <= present:
                add.append(ln)
                present |= st
        if add:
            text = (text + "\n\n" if text.strip() else "") + "\n".join(add)
        for fb in foreign:
            text = (text + "\n\n" if text.strip() else "") + fb
        for origins, joined in items:
            src = "".join(f"// crate-map: from {o}\n" for o in origins.split(", "))
            text = (text + "\n\n" if text.strip() else "") + \
                f"{ITEMS_BEGIN} [{tag}]\n{src}{joined}\n{CRATE_MAP_END}"
        return text + "\n"

    def _test_internal_uses(self, mod, pkg, chunks, placed):
        """`#[cfg(test)] use <package>::internal::<mod>::<Name>;` for every `[internal]` item of
        module `mod` placed in another package whose name a test module of `mod` kept in `pkg`
        (a `#[cfg(test)] mod x;` of `chunks`, file `<mod>/x.cairo`) names."""
        tests = []
        for _o, t in chunks:
            m = re.match(r"\s*#\[cfg\(test\)\]\s*mod\s+(\w+)\s*;", cc.mask(t))
            if m:
                p = os.path.join(ROOT, package_dir(pkg), "src", mod[: -len(".cairo")], m.group(1) + ".cairo")
                if os.path.exists(p):
                    tests.append(cc.mask(open(p, encoding="utf-8").read()))
        if not tests:
            return []
        out = []
        for (of, lab), (qp, qm) in sorted(placed.items()):
            if of != mod or qp == pkg or qm != INTERNAL + mod:
                continue
            if any(re.search(rf"\b{re.escape(lab)}\b", t) for t in tests):
                out.append(f"#[cfg(test)]\nuse {qp}::{qm[: -len('.cairo')].replace('/', '::')}::{lab};")
        return out

    def _internal_uses(self, text, f, pkg, placed, idx):
        """`text` (module file `f` of package `pkg`) with every private `use` statement that names an
        `[internal]` item rewritten to its `internal::` path; the other statements untouched."""
        if not self.internal:
            return text
        masked = cc.mask(text)
        names = set(IDENT.findall(masked))
        out, last = [], 0
        for m in USE.finditer(masked):
            stmt = text[m.start():m.end()]
            if stmt.lstrip().startswith("pub"):
                continue
            hit = False
            for full in expand_use(m.group(1)):
                it = idx.get(full.split(" as ")[0].split("::")[-1].strip())
                if it is not None and self.is_internal(it, self.crate_of(it, idx)):
                    hit = True
            if not hit:
                continue
            new = self._rewrite_use(stmt.strip(), f, f, pkg, names, placed, idx)
            lead = stmt[: len(stmt) - len(stmt.lstrip())]
            out.append(text[last:m.start()] + lead + (new or ""))
            last = m.end()
        out.append(text[last:])
        return "".join(out)

    def _rewrite_use(self, stmt, origin, mod, pkg, used, placed, idx):
        """A `use` statement of the original file for a piece: only the names the piece uses,
        `crate::` paths pointing at the package that defines the name."""
        m = USE.match(stmt.strip() + ("" if stmt.strip().endswith(";") else ";"))
        if m is None:
            return stmt
        head = stmt.strip()[: stmt.strip().index("use")]
        keep = []
        for full in expand_use(m.group(1)):
            alias = None
            if " as " in full:
                full, alias = [x.strip() for x in full.split(" as ")]
            segs = [s.strip() for s in full.split("::")]
            local = alias or segs[-1]
            if local not in used and segs[-1] != "*":
                continue
            if segs[0] in ("crate", "super", "self"):
                cur = origin[: -len(".cairo")].split("/")
                if segs[0] == "crate":
                    cur, rest = [], segs[1:]
                elif segs[0] == "self":
                    rest = segs[1:]
                else:
                    rest = segs
                while rest and rest[0] == "super":
                    cur, rest = cur[:-1], rest[1:]
                absolute = cur + rest
                target_pkg = pkg
                it = idx.get(absolute[-1])
                if it is not None:
                    tc = self.crate_of(it, idx)
                    target_pkg = self.crates[tc]
                    q = self.module_of(it, tc, idx)[: -len(".cairo")].split("/")
                    # a name another sub-crate defines, imported through a re-export of 0.1.0
                    # (`crate::geometry::Rotation3`), is imported from its defining module: the
                    # sub-crates keep only the re-exports their own code uses (WP 9-NS4)
                    if q != absolute[:-1] and ("/".join(absolute[:-1]) + ".cairo" == it.file
                                               or target_pkg not in (pkg, self.facade_package)):
                        absolute = q + absolute[-1:]
                pl = placed.get(("/".join(absolute[:-1]) + ".cairo", absolute[-1]))
                if pl is not None:
                    target_pkg = pl[0]
                    absolute = pl[1][: -len(".cairo")].split("/") + absolute[-1:]
                elif it is None:
                    # a module (`crate::base::errors`): the package whose `src/` holds its file
                    # when this package does not (WP 9-NS4)
                    rel = os.path.join(*absolute) + ".cairo"
                    if not os.path.exists(os.path.join(ROOT, package_dir(pkg), "src", rel)):
                        for p in self.packages():
                            if os.path.exists(os.path.join(ROOT, package_dir(p), "src", rel)):
                                target_pkg = p
                                break
                root = "crate" if target_pkg == pkg else target_pkg
                full = "::".join([root] + absolute)
            keep.append(full + (f" as {alias}" if alias else ""))
        if not keep:
            return None
        if len(keep) == 1:
            return f"{head}use {keep[0]};"
        return f"{head}use {{{', '.join(keep)}}};".replace("use {", "use {", 1)


INTERNAL_DOC = """//! Internal, no stability promise: the crate-private items of `{module}` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

"""


def _defined(masked):
    """The names a (comment-free) module text defines at the top level: its items and the traits of
    its `#[generate_trait]` impls."""
    out = set(re.findall(r"^(?:pub(?:\(crate\))?\s+)?(?:impl|trait|struct|enum|fn|const|type)\s+(\w+)", masked, re.M))
    out |= set(re.findall(r"#\[generate_trait\]\s*(?:pub(?:\(crate\))?\s+)?impl\s+\w+[^{;]*?\bof\s+(\w+)", masked))
    return out


def publish(chunk):
    """An `[internal]` item made `pub` (0.1.0: `pub(crate)` or private): its first item line."""
    masked = cc.mask(chunk)
    m = re.search(r"^(pub\(crate\)\s+|pub\s+)?(?=(impl|trait|struct|enum|fn|const|type|mod)\s)", masked, re.M)
    if m is None:
        return chunk
    return chunk[: m.start()] + "pub " + chunk[m.end(1) if m.group(1) else m.start():]


def use_local(full):
    """The local name a `use` path binds (`a::b::C as D` -> `D`)."""
    return full.split(" as ")[-1].split("::")[-1].strip()


def local_names(text):
    """The names a module text binds at the top level: its items and its non-glob imports."""
    masked = cc.mask(text)
    out = _defined(masked)
    for m in USE.finditer(masked):
        out |= {use_local(x) for x in expand_use(m.group(1))} - {"*"}
    return out


def keep_uses(line, keep):
    """A (private) `use` line with only the paths `keep(full path)` accepts; None if none is left."""
    s = line.strip()
    m = USE.match(s if s.endswith(";") else s + ";")
    if m is None or not s.startswith("use "):
        return line
    fulls = [x.strip() for x in expand_use(m.group(1))]
    kept = [x for x in fulls if keep(x)]
    if not kept:
        return None
    if len(kept) == len(fulls):
        return line
    return "\n".join(f"use {x};" for x in kept)


def statements(text):
    """The `use` paths and `mod` declarations of a text ({"use a::b::C", "mod x"}), comparable
    whatever the grouping (`use a::{B, C};`)."""
    out = set()
    masked = cc.mask(text)
    for m in USE.finditer(masked):
        for full in expand_use(m.group(1)):
            out.add("use " + re.sub(r"\s+", "", full))
    for m in re.finditer(r"^\s*(?:pub\s+)?mod\s+(\w+)\s*[;{]", masked, re.M):
        out.add("mod " + m.group(1))
    return out


def load(path=None):
    return CrateMap(path)


def committed_path(path, marker):
    """Where the committed text of a hand-written file with a generated block lives: `path` (in
    the facade) if it holds `marker`, else the same module file of the package that does (the file
    moved with its crate), else `path`."""
    from pathlib import Path

    path = Path(path)
    if not path.exists() or marker in path.read_text():
        return path
    cm = load()
    lib = Path(ROOT) / package_dir(cm.facade_package) / "src"
    if not path.is_relative_to(lib):
        return path
    rel = path.relative_to(lib)
    for pkg in cm.packages():
        other = Path(ROOT) / package_dir(pkg) / "src" / rel
        if other.exists() and marker in other.read_text():
            return other
    return path


def route_outputs(outputs, tmp, tag):
    """The generators' hook. `outputs`: {committed Path: generated Path} (formatted files).
    Single-crate mode: returned as is. Split mode: the library files (`crates/nalgebra/src/`) are
    routed (`CrateMap.route`), written under `tmp` and formatted with `scarb fmt`; the result maps
    every routed repository path to its formatted file, the other outputs unchanged."""
    import shutil
    import subprocess
    from pathlib import Path

    cm = load()
    if cm.single:
        return outputs
    lib = Path(ROOT) / package_dir(cm.facade_package) / "src"
    mine = {dst: gen for dst, gen in outputs.items() if Path(dst).is_relative_to(lib)}
    routed = cm.route({str(Path(dst).relative_to(lib)): Path(gen).read_text() for dst, gen in mine.items()},
                      prefix="", tag=tag)
    tmp = Path(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "src").mkdir(parents=True)
    (tmp / "Scarb.toml").write_text(FMT_MANIFEST)
    (tmp / "src" / "lib.cairo").write_text("")
    out = {dst: gen for dst, gen in outputs.items() if dst not in mine}
    for rel, text in routed.items():
        path = tmp / "src" / "routed" / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        out[Path(ROOT) / rel] = path
    subprocess.run(["scarb", "fmt"], cwd=tmp, check=True)
    return out


# --- command line --------------------------------------------------------------------------------


def split_map_text(path=DEFAULT_MAP):
    """The map's text with every crate hosted by `nalgebra_<crate>` (the facade by `nalgebra`)."""
    text = open(path).read()
    facade = tomllib.loads(text).get("facade", "facade")

    def sub(m):
        c = m.group(1)
        return f'{c} = "{"nalgebra" if c == facade else "nalgebra_" + c}"'

    m = re.search(r"^\[crates\]\n(.*?)(?=\n\n|\n\[|\Z)", text, re.S | re.M)
    body = re.sub(r'^(\w+) = "[^"]*"', sub, m.group(1), flags=re.M)
    return text[: m.start(1)] + body + text[m.end(1):]


# NS1's crate names (`layouts.py`, `plan.py`) -> the final names of the map (WP 9-NS1b: renamed,
# five merges), for `--compare-plan` against a plan of `plan.py`
NS1_NAMES = {
    "dim3v": "static3", "dim3": "static3", "types5": "shapes5", "dim4": "static4",
    "geometry": "geometry4", "types6": "shapes6", "dim5": "static5", "dim5a": "static5",
    "dim6a": "static6_tall", "dim6": "static6_wide", "dim6b": "static6_wide", "edition": "blocks",
    "kronecker": "blocks", "geometry_nd": "geometry6", "linalg": "linalg4",
    "linalg_svd": "linalg_svd_eigen4", "linalg_pivot": "linalg_pivot4",
    "linalg_spectral": "linalg_spectral4", "linalg5_pivot": "linalg_pivot5",
    "linalg5_spectral": "linalg_spectral5", "linalg6_pivot": "linalg_pivot6",
    "linalg6_bidiagonal": "linalg_spectral6", "linalg6_spectral": "linalg_spectral6",
}


def compare_plan(cm, edges_path, plan_path, generated_only):
    import json

    from prototype import Splitter

    edges = json.load(open(edges_path))
    plan = json.load(open(plan_path))
    sp = Splitter(edges, plan)
    texts = cm.library_files()
    placed = cm.place_all(texts)
    gen = None
    if generated_only:
        gen = {f for f, t in texts.items() if t.startswith("// Generated by tools/")} | generated_files()
    total = same = 0
    diffs = collections.Counter()
    examples = collections.defaultdict(list)
    for f, rows in placed.items():
        if gen is not None and f not in gen:
            continue
        ef = edges["files"].get(f)
        if ef is None:
            continue
        by_label = collections.defaultdict(list)
        for k, it in enumerate(ef["items"]):
            by_label[it["gen"] or it["name"]].append(f"{f}#{k}")
        seen = collections.Counter()
        for it, c, mod in rows:
            nids = by_label.get(it.label)
            if not nids:
                continue
            nid = nids[min(seen[it.label], len(nids) - 1)]
            seen[it.label] += 1
            pc = plan["items"][nid]
            want = (pc if pc in cm.rank else NS1_NAMES.get(pc, pc), sp.module_of[nid])
            got = (c, mod)
            total += 1
            if want == got:
                same += 1
            else:
                key = (f"{want[0]}->{got[0]}" if want[0] != got[0] else "module", it.kind, it.of or "")
                diffs[key] += 1
                examples[key].append(f"{it.label} [{f}] plan {want} map {got}")
    print(f"{same} / {total} items placed as the plan ({total - same} differ)")
    for k, v in diffs.most_common():
        print(f"  {v:5d} {k}: {'; '.join(examples[k][:3])}")
    return total - same


def compare_tree(cm, checkout, proto):
    """File-level and item-level comparison of a checkout written by the generators in split
    mode with the output of `prototype.py`, over the items of the generated files."""
    gen = generated_files()
    lib = cm.library_files()
    labels = collections.Counter()
    for f in gen:
        if f in lib:
            labels.update(it.label for it in parse(f, lib[f]) if it.kind not in ("use", "mod"))
    # names also defined by a hand-written file (a private homonym): ambiguous, left out
    for f in set(lib) - gen:
        for it in parse(f, lib[f]):
            labels.pop(it.label, None)

    def collect(root, package_of):
        out = collections.Counter()
        for d in sorted(os.listdir(os.path.join(root, "crates"))):
            pkg = package_of(d)
            if pkg is None or os.path.islink(os.path.join(root, "crates", d)):
                continue
            src = os.path.join(root, "crates", d, "src")
            for base, _dirs, files in os.walk(src):
                for fn in files:
                    if not fn.endswith(".cairo"):
                        continue
                    full = os.path.join(base, fn)
                    rel = os.path.relpath(full, src)
                    text = open(full, encoding="utf-8").read()
                    for it in parse(rel, text):
                        if it.kind in ("use", "mod") or it.label not in labels:
                            continue
                        if TEST_ONLY.search(text[it.start:it.end].split("\n{", 1)[0].split(" {", 1)[0]):
                            # test-only code: excluded by prototype.py, moved with its module here
                            continue
                        out[(pkg, rel, it.label)] += 1
        return out

    pkgs = {package_dir(p).split("/")[-1]: p for p in cm.packages()}
    mine = collect(checkout, pkgs.get)
    theirs = collect(proto, lambda d: None if d == "path_proof" else
                     ("nalgebra" if d == "facade" else "nalgebra_" + d))
    fm = {(p, m) for p, m, _l in mine}
    ft = {(p, m) for p, m, _l in theirs}
    print(f"files holding generated items: {len(fm)} written by the generators, {len(ft)} by "
          f"prototype.py, {len(fm & ft)} in common, {len(fm - ft)} only generators, "
          f"{len(ft - fm)} only prototype")
    for x in sorted(fm - ft)[:20]:
        print("  only generators:", x)
    for x in sorted(ft - fm)[:20]:
        print("  only prototype:", x)
    only_m, only_t = mine - theirs, theirs - mine
    print(f"generated items: {sum(mine.values())} placed by the generators, {sum(theirs.values())} "
          f"by prototype.py, {sum(only_m.values())} / {sum(only_t.values())} differ")
    for x in sorted(only_m)[:20]:
        print("  only generators:", x)
    for x in sorted(only_t)[:20]:
        print("  only prototype:", x)
    return len(fm ^ ft) + sum(only_m.values()) + sum(only_t.values())


def generated_files():
    """Module files (relative to `src/`) the two generators own (`shapegen.py`, `generate.py`)."""
    sys.path.insert(0, os.path.join(ROOT, "tools", "shapegen"))
    sys.path.insert(0, os.path.join(ROOT, "tools", "linalggen"))
    sys.dont_write_bytecode = True
    import dynamic
    import generate
    import shapes
    from model import ALL_SHAPES

    out = {f"base/{s.module}.cairo" for s in ALL_SHAPES}
    out |= {f"base/{m}.cairo" for m in shapes.SHARED_MODULES}
    out |= {f"linalg/{m}.cairo" for m in shapes.LINALG_MODULES}
    out |= {"base/dynamic/shapes.cairo", "macros.cairo"}
    out |= {f"base/dynamic/{m}.cairo" for m in dynamic.TYPE_MODULES.values()}
    lin = "crates/nalgebra/src/"
    out |= {k[len(lin):] for k in generate.outputs() if k.startswith(lin)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--map", help="crate map (default: tools/split/crates.toml)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--place", action="store_true")
    ap.add_argument("--compare-plan", nargs=2, metavar=("EDGES", "PLAN"))
    ap.add_argument("--generated", action="store_true", help="--compare-plan: generated files only")
    ap.add_argument("--split-map", metavar="OUT")
    ap.add_argument("--compare-tree", nargs=2, metavar=("CHECKOUT", "PROTO"),
                    help="a checkout the generators wrote in split mode against prototype.py's output")
    a = ap.parse_args()
    if a.split_map:
        with open(a.split_map, "w") as f:
            f.write(split_map_text(a.map or DEFAULT_MAP))
        return 0
    cm = load(a.map)
    if a.show:
        print(f"map: {os.path.relpath(cm.path, ROOT)} ({'single-crate' if cm.single else 'split'} mode)")
        for c in cm.order:
            print(f"  {c:20s} -> {cm.crates[c]}")
        print(f"{len(cm.rules)} rules")
    if a.place:
        placed = cm.place_all(cm.library_files())
        counts = collections.Counter(c for rows in placed.values() for _it, c, _m in rows)
        for c in cm.order:
            print(f"  {c:20s} {counts[c]:6d} items")
    if a.compare_tree:
        return 1 if compare_tree(cm, *a.compare_tree) else 0
    if a.compare_plan:
        return 1 if compare_plan(cm, *a.compare_plan, a.generated) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
