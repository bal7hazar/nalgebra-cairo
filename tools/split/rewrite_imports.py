#!/usr/bin/env python3
"""Rewrite the `use nalgebra::...` lines of test packages to the sub-crates of the crate map
(docs/SPLIT.md §5; WP 9-NS2, used by the move WPs NS3..NS11).

Every name a `use nalgebra::<path>::X` statement imports is resolved against the library through
the crate map (`tools/split/crates.toml`, or `$NALGEBRA_CRATE_MAP` / `--map`): the crate that
defines X and the module X lives in (its 0.1.0 module, or the anchor module an impl moved to).
The statement becomes one `use <package>::<module>::{...};` per target module
(`use nalgebra_core::base::matrix3::{Matrix3, Matrix3Trait};`); names the facade still hosts
(root functions, macros) keep their `nalgebra::` path, and a statement whose names all stay in
the facade is left byte for byte. The package's `[dependencies]` line `nalgebra = ...` becomes
one path dependency per sub-crate it now imports (lowest crate first), plus the `nalgebra` line
itself if the facade is still used.

Dry run by default (a unified diff on stdout); `--write` applies it (run `scarb fmt` afterwards).
In single-crate mode (the committed map) every name stays in `nalgebra`: nothing changes.

    python3 tools/split/rewrite_imports.py                          # every test package, dry run
    python3 tools/split/rewrite_imports.py crates/tests_base --write
    NALGEBRA_CRATE_MAP=/tmp/split.toml python3 tools/split/rewrite_imports.py crates/tests_linalg

Not rewritten (reported as warnings): glob imports (`use nalgebra::*`), module imports
(`use nalgebra::base::errors;`), names the library does not define at top level (constants of an
inline `errors` module...) or defines several times without a module hint, and inline paths
(`nalgebra::base::X::new()` in code). Generated test packages (`tools/shapegen/tests_*.py`,
`tools/linalggen`) must get the same rewrite in their generator (`rewrite_text`), not by hand.
"""
import argparse
import collections
import difflib
import glob
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import cratemap  # noqa: E402
from edges import expand_use  # noqa: E402

USE_NALGEBRA = re.compile(r"^(?P<indent>[ \t]*)(?P<vis>pub\s+)?use\s+nalgebra(?P<tree>::[^;]*);", re.M)
INLINE = re.compile(r"(?<![\w:])nalgebra::\w")


class Resolver:
    def __init__(self, cm):
        self.cm = cm
        placed = cm.place_all(cm.library_files())
        self.by_name = collections.defaultdict(list)
        for f, rows in placed.items():
            for it, crate, module in rows:
                if it.kind in ("use", "mod"):
                    continue
                for nm in {it.name, it.gen} - {None, ""}:
                    self.by_name[nm].append((f, cm.crates[crate], module, crate))

    def resolve(self, segs):
        """(package, module path segments, crate) of `nalgebra::<segs>`, or (None, reason)."""
        name, hint = segs[-1], "/".join(segs[:-1])
        cands = self.by_name.get(name, [])
        if not cands:
            return None, f"`{name}` is not a top-level item of the library"
        if len({(p, m) for _f, p, m, _c in cands}) > 1:
            exact = [c for c in cands if c[0][: -len(".cairo")] == hint]
            under = [c for c in cands if hint and c[0].startswith(hint + "/")]
            cands = exact or under or cands
        if len({(p, m) for _f, p, m, _c in cands}) > 1:
            return None, f"`{name}` is defined {len(cands)} times ({', '.join(sorted(c[0] for c in cands))})"
        _f, pkg, module, crate = cands[0]
        return (pkg, module[: -len(".cairo")].split("/"), crate), None


def rewrite_text(text, resolver, where="", warnings=None):
    """The text with its `use nalgebra::...` statements rewritten; the packages it now imports."""
    cm = resolver.cm
    used = set()
    warnings = warnings if warnings is not None else []

    def sub(m):
        paths = expand_use("nalgebra" + m.group("tree"))
        groups = collections.OrderedDict()
        facade_only = True
        for full in paths:
            alias = None
            if " as " in full:
                full, alias = [x.strip() for x in full.split(" as ")]
            segs = [s.strip() for s in full.split("::")][1:]
            last = segs[-1] if segs else ""
            res, why = (None, "glob import") if last == "*" else (
                (None, "module import") if (last == "self" or re.fullmatch(r"[a-z_][a-z0-9_]*", last)
                                            and not resolver.by_name.get(last)) else resolver.resolve(segs))
            item = last + (f" as {alias}" if alias else "")
            if res is None or res[0] == cm.facade_package:
                if res is None:
                    warnings.append(f"{where}: kept `nalgebra::{'::'.join(segs)}` ({why})")
                used.add(cm.facade_package)
                groups.setdefault(("nalgebra", tuple(segs[:-1])), []).append(item)
                continue
            facade_only = False
            pkg, module, _crate = res
            used.add(pkg)
            groups.setdefault((pkg, tuple(module)), []).append(item)
        if facade_only:
            return m.group(0)
        out = []
        for (pkg, module), names in groups.items():
            prefix = "::".join([pkg, *module])
            names = sorted(set(names))
            body = names[0] if len(names) == 1 else "{" + ", ".join(names) + "}"
            out.append(f"{m.group('indent')}{m.group('vis') or ''}use {prefix}::{body};")
        return "\n".join(out)

    new = USE_NALGEBRA.sub(sub, text)
    for m in INLINE.finditer(new):
        line = new.count("\n", 0, m.start()) + 1
        if not new[new.rfind("\n", 0, m.start()) + 1 : m.start()].lstrip().startswith(("use ", "pub use ", "//")):
            warnings.append(f"{where}:{line}: inline path `nalgebra::...` left unchanged")
    return new, used


def rewrite_manifest(text, pkg_dir, used, cm):
    """`[dependencies]`: the `nalgebra = ...` line replaced by the sub-crates the sources import."""
    m = re.search(r"^nalgebra\s*=.*$", text, re.M)
    if m is None or used <= {cm.facade_package}:
        return text
    subs = sorted((p for p in used if p != cm.facade_package), key=cm.package_rank)
    lines = []
    for p in subs:
        rel = os.path.relpath(os.path.join(ROOT, cratemap.package_dir(p)), pkg_dir)
        lines.append(f'{p} = {{ path = "{rel}" }}')
    if cm.facade_package in used:
        lines.append(m.group(0))
    return text[: m.start()] + "\n".join(lines) + text[m.end():]


def test_packages():
    return sorted(d for pat in ("crates/tests_*", "crates/shapes_tests_*")
                  for d in glob.glob(os.path.join(ROOT, pat)) if os.path.isfile(os.path.join(d, "Scarb.toml")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("packages", nargs="*", help="test package directories (default: every test package)")
    ap.add_argument("--map", help="crate map (default: tools/split/crates.toml or $NALGEBRA_CRATE_MAP)")
    ap.add_argument("--write", action="store_true", help="apply (default: dry run, a diff)")
    a = ap.parse_args(argv)
    cm = cratemap.load(a.map)
    resolver = Resolver(cm)
    pkgs = [os.path.abspath(p) for p in a.packages] or test_packages()
    changed_files = 0
    warnings = []
    for d in pkgs:
        used = set()
        edits = {}
        for f in sorted(glob.glob(os.path.join(d, "**", "*.cairo"), recursive=True)):
            text = open(f, encoding="utf-8").read()
            new, u = rewrite_text(text, resolver, os.path.relpath(f, ROOT), warnings)
            used |= u
            if new != text:
                edits[f] = (text, new)
        man = os.path.join(d, "Scarb.toml")
        mt = open(man, encoding="utf-8").read()
        mn = rewrite_manifest(mt, d, used, cm)
        if mn != mt:
            edits[man] = (mt, mn)
        for f, (old, new) in edits.items():
            changed_files += 1
            if a.write:
                with open(f, "w", encoding="utf-8") as fh:
                    fh.write(new)
            else:
                rel = os.path.relpath(f, ROOT)
                sys.stdout.writelines(difflib.unified_diff(
                    old.splitlines(True), new.splitlines(True), f"a/{rel}", f"b/{rel}"))
        if used - {cm.facade_package}:
            print(f"# {os.path.relpath(d, ROOT)}: imports "
                  + ", ".join(sorted(used, key=cm.package_rank)), file=sys.stderr)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    mode = "single-crate" if cm.single else "split"
    print(f"rewrite_imports ({mode} mode): {len(pkgs)} package(s), {changed_files} file(s) "
          f"{'rewritten' if a.write else 'to rewrite'}, {len(warnings)} warning(s)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
