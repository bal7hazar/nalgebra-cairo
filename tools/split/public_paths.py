#!/usr/bin/env python3
"""The public surface of `nalgebra` and its proof across the package split (docs/SPLIT.md §3.4,
§12.3; WP 9-NS2). The CI job `Path proof` runs `--check`.

A PUBLIC PATH is `nalgebra::<module>::<name>` for
  * every public module reachable from `lib.cairo` through `pub mod` (file or inline), test-only
    modules excluded, and every module named `internal` (and below) excluded: former
    `pub(crate)` items that sub-crates must share live there with no stability promise and the
    facade never re-exports them (SPLIT §12.3);
  * every public item of such a module: `struct`, `enum`, `trait`, `fn`, `const`, `type`,
    `macro`, `impl` (named by users as `nalgebra::geometry::Point2Index`), and the trait of a
    public `#[generate_trait]` impl;
  * every name of every `pub use` of such a module (`pub use base::{Matrix3, ...}` at the root);
    a glob `pub use <package>::<path>::*` (the facade of a split) contributes every public name
    of the target module, followed across the packages of the crate map.
Code behind Scarb features counts (every feature of 0.1.0 is a default one).

    python3 tools/split/public_paths.py                 # print the current surface
    python3 tools/split/public_paths.py --freeze v0.1.0 # rewrite public_paths_0.1.0.txt from a tag
    python3 tools/split/public_paths.py --check         # the proof (CI): 1 + 2 below
    python3 tools/split/public_paths.py --check --no-build   # 1 only (seconds)

`--check`
  1. the current facade exports EXACTLY the frozen 0.1.0 surface (`public_paths_0.1.0.txt`) plus
     the public paths added since (`public_paths_added.txt`, one per line, optional): nothing
     more, nothing less, the difference is printed. No allowance: the moves NS3..NS10 accepted
     the "transient" anchor paths (a public impl moved to an anchor module of a crate the facade
     package still hosted, docs/SPLIT.md §15); since WP 9-NS11a the facade package hosts the
     facade alone, so any extra path fails;
  2. builds a consumer crate (a temporary directory, `nalgebra` by path with its default
     features) with one `use nalgebra::<path>;` per path and the `usage` module of SPLIT §3.4
     (methods, operators, products, norms, solves, LU, a cross-crate product, views, geometry
     and the construction macros with the imports a 0.1.0 user writes). `scarb build` failing
     fails the proof.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import consumer_cost as cc  # noqa: E402
import cratemap  # noqa: E402
from edges import expand_use, items as cut_items  # noqa: E402

FROZEN = os.path.join(HERE, "public_paths_0.1.0.txt")
ADDED = os.path.join(HERE, "public_paths_added.txt")
PUBLIC_KINDS = ("struct", "enum", "trait", "fn", "const", "type", "macro", "impl")
EXTERNAL = {"core", "starknet", "simba", "fixed"}


def kept(text):
    """Masked text with the test-only items blanked (offsets kept)."""
    m = cc.mask(text)
    out = list(m)
    for a in re.finditer(r"#\[cfg\(", m):
        close = cc.match_close(m, a.start() + 1, "[", "]")
        if close < 0 or not cc.test_only(m[a.end(): close - 1]):
            continue
        end = cc.item_end(m, close + 1)
        for i in range(a.start(), max(end, close)):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


class Tree:
    """The module trees of the packages of a source root (a checkout, or a `git archive`)."""

    def __init__(self, root, cm):
        self.root = root
        self.cm = cm
        self.cache = {}

    def package_src(self, package):
        d = os.path.join(self.root, cratemap.package_dir(package), "src")
        return d if os.path.isdir(d) else None

    def module(self, package, segs):
        """(masked text, raw text, file path) of module `segs` of `package`, or None."""
        key = (package, tuple(segs))
        if key in self.cache:
            return self.cache[key]
        res = None
        src = self.package_src(package)
        if src is not None:
            if not segs:
                p = os.path.join(src, "lib.cairo")
                if os.path.exists(p):
                    raw = open(p, encoding="utf-8").read()
                    res = (kept(raw), raw, p)
            else:
                parent = self.module(package, segs[:-1])
                if parent is not None:
                    res = self.child(parent, segs[-1])
        self.cache[key] = res
        return res

    def child(self, parent, name):
        masked, raw, path = parent
        for kind, nm, _g, _o, s, e in cut_items(masked):
            if kind != "mod" or nm != name:
                continue
            head = masked[s:e]
            brace = head.find("{")
            if brace >= 0 and (head.find(";") < 0 or brace < head.find(";")):
                close = cc.match_close(masked, s + brace, "{", "}")
                return (masked[s + brace + 1: close - 1], raw[s + brace + 1: close - 1], path)
            base = path[: -len(".cairo")] if not path.endswith("lib.cairo") else os.path.dirname(path)
            p = os.path.join(base, name + ".cairo")
            if os.path.exists(p):
                r = open(p, encoding="utf-8").read()
                return (kept(r), r, p)
        return None

    def exports(self, package, segs, seen=None):
        """{name: kind} of the public names of a module ("mod" for its public child modules)."""
        seen = seen or set()
        key = (package, tuple(segs))
        if key in seen:
            return {}
        seen.add(key)
        mod = self.module(package, segs)
        if mod is None:
            return {}
        masked, raw, _p = mod
        out = {}
        for kind, name, gen, _of, s, e in cut_items(masked):
            text = masked[s:e]
            if not re.match(r"pub\s", text):
                continue
            if kind == "mod":
                if name != "internal":
                    out[name] = "mod"
            elif kind in PUBLIC_KINDS and name:
                out[name] = kind
                if gen:
                    out[gen] = "trait"
            elif kind == "use":
                tree = re.sub(r"\s+", " ", raw[s:e]).split("use", 1)[1].rsplit(";", 1)[0]
                for full in expand_use(tree.strip()):
                    alias = None
                    if " as " in full:
                        full, alias = [x.strip() for x in full.split(" as ")]
                    parts = [x.strip() for x in full.split("::")]
                    if parts[-1] == "*":
                        target = self.resolve(package, segs, parts[:-1])
                        if target is not None:
                            for n, k in self.exports(*target, seen=seen).items():
                                out.setdefault(n, k)
                        continue
                    if parts[-1] == "self":
                        parts = parts[:-1]
                    out[alias or parts[-1]] = "use"
        return out

    def resolve(self, package, segs, parts):
        """(package, segments) a `use` path names, from module `segs` of `package`."""
        if not parts:
            return None
        if parts[0] == "crate":
            return package, parts[1:]
        if parts[0] == "super" or parts[0] == "self":
            cur = list(segs)
            while parts and parts[0] in ("super", "self"):
                if parts[0] == "super":
                    cur = cur[:-1]
                parts = parts[1:]
            return package, cur + parts
        if parts[0] in EXTERNAL:
            return None
        if self.package_src(parts[0]) is not None:
            return parts[0], parts[1:]
        return package, list(segs) + parts  # a child module

    def public_modules(self, package):
        """Module paths (segment lists) of the public module tree of `package`."""
        out = []

        def walk(segs):
            out.append(segs)
            for name, kind in sorted(self.own_modules(package, segs).items()):
                walk(segs + [name])

        walk([])
        return out

    def own_modules(self, package, segs):
        mod = self.module(package, segs)
        if mod is None:
            return {}
        masked = mod[0]
        return {nm: "mod" for kind, nm, _g, _o, s, e in cut_items(masked)
                if kind == "mod" and re.match(r"pub\s", masked[s:e]) and nm != "internal"
                and self.module(package, segs + [nm]) is not None}


def surface(root, package="nalgebra", cm=None):
    """Sorted public paths of `package` in the checkout at `root`."""
    cm = cm or cratemap.load()
    tree = Tree(root, cm)
    paths = set()
    for segs in tree.public_modules(package):
        prefix = "::".join([package] + segs)
        if segs:
            paths.add(prefix)
        for name in tree.exports(package, segs):
            if name == "internal":
                continue
            paths.add(f"{prefix}::{name}")
    return sorted(paths)


def surface_of_ref(ref):
    tmp = tempfile.mkdtemp(prefix="public_paths_")
    try:
        archive = subprocess.run(["git", "archive", ref, "crates"], cwd=ROOT, check=True,
                                 capture_output=True).stdout
        subprocess.run(["tar", "-x", "-C", tmp], input=archive, check=True)
        return surface(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def read_list(path):
    if not os.path.exists(path):
        return []
    return [line.strip() for line in open(path) if line.strip() and not line.startswith("#")]


def workspace_version(name):
    text = open(os.path.join(ROOT, "Scarb.toml")).read()
    m = re.search(rf'^{name}\s*=\s*"([^"]+)"', text, re.M)
    return m.group(1) if m else None


def build_consumer(paths, keep=None):
    """`scarb build` of a consumer naming every path + the usage module; True if it builds."""
    sys.path.insert(0, HERE)
    from facade import USAGE

    d = keep or tempfile.mkdtemp(prefix="path_proof_")
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    cm = cratemap.load()
    nalgebra = os.path.join(ROOT, cratemap.package_dir(cm.facade_package))
    fixed = workspace_version("fixed")
    with open(os.path.join(d, "Scarb.toml"), "w") as f:
        f.write('[package]\nname = "path_proof"\nversion = "0.1.0"\nedition = "2024_07"\n\n'
                f'[dependencies]\nnalgebra = {{ path = "{nalgebra}" }}\n'
                + (f'fixed = "{fixed}"\n' if fixed else ""))
    body = [f"// Every public path of nalgebra ({len(paths)}), generated by tools/split/public_paths.py."]
    for i, p in enumerate(paths):
        body.append(f"#[allow(unused_imports)]\nmod p{i} {{\n    use {p};\n}}")
    body.append(USAGE)
    with open(os.path.join(d, "src", "lib.cairo"), "w") as f:
        f.write("\n".join(body) + "\n")
    print(f"path proof: building a consumer of {len(paths)} paths in {d}", flush=True)
    r = subprocess.run(["scarb", "--manifest-path", os.path.join(d, "Scarb.toml"), "build"])
    if keep is None and r.returncode == 0:
        shutil.rmtree(d, ignore_errors=True)
    return r.returncode == 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="the proof (surface equality + consumer build)")
    ap.add_argument("--no-build", action="store_true", help="--check without the consumer build")
    ap.add_argument("--freeze", metavar="REF", help="write public_paths_0.1.0.txt from a git ref")
    ap.add_argument("--root", default=ROOT, help="checkout to read (default: this repository)")
    ap.add_argument("--keep", metavar="DIR", help="build the consumer in DIR and keep it")
    a = ap.parse_args(argv)
    if a.freeze:
        paths = surface_of_ref(a.freeze)
        with open(FROZEN, "w") as f:
            f.write(f"# Public paths of nalgebra {a.freeze} (tools/split/public_paths.py --freeze "
                    f"{a.freeze}): {len(paths)} paths. Never edited by hand.\n")
            f.write("\n".join(paths) + "\n")
        print(f"{len(paths)} public paths frozen in {os.path.relpath(FROZEN, ROOT)}")
        return 0
    paths = surface(a.root)
    if not a.check:
        print("\n".join(paths))
        return 0
    expected = set(read_list(FROZEN)) | set(read_list(ADDED))
    if not expected:
        sys.exit(f"error: {FROZEN} is missing (--freeze v0.1.0)")
    have = set(paths)
    missing, extra = sorted(expected - have), sorted(have - expected)
    for p in missing:
        print(f"MISSING {p}")
    for p in extra:
        print(f"EXTRA   {p}")
    print(f"public paths: {len(expected)} expected (0.1.0 + added), {len(have)} exported, "
          f"{len(missing)} missing, {len(extra)} extra")
    if missing or extra:
        print("the facade must export exactly the 0.1.0 surface (docs/SPLIT.md §12.3); a new public "
              "item is added to tools/split/public_paths_added.txt in its own PR", file=sys.stderr)
        return 1
    if a.no_build:
        return 0
    if not build_consumer(paths, a.keep):
        print("path proof: the consumer does not build", file=sys.stderr)
        return 1
    print("path proof: every public path resolves and the usage checks compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
