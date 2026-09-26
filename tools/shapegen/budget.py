#!/usr/bin/env python3
"""Compile-memory ablations of `crates/nalgebra` (WP 8.1d; `DESIGN.md` §2.11, raw results:
`budget-results.jsonl`).

Copies the library, `crates/testing` and one test package into a throw-away workspace
(`/tmp/nalgebra-budget/<variant>`), applies ONE variant (a module removed, a family duplicated,
a feature switched off...), then measures a COLD build (`SCARB_INCREMENTAL=false`, empty
`target/`) with `/usr/bin/time -v`: peak RSS (`Maximum resident set size`), user + system CPU
and wall time. The machine-wide build lock is taken BEFORE the timer starts, so waiting for it
is not counted.

    tools/shapegen/budget.py <variant> [--target lib|test] [--package PKG] [--keep]

Two kinds of variants:

* removal (`-x`): the module and its re-exports are removed; exact, but only possible for the
  leaves nothing else uses (`linalg`, `statistics`, `blas`; `cg` together with the geometry
  modules that use it);
* duplication (`+x`): a second copy of the code is compiled next to the original (a copied
  module, or every generated method of a family rendered twice under a `_dup` name); the growth
  estimates the marginal cost of the parts that cannot be removed (the base families and
  `geometry` are used everywhere). Calibrated against the removals (`+statistics` / `-statistics`,
  `+linalg` / `-linalg`).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

TOOL = Path(__file__).resolve().parent
ROOT = TOOL.parents[1]
sys.path.insert(0, str(TOOL))

WORK = Path(os.environ.get("BUDGET_WORK", "/tmp/nalgebra-budget"))
LOCK = os.environ.get("HEAVY_BUILD_LOCK", str(Path.home() / "orchestrator" / "heavy-build.lock"))
RESULTS = WORK / "results.jsonl"
MEMBERS = ["crates/nalgebra", "crates/testing"]
SRC = "crates/nalgebra/src"


# --------------------------------------------------------------------------------------------
# Text surgery on the copied workspace
# --------------------------------------------------------------------------------------------


def drop_statements(text: str, pattern: str) -> str:
    """`text` without the statements (up to their `;`, multi-line) starting with `pattern`, and
    without the attribute lines right above them."""
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        if re.match(pattern, lines[i].strip()):
            while out and out[-1].strip().startswith("#[cfg(feature"):
                out.pop()
            while not lines[i].rstrip().endswith(";"):
                i += 1
            i += 1
            continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def drop_names(text: str, names: set[str]) -> str:
    """`text` with `names` removed from every `use a::{..}` list (empty lists removed)."""
    def fix(m: re.Match) -> str:
        kept = [n for n in re.split(r"\s*,\s*", m.group(2).strip().rstrip(",")) if n
                and n not in names]
        if not kept:
            return ""
        return m.group(0).replace(m.group(2), ", ".join(kept))
    return re.sub(r"(?:#\[cfg\(feature: '\w+'\)\]\s*)?((?:pub )?use [\w:]+::)\{([^}]*)\};",
                  fix, text)


def remove_module(ws: Path, parent: str, module: str, file_or_dir: list[str]):
    """Removes `pub mod <module>;` and `pub use <module>::..;` from `parent` (a `.cairo` file of
    the copy), the module's files, and every name it exported from `lib.cairo`'s lists."""
    p = ws / SRC / parent
    text = p.read_text()
    names = set()
    for m in re.finditer(rf"pub use {module}::(\{{[^}}]*\}}|\w+);", text):
        names |= {n.strip() for n in m.group(1).strip("{}").split(",") if n.strip()}
    text = drop_statements(text, rf"(pub )?mod {module};")
    text = drop_statements(text, rf"pub use {module}::")
    p.write_text(text)
    for rel in file_or_dir:
        f = ws / SRC / rel
        shutil.rmtree(f) if f.is_dir() else f.unlink(missing_ok=True)
    lib = ws / SRC / "lib.cairo"
    lib.write_text(drop_names(lib.read_text(), names))
    return names


def copy_module(ws: Path, parent: str, module: str, copy: str):
    """Adds `mod <copy>;` next to `mod <module>;`: a second copy of the module's files, with its
    `crate::...::<module>` paths pointing to the copy."""
    base = ws / SRC / Path(parent).parent if parent != "lib.cairo" else ws / SRC
    sub = "" if parent == "lib.cairo" else Path(parent).stem + "/"
    src_file, src_dir = base / f"{sub}{module}.cairo", base / f"{sub}{module}"
    path = f"crate::{sub.replace('/', '::')}{module}"
    new = f"crate::{sub.replace('/', '::')}{copy}"
    shutil.copyfile(src_file, base / f"{sub}{copy}.cairo")
    if src_dir.is_dir():
        shutil.copytree(src_dir, base / f"{sub}{copy}")
    for f in [base / f"{sub}{copy}.cairo", *(base / f"{sub}{copy}").rglob("*.cairo")]:
        if f.is_file():
            f.write_text(re.sub(rf"{re.escape(path)}\b", new, f.read_text()))
    p = ws / SRC / parent
    p.write_text(p.read_text().replace(f"pub mod {module};",
                                       f"pub mod {module};\n#[allow(unused)]\nmod {copy};", 1))


# --------------------------------------------------------------------------------------------
# Generator-level variants (monkeypatched `tools/shapegen` modules, output copied into the copy)
# --------------------------------------------------------------------------------------------


def dup_fns(fns):
    import library as L
    out = list(fns)
    for f in fns:
        sig = re.sub(rf"\bfn {f.name}\b", f"fn {f.name}_dup", f.sig, count=1)
        out.append(L.Fn(f.name + "_dup", "", sig, f.body, f.inline))
    return out


def regenerate(ws: Path, patch):
    """Runs the library part of the generator with `patch()` applied, into the copy."""
    import importlib
    import completion
    import functional
    import library
    import views
    import shapes
    import shapegen
    patch(completion=completion, functional=functional, views=views, shapes=shapes,
          library=library)
    tmp = WORK / ".gen"
    shutil.rmtree(tmp, ignore_errors=True)
    outputs = shapegen.library_outputs(tmp)
    for dst, gen in outputs.items():
        rel = dst.relative_to(ROOT)
        shutil.copyfile(gen, ws / rel)
    shutil.rmtree(tmp, ignore_errors=True)
    for m in (completion, functional, views, shapes, library):
        importlib.reload(m)


def dup_family(module_name: str, only=None):
    def patch(**mods):
        m = mods[module_name]
        orig = m.missing

        def missing(s, have):
            fns = orig(s, have)
            dup = [f for f in fns if only is None or only(f)]
            return fns + dup_fns(dup)[len(dup):]
        m.missing = missing
    return patch


def drop_family(module_name: str, items: bool = True):
    def patch(**mods):
        m = mods[module_name]
        m.missing = lambda s, have: []
        if items:
            m.items = lambda s: []
            m.uses = lambda s: []
    return patch


def keep_only(module_name: str, drop):
    def patch(**mods):
        m = mods[module_name]
        orig = m.missing
        m.missing = lambda s, have: [f for f in orig(s, have) if not drop(f)]
    return patch


def closure(f) -> bool:
    return "core::ops::Fn<" in f.sig or "Func" in f.sig


def swizzle(f) -> bool:
    return re.fullmatch(r"[xyzwab]{2,3}", f.name) is not None


def strip_inline(ws: Path, files):
    n = 0
    for f in files:
        text = f.read_text()
        n += text.count("#[inline(always)]\n")
        f.write_text(re.sub(r"^[ \t]*#\[inline\(always\)\]\n", "", text, flags=re.M))
    return n


def generated_files(ws: Path):
    return [f for f in (ws / SRC / "base").glob("*.cairo")
            if f.read_text().startswith("// Generated by tools/shapegen")]


P11A = ["transform", "transform2", "transform3", "affine2", "affine3", "projective2",
        "projective3"]


def remove_p11a(ws: Path):
    for m in P11A:
        remove_module(ws, "geometry.cairo", m, [f"geometry/{m}.cairo", f"geometry/{m}"])
    # `Orthographic3` / `Perspective3` convert into `Projective3`: drop those methods.
    for f in ("orthographic3", "perspective3"):
        p = ws / SRC / "geometry" / f"{f}.cairo"
        text = p.read_text()
        text = re.sub(r"use super::projective3::[^;]*;\n", "", text)
        text = re.sub(r"(\s*///[^\n]*\n)*\s*(#\[[^\n]*\]\n\s*)*fn (as|to)_projective\(.*?\n    \}\n",
                      "\n", text, flags=re.S)
        p.write_text(text)


def set_features(ws: Path, package: str, spec: str):
    p = ws / "crates" / package / "Scarb.toml"
    text = p.read_text()
    text = re.sub(r'nalgebra = \{ path = "../nalgebra"[^}\n]*\}',
                  f'nalgebra = {{ path = "../nalgebra"{", " + spec if spec else ""} }}', text)
    p.write_text(text)


def split_module(ws: Path, module: str):
    """`base/<module>.cairo` as one submodule per `#[generate_trait]` impl (`base/<module>/*.cairo`),
    the head of the file (`use` lines, private helpers) repeated in each."""
    p = ws / SRC / "base" / f"{module}.cairo"
    text = p.read_text()
    starts = [m.start() for m in re.finditer(r"(?:///[^\n]*\n)*#\[generate_trait\]", text)]
    head = text[:starts[0]]
    uses = "\n".join(re.sub(r"\bsuper::", "crate::base::", line)
                     for line in head.split("\n") if line.startswith("use "))
    parts = [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]
    (ws / SRC / "base" / module).mkdir()
    mods, exports = [], []
    for i, part in enumerate(parts):
        trait = re.search(r"\bof (\w+)<", part).group(1)
        name = f"part{i}"
        (ws / SRC / "base" / module / f"{name}.cairo").write_text(
            "#[allow(unused_imports)]\n" + uses.replace("\nuse ", "\n#[allow(unused_imports)]\nuse ")
            + "\n\n" + part)
        mods.append(f"pub mod {name};")
        exports.append(f"pub use {name}::{trait};")
    p.write_text("\n".join(line for line in head.split("\n") if not line.startswith("use "))
                 + "\n" + "\n".join(mods + exports) + "\n")


def strip_derives(ws: Path, derives: set[str]):
    for f in (ws / SRC).rglob("*.cairo"):
        text = f.read_text()

        def fix(m: re.Match) -> str:
            kept = [d.strip() for d in m.group(1).split(",") if d.strip() not in derives]
            return f"#[derive({', '.join(kept)})]"
        f.write_text(re.sub(r"#\[derive\(([^)]*)\)\]", fix, text))


def empty_tests(ws: Path):
    """The test package reduced to one test: the library's own cost in a test unit."""
    for pkg in (ws / "crates").iterdir():
        if pkg.name.startswith(("shapes_tests", "tests_")):
            shutil.rmtree(pkg / "src")
            (pkg / "src").mkdir()
            (pkg / "src" / "lib.cairo").write_text(
                "#[cfg(test)]\nmod tests {\n    use nalgebra::Vector3;\n\n    #[test]\n"
                "    fn one() {\n        let v: Vector3<u32> = Vector3 { x: 1, y: 2, z: 3 };\n"
                "        assert!(v.x == 1);\n    }\n}\n")


VARIANTS = {
    "full": ("HEAD", lambda ws: None),
    "-linalg": ("without `linalg`", lambda ws: remove_module(ws, "lib.cairo", "linalg",
                                                             ["linalg.cairo", "linalg"])),
    "-statistics": ("without `base::statistics`", lambda ws: remove_module(
        ws, "base.cairo", "statistics", ["base/statistics.cairo"])),
    "-blas": ("without `base::blas`", lambda ws: remove_module(
        ws, "base.cairo", "blas", ["base/blas.cairo"])),
    "-statistics-blas": ("without `statistics` and `blas`", lambda ws: (
        remove_module(ws, "base.cairo", "statistics", ["base/statistics.cairo"]),
        remove_module(ws, "base.cairo", "blas", ["base/blas.cairo"]))),
    "-p11a": ("without the geometry modules that use `cg` (P11a)", remove_p11a),
    "-cg-p11a": ("without `cg` and P11a", lambda ws: (
        remove_p11a(ws), remove_module(ws, "base.cairo", "cg", ["base/cg.cairo"]))),
    "+statistics": ("`statistics` twice", lambda ws: copy_module(
        ws, "base.cairo", "statistics", "statistics_dup")),
    "+linalg": ("`linalg` twice", lambda ws: copy_module(ws, "lib.cairo", "linalg",
                                                          "linalg_dup")),
    "+geometry": ("`geometry` twice", lambda ws: copy_module(ws, "lib.cairo", "geometry",
                                                              "geometry_dup")),
    "+blas": ("`blas` twice", lambda ws: copy_module(ws, "base.cairo", "blas", "blas_dup")),
    "+cg": ("`cg` twice", lambda ws: copy_module(ws, "base.cairo", "cg", "cg_dup")),
    "+completion": ("the completion methods (P02) twice",
                    lambda ws: regenerate(ws, dup_family("completion"))),
    "+functional": ("the functional methods (P03) twice",
                    lambda ws: regenerate(ws, dup_family("functional"))),
    "-functional": ("without the functional family (P03: methods, operators, `Norm` impls)",
                    lambda ws: regenerate(ws, drop_family("functional"))),
    "-functional-methods": ("without the functional methods (P03), its items kept",
                            lambda ws: regenerate(ws, drop_family("functional", False))),
    "-closures": ("without the functional methods taking a closure (`map`, `fold`, `apply`...)",
                  lambda ws: regenerate(ws, keep_only("functional", closure))),
    "-functional-plain": ("without the functional methods taking no closure (edition, in place)",
                          lambda ws: regenerate(ws, keep_only("functional",
                                                              lambda f: not closure(f)))),
    "+views": ("the view methods (P04/P05, swizzles included) twice",
               lambda ws: regenerate(ws, dup_family("views"))),
    "+swizzles": ("the swizzle methods twice",
                  lambda ws: regenerate(ws, dup_family("views", swizzle))),
    "split-statistics": ("`statistics` as 36 submodules (one per shape)",
                         lambda ws: split_module(ws, "statistics")),
    "split-blas": ("`blas` as one submodule per `#[generate_trait]`",
                   lambda ws: split_module(ws, "blas")),
    "-debug-hash": ("no derived `Debug` / `Hash` in the library",
                    lambda ws: strip_derives(ws, {"Debug", "Hash"})),
    "empty-lib": ("an empty library (the fixed cost: corelib, simba, the plugins)",
                  lambda ws: (ws / SRC / "lib.cairo").write_text("pub fn f() {}\n")),
    "empty-tests": ("the test package reduced to one trivial test", empty_tests),
    "-inline": ("no `#[inline(always)]` in the generated base modules",
                lambda ws: strip_inline(ws, generated_files(ws))),
    "-inline-all": ("no `#[inline(always)]` anywhere in the library",
                    lambda ws: strip_inline(ws, (ws / SRC).rglob("*.cairo"))),
}


# --------------------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------------------


def make_workspace(label: str, package: str | None) -> Path:
    ws = WORK / re.sub(r"[^\w.-]", "_", label)
    shutil.rmtree(ws, ignore_errors=True)
    members = MEMBERS + ([f"crates/{package}"] if package else [])
    if package:
        deps = re.findall(r'path = "\.\./(\w+)"', (ROOT / "crates" / package / "Scarb.toml").read_text())
        members += [f"crates/{d}" for d in deps if f"crates/{d}" not in members]
    for m in members:
        shutil.copytree(ROOT / m, ws / m, ignore=shutil.ignore_patterns("target", ".snfoundry_cache"))
    manifest = (ROOT / "Scarb.toml").read_text()
    manifest = re.sub(r"members = \[[^\]]*\]",
                      "members = [" + ", ".join(f'"{m}"' for m in members) + "]", manifest)
    (ws / "Scarb.toml").write_text(manifest)
    shutil.copyfile(ROOT / "Scarb.lock", ws / "Scarb.lock")
    return ws


def measure(ws: Path, args: list[str]) -> dict:
    env = dict(os.environ, HEAVY_BUILD_LOCK_HELD="1")
    env.setdefault("SCARB_INCREMENTAL", "false")
    shutil.rmtree(ws / "target", ignore_errors=True)
    timing = ws / "time.txt"
    cmd = ["flock", LOCK, "nice", "-n", "10", "/usr/bin/time", "-v", "-o", str(timing), *args]
    run = subprocess.run(cmd, cwd=ws, env=env, capture_output=True, text=True)
    log = run.stdout + run.stderr
    t = timing.read_text() if timing.exists() else ""

    def field(name: str) -> float:
        m = re.search(rf"{re.escape(name)}: ([\d.:]+)", t)
        if not m:
            return float("nan")
        v = m.group(1)
        if ":" in v:
            parts = [float(x) for x in v.split(":")]
            return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
        return float(v)
    ok = run.returncode == 0
    return {"ok": ok, "rss_mb": round(field("Maximum resident set size (kbytes)") / 1024),
            "cpu_s": round(field("User time (seconds)") + field("System time (seconds)"), 1),
            "wall_s": round(field("Elapsed (wall clock) time (h:mm:ss or m:ss)"), 1),
            "units": re.findall(r"Compiling (.*?) v", log), "log": "" if ok else log[-4000:],
            "output": log}


def library_lines(ws: Path) -> int:
    return sum(len(f.read_text().split("\n")) for f in (ws / SRC).rglob("*.cairo"))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("variant", help=", ".join(VARIANTS))
    p.add_argument("--target", choices=["lib", "test", "selftest"], default="lib",
                   help="lib: `scarb build -p nalgebra`; test: `scarb build --test` of --package; "
                        "selftest: `scarb build --test -p nalgebra` (the library's own tests)")
    p.add_argument("--package", default="shapes_tests_core",
                   help="the test package (directory under crates/) of --target test")
    p.add_argument("--features", default=None,
                   help="replaces the test package's dependency options on nalgebra: e.g. "
                        "'default-features = false', or '' for the defaults")
    p.add_argument("--scarb-args", default=None, help="extra arguments of the scarb command")
    p.add_argument("--snforge", action="store_true",
                   help="--target test: measure `snforge test` (build, CASM, run) and keep its "
                        "output in <work>/<variant>.snforge.txt")
    p.add_argument("--keep", action="store_true", help="keep the copied workspace")
    args = p.parse_args()
    desc, apply = VARIANTS[args.variant]
    package = args.package if args.target == "test" else None
    label = f"{args.variant}-{args.target}"
    ws = make_workspace(label, package)
    apply(ws)
    if args.features is not None and package:
        set_features(ws, package, args.features)
    if args.target == "lib":
        cmd = ["scarb", "build", "-p", "nalgebra"]
    elif args.target == "selftest":
        cmd = ["scarb", "build", "--test", "-p", "nalgebra"]
    else:
        name = re.search(r'name = "([^"]+)"',
                         (ws / "crates" / package / "Scarb.toml").read_text()).group(1)
        cmd = (["snforge", "test", "-p", name] if args.snforge
               else ["scarb", "build", "--test", "-p", name])
    if args.scarb_args:
        cmd += args.scarb_args.split()
    started = time.time()
    r = measure(ws, cmd)
    r.update(variant=args.variant, desc=desc, target=args.target,
             package=package, features=args.features, scarb_args=args.scarb_args,
             incremental=os.environ.get("SCARB_INCREMENTAL", "false"),
             lines=library_lines(ws), at=time.strftime("%Y-%m-%d %H:%M"),
             total_s=round(time.time() - started))
    r["snforge"] = args.snforge
    output = r.pop("output")
    if args.snforge:
        (WORK / f"{label}.snforge.txt").write_text(output)
        m = re.search(r"Tests: (\d+) passed, (\d+) failed", output)
        r["tests"] = m.group(0) if m else "?"
    WORK.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a") as f:
        f.write(json.dumps({k: v for k, v in r.items() if k != "log"}) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "log"}))
    if not r["ok"]:
        print(r["log"], file=sys.stderr)
    if not args.keep:
        shutil.rmtree(ws, ignore_errors=True)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
