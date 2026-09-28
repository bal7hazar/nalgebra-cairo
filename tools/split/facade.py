#!/usr/bin/env python3
"""Facade prototype and public-path proof for a prototype split (WP 9-NS1 helper).

1. Rewrites the `facade` crate of a prototype workspace (`prototype.py`) into the facade `nalgebra`
   of the plan: the module tree of `crates/nalgebra/src` (public modules only), each facade module
   re-exporting `nalgebra_<crate>::<same path>::*` from every sub-crate that hosts that module, plus
   the original `pub use` lists of the module (root re-exports, `base::{Matrix3, ...}`), verbatim.
   The root functions (`root.cairo`) and the macros (`macros.cairo`) are the facade's own items.
2. Writes a consumer crate `path_proof` that names EVERY public path of 0.1.0 (every public item
   of every public module, every name of every `pub use`), calls the construction macros and
   resolves methods / operators / products / norms / solves / decompositions with the imports a
   0.1.0 user writes. `scarb build -p path_proof` succeeding is the proof that the facade keeps
   the paths.

    python3 tools/split/facade.py EDGES.json PROTO_DIR
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import consumer_cost as cc  # noqa: E402
from edges import USE, expand_use, kept_text_and_excluded  # noqa: E402

SRC = os.path.join(ROOT, "crates", "nalgebra", "src")
OWN = {"root", "macros"}  # modules whose items the facade itself hosts


def public_modules():
    """{module path ("base/matrix3"): [public child names]} for the modules reachable through
    `pub mod` declarations, test-only ones excluded."""
    out = {}

    def walk(path, mod):
        text, excluded = kept_text_and_excluded(path)
        children = []
        for m in re.finditer(r"^[ \t]*(pub(?:\([\w:]+\))?[ \t]+)?mod[ \t]+(\w+)[ \t]*;", text, re.M):
            if m.group(1) is None or m.group(1).strip() != "pub":
                continue
            name = m.group(2)
            child = cc.module_file(path, name)
            if child is None:
                continue
            children.append(name)
            walk(child, f"{mod}/{name}" if mod else name)
        for m in re.finditer(r"^pub mod (\w+) \{", text, re.M):
            children.append(m.group(1))
            out[f"{mod}/{m.group(1)}" if mod else m.group(1)] = []
        out[mod] = children

    walk(os.path.join(SRC, "lib.cairo"), "")
    return out


def pub_uses(mod):
    """The `pub use` statements of module `mod` (with the attributes right above them)."""
    path = os.path.join(SRC, (mod or "lib") + ".cairo")
    if not os.path.exists(path):
        return []
    raw = open(path).read()
    _, excluded = kept_text_and_excluded(path)
    masked = cc.mask(raw)
    out = []
    for m in re.finditer(r"^pub use [^;]*;", masked, re.M):
        if raw.count("\n", 0, m.start()) in excluded:
            continue
        start = m.start()
        lines_above = raw[:start].split("\n")[:-1]
        attrs = []
        while lines_above and lines_above[-1].strip().startswith("#["):
            attrs.insert(0, lines_above.pop())
        out.append("\n".join(attrs + [raw[m.start() : m.end()]]))
    return out


def main():
    edges = json.load(open(sys.argv[1]))
    proto = os.path.abspath(sys.argv[2])
    crates_dir = os.path.join(proto, "crates")
    subs = sorted(c for c in os.listdir(crates_dir) if c not in ("facade", "path_proof"))
    mods = public_modules()
    fsrc = os.path.join(crates_dir, "facade", "src")
    # keep the facade's own modules (root, macros) as the splitter emitted them
    own = {}
    for m in OWN:
        p = os.path.join(fsrc, m + ".cairo")
        if os.path.exists(p):
            own[m] = open(p).read()
    own_dirs = {m: os.path.join(fsrc, m) for m in OWN if os.path.isdir(os.path.join(fsrc, m))}
    for root, dirs, files in os.walk(fsrc, topdown=False):
        for f in files:
            full = os.path.join(root, f)
            rel = os.path.relpath(full, fsrc)
            if rel.split("/")[0] in OWN and rel.split("/")[0] + ".cairo" != rel:
                continue
            os.remove(full)
    moved = json.load(open(os.path.join(proto, "prototype.json"))).get("moved_items", {})
    moved_by_origin = collections.defaultdict(list)
    for nid, (c, target) in moved.items():
        origin, k = nid.rsplit("#", 1)
        it = edges["files"][origin]["items"][int(k)]
        if it.get("public") and c != "facade":
            moved_by_origin[origin[: -len(".cairo")]].append(
                f"pub use nalgebra_{c}::{target[: -len('.cairo')].replace('/', '::')}::{it['name']};")
    first_host = {}
    for mod, children in sorted(mods.items()):
        if mod and not os.path.exists(os.path.join(SRC, mod + ".cairo")):
            continue  # an inline module: re-exported by its parent (or the facade's own file)
        path = os.path.join(fsrc, (mod or "lib") + ".cairo")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if mod in OWN:
            body = own.get(mod, "")
        else:
            hosts = [c for c in subs
                     if os.path.exists(os.path.join(crates_dir, c, "src", mod + ".cairo"))] if mod else []
            lines = [f"pub use nalgebra_{c}::{mod.replace('/', '::')}::*;" for c in hosts]
            if hosts:
                first_host[mod] = hosts[0]
            # children the facade declares itself shadow the globbed modules of the same name;
            # an inline `errors` module exists in every sub-crate: one explicit re-export wins
            body = "\n".join(lines) + "\n"
        decls = []
        # public impls the split moves to the module of their anchor type keep their 0.1.0 path
        body += "\n".join(sorted(moved_by_origin.get(mod, []))) + "\n"
        for ch in children:
            if mod in OWN:
                break
            chmod = f"{mod}/{ch}" if mod else ch
            if chmod in mods and not os.path.exists(os.path.join(SRC, chmod + ".cairo")) and \
                    os.path.exists(os.path.join(SRC, (mod or "lib") + ".cairo")) and not mods[chmod]:
                # an inline module (`pub mod errors { .. }`): re-exported from one sub-crate
                hosts = [c for c in subs
                         if os.path.exists(os.path.join(crates_dir, c, "src", chmod + ".cairo"))]
                if hosts:
                    decls.append(f"pub use nalgebra_{hosts[0]}::{chmod.replace('/', '::')};")
                continue
            decls.append(f"pub mod {ch};")
        body += "\n" + "\n".join(decls) + "\n\n" + "\n".join(pub_uses(mod)) + "\n"
        with open(path, "w") as f:
            f.write(body)
    # facade manifest: every sub-crate
    man = os.path.join(crates_dir, "facade", "Scarb.toml")
    text = open(man).read()
    text = text.replace('name = "nalgebra_facade"', 'name = "nalgebra"')
    deps = "".join(f'nalgebra_{c} = {{ path = "../{c}" }}\n' for c in subs)
    text = re.sub(r"(\[dependencies\]\n)(.*)", lambda m: m.group(1) + 'simba = "0.2.0"\n' + deps, text,
                  flags=re.S)
    with open(man, "w") as f:
        f.write(text)
    write_proof(edges, proto, mods)


def write_proof(edges, proto, mods):
    paths = set()
    for mod, _children in mods.items():
        f = edges["files"].get((mod or "lib") + ".cairo")
        if f is not None:
            for it in f["items"]:
                if not it.get("public"):
                    continue
                if it["kind"] in ("struct", "enum", "trait", "fn", "const", "type"):
                    paths.add(f"{mod.replace('/', '::')}::{it['name']}".lstrip(":"))
                if it["kind"] == "impl" and it["gen"]:
                    paths.add(f"{mod.replace('/', '::')}::{it['gen']}".lstrip(":"))
        for stmt in pub_uses(mod):
            tree = stmt.split("pub use", 1)[1].rsplit(";", 1)[0]
            for full in expand_use(tree):
                name = full.split(" as ")[-1].split("::")[-1].strip()
                if name == "*" or name == "self":
                    continue
                paths.add(f"{mod.replace('/', '::')}::{name}".lstrip(":"))
    paths = sorted(paths)
    d = os.path.join(proto, "crates", "path_proof")
    os.makedirs(os.path.join(d, "src"), exist_ok=True)
    with open(os.path.join(d, "Scarb.toml"), "w") as f:
        f.write('[package]\nname = "path_proof"\nversion = "0.1.0"\nedition = "2024_07"\n\n'
                '[dependencies]\nnalgebra = { path = "../facade" }\nfixed = "0.4.0"\n')
    body = ["// Every public path of nalgebra 0.1.0, through the facade prototype."]
    for i, p in enumerate(paths):
        body.append(f"#[allow(unused_imports)]\nmod p{i} {{\n    use nalgebra::{p};\n}}")
    body.append(USAGE)
    with open(os.path.join(d, "src", "lib.cairo"), "w") as f:
        f.write("\n".join(body) + "\n")
    ws = os.path.join(proto, "Scarb.toml")
    t = open(ws).read()
    if "crates/path_proof" not in t:
        t = t.replace('members = [\n', 'members = [\n    "crates/path_proof",\n', 1)
        with open(ws, "w") as f:
            f.write(t)
    print(f"{len(paths)} public paths written to {d}/src/lib.cairo")


USAGE = """
// Method resolution, operators, products, norms, solves, decompositions and macros with the
// imports a 0.1.0 user writes.
mod usage {
    use fixed::Fixed;
    use nalgebra::{
        EuclideanNorm, Isometry3, Isometry3Trait, Matrix3, Matrix3Trait, Matrix6, Matrix6Trait,
        MatrixMul, MatrixSolve, MatrixTrMul, Matrix5x3, Matrix3x5, Matrix5, FixedView, Matrix2,
        UnitQuaternion, UnitQuaternionTrait, Vector3, Vector3Trait, dmatrix, matrix, vector, point,
    };
    use nalgebra::linalg::Matrix3LuTrait;

    fn methods(m: Matrix3<Fixed>, v: Vector3<Fixed>) -> Fixed {
        let t = m.transpose();
        let p = m * t + m;
        let w = p.mul_mat(v);
        let s = m.tr_mul(m);
        let n = s.apply_norm(EuclideanNorm {});
        let x = m.solve_lower_triangular(w);
        let lu = m.lu();
        let _ = lu;
        let _ = x;
        n + v.norm() + m.determinant()
    }

    fn cross_crate(a: Matrix5x3<Fixed>, b: Matrix3x5<Fixed>, m6: Matrix6<Fixed>) -> Fixed {
        let c: Matrix5<Fixed> = a.mul_mat(b);
        let d: Matrix2<Fixed> = FixedView::fixed_view(m6, 0, 0);
        let _ = d;
        c.m11 + m6.trace()
    }

    fn geometry(q: UnitQuaternion<Fixed>, iso: Isometry3<Fixed>, v: Vector3<Fixed>) -> Vector3<Fixed> {
        iso.transform_vector(q.transform_vector(v))
    }

    fn macros() -> Matrix3<Fixed> {
        let _v = vector![Fixed { raw: 1 }, Fixed { raw: 2 }, Fixed { raw: 3 }];
        let _p = point![Fixed { raw: 1 }, Fixed { raw: 2 }];
        let _d = dmatrix![Fixed { raw: 1 }, Fixed { raw: 2 }];
        matrix![
            Fixed { raw: 1 }, Fixed { raw: 0 }, Fixed { raw: 0 };
            Fixed { raw: 0 }, Fixed { raw: 1 }, Fixed { raw: 0 };
            Fixed { raw: 0 }, Fixed { raw: 0 }, Fixed { raw: 1 }
        ]
    }
}
"""


if __name__ == "__main__":
    main()
