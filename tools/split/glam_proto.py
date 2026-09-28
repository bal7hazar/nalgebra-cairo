#!/usr/bin/env python3
"""`nalgebra_glam` on the sub-crates of a prototype split (WP 9-NS1 helper).

Copies `crates/nalgebra_glam/src` (library files only) into `PROTO/crates/glam_bridge`, rewrites
every `use nalgebra::{..}` to the sub-crate that defines each name (plan + prototype relocation
map), and depends on those sub-crates only (plus `glam`, `fixed`). Building it proves which
sub-crates `nalgebra_glam` needs; measuring an empty consumer of it gives its closure cost.

    python3 tools/split/glam_proto.py EDGES.json PLAN.json PROTO_DIR
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from edges import expand_use  # noqa: E402

GLAM = os.path.join(ROOT, "crates", "nalgebra_glam", "src")


def main():
    edges = json.load(open(sys.argv[1]))
    plan = json.load(open(sys.argv[2]))
    proto = os.path.abspath(sys.argv[3])
    moved = json.load(open(os.path.join(proto, "prototype.json"))).get("moved_items", {})
    where = {}
    for name, nid in edges["items"].items():
        c = plan["items"][nid]
        mod = moved.get(nid, [c, nid.rsplit("#", 1)[0]])[1]
        where[name] = (c, mod[: -len(".cairo")].replace("/", "::"))
    dst = os.path.join(proto, "crates", "glam_bridge")
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "src"))
    used = set()
    for f in os.listdir(GLAM):
        if not f.endswith(".cairo") or f in ("tests.cairo", "benches.cairo"):
            continue
        text = open(os.path.join(GLAM, f)).read()
        text = re.sub(r"#\[cfg\(test\)\]\s*mod \w+;\n", "", text)

        def repl(m):
            out = []
            for full in expand_use(m.group(1)):
                segs = [s.strip() for s in full.split("::")]
                name = segs[-1]
                if segs[0] != "nalgebra" or name not in where:
                    out.append(f"use {full};")
                    continue
                c, mod = where[name]
                used.add(c)
                out.append(f"use nalgebra_{c}::{mod}::{name};")
            return "\n".join(out)

        text = re.sub(r"^use (nalgebra::[^;]*);", repl, text, flags=re.M)
        with open(os.path.join(dst, "src", f), "w") as fh:
            fh.write(text)
    deps = "".join(f'nalgebra_{c} = {{ path = "../{c}" }}\n' for c in sorted(used))
    with open(os.path.join(dst, "Scarb.toml"), "w") as fh:
        fh.write('[package]\nname = "glam_bridge"\nversion = "0.1.0"\nedition = "2024_07"\n\n'
                 f'[dependencies]\n{deps}glam = "0.4.0"\nfixed = "0.4.0"\nsimba = "0.2.0"\n')
    ws = os.path.join(proto, "Scarb.toml")
    t = open(ws).read()
    if "crates/glam_bridge" not in t:
        t = t.replace("members = [\n", 'members = [\n    "crates/glam_bridge",\n', 1)
        with open(ws, "w") as fh:
            fh.write(t)
    print("nalgebra_glam depends on:", ", ".join(sorted(used)))


if __name__ == "__main__":
    main()
