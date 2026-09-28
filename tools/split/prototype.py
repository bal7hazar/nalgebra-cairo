#!/usr/bin/env python3
"""Emit a THROWAWAY workspace that realises a planned split of `nalgebra` (WP 9-NS1 helper).

Mechanical, for measurement only (never committed as code of the library): every top-level item of
`crates/nalgebra/src` is copied, with its doc comments and attributes, into the crate that
`plan.py --json` assigned it to, at the SAME module path (`base/matrix3.cairo` stays
`base::matrix3` in whichever crate), except the impls with type arguments whose own module does
not hold their struct / trait in that crate: Cairo finds such an impl only in the module of its
trait or of one of its argument types, so they move to the module of an anchor in the crate. Then:

  * the `use` block of every emitted file is regenerated: external imports (`core::`, `simba::`)
    are copied, every crate item the file's items name (and every trait the original file imported,
    for method resolution) is imported from its defining crate (`crate::` or `nalgebra_<crate>::`);
  * inline paths `crate::a::b::Name` / `super::x::Name` are rewritten the same way;
  * visibility is widened (`pub(crate)` -> `pub`, private items and struct fields -> `pub`): a
    sub-crate uses items that were crate-private;
  * each crate declares the features of `nalgebra` (all on by default) and depends on the lower
    crates it imports from (plus `simba`).

    python3 tools/split/prototype.py EDGES.json PLAN.json OUT_DIR

Items referenced from a HIGHER crate are not imported (that would be a cycle): the compiler then
reports them, which is how a plan is proved wrong. `OUT_DIR/prototype.json` lists them.
"""
import collections
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import consumer_cost as cc  # noqa: E402
from edges import USE, kept_text_and_excluded  # noqa: E402

SRC = os.path.join(ROOT, "crates", "nalgebra", "src")
PREFIX = "nalgebra_"
EXTERNAL_ROOTS = ("core", "simba", "fixed", "starknet")
INLINE_PATH = re.compile(r"\b(crate|super(?:::super)*)((?:::\w+)+)")
TOP_ITEM = re.compile(
    r"^(?P<vis>pub(?:\([\w:]+\))?\s+)?(?P<kw>impl|trait|fn|struct|enum|const|type)\b", re.M
)


def expand_use(tree):
    """`a::{b, c::{d, e}}` -> ["a::b", "a::c::d", "a::c::e"] (aliases kept: "a::b as x")."""
    tree = " ".join(tree.split())
    i = tree.find("{")
    if i < 0:
        return [tree.strip()]
    prefix = tree[:i]
    inner = tree[i + 1 : tree.rfind("}")]
    parts, depth, cur = [], 0, ""
    for ch in inner:
        if ch == "," and depth == 0:
            parts.append(cur)
            cur = ""
            continue
        depth += ch == "{"
        depth -= ch == "}"
        cur += ch
    parts.append(cur)
    out = []
    for part in parts:
        if part.strip():
            out.extend(prefix + x for x in expand_use(part.strip()))
    return out


def modpath(path):
    return path[: -len(".cairo")].replace("/", "::")


def feature_table():
    text = open(os.path.join(ROOT, "crates", "nalgebra", "Scarb.toml")).read()
    m = re.search(r"^\[features\]\n(.*?)(?=^\[)", text, re.S | re.M)
    return "[features]\n" + m.group(1)


class Splitter:
    def __init__(self, edges, plan):
        self.edges = edges
        self.order = plan["crates"]
        self.rank = {c: i for i, c in enumerate(self.order)}
        self.item_crate = plan["items"]
        self.defs = edges["items"]  # unique name -> "file#k"
        self.deps = collections.defaultdict(set)
        self.upward = collections.Counter()
        self.relocated = collections.Counter()
        self.anchor_file = {}  # struct / trait name -> file
        for p, f in edges["files"].items():
            for it in f["items"]:
                if it["kind"] in ("struct", "trait"):
                    self.anchor_file.setdefault(it["name"], p)
        self.local = collections.defaultdict(dict)
        for p, f in edges["files"].items():
            for k, it in enumerate(f["items"]):
                for nm in (it["name"], it["gen"]):
                    if nm:
                        self.local[p][nm] = f"{p}#{k}"
        self.per = collections.defaultdict(lambda: collections.defaultdict(list))
        self.module_of = {}
        for nid, c in self.item_crate.items():
            p, k = nid.rsplit("#", 1)
            k = int(k)
            t = self.target_module(p, k, c)
            self.module_of[nid] = t
            self.per[c][t].append((p, k))
        self.module_crates = collections.defaultdict(set)
        for c in self.per:
            for p in self.per[c]:
                self.module_crates[p].add(c)
        self.cache = {}

    def crate_of_name(self, name):
        nid = self.defs.get(name)
        return None if nid is None else self.item_crate[nid]

    def target_module(self, p, k, c):
        it = self.edges["files"][p]["items"][k]
        if not (it["kind"] == "impl" and it["args"] and not it["gen"]):
            return p
        anchors = [a for a in it["args"] if a in self.anchor_file]
        if it["of"] in self.anchor_file:
            anchors.append(it["of"])
        here = [self.anchor_file[a] for a in anchors if self.crate_of_name(a) == c]
        if not here or p in here:
            return p
        self.relocated[(p, here[0])] += 1
        return here[0]

    def where(self, name, origin=None):
        """(defining module, crate) of `name` seen from the original file `origin`."""
        nid = self.local.get(origin, {}).get(name) or self.defs.get(name)
        if nid is None:
            return None
        return self.module_of[nid], self.item_crate[nid]

    def ref(self, name, c, p, count=True, origin=None):
        """Path prefix to name `name` from module `p` of crate `c` ("" if same module)."""
        w = self.where(name, origin)
        if w is None:
            return None
        q, y = w
        if y == c:
            return "crate" if q != p else ""
        if self.rank[y] < self.rank[c]:
            self.deps[c].add(y)
            return PREFIX + y
        if count:
            self.upward[(c, y, name)] += 1
        return None

    def source(self, o):
        if o not in self.cache:
            raw = open(os.path.join(SRC, o), encoding="utf-8").read()
            _kept, excluded = kept_text_and_excluded(os.path.join(SRC, o))
            self.cache[o] = (raw, excluded)
        return self.cache[o]

    def emit(self, c, p, pks):
        files = self.edges["files"]
        own = set()
        for o, k in pks:
            it = files[o]["items"][k]
            own.update(x for x in (it["name"], it["gen"]) if x)
        wanted = {}  # name -> (origin, counted)
        for o, k in pks:
            for x in files[o]["items"][k]["refs"]:
                wanted[x] = (o, True)
        for o in {o for o, _ in pks}:
            for x in files[o]["imports"]:
                if x in self.defs and x not in wanted:
                    wanted[x] = (o, False)
        uses, ext = set(), set()
        for name in sorted(set(wanted) - own):
            o, counted = wanted[name]
            w = self.where(name, o)
            if w is None:
                continue
            q, y = w
            if q == p and y == c:
                continue
            r = self.ref(name, c, p, counted, o)
            if r:
                uses.add(f"use {r}::{modpath(q)}::{name};")
        chunks = []
        for o in sorted({o for o, _ in pks}):
            raw, excluded = self.source(o)
            renames = {}
            for m in USE.finditer(cc.mask(raw)):
                if raw.count("\n", 0, m.start()) in excluded:
                    continue
                for full in expand_use(raw[m.start(1) : m.end(1)]):
                    alias = ""
                    if " as " in full:
                        full, alias = [x.strip() for x in full.split(" as ")]
                    segs = [x.strip() for x in full.split("::")]
                    if segs[0] in EXTERNAL_ROOTS:
                        ext.add(full + (f" as {alias}" if alias else ""))
                        continue
                    cur = modpath(o).split("::")
                    if segs[0] == "crate":
                        cur, segs = [], segs[1:]
                    elif segs[0] == "self":
                        segs = segs[1:]
                    while segs and segs[0] == "super":
                        cur, segs = cur[:-1], segs[1:]
                    absolute = cur + segs
                    local_name = alias or absolute[-1]
                    if "/".join(absolute[:-1]) + ".cairo" in self.inline:
                        # a constant of an inline `errors` module (emitted in every crate)
                        uses.add(f"use crate::{'::'.join(absolute)}" + (f" as {alias};" if alias else ";"))
                        continue
                    if not re.fullmatch(r"[a-z_][a-z_0-9]*", absolute[-1]):
                        continue
                    target = "/".join(absolute) + ".cairo"
                    hosts = [h for h in self.module_crates.get(target, ()) if self.rank[h] <= self.rank[c]]
                    if not hosts:
                        continue
                    h = max(hosts, key=lambda x: self.rank[x])
                    if h != c:
                        self.deps[c].add(h)
                    if o != p:
                        # an item moved to another module: the module alias must not clash
                        new_alias = f"{local_name}_{o.replace('/', '_')[:-6]}"
                        renames[local_name] = new_alias
                        local_name = new_alias
                    pre = "crate" if h == c else PREFIX + h
                    uses.add(f"use {pre}::{'::'.join(absolute)} as {local_name};")
            for k in sorted(k for oo, k in pks if oo == o):
                it = files[o]["items"][k]
                a, b = it["span"]
                la = raw.count("\n", 0, a)
                lines = raw[a:b].split("\n")
                text = "\n".join("" if (la + i) in excluded else l for i, l in enumerate(lines))

                def rewrite(m):
                    segs = m.group(2).split("::")[1:]
                    for i, s in enumerate(segs):
                        w = self.where(s, o)
                        if w is not None:
                            r = self.ref(s, c, p, True, o)
                            if r is None:
                                return m.group(0)
                            rest = "::".join(segs[i + 1 :])
                            head = (r or "crate") + "::" + modpath(w[0]) + "::" + s
                            return head + ("::" + rest if rest else "")
                    return m.group(0)

                text = INLINE_PATH.sub(rewrite, text)
                for old, new in renames.items():
                    text = re.sub(r"(?<![\w:])" + old + r"::", new + "::", text)
                text = text.replace("pub(crate)", "pub")
                mm = TOP_ITEM.search(text)
                if mm and not mm.group("vis"):
                    text = text[: mm.start("kw")] + "pub " + text[mm.start("kw") :]
                if it["kind"] == "struct":
                    text = re.sub(r"^(\s+)(?!pub\b)([a-z_]\w*\s*:)", r"\1pub \2", text, flags=re.M)
                chunks.append(text)
        seen, ext_lines = set(), []
        for full in sorted(ext):
            name = full.split(" as ")[-1].split("::")[-1].strip()
            if name in seen or name in own:
                continue
            seen.add(name)
            ext_lines.append(f"use {full};")
        imported = {u.split(" as ")[-1].split("::")[-1].rstrip(";").strip() for u in uses}
        ext_lines = [e for e in ext_lines if e.split(" as ")[-1].split("::")[-1].rstrip(";").strip() not in imported]
        return ext_lines, sorted(uses), chunks

    def inline_modules(self):
        """{"geometry/point/errors.cairo": body} for the inline `pub mod errors { .. }` blocks
        (constants only), emitted as file modules in every crate."""
        out = {}
        for o in self.edges["files"]:
            raw, _ = self.source(o)
            masked = cc.mask(raw)
            for m in re.finditer(r"^pub mod (\w+) \{", masked, re.M):
                close = cc.match_close(masked, m.end() - 1, "{", "}")
                out[o[: -len(".cairo")] + "/" + m.group(1) + ".cairo"] = raw[m.end() : close]
        return out

    def run(self, out):
        files_out = collections.defaultdict(dict)
        inline = self.inline_modules()
        self.inline = set(inline)
        for p in inline:
            for c in self.order:
                self.module_crates[p].add(c)
        for c in self.order:
            for p, pks in sorted(self.per[c].items()):
                files_out[c][p] = self.emit(c, p, pks)
            if c in files_out:
                for p, body in inline.items():
                    files_out[c][p] = ([], [], [body])
        if os.path.isdir(out):
            shutil.rmtree(out)
        os.makedirs(out)
        feats = feature_table()
        members = []
        for c in self.order:
            if c not in files_out:
                continue
            members.append(f"crates/{c}")
            srcdir = os.path.join(out, "crates", c, "src")
            os.makedirs(srcdir)
            files = files_out[c]
            allpaths = set(files)
            for p in list(files):
                parts = p[: -len(".cairo")].split("/")
                for i in range(1, len(parts)):
                    allpaths.add("/".join(parts[:i]) + ".cairo")
            children = collections.defaultdict(set)
            for p in allpaths:
                parts = p[: -len(".cairo")].split("/")
                parent = "/".join(parts[:-1]) + ".cairo" if len(parts) > 1 else "lib.cairo"
                children[parent].add(parts[-1])
            for p in sorted(allpaths | {"lib.cairo"}):
                ext, uses, chunks = files.get(p, ([], [], []))
                mods = [f"pub mod {x};" for x in sorted(children.get(p, ()))]
                body = "\n".join(ext + uses) + "\n\n" + "\n".join(mods) + "\n\n" + "\n\n".join(chunks) + "\n"
                dst = os.path.join(srcdir, p)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with open(dst, "w") as f:
                    f.write(body)
            dep_lines = "\n".join(f'{PREFIX}{d} = {{ path = "../{d}" }}' for d in sorted(self.deps[c]))
            with open(os.path.join(out, "crates", c, "Scarb.toml"), "w") as f:
                f.write(
                    f'[package]\nname = "{PREFIX}{c}"\nversion = "0.1.0"\nedition = "2024_07"\n'
                    'cairo-version = "2.19.4"\nexperimental-features = '
                    '["associated_item_constraints", "user_defined_inline_macros"]\n\n'
                    f'{feats}\n[dependencies]\nsimba = "0.2.0"\n{dep_lines}\n'
                )
        with open(os.path.join(out, "Scarb.toml"), "w") as f:
            f.write("[workspace]\nmembers = [\n" + "".join(f'    "{m}",\n' for m in members) + "]\n")
        rep = {
            "deps": {c: sorted(v) for c, v in self.deps.items()},
            "upward": [[c, y, n, v] for (c, y, n), v in self.upward.most_common()],
            "relocated": [[a, b, v] for (a, b), v in self.relocated.most_common()],
        }
        with open(os.path.join(out, "prototype.json"), "w") as f:
            json.dump(rep, f, indent=1)
        print(f"impls moved to the module of an anchor type: {sum(self.relocated.values())}")
        print("crate deps:")
        for c in self.order:
            if c in files_out:
                print(f"  {c}: {', '.join(sorted(self.deps[c])) or '-'}")
        print(f"upward references (not imported): {sum(self.upward.values())}")
        for (c, y, n), v in self.upward.most_common(30):
            print(f"  {c} -> {y}: {n} ({v})")


def main():
    edges = json.load(open(sys.argv[1]))
    plan = json.load(open(sys.argv[2]))
    Splitter(edges, plan).run(sys.argv[3])


if __name__ == "__main__":
    main()
