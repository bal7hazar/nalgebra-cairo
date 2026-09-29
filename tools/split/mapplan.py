#!/usr/bin/env python3
"""The plan of the crate map, for `prototype.py` (WP 9-NS1b): build the split workspace FROM THE MAP.

`tools/split/crates.toml` is the only place the crate names and the placement rules live. This
helper places every item of the item graph (`edges.py`) with the map's rules (`cratemap.py`, the
same placement the generators use), optionally after merging crates, and writes a plan JSON in
`plan.py --json`'s format (`crates`, `deps`, `lines`, `items`; plus `band_split` and the map's
declared `closures`), which
`prototype.py`, `facade.py` and `glam_proto.py` read:

    python3 tools/split/edges.py crates/nalgebra --json E.json
    python3 tools/split/mapplan.py E.json PLAN.json [--map M] [--merge NEW=a+b ...] [--lines]
    python3 tools/split/prototype.py E.json PLAN.json PROTO

`--merge NEW=a+b` hosts crates `a` and `b` in one crate `NEW` (a new name or one of the two) at the
position of the LOWER of them in the table (the later one: a crate only depends on crates above
it); a merge that makes a crate depend on a crate below it is refused. `--lines` prints the lines
and direct dependencies per crate. Exit code 1 if an item of the graph is left unplaced.
"""
import argparse
import collections
import json
import os
import re
import sys
import tempfile
import tomllib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import cratemap  # noqa: E402
from layouts import band_of_label, final as NS1_FINAL  # noqa: E402
from plan import Graph  # noqa: E402

# crate-internal kernel traits holding every dimension, split per band by the moves (docs/SPLIT.md
# §3.3): {generated trait: {band: the item whose crate hosts that band's methods}}
BAND_SPLIT = {"SvdRightTrait": {"5": "Svd5", "6": "Svd6"}}


def merged_map_text(text, merges):
    """The map's text with the crates of each merge `(new, [a, b])` hosted by one crate `new`."""
    m = re.search(r"^\[crates\]\n(.*?)(?=\n\n|\n\[|\Z)", text, re.S | re.M)
    rows = [re.match(r'^(\w+) = "([^"]*)"', line) for line in m.group(1).splitlines()]
    order = [r.group(1) for r in rows if r]
    host = {r.group(1): r.group(2) for r in rows if r}
    rename = {}
    for new, parts in merges:
        for p in parts:
            if p not in host:
                sys.exit(f"--merge: unknown crate {p!r}")
        last = max(parts, key=order.index)
        for p in parts:
            rename[p] = new
        pkg = host[last]
        order = [new if c == last else c for c in order if c not in parts or c == last]
        host[new] = pkg if pkg == "nalgebra" else "nalgebra_" + new
    body = "\n".join(f'{c} = "{host[c]}"' for c in order)
    out = text[: m.start(1)] + body + text[m.end(1):]

    def sub(mm):
        return mm.group(1) + rename.get(mm.group(2), mm.group(2)) + mm.group(3)

    # rule targets: `crate = "x"`, `default = "x"`, the values of `bands = { "4" = "x" }`
    head, rules = out.split("[[rule]]", 1) if "[[rule]]" in out else (out, "")
    rules = re.sub(r'(\b(?:crate|default) = ")(\w+)(")', sub, rules)
    rules = re.sub(r'("\d+" = ")(\w+)(")', sub, rules)
    return head + ("[[rule]]" + rules if rules else ""), rename


def place(cm, edges):
    """{edges item id: crate} by the map's rules (items matched per file by label, in order)."""
    texts = cm.library_files()
    placed = cm.place_all(texts)
    items = {}
    for f, ef in edges["files"].items():
        by_label = collections.defaultdict(list)
        for it, c, _mod in placed.get(f, ()):
            by_label[it.label].append(c)
        seen = collections.Counter()
        for k, it in enumerate(ef["items"]):
            label = it["gen"] or it["name"]
            cs = by_label.get(label)
            if not cs:
                continue
            items[f"{f}#{k}"] = cs[min(seen[label], len(cs) - 1)]
            seen[label] += 1
    return items


def pruned_graph(edges, text):
    """`plan.py`'s item graph without the edges the moves remove or the analysis over-approximates."""
    g = Graph(edges)
    # the edges the moves remove (band-split kernel traits, `Normed` kernel wrappers, the
    # over-approximated calls of shared linalg files): `plan.py`'s, layout-independent
    for n in g.nodes:
        for m in list(g.deps[n]):
            if NS1_FINAL.ignore_edge(g, n, m):
                g.deps[n].discard(m)
    d = tomllib.loads(text)
    # `[label_bands]`: the dimension of the items whose name does not say it (`UnitQuaternion` is
    # 3D, `Matrix4CgTrait` the 3D homogeneous coordinates), regexes (full match) on the label
    label_bands = [(re.compile(k), v) for k, v in d.get("label_bands", {}).items()]

    def band(n):
        lab = g.label(n).split(":")[0]
        for rx, b in label_bands:
            if rx.fullmatch(lab):
                return b
        return band_of_label(lab)

    # `kernel_wrappers`: impls whose bodies call a higher crate's inherent trait until the move gives
    # them a type-level kernel (an `#[inline(always)]` wrapper: no step change, docs/SPLIT.md §3.3)
    wrappers = set(d.get("kernel_wrappers", []))
    for n in g.nodes:
        if g.label(n).split(":")[0] in wrappers:
            for m in list(g.deps[n]):
                if g.kind(m) == "inherent":
                    g.deps[n].discard(m)
    if d.get("strict_calls"):
        # (WP 9-NS13) a method-call edge (the item does not name the target) from an item of
        # dimension b to an item of a LARGER dimension is the over-approximation of `plan.py`
        # (every trait in scope with a method of that name); the prototype build proves the cut
        named = collections.defaultdict(set)
        for n in g.nodes:
            for mid in g.members[n]:
                named[n].update(g.find(t) for t in g.info[mid]["ref_ids"])
        for n in g.nodes:
            bn = band(n)
            if bn is None:
                continue
            for m in list(g.deps[n]):
                bm = band(m)
                if bm is not None and bm > bn and m not in named[n]:
                    g.deps[n].discard(m)
    return g


def build_plan(edges, map_path, merges):
    text = open(map_path).read()
    rename = {}
    if merges:
        text, rename = merged_map_text(text, merges)
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(text)
        tmp = fh.name
    try:
        cm = cratemap.load(tmp)
    finally:
        os.unlink(tmp)
    items = place(cm, edges)
    g = pruned_graph(edges, text)
    missing = [nid for n in g.nodes for nid in g.members[n] if nid not in items]
    # a node (a trait + its single impl) is one unit: its members follow its first placed member
    for n in g.nodes:
        cs = [items[m] for m in g.members[n] if m in items]
        for m in g.members[n]:
            if m not in items and cs:
                items[m] = cs[0]
    crate = {n: items[g.members[n][0]] for n in g.nodes if g.members[n][0] in items}
    lines = collections.Counter()
    for n, c in crate.items():
        lines[c] += g.lines[n]
    deps = collections.defaultdict(set)
    for n in crate:
        for m in g.deps[n]:
            if m in crate and crate[m] != crate[n]:
                deps[crate[n]].add(crate[m])
    order = [c for c in cm.order if lines[c]]
    rank = {c: i for i, c in enumerate(order)}
    upward = sorted((c, d) for c, ds in deps.items() for d in ds if d in rank and rank[d] > rank[c])
    closures = {}
    for name, members in tomllib.loads(text).get("closures", {}).items():
        out = []
        for m in members:
            m = rename.get(m, m)
            if m not in out:
                out.append(m)
        closures[name] = out
    band = {}
    # the map's own `[band_split]` table (the per-dimension re-cut, WP 9-NS13), else BAND_SPLIT
    for gen, per in (tomllib.loads(text).get("band_split") or BAND_SPLIT).items():
        band[gen] = {b: items.get(edges["items"].get(nm)) for b, nm in per.items()}
    return {
        "crates": order,
        "deps": {c: sorted(v) for c, v in deps.items()},
        "lines": dict(lines),
        "items": items,
        "band_split": band,
        "closures": closures,
        "upward": upward,
        "unplaced": [m for m in missing if m not in items],
        "stubs": sorted(tomllib.loads(text).get("kernel_wrappers", [])),
        "map": text,
    }


def parse_merge(spec):
    new, parts = spec.split("=", 1)
    parts = [p for p in parts.split("+") if p]
    if len(parts) < 2:
        sys.exit(f"--merge {spec}: expected NEW=a+b")
    return new, parts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("edges")
    ap.add_argument("out")
    ap.add_argument("--map", default=cratemap.DEFAULT_MAP)
    ap.add_argument("--merge", action="append", default=[], metavar="NEW=a+b")
    ap.add_argument("--lines", action="store_true")
    a = ap.parse_args()
    edges = json.load(open(a.edges))
    plan = build_plan(edges, a.map, [parse_merge(s) for s in a.merge])
    with open(a.out, "w") as f:
        json.dump(plan, f, indent=1, sort_keys=True)
    if a.lines:
        for c in plan["crates"]:
            print(f"  {c:22s} {plan['lines'][c]:7,d}  {', '.join(plan['deps'].get(c, [])) or '-'}")
    for c, d in plan["upward"]:
        print(f"error: {c} depends on {d}, placed below it in the table", file=sys.stderr)
    if plan["unplaced"]:
        print(f"error: {len(plan['unplaced'])} items unplaced: {plan['unplaced'][:5]}", file=sys.stderr)
    return 1 if plan["unplaced"] or plan["upward"] else 0


if __name__ == "__main__":
    sys.exit(main())
