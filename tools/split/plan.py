#!/usr/bin/env python3
"""Place the items of `nalgebra` into planned sub-crates and check the plan (WP 9-NS1 helper).

Input: the JSON of `tools/split/edges.py`. A LAYOUT (see `layouts.py`) gives an ordered list of
crates (lowest first) and a `home(node)` rule. The solver

  1. builds the item graph: one node per top-level item, the single impl of a trait declared in
     the same file merged with its trait (inherent traits: `Matrix3Trait` + `Matrix3Impl`);
     edges: names referenced (types, traits, constants, functions) and method calls resolved
     against the traits in scope of the file (over-approximation: every trait in scope that has a
     method of that name);
  2. lifts every node to the highest crate among its home and its dependencies (fixpoint), so
     every dependency points to the same or a LOWER crate (the crate graph is acyclic);
  3. checks Cairo's impl lookup (measured in docs/SPLIT.md §1): an impl of a trait with type
     arguments is found without an import only in the module of the trait or of one of its
     argument types, so an impl must end in the crate of its trait or of one of its argument
     structs (`anchor`); `anchor` violations are listed (they would need an import at every use
     site, i.e. a breaking change);
  4. reports the library lines per crate, the crate dependencies and the lifted nodes.

    python3 tools/split/plan.py /tmp/edges.json LAYOUT [--json assignment.json] [--verbose]
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

CORE_TRAITS = {
    "Add", "Sub", "Neg", "Mul", "Div", "Rem", "AddAssign", "SubAssign", "MulAssign", "DivAssign",
    "PartialOrd", "PartialEq", "Bounded", "One", "Zero", "Default", "Sum", "Product", "IndexView",
    "Index", "Into", "TryInto", "Serde", "Debug", "Display", "Hash", "Copy", "Drop", "Felt252DictValue",
}

# Cairo corelib method names (`Option::unwrap`, `Into::into`...): a call of one of them is never
# resolved to a nalgebra trait of ANOTHER top-level module by the last, "any trait in scope",
# fallback below (WP 9-NS13: `.unwrap()` in `Matrix2AngleTrait` is not `UnitComplexTrait::unwrap`)
CORE_METHODS = {
    "unwrap", "expect", "into", "try_into", "clone", "len", "append", "span", "at", "is_some",
    "is_none", "unwrap_or", "is_empty", "pop_front", "get", "index", "new",
}

SHAPE_RE = re.compile(r"(?:^|/)(matrix|vector|row_vector)(\d)(?:x(\d))?\.cairo$")


def shape_of(path):
    """(rows, cols) of a base shape file, None otherwise."""
    m = re.match(r"base/(matrix|vector|row_vector)(\d)(?:x(\d))?\.cairo$", path)
    if not m:
        return None
    a = int(m.group(2))
    b = int(m.group(3) or a)
    if m.group(1) == "vector":
        return (a, 1)
    if m.group(1) == "row_vector":
        return (1, a)
    return (a, b)


def dims_in_name(path):
    """(rows, cols) encoded in a generated file name (`lu3.cairo`, `svd5x2.cairo`...), or None."""
    m = re.search(r"(\d)(?:x(\d))?\.cairo$", path)
    if not m:
        return None
    a = int(m.group(1))
    return (a, int(m.group(2) or a))


class Graph:
    def __init__(self, data):
        self.data = data
        self.info = {}
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        impls_of = collections.Counter()
        for p, f in data["files"].items():
            for it in f["items"]:
                if it["kind"] == "impl" and it["of"]:
                    impls_of[(p, it["of"])] += 1
        self.local_traits = {}
        for p, f in data["files"].items():
            local = {}
            for k, it in enumerate(f["items"]):
                nid = f"{p}#{k}"
                parent[nid] = nid
                self.info[nid] = it
                if it["kind"] == "trait":
                    local[it["name"]] = nid
            for k, it in enumerate(f["items"]):
                nid = f"{p}#{k}"
                if it["kind"] == "impl" and it["of"] in local and impls_of[(p, it["of"])] == 1:
                    parent[find(nid)] = find(local[it["of"]])
        self.find = find
        self.members = collections.defaultdict(list)
        for nid in self.info:
            self.members[find(nid)].append(nid)
        self.nodes = sorted(self.members)
        self.lines = {n: sum(self.info[m]["lines"] for m in self.members[n]) for n in self.nodes}
        # trait name -> node, methods
        self.trait_node = {}
        self.trait_methods = collections.defaultdict(set)
        for nid, it in self.info.items():
            if it["kind"] == "trait" or it["gen"]:
                nm = it["name"] if it["kind"] == "trait" else it["gen"]
                self.trait_node[nm] = find(nid)
                for m in it.get("methods", []):
                    self.trait_methods[nm].add(m["name"])
        item_of = data["items"]
        self.deps = collections.defaultdict(set)
        for nid, it in self.info.items():
            p = nid.split("#")[0]
            r = find(nid)
            for t in it["ref_ids"]:
                if find(t) != r:
                    self.deps[r].add(find(t))
            scope = set(data["files"][p]["imports"]) | {
                (self.info[m]["name"] if self.info[m]["kind"] == "trait" else self.info[m]["gen"])
                for m in self.info
                if m.startswith(p + "#") and (self.info[m]["kind"] == "trait" or self.info[m]["gen"])
            }
            top = p.split("/")[0]
            for c in it.get("calls", []):
                cands = [tr for tr in scope if tr in self.trait_node and c in self.trait_methods[tr]]
                # receiver types are unknown: prefer the traits of the same file, then of the same
                # top-level module (base / geometry / linalg), then any trait in scope
                for pick in (
                    [tr for tr in cands if self.trait_node[tr].startswith(p + "#")],
                    [tr for tr in cands if self.trait_node[tr].split("/")[0] == top],
                    [] if c in CORE_METHODS else cands,
                ):
                    if pick:
                        break
                for tr in pick:
                    t = self.trait_node[tr]
                    if t != r:
                        self.deps[r].add(t)
        self.struct_node = {
            it["name"]: find(nid) for nid, it in self.info.items() if it["kind"] == "struct"
        }

    def file(self, n):
        return n.split("#")[0]

    def label(self, n):
        it = self.info[n]
        if it["gen"]:
            return it["gen"]
        if it["kind"] == "impl":
            return f"{it['name']}: {it['of']}<{','.join(it['args'])}>"
        return it["name"]

    def kind(self, n):
        """struct | typebound (core trait of one nalgebra type) | generic:<Trait> | inherent | other."""
        it = self.info[n]
        if it["kind"] in ("struct", "enum"):
            return "struct"
        if it["kind"] == "impl" and len(self.members[n]) == 1 and not it["gen"]:
            if it["of"] in CORE_TRAITS and len(it["args"]) <= 1:
                return "typebound"
            if it["args"]:
                return "generic:" + (it["of"] or "?")
        if it["kind"] == "trait" or it["gen"] or len(self.members[n]) > 1:
            return "inherent"
        return "other"

    def is_impl_anchored(self, n):
        """Impl of a trait with type arguments (found through the trait's or a type's module)."""
        it = self.info[n]
        return it["kind"] == "impl" and not it["gen"] and len(self.members[n]) == 1 and bool(it["args"])


def homes(g, layout):
    rank = {c: i for i, c in enumerate(layout.CRATES)}
    crate = {n: layout.home(g, n) for n in g.nodes}
    for n, c in getattr(layout, "pull", lambda g: {})(g).items():
        # `n` must sit in crate `c` or lower, and so must everything it depends on
        stack, seen = [n], set()
        while stack:
            m = stack.pop()
            if m in seen:
                continue
            seen.add(m)
            if rank.get(crate[m], -1) > rank[c]:
                crate[m] = c
            stack.extend(g.deps[m])
    return crate


def solve(g, layout, verbose=False):
    order = layout.CRATES
    rank = {c: i for i, c in enumerate(order)}
    crate = homes(g, layout)
    for n, c in crate.items():
        if c not in rank:
            raise SystemExit(f"unknown crate {c} for {g.label(n)} ({n})")
    ignore = getattr(layout, "ignore_edge", None)
    ignored = collections.Counter()
    if ignore is not None:
        for n in g.nodes:
            for m in list(g.deps[n]):
                if ignore(g, n, m):
                    g.deps[n].discard(m)
                    ignored[(g.label(n).split(":")[0], g.label(m).split(":")[0])] += 1
    g.ignored = ignored
    lifted_by = {}
    changed = True
    it = 0
    while changed:
        changed = False
        it += 1
        for n in g.nodes:
            for m in g.deps[n]:
                if rank[crate[m]] > rank[crate[n]]:
                    crate[n] = crate[m]
                    lifted_by[n] = m
                    changed = True
            # a core-trait impl of one nalgebra type (`Add<Matrix3>`) lives in the module of its
            # type: the struct follows it
            if g.kind(n) == "typebound":
                for a in g.info[n]["args"]:
                    st = g.struct_node.get(a)
                    if st is not None and rank[crate[n]] > rank[crate[st]]:
                        crate[st] = crate[n]
                        lifted_by[st] = n
                        changed = True
    # anchors
    violations = []
    for n in g.nodes:
        if not g.is_impl_anchored(n):
            continue
        it_ = g.info[n]
        allowed = set()
        tn = g.trait_node.get(it_["of"])
        if tn is not None:
            allowed.add(crate[tn])
        for a in it_["args"]:
            s = g.struct_node.get(a)
            if s is not None:
                allowed.add(crate[s])
        if it_["of"] in CORE_TRAITS and not allowed:
            continue
        if tn is not None and not g.info[tn].get("public", True):
            # a crate-private trait: its callers are ours and import the impl
            continue
        if crate[n] not in allowed:
            violations.append((n, crate[n], sorted(allowed)))
    return crate, lifted_by, violations


def report(g, layout, crate, lifted_by, violations, verbose=False):
    lines = collections.Counter()
    for n, c in crate.items():
        lines[c] += g.lines[n]
    cdeps = collections.defaultdict(set)
    for n in g.nodes:
        for m in g.deps[n]:
            if crate[m] != crate[n]:
                cdeps[crate[n]].add(crate[m])
    print("| crate | lines | depends on |")
    print("|---|---:|---|")
    for c in layout.CRATES:
        if lines[c]:
            print(f"| {c} | {lines[c]:,} | {', '.join(sorted(cdeps[c])) or '-'} |")
    print(f"| total | {sum(lines.values()):,} | |")
    home = homes(g, layout)
    lifted = [n for n in g.nodes if crate[n] != home[n]]
    agg = collections.Counter()
    for n in lifted:
        agg[(home[n], crate[n])] += g.lines[n]
    print("\nlifted (home -> crate: lines):")
    for (h, c), v in agg.most_common():
        print(f"  {h} -> {c}: {v}")
    if verbose:
        for n in sorted(lifted, key=lambda n: -g.lines[n])[:60]:
            chain = []
            m = n
            while m in lifted_by and len(chain) < 6:
                m = lifted_by[m]
                chain.append(f"{g.label(m)} [{g.file(m)}]")
            print(f"    {g.lines[n]:6d} {g.label(n)} [{g.file(n)}] {home[n]}->{crate[n]} via {' <- '.join(chain)}")
    ign = getattr(g, "ignored", {})
    if ign:
        print(f"\nedges ignored (crate-internal items the moves split per band): {len(ign)}")
        for (a, b), v in sorted(ign.items())[:20]:
            print(f"  {a} -> {b}")
    print(f"\nanchor violations: {len(violations)}")
    vagg = collections.Counter()
    for n, c, allowed in violations:
        vagg[(g.info[n]["of"], c, tuple(allowed))] += 1
    for k, v in vagg.most_common(40):
        print(f"  {v:4d} x {k[0]} in {k[1]}, allowed {list(k[2])}")
    return lines, cdeps


def main():
    data = json.load(open(sys.argv[1]))
    import layouts

    layout = getattr(layouts, sys.argv[2])
    g = Graph(data)
    crate, lifted_by, violations = solve(g, layout)
    lines, cdeps = report(g, layout, crate, lifted_by, violations, "--verbose" in sys.argv)
    if "--json" in sys.argv:
        out = {
            "crates": [c for c in layout.CRATES if lines[c]],
            "deps": {c: sorted(v) for c, v in cdeps.items()},
            "lines": dict(lines),
            "items": {m: crate[n] for n in g.nodes for m in g.members[n]},
            "violations": [[n, c, a] for n, c, a in violations],
        }
        with open(sys.argv[sys.argv.index("--json") + 1], "w") as f:
            json.dump(out, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
