#!/usr/bin/env python3
"""Cross-file references of a Cairo crate, per top-level item (WP 9-NS1 helper).

Every library file (the `mod` tree of `consumer_cost.py`, test-only code excluded) is cut into its
top-level items (`impl`, `trait`, `fn`, `struct`, `const`, `type`, `macro`...). Every name defined
exactly once in the crate (structs, traits, generated traits `of X` of `#[generate_trait]` impls,
top-level functions, constants, aliases, impls, macros) is a node; an item references a name when
the identifier occurs in its body. The output (JSON) lists per file its lines and its items with
the names they reference outside the file:

    python3 tools/split/edges.py crates/nalgebra --json /tmp/edges.json

`group_of(path)` in `tools/split/plan.py` maps files to planned crates; `plan.py` consumes this JSON.
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "scripts"))
sys.path.insert(0, HERE)
import consumer_cost as cc  # noqa: E402
from files import files  # noqa: E402

ITEM = re.compile(
    r"^(?:pub(?:\([\w:]+\))?\s+)?(impl|trait|fn|struct|enum|const|type|mod|use|macro)\s+(\w+)?",
    re.M,
)
IDENT = re.compile(r"\b[A-Za-z_]\w*\b")
# method calls `.m(` (path calls `Trait::m(` already name their trait)
CALL = re.compile(r"\.\s*([a-z_]\w*)\s*(?:::<[^>]*>)?\(")
USE = re.compile(r"^\s*(?:pub(?:\([\w:]+\))?\s+)?use\s+([^;]*);", re.M)


def kept_text(path):
    """Masked text of `path` with the test-only lines blanked (consumer_cost's exclusion)."""
    return kept_text_and_excluded(path)[0]


def kept_text_and_excluded(path):
    """(masked text with the test-only lines blanked, set of the excluded line numbers)."""
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = cc.mask(text)
    lines = text.split("\n")
    starts = [0] + [k + 1 for k, c in enumerate(m) if c == "\n"]

    def line_of(pos):
        lo, hi = 0, len(starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    excluded = set()
    for a in re.finditer(r"#\[cfg\(", m):
        close = cc.match_close(m, a.start() + 1, "[", "]")
        if close < 0 or not cc.test_only(m[a.end() : close - 1]):
            continue
        first = line_of(a.start())
        while first > 0 and re.match(r"\s*(///|#\[)", lines[first - 1]):
            first -= 1
        last = line_of(max(cc.item_end(m, close + 1) - 1, close))
        excluded.update(range(first, last + 1))
    # excluded lines blanked with spaces: offsets stay those of the raw file
    kept = "\n".join(" " * len(l) if k in excluded else l for k, l in enumerate(m.split("\n")))
    return kept, excluded


def items(text):
    """[(kind, name, generated_trait, implemented_trait, start, end)] of the top-level items of a masked text."""
    out, depth, i, n = [], 0, 0, len(text)
    line_start = True
    while i < n:
        c = text[i]
        if line_start and depth == 0:
            mm = ITEM.match(text, i)
            if mm:
                # attributes right above belong to the item
                attr_start = i
                end = cc.item_end(text, i)
                kind, name = mm.group(1), mm.group(2) or ""
                gen = of = None
                if kind == "impl":
                    head = text[i:end].split("{", 1)[0]
                    g = re.search(r"\bof\s+(?:\w+::)*(\w+)", head)
                    of = g.group(1) if g else None
                    before = text[max(0, i - 200) : i]
                    if g and "#[generate_trait]" in before.split("\n}")[-1]:
                        gen = g.group(1)
                out.append((kind, name, gen, of, attr_start, end))
                i = end
                continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        line_start = c == "\n"
        i += 1
    return out


def methods(body):
    """[(name, start, end)] of the `fn` items at depth 1 of an impl / trait body."""
    out, depth = [], 0
    i = body.find("{")
    if i < 0:
        return out
    j = i + 1
    while j < len(body):
        c = body[j]
        if depth == 0:
            mm = re.match(r"fn\s+(\w+)", body[j:])
            if mm and (j == 0 or not (body[j - 1].isalnum() or body[j - 1] == "_")):
                k = body.find("{", j)
                k2 = body.find(";", j)
                if k2 != -1 and (k == -1 or k2 < k):
                    out.append((mm.group(1), j, k2 + 1))
                    j = k2 + 1
                    continue
                e = cc.match_close(body, k, "{", "}")
                out.append((mm.group(1), j, e + 1))
                j = e + 1
                continue
        if c in "{([":
            depth += 1
        elif c in "})]":
            depth -= 1
        j += 1
    return out


def crossrefs(body, resolve, own, p, imported=()):
    """Crate names referenced by `body` (other items, same file included; a name defined in the
    file wins over a homonym elsewhere): capitalised names (types, traits, impls, constants)
    anywhere, lower-case functions when called through a path (`kernels::f(`) or defined in the
    same file."""
    out = set()
    for mm in IDENT.finditer(body):
        x = mm.group(0)
        if x not in resolve or x in own:
            continue
        seg = re.search(r"(\w+)::$", body[max(0, mm.start() - 64) : mm.start()])
        local_call = resolve[x].startswith(p + "#") and body[mm.end() : mm.end() + 1] in ("(", "<")
        if x[0].isupper() or local_call or x in imported or (seg is not None and seg.group(1)[0].islower()):
            out.add(x)
    return sorted(out)


def raw_start(raw_lines, starts, pos):
    """Offset of the first line of the doc comments / attributes right above `pos`."""
    lo, hi = 0, len(starts) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if starts[mid] <= pos:
            lo = mid
        else:
            hi = mid - 1
    k = lo
    while k > 0:
        prev = raw_lines[k - 1].strip()
        if prev.startswith("///") or prev.startswith("#["):
            k -= 1
            continue
        if prev.endswith("]") and not prev.startswith("//"):
            # the last line of a multi-line attribute
            j = k - 1
            while j > 0 and j > k - 30 and not raw_lines[j].strip().startswith("#["):
                j -= 1
            if raw_lines[j].strip().startswith("#["):
                k = j
                continue
        break
    return starts[k]


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


def import_map(p, text, local_defs):
    """{local name: "file#k"} of the `use` statements of file `p` that name an item of a module
    file of the crate directly (re-exports through module roots are left to the global lookup)."""
    out = {}
    for u in USE.findall(text):
        for full in expand_use(u):
            alias = None
            if " as " in full:
                full, alias = [x.strip() for x in full.split(" as ")]
            segs = [x.strip() for x in full.split("::")]
            cur = p[: -len(".cairo")].split("/")
            if segs[0] == "crate":
                cur, segs = [], segs[1:]
            elif segs[0] == "self":
                segs = segs[1:]
            elif segs[0] != "super":
                continue
            while segs and segs[0] == "super":
                cur, segs = cur[:-1], segs[1:]
            if not segs:
                continue
            q = "/".join(cur + segs[:-1]) + ".cairo"
            nid = local_defs.get(q, {}).get(segs[-1])
            if nid:
                out[alias or segs[-1]] = nid
    return out


def main():
    crate = sys.argv[1]
    src = os.path.join(crate, "src")
    lines = files(crate)
    texts = {p: kept_text(os.path.join(src, p)) for p in lines}
    defs = {}
    per_file_items = {}
    for p, t in texts.items():
        its = items(t)
        per_file_items[p] = its
        k = 0
        for kind, name, gen, of, s, e in its:
            if kind in ("use", "mod"):
                continue
            for nm in (name, gen):
                if nm:
                    defs.setdefault(nm, set()).add((p, k))
            k += 1
    unique = {k: next(iter(v))[0] for k, v in defs.items() if len(v) == 1}
    local_defs = collections.defaultdict(dict)
    for nm, locs in defs.items():
        for q, k in locs:
            local_defs[q][nm] = f"{q}#{k}"
    unique_item = {k: "%s#%d" % next(iter(v)) for k, v in defs.items() if len(v) == 1}
    res = {}
    for p, t in texts.items():
        with open(os.path.join(src, p), encoding="utf-8") as f:
            raw = f.read()
        raw_lines = raw.split("\n")
        starts = [0]
        for line in raw_lines[:-1]:
            starts.append(starts[-1] + len(line) + 1)
        its = []
        resolve = dict(unique_item)
        imap = import_map(p, t, local_defs)
        resolve.update(imap)
        resolve.update(local_defs[p])
        for kind, name, gen, of, s, e in per_file_items[p]:
            if kind in ("use", "mod"):
                continue
            body = t[s:e]
            refs = crossrefs(body, resolve, {name, gen}, p, imap)
            ref_ids = [resolve[x] for x in refs]
            calls = sorted(set(CALL.findall(body)))
            args = []
            if kind == "impl" and of:
                head = body.split("{", 1)[0]
                tail = head[head.find(" of ") + 4 :] if " of " in head else ""
                args = sorted({x for x in IDENT.findall(tail) if x in unique and x != of})
            head_vis = body[: body.find(kind)].strip().split()[-1:] if body.find(kind) > 0 else []
            entry = {"kind": kind, "name": name, "gen": gen, "of": of, "lines": body.count("\n") + 1,
                     "public": bool(head_vis) and head_vis[0] == "pub",
                     "span": [raw_start(raw_lines, starts, s), e],
                     "refs": refs, "ref_ids": ref_ids, "calls": calls, "args": args}
            if kind in ("impl", "trait"):
                ms = []
                for mn, ms_, me in methods(body):
                    mb = body[ms_:me]
                    mrefs = crossrefs(mb, resolve, {name, gen}, p, imap)
                    ms.append({"name": mn, "lines": mb.count("\n") + 1, "refs": mrefs})
                entry["methods"] = ms
            its.append(entry)
        imports = sorted({x for u in USE.findall(t) for x in IDENT.findall(u.split("::")[-1] if "{" not in u else u[u.index("{"):])})
        res[p] = {"lines": lines[p], "items": its, "imports": imports, "import_ids": imap}
    out = {"defs": unique, "items": unique_item, "files": res}
    path = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else "/dev/stdout"
    with open(path, "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    main()
