#!/usr/bin/env python3
"""Prepare and check a release of the workspace packages. It never publishes.

Publishing is done by hand by the orchestrator, one `scarb publish -p <package>` per package, on the
project manager's go naming package, version, commit and archive sha256 (several packages may share
one request and one go). This script writes that request and reads the registry back afterwards.

  python3 scripts/release.py request --commit SHA            # build the archives, write the request
  python3 scripts/release.py request --commit SHA --dry-run --packages a,b --out /tmp/r.md
  python3 scripts/release.py verify --request docs/releases/V.md [--packages a] [--wait]
  python3 scripts/release.py --self-test                     # pure logic: no network, no scarb, no git

Package set and order (no name is hard-coded): `scarb metadata` lists the workspace packages; the
published ones are those whose manifest does not say `publish = false`. Their path dependencies
(normal ones: dev-dependencies are not needed to publish) give a graph; the packages another
published package depends on come first, in topological order (ties by name), then the "leaf"
packages nothing depends on (the facade, then the bridges), the one with the most workspace
dependencies first. A published package with a path dependency on an unpublished one is refused.

Request mode, from a clean checkout detached at the release commit (`--commit`). It refuses
(`--dry-run` reports the same checks and goes on) unless:
  * it covers every published package (`--packages`, a subset, is for dry runs only);
  * the working tree is clean, HEAD is detached and is `--commit`;
  * every check run of that commit on GitHub (`gh api .../commits/SHA/check-runs`) has completed
    with success / neutral / skipped, and one of them is `Consumer cost` (the enforcing
    package-granularity gate, docs/SPLIT.md §12.2) with success (skipped refuses);
  * every published package has the workspace version (`[workspace.package] version`);
  * that version of each package is not on the registry index yet.
It then builds each package's archive, in publication order, with `scarb package --no-verify -p NAME`
(through the shims: on the VPS it takes the heavy-build lock): the archive bytes are the same with or
without the verification, and the verification (building the unpacked archive against the registry)
cannot pass before the package's workspace dependencies are on the registry at the new version; each
`scarb publish -p NAME` verifies its package, in order, once they are. The archive records the commit
(`VCS.json`), so `scarb publish` from the same clean checkout rebuilds the same bytes. It writes
`docs/releases/VERSION.md` (or `--out`): a table (package, version, commit, sha256, archive size
compressed and unpacked) in publication order, then the ordered publication commands.

Verify mode reads the request's table back and compares, for each package (or `--packages`), the
registry index checksum with the requested sha256. `--wait` polls the index every VERIFY_DELAY s
until it lists every package checked or `--verify-timeout` s have passed (a wall-clock cap: the last
check may start just before it and take up to 30 s, one index fetch, per package checked). Exit 0
only when every package checked is on the index with the requested checksum.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

REQUIRED_CHECK = "Consumer cost"
OK_CONCLUSIONS = {"success", "neutral", "skipped"}
VERIFY_DELAY = 20  # seconds between two polls of the registry index
REQUEST_COLUMNS = ("#", "Package", "Version", "Commit", "sha256", "Compressed (bytes)", "Unpacked (bytes)")


# ------------------------------------------------------------------------------------------------
# Pure logic
# ------------------------------------------------------------------------------------------------


def publication_order(packages):
    """`packages`: {name: [workspace path dependencies that are published]} -> the order.
    Non-leaves (something depends on them) in topological order, ties by name; then the leaves,
    the most dependencies first (ties by name)."""
    dependents = {n: set() for n in packages}
    for n, deps in packages.items():
        for d in deps:
            if d not in packages:
                raise ValueError(f"`{n}` depends on `{d}`, which is not published")
            dependents[d].add(n)
    leaves = sorted((n for n in packages if not dependents[n]), key=lambda n: (-len(packages[n]), n))
    inner = [n for n in packages if dependents[n]]
    done, order = set(), []
    remaining = set(inner)
    while remaining:
        ready = sorted(n for n in remaining if all(d in done for d in packages[n]))
        if not ready:
            raise ValueError(f"dependency cycle among {sorted(remaining)}")
        n = ready[0]
        order.append(n)
        done.add(n)
        remaining.remove(n)
    for n in leaves:
        missing = [d for d in packages[n] if d not in done]
        if missing:  # a leaf depending on another leaf cannot happen (it would not be a leaf)
            raise ValueError(f"`{n}`: dependencies {missing} not ordered before it")
        order.append(n)
        done.add(n)
    return order


def select_packages(order, wanted):
    """The packages of ORDER named by `--packages` (comma-separated, None: all), in ORDER."""
    if not wanted:
        return list(order)
    names = [w.strip() for w in wanted.split(",") if w.strip()]
    unknown = [n for n in names if n not in order]
    if unknown:
        raise ValueError(f"not a published package of the workspace: {', '.join(unknown)}")
    return [n for n in order if n in names]


def index_prefix(name):
    """The cargo-style index directory of a package name (`nalgebra` -> `na/lg`)."""
    n = name.lower()
    if len(n) <= 2:
        return str(len(n))
    if len(n) == 3:
        return f"3/{n[0]}"
    return f"{n[:2]}/{n[2:4]}"


def check_runs_verdict(runs, required=REQUIRED_CHECK):
    """(ok, problems) for the check runs of a commit (`gh api` JSON `check_runs`). Every run must
    have completed with success / neutral / skipped, and the REQUIRED one with success: a skipped
    enforcing gate (CI path gating) vouches for nothing."""
    problems = []
    if not runs:
        problems.append("no check run on the release commit")
    names = set()
    for r in runs:
        names.add(r["name"])
        if r["status"] != "completed":
            problems.append(f"`{r['name']}` is {r['status']}")
        elif r["name"] == required and r["conclusion"] != "success":
            problems.append(f"`{r['name']}` concluded {r['conclusion']} (it must succeed)")
        elif r["conclusion"] not in OK_CONCLUSIONS:
            problems.append(f"`{r['name']}` concluded {r['conclusion']}")
    if required not in names:
        problems.insert(0, f"no `{required}` check run on the release commit")
    return not problems, problems


def request_problems(packages, dry_run):
    """Why the options cannot make a real request: a release request lists every published
    package, so `--packages` is for dry runs only."""
    if packages and not dry_run:
        return ["`--packages` is for a dry run only: a release request lists every published package"]
    return []


def checkout_problems(head, commit, detached, dirty):
    """Why the checkout is not a clean checkout detached at COMMIT (empty: it is)."""
    problems = []
    if dirty:
        problems.append("the working tree is not clean")
    if not detached:
        problems.append("HEAD is not detached (check out the release commit with `git checkout --detach SHA`)")
    if head != commit:
        problems.append(f"HEAD {head[:10]} is not the release commit {commit[:10]}")
    return problems


def render_request(version, commit, rows, header, dry_run_problems=None):
    """The request document. `rows`: [{name, sha256, compressed, unpacked}] in publication order;
    `header`: lines placed under the title (tooling, CI, registry). `dry_run_problems`: None for a
    real request, else the checks a real request would have refused on (the document says so)."""
    out = [f"# Release request: nalgebra-cairo {version}", ""]
    if dry_run_problems is not None:
        out += ["**DRY RUN: not a release request.** Checks a real request refuses on:", ""]
        out += [f"- {p}" for p in dry_run_problems] or ["- none"]
        out.append("")
    else:
        out += ["Status: requested, not published.", ""]
    out += header + [""]
    out.append("| " + " | ".join(REQUEST_COLUMNS) + " |")
    out.append("|" + "|".join("---:" if c in ("#",) or "bytes" in c else "---" for c in REQUEST_COLUMNS) + "|")
    for k, r in enumerate(rows, 1):
        unpacked = "n/a" if r["unpacked"] is None else str(r["unpacked"])
        out.append(f"| {k} | `{r['name']}` | {version} | `{commit}` | `{r['sha256']}` | "
                   f"{r['compressed']} | {unpacked} |")
    out += ["", "## Publication", "",
            "By hand, from the same clean checkout detached at the commit, one package at a time in the",
            "order of the table, on the project manager's go naming every row. `scarb publish` verifies",
            "each package by building it against the registry, so a package is published only once the",
            "index lists the packages before it: the `verify --wait` line after each publish waits for",
            "that and compares the registry checksum with the table.", "", "```sh"]
    for r in rows:
        out.append(f"scarb publish -p {r['name']}")
        out.append(f"python3 scripts/release.py verify --request docs/releases/{version}.md "
                   f"--packages {r['name']} --wait")
    out += ["```", ""]
    return "\n".join(out)


ROW = re.compile(r"^\|\s*\d+\s*\|\s*`([^`]+)`\s*\|\s*([^|\s]+)\s*\|\s*`([0-9a-f]{40})`\s*\|"
                 r"\s*`([0-9a-f]{64})`\s*\|")


def parse_request(text):
    """(version, [{name, version, commit, sha256}]) from a request document; a dry run is refused."""
    if "**DRY RUN" in text:
        raise ValueError("this is a dry-run document, not a release request")
    rows = []
    for line in text.splitlines():
        m = ROW.match(line)
        if m:
            rows.append({"name": m[1], "version": m[2], "commit": m[3], "sha256": m[4]})
    if not rows:
        raise ValueError("no package row found")
    versions = {r["version"] for r in rows}
    commits = {r["commit"] for r in rows}
    if len(versions) != 1 or len(commits) != 1:
        raise ValueError(f"one version and one commit expected, found {sorted(versions)} / {sorted(commits)}")
    names = [r["name"] for r in rows]
    if len(set(names)) != len(names):
        raise ValueError("a package is listed twice")
    return versions.pop(), rows


def compare_checksums(rows, records):
    """[(name, state, detail)] with state `ok`, `missing` (not on the index) or `mismatch`.
    `records`: {name: index record or None}; the index writes checksums as `sha256:<hex>`."""
    result = []
    for r in rows:
        rec = records.get(r["name"])
        if rec is None:
            result.append((r["name"], "missing", "not on the index at this version"))
        elif rec.get("cksum") != "sha256:" + r["sha256"]:
            result.append((r["name"], "mismatch", f"index {rec.get('cksum')} != request sha256:{r['sha256']}"))
        else:
            result.append((r["name"], "ok", rec.get("cksum")))
    return result


def wait_until(check, timeout, delay, sleep=time.sleep, clock=time.monotonic):
    """Call `check()` (True when done) until it is done or TIMEOUT s of wall-clock time have passed,
    DELAY s apart; no check starts after the deadline. Returns whether it is done."""
    deadline = clock() + timeout
    while True:
        if check():
            return True
        left = deadline - clock()
        if left <= 0:
            return False
        sleep(min(delay, left))


def self_test():
    g = {"core": [], "a": ["core"], "b": ["core", "a"], "facade": ["core", "a", "b"],
         "bridge": ["core", "a"]}
    assert publication_order(g) == ["core", "a", "b", "facade", "bridge"], publication_order(g)
    try:
        publication_order({"x": ["y"], "y": ["x"], "z": ["x"]})
        raise AssertionError("cycle not detected")
    except ValueError:
        pass
    try:
        publication_order({"x": ["unpublished"]})
        raise AssertionError("unpublished dependency not detected")
    except ValueError:
        pass
    order = publication_order(g)
    assert select_packages(order, None) == order
    assert select_packages(order, "b, core") == ["core", "b"]  # publication order, not the given one
    try:
        select_packages(order, "core,nope")
        raise AssertionError("unknown package not detected")
    except ValueError:
        pass
    assert index_prefix("nalgebra") == "na/lg" and index_prefix("abc") == "3/a"
    assert index_prefix("ab") == "2" and index_prefix("a") == "1"
    ok, _ = check_runs_verdict([{"name": "Consumer cost", "status": "completed", "conclusion": "success"},
                                {"name": "Oracle", "status": "completed", "conclusion": "skipped"}])
    assert ok
    ok, p = check_runs_verdict([{"name": "X", "status": "completed", "conclusion": "success"}])
    assert not ok and "Consumer cost" in p[0]
    ok, p = check_runs_verdict([{"name": "Consumer cost", "status": "in_progress", "conclusion": None}])
    assert not ok
    ok, p = check_runs_verdict([{"name": "Consumer cost", "status": "completed", "conclusion": "failure"}])
    assert not ok
    for skipped in ("skipped", "neutral"):  # the enforcing gate must have run and succeeded
        ok, p = check_runs_verdict([{"name": "Consumer cost", "status": "completed", "conclusion": skipped}])
        assert not ok and "must succeed" in p[0], p
    assert request_problems(None, True) == [] and request_problems("a", True) == []
    assert request_problems(None, False) == [] and "dry run" in request_problems("a,b", False)[0]
    sha, other = "a" * 40, "b" * 40
    assert checkout_problems(sha, sha, True, False) == []
    assert len(checkout_problems(other, sha, False, True)) == 3
    assert "detached" in checkout_problems(sha, sha, False, False)[0]
    # request -> parse round trip, in order, and the commands
    rows = [{"name": "core", "sha256": "1" * 64, "compressed": 120, "unpacked": 400},
            {"name": "a", "sha256": "2" * 64, "compressed": 80, "unpacked": None}]
    doc = render_request("9.9.9", sha, rows, ["Commit: `" + sha + "`"])
    version, back = parse_request(doc)
    assert version == "9.9.9" and [r["name"] for r in back] == ["core", "a"]
    assert back[1] == {"name": "a", "version": "9.9.9", "commit": sha, "sha256": "2" * 64}
    assert doc.index("scarb publish -p core") < doc.index("scarb publish -p a") and "| n/a |" in doc
    assert "DRY RUN" not in doc
    dry = render_request("9.9.9", sha, rows, [], dry_run_problems=["HEAD is not detached"])
    assert "DRY RUN" in dry and "HEAD is not detached" in dry
    core_row = next(line for line in doc.splitlines() if line.startswith("| 1 |"))
    for bad in (dry, "# nothing\n", doc + "\n" + core_row.replace("| 1 |", "| 3 |") + "\n",
                doc.replace("| 2 | `a` | 9.9.9 |", "| 2 | `a` | 9.9.8 |")):
        try:
            parse_request(bad)
            raise AssertionError("a bad request was parsed")
        except ValueError:
            pass
    # verify: index checksums against the request
    res = compare_checksums(back, {"core": {"v": "9.9.9", "cksum": "sha256:" + "1" * 64}, "a": None})
    assert res[0][1] == "ok" and res[1][1] == "missing"
    res = compare_checksums(back[:1], {"core": {"v": "9.9.9", "cksum": "sha256:" + "f" * 64}})
    assert res[0][1] == "mismatch"
    # wait_until: a wall-clock cap, no check after the deadline
    now, naps, calls = [0.0], [], []

    def nap(s):
        naps.append(s)
        now[0] += s

    def done_at(k):
        def check():
            calls.append(now[0])
            return len(calls) >= k
        return check
    assert wait_until(done_at(3), 900, 20, nap, lambda: now[0]) and len(calls) == 3 and naps == [20, 20]
    now[0], calls[:], naps[:] = 0.0, [], []
    assert not wait_until(done_at(99), 50, 20, nap, lambda: now[0])
    assert naps == [20, 20, 10] and calls == [0, 20, 40, 50]  # the last nap is cut to the deadline
    now[0], calls[:], naps[:] = 0.0, [], []
    assert not wait_until(done_at(99), 0, 20, nap, lambda: now[0]) and len(calls) == 1 and not naps
    print("self-test: ok")
    return 0


# ------------------------------------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------------------------------------


def sh(cmd, check=True, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if check and p.returncode != 0:
        sys.exit(f"error: {' '.join(cmd)} failed ({p.returncode}):\n{p.stderr[-2000:]}")
    return p


def workspace(scarb):
    meta = json.loads(sh([scarb, "metadata", "--format-version", "1", "--no-deps"]).stdout)
    root = meta["workspace"]["root"]
    with open(os.path.join(root, "Scarb.toml"), "rb") as f:
        ws_version = tomllib.load(f).get("workspace", {}).get("package", {}).get("version")
    by_manifest = {}
    pkgs = {}
    for p in meta["packages"]:
        with open(p["manifest_path"], "rb") as f:
            package = tomllib.load(f).get("package", {})
        by_manifest[os.path.normpath(p["manifest_path"])] = p["name"]
        pkgs[p["name"]] = {"version": p["version"], "publish": package.get("publish", True) is not False,
                           "deps": p["dependencies"]}
    graph = {}
    for n, p in pkgs.items():
        if not p["publish"]:
            continue
        deps = []
        for d in p["deps"]:
            if d.get("kind") is not None or not d["source"].startswith("path+file://"):
                continue
            dep = by_manifest.get(os.path.normpath(d["source"][len("path+file://"):]), d["name"])
            deps.append(dep)
        graph[n] = sorted(set(deps))
    return root, ws_version, pkgs, graph


def registry_index(registry):
    with urllib.request.urlopen(f"{registry.rstrip('/')}/api/v1/index/config.json", timeout=30) as r:
        return json.load(r)["index"]


def index_entry(template, name, version):
    """The index record of NAME at VERSION, or None (404 or version absent)."""
    url = template.replace("{prefix}", index_prefix(name)).replace("{package}", name)
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            records = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    for rec in records:
        if rec.get("v") == version:
            return rec
    return None


def gh_check_runs(repo, sha):
    """The check runs of SHA, or (None, the error) when GitHub cannot give them (e.g. not pushed)."""
    p = sh(["gh", "api", "--paginate", f"repos/{repo}/commits/{sha}/check-runs",
            "-q", ".check_runs[] | {name, status, conclusion}"], check=False)
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["gh api failed"])[-1]
    return [json.loads(line) for line in p.stdout.splitlines() if line.strip()], None


def unpacked_size(path):
    """The total size of the files in a `.tar.zst` archive, or None when no zstd decoder is found."""
    try:
        from compression import zstd  # Python >= 3.14
        with zstd.open(path) as z, tarfile.open(fileobj=z, mode="r|") as t:
            return sum(m.size for m in t if m.isfile())
    except ImportError:
        pass
    if shutil.which("zstd") is None:
        return None
    with subprocess.Popen(["zstd", "-dcq", path], stdout=subprocess.PIPE) as p:
        with tarfile.open(fileobj=p.stdout, mode="r|") as t:
            total = sum(m.size for m in t if m.isfile())
    return total if p.returncode == 0 else None


def build_archive(scarb, root, name, version):
    """`scarb package --no-verify -p NAME` (see the docstring), then the archive's sha256 and sizes."""
    p = subprocess.run([scarb, "package", "--no-verify", "-p", name], cwd=root,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        sys.exit(f"error: scarb package -p {name} failed ({p.returncode}):\n{p.stdout[-2000:]}")
    path = os.path.join(root, "target", "package", f"{name}-{version}.tar.zst")
    with open(path, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    return {"name": name, "sha256": digest, "compressed": os.path.getsize(path),
            "unpacked": unpacked_size(path)}


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


def request(args):
    refused = request_problems(args.packages, args.dry_run)
    if refused:
        sys.exit(f"error: {refused[0]}")
    root, version, pkgs, graph = workspace(args.scarb)
    try:
        order = publication_order(graph)
        chosen = select_packages(order, args.packages)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"workspace version: {version}; {len(order)} published packages, {len(chosen)} in this request")
    problems = []

    wrong = [f"{n} {pkgs[n]['version']}" for n in chosen if pkgs[n]["version"] != version]
    if wrong:
        problems.append(f"packages not at the workspace version {version}: {', '.join(wrong)}")

    commit = sh(["git", "rev-parse", "--verify", "--quiet", args.commit + "^{commit}"], cwd=root).stdout.strip()
    head = sh(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    detached = sh(["git", "symbolic-ref", "-q", "HEAD"], check=False, cwd=root).returncode != 0
    dirty = bool(sh(["git", "status", "--porcelain"], cwd=root).stdout.strip())
    problems += checkout_problems(head, commit, detached, dirty)

    repo = args.repo or sh(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                           cwd=root).stdout.strip()
    runs, error = gh_check_runs(repo, commit)
    ok, ci = check_runs_verdict(runs or [])
    if error:
        ci = [f"cannot read the check runs of {commit[:10]}: {error}"]
        runs = []
    if len(ci) > 6:
        ci = ci[:5] + [f"... and {len(ci) - 5} more"]
    print(f"CI of {commit[:10]} ({repo}): {'green, `' + REQUIRED_CHECK + '` included' if ok else 'NOT green'}")
    problems += [f"CI: {p}" for p in ci]

    template = registry_index(args.registry)
    listed = [n for n in chosen if index_entry(template, n, version) is not None]
    if listed:
        problems.append(f"already on the registry at {version} (bump the workspace version): "
                        + ", ".join(listed[:5]) + (f" and {len(listed) - 5} more" if len(listed) > 5 else ""))

    if problems:
        print("\nchecks that refuse a release request:" if not args.dry_run else "\nchecks a real request refuses on:")
        for p in problems:
            print(f"  - {p}")
        if not args.dry_run:
            return 1

    rows = []
    for k, n in enumerate(chosen, 1):
        print(f"  {k:2}/{len(chosen)} scarb package --no-verify -p {n}", flush=True)
        rows.append(build_archive(args.scarb, root, n, version))
    tool = sh([args.scarb, "--version"], cwd=root).stdout.splitlines()[0].strip()
    subject = sh(["git", "log", "-1", "--format=%s", commit], cwd=root).stdout.strip()
    header = [
        f"Commit: `{commit}` ({subject}).",
        f"Archives: `scarb package --no-verify -p <package>` with {tool}, "
        f"{time.strftime('%Y-%m-%d', time.gmtime())}, from a checkout detached at the commit; sha256 of "
        f"`target/package/<package>-{version}.tar.zst`. The registry checksum must equal it.",
        f"CI of the commit ({repo}): {'green, `' + REQUIRED_CHECK + '` included' if ok else 'NOT green'}, "
        f"{len(runs)} check runs.",
        f"Registry: {args.registry}; {'none of these packages is' if not listed else 'some packages are'} "
        f"on its index at {version}.",
        f"Packages: {len(rows)} of the {len(order)} published packages of the workspace, in publication order.",
    ]
    doc = render_request(version, commit, rows, header, problems if args.dry_run else None)
    out = args.out or os.path.join(root, "docs", "releases", f"{version}.md")
    with open(out, "w") as f:
        f.write(doc)
    print(f"\nwrote {out} ({len(rows)} packages){' - DRY RUN' if args.dry_run else ''}")
    return 0


def verify(args):
    try:
        with open(args.request) as f:
            version, rows = parse_request(f.read())
        names = [r["name"] for r in rows]
        chosen = select_packages(names, args.packages)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {args.request}: {e}")
    rows = [r for r in rows if r["name"] in chosen]
    template = registry_index(args.registry)
    records = {}

    def check():
        for r in rows:
            if records.get(r["name"]) is None:
                records[r["name"]] = index_entry(template, r["name"], version)
        return all(records[r["name"]] is not None for r in rows)

    if args.wait:
        print(f"waiting up to {args.verify_timeout} s for {len(rows)} package(s) at {version} on the index "
              f"(a check every {VERIFY_DELAY} s)", flush=True)
        wait_until(check, args.verify_timeout, VERIFY_DELAY)
    else:
        check()
    result = compare_checksums(rows, records)
    for name, state, detail in result:
        print(f"  {state:8} {name} {version}: {detail}")
    bad = [r for r in result if r[1] != "ok"]
    print(f"\n{len(result) - len(bad)} of {len(result)} package(s) on the index with the requested checksum")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description="Release request and registry check (never publishes).")
    ap.add_argument("--self-test", action="store_true", help="checks of the pure logic, no network, no scarb")
    sub = ap.add_subparsers(dest="mode")
    rq = sub.add_parser("request", help="build the archives and write docs/releases/VERSION.md")
    rq.add_argument("--commit", required=True, help="the release commit (HEAD must be detached at it)")
    rq.add_argument("--packages", help="with --dry-run only: comma-separated subset (default: every published package)")
    rq.add_argument("--dry-run", action="store_true", help="report the refusing checks and go on")
    rq.add_argument("--out", help="output file (default: docs/releases/VERSION.md)")
    rq.add_argument("--repo", help="GitHub OWNER/REPO (default: from `gh repo view`)")
    rq.add_argument("--scarb", default="scarb")
    rq.add_argument("--registry", default="https://scarbs.xyz")
    vf = sub.add_parser("verify", help="compare the registry index checksums with a request")
    vf.add_argument("--request", required=True, help="a docs/releases/VERSION.md request")
    vf.add_argument("--packages", help="comma-separated subset of the request (default: all of it)")
    vf.add_argument("--wait", action="store_true", help="poll the index until it lists them")
    vf.add_argument("--verify-timeout", type=int, default=900,
                    help="with --wait: wall-clock seconds to wait for the index (default 900)")
    vf.add_argument("--registry", default="https://scarbs.xyz")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.mode is None:
        ap.error("a mode is needed: request, verify or --self-test")
    if args.mode == "request":
        if tomllib is None:
            sys.exit("error: Python 3.11+ is needed (tomllib)")
        return request(args)
    return verify(args)


if __name__ == "__main__":
    sys.exit(main())
