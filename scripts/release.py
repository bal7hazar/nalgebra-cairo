#!/usr/bin/env python3
"""Publish every published package of the workspace on the registry, in dependency order, resumably.

DRY RUN BY DEFAULT: it prints the checks, the publication order and the commands, and publishes
nothing. `--publish` publishes (the release go of docs/ORCHESTRATOR.md "Releases" is a precondition,
not something this script can check).

  python3 scripts/release.py                 # dry run: checks + order + commands
  python3 scripts/release.py --publish       # publish (resumes from the state file if any)
  python3 scripts/release.py --self-test     # checks of the pure logic, no network, no scarb

Package set and order (no name is hard-coded): `scarb metadata` lists the workspace packages; the
published ones are those whose manifest does not say `publish = false`. Their path dependencies
(normal ones: dev-dependencies are not needed to publish) give a graph; the packages another
published package depends on come first, in topological order (ties by name), then the "leaf"
packages nothing depends on (the facade, then the bridges), the one with the most workspace
dependencies first. A published package with a path dependency on an unpublished one is refused.

Refuses to start (with `--publish`; the dry run reports the same checks) unless:
  * the working tree is clean and HEAD is `origin/main` (the release commit);
  * every check run of the release commit on GitHub (`gh api .../commits/SHA/check-runs`) has
    completed with success / neutral / skipped, and one of them is `Consumer cost` (the enforcing
    package-granularity gate, docs/SPLIT.md §12.2) with success;
  * every published package has the workspace version (`[workspace.package] version`);
  * that version of each package is NOT on the registry index yet, except the packages the state
    file records as published by an earlier run of this release (resume).
  * `SCARB_REGISTRY_AUTH_TOKEN` is set (never printed).

Publication: one package at a time, `scarb publish -p NAME`; then the registry index is polled until
it lists NAME at the version, and its checksum is compared with the local archive
(`target/package/NAME-VERSION.tar.zst`) before the next package starts.

Resume: the state file `target/release/state-VERSION.json` records the release commit and every
package once verified (and the one in flight). After a failure at package k, run the same command
again: the packages before k are checked on the index and skipped, k is published again (or, if the
index already lists it, verified and recorded), then k+1... A state file of another commit is refused,
with one exception: a release may continue at a later commit (a fix of a package not published yet)
when `git diff --name-only OLD HEAD` touches no file of an already published package (its directory,
and its `readme` / `license-file` when they live elsewhere) and neither the workspace `Scarb.toml`
nor `Scarb.lock`: what the registry holds is then exactly what HEAD would publish. The check and
its outcome are printed; the state file then records HEAD and the commit it continued from.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

REQUIRED_CHECK = "Consumer cost"
OK_CONCLUSIONS = {"success", "neutral", "skipped"}


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


def index_prefix(name):
    """The cargo-style index directory of a package name (`nalgebra` -> `na/lg`)."""
    n = name.lower()
    if len(n) <= 2:
        return str(len(n))
    if len(n) == 3:
        return f"3/{n[0]}"
    return f"{n[:2]}/{n[2:4]}"


def check_runs_verdict(runs, required=REQUIRED_CHECK):
    """(ok, problems) for the check runs of a commit (`gh api` JSON `check_runs`)."""
    problems = []
    if not runs:
        problems.append("no check run on the release commit")
    names = set()
    for r in runs:
        names.add(r["name"])
        if r["status"] != "completed":
            problems.append(f"`{r['name']}` is {r['status']}")
        elif r["conclusion"] not in OK_CONCLUSIONS:
            problems.append(f"`{r['name']}` concluded {r['conclusion']}")
    if required not in names:
        problems.insert(0, f"no `{required}` check run on the release commit")
    return not problems, problems


def plan_resume(order, state, on_index):
    """What to do for each package: `skip` (the state records it and the index lists it),
    `verify` (the index lists it: only allowed for the package the state says was in flight),
    `publish`; plus the refusals. `on_index`: {name: bool}."""
    actions, refusals = {}, []
    published = set(state.get("published", [])) if state else set()
    in_flight = state.get("in_flight") if state else None
    for n in order:
        if n in published:
            if on_index[n]:
                actions[n] = "skip"
            else:
                actions[n] = "verify"  # recorded but not visible yet: wait for the index
        elif on_index[n]:
            if n == in_flight:
                actions[n] = "verify"
            else:
                actions[n] = "refuse"
                refusals.append(f"`{n}` is already on the registry at this version "
                                "(bump the workspace version)")
        else:
            actions[n] = "publish"
    return actions, refusals


WORKSPACE_FILES = ("Scarb.toml", "Scarb.lock")


def continuation_problems(changed, published_paths):
    """Whether a release begun at an older commit may continue at HEAD. `changed`: the files
    `git diff --name-only OLD HEAD` lists (relative to the repository root); `published_paths`:
    {package: [its directory, and its readme / license file when outside it]} for every package
    already on the registry at this version. Returns the problems (empty: it may continue)."""
    problems = []
    for f in changed:
        if f in WORKSPACE_FILES:
            problems.append(f"{f} changed (it reaches every published package)")
            continue
        for n, paths in sorted(published_paths.items()):
            if any(f == q or f.startswith(q.rstrip("/") + "/") for q in paths):
                problems.append(f"{f} changed, and it belongs to {n}, already published")
                break
    return problems


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
    order = ["core", "a", "b"]
    acts, ref = plan_resume(order, None, {"core": False, "a": False, "b": False})
    assert acts == {"core": "publish", "a": "publish", "b": "publish"} and not ref
    acts, ref = plan_resume(order, None, {"core": True, "a": False, "b": False})
    assert acts["core"] == "refuse" and ref
    state = {"published": ["core"], "in_flight": "a"}
    acts, ref = plan_resume(order, state, {"core": True, "a": True, "b": False})
    assert acts == {"core": "skip", "a": "verify", "b": "publish"} and not ref
    published = {"core": ["crates/core"], "a": ["crates/a", "README.md"]}
    assert continuation_problems(["crates/b/Scarb.toml", "docs/PLAN.md"], published) == []
    assert continuation_problems(["crates/core/src/lib.cairo"], published)
    assert continuation_problems(["crates/core_extra/Scarb.toml"], published) == []
    assert continuation_problems(["README.md"], published)
    assert continuation_problems(["Scarb.lock"], published) and continuation_problems(["Scarb.toml"], published)
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
            manifest = tomllib.load(f)
        package = manifest.get("package", {})
        publish = package.get("publish", True) is not False
        by_manifest[os.path.normpath(p["manifest_path"])] = p["name"]
        pkg_dir = os.path.dirname(os.path.normpath(p["manifest_path"]))
        paths = [os.path.relpath(pkg_dir, root)]
        for key in ("readme", "license-file"):
            value = package.get(key)
            if isinstance(value, str):
                full = os.path.normpath(os.path.join(pkg_dir, value))
                if not full.startswith(pkg_dir + os.sep):
                    paths.append(os.path.relpath(full, root))
        pkgs[p["name"]] = {"version": p["version"], "publish": publish, "deps": p["dependencies"],
                           "paths": paths}
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
    out = sh(["gh", "api", "--paginate", f"repos/{repo}/commits/{sha}/check-runs",
              "-q", ".check_runs[] | {name, status, conclusion}"]).stdout
    return [json.loads(line) for line in out.splitlines() if line.strip()]


def local_checksum(root, name, version):
    path = os.path.join(root, "target", "package", f"{name}-{version}.tar.zst")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Resumable, verified release of the workspace packages.")
    ap.add_argument("--publish", action="store_true", help="really publish (default: dry run)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--registry", default="https://scarbs.xyz")
    ap.add_argument("--repo", help="GitHub OWNER/REPO (default: from `gh repo view`)")
    ap.add_argument("--scarb", default="scarb")
    ap.add_argument("--verify-timeout", type=int, default=900, help="seconds to wait for the index")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if tomllib is None:
        sys.exit("error: Python 3.11+ is needed (tomllib)")

    root, version, pkgs, graph = workspace(args.scarb)
    problems = []
    try:
        order = publication_order(graph)
    except ValueError as e:
        sys.exit(f"error: {e}")
    print(f"workspace version: {version}; {len(order)} published packages")

    # 1. Versions
    wrong = [f"{n} {pkgs[n]['version']}" for n in order if pkgs[n]["version"] != version]
    if wrong:
        problems.append(f"packages not at the workspace version {version}: {', '.join(wrong)}")

    # 2. Release commit: clean tree, HEAD = origin/main, CI green with `Consumer cost`
    sh(["git", "fetch", "-q", "origin", "main"], check=False, cwd=root)
    head = sh(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    main_sha = sh(["git", "rev-parse", "origin/main"], cwd=root).stdout.strip()
    if sh(["git", "status", "--porcelain"], cwd=root).stdout.strip():
        problems.append("the working tree is not clean")
    if head != main_sha:
        problems.append(f"HEAD {head[:10]} is not origin/main {main_sha[:10]} (the release commit)")
    repo = args.repo or sh(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                           cwd=root).stdout.strip()
    ok, ci = check_runs_verdict(gh_check_runs(repo, head))
    if len(ci) > 6:
        ci = ci[:5] + [f"... and {len(ci) - 5} more"]
    print(f"CI of {head[:10]} ({repo}): {'green, `' + REQUIRED_CHECK + '` included' if ok else 'NOT green'}")
    problems += [f"CI: {p}" for p in ci]

    # 3. Registry: versions not published yet, or recorded by the state file (resume)
    state_path = os.path.join(root, "target", "release", f"state-{version}.json")
    state = None
    if os.path.exists(state_path):
        with open(state_path) as f:
            state = json.load(f)
        print(f"resuming: {len(state.get('published', []))} package(s) already published "
              f"({state_path})")
    template = registry_index(args.registry)
    on_index = {n: index_entry(template, n, version) is not None for n in order}
    if state is not None and state.get("commit") != head:
        old = state.get("commit", "")
        if sh(["git", "merge-base", "--is-ancestor", old, head], check=False, cwd=root).returncode != 0:
            problems.append(f"{state_path} belongs to commit {old[:10]}, which is not an ancestor of HEAD")
        else:
            changed = [f for f in sh(["git", "diff", "--name-only", old, head], cwd=root).stdout.split("\n") if f]
            done = [n for n in order if on_index[n] and
                    (n in state.get("published", []) or n == state.get("in_flight"))]
            found = continuation_problems(changed, {n: pkgs[n]["paths"] for n in done})
            print(f"continuing the release begun at {old[:10]}: `git diff --name-only {old[:10]} {head[:10]}` "
                  f"lists {len(changed)} file(s); checked against the {len(done)} package(s) already on the "
                  f"index and {', '.join(WORKSPACE_FILES)}: {'refused' if found else 'none of them is touched'}")
            for f in changed:
                print(f"    {f}")
            problems += [f"cannot continue at HEAD: {p}" for p in found]
            if not found:
                state["continued_from"] = state.get("continued_from", []) + [old]
                state["commit"] = head
    actions, refusals = plan_resume(order, state, on_index)
    problems += refusals

    # 4. Token
    if not os.environ.get("SCARB_REGISTRY_AUTH_TOKEN"):
        problems.append("SCARB_REGISTRY_AUTH_TOKEN is not set")

    print("\npublication order:")
    for k, n in enumerate(order, 1):
        deps = ", ".join(graph[n]) or "-"
        print(f"  {k:2}. {n} {version}  [{actions[n]}]  (after: {deps})")
        if actions[n] == "publish":
            print(f"      $ {args.scarb} publish -p {n}; then wait for {n} {version} on the index and "
                  f"compare its checksum with target/package/{n}-{version}.tar.zst")
    if problems:
        print("\nrefusing to start:" if args.publish else "\nchecks that would refuse the release:")
        for p in problems:
            print(f"  - {p}")
    if not args.publish:
        print("\ndry run: nothing published (pass --publish after the release go)")
        return 1 if problems else 0
    if problems:
        return 1

    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    state = state or {"version": version, "commit": head, "published": [], "in_flight": None}

    def save():
        with open(state_path + ".tmp", "w") as f:
            json.dump(state, f, indent=2)
        os.replace(state_path + ".tmp", state_path)

    for n in order:
        if actions[n] == "skip":
            print(f"{n}: already published and on the index, skipped")
            continue
        if actions[n] == "publish":
            state["in_flight"] = n
            save()
            print(f"{n}: scarb publish -p {n}", flush=True)
            p = subprocess.run([args.scarb, "publish", "-p", n], cwd=root)
            if p.returncode != 0:
                sys.exit(f"error: publishing {n} failed; fix and run the same command again to resume")
        deadline = time.monotonic() + args.verify_timeout
        rec = None
        while rec is None:
            rec = index_entry(template, n, version)
            if rec is None:
                if time.monotonic() > deadline:
                    sys.exit(f"error: {n} {version} not on the index after {args.verify_timeout} s; "
                             "run again to resume (it will be verified, not republished)")
                time.sleep(20)
        local = local_checksum(root, n, version)
        if local is not None and rec.get("cksum") != local:
            sys.exit(f"error: {n} {version}: index checksum {rec.get('cksum')} != local archive {local}")
        state["published"].append(n)
        state["in_flight"] = None
        save()
        print(f"{n}: verified on the index ({rec.get('cksum')})", flush=True)
    print(f"\nreleased {len(order)} packages at {version}; tag v{version} per docs/ORCHESTRATOR.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
