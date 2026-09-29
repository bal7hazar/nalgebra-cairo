#!/usr/bin/env python3
"""Manifest compatibility of the facade's no-op features (docs/SPLIT.md §17, WP 9-NS11b).

`statistics`, `blas`, `dynamic`, `sparse` and `io` save nothing since the package split (their code
lives in sub-crates the facade always depends on) but stay declared in 0.1.1 so that a consumer
manifest naming them keeps building; they are removed in 0.2.0. This script builds two throwaway
consumers of the workspace facade (by path), each naming the five features:

  * `default-features = false, features = [the five]`;
  * default features plus `features = [the five]`;

and each consumer's code names one item behind each feature, so the build also proves the feature
still exposes what it exposed in 0.1.0.

Usage
  python3 scripts/facade_features.py               # build both consumers (CI job `Facade features`)
  python3 scripts/facade_features.py --resolve     # `scarb metadata` only (cheap, no build)
  python3 scripts/facade_features.py --dry-run     # print the manifests
Exit status: 0 when every consumer resolves (and builds), 1 otherwise.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

FACADE = "nalgebra"
NO_OP = ["statistics", "blas", "dynamic", "sparse", "io"]
# One public 0.1.0 path per no-op feature (tools/split/public_paths_0.1.0.txt).
ITEMS = {
    "statistics": "nalgebra::base::statistics::Matrix2StatisticsTrait",
    "blas": "nalgebra::base::blas::Matrix1BlasTrait",
    "dynamic": "nalgebra::DMatrix",
    "sparse": "nalgebra::sparse::CsMatrix",
    "io": "nalgebra::io::cs_matrix_from_matrix_market_str",
}
CONSUMERS = {
    "no_default": {"default-features": False},
    "defaults": {},
}


def facade_root(scarb):
    meta = json.loads(subprocess.run([scarb, "metadata", "--format-version", "1", "--no-deps"],
                                     check=True, stdout=subprocess.PIPE, text=True).stdout)
    for p in meta["packages"]:
        if p["name"] == FACADE:
            return p["root"], p["edition"]
    sys.exit(f"error: no package `{FACADE}` in the workspace")


def manifest(name, root, edition, opts):
    parts = [f"path = {json.dumps(root)}"]
    if opts.get("default-features") is False:
        parts.append("default-features = false")
    parts.append("features = " + json.dumps(NO_OP))
    return (f'[package]\nname = "facade_features_{name}"\nversion = "0.1.0"\nedition = "{edition}"\n\n'
            f"[dependencies]\n{FACADE} = {{ {', '.join(parts)} }}\n")


def source():
    uses = "".join(f"#[allow(unused_imports)]\nuse {ITEMS[f]}; // feature `{f}`\n" for f in NO_OP)
    return uses + "\npub fn facade_features_answer() -> u32 {\n    42\n}\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--resolve", action="store_true", help="resolve only (`scarb metadata`)")
    ap.add_argument("--dry-run", action="store_true", help="print the manifests, run nothing")
    ap.add_argument("--scarb", default="scarb")
    args = ap.parse_args()
    root, edition = facade_root(args.scarb)
    failed = []
    with tempfile.TemporaryDirectory(prefix="facade_features_") as tmp:
        for name, opts in CONSUMERS.items():
            d = os.path.join(tmp, name)
            os.makedirs(os.path.join(d, "src"))
            text = manifest(name, root, edition, opts)
            with open(os.path.join(d, "Scarb.toml"), "w") as f:
                f.write(text)
            with open(os.path.join(d, "src", "lib.cairo"), "w") as f:
                f.write(source())
            print(f"## consumer `{name}`\n```toml\n{text}```", flush=True)
            if args.dry_run:
                continue
            cmd = ([args.scarb, "metadata", "--format-version", "1"] if args.resolve
                   else [args.scarb, "build"])
            p = subprocess.run(cmd, cwd=d, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            ok = p.returncode == 0
            print(f"`{' '.join(cmd[1:])}`: {'ok' if ok else 'FAILED'}", flush=True)
            if not ok:
                print(p.stdout[-4000:])
                failed.append(name)
    if failed:
        print(f"error: the no-op features {NO_OP} no longer resolve / build for: {', '.join(failed)}")
        return 1
    if not args.dry_run:
        print(f"every consumer naming {', '.join(NO_OP)} {'resolves' if args.resolve else 'builds'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
