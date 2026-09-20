#!/usr/bin/env python3
"""Turn `snforge test` output into a gas report.

Convention: benchmark tests are named `bench_<group>__<variant>`. Within a group, the optional
`baseline` variant measures the fixed overhead (inputs + assertions) and is subtracted from every
other variant to obtain the net cost of the operation.

Reports are keyed by the module path of the test (`package::module::…::tests`), so a CI shard
that runs only part of the workspace (`snforge test -p nalgebra nalgebra::base`) can check its
slice of the snapshot with `--filter nalgebra::base`.

Snapshots are split per CI shard (`simba`, `nalgebra::base`, ...) into `gas/<shard>.json` +
`gas/<shard>.md`, so parallel PRs on different modules never touch a common file.

Usage:
    snforge test --workspace | python3 scripts/gas_report.py --update gas/
    snforge test -p nalgebra nalgebra::base | python3 scripts/gas_report.py --check gas/ --filter nalgebra::base
    snforge test -p simba | python3 scripts/gas_report.py            # print a report
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

LINE = re.compile(r"\[PASS\]\s+(\S+)\s+\(.*l2_gas:\s*~?(\d+)\)")
NAME = re.compile(r"^bench_(?P<group>.+?)__(?P<variant>.+)$")


def parse(stream):
    """Return {module path: {group: {variant: gas}}}."""
    report = defaultdict(lambda: defaultdict(dict))
    for line in stream:
        match = LINE.search(line)
        if not match:
            continue
        path, gas = match.group(1), int(match.group(2))
        parts = path.split("::")
        name = NAME.match(parts[-1])
        if not name:
            continue
        report["::".join(parts[:-1])][name.group("group")][name.group("variant")] = gas
    return report


def net(variants):
    """Return {variant: (raw, net)} with the group baseline subtracted when present."""
    base = variants.get("baseline")
    return {
        variant: (gas, gas - base if base is not None else None)
        for variant, gas in variants.items()
        if variant != "baseline"
    }


def to_markdown(report):
    out = ["# Gas report", "", "Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.", ""]
    for module in sorted(report):
        out += [f"## {module}", ""]
        for group in sorted(report[module]):
            rows = net(report[module][group])
            if not rows:
                continue
            ranked = sorted(rows.items(), key=lambda kv: (kv[1][1] if kv[1][1] is not None else kv[1][0], kv[0]))
            best = ranked[0][1][1] if ranked[0][1][1] is not None else ranked[0][1][0]
            out += [f"### {group}", "", "| variant | raw | net | vs best |", "|---|---:|---:|---:|"]
            for variant, (raw, delta) in ranked:
                value = delta if delta is not None else raw
                ratio = f"x{value / best:.2f}" if best > 0 else "-"
                out.append(f"| `{variant}` | {raw} | {delta if delta is not None else '-'} | {ratio} |")
            out.append("")
    return "\n".join(out)


def shard(module):
    """CI shard of a module path: the package, or `nalgebra::<top module>` for the main crate."""
    parts = module.split("::")
    return "::".join(parts[:2]) if parts[0] == "nalgebra" and len(parts) > 1 else parts[0]


def split(report):
    """Return {shard: sub-report}."""
    shards = defaultdict(dict)
    for module, groups in report.items():
        shards[shard(module)][module] = groups
    return shards


def flatten(report, prefix=""):
    return {
        f"{module}::{group}__{variant}": gas
        for module, groups in report.items()
        if module.startswith(prefix)
        for group, variants in groups.items()
        for variant, gas in variants.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--update", metavar="DIR", help="write `<shard>.json` and `<shard>.md` snapshots into DIR")
    parser.add_argument("--check", metavar="DIR", help="compare against the snapshots in DIR; exit 1 on any difference")
    parser.add_argument("--filter", default="", help="with --check: only compare modules with this prefix")
    args = parser.parse_args()

    report = parse(sys.stdin)
    if not report and not args.check:
        sys.exit("no benchmark found in input")
    # With --check, an empty run is fine as long as the snapshot slice is empty too (a package with
    # no bench yet); a missing run against a non-empty slice fails as "present -> absent".

    if args.update:
        for name, sub in split(report).items():
            base = os.path.join(args.update, name.replace("::", "-"))
            with open(base + ".json", "w") as file:
                json.dump(sub, file, indent=2, sort_keys=True)
                file.write("\n")
            with open(base + ".md", "w") as file:
                file.write(to_markdown(sub) + "\n")
    if args.check:
        expected = {}
        for path in glob.glob(os.path.join(args.check, "*.json")):
            with open(path) as file:
                expected.update(flatten(json.load(file), args.filter))
        actual = flatten(report, args.filter)
        diffs = [
            f"{key}: {expected.get(key, 'absent')} -> {actual.get(key, 'absent')}"
            for key in sorted(set(expected) | set(actual))
            if expected.get(key) != actual.get(key)
        ]
        if diffs:
            sys.exit("gas snapshot mismatch (regenerate with --update if intended):\n  " + "\n  ".join(diffs))
    if not (args.update or args.check):
        print(to_markdown(report))


if __name__ == "__main__":
    main()
