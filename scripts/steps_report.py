#!/usr/bin/env python3
"""Turn `snforge test --tracked-resource cairo-steps --detailed-resources` output into a steps report.

Convention (the sibling of `scripts/gas_report.py`): probe tests are named `probe_<group>__<variant>`.
Within a group, the `baseline` variant measures the fixed overhead (inputs + assertion) and is
subtracted from every other variant (`op`) to obtain the net Cairo steps of the operation. Cairo steps
are path- and platform-free: a snapshot generated on any machine is the one CI checks.

`snforge` prints the steps of a test on the line after its `[PASS]` line only with
`--detailed-resources`; the tests run in any order, the report is keyed by the module path.

Snapshots are one per package (`nalgebra_probes_steps`, ...): `steps/<package>.json` + `steps/<package>.md`.

Usage:
    snforge test -p nalgebra_probes_steps --tracked-resource cairo-steps --detailed-resources \\
        | python3 scripts/steps_report.py --update steps/
    snforge test -p nalgebra_probes_steps --tracked-resource cairo-steps --detailed-resources \\
        | python3 scripts/steps_report.py --check steps/
    snforge test ... | python3 scripts/steps_report.py            # print a report
    python3 scripts/steps_report.py --self-test                  # checks of this script's own logic
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

PASS = re.compile(r"^\[PASS\]\s+(\S+)\s+\(")
FAIL = re.compile(r"^\[(FAIL|IGNORE)\]\s+(\S+)")
STEPS = re.compile(r"^\s*steps:\s*(\d+)\s*$")
NAME = re.compile(r"^probe_(?P<group>.+?)__(?P<variant>.+)$")


def parse(lines):
    """Return {module path: {group: {variant: steps}}}; exit on a failed test, on a probe without a
    `steps:` line (run without `--detailed-resources`) and on a group without a baseline."""
    report = defaultdict(lambda: defaultdict(dict))
    pending = None
    for line in lines:
        fail = FAIL.match(line)
        if fail:
            sys.exit(f"{fail.group(2)}: {fail.group(1)}: a probe that does not pass has no steps")
        match = PASS.match(line)
        if match:
            if pending:
                sys.exit(f"{pending}: no `steps:` line (run snforge with --detailed-resources)")
            pending = match.group(1)
            continue
        steps = STEPS.match(line)
        if steps and pending:
            parts = pending.split("::")
            name = NAME.match(parts[-1])
            if name:
                report["::".join(parts[:-1])][name.group("group")][name.group("variant")] = int(steps.group(1))
            pending = None
    if pending:
        sys.exit(f"{pending}: no `steps:` line (run snforge with --detailed-resources)")
    for module, groups in report.items():
        for group, variants in groups.items():
            if "baseline" not in variants:
                sys.exit(f"{module}::probe_{group}: no `baseline` variant (net steps need one)")
    return report


def net(variants):
    """Return {variant: (raw, net)} with the group baseline subtracted."""
    base = variants["baseline"]
    return {variant: (steps, steps - base) for variant, steps in variants.items() if variant != "baseline"}


def to_markdown(report):
    out = [
        "# Steps report",
        "",
        "Cairo steps per probe (`snforge test --tracked-resource cairo-steps --detailed-resources`);"
        " `net` = raw - group baseline.",
        "",
    ]
    for module in sorted(report):
        out += [f"## {module}", "", "| probe | variant | baseline | raw | net steps |", "|---|---|---:|---:|---:|"]
        for group in sorted(report[module]):
            variants = report[module][group]
            for variant, (raw, delta) in sorted(net(variants).items()):
                out.append(f"| `{group}` | `{variant}` | {variants['baseline']} | {raw} | {delta} |")
        out.append("")
    return "\n".join(out)


def package(module):
    """Snapshot name of a module path: its package (the first segment)."""
    return module.split("::")[0]


def split(report):
    """Return {package: sub-report}."""
    packages = defaultdict(dict)
    for module, groups in report.items():
        packages[package(module)][module] = groups
    return packages


def flatten(report):
    return {
        f"{module}::{group}__{variant}": steps
        for module, groups in report.items()
        for group, variants in groups.items()
        for variant, steps in variants.items()
    }


def diff(expected, actual):
    """The lines of a snapshot mismatch (empty when equal)."""
    return [
        f"{key}: {expected.get(key, 'absent')} -> {actual.get(key, 'absent')}"
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    ]


def self_test():
    sample = """\
Running 6 test(s) from src/
[PASS] pkg::m::probe_dot__op (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 107
        memory holes: 0

[PASS] pkg::m::probe_dot__baseline (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 90

[PASS] pkg::n::probe_cross__baseline (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 100
[PASS] pkg::n::bench_other__baseline (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 5
[PASS] pkg::n::probe_cross__op (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 147
[PASS] other::k::probe_a__baseline (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 1
[PASS] other::k::probe_a__op (l1_gas: ~0, l1_data_gas: ~0, l2_gas: ~40000)
        steps: 1
Tests: 7 passed, 0 failed, 0 ignored, 0 filtered out
"""
    report = parse(sample.splitlines())
    assert dict(report["pkg::m"]["dot"]) == {"op": 107, "baseline": 90}, report
    assert dict(report["pkg::n"]["cross"]) == {"op": 147, "baseline": 100}
    assert "bench_other" not in str(dict(report)), "a test that is not a probe is ignored"
    assert net(report["pkg::m"]["dot"]) == {"op": (107, 17)}
    assert net(report["other::k"]["a"]) == {"op": (1, 0)}, "a free operation has 0 net steps"
    md = to_markdown(report)
    assert "| `dot` | `op` | 90 | 107 | 17 |" in md and "| `cross` | `op` | 100 | 147 | 47 |" in md, md
    assert sorted(split(report)) == ["other", "pkg"]
    flat = flatten(report)
    assert flat["pkg::m::dot__op"] == 107 and flat["pkg::m::dot__baseline"] == 90
    assert diff(flat, flat) == []
    changed = dict(flat, **{"pkg::m::dot__op": 108})
    removed = {k: v for k, v in flat.items() if k != "pkg::n::cross__op"}
    assert diff(flat, changed) == ["pkg::m::dot__op: 107 -> 108"], diff(flat, changed)
    assert diff(flat, removed) == ["pkg::n::cross__op: 147 -> absent"]
    assert diff(removed, flat) == ["pkg::n::cross__op: absent -> 147"]

    def refused(text, needle):
        try:
            parse(text.splitlines())
        except SystemExit as exit_:
            assert needle in str(exit_), (needle, exit_)
            return
        raise AssertionError(f"not refused: {needle}")

    refused("[PASS] p::probe_x__baseline (l2_gas: ~1)\n[PASS] p::probe_x__op (l2_gas: ~1)\n        steps: 3\n",
            "no `steps:` line")
    refused("[PASS] p::probe_x__baseline (l2_gas: ~1)\n", "no `steps:` line")
    refused("[PASS] p::probe_x__op (l2_gas: ~1)\n        steps: 3\n", "no `baseline` variant")
    refused("[FAIL] p::probe_x__op\n", "FAIL")
    print("self-test: ok")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--update", metavar="DIR", help="write `<package>.json` and `<package>.md` snapshots into DIR")
    parser.add_argument("--check", metavar="DIR", help="compare against the snapshots in DIR; exit 1 on any difference")
    parser.add_argument("--self-test", action="store_true", help="check the script's own logic and exit")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    report = parse(sys.stdin)
    if not report:
        sys.exit("no probe found in input (was snforge run with --detailed-resources?)")

    if args.update:
        os.makedirs(args.update, exist_ok=True)
        for name, sub in split(report).items():
            base = os.path.join(args.update, name)
            with open(base + ".json", "w") as file:
                json.dump(sub, file, indent=2, sort_keys=True)
                file.write("\n")
            with open(base + ".md", "w") as file:
                file.write(to_markdown(sub) + "\n")
    if args.check:
        expected = {}
        for path in glob.glob(os.path.join(args.check, "*.json")):
            with open(path) as file:
                expected.update(flatten(json.load(file)))
        diffs = diff(expected, flatten(report))
        if diffs:
            sys.exit("steps snapshot mismatch (regenerate with --update if intended):\n  " + "\n  ".join(diffs))
    if not (args.update or args.check):
        print(to_markdown(report))


if __name__ == "__main__":
    main()
