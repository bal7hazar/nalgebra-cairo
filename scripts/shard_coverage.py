#!/usr/bin/env python3
"""Check that the CI shards of each package run every test of that package exactly once.

Input (stdin): one line `<package> Tests: P passed, F failed, I ignored, X filtered out` per
`snforge test -p <package> <filter>` run, collected by the `library` jobs. For every package the
runs must agree on the package's test count (P + F + I + X), and the tests they selected
(P + F + I) must add up to it: fewer means tests that no filter selects (they would never run in
CI, e.g. a new module whose name matches no shard prefix), more means overlapping filters.

    cat bench/*/coverage.out | python3 scripts/shard_coverage.py
"""

import re
import sys
from collections import defaultdict

LINE = re.compile(r"^(\S+) Tests: (\d+) passed, (\d+) failed, (\d+) ignored, (\d+) filtered out")


def main():
    totals = defaultdict(set)
    selected = defaultdict(int)
    runs = defaultdict(int)
    for raw in sys.stdin:
        m = LINE.match(raw.strip())
        if not m:
            continue
        package, passed, failed, ignored, filtered = m.group(1), *map(int, m.groups()[1:])
        totals[package].add(passed + failed + ignored + filtered)
        selected[package] += passed + failed + ignored
        runs[package] += 1
    if not totals:
        sys.exit("no `Tests:` summary found")
    errors = []
    for package in sorted(totals):
        if len(totals[package]) != 1:
            errors.append(f"{package}: runs disagree on the test count {sorted(totals[package])}")
            continue
        total = totals[package].pop()
        if selected[package] < total:
            errors.append(f"{package}: {total - selected[package]} of {total} tests are selected by "
                          "no CI filter (they never run): add their module to a shard's filters")
        elif selected[package] > total:
            errors.append(f"{package}: {selected[package] - total} tests are selected by two "
                          "filters (overlapping shards)")
        print(f"{package}: {selected[package]} / {total} tests in {runs[package]} run(s)")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
