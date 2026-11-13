"""Tests of gas_report.py: default stdout output and the --md option."""

import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "gas_report.py")
FIXTURE = os.path.join(HERE, "fixture_snforge.txt")

EXPECTED = """# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## scalar::abs::tests

### abs_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 17430 | 1590 | x1.00 |
| `felt_p` | 17750 | 1910 | x1.20 |

### noop

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `only` | 500 | - | x1.00 |
"""


def run(*args):
    with open(FIXTURE) as stdin:
        return subprocess.run(
            [sys.executable, SCRIPT, *args], stdin=stdin, capture_output=True, text=True, check=True
        ).stdout


class GasReportTest(unittest.TestCase):
    def test_default_stdout(self):
        self.assertEqual(run(), EXPECTED + "\n")

    def test_md_writes_table_and_keeps_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "GAS.md")
            self.assertEqual(run("--md", path), run())
            with open(path) as file:
                self.assertEqual(file.read(), EXPECTED + "\n")


if __name__ == "__main__":
    unittest.main()
