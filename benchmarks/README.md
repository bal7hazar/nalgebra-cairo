# Design-time benchmarks

Reproducible Sierra-gas micro-benchmarks that drove [the design](../docs/DESIGN.md). Synthesis in
[docs/BENCHMARK.md](../docs/BENCHMARK.md). Not part of the library workspace.

| Package | Question | Report |
|---|---|---|
| [`primitives`](primitives/README.md) | What does each Cairo building block cost? Math vs bitwise vs loops | [GAS](primitives/GAS.md) |
| [`scalar`](scalar/README.md) | Which fixed-point representation and which algorithms? | [GAS](scalar/GAS.md) |
| [`layout`](layout/README.md) | Struct vs array vs span, generic vs concrete, inlining, lazy rescaling | [GAS](layout/GAS.md) |
| [`libs`](libs/README.md) | How do alexandria, origami, cubit and orion perform? (standalone package) | [GAS](libs/GAS.md) |
| `harness` | `black_box` and conventions shared by the above | |

```bash
cd benchmarks
snforge test -p scalar | python3 scripts/gas_report.py            # print a report
snforge test --workspace | python3 scripts/gas_report.py --check gas/      # or --update gas/
cd libs && snforge test | python3 ../scripts/gas_report.py --check gas/
```

Conventions: tests are named `bench_<group>__<variant>`, are `#[inline(never)]`, take their inputs
through `harness::black_box`, assert their result, and each group has a `baseline` variant.
Re-run everything after a toolchain bump: rankings depend on the Sierra gas cost table.
