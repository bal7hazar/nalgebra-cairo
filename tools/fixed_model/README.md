# fixed_model

Bit-exact Python integer model of `simba::fixed` (Q32.32 on `i64`) and test-vector generator.

| file | role |
|---|---|
| `fixed_model.py` | Reference semantics of every kernel on raw integers: floor rounding once per output, `Overflow` / `DivisionByZero` / `SqrtNegative` where Cairo panics, floor-rounded constants (`constants()`, needs `mpmath`). `python3 fixed_model.py` self-checks the model against exact rational arithmetic and prints the constants. |
| `gen_vectors.py` | Emits `crates/simba/src/fixed/tests_generated.cairo`: 256 pseudo-random vectors per kernel (fixed seed, 64 rows per test) with expectations from the model, plus a test pinning the constants. |

```sh
python3 tools/fixed_model/fixed_model.py           # self-check + constants
python3 tools/fixed_model/gen_vectors.py           # regenerate (runs `scarb fmt -p simba`)
python3 tools/fixed_model/gen_vectors.py --check   # is the committed file up to date?
```

Changing the numeric behaviour of a kernel is a breaking change (AGENTS.md, numeric rules): change
the model first, regenerate, and the Cairo implementation must follow bit for bit.
