# polygen

Generator of the **typed polynomial kernels** of `simba::fixed` (DESIGN D6): range reductions,
Horner evaluations and final roundings, emitted as `core::internal::bounded_int` straight-line
Cairo whose bounds are proven here by interval arithmetic and re-checked by the Cairo compiler.

| file | role |
|---|---|
| `typed.py` | `Kernel` / `Module`: records exact integer operations with their exact result intervals, emits the Cairo, replays the same program on Python integers (`run`) and serialises it (`record`). |
| `polygen.py` | Scales, Chebyshev fits, Horner scheduling, the kernel set, the accuracy report and the three outputs. |

Outputs (committed, reproducible byte for byte, never edited by hand):

| output | content |
|---|---|
| `crates/simba/src/fixed/kernels/poly.cairo` | the shipped kernels |
| `crates/simba/src/fixed/kernels/poly_alternatives.cairo` | the degrees that lost the accuracy / gas comparison, `#[cfg(test)]` only (AGENTS.md rule 8) |
| `tools/fixed_model/poly_ops.py` | the same programs as data, so the bit-exact model of `tools/fixed_model` runs **the same** integer operations as the Cairo code and cannot drift from it |

```sh
python3 tools/polygen/polygen.py           # write the three files (runs `scarb fmt -p simba`)
python3 tools/polygen/polygen.py --check   # exit 1 if a committed file is stale
python3 tools/polygen/polygen.py --report  # fit error and end-to-end error of every candidate
```

The default mode and `--check` also replay every generated kernel through
`tools/fixed_model/fixed_model.py` against mpmath at 200 bits (`self_check_transcendental`), so a
divergence is a hard error. `--report` needs about ten seconds.

## Why interval types

The Sierra `bounded_int_*` libfuncs take the **exact** interval of each operation as its result
type. Computing those intervals with Python integers (`typed.Val`) and letting the compiler
re-check them removes every overflow check and every sign branch from the evaluation: a Horner
step is `mul, add const` and, once every three steps, one `div_rem`. Intermediates keep 64
fractional bits, so the only rounding of a kernel is its documented final one (to nearest, ties
up). A bound that is off by one is a compile error, not a wrong result.

## Design choices, and how to revisit them

- **Scale of reduced angles and ratios: 2^61** (`A`). `pi/4 * 2^61` is within 0.13 of an integer,
  so reducing the largest representable angle (`|x| < 2^31` rad) drifts by less than 2.9 raw
  Q32.32 units instead of the `|x| / 8` ulp a Q32.32 `TAU` would give. Every divisor stays below
  2^123, the limit of the cheapest `div_rem` algorithm.
- **Scale of the squared argument: 2^36** (`Z`), accumulators at 2^64 (`ACC`). A small variable
  lets one accumulator absorb three multiplications between two `div_rem`s (`MAX_SHIFT`).
- **Every fit is `1 + z * G(z)` with the leading 1 exact**, so `sin(0) = 0`, `cos(0) = 1`,
  `atan(0) = 0`, `asin(0) = 0` hold bit for bit and the functions are the identity for tiny
  arguments.
- **Degrees** are chosen on the measured trade-off, `--report` for the accuracy and the
  `bench_sin__*` / `bench_atan2__*` / `bench_asin__*` groups of `simba::fixed::transcendental`
  for the gas. Change `SIN_FIT` / `COS_FIT` / `ATAN_FIT` / `ASIN_FIT`, regenerate, re-run the
  gate: the losing degrees move to `poly_alternatives.cairo` automatically.

Changing any of this changes the last bit of a public function, which is a breaking change
(AGENTS.md, numeric rules): regenerate, update the expectations of `tests_generated.cairo`
(`tools/fixed_model/gen_vectors.py`) and the numeric spec in
`crates/simba/src/fixed/transcendental.cairo`.
