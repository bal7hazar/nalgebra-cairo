# API-1 — API parity close-out (WP 11-API-1)

Date: 2026-10-03. Status: report; nothing here edits `docs/PLAN.md` (the proposed rows of section 4 are
for the orchestrator). Inventory of record: `docs/API_PARITY.md`, nalgebra-rs 0.35.0 and simba-rs 0.10.2.

## 1. The last `missing` item

`nalgebra::io::cs_matrix_from_matrix_market` was the only `missing` item (module `io`: 1 ported, 1
missing). It takes a `Path` and reads a file. A Cairo program has no file system and no path type
(`AGENTS.md`, `docs/DESIGN.md`: no I/O the Cairo VM cannot have), so it cannot be ported as written.
Its content-taking sibling `cs_matrix_from_matrix_market_str` is already ported, with upstream's
grammar (`crates/nalgebra/src/io.cairo` says so in its module documentation).

Decision (WP 11-API-1): a documented exclusion, not a port and not a rename. The closed reason list of
`scripts/api_parity.py` had no reason for it, so one is added, `fs` ("File-system access"), with one
`EXCLUDE` rule that names this single function. A rename to the `_str` form would have mapped two
upstream functions onto one Cairo function and hidden that the path form does not exist. Reverse it by
deleting the rule and the reason: the item is `missing` again and the work package P20 lists it again.

Effect on `docs/API_PARITY.md` (regenerated, `--check` passes): `io` 100.0 %, total 1885 ported, 0
partial, 0 missing, 551 excluded of 2436 items, coverage 100.0 %; P20 has 0 items; the reason table gains
`fs` (1 item).

## 2. nalgebra-rs: latest release against the 0.35.0 inventory

Method. The latest release was read from the crates.io index (`max_version`, `max_stable_version`: both
0.35.0; published 2026-05-24; the later-dated 0.34.2 and 0.33.3 of 2026-03-28 are backport releases of
older lines). dimforge/nalgebra was cloned read-only into a temporary directory outside the repository
(removed afterwards), with a second worktree at tag `v0.35.0`. Two dry runs of
`python3 scripts/api_parity.py --refresh --nalgebra-rs <checkout> --simba <checkout>` (the script
rewrites the generated document, so each run was followed by `git diff` and reverted; no refreshed
inventory is committed):

| Run | Sources | Result against the committed inventory |
|---|---|---|
| tag `v0.35.0` + simba 0.10.2 (cargo registry) | the released sources | identical: the committed inventory reproduces from the tag |
| upstream `main` (`b2466e6c`, 2026-09-30) + simba `main` (`16a77ff`, 2026-08-07) | unreleased | identical: no item added, removed or renamed |

Finding: **there is no newer release than 0.35.0, so no API delta exists to port.** Unreleased `main`
is 3 commits ahead of the tag and none changes a public signature (a `git diff v0.35.0..main` over `src/`
touches `linalg/cholesky.rs` and `linalg/schur.rs` only, with no changed `pub` item; the rest is tests):

| Commit (upstream, 2026-06-18 to 2026-09-30) | Effect | Cairo relevance |
|---|---|---|
| `Cholesky::new` returns `None` for non-positive-definite complex matrices | takes the real part of each pivot and requires it positive | none: complex scalars are out of scope (`docs/PLAN.md`); for real scalars the check is the previous one |
| exceptional shift in the real Schur iteration | a Wilkinson-Reinsch exceptional shift every 10 stalled QR steps, a stall counter reset at each deflation | behavioural: it changes which matrices converge, not the API; see row R3 below |
| test flake: epsilon of `f64::symmetric_eigen` test | tests only | none |

The `--refresh` mode has no dry or report mode of its own; the revert after each run is the dry run. The
note belongs in the script's usage if the mode is used again (not changed in this lot: allowlist).

## 3. simba-cairo against simba-rs `RealField` / `ComplexField`

Sources: simba-rs 0.10.2 `src/scalar/{field,complex,real}.rs` (the three traits `parse_simba` reads:
`Field` declares no method, `ComplexField` 55 methods, `RealField` 24); simba-cairo 0.2.0 (`Real`,
`Transcendental`); `fixed` 0.4.0, the one scalar behind it (Q32.32, fixed-point, panics on overflow,
no NaN, no infinity). Status: **provided** (same name and meaning), **fused** (provided under a Cairo
name by a fused kernel), **lacks** (a gap with a Cairo answer), **excluded by design**.

### 3.1 `RealField` (24)

| simba-rs method | simba-cairo | Status | Note |
|---|---|---|---|
| `is_sign_positive`, `is_sign_negative` | `Real` | provided | |
| `max`, `min`, `clamp` | `Real` | provided | |
| `atan2` | `Transcendental::atan2(y, x)` | provided | free-standing form `atan2(y, x)` |
| `min_value`, `max_value` | `Real` | provided | `Option<T>` as upstream |
| `pi`, `two_pi`, `frac_pi_2`, `frac_pi_3`, `frac_pi_4`, `frac_pi_6`, `frac_1_pi`, `e`, `ln_2`, `ln_10` | `Real` | provided | |
| `copysign` | none (`fixed` 0.4.0 has `copysign`) | lacks | one forwarding method |
| `frac_pi_8`, `frac_2_pi` | none (`fixed` 0.4.0 has both constants) | lacks | two forwarding constants |
| `frac_2_sqrt_pi`, `log2_e`, `log10_e` | none (not in `fixed`) | lacks | three constants rounded to Q32.32, derivable from the reference values |

### 3.2 `ComplexField` (55), real-scalar methods (42)

| simba-rs method | simba-cairo | Status | Note |
|---|---|---|---|
| `abs`, `signum`, `floor`, `mul_add`, `recip`, `sqrt` | `Real` | provided | `abs` returns `Self` on a real scalar |
| `sin`, `cos`, `sin_cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `sinhc`, `coshc` | `Transcendental` | provided | `sinhc` / `coshc` since 0.2.0 |
| `ceil`, `round`, `trunc`, `fract` | none (`fixed` has all four) | lacks | four forwarding methods; `floor` is the only rounding in `Real` |
| `powi` | none (`fixed` has `powi`) | lacks | forwarding |
| `exp2`, `exp_m1`, `log2`, `log10`, `ln_1p`, `log(base)`, `powf` | none (`fixed::exp` has all seven) | lacks | forwarding; `fixed::exp::ExpTrait` is already imported by `scalar.cairo` |
| `sinh_cosh` | none | lacks | default method upstream (`(sinh, cosh)`); the fused form shares one exponential |
| `sinc`, `cosc` | none | lacks | default methods upstream (`sin(x) / x` and its derivative form); Q32.32 needs the `x = 0` branch spelled out |
| `asinh`, `acosh`, `atanh` | none (not in `fixed`) | lacks | need closed forms over `ln` / `sqrt`; the domain panics must follow `fixed`'s |
| `cbrt` | none (not in `fixed`) | lacks | needs a root kernel; no use found in the nalgebra-rs 0.35.0 library sources |
| `hypot` | `Real::norm2(x, y)` | fused | named `norm2`, floor of the exact root; documented exception of `docs/API_PARITY.md` |
| `try_sqrt` | none | lacks | `Option<Self>` (`None` for a negative input) instead of the `'Fixed: sqrt negative'` panic; one comparison over `sqrt` |
| `is_finite` | none | excluded by design | every `Fixed` is finite (no NaN, no infinity; overflow panics): the answer is constant `true` |

### 3.3 `ComplexField` (55), complex methods (13)

`from_real`, `real`, `imaginary`, `modulus`, `modulus_squared`, `argument`, `norm1`, `scale`, `unscale`,
`to_polar`, `to_exp`, `conjugate`, `powc`: **excluded by design**, with the complex scalar (`num_complex`'s
`Complex` is the `interop` exclusion, and complex numbers are listed under `docs/PLAN.md` "Out of
scope"): for a real scalar they are identities (`real` = self, `imaginary` = 0, `conjugate` = self,
`modulus` = `abs`, `norm1` = `abs`, `scale` = `*`, `unscale` = `/`), so no generic code needs them as
long as the library is real-only. The nalgebra-rs sources use them mostly in the complex branches
of the decompositions.

### 3.4 Cairo-only items (the documented exception)

The fused kernels, `from_int`, `from_ratio`, `abs_diff_eq`, `rem`, `sqr`, `div*`, `wide_*`, `lerp`,
`norm*`, `sum_prod*`, `diff_prod` and four constants, 35 items, are the owner ruling of 2026-09-24
(`docs/API_PARITY.md`, "Scalar layer"); they are not gaps. Items upstream gets from supertraits
(`Zero`, `One`, `AbsDiffEq`, `RelativeEq`, `UlpsEq`, `Signed`, `FromPrimitive`, `SubsetOf`, `SimdValue`)
are `zero`, `one`, `default_epsilon` and `abs_diff_eq` here; the conversions (`SubsetOf`,
`SupersetOf<f32/f64>`, `FromPrimitive`) are `from_int` / `from_ratio`, because Cairo has no `f64`.

### 3.5 Count

| | Methods | Provided | Fused | Lacks | Excluded by design |
|---|---:|---:|---:|---:|---:|
| `RealField` | 24 | 18 | 0 | 6 | 0 |
| `ComplexField`, real scalar | 42 | 20 | 1 | 20 | 1 |
| `ComplexField`, complex | 13 | 0 | 0 | 0 | 13 |
| **Total** | **79** | **38** | **1** | **26** | **14** |

(The traits share no method name, so nothing is counted twice.)

None of the 26 gaps blocks the 100 % parity of nalgebra-cairo: the library is at 100 % with `Real` and
`Transcendental` as they are, because every item it ports compiles against them. The gaps are the
distance between simba-cairo and simba-rs's scalar API, a separate repository's surface.

## 4. Proposed rows for the plan

For the orchestrator to place; none is entered in `docs/PLAN.md`. Size: S = one small PR (a trait method
or constant with its tests and gas snapshot), M = one PR with a design choice or a new kernel. Crate:
`simba-cairo` (repository `simba-cairo`, crate `simba`) unless noted.

| Row | Title | Size | Crate | Content |
|---|---|---|---|---|
| R1 | simba `Real` / `Transcendental`: forward what `fixed` 0.4.0 already has | S | `simba` (simba-cairo) | `ceil`, `round`, `trunc`, `fract`, `copysign`, `powi`, `exp2`, `exp_m1`, `log2`, `log10`, `ln_1p`, `log`, `powf`, constants `frac_pi_8`, `frac_2_pi`; each a one-line `#[inline(always)]` call, a test against `fixed`'s own result, gas snapshot rows. A 0.3.0 MINOR of simba (new trait methods break implementors; the only implementor is `Fixed`) |
| R2 | simba: methods needing a small kernel | M | `simba` | `asinh`, `acosh`, `atanh`, `sinh_cosh`, `sinc`, `cosc`, `cbrt`, `try_sqrt`, constants `frac_2_sqrt_pi`, `log2_e`, `log10_e`; closed forms with the Q32.32 rounding rule documented (rounded once), oracle vectors from mpmath, domain panics as `fixed`'s. `cbrt` may be split off or dropped if no consumer appears (no use in the nalgebra-rs 0.35.0 library sources) |
| R3 | Follow upstream's Schur exceptional shift | S | `linalg_spectral3`..`6` (nalgebra-cairo) | Unreleased upstream change (section 2): only when upstream releases it (0.35.1 or 0.36); then an owner decision whether the Cairo Schur, whose loops are bounded by the shape, takes it, with a stalling-matrix test. Not before the release: no row to start now |
| R4 | Track upstream releases | S, recurring | `scripts/api_parity.py` | When a release newer than 0.35.0 exists: bump `VERSION` / `SIMBA_VERSION`, run `--refresh` on the new tag, review the delta as in section 2. Add a `--report` (diff against the committed inventory, no write) to the script so the dry run is a mode instead of a revert |

No row for `is_finite` and the complex methods (excluded by design); no row for the `Signed`, `RelativeEq`
and `UlpsEq` supertraits (covered by `abs_diff_eq` and `default_epsilon`).

## 5. Commands run

```
python3 scripts/api_parity.py && python3 scripts/api_parity.py --check
python3 scripts/api_parity.py --refresh --nalgebra-rs <v0.35.0 worktree> --simba <cargo registry simba-0.10.2>   # then git diff, empty
python3 scripts/api_parity.py --refresh --nalgebra-rs <upstream main> --simba <upstream main>                   # then git diff, empty; git checkout docs/API_PARITY.md
```

Upstream clones were in a temporary directory outside the repository and were removed by the thread that
created them.
