# tools/oracle

Offline generator of **golden test vectors** for nalgebra-cairo (DESIGN D7, ROADMAP WP 1.3).
The oracle is the upstream Rust [`nalgebra`](https://crates.io/crates/nalgebra) crate (pinned to
`0.35.0`): inputs are drawn as raw Q32.32 values, evaluated upstream, and the results are floored
back to Q32.32 together with a suggested tolerance in raw units (ulp).

It is a standalone cargo crate (never a dependency of the Cairo workspace). Its outputs are
committed: `vectors/<suite>.json`. Cairo test data is derived from them with `emit-cairo`.

## Usage

Run from `tools/oracle`:

```
cargo run --release -- list                          # suites, ops, case counts, max tolerance
cargo run --release -- all                           # regenerate every vectors/<suite>.json
cargo run --release -- gen vector3 quaternion        # regenerate some suites
cargo run --release -- check                         # fail if vectors/ is stale (also a unit test)
cargo run --release -- emit-cairo vector3 --out /path/to/oracle_vector3.cairo
```

Options:

| Flag | Default | Meaning |
|---|---|---|
| `--cases N` | 32 | random cases per op (split evenly between the op's distributions; ops with more than 24 / 64 input scalars get N/2 / N/4) |
| `--seed S` | 1 | seed of the ChaCha8 streams |
| `--out PATH` | `vectors` | output directory (`all`, `gen`, `check`) or output file (`emit-cairo`) |
| `--from DIR` | — | `emit-cairo`: read `DIR/<suite>.json` instead of regenerating (use `--from vectors` to emit exactly what is committed) |
| `--dist a,b` | all | `emit-cairo`: keep only these distributions (`small`, `unit`, `medium`, `large`, `special`) |
| `--ops a,b` | all | `emit-cairo`: keep only these ops |
| `--max-per-dist K` | all | `emit-cairo`: keep at most K cases per distribution of each op |

The committed vectors use the defaults (`--seed 1 --cases 32`); `cargo test` fails if they are
stale. Quality gate: `cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test`.

## Number model

- A scalar is a raw `i64`, `value = raw / 2^32` (Q32.32, DESIGN D2).
- **Inputs are generated as raw integers** (`|raw| <= 2^53`, so `raw as f64 / 2^32` is exact): the
  Rust oracle and the Cairo code see bit-identical inputs.
- Expected outputs are `floor(result * 2^32)` (rounding = floor, like the Cairo scalar). A case
  whose outputs do not fit Q32.32 is resampled: **every expected output fits**; intermediates of
  a naive implementation may not (e.g. the scaled sum of squares of a `large` vector: DESIGN D6
  computes norms on the unscaled sum).
- Two evaluation paths, both upstream nalgebra:
  - **f64** for everything;
  - **exact integers** for homogeneous polynomial ops (dot, cross, perp, scale, component_mul,
    norm_squared, matrix products, outer, 2x2 / 3x3 determinant, complex and Hamilton products):
    nalgebra is instantiated on `i128` raws (or a hand-written kernel cross-checked against
    upstream in f64), the wide result is shifted right (= floor). The expected value is then the
    exact floor of the true result, and the f64 path only serves as a cross-check (asserted).
- Unit vectors, unit complex numbers, unit quaternions and rotation matrices are normalised in f64
  **then quantised component-wise**: their norm is 1 within about 2 ulp, not exactly 1. The oracle
  takes them as unit without renormalising (`Unit::new_unchecked`), which is what the Cairo types do.

### Input distributions

Every case carries a `dist` label; cases of an op are ordered by distribution (index ranges are
printed in the Cairo doc comment).

| Label | Scalars / components | Structured matrices (scale) |
|---|---|---|
| `special` | hand-picked (0, 1, pi/2, axes...), explicit tolerance, emitted first | — |
| `small` | uniform in (-1, 1) | 0.05 .. 0.5 |
| `unit` | magnitude uniform in [0.5, 2), random sign | 1 |
| `medium` | magnitude log-uniform in [2, 1e3), random sign | log-uniform 2 .. 100 |
| `large` | magnitude log-uniform in [1e3, 4e4), random sign (squares fit) | not used |

Rotations (unit complex, unit quaternion, rotation matrices, axes) are always labelled `unit`;
angles are uniform in (-pi, pi), interpolation parameters in [0, 1], similarity scalings in
[0.25, 4], scaled axes have an angle in [0.05, 3].

Structured matrices (row-major, `Q1 * diag(sigma) * Q2^T` with Haar-ish orthogonal factors):

- **well-conditioned** (`try_inverse`, LU, QR, SVD): singular values log-uniform in [0.25, 2] x scale
  (condition number <= 8);
- **SPD** (Cholesky, UDU / LDLT, eigen): eigenvalues log-uniform in [0.25, 2] x scale, exactly
  symmetric in raw units;
- **symmetric** (eigen): independent entries of the distribution, mirrored;
- **exactly singular** (`*_try_inverse_singular`): small integer entries, one row is an integer
  combination of the others, rows shuffled. Every fixed-point product is exact, so a fixed-point
  determinant is exactly 0 and `try_inverse` must return `None` (expected `is_some = false`);
- **near-singular**, FLAGGED in separate ops (`*_near_singular`): one singular value log-uniform in
  [1e-4, 1e-2], the others in [0.5, 2] (condition number 1e2..1e4). Tolerances there are large by
  construction; these ops document behaviour under ill-conditioning, they are not precision tests.

Degenerate configurations are rejected and resampled: nearly parallel vectors for `angle` /
`rotation_between` / `face_towards`, slerp with `|dot|` outside (0.05, 0.95), gimbal lock
(`|pitch| >= 1.45`), branch cuts of angles, tolerances above 2^22 ulp (1e-3).

## Conventions

- **Matrices are emitted ROW-major**: `[[m11, m12, ..], [m21, m22, ..], ..]`, i.e. the argument
  order of upstream `Matrix3::new(m11, m12, m13, m21, ...)`. Beware that upstream *storage*,
  `as_slice()` and serde are COLUMN-major: never feed these arrays to a column-major constructor.
  6x6 matrices are plain row-major arrays (the Cairo 2x2-blocks-of-`Matrix3` layout is an
  implementation detail).
- **Quaternions are `(w, i, j, k)`**, the argument order of `Quaternion::new` (upstream storage is
  `[i, j, k, w]`, glam is `xyzw`).
- **Unit complex numbers are `(re, im)`** = `(cos, sin)`.
- Vectors and points are `(x, y[, z[, w]])`; isometries `(translation, rotation)`; similarities
  `(translation, rotation, scaling)`; euler angles `(roll, pitch, yaw)` as upstream
  (`R = Rz(yaw) * Ry(pitch) * Rx(roll)`).
- **Angles are in radians**, as raw Q32.32.
- Eigenvalues are sorted **ascending**, singular values **descending**. Eigenvectors, `U`, `V` are
  not emitted (sign / order ambiguous): check them by reconstruction.
- `qr{n}_q_r` follows upstream's Householder signs (`diag(r)` may be negative). For another
  convention, flip the sign of row `i` of `r` and column `i` of `q`, or check by reconstruction.
- `udu{n}_u_d` is upstream's `UDU` (`a = u * diag(d) * u^T`, `u` unit **upper**). DESIGN D6 calls
  for LDLT (`l` unit **lower**): `ldlt{n}_l_d` provides those factors (derived from upstream's
  Cholesky). `udu{n}_solve` / `udu{n}_inverse` are valid for both.
- `unit_quaternion_from_rotation_matrix` has upstream's sign; compare up to sign if the branch
  selection differs.
- `unit_quaternion_slerp` takes the shortest arc (upstream negates `b` when `dot < 0`);
  `unit_quaternion_nlerp` does not; `unit_complex_slerp` is `a * new((b / a).angle() * t)`.
- `unit_quaternion_mean_of` (suite `unit_quaternion_completion`) is upstream's `mean_of`
  UN-PERMUTED: upstream builds `Quaternion::new(v[0], v[1], v[2], v[3])` from an eigenvector stored
  as `(i, j, k, w)`, so its `(w, i, j, k)` is the true `(i, j, k, w)`. The sign makes the largest
  component positive; cases whose second eigenvalue exceeds 0.8 of the first are rejected.
- `unit_quaternion_from_matrix` / `unit_complex_from_matrix` run upstream's iteration bounded to
  1000 steps and keep a case only when it agrees with the SVD (3D) or closed-form (2D) maximiser of
  `tr(R^T m)`; the 3D result has the sign of upstream's final `from_rotation_matrix`.
- `quaternion_half` is upstream's `q / 2` (nearest) floored by the oracle: 1 ulp.
- Suite `pose_completion` (WP 8.4-P09b): the rotation-matrix poses `IsometryMatrix2/3` /
  `SimilarityMatrix2/3` are emitted as SEPARATE fields — translation, ROW-major rotation matrix
  (then scaling) — not as one group. The look-at ops (`isometry_matrix3_look_at_rh` / `_lh`,
  `isometry3_look_at_lh`) use the `SensMag` policy with `mag = 32`: the translation
  `rotation · (-eye)` multiplies the few-ulp error of the normalised frame by `|eye|`.
  `quaternion_exp_real` / `_sinh_real` / `_cosh_real` pin upstream's identity / zero / identity
  for a real quaternion (tolerance 0), `rotation2_renormalize` is upstream's
  `from_matrix_eps(m, eps, 0, guess)` on a rotation plus a drift of at most 2^-10 per entry.
- Suite `cg` (WP 8.3-P07): upstream `base/cg.rs` on `Matrix3` / `Matrix4` (rotation constructors,
  `face_towards`, `look_at_rh` / `_lh` with the `SensMag` look-at policy, `transform_point` /
  `transform_vector`). The transforms keep the normaliser away from zero (`|n| >= 1/4`).
  `transform_vector` divides BEFORE the product (upstream's `m * (v / n)`): its `SensMag` policy
  (`mag = 2`) covers the half-ulp quotient rounding multiplied by the entries of `m`.
- Suite `dual_quaternion` (WP 8.4-P12): dual quaternions are `(real (w, i, j, k), dual (w, i, j,
  k))`. A unit dual quaternion input (`Gen::UnitDual`) is a quantised unit quaternion `r` and a
  translation `t` of the case's magnitude class, then upstream's `from_parts(t, r)` in f64 with its
  dual part quantised: both sides see the same raw eight components. The dual-quaternion product
  and the products by a unit quaternion are exact-integer ops (tolerance 0). The chained products
  (inverse, divisions, point transforms, the products with isometries) use a constant 8 ulp: the
  sensitivity policy would count the derivatives with respect to the rotation (of the size of the
  translation) as rounding errors, which these kernels never commit on their exact inputs.
  `nlerp` and `sclerp` use `sensitivity + magnitude` with `k = 0` (`mag` 4 and 8): a rounded
  unit-scale factor (the norm, the screw) multiplies the dual part. `unit_dual_quaternion_sclerp`
  is upstream's `try_sclerp` with `epsilon = 2^-32` (the Cairo `default_epsilon`).
- Suite `transform` (WP 8.4-P11a): an `Affine2/3` input is `(linear, translation)`, the ROW-major
  linear block then the translation, the last row `(0, .., 0, 1)` implicit; `Projective2/3`
  inputs are whole ROW-major matrices. The inverted linear blocks are well-conditioned. The ops
  that apply a rounded inverse to large data (`inverse_transform_*`, `affine3_div_projective3`)
  use `sensitivity + magnitude` (`mag = 4`): the one-ulp roundings of the inverse's entries are
  multiplied by the coordinates of the point / the entries of the dividend.
- The scalar suite and every transcendental inside nalgebra go through the pure-Rust `libm`
  (`libm-force`), not the platform libm.

## Tolerance policy

`tol` is a **suggested** bound on `|got - expected|` in raw units, applied to every output scalar of
the case. It answers "how far can a sane fixed-point implementation (floor, one rescale per output
scalar, fused kernels) be from the floored real result", not "how accurate is f64". The policy of
each op is written in its JSON `tolerance` field and in the Cairo doc comment.

| Policy | Value | Used for |
|---|---|---|
| exact | 0 | results that involve no rounding (add, sub, neg, transpose, trace, min / max, conjugate, translations, `cross_matrix`...) and exact-integer ops (dot, cross, products...): a single rescale per output scalar reproduces the floor **bit for bit**; an implementation that rescales each product separately is off by up to `terms - 1` ulp and violates AGENTS.md rule 4 |
| constant | `n` ulp | error independent of the inputs: `sqrt` / `norm` (1-2), lerp (2), Hamilton product (1: exact expectation, two fused kernels allowed), trig of DESIGN D6 accuracy (sin / cos 4, atan2 12, asin / acos 8, ln 8), isometry2 (2) |
| sensitivity | `ceil(base + k * A)` | everything with intermediate roundings. `A = max_out sum_in \|d out / d in\|` is measured by central differences (step 2^-20): `k` one-ulp errors committed on intermediates propagate to first order like input perturbations. `base` covers the final floor and the transcendental kernels (8 - 32 ulp). `k` = 2 - 3 for normalize / rotations / isometries, 4 - 8 for trig-based ops, `2n` for `n x n` factorizations |
| sensitivity + magnitude | `ceil(base + k * A + mag * max\|input\|)` | eigenvalues, singular values, QR factors: rounded unit-scale rotations / reflectors are applied to the input matrix |
| model | op-specific | `det3`: `6 * amax + 6`; `det4`: `24 * amax^2 + 24 * amax + 8` (+ f64 term); `sin` / `cos`: `4 + \|x\| / 8` (range reduction by a Q32.32 `2*pi` constant); `tan`: `2 * (4 + \|x\| / 8) * (1 + tan^2)`; `exp`: `4 + 4 * exp(x)`; `inv_sqrt`: `2 + 1 / x` |

The f64 oracle's own error is handled separately:

- below 2^45 raw, an expected value whose floor could change with the last bits of the f64 result
  (fractional part within `max(2^-10, |s| * 2^-48)` of an integer) is **resampled**, so the floor is
  the floor of the real result;
- above 2^45 raw, `ceil(|expected| * 2^-49) + 1` ulp is added to the tolerance;
- ops that cancel large products (dot, cross, determinants, Hamilton product) use the exact path.

Tolerances are a starting point: a Cairo test may tighten one (and should, when the implementation
is provably exact) or loosen one with a justification in the test. A failure by a few ulp above
`tol` deserves an error analysis, not a silent bump.

## Output formats

### JSON (`vectors/<suite>.json`, committed)

```json
{
"suite": "vector3", "description": "...",
"format": { "scalar": "Q32.32 raw i64, value = raw / 2^32", "matrix_order": "row-major: ...", ... },
"generator": { "name": "nalgebra-oracle", "version": "0.1.0", "nalgebra": "0.35.0", "seed": 1, "cases": 32 },
"ops": [
  { "name": "vector3_dot", "doc": "a.dot(&b)", "tolerance": "...",
    "inputs":  [{ "name": "a", "kind": { "vector": 3 }, "layout": "(x, y, z)" }, ...],
    "outputs": [{ "name": "dot", "kind": "scalar", "layout": "raw" }],
    "cases": [ { "dist": "small", "in": [[..], [..]], "out": [..], "tol": 0 }, ... ] }
]}
```

`kind` is `"scalar"`, `"bool"`, `{"vector": n}`, `{"matrix": [rows, cols]}` (array of rows) or
`{"group": [kinds]}`. One case per line (line-oriented diffs). Raw values exceed 2^53: parse them as
64-bit integers (Python does; JavaScript `JSON.parse` does not).

### Cairo (`emit-cairo`)

One module per suite, one function per op, **raw integers only** (no dependency on nalgebra-cairo
types, so the data compiles before the types exist and survives their refactoring):

```cairo
/// `vector3_dot`: a.dot(&b)
/// Layout: (a: (x, y, z), b: (x, y, z), dot: raw, tol: ulp)
/// Tolerance: ...
/// Cases: [0..8) small, [8..16) unit, [16..24) medium, [24..32) large.
#[cairofmt::skip]
pub fn vector3_dot_cases() -> Span<((i64, i64, i64), (i64, i64, i64), i64, u64)> {
    const CASES: [((i64, i64, i64), (i64, i64, i64), i64, u64); 32] = [ ... ];
    CASES.span()
}
```

- tuple = inputs in order, then outputs in order, then `tol: u64`;
- scalar `i64`; flag `bool`; vector / point / quaternion / complex: tuple of `i64`; matrix
  `[[i64; cols]; rows]` (row-major, destructure with `let [[m11, m12], [m21, m22]] = m;`);
  isometry `((x, y, z), (w, i, j, k))`; similarity `((x, y, z), (w, i, j, k), scaling)`;
- the data is a `const` array (no runtime array building), `#[cairofmt::skip]` keeps one case per
  line, and the file passes `scarb fmt --check` (scarb 2.19.4) as emitted;
- typical use:

```cairo
let mut cases = oracle_vector3::vector3_dot_cases();
while let Some(case) = cases.pop_front() {
    let ((ax, ay, az), (bx, by, bz), expected, tol) = *case;
    let got = Vector3 { x: Fixed { raw: ax }, .. }.dot(..);
    assert_abs_diff_le(got.raw, expected, tol);
}
```

The emitted modules of all suites were compiled and iterated with scarb 2.19.4 / `cairo-test` in a
throw-away package (21 modules, 1.8 MB of source, 3 s build), including a bit-exact check of
`vector3_dot` / `vector3_cross` against an `i128` reference written in Cairo.

## Determinism

- ChaCha8 (`rand_chacha`), one stream per op keyed by `(seed, fnv1a(op name))`: adding, removing or
  reordering ops never changes the vectors of the others. Only `next_u64` is consumed; every
  distribution is implemented here.
- No timestamp, host name or path in any output; stable op and case order; hand-written JSON layout.
- No platform dependence: nalgebra is built without `std` (no `matrixmultiply` runtime CPU
  dispatch) and with `libm-force`; the generator itself only uses `libm` and IEEE-exact operations.
  Debug and release builds produce identical files (checked by `cargo test`, which compares the
  committed release output with a debug regeneration).
- `Cargo.lock` is committed; nalgebra is pinned with `=0.35.0` and a test checks that
  `NALGEBRA_VERSION` matches the lockfile. Bumping nalgebra is a dedicated change that regenerates
  the vectors and reviews the diff.

## Layout

```
src/fixed.rs        Q32.32 quantisation, ambiguity check, exact rescale
src/gen.rs          RNG streams, distributions, structured matrix generators
src/engine.rs       Op builder, case generation, tolerance derivation, field constructors
src/model.rs        serde data model (Kind, Value, Case, OpData, SuiteData), Cairo type / literal
src/emit.rs         JSON writer / reader, Cairo emitter
src/suites/*.rs     op definitions: scalar, base (vectors, matrices, points, 6D), geometry, linalg
vectors/*.json      committed golden vectors (seed 1, 32 cases)
```

Adding an op: append an `Op::new(name, upstream_expression)` to the relevant suite with its inputs
(`iv`, `iuq`, `with(field, Gen::..)`), outputs, distributions, tolerance policy and an `eval` closure
calling upstream on flat row-major `f64` slices (plus `ring!(kernel)` for an exact bilinear kernel),
then run `cargo run --release -- all` and commit the JSON.
