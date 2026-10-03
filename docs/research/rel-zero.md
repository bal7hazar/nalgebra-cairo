# `is_sign_positive` at an exact zero: nalgebra-cairo with simba 0.3.0 (WP 12-REL-a)

simba 0.3.0 (published 2026-10-03) changes one result that nalgebra-cairo uses:
`Real::is_sign_positive(0)` is now `true` (`self >= 0`, forwarding to fixed 0.5.0's
`is_sign_positive`), where simba 0.2.0 gave `false` (`self > 0`, fixed's `is_positive`). simba-rs
gives `true` for `+0.0`, and Q32.32 has no negative zero. This note takes the six
`R::is_sign_positive` call sites of the workspace one by one: what nalgebra-rs 0.35.0 does at the
same input, what nalgebra-cairo returns with simba 0.2.0 (today) and with simba 0.3.0, and the test
lot REL-b should add. This lot changes no code at those sites.

## Result

| Site | Can it see an exact 0? | nalgebra-rs 0.35.0 | simba 0.2.0 (today) | simba 0.3.0, code unchanged | For REL-b |
|---|---|---|---|---|---|
| `crates/geometry3/src/geometry/unit_quaternion.cairo:418`, `from_rotation_matrix`, trace branch | **yes**: every rotation of 120° (trace `1 + 2cos θ = 0`), e.g. the cyclic permutations | `tr > 0` (`simd_gt`): at `tr = 0` it takes a diagonal branch | `tr > 0`: same branch and same bits as upstream | `tr >= 0`: the trace branch, **the opposite sign** of upstream's (same rotation) | **change the test to `tr > R::zero()`** (upstream's own comparison) and pin the inverse cyclic permutation (below) |
| `crates/geometry3/src/internal/geometry/unit_quaternion.cairo:203`, `shepperd_sign`, `w` | no: the branch needs `3w² - \|v\|² > 0`, so `w != 0` | n/a (internal) | unchanged | unchanged | nothing |
| same file `:205`, `i` | no: the branch needs `\|i\| > \|j\|` and `\|i\| > \|k\|`, so `i != 0` | n/a | unchanged | unchanged | nothing |
| same file `:207`, `j` | no: the branch needs `\|j\| > \|k\|`, so `j != 0` | n/a | unchanged | unchanged | nothing |
| same file `:209`, `k` | only for `i = j = k = 0` with `3w²` rounding to 0 (`\|w\|` below about `2^-16.8`): not a unit quaternion, and no caller passes one | upstream would see the identity rotation: `w > 0` | `w` negated (`-2^-32` for `w = 2^-32`) | `w` kept (`+2^-32`): upstream's sign | optional: a module test pinning the new value; no public result changes |
| `crates/geometry2/src/geometry/unit_complex.cairo:755`, `axis_angle` | no: `ang == 0` returns `None` one line earlier | the same structure (`is_zero`, then `is_sign_positive`) | `None` at 0 | `None` at 0 | nothing (already pinned: `crates/tests_geometry/src/unit_complex/tests_ext.cairo:217`) |

So with the code unchanged, simba 0.3.0 moves one public result, and it moves it **away from**
nalgebra-rs, not closer: `UnitQuaternion::from_rotation_matrix` (and every function that goes
through it, below) at a rotation matrix whose computed trace is exactly 0. With the one-token
change at line 418, simba 0.3.0 changes no result of these six sites for any input a caller can
pass; the only change left is site 209's unreachable input, where the new sign is upstream's.

## Site 418 in detail

The function's own doc says: "The sign is upstream's (the selected branch fixes it; `w > 0` only in
the trace branch) ... the oracle vectors are matched in sign too." Upstream selects the trace branch
with `tr.simd_gt(T::zero())` (`nalgebra-0.35.0/src/geometry/quaternion_construction.rs:345`); the Cairo
code wrote `R::is_sign_positive(tr)`, which meant `tr > 0` until simba 0.3.0.

Input: the inverse of the 120° turn about `(1, 1, 1)`, the cyclic permutation with rows
`[[0, 1, 0], [0, 0, 1], [1, 0, 0]]` (`third().inverse()` of `crates/tests_geometry/src/rotation3/tests.cairo`;
also `third() * third()`). Every operation of both branches is exact on it (`sqrt(1) = 1`, halves).
Measured, `(w, i, j, k)`:

| | `w` | `i` | `j` | `k` |
|---|---:|---:|---:|---:|
| nalgebra-rs 0.35.0 (f64) | -0.5 | 0.5 | 0.5 | 0.5 |
| nalgebra-cairo, simba 0.2.0 (raw, `2^32` = 1) | -2147483648 | 2147483648 | 2147483648 | 2147483648 |
| nalgebra-cairo, simba 0.3.0, code unchanged (raw) | 2147483648 | -2147483648 | -2147483648 | -2147483648 |

The 120° turn itself (rows `[[0, 0, 1], [1, 0, 0], [0, 1, 0]]`, the `third()` of the existing exact
test `test_from_rotation_matrix_exact`) gives `(0.5, 0.5, 0.5, 0.5)` in all three columns: both
branches agree on it, so that existing test does not move either way.

Callers of `from_rotation_matrix` in the published crates (each moves with it at such an input), in
`crates/geometry3/src/geometry/`: `unit_quaternion.cairo` `into` from `Rotation3` (89),
`from_basis_unchecked` (652), `face_towards` (695, and the `look_at_*` built on it), `mul_rotation`
(744), `div_rotation` (752); `rotation3.cairo` `mul_unit_quaternion` (296), `div_unit_quaternion`
(303), `from_matrix_eps` (its guess, 740), `slerp` (751), `try_slerp` (764); `isometry3.cairo`
`face_towards` (339), `look_at_rh` (352), `into` from `Rotation3` (801); `isometry_matrix3.cairo`
`into` (727); `unit_dual_quaternion.cairo` `into` from `Rotation3` (1098); and the `nalgebra_glam`
conversions (`glam_rotation.cairo:56`, `glam_isometry.cairo:185`, `glam_similarity.cairo:133`).

**Recommended for REL-b** (code at the site, "only if needed to match nalgebra-rs"): replace
`R::is_sign_positive(tr)` by `tr > R::zero()` at line 418, which is upstream's comparison and keeps
today's bits for every input, and add next to `test_from_rotation_matrix_exact`:

```cairo
// Trace exactly 0 (the -120° turn about (1, 1, 1)): upstream's `tr > 0` takes the k branch,
// so w < 0 here (simba >= 0.3.0 makes `is_sign_positive(0)` true; the test is `tr > 0`).
assert!(
    UnitQuaternionTrait::from_rotation_matrix(r3i([[0, 1, 0], [0, 0, 1], [1, 0, 0]]))
        == uqt((HALF_RAW, HALF_RAW, HALF_RAW, -HALF_RAW)),
);
```

(`uqt` takes `(i, j, k, w)`; `HALF_RAW = 0x80000000`.) This test fails on simba 0.3.0 without the
code change, and passes on both simba versions with it. If REL-b keeps the code unchanged instead,
the same test pins `uqt((-HALF_RAW, -HALF_RAW, -HALF_RAW, HALF_RAW))` and the CHANGELOG must say
that the sign differs from upstream's at a zero trace.

## Site 209 in detail

`shepperd_sign` applies upstream's sign convention to the result of `from_matrix_eps` (both its
paths: `closest_rotation`, which returns a normalised quaternion or the identity for the zero
matrix, and Müller's iteration, a product of unit quaternions). Its last branch reads `k` when
`|k| >= |j|` and not (`|i| > |j|` and `|i| > |k|`); `k = 0` there forces `i = j = 0`, and the branch
also needs `3w² - |v|² <= 0` after the wide rescale, i.e. `3w²` below half an ulp. A unit
quaternion has `|w| = 1` when `v = 0`, so no caller reaches it. Measured on the degenerate input
`(w, i, j, k) = (2^-32, 0, 0, 0)` (raw `(1, 0, 0, 0)`): simba 0.2.0 returns raw `(-1, 0, 0, 0)`,
simba 0.3.0 raw `(1, 0, 0, 0)`; upstream's convention for the rotation it stands for (the identity:
trace 3) is `w > 0`, so the new value is upstream's sign. A module test in
`crates/geometry3/src/internal/geometry/unit_quaternion.cairo` (it has none yet) may pin it; no
public result depends on it.

## Existing tests, goldens and oracle vectors

Static search of every 3x3 integer or raw literal in `crates/**/*.cairo` (3 447 literals): 57 have
a zero trace; the rotation matrices among them are the cyclic permutation `third()` (three test
files, no change: measured above) and its inverse in `test_inverse_is_the_transpose`
(`crates/tests_geometry/src/rotation3/tests.cairo:59`), which compares matrices only. The oracle
vectors of `from_rotation_matrix` and `from_matrix` (`tools/oracle`, random rotations) have no zero
trace. A matrix with trace exactly 0 built at run time (such as `third() * third()`, or a 120° turn
that rounds to a zero trace) cannot be found statically: REL-b's full test run with simba 0.3.0
lists it. With the recommended change at line 418, no existing test, golden or oracle vector can
move through these six sites.

Not covered here (outside item 2): simba 0.3.0 also forwards some functions to fixed 0.5.0 (the R1
forwards) and fixed 0.5.0 is a new dependency version; REL-b's full run is what shows whether any
other nalgebra-cairo result moves.

## How it was measured

- simba 0.3.0's source: the published archive (`https://scarbs.xyz/api/v1/dl/simba/0.3.0`, sha256
  `b2477769287bb0f3e920410c480ab6bf8117cef5f6b960b35f056c044863de13`, equal to the release go),
  `src/scalar.cairo`: `fn is_sign_positive(self: Fixed) -> bool { FixedTrait::is_sign_positive(self) }`,
  documented `self >= 0`; simba 0.2.0: `FixedTrait::is_positive(self)` (`self.raw > 0`).
- nalgebra-rs: a scratch binary on `nalgebra = "=0.35.0"` (`default-features = false`, `alloc`,
  `libm-force`, the oracle's lock file), `cargo run --offline`:
  `from_rotation_matrix(third^T): w=-0.5 i=0.5 j=0.5 k=0.5`, `from_rotation_matrix(third): w=0.5 i=0.5 j=0.5 k=0.5`,
  `UnitComplex::identity().axis_angle() = None`, `UnitComplex::new(2^-32).axis_angle() = Some(([[1.0]], 2.3283064365386963e-10))`.
- nalgebra-cairo: a scratch snforge package outside the workspace depending by path on this
  branch's `nalgebra_types3`, `nalgebra_geometry2`, `nalgebra_geometry3` (Scarb 2.20.1, snforge
  0.64.0, under `prlimit --as=8589934592`), run once with simba 0.2.0 and once with
  `[patch.scarbs-xyz] simba = { path = <the published 0.3.0 archive, unpacked> }` (resolved to
  simba 0.3.0 / fixed 0.5.0 for the whole graph). Both runs: `axis_angle(identity)` is `None`,
  `axis_angle(new(2^-32 raw 1))` is `(1, 2^-32)`; the other values are in the tables above.
