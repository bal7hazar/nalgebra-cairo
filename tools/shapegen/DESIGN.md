# `tools/shapegen`: generator of the static shapes (design, WP 8.1a)

nalgebra-rs has ONE generic type, `Matrix<T, R, C, S>`, and names 48 static aliases of it
(`src/base/alias.rs`: `Matrix1..6`, `MatrixRxC` for R ≠ C in 1..6, `Vector1..6`, `RowVector1..6`)
plus 6 `UnitVector1..6` (= `Unit<VectorN>`): the "54 shapes" of PLAN M8. Cairo has no const
generics, so every shape is a named-field `Copy` struct with its own impls. `shapegen.py` writes
them from one model of a shape and one template per method family; this document fixes the
naming, the architecture, the test strategy, the file layout and the CI budget of WP 8.1b, and
records the measurements taken on the prototype (`proto/`).

Every figure below was measured on this machine (scarb 2.19.4, snforge 0.61.0, `fixed` 0.3.0)
with `measure.sh` or the prototype's benches (`proto/GAS.md`); wall times carry ±20 % noise (shared
machine), CPU times and peak RSS are the stable figures.

## 1. Shapes and names

### 1.1 One Cairo type per shape, upstream aliases as `type` aliases

The 48 matrix aliases name only **36 distinct shapes** (R, C in 1..6): upstream `Vector3` IS
`Matrix3x1`, `RowVector3` IS `Matrix1x3`, `Vector1` IS `RowVector1` IS `Matrix1`. The generator
emits ONE struct per shape and the other upstream names as Cairo type aliases:

| shape | struct (canonical) | aliases | fields (declaration order) |
|---|---|---|---|
| 1x1 | `Matrix1` | `Vector1`, `RowVector1` | `x` |
| N x 1, N ≥ 2 | `VectorN` | `MatrixNx1` | `x y z w a b` (first N) |
| 1 x N, N ≥ 2 | `RowVectorN` | `Matrix1xN` | `x y z w a b` (first N) |
| N x N, N ≥ 2 | `MatrixN` | — | `m11 m21 .. mNN`, column-major |
| R x C otherwise | `MatrixRxC` | — | `m11 m21 .. mRC`, column-major |

- Field names are upstream's `Deref` targets (`base/coordinates.rs`: `X`, `XY`..`XYZWAB` for one
  column or one row, `M2x2`..`M6x6` otherwise), so `v.x`, `m.m23` read like upstream.
- Fields are declared column-major: `Serde` writes upstream's storage order (tested for every
  shape: the generated tests compare the `Serde` image with the column-major raws).
- `new` takes its arguments in ROW-major order like upstream's `Matrix2x3::new`.
- Methods live in `#[generate_trait] pub impl <Struct>Impl of <Struct>Trait`: the trait carries
  the canonical name (`Vector3Trait::new` also builds a `Matrix3x1`; Cairo cannot attach a second
  trait name to an alias without a second, duplicate impl).

**Decision: `VectorN` is the struct, `MatrixNx1` an alias (not two structs).** Measured:

- gas: identical by construction (same type; the prototype's `test_*_new_zeros_from_element`
  binds `Matrix3x1<Fixed>` / `RowVector1<Fixed>` values to `Vector3<Fixed>` / `Matrix1<Fixed>`);
- compile cost of the alternative (12 extra structs with the full surface, plus conversions and the
  products involving both names): per shape, the library costs ≈ 22 MB / 0.1 s of `scarb build`
  (36 shapes: 1,596 MB vs 820 MB for none) and the first family of generated tests ≈ 90 MB / 1 s of
  test build (36 shapes: +3.3 GB) — two structs would pay both twice for no behaviour;
- semantics: upstream code passes a `Vector3` where a `Matrix3x1` is expected (`m.column(0)`,
  `a * v`); only an alias keeps that true. A separate struct would also double the `MatrixMul`
  impls whose output is a column vector.

### 1.2 `Vector6` / `Matrix6`: flat, like upstream (changes DESIGN D4)

The hand-written `Vector6` is two `Vector3` blocks (`v.a.x`) and `Matrix6` four `Matrix3` blocks
(`m.m21.m32`), a spatial-algebra layout whose `Serde` order differs from upstream's. The generator
has one layout for all shapes: `Vector6 { x, y, z, w, a, b }`, `Matrix6 { m11 .. m66 }`. Measured
against the hand-written block `Vector6` (`proto/src/compare.cairo`): `new`, `add`, `sub`, `neg`,
`scale`, `dot`, `norm_squared` have **identical gas**, `zeros` is 300 gas cheaper flat, and every
output is bit-identical. Block access becomes P05's `fixed_view::<3, 3>(i, j)` / `fixed_rows`,
which only move fields (free, BENCHMARK §5). Cost: a breaking change of field paths for users of
`Vector6` / `Matrix6` (`linalg` `Lu6` / `Ldlt6` / `Cholesky6`, rapier's multibody code) —
**owner decision Q2 (2026-09-24): flat**, done in WP 8.1b-2 (§2.4).

### 1.3 Products between shapes: `MatrixMul::mul_mat`

Cairo's `Mul<T>` is homogeneous (`fn mul(lhs: T, rhs: T) -> T`), so `*` can only be the square
product. Every conformable product goes through ONE generic trait with an associated output type
(`proto/src/matrix_mul.cairo`):

```cairo
pub trait MatrixMul<Lhs, Rhs> {
    type Output;
    fn mul_mat(self: Lhs, rhs: Rhs) -> Self::Output;
}
// generated, one impl per conformable pair: 6 x 6 x 6 = 216 impls for the 36 shapes
pub impl Matrix2x3MulMatrix3x2<T, impl R: Real<T>, ...> of MatrixMul<Matrix2x3<T>, Matrix3x2<T>> {
    type Output = Matrix2<T>;
    fn mul_mat(self: Matrix2x3<T>, rhs: Matrix3x2<T>) -> Matrix2<T> { .. }
}
```

- Type-checks and infers as upstream's `*` does: `m.mul_mat(v)` (`Matrix2x3 * Vector3 → Vector2`),
  chains (`s.mul_mat(b).mul_mat(a)`), row × column (`RowVector3 * Vector3 → Matrix1`), outer
  products (`Vector3 * RowVector3 → Matrix3`) — all compiled and tested in the prototype.
- The trait must be imported (`use nalgebra::MatrixMul;`), like every `*Trait` today (measured:
  E0002 "method not found" otherwise) → re-export it at the crate root.
- Square shapes keep `*` / `*=` (upstream operators) and their `MatrixMul` impl delegates to `*`:
  `matrix3_mul_matrix3`: `mul_mat` 22,150 = `operator` 22,150 net gas.
- `mul_vec` is kept on square matrices by the prototype, delegating to `mul_mat` (equal gas:
  `matrix3_mul_vec` 6,140 generated = hand-written), because the existing API and WP 8.0 use it.
- The same pattern gives upstream names to every other shape-dependent method (`tr_mul`, `ad_mul`,
  `kronecker`, `component_mul` stays homogeneous): a generic trait with an associated `Output`
  keeps the upstream METHOD name; only operators need a rename.

Against `scripts/api_parity.py`'s RENAMES policy ("heterogeneous operators are named methods",
DESIGN D4): upstream's `impl:Mul<Matrix>` on `Matrix` becomes ONE rename rule, valid on every
shape, square ones included, using the script's existing owner redirection:

```python
rule(r"Matrix|SquareMatrix|Vector|RowVector|Matrix\w+|Vector\d|RowVector\d", r"impl:Mul<Matrix>",
     r"MatrixMul::mul_mat", "conformable products are `mul_mat` (Cairo's `Mul` is homogeneous)"),
```

`mul_vec` is then a second Cairo name for the same upstream item; strict parity ("neither more nor
less") wants one — **open question Q1** (recommendation: `mul_mat` everywhere, `mul_vec` removed
once rapier-cairo has migrated, or kept as the documented rename of `M * v` if the owner prefers
the geometric name; the gas is identical either way).

### 1.4 `Index`

`m[i]` (column-major linear index, upstream `Index<usize>`) and `m[(i, j)]` (upstream
`Index<(usize, usize)>`) are `IndexView` impls (Cairo's `Index` takes `ref self`; the parity script
maps both to `Index<..>`), panicking with `'Matrix index out of bounds'` (upstream's message).
Variants measured (net gas, last component):

| op | `Matrix3` | `Matrix2x3` |
|---|---:|---:|
| `m[i]`: one `match` on `i` | 870 | 1,300 |
| `m[(i, j)]`: `match j { .. match i }` (**kept**) | 1,200 | 1,000 |
| `m[(i, j)]`: row check, `i + j * R`, one `match` (`alt_linear`, loser) | 2,480 | 2,810 |

## 2. Generator architecture

`shapegen.py` (prototype, 1,264 lines of Python, standard library only) is organised as:

1. **Shape model** (`Shape(r, c)`): canonical name, aliases, module name, field of `(i, j)`,
   column-major / row-major field lists, transposed shape.
2. **Kernel selection** (AGENTS.md rule 4, one rounding per output scalar), reproducing the
   hand-written choices exactly (that is what makes the gas equal):

   | terms K | expression |
   |---|---|
   | 1 | `a * b` |
   | 2-4 | `Real::sum_prodK` (`norm_squaredK`, `normK` for squares) |
   | 5-6 (products) | `Fused::sum_prod5/6`: private `#[inline(always)]` helpers, one `Real::Wide` chain |
   | 5+ (single-output reductions) | `let w = R::wide_add_prod(..)` chain, `wide_rescale` / `wide_sqrt` |

   `Fused::sum_prod6` costs exactly the in-place chain (`row_vector6_mul_vector6`: `fused_helper`
   2,580 = `alt_nested` 2,580 = hand-written `Vector6::dot` 2,580) and cuts the generated library
   by a fifth (36 shapes: 36,558 → 29,284 lines). `simba::Real` stops at `sum_prod4`; the helpers
   are private, so they add no public item.
3. **Inlining rule**, also reproduced from the hand-written shapes: `#[inline(always)]` on
   constructors, component-wise ops, `transpose`, reductions and on products whose output is a
   vector of at most 4 components (`Matrix2/3/4::mul_vec`); default inlining for matrix outputs
   (`Matrix2/3/4/6 * Matrix`, `Matrix6::mul_vec`).
4. **Families**: one function per method family, `family_x(shape, shapes, out)`, appending methods
   to the shape's `#[generate_trait]` impl and top-level items (operators, `MatrixMul`, `Index`)
   to the shape's file. A family decides which shapes it applies to (`identity` / `*` on square
   shapes, `transpose` / products only when the result shape is generated, `cross` on `Vector3`,
   swizzles per dimension, `into_scalar` on `Matrix1`). The prototype's families: construction,
   arithmetic (`Add` / `Sub` / `Neg` / assigns / `scale`), transpose, products, reductions (`dot`,
   `norm_squared`), index.
5. **Specialisations** (`specialisations/<module>.cairo`): hand-written kernels spliced VERBATIM.
   A `// @method <name>` section replaces the generated method of that name or adds a method the
   templates do not have. The prototype carries three: `Matrix2::determinant`,
   `Matrix3::determinant` (copied from `crates/nalgebra`), `Vector3::cross` — all bit-identical and
   at the same gas as the originals (`matrix3_determinant` 10,350 = 10,350, `vector3_cross` 5,540 =
   5,540). 8.1b extends the format with `// @item` (top-level impls such as `Vector3AngleImpl`,
   private helpers) and `// @use` sections.
6. **Rendering**: string templates, then `scarb fmt` on the generated package (the generator never
   formats by hand); `--check` regenerates into `tools/shapegen/.tmp-proto` and compares byte for
   byte (the polygen precedent).

### 2.1 Determinant and inverse per shape (for 8.2 / P14)

Upstream `determinant` is closed-form for 1, 2, 3 and LU otherwise; `try_inverse` uses
`do_inverse2/3/4` then LU. The generator follows it with specialisations: dimension 1 is a template
(`m11`, `1 / m11`), 2 / 3 / 4 are the hand-written closed forms of `Matrix2/3/4` (the 4x4
determinant from 2x2 minors is nalgebra-cairo's measured choice, DESIGN D6), 5 and 6 delegate to the
LU kernels (`Lu6` exists, `Lu5` has to be generated from the same template as P14 decides).

### 2.2 Preserving the hand-written kernels bit for bit (the 8.1b migration protocol)

For each of `Vector2/3/4/6`, `Matrix2/3/4/6`:

1. every hand-written method whose body differs from its template output goes to
   `specialisations/<module>.cairo` verbatim; methods equal to the template are dropped;
2. a staging package generated with `--compare` (the prototype's `compare.cairo`: bit-identity on
   three random inputs + one `__generated` / `__handwritten` bench per method against the current
   crate) must show `=` on every row before the hand-written file is replaced;
3. the existing `tests.cairo` / `benches.cairo` / oracle files of these shapes stay as they are
   and keep running against the generated types; `gas/nalgebra-base.json` must not move (the CI
   gate enforces it).

Prototype result for the overlapping surface (net gas; raw figures equal the committed
`gas/nalgebra-base.json` entries, e.g. `matrix3_mul` 45,690, `vector3_dot` 19,120):

| group | generated | hand-written | | group | generated | hand-written |
|---|---:|---:|---|---|---:|---:|
| `vector2_add` | 1,580 | 1,580 | | `matrix2_mul` | 9,950 | 9,950 |
| `vector2_dot` | 1,780 | 1,780 | | `matrix2_mul_vec` | 3,660 | 3,660 |
| `vector2_scale` | 3,260 | 3,260 | | `matrix2_scale` | 6,620 | 6,620 |
| `vector2_norm_squared` | 1,780 | 1,780 | | `matrix2_transpose` | 400 | 400 |
| `vector3_add` | 2,420 | 2,420 | | `matrix2_norm_squared` | 2,180 | 2,180 |
| `vector3_dot` | 1,980 | 1,980 | | `matrix2_determinant` | 1,780 | 1,780 |
| `vector3_cross` | 5,540 | 5,540 | | `matrix3_add` | 7,460 | 7,460 |
| `vector3_scale` | 4,940 | 4,940 | | `matrix3_mul` | 22,150 | 22,150 |
| `vector3_norm_squared` | 1,980 | 1,980 | | `matrix3_mul_vec` | 6,140 | 6,140 |
| `vector6_add` (flat vs blocks) | 4,940 | 4,940 | | `matrix3_scale` | 15,020 | 15,020 |
| `vector6_dot` | 2,580 | 2,580 | | `matrix3_transpose` | 900 | 900 |
| `vector6_scale` | 9,980 | 9,980 | | `matrix3_norm_squared` | 3,180 | 3,180 |
| `vector6_zeros` | −1,200 | −900 | | `matrix3_determinant` | 10,350 | 10,350 |

All 49 compared groups (`new`, `zeros`, `identity`, `sub`, `neg` included, see `proto/GAS.md`)
are equal except `vector6_zeros` (flat layout 300 cheaper); 49 `*_bit_identical` tests pass.
(`zeros` / `identity` nets are negative because their baseline compares two black-boxed values.)

### 2.3 WP 8.1b-1 outcome: `Vector2/3/4`, `Matrix2/3/4` generated

The six library files are written by `shapegen.py` from `library.py` (templates that reproduce the
hand-written families as they were: explicit `VectorNTrait` + `VectorNImpl` for the vectors,
`#[generate_trait]` for the matrices, `R::zero()` / `R::one()`, the measured kernel and inlining
of every method, the doc comments with their line breaks) and `specialisations/<module>.cairo`.
The generated files differ from the hand-written ones by their two-line header and five blank
lines between crate-internal matrix methods; `tools/shapegen/COMPARE.md` (`compare.sh`, run
before the switch) shows 235 / 235 public operations bit-identical at equal gas, and
`gas/nalgebra-base.json` did not move. Specialisation format (`library.py`): `// @method`,
`// @internal` (crate-internal matrix trait), `// @use`, `// @doc internal`, `// @item <name>
<anchor>` (`struct`, `impl`, `end`); the inline `#[cfg(test)] mod tests` of `Matrix2/3/4` is
spliced from `specialisations/<module>_tests.cairo` (moved to `base/matrixN/tests.cairo` by
8.1b-2). The prototype keeps its own specialisations in `specialisations/proto/`. Unifying the vector / matrix / prototype templates is 8.1b-3's work,
under the same gas gate.

### 2.4 WP 8.1b-2 outcome: flat `Vector6` / `Matrix6` generated

`Vector6 { x, y, z, w, a, b }` and `Matrix6 { m11 .. m66 }` (column-major, `new` row-major) are
written by the same templates, restricted to the surface of the block types they replaced
(`VECTOR_SURFACE`, `MATRIX_SURFACE`: no method added). Above 4 terms `simba::Real` has no
`sum_prodN`, so the 6-term sums are ONE `Real::Wide` chain each (`wide_chain`: `let w = ..`
statements, the hand-written vector form; `wide_expr`: one nested expression, the hand-written
matrix form), `Matrix6::mul_vec` / `tr_mul_vec` keep default inlining, and the `Matrix6`
comparisons (`is_identity`, `abs_diff_eq`) are `#[inline(always)]` 36-term chains: measured
cheaper than the former four `Matrix3` calls and than the chain as a call in every case
(`bench_matrix6_{is_identity,abs_diff_eq}__alt_*`). The specialisation files carry only `// @doc`
sections (the measured notes of the old doc comments; `@doc <method>`, `@doc Mul`, `@doc module`
are new in 8.1b-2). The inline `mod tests` of `Matrix2/3/4` moved to `base/matrixN/tests.cairo`
(same test paths and gas keys); every generated shape declares its `base/<module>/*.cairo` test
modules.

`compare.sh 8d7ccec` (the commit with the hand-written block types; `compare.py` writes their
inputs as block literals and reads a `Matrix6` result through its flat column-major image) shows
281 / 281 operations bit-identical, 274 at equal gas and 7 cheaper (`zeros`, `identity`,
`from_diagonal*` of the flat layout, and the two comparisons), none dearer (`COMPARE.md`). In the
crate, 67 figures of `gas/` drop (6D shapes, `Lu6`, `Ldlt6`, `Cholesky6`: the flat builders and
literals) and none rises. `Matrix6::trace` now sums in upstream's left-to-right order instead of
`(m11 + m22 + m33) + (m44 + m55 + m66)`: equal whenever both are defined; only an overflowing
partial sum could panic in one order and not the other.

### 2.5 WP 8.1b-3 outcome: the 36 shapes, `mul_mat` / `tr_mul` everywhere

`shapes.py` (templates of the 28 new shapes, the aliases, the products, the generated blocks) and
`model.py` (the `Shape` model, shared with the prototype) complete the generator:

- **Shapes**: `Matrix1`, `Matrix5`, `Vector5`, `RowVector2..6` and the 20 `MatrixRxC` get their
  own file (`base/<module>.cairo`), with the surface that the parity items ported on the former
  shapes require on every candidate (`OWNER_CANDIDATES` / `DIM_ONLY`): `new`, `zeros`,
  `transpose`, `abs`, `scale`, `abs_diff_eq`, `+ - -x += -=`; on squares `identity`,
  `from_diagonal(_element)`, `diagonal`, `trace`, `is_identity`, `* *=`; on column vectors `lerp`;
  on vectors the `[T; N]` conversions; on `Matrix1` `into_scalar` / `to_scalar` / `as_scalar`
  (by value). The former `Vector2/3/4/6` gain `transpose` (a `RowVectorN`). The rest of P02-P05
  comes with 8.2 on all shapes at once.
- **Aliases**: `Vector1` / `RowVector1` (= `Matrix1`), `MatrixNx1` / `Matrix1xN`, `UnitVector1..6`
  (= `Unit<VectorN>`), in the file of their shape: all 54 static names of `alias.rs` exist.
  Upstream has no `Matrix1x1`, so there is none (strict parity).
- **Products**: `MatrixMul::mul_mat` (216 impls) and the new generic `MatrixTrMul::tr_mul` (216
  impls, upstream's `tr_mul` taking any operand with as many rows), in the file of the left
  operand. `mul_vec` / `tr_mul_vec` and the square-only `tr_mul` method are gone (owner ruling Q1
  below); `*` / `*=` stay on the squares and the square `mul_mat` delegates to `*`. Kernels:
  `sum_prodK` up to 4 terms, the private `Fused::sum_prod5/6` above (`base/kernels.cairo`), except
  `Matrix6 * Vector6`, which keeps the nested chain of the former `mul_vec` (`LEGACY_WIDE`).
  `tr_mul` is `mul_mat` of the transposed fields: the transpose only relabels values, so it costs
  exactly the former column-reading kernels (`bench_matrix{2,3,4,6}_tr_mul*`, unchanged in
  `gas/`) for half the product code. `gas/` did not move: every former call site of
  `mul_vec` / `tr_mul_vec` / `tr_mul` costs what it cost.
- **Blocks**: the module declarations and re-exports of `base.cairo` / `lib.cairo` are generated
  between `// shapegen: begin` / `// shapegen: end`. `scarb fmt` sorts every run of `mod` / `use`
  items and moves a comment with the item it precedes, so each block is its own run, generated
  already sorted, and the generator checks after formatting that every generated line (and only
  those) is still between the markers.
- **Tests**: `tests_core.py` writes the test-only package `crates/shapes_tests_core` (§3.3 Tier
  A on the 36 shapes, 650 tests + 33 benches), checked by `--check` like the library.
- `compare.py` / `compare.sh` (the 8.1b-1/2 migration proofs) and the prototype (`proto/`,
  whose `compare.cairo` calls the former `mul_vec`) are frozen evidence of those WPs: run them
  against the commit they were written for (`compare.sh 8d7ccec`), not against this tree.

Compile budget (cold builds, this machine, peak RSS): library 1.37 GB / 4.9 s → 2.10 GB / 9.5 s;
`nalgebra` unit-test crate 12.5 GB / 100 s → 13.3 GB / 117 s; `shapes_tests_core` 7.0 GB / 93 s.

### 2.6 WP 8.2a outcome: the base completion (API_PARITY P02) on the 36 shapes

`completion.py` holds one template per upstream family (`norm.rs`, `componentwise.rs`,
`min_max.rs`, `construction.rs`, `matrix.rs`, `ops.rs`, `conversion.rs`, `interpolation.rs`) and
applies it to every shape; a shape that already has a method of that name keeps it (the kernels
of `library.py` / `specialisations/` and `shapes.py` are untouched, so `gas/` did not move: every
committed figure is byte-identical). `shapes.py` appends the missing methods to the new shapes,
`library.py` to `Vector2/3/4/6` (their explicit trait) and `Matrix2/3/4/6` (a `base completion`
section); both add the `<S>AngleTrait` (`Transcendental`: `angle`, `lp_norm`, `slerp`) and the
top-level items. Decisions:

- **Kernels**: sums of products / squares as in §2 (`sum_prodK`, `normK`, one `Real::Wide` chain
  above 4 terms); a common divisor through the prepared-divisor `Real::divN`, chunked to the
  fewest calls (`div_chunks`: `Matrix6::unscale` = `div16 + div16 + div4`, 159 090 gas against
  160 160 for four `div9`; `Matrix2x3::unscale` = one `div6`, 18 930 against 29 640 per
  component).
- **`from_row_slice` / `from_column_slice`**: ONE fixed-size-array read (`Span -> @Box<[T; N]>`,
  one length check): `Matrix3` 1 500 gas net against 9 860 for `*data[k]` per component.
- **`from_fn`**: any `core::ops::Fn<F, (usize, usize)>` whose `Output` converts `Into<T>`: the
  `Output = T` constraint needs the `associated_item_constraints` experimental feature (not
  enabled in `crates/nalgebra`).
- **`get` / `index`**: the generic `MatrixIndex<M, I>` (`base/matrix_index.cairo`, impls for
  `usize` column-major and `(usize, usize)` in each shape's module), `m[i]` / `m[(i, j)]` the
  `IndexView` impls (nested `match`, DESIGN §1.4); panics `nalgebra: index out of bounds`
  (`base/errors.cairo`, generated).
- **`slerp`**: the Kahan half-angle form of `angle` (`θ = 2 atan2(|u - v|, |u + v|)`, `sin θ =
  |u - v| |u + v| / 2`): upstream's `acos(c)` / `sqrt(1 - c²)` loses the last bit of `1 - c²`
  near `c = 1` in fixed point (a unit vector interpolated with itself came out √2 too long);
  155 000 gas against 141 180 (`bench_vector4_slerp__alt_acos`).
- **`Normalizable` is renamed `Normed`** (upstream's trait, implemented for `Vector1..6`), a
  breaking change of a trait name that only `Unit` used.
- Cairo-imposed forms (`api_parity.py` `RENAMES`): `k * m` is `m.scale(k)`, `m * p` / `m * r`
  are `mul_mat`, `m / r` is `m.div_rotation(r)`, `SubsetOf` is `cast`, `ad_mul` a default
  method of `MatrixTrMul`, `eq` the derived `PartialEq`.

Tests: `tests_ops.py` writes four test-only packages, `crates/shapes_tests_{norms,componentwise,
construction,structure}` (one package measured 13.8 GB, two 9.0 / 9.3 GB; four 5.8 / 6.1 / 4.1 /
6.8 GB peak `snforge test`, library 2.9 GB), 392 tests: Tier A (§3.3) with the integer model
extended to `isqrt` of the exact sum (norms) and the half-to-even quotient (`Fraction`, every
division), Tier B float comparisons for `angle`, `lp_norm(3)`, `slerp`, `orthonormalize` and
`orthonormal_subspace_basis`, and the benches of `gas/nalgebra_shapes_tests_norms`.

### 2.7 WP 8.2b outcome: the functional and in-place variants (API_PARITY P03) on the 36 shapes

`functional.py` holds the P03 templates, applied like `completion.py` (a shape keeps a method it
already has; no existing kernel changes, `gas/` did not move): `shapes.py` appends them to the new
shapes, `library.py` to `Vector2/3/4/6` and to a `functional and in-place variants (WP 8.2b)`
section of `Matrix2/3/4/6`. Decisions:

- **Closures**: `core::ops::Fn` bounds like `from_fn`. `map` / `zip_map` / `map_diagonal` return
  `S<Func::Output>` (no conversion); `fold` / `zip_fold` / `apply*` / `fill_with` convert the
  closure's output `Into` the accumulator / scalar (`Output = T` needs the experimental
  `associated_item_constraints`); `fold_with`'s accumulator is `init_f`'s output. A Cairo closure
  takes values and cannot mutate its captures, so upstream's `FnMut(&mut T)` (`apply`,
  `zip_apply`, `zip_zip_apply`) returns the new component instead.
- **`#[inline]` hint**: `#[inline(always)]` is refused on functions with impl generic parameters
  (E2143); the hint removes the call: `apply_norm` = `norm()` (`bench_matrix3_apply_norm`), `map`
  11 090 → 8 660, `fold` 5 090 → 2 960, `swap` 16 420 → 9 360 net gas.
- **Runtime positions** (`fill_row` / `fill_column`, `set_row` / `set_column`, `fill_*_triangle`
  shifts): ONE `match` selecting a whole literal, 2.1-2.7 times cheaper than one comparison per
  component (`bench_matrix4_fill_{row,lower_triangle}__alt_per_component`). `swap_rows` /
  `swap_columns`: two reads and two writes (`<S>EditTrait`, private); the nested `match` on both
  indices is 940 gas cheaper on `Matrix3` but generates R² literals per shape (≈ 40 000 lines per
  method over the 36 shapes): kept as `bench_matrix3_swap_rows__alt_pair_match`.
- **In-place forms** delegate to the by-value kernels (`self = self.op()`), bit-identical;
  `mul_to` / `tr_mul_to` / `ad_mul_to` are default methods of `MatrixMul` / `MatrixTrMul`
  (`mul_to` = `mul_mat`, 12 550 = 12 550 net gas on `Matrix2x3 * Matrix3x2`).
- **`Norm<N, M, T>`** (`base/norm.cairo`): upstream's `Norm<T>` is generic over the matrix; Cairo
  has no common matrix type, so the trait is generic over the marker AND the shape, implemented in
  each shape's module for the four markers (`Matrix3EuclideanNorm`...). `api_parity.py` attributes
  its methods to those implementors (`CROSS_FILE_TRAITS`).

Tests: `tests_functional.py` writes `crates/shapes_tests_{functional,edition,inplace}` (Tier A
with the integer model of §3.3; `LpNorm` on the `TRANSCENDENTAL_SAMPLE` only, `lp_norm`'s
`exp` / `ln` instantiation being heavy) and the benches of `gas/nalgebra_shapes_tests_functional`.
Peak `scarb build --test`: 6.4 / 3.3 / 6.4 GB (one package for the last two measured 7.9 GB).

### 2.8 WP 8.2c outcome: swizzles, rows, columns and blocks (API_PARITY P04, P05) on the 36 shapes

`views.py` holds the P04 / P05 templates, applied like `functional.py` (a shape keeps a method it
already has; no existing kernel changes): `shapes.py` appends them to the new shapes, `library.py`
to `Vector2/3/4/6` and to a `swizzles, rows, columns and blocks (WP 8.2c)` section of
`Matrix2/3/4/6`. Decisions:

- **Owned copies, the output type as the size.** Cairo has neither borrowed views nor const
  generics: `row(i)` / `column(j)` return a `RowVectorC` / `VectorR`, and upstream's
  const-generic views are the methods of generic traits whose OUTPUT type is the size, inferred
  like `Into`'s (`let b: Matrix2x3<Fixed> = m.fixed_view(1, 0);` for `m.fixed_view::<2, 3>(1,
  0)`): `FixedRows` (`fixed_rows`, `select_rows`, default `rows` / `rows_range`), `FixedColumns`
  (likewise), `FixedView` (`fixed_view`, default `view` / `fixed_slice` / `slice`), in
  `base/matrix_view.cairo`, one impl per (shape, output shape) pair in the module of the source
  shape (126 + 126 + 441), so a size that does not fit is a compile error like upstream's. The
  runtime-sized forms (`rows(i, n)`, `view(start, shape)`, `rows_range(a..b)`, `row_part(i, n)`,
  `select_rows(span)`, `resize(r, c, v)`) panic with `nalgebra: dimension mismatch` when the
  requested size is not the output type's (their size check folds away for a literal size:
  `bench_matrix4_rows` = `fixed_rows`). `RowPart` / `ColumnPart` / `FixedResize` are blanket
  impls over crate-private helper traits (`ShapeDims`, `RowVectorLen`, `ColumnVectorLen`,
  `PadTo6`, `CropFrom6`), which resolve from other crates. `MatrixKronecker` (`type Output`) has
  one impl per pair whose product fits in 6x6 (196).
- **Kernels.** A runtime position selects ONE literal through a `match`, nested for a block:
  `Matrix4 -> Matrix2` 3,810 net gas against 4,710 for `fixed_columns` then `fixed_rows` and 8,540
  for a crop of the `Matrix6` canvas after a bounds check; `Matrix6 -> Matrix3` 7,200 against
  18,100 composed (`bench_matrix{4,6}_fixed_view__alt_*`). `fixed_resize` pads to the `Matrix6`
  canvas and crops (constant positions): struct moves are free once inlined, so it costs exactly
  the direct literal (`bench_matrix2x4_fixed_resize`: 1,900 = 1,900) for 72 small impls instead
  of 1,296 literals. `kronecker` is one floored product per component.
- **The internal `column1..N` / `row1..N`** of `Matrix2/3/4InternalTrait` stay: `linalg` calls
  them, `row2()` returns a `Vector3` where `row(1)` returns a `RowVector3`, and the compile-time
  position is cheaper (`base/matrix3/benches_views.cairo`: `column2()` 300 net gas, `column(j)`
  1,210). The public `row` / `column` reuse the private `row_at` / `column_at` of `swap_rows`.
- **`from_rows` takes column vectors** (`VectorC`, one argument per row), the form of the former
  `Matrix2/3/4::from_rows` that `linalg` calls (upstream: a slice of row vectors).
- **Edition into the neighbouring shape**: `insert_row` / `insert_column` / `remove_row` /
  `remove_column` return the shape with one row / column more or less when it exists (`DIM_ONLY`:
  no 7-row, no 0-row shape); upstream's `insert_rows(i, n)`, `remove_rows`, `resize_vertically`
  and the fixed variants stay with the dynamic matrices (P13).
- **Swizzles**: the `impl_swizzle!` set on the column vectors (`Vector1` is `Matrix1`) in
  `<S>Trait`, and on the six points in `base/point_swizzle.cairo` (`Point1..6SwizzleTrait`;
  `Point3::xy`, `Vector3/4::xy` and `Vector4::xyz` predate them and keep their traits).
- **Properties**: `is_orthogonal(ulps)` is `selfᵀ * self` (fused `tr_mul`) against the identity
  (the 1x1 `norm_squared` on column vectors); `is_invertible` / `is_special_orthogonal` exist
  where `try_inverse` / `determinant` do (`Matrix2/3/4/6`, `Lu6` for 6).

Compile budget (cold `SCARB_INCREMENTAL=false`, peak RSS, this machine): library
(`scarb build -p nalgebra`) 4.54 → 5.26 GB (5.9 GB with every runtime-sized form in each impl,
4.97 GB with the composed `fixed_view`); `nalgebra` unit-test crate 7.46 → 8.76 GB; every
generated test package about +1.1 GB (`shapes_tests_edition` 5.53 → 6.71 GB,
`shapes_tests_core` 8.70 → 9.80 GB). Tests: `tests_views.py` writes
`crates/shapes_tests_{views,blocks}` (Tier A on the 36 shapes and the six points: 121 / 110
tests, 7.1 / 6.5 GB; one package measured 7.5 GB) and the benches of
`gas/nalgebra_shapes_tests_views`.

### 2.9 WP 8.3-P07 outcome: the homogeneous / computer-graphics helpers (API_PARITY P07)

`cg.py` writes upstream's `base/cg.rs` into ONE shared module, `base/cg.cairo` (upstream's
file, no growth of the 36 shape files): `Matrix1..6CgTrait` (any `Real` scalar) and
`Matrix3/4CgAngleTrait` (`Transcendental`: the rotation constructors). Decisions:

- **Dimensions.** The helpers whose argument or result is a `D - 1` vector exist on
  `Matrix2..6` (`Vector1` is `Matrix1`); `Matrix1` has the uniform `new_scaling` (the identity,
  like upstream's) and `append_scaling` / `prepend_scaling` (which scale no row / column and
  return `self`). `api_parity.py` `DIM_ONLY` lists the others without `Matrix1`.
- **Delegation.** The rotation, observer and view constructors are upstream's one-liners:
  `Rotation2/3::...(..).to_homogeneous()`, `IsometryMatrix3::{face_towards, look_at_rh, look_at_lh,
  rotation_wrt_point}(..).to_homogeneous()` (same bits as the geometry types, checked by the
  generated tests; their numerics by the oracle suite `cg` in `crates/tests_base`).
  `Matrix4::new_perspective` / `new_orthographic` come with `Perspective3` / `Orthographic3`.
- **Kernels.** Scalings: one floored product per scaled component. Every sum of products with an
  exact addend (`append_translation`, `prepend_translation`, `transform_point`) is ONE
  `Real::Wide` chain closed by `wide_add` (`R::mul_add` for one product): bit-identical to the
  sum rounded then added, and measured cheaper than `sum_prodK(..) + c`
  (`bench_matrix4_prepend_translation`: 12,320 against 19,810 net; `append_translation` 24,960
  against 37,570; `transform_point` 27,630 against 29,790; the loser is rendered from the same
  template, `cg.CG_ADDEND`, as `alt_plain_*` in `crates/shapes_tests_cg/src/benches.cairo`).
  `transform_vector` / `transform_point` divide by the normaliser with `R::div` / `R::div3/4/5`
  (bit-identical to `/`) in upstream's order (the vector before the product, the point after),
  and take upstream's other branch (no division) when it is zero.
  `new_nonuniform_scaling_wrt_point` is `R::mul_add(-pt, scaling, pt)`: one rounding of the
  exact value.
- **Tests.** `tests_cg.py` writes `crates/shapes_tests_cg`: every `Matrix{N}CgTrait` operation is
  modelled EXACTLY on integer raws (71 tests with the delegation checks and the benches; peak
  `scarb build --test` 6.6 GB).

## 3. Generated tests under the compile budget

### 3.1 What the budget is

The existing `nalgebra` unit-test crate already needs **106.6 s wall (289 s CPU) and 13.2 GB peak**
to compile (`scarb build --test -p nalgebra`; its library alone: 5.3 s, 1.39 GB), close to the 16 GB
of a GitHub `ubuntu-latest` runner. CI shards only split the Sierra→CASM compilation and the
execution (`snforge test -p nalgebra <filter>`); every shard compiles the whole test crate.
Integration-test files do not help either: generating the tests as one `tests/<family>.cairo`
crate per family gave 48.0 s / 4.9 GB against 39.3 s / 5.1 GB as one unit-test crate (Scarb builds
them in the same process). **Generated tests therefore go to separate test-only packages**, one
CI job each (`snforge test -p <package>`), each compiling only itself plus the library.

### 3.2 Measured cost of the generated tests (all 36 shapes, first family)

`measure.sh all <label> [--test-families f]` (library + tests; wall s (CPU s), peak RSS):

| configuration | lib lines | test lines / tests | `scarb build` | `scarb build --test` | `snforge test` |
|---|---:|---:|---|---|---|
| no shape (fixed cost) | 133 | 0 / 2 | 3.1 s (6.5), 820 MB | 3.4 s (7.5), 940 MB | 0.3 s |
| prototype, 11 shapes | 3,038 | 1,936 / 118 | 3.3 s (6.7), 925 MB | 5.6 s (14.5), 1,423 MB | 2.2 s |
| 36 shapes, v1 tests (struct literals, derived `==` per assertion) | 36,567 | 22,774 / 504 | 5.8 s, 1,673 MB | 47.9 s, **7,087 MB** | 21.4 s |
| 36 shapes, v3 tests (`load` / `assert_raws` helpers, loops) | 29,339 | 12,252 / 504 | 6.7 s, 1,614 MB | 43.2 s (105), **5,057 MB** | 43.0 s (137) |

Per family, 36 shapes (test build; the library alone is ≈ 6.3 s / 1.7 GB):

| family (tests) | v1 | v2 (helpers) |
|---|---|---|
| construction: `new`, `zeros`, `from_element`, `repeat`, `identity`, aliases (36) | 15.0 s, 2,580 MB | 8.5 s, 2,032 MB |
| arithmetic: `+ - -x += -= scale`, overflow panic (72) | 19.5 s, 3,144 MB | 20.8 s, 2,652 MB |
| transpose (36) | 8.6 s, 2,107 MB | — |
| reductions: `dot`, `norm_squared` (36) | 7.1 s, 1,935 MB | — |
| index: both impls, out-of-bounds panics (108) | 17.8 s, 3,556 MB | 10.0 s, 2,287 MB |
| products: 216 `mul_mat` + `*`, `*=`, `mul_vec` (216) | 14.4 s, 3,016 MB | 14.2 s, 3,097 MB |

Findings that become rules for 8.1b / 8.2:

- the cost is the code instantiated per test, not the number of tests: a derived `==` or a struct
  literal on a 36-field shape expands inline at every assertion. Shape values are loaded from and
  compared with column-major raw spans through two generic helpers (`load`, `assert_raws`:
  `Serde` image, one monomorphisation per shape), index-like checks run in loops (one expansion of
  the `match` per test instead of one per component): construction and index 2-4x cheaper, total
  peak 7.1 → 5.1 GB. (The snforge phase went from 21 to 43 s wall on these runs; CPU-bound noise is
  large here, the memory drop is the robust result.)
- products are pure library instantiation (216 kernels): their cost does not shrink with the test
  style, it is the price of testing every conformable pair once.
- the library itself scales mildly with the surface: 8 renamed copies of every trait method
  (`--replicate 8`: 3.7k methods over 36 shapes, 67.6k lines) build in 8.2 s (18.5 CPU),
  2,186 MB, against 7.7 s, 1,596 MB for one copy.

### 3.3 Strategy

- **Tier A, every shape (36), every generated method, one exact case per (shape, family)**: the
  prototype's generated tests. Expectations come from a bit-exact integer model inside the
  generator, so they cost no oracle run: the prototype models the floor kernels (the floor of the
  exact sum for products, `dot`, `norm_squared`, `scale`); 8.2 adds `isqrt` of the exact sum of
  squares for norms and half-to-even rounding of the exact quotient (`Fraction`) for `/`,
  `fixed`'s nearest division. They catch field-mapping bugs and Sierra-level failures of each instantiation; panic
  messages are checked once per (shape, family) (`#[should_panic(expected: ..)]`).
- **Tier B, properties and oracle vectors on a sample**: the existing hand-written tests, oracle
  files (`tools/oracle`) and benches of `Vector2/3/4/6`, `Matrix2/3/4/6` stay in `crates/nalgebra`
  (they are the regression suite of the migrated kernels); new numerics (norms, `normalize`,
  inverse, LU-based determinant) get oracle vectors on `{Matrix1, Vector2, Vector5, RowVector3,
  Matrix2x3, Matrix3x2, Matrix5, Matrix6x4}` — one shape per kind (1x1, column, row, square,
  wide, tall) and per kernel size (K = 1..6).
- Test-file rules: no struct literal or derived `==` per assertion (use `load` / `assert_raws`),
  loops for per-component checks, at most one `#[should_panic]` test per (shape, family), no
  generated test file over ~2,000 lines (the prototype's largest family file is 459 lines; its
  `compare.cairo`, 6,088 lines of one-off migration proofs, is not a pattern for the library).

## 4. File layout

Library (`crates/nalgebra/src/base/`, generated files carry the `// Generated by
tools/shapegen/shapegen.py` header and are checked by `shapegen.py --check`):

```
base/<module>.cairo          one file per canonical shape (36): struct, aliases, <Struct>Trait impl,
                             operators, Index, the MatrixMul impls where it is the left operand
                             (vector3.cairo, matrix2x3.cairo, row_vector3.cairo, ...: the paths of
                             the 8 existing shapes do not change)
base/matrix_mul.cairo        the MatrixMul trait
base/kernels.cairo           private Fused::sum_prod5/6
base/<module>/{tests,benches,oracle}.cairo   hand-written, only for the 8 existing shapes
tools/shapegen/specialisations/<module>.cairo  verbatim hand-written kernels
```

- One file per shape, not per family: AGENTS.md's "one type per file", localised diffs, and
  `api_parity.py` attributes items to owners by trait name (`Matrix2x3Trait` → `Matrix2x3`) and
  by impl name prefix (`Matrix2x3MulMatrix3x2` → `Matrix2x3`).
- Module declarations: `base.cairo` and the re-exports of `lib.cairo` (canonical structs, aliases,
  traits, `MatrixMul`) are generated blocks between `// shapegen: begin` / `// shapegen: end`
  markers (orchestrator-owned files: 8.1b's brief must allow it).
- Generated tests: test-only workspace members (`publish = false`, dependency on `nalgebra`),
  e.g. `crates/nalgebra_shapes_tests_<group>/src/{lib,<family>}.cairo`, one per group of families
  (§5). They are outside `crates/nalgebra/src`, so `api_parity.py` never scans them.

### 4.1 `scripts/api_parity.py` changes (done in WP 8.1b-3)

Ran the script's Cairo parser on the prototype (`parse_cairo` with `ROOT` redirected to a copy):
struct types, traits, methods, operators and derives are found per shape, but

1. `pub type` aliases are not types (only `pub struct|enum` are): add `pub type X<T> = Y<T>;` as
   `type:X` (and let owner `X` resolve to `Y`'s items) — 12 upstream names depend on it;
2. the `MatrixMul` impls live in other files than the trait, so `mul_mat` is attributed to owner
   `MatrixMul` (the generic-trait branch only searches the trait's own file): either search every
   file, or use the RENAMES redirection `MatrixMul::mul_mat` of §1.3 (works as is);
3. `RowVectorN` operands normalise to `Add<RowVector2>` instead of `Add<Matrix>`
   (`impl_rhs_family`): map `RowVector\d` / `Matrix\d(x\d)?` / `Vector\d` to `Matrix`;
4. `OWNER_CANDIDATES` grows to: `"Matrix"`: the 36 canonical structs; `"SquareMatrix"`:
   `Matrix1..6`; `"Vector"`: `Matrix1, Vector2..6`; `"RowSVector"` / `"RowVector"`: `Matrix1,
   RowVector2..6`; one entry per alias (`"Matrix3x1": ["Vector3"]`, `"Vector1": ["Matrix1"]`, ...);
   `DIM_ONLY` loses `transpose`, `tr_mul`, `from_rows`, `from_columns`, `Mul<Matrix>` (generated on
   every shape) and gains per-dimension entries for the swizzles (P04) and `into_scalar` /
   `as_scalar` / `to_scalar` (`Matrix1`).

## 5. CI impact and shard plan

Measured (this machine, 4 build threads): see §3.2. Extrapolation to the full static surface
(P01-P05 ≈ 120 methods per shape, ≈ 8x the prototype's families, 216 product impls unchanged):

| unit | today | after 8.1b (first family) | after 8.2 (P02-P05) |
|---|---|---|---|
| `scarb build` of `nalgebra` (also paid by every consumer, measured: a crate using one function of the 36-shape library builds in 7.5 s / 1.73 GB vs 4.7 s / 0.89 GB with 2 shapes) | 5.3 s, 1.39 GB | ≈ 9 s, ≈ 2.2 GB | ≈ 11 s, ≈ 2.8 GB |
| `nalgebra` unit-test crate (hand-written tests unchanged) | 107 s, 13.2 GB | ≈ 110 s, ≈ 14 GB | ≈ 115 s, ≈ 14.6 GB ⚠ |
| generated Tier-A tests, all shapes | — | 43 s, 5.1 GB (1 package) | ≈ 180 s, ≈ 17 GB above the library → 5 packages |

Per method family over 36 shapes: ≈ 1.5 s and 0.14 GB of test build per method (v2 figures:
2.27 GB / 23.5 s for 16 non-product methods), products 1.35 GB / 7.8 s.

Shard plan (one CI job each, `snforge test -p <package>`; target ≤ 60 s and ≤ 6 GB per build):

| job | content | estimated build |
|---|---|---|
| `nalgebra` (existing 8 shards) | hand-written tests, oracle, benches | 13-15 GB ⚠ (see Q4) |
| `shapes_core` | P01: construction, arithmetic, transpose, products, dot, index | 43 s, 5.1 GB (measured) |
| `shapes_norms` | P02: norms, `normalize`, component-wise, min/max, `cast`, conversions | ≈ 45 s, ≈ 6 GB |
| `shapes_functional` | P03: `map` / `zip` / fold, `*_mut` / `*_assign` in place | ≈ 45 s, ≈ 6 GB |
| `shapes_swizzles` | P04 (column vectors, 6 shapes; 2-3 letter swizzles) | ≈ 20 s, ≈ 3 GB |
| `shapes_views` | P05: rows, columns, `fixed_view`, blocks, triangles | ≈ 45 s, ≈ 6 GB |

Each new package is re-measured with `measure.sh` (or `/usr/bin/time` on its test build) in the
PR that creates it; a package over 6 GB splits by shape group (`R ≤ 3` / `R ≥ 4`).

## 6. Open questions (owner / orchestrator)

- **Q1** (**decided**, owner 2026-09-24: `mul_mat` everywhere, `mul_vec` / `tr_mul_vec` removed,
  `*` on the squares; done in WP 8.1b-3, §2.5) product naming: `mul_mat` for every conformable product (square included) and `*` for
  the square operator; does `mul_vec` stay as a second name (today's API, WP 8.0's rename of
  `M * v`) or go (strict parity)? Alternatives considered: `mul_matrix` (longer), per-shape names
  (`mul_3x4`: 216 names), a generic `MatrixMul::mul` (collides with `core::traits::Mul::mul` on
  square shapes: ambiguous method).
- **Q2** flat `Vector6` / `Matrix6` (upstream layout and `Serde` order, equal gas) instead of the
  blocks of DESIGN D4 — a breaking change of field paths for `linalg` and rapier-cairo.
  **Decided: flat** (owner, 2026-09-24; WP 8.1b-2, §2.4).
- **Q3** the generated-tests packages are new workspace members and CI jobs (§5).
- **Q4** the `nalgebra` unit-test crate is at 13.2 GB of 16 GB: moving the hand-written `base`
  tests of the 8 migrated shapes into a `shapes_legacy` test package (or splitting `geometry` /
  `linalg` tests out) before 8.2 keeps it clear of the runner's limit.
- **Q5** `simba::Real` escalation for P02: `unscale` / `normalize` divide every component by ONE
  divisor; `Real` offers `div3/4/5/6/9/16` only. Shapes have 1-36 components (8, 10, 12, 15, 18,
  20, 24, 25, 30, 36 missing): a prepared-divisor API (`Real::Divisor`, `divisor(d)`,
  `div_by(x, divisor)`, bit-identical to `/`) would serve every size with one preparation.
- **Q6** `identity` on non-square shapes (upstream defines it: ones on the diagonal): the prototype
  generates it on square shapes only (today's `DIM_ONLY`); 8.2 should generate it everywhere.

## 7. Usage

```
python3 tools/shapegen/shapegen.py            # regenerate the library shapes, the base.cairo /
                                              # lib.cairo blocks, crates/shapes_tests_core and
                                              # tools/shapegen/proto (runs scarb fmt)
python3 tools/shapegen/shapegen.py --check    # fail if the committed output is stale
snforge test -p nalgebra_shapes_tests_core                 # the generated Tier-A tests
cd tools/shapegen/proto && scarb build && snforge test     # 345 tests
tools/shapegen/measure.sh all all36                        # compile budget of all 36 shapes
tools/shapegen/measure.sh all x8 --no-tests --replicate 8  # library growth with the surface
```
