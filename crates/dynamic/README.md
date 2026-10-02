# nalgebra_dynamic

The dynamically sized matrices (upstream `nalgebra::base::dynamic`: `DMatrix`, `DVector`,
`RowDVector`) of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of
the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::dynamic::dmatrix`: `DMatrix` (its dimensions are runtime values, column-major storage)
  and the partially dynamic aliases `Matrix2xX..Matrix6xX`, `MatrixXx2..MatrixXx6`;
- `base::dynamic::dvector`, `base::dynamic::row_dvector`: `DVector` (`MatrixXx1`), `RowDVector`
  (`Matrix1xX`);
- the conversions of the static shapes to and from them, and `base::dynamic::shapes` (its dynamic
  part): the edition forms of the static shapes whose result is dynamic (`insert_columns`,
  `remove_columns`...) with their `<Shape>DynamicTrait`s. The fixed-size edition of the same module
  (`InsertFixedRows`, `RemoveFixedColumns` & co., 360 impls) is in
  [`nalgebra_blocks`](../blocks/README.md) since WP 9-NS12b; the facade re-exports both parts at the
  0.1.0 path;
- `base::dynamic::convolution`: `convolve_full` / `convolve_same` / `convolve_valid` of the
  column vectors.

Upstream's module paths are kept: `nalgebra_dynamic::base::dynamic::DMatrix` is
`nalgebra::base::dynamic::DMatrix` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), the types of every dimension
([`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md)), the method
crates of the static shapes ([`nalgebra_static_core`](../static_core/README.md),
[`nalgebra_static2`](../static2/README.md), [`nalgebra_static3`](../static3/README.md),
[`nalgebra_static4`](../static4/README.md), [`nalgebra_static5`](../static5/README.md),
[`nalgebra_static6_tall`](../static6_tall/README.md),
[`nalgebra_static6_wide`](../static6_wide/README.md)); it does not depend on `nalgebra_blocks`.

## When to depend on it

Depend on `nalgebra_dynamic` (and the crates it depends on) when you need matrices whose size is
only known at run time. It depends on every method crate, so it is a dimension 5-6 build: no declared
closure names it; its marginal cost over its direct dependencies (1.5 s / 0.26 GB on the
runner, CI run 37019191074 of the Scarb 2.20.1 bump, 10,442 lines, under the 5 s / 1 GB gate since the fixed-size edition moved to
`nalgebra_blocks`) is in `docs/PACKAGES.md`. For upstream's whole API at upstream's paths, depend on the facade
`nalgebra`, which re-exports every sub-crate at the 0.1.0 paths (behind its feature `dynamic`, on by
default: `nalgebra` always depends on this crate, the feature gates its re-exports only).

## Features

`closures` (on by default, forwarded by `nalgebra`'s feature of the same name): the methods that
take a closure (`map`, `fold`, `apply`, `zip_*`...). It gates this crate's own methods only and no
longer forwards to the static method crates (WP 9-NS12b: none of these methods calls their closure
methods; a user of those traits depends on them directly, with their own `closures`; the facade
forwards to each of them).

## `internal`

This crate has no `internal` module. The fields of `DMatrix` (`data`, `nrows`, `ncols`) and of
`DVector` (`data`), crate-private in `nalgebra` 0.1.0, are `pub` because `nalgebra_sparse` builds
and reads them: **internal, no stability promise**; build these types with the constructors of
`DMatrixTrait` / `DVectorTrait`.

## License

MIT.
