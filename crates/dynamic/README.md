# nalgebra_dynamic

The dynamically sized matrices (upstream `nalgebra::base::dynamic`: `DMatrix`, `DVector`,
`RowDVector`) of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of
the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::dynamic::dmatrix`: `DMatrix` (its dimensions are runtime values, column-major storage)
  and the partially dynamic aliases `Matrix2xX..Matrix6xX`, `MatrixXx2..MatrixXx6`;
- `base::dynamic::dvector`, `base::dynamic::row_dvector`: `DVector` (`MatrixXx1`), `RowDVector`
  (`Matrix1xX`);
- the conversions of the static shapes to and from them, and `base::dynamic::shapes`: the edition
  forms of the static shapes whose result is dynamic or another static shape (`insert_columns`,
  `remove_fixed_rows`...), with `InsertFixedRows`, `RemoveFixedColumns` & co.;
- `base::dynamic::convolution`: `convolve_full` / `convolve_same` / `convolve_valid` of the
  column vectors.

Upstream's module paths are kept: `nalgebra_dynamic::base::dynamic::DMatrix` is
`nalgebra::base::dynamic::DMatrix` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), the types of every dimension
([`nalgebra_types2`](../types2/README.md) .. [`nalgebra_types6`](../types6/README.md)), the method
crates of the static shapes ([`nalgebra_static_core`](../static_core/README.md),
[`nalgebra_static2`](../static2/README.md) .. [`nalgebra_static5`](../static5/README.md),
[`nalgebra_static6_tall`](../static6_tall/README.md),
[`nalgebra_static6_wide`](../static6_wide/README.md)) and
[`nalgebra_blocks`](../blocks/README.md) (padding / cropping to the 6x6 kernels).

## When to depend on it

Depend on `nalgebra_dynamic` (and the crates it depends on) when you need matrices whose size is
only known at run time. It depends on every method crate, so it is a dimension 5-6 build: no declared
closure names it; its marginal cost over its direct dependencies is in `docs/PACKAGES.md`
(docs/SPLIT.md §20: it still has to be cut under the 5 s / 1 GB marginal gate). For upstream's whole API at upstream's paths, depend on the facade
`nalgebra`, which re-exports every sub-crate at the 0.1.0 paths (behind its feature `dynamic`, on by
default: `nalgebra` always depends on this crate, the feature gates its re-exports only).

## Features

`closures` (on by default, forwarded by `nalgebra`'s feature of the same name): the methods that
take a closure (`map`, `fold`, `apply`, `zip_*`...). It forwards to the feature of the same name
of the static method crates.

## `internal`

This crate has no `internal` module. The fields of `DMatrix` (`data`, `nrows`, `ncols`) and of
`DVector` (`data`), crate-private in `nalgebra` 0.1.0, are `pub` because `nalgebra_sparse` builds
and reads them: **internal, no stability promise**; build these types with the constructors of
`DMatrixTrait` / `DVectorTrait`.

## License

MIT.
