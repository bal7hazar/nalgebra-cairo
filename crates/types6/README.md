# nalgebra_types6

The types of dimension 6 (shapes with max(rows, cols) = 6, their core-trait impls, products, indexing, solve / permutation / Givens / LU-step / Householder impls, without the edit kernels); Point6, Translation6.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- the shapes with max(rows, columns) = 6 (`Matrix6`, `Matrix2x6`, `Matrix6x5`, `Vector6`, `RowVector6`...), their core-trait impls, products, indexing and solve impls, without the edit kernels (they sit in `nalgebra_static6_tall`, so that this crate keeps its line margin under the 40,000-line gate);
- `Point6`, `Translation6` and the permutation / Householder impls of dimension 6.

Upstream's module paths are kept: `nalgebra_types6::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md) and `simba`.

## When to depend on it

It is also pulled in by 18 other declared closures, the cheapest being `static5_pivot` (9.4 s / 3.26 GB, 20 s / 4.5 GB).

Depend on the `types` crates when you need the types and their operators without the named methods (those are in the `static` crates). It is the largest crate of the library (37,463 lines).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`, `internal::geometry`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
