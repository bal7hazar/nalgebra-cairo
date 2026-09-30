# nalgebra_static5

The methods of the dimension-5 shapes (Matrix5, Vector5, Matrix2x5, Matrix5x3...).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- the shapes of dimension 5 (`Matrix5`, `Vector5`, `RowVector5`, `Matrix2x5`, `Matrix5x3`...): one inherent trait per shape (`Matrix5Trait`...), with the constructors, accessors, products, norms, edition and the methods that take a closure (feature `closures`).

Upstream's module paths are kept: `nalgebra_static5::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static2`](../static2/README.md), [`nalgebra_static3`](../static3/README.md), [`nalgebra_static4`](../static4/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

`nalgebra_static5` is a member of the declared closures `static5` (12.9 s / 3.05 GB, 20 s / 4.5 GB), `static5_factor` (10.9 s / 3.13 GB, 20 s / 4.5 GB), `static5_svd` (10.4 s / 3.53 GB, 20 s / 4.5 GB), `static5_pivot` (9.4 s / 3.26 GB, 20 s / 4.5 GB), `static5_spectral` (14.8 s / 3.35 GB, 20 s / 4.5 GB), `static5_geometry` (13.3 s / 3.14 GB, 20 s / 4.5 GB). It is also pulled in by 10 other declared closures, the cheapest being `pivot6` (12.8 s / 4.30 GB, 20 s / 4.5 GB).

Depend on it for the named methods of the dimension-5 shapes; dimension 5 and above are documented closures of their own (facade README, "Dimensions 5 and 6").

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
