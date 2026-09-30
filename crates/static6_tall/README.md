# nalgebra_static6_tall

The methods of the shapes with 6 rows and fewer columns (Matrix6x1..Matrix6x5, Vector6); the dimension-6 edit kernels.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- the shapes with 6 rows and fewer columns (`Matrix6x1`..`Matrix6x5`, `Vector6`): one inherent trait per shape, with the constructors, accessors, products, norms, edition and the methods that take a closure (feature `closures`);
- the dimension-6 edit kernels (crate-private in `nalgebra` 0.1.0, under `internal`).

The dimension-6 methods are cut in two crates (`static6_tall`, `static6_wide`): the one documented exception to "a number in a crate name is exactly one dimension", needed to keep each under the 40,000-line gate.

Upstream's module paths are kept: `nalgebra_static6_tall::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static2`](../static2/README.md), [`nalgebra_static3`](../static3/README.md), [`nalgebra_static4`](../static4/README.md), [`nalgebra_static5`](../static5/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

It is pulled in by 10 declared closures, the cheapest being `pivot6` (12.8 s / 4.30 GB, 20 s / 4.5 GB).

Depend on it (or on `nalgebra_static6_wide`, which pulls it) for the methods of the dimension-6 shapes; dimension 5-6 closures have their own budget (facade README, "Dimensions 5 and 6").

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
