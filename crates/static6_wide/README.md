# nalgebra_static6_wide

The methods of the shapes with 6 columns (Matrix6, Matrix2x6..Matrix5x6, RowVector6); Lu6, Perm6.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- the shapes with 6 columns (`Matrix6`, `Matrix2x6`..`Matrix5x6`, `RowVector6`): one inherent trait per shape, with the constructors, accessors, products, norms, edition and the methods that take a closure (feature `closures`);
- `Lu6` and `Perm6` (used by `Matrix6`, so they cannot sit in a decomposition crate).

This is the second half of the dimension-6 methods: `nalgebra_static6_wide` depends on [`nalgebra_static6_tall`](../static6_tall/README.md) (the pair is one declared closure, `static6`).

Upstream's module paths are kept: `nalgebra_static6_wide::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static6_tall`](../static6_tall/README.md) and `simba`.

Feature `closures` (on by default, forwarded by the facade's feature of the same name): the methods that take a closure (`map`, `fold`, `apply`, `zip_*`...).

## When to depend on it

`nalgebra_static6_wide` is a member of the declared closures `static6` (17.3 s / 3.93 GB, 20 s / 4.5 GB), `static6_factor` (17.7 s / 4.11 GB, 20 s / 4.5 GB), `static6_svd` (21.5 s / 4.70 GB, documented, not gated), `static6_pivot` (18.5 s / 4.30 GB, 20 s / 4.5 GB), `static6_spectral` (21.5 s / 4.64 GB, documented, not gated), `static6_geometry` (18.0 s / 4.01 GB, 20 s / 4.5 GB). It is also pulled in by `pivot6` (12.8 s / 4.30 GB, 20 s / 4.5 GB), `static4_norm` (18.7 s / 4.04 GB, documented, not gated).

Depend on it for the methods of `Matrix6` and the shapes with 6 columns; it pulls `nalgebra_static6_tall`. Dimension 5-6 closures have their own budget (facade README, "Dimensions 5 and 6").

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::linalg`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
