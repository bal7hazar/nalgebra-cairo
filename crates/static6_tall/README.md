# nalgebra_static6_tall

The methods of the shapes with 6 rows and fewer columns of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- the METHODS of the static shapes with 6 rows and fewer columns: `Matrix6x2Trait`..`Matrix6x5Trait`,
  `Vector6Trait` (`Matrix6x1` is `Vector6`), `UnitVector6Trait` and their `...AngleTrait` forms
  (trigonometry, `exp`, `ln`...): constructors, component-wise operations, reductions, norms,
  interpolation, the base completion of upstream's `Matrix` / `Vector` API;
- the crate-private edit kernels of every dimension-6 shape (`Matrix6EditTrait`,
  `Matrix2x6EditTrait`..., under `internal`, see below).

The types themselves (`Matrix6x2`, `Vector6`...) are in [`nalgebra_shapes6`](../shapes6/README.md);
this crate depends on it, on [`nalgebra_core`](../core/README.md),
[`nalgebra_shapes5`](../shapes5/README.md) and on the method crates of the smaller dimensions
([`nalgebra_static3`](../static3/README.md), [`nalgebra_static4`](../static4/README.md),
[`nalgebra_static5`](../static5/README.md)). The methods of the shapes with 6 COLUMNS (`Matrix6`,
`Matrix2x6`..`Matrix5x6`, `RowVector6`) are in [`nalgebra_static6_wide`](../static6_wide/README.md).
Upstream's module paths are kept: `nalgebra_static6_tall::base::vector6::Vector6Trait` is
`nalgebra::base::vector6::Vector6Trait` (docs/SPLIT.md §3.1).

## When to depend on it

Depend on `nalgebra_static6_tall` (and the crates it depends on) when you need the methods of
6-vectors or of the 6-row matrices (spatial vectors, 6-row Jacobians) without the rest of the
library. For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which
re-exports every sub-crate at the 0.1.0 paths.

## Features

`closures` (default): the methods of the shapes that take a closure (`map`, `fold`, `apply`,
`zip_*`...), as in `nalgebra` (whose feature of the same name forwards to this one).

## `internal`

`nalgebra_static6_tall::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_static6_tall` use (the edit kernels of the dimension-6 shapes, called by the
methods of `static6_wide` and by the block and view impls). They are public only for those crates:
**internal, no stability promise**, they may change or disappear in any release, and the facade
never re-exports them.

## License

MIT.
