# nalgebra_static6_wide

The methods of the shapes with 6 columns of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- the METHODS of the static shapes with 6 columns: `Matrix6Trait`, `Matrix2x6Trait`..`Matrix5x6Trait`,
  `RowVector6Trait` and their `...AngleTrait` forms (trigonometry, `exp`, `ln`...): constructors,
  component-wise operations, reductions, norms, interpolation, inverse and determinant of `Matrix6`,
  the base completion of upstream's `Matrix` / `Vector` API;
- `linalg::lu::lu6` (`Lu6`, `Lu6Trait`, `Matrix6LuTrait`: the LU factorisation of `Matrix6`, which
  `Matrix6::is_invertible` / `try_inverse` run) and `Perm6` (`linalg::lu`, the row permutation of a
  6x6 factorisation) with its `PermuteRows` / `PermuteColumns` impls on every dimension-6 shape.

The types themselves (`Matrix6`, `RowVector6`...) are in [`nalgebra_shapes6`](../shapes6/README.md);
this crate depends on it, on [`nalgebra_core`](../core/README.md),
[`nalgebra_shapes5`](../shapes5/README.md) and on the method crates below it (up to
[`nalgebra_static6_tall`](../static6_tall/README.md), whose edit kernels it calls). The other LU
factorisations (`Lu2`..`Lu4`) and the decompositions of dimension 6 are in the `linalg` crates.
Upstream's module paths are kept: `nalgebra_static6_wide::base::matrix6::Matrix6Trait` is
`nalgebra::base::matrix6::Matrix6Trait`, `nalgebra_static6_wide::linalg::lu::lu6::Lu6` is
`nalgebra::linalg::lu::lu6::Lu6` (docs/SPLIT.md §3.1).

## When to depend on it

Depend on `nalgebra_static6_wide` (and the crates it depends on) when you need the methods of 6x6
matrices (spatial inertia, 6-column Jacobians) or their LU factorisation without the rest of the
library. For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which
re-exports every sub-crate at the 0.1.0 paths.

## Features

`closures` (default): the methods of the shapes that take a closure (`map`, `fold`, `apply`,
`zip_*`...), as in `nalgebra` (whose feature of the same name forwards to this one).

## `internal`

`nalgebra_static6_wide::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_static6_wide` use (the crate-private kernels of `Lu6`). They are public only
for those crates: **internal, no stability promise**, they may change or disappear in any release,
and the facade never re-exports them.

## License

MIT.
