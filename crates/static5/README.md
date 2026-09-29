# nalgebra_static5

The methods of every dimension-5 shape of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- the METHODS of the static shapes of dimension 5: `Matrix5Trait`, `Matrix5x2Trait`..`Matrix5x4Trait`,
  `Matrix2x5Trait`..`Matrix4x5Trait`, `Vector5Trait`, `RowVector5Trait`, `UnitVector5Trait` and their
  `...AngleTrait` forms (trigonometry, `exp`, `ln`...): constructors, component-wise operations,
  reductions, norms, interpolation, inverse and determinant of `Matrix5`, the base completion of
  upstream's `Matrix` / `Vector` API.

The types themselves (`Matrix5`, `Vector5`...) are in [`nalgebra_shapes5`](../shapes5/README.md),
those of dimension 6 (which some methods return: `insert_row`, `to_homogeneous`...) in
[`nalgebra_shapes6`](../shapes6/README.md); this crate depends on both, on
[`nalgebra_core`](../core/README.md), [`nalgebra_static3`](../static3/README.md) and
[`nalgebra_static4`](../static4/README.md). Upstream's module paths are kept:
`nalgebra_static5::base::matrix5::Matrix5Trait` is `nalgebra::base::matrix5::Matrix5Trait`
(docs/SPLIT.md §3.1).

## When to depend on it

Depend on `nalgebra_core` + `nalgebra_shapes5` + `nalgebra_static5` (and the crates it depends on)
when you need the methods of the 5-dimensional shapes without the rest of the library. For upstream's
whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at
the 0.1.0 paths.

## Features

`closures` (default): the methods of the shapes that take a closure (`map`, `fold`, `apply`,
`zip_*`...), as in `nalgebra` (whose feature of the same name forwards to this one).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
