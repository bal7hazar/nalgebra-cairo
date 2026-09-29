# nalgebra_blocks

Row / column blocks, resize, pad / crop and the Kronecker products of the static shapes of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::matrix_view`: `FixedRows`, `FixedColumns` (upstream `fixed_rows`, `rows`,
  `fixed_columns`, `columns`...: owned copies whose size is the output type) and `FixedResize`
  (`fixed_resize`, `resize`), with their impls for every pair of shapes (in the trait's module);
- `base::matrix_kronecker`: `MatrixKronecker` (`kronecker`, upstream `base/ops.rs`) and its 196
  impls.

Upstream's module paths are kept: `nalgebra_blocks::base::matrix_view::FixedRows` is
`nalgebra::base::matrix_view::FixedRows` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_shapes5`](../shapes5/README.md),
[`nalgebra_shapes6`](../shapes6/README.md) and [`nalgebra_static6_tall`](../static6_tall/README.md).

## When to depend on it

Depend on `nalgebra_blocks` (and the crates it depends on) when you need row / column blocks,
resizing or Kronecker products of the static shapes without the rest of the library. For upstream's
whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at
the 0.1.0 paths.

## `internal`

`nalgebra_blocks::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_blocks` use: `PadTo6`, `CropFrom6` and `ShapeDims` (padding a shape onto the
`Matrix6` canvas and back, used by `FixedView` in `nalgebra_views` and by the dynamic shapes), with
their impls. They are public only for those crates: **internal, no stability promise**, they may
change or disappear in any release, and the facade never re-exports them.

## License

MIT.
