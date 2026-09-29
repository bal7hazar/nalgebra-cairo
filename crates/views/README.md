# nalgebra_views

`FixedView`, the owned fixed-size views of the static shapes, of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::matrix_view`: `FixedView` (upstream `fixed_view`, `view`...: an owned copy whose size is the
  output type) and its 441 impls (one per pair of source and output shapes, in the trait's module),
  with the runtime-sized helpers `RowPart` and `ColumnPart`.

Upstream's module paths are kept: `nalgebra_views::base::matrix_view::FixedView` is
`nalgebra::base::matrix_view::FixedView` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_blocks`](../blocks/README.md), [`nalgebra_core`](../core/README.md),
[`nalgebra_shapes5`](../shapes5/README.md) and [`nalgebra_shapes6`](../shapes6/README.md).

## When to depend on it

Depend on `nalgebra_views` (and the crates it depends on) when you need `fixed_view` / `view` on the
static shapes without the rest of the library. For upstream's whole API at upstream's paths, depend
on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
