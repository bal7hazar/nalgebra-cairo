# nalgebra_statistics

The statistics of the static shapes (upstream `nalgebra::base::statistics`) of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::statistics`: one `...StatisticsTrait` per static shape (`Matrix3StatisticsTrait`...):
  `sum`, `product`, `mean`, `variance` of all the components, of each column (`row_*`) and of each
  row (`column_*`), min / max over rows and columns.

Upstream's module paths are kept: `nalgebra_statistics::base::statistics::Matrix3StatisticsTrait` is
`nalgebra::base::statistics::Matrix3StatisticsTrait` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_shapes5`](../shapes5/README.md) and
[`nalgebra_shapes6`](../shapes6/README.md).

## When to depend on it

Depend on `nalgebra_statistics` (and the crates it depends on) when you need the statistics of the
static shapes without the rest of the library. For upstream's whole API at upstream's paths, depend
on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0 paths (behind its feature
`statistics`, on by default: `nalgebra` always depends on this crate, the feature gates its
re-exports only).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
