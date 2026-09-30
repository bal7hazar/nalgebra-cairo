# nalgebra_statistics6

base::statistics of the dimension-6 shapes (one inherent trait per shape).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::statistics` of the shapes of dimension 6: one inherent trait per shape (`Matrix6StatisticsTrait`...) with `sum`, `product`, `mean`, `variance`, `column_mean`, `row_variance`, `min`, `max`...

Upstream's module paths are kept: `nalgebra_statistics6::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md) and `simba`.

## When to depend on it

`nalgebra_statistics6` is a member of the declared closure `static4_statistics_all` (11.9 s / 2.79 GB, 15 s / 3 GB).

Depend on it for the statistics (`mean`, `variance`...) of the dimension-6 shapes; "static 2-4 + statistics 2-4" is a small closure, the family whole (2-6) costs 11.9 s / 2.79 GB.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind its feature `statistics`, on by default and a no-op since the split).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
