# nalgebra_views

FixedView and its impls: fixed-size views of the static shapes (upstream base::matrix_view).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::matrix_view`: `FixedView` and its impls (`fixed_view`, `fixed_rows`, `fixed_columns` of any static shape, 441 impls).

Upstream's module paths are kept: `nalgebra_views::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_blocks`](../blocks/README.md) and `simba`.

## When to depend on it

`nalgebra_views` is a member of the declared closure `static4_views` (16.7 s / 4.06 GB, documented, not gated).

Advanced use: the generic `FixedView::fixed_view(m, i, j)` over any shape. `fixed_view` as a method of the shapes does not need this crate. Its dimension 5-6 impls can only sit here (a generic trait's impls live in its crate or in a type's), so "static 2-4 + views" is a dimension 5-6 closure (docs/SPLIT.md §18.4, §18.7).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
