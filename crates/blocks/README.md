# nalgebra_blocks

Row / column blocks, resize, pad / crop, Kronecker products, the fixed-size edition of the static shapes (upstream base::edition, base::blocks).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::edition`, `base::blocks`: row / column blocks, resizing, padding / cropping (the `Fixed*` block traits, `resize`, `pad`, `crop`...), and `base::matrix_kronecker`: Kronecker products of the static shapes;
- `base::dynamic::shapes` (its part in this crate): the fixed-size edition of the static shapes, `InsertFixedColumns`, `InsertFixedRows`, `RemoveFixedColumns`, `RemoveFixedRows` with their 360 impls (`insert_fixed_columns`, `remove_fixed_rows`...). They run on this crate's `Matrix6` canvas and name no dynamic type, so they moved here from `nalgebra_dynamic` (WP 9-NS12b); the facade re-exports both parts at the 0.1.0 path.

Upstream's module paths are kept: `nalgebra_blocks::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static6_tall`](../static6_tall/README.md) and `simba`.

## When to depend on it

`nalgebra_blocks` is a member of the declared closure `static4_blocks` (17.4 s / 3.87 GB, documented, not gated). It is also pulled in by `static4_views` (17.9 s / 4.18 GB, documented, not gated).

Advanced use: the generic block traits and the Kronecker products. The everyday methods (`insert_*`, `remove_*`, `fixed_rows`, `fixed_columns`) do not need this crate; it pulls `nalgebra_static6_tall` for the dimension-6 edit kernels and holds the fixed-size edition (`insert_fixed_rows`...), so "static 2-4 + blocks" is a dimension 5-6 closure (docs/SPLIT.md §18.4, §18.7).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
