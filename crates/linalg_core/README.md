# nalgebra_linalg_core

Shared linalg kernels: the Householder kernels and steps (`clear_column_unchecked`, `assemble_q`), `reflection_axis_mut`, balancing, the SVD filters, the `exp` Padé message.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `linalg::balancing`, `linalg::householder`, `linalg::householder_steps`: the Householder kernels and steps (`clear_column_unchecked`, `assemble_q`), `reflection_axis_mut`, balancing, the SVD filters (`SvdRightTrait`), the `exp` Padé message.

Upstream's module paths are kept: `nalgebra_linalg_core::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md) and `simba`.

Features (all on by default, forwarded by the facade's features of the same name): `svd`, `hessenberg`, `exp`. Each gates the module of that name; `nalgebra` with `default-features = false` leaves them out.

## When to depend on it

It is pulled in by 7 declared closures, the cheapest being `static3_svd` (5.8 s / 1.23 GB, 15 s / 3 GB).

Depend on it only as a building block: it holds the kernels the `linalg_*` packages share (every decomposition package of a dimension above 1 that uses them depends on it).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind the features of the same names, on by default and still real: they gate code inside this package).

## `internal`

The modules under `internal` (`internal::linalg`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
