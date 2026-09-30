# nalgebra_linalg_svd_eigen6

SVD and symmetric eigen of dimension 6.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `linalg::svd`: SVD;
- `linalg::symmetric_eigen6`: symmetric eigen.

Upstream's module paths are kept: `nalgebra_linalg_svd_eigen6::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_linalg_core`](../linalg_core/README.md), [`nalgebra_linalg_svd_eigen2`](../linalg_svd_eigen2/README.md), [`nalgebra_linalg_svd_eigen3`](../linalg_svd_eigen3/README.md), [`nalgebra_linalg_svd_eigen4`](../linalg_svd_eigen4/README.md), [`nalgebra_linalg_svd_eigen5`](../linalg_svd_eigen5/README.md) and `simba`.

Features (all on by default, forwarded by the facade's features of the same name): `eigen`, `svd`. Each gates the module of that name; `nalgebra` with `default-features = false` leaves them out.

## When to depend on it

`nalgebra_linalg_svd_eigen6` is a member of the declared closures `static6_svd` (21.5 s / 4.70 GB, documented, not gated), `svd_eigen6` (12.1 s / 3.00 GB, 20 s / 4.5 GB).

Depend on the `linalg_svd_eigen` packages for the SVD and the symmetric eigen decomposition (`svd()`, `singular_values()`, `symmetric_eigen()`, `pseudo_inverse()`...).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths (behind the features of the same names, on by default and still real: they gate code inside this package).

## `internal`

The modules under `internal` (`internal::linalg`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
