# nalgebra_types4

The types of dimension 4 (shapes with max(rows, cols) = 4, their core-trait impls, products,
indexing, solve / permutation / Givens / LU-step / Householder impls); Point4, Translation4, Perm4,
Reflection4.

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust
`nalgebra` crate on `fixed::Fixed` (Q32.32), split per dimension (docs/SPLIT.md §18: a number in a
crate name is exactly that dimension). Upstream's module paths are kept
(`nalgebra_types4::<module path>` is `nalgebra::<module path>`); most users depend on the facade `nalgebra`,
which re-exports every sub-crate at its 0.1.0 paths. Modules under `internal` hold items that are
crate-private in `nalgebra` 0.1.0: no stability promise.

Minimal README of the re-cut (WP 9-R1); the full one comes with move R3.
