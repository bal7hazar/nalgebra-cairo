# nalgebra_core

The generic trait declarations (MatrixMul, MatrixTrMul, MatrixIndex, Norm, Normed / Unit, MatrixSolve, PermuteRows, GivensRotate, LuSteps, Householder / balancing, TransformMul), errors, fused kernels, and the dimension-1 types (Matrix1, Vector1, Point1, Translation1, Perm1, Reflection1 with its constructors and accessors).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::matrix_mul`, `base::matrix_tr_mul`, `base::matrix_index`, `base::solve`, `base::norm`, `base::unit`: the generic trait declarations (`MatrixMul`, `MatrixTrMul`, `MatrixIndex`, `MatrixSolve`, `Norm`, `Normed`, `Unit`); their impls sit with the types of each dimension (Cairo's impl lookup);
- `base::errors`, the fused kernels and `base::matrix1`: the dimension-1 shapes `Matrix1`, `Vector1`, `RowVector1`;
- `geometry::point`, `geometry::point1`, `geometry::translation1`, `geometry::reflection1`, `geometry::transform`: `Point1`, `Translation1`, `Reflection1` (with its constructors and accessors), and the declarations `TransformMul`, `TransformDiv`, `TransformSetCategory`;
- `linalg::lu`, `linalg::lu_steps`, `linalg::permutation_sequence`: `Perm1`, the declarations `PermuteRows`, `GivensRotate`, `LuSteps`, the Householder and balancing traits.

Upstream's module paths are kept: `nalgebra_core::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on `simba`.

## When to depend on it

It is the base of every closure (it costs 0.7 s / 0.03 GB over an empty consumer, the release commit's CI run, `docs/PACKAGES.md`).

Depend on `nalgebra_core` for the generic trait declarations and the dimension-1 types only; every other sub-crate depends on it.

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

The modules under `internal` (`internal::base`, `internal::geometry`, `internal::linalg`) hold items that are crate-private in `nalgebra` 0.1.0 and that the split had to make `pub` to share them across crates: no stability promise.

## License

MIT.
