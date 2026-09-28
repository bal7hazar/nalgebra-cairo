# nalgebra_core

The lowest sub-crate of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo
port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- the structs of the 36 static shapes (`Matrix1`..`Matrix6`, `MatrixRxC`, `Vector2`..`Vector6`,
  `RowVector2`..`RowVector6`) and their core-trait impls (operators, conversions, `Serde`...);
- the products (`MatrixMul`, `MatrixTrMul`), indexing (`MatrixIndex`), solve kernels, permutations
  (`Perm2`..`Perm4`, `PermuteRows`, `PermuteColumns`) and Givens rotations (`GivensRotation`,
  `GivensRotate`) of the shapes up to 4x4, and the LU steps (`gauss_step`, `gauss_step_swap`);
- the declarations of the generic traits every other crate implements (`MatrixMul`, `MatrixTrMul`,
  `MatrixIndex`, `Norm`, `MatrixSolve`, `PermuteRows`, `GivensRotate`...);
- `Point1`..`Point4` (`Point2` / `Point3` in `base`), `Translation1`, `Translation4`, `Unit` and
  `Normed`, the panic messages (`base::errors`) and the fused kernels.

Upstream's module paths are kept: `nalgebra_core::base::matrix3::Matrix3` is
`nalgebra::base::matrix3::Matrix3`. The METHODS of the shapes (`Matrix3Trait`...) live in the
crates above (`nalgebra_static3`, `nalgebra_static4`...), as do the geometry, the decompositions
and the other families of upstream's `base` (docs/SPLIT.md §3.1).

## When to depend on it

Depend on `nalgebra_core` when you need the types themselves (fields, operators, products,
conversions) without the rest of the library: it is the cheapest build of the family. For the
methods, add the crate of the dimension you use; for upstream's whole API at upstream's paths,
depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

`nalgebra_core::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_core` use (kernels, crate-private traits such as `SolveKernel`). They are
public only for those crates: **internal, no stability promise**, they may change or disappear in
any release, and the facade never re-exports them.

## License

MIT.
