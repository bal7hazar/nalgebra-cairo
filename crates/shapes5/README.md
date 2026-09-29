# nalgebra_shapes5

The types of dimension 5 of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo
port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- the structs of the nine shapes of dimension 5 (`Matrix5`, `Matrix2x5`, `Matrix3x5`, `Matrix4x5`,
  `Matrix5x2`, `Matrix5x3`, `Matrix5x4`, `Vector5`, `RowVector5`, with the aliases `Matrix5x1`,
  `Matrix1x5`, `UnitVector5`) and their core-trait impls (operators, conversions, `Serde`...);
- their products (`MatrixMul`, `MatrixTrMul`, with every shape up to dimension 5), indexing
  (`MatrixIndex`), solve kernels, permutations (`Perm5`, `PermuteRows`, `PermuteColumns`), Givens
  rotations (`GivensRotate`), the LU steps of `Matrix5` and the edit kernels of these shapes;
- `Point4Trait` and `Translation4Trait`, the methods of `Point4` / `Translation4` (they build
  dimension-5 types: `to_homogeneous`...).

The traits they implement and the types up to 4x4 are in [`nalgebra_core`](../core/README.md),
which this crate depends on (with `nalgebra_static3`). The METHODS of the dimension-5 shapes
(`Matrix5Trait`, `Vector5Trait`...) are in the crates above. Upstream's module paths are kept:
`nalgebra_shapes5::base::matrix5::Matrix5` is `nalgebra::base::matrix5::Matrix5`
(docs/SPLIT.md §3.1). A product sits in the module of its larger operand
(`Matrix2 * Matrix2x5` in `base::matrix2x5`); the facade `nalgebra` keeps it at its 0.1.0 path.

## When to depend on it

Depend on `nalgebra_shapes5` when you need the dimension-5 types themselves (fields, operators,
products, conversions), typically as a dependency of `nalgebra_static4` / `nalgebra_geometry4`
(homogeneous coordinates of dimension 4 are 5-vectors and 5x5 matrices). For upstream's whole API
at upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0
paths.

## `internal`

`nalgebra_shapes5::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_shapes5` use (the edit kernels of the shapes, `slerp_unit`). They are public
only for those crates: **internal, no stability promise**, they may change or disappear in any
release, and the facade never re-exports them.

## License

MIT.
