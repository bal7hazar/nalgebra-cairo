# nalgebra_shapes6

The types of dimension 6 of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo
port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- the structs of the eleven shapes of dimension 6 (`Matrix6`, `Matrix2x6`..`Matrix5x6`,
  `Matrix6x2`..`Matrix6x5`, `Vector6`, `RowVector6`, with the aliases `Matrix6x1`, `Matrix1x6`,
  `UnitVector6`) and their core-trait impls (operators, conversions, `Serde`, `Sum`...);
- their products (`MatrixMul`, `MatrixTrMul`, with every shape; a product sits in the module of its
  larger operand, so `Matrix2x5 * Matrix5x6` is in `base::matrix5x6`), indexing (`MatrixIndex`),
  solve kernels, permutations (`Perm1`..`Perm5` on these shapes: `PermuteRows`, `PermuteColumns`),
  Givens rotations (`GivensRotate`), the LU steps of `Matrix6`, the `Normed` impl of `Vector6`;
- the reflections `Reflection1`..`Reflection4` applied to these shapes (`Reflection2Columns` on a
  `Matrix2x6`...) and `Matrix6CgTrait` (upstream `base/cg.rs` for `Matrix6`: homogeneous coordinates
  of dimension 5).

The traits they implement and the smaller types are in [`nalgebra_core`](../core/README.md) and
[`nalgebra_shapes5`](../shapes5/README.md), the reflections of dimension 1 to 4 in
[`nalgebra_geometry4`](../geometry4/README.md); this crate depends on those three and on
[`nalgebra_static3`](../static3/README.md). The METHODS of the dimension-6 shapes (`Matrix6Trait`,
`Vector6Trait`...), their edit kernels, `Perm6` and `Lu6` are in the crates above (`static6_tall`,
`static6_wide`). Upstream's module paths are kept: `nalgebra_shapes6::base::matrix6::Matrix6` is
`nalgebra::base::matrix6::Matrix6` (docs/SPLIT.md §3.1); the facade `nalgebra` keeps every impl at its
0.1.0 path.

## When to depend on it

Depend on `nalgebra_shapes6` when you need the dimension-6 types themselves (fields, operators,
products, conversions), typically as a dependency of `nalgebra_geometry6` (points and translations of
dimension 5 have 6x6 homogeneous matrices) or of the dimension-6 method crates. For upstream's whole
API at upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the
0.1.0 paths.

## `internal`

`nalgebra_shapes6::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_shapes6` use (`slerp_unit` of `Vector6`). They are public only for those
crates: **internal, no stability promise**, they may change or disappear in any release, and the
facade never re-exports them.

## License

MIT.
