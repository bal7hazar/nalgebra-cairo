# nalgebra_geometry6

The points, translations, scales and reflections of dimension 5 and 6 of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- upstream's `geometry` types of dimension 5 and 6: `Point5` / `Point6`, `Translation5` /
  `Translation6`, the non-uniform scales `Scale5` / `Scale6` and the reflections `Reflection5` /
  `Reflection6` (with their `Reflection5Columns` / `Reflection5Rows`... traits), each with its methods
  (`Point5Trait`, `Translation6Trait`, `to_homogeneous`...) and core-trait impls;
- the swizzles of `Point5` / `Point6` (`Point5SwizzleTrait`, `Point6SwizzleTrait`, upstream
  `base/swizzles.rs`) and `translation5.into()` (`Matrix6FromTranslation5`).

The vectors and matrices they are built on are in [`nalgebra_core`](../core/README.md),
[`nalgebra_shapes5`](../shapes5/README.md) and [`nalgebra_shapes6`](../shapes6/README.md): this crate
depends on those three only. Upstream's module paths are kept:
`nalgebra_geometry6::geometry::point5::Point5` is `nalgebra::geometry::point5::Point5`
(docs/SPLIT.md §3.1). The points and translations of dimension 1 to 4 are in `nalgebra_core` /
`nalgebra_static3` / `nalgebra_shapes5`, the scales and reflections of dimension 1 to 4 in
`nalgebra_geometry4`.

## When to depend on it

`Point6` / `Translation5` / `Reflection6` → `nalgebra_core` + `nalgebra_static3` + `nalgebra_shapes5`
+ `nalgebra_static4` + `nalgebra_geometry4` + `nalgebra_shapes6` + `nalgebra_geometry6` (the
transitive closure of its dependencies).

For upstream's whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports every
sub-crate at the 0.1.0 paths.

## License

MIT.
