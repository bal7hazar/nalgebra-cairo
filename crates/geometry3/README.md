# nalgebra_geometry3

The 3D geometry: quaternions, unit quaternions, Rotation3 methods, isometries, similarities, dual
quaternions, AbstractRotation, point / translation / scale 3 methods, Matrix4 homogeneous (cg).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust
`nalgebra` crate on `fixed::Fixed` (Q32.32), split per dimension (docs/SPLIT.md §18: a number in a
crate name is exactly that dimension). Upstream's module paths are kept
(`nalgebra_geometry3::<module path>` is `nalgebra::<module path>`); most users depend on the facade
`nalgebra`, which re-exports every sub-crate at its 0.1.0 paths. Modules under `internal` hold items
that are crate-private in `nalgebra` 0.1.0: no stability promise.

Minimal README of the re-cut (WP 9-R1); the full one comes with move R3.
