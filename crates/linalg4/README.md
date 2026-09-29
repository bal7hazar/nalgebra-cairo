# nalgebra_linalg4

LU, Cholesky, LDLᵀ / UDU, Cholesky update and QR up to 4x4, with the Householder reflections,
of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `linalg::lu`: `Lu2`, `Lu3`, `Lu4` (`P·A = L·U`, partial pivoting) and `Matrix2LuTrait` ..
  `Matrix4LuTrait`; the permutations `Perm1` .. `Perm4` are in `nalgebra_core`, `Lu6` in
  `nalgebra_static6_wide`;
- `linalg::cholesky` (`Cholesky2` .. `Cholesky4`), `linalg::cholesky_update` (their rank-one
  updates, column insertion / removal), `linalg::udu` (`Udu2` .. `Udu4`), `linalg::qr` (`Qr` of
  every shape up to 4 rows and columns), `linalg::inverse` (`Matrix2InverseTrait` ..
  `Matrix4InverseTrait`);
- the building blocks `linalg::householder::reflection_axis_mut` and
  `linalg::householder_steps::{clear_column_unchecked, clear_row_unchecked, assemble_q}`.

Upstream's module paths are kept: `nalgebra_linalg4::linalg::lu::lu3::Lu3` is
`nalgebra::linalg::lu::lu3::Lu3` (docs/SPLIT.md §3.1). The dimension-5 / 6 decompositions are in
the facade `nalgebra` (until `nalgebra_linalg5` / `nalgebra_linalg6`), and so is
`linalg::lu_steps::try_invert_to`, whose `Matrix6` form runs `Lu6`. It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_static3`](../static3/README.md),
[`nalgebra_shapes5`](../shapes5/README.md) and [`nalgebra_static4`](../static4/README.md).

## When to depend on it

Depend on `nalgebra_core` + `nalgebra_static3` + `nalgebra_shapes5` + `nalgebra_static4` +
`nalgebra_linalg4` (the declared closure `static4_factor`) when you need the factorisations of the
shapes up to 4x4 (solving, inverting, least squares) without the rest of the library. For upstream's
whole API at upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at
the 0.1.0 paths.

## Features

All on by default, as in `nalgebra` (whose features of the same names forward to these): `qr`
(`linalg::qr`), `cholesky_update` (`linalg::cholesky_update`) and `hessenberg`
(`linalg::householder_steps`, grouped with the Hessenberg reductions in `nalgebra` 0.1.0).

## `internal`

`nalgebra_linalg4::internal` holds items that `nalgebra` 0.1.0 keeps crate-private but that the
crates above `nalgebra_linalg4` use: the Householder kernels of the reductions
(`HouseholderKernelTrait`), the `LDLᵀ` kernel of `udu` (`Ldlt2` .. `Ldlt4`) and the internal
traits of `Lu2` .. `Lu4` and `Qr2` .. `Qr4` (named by the in-crate tests of `nalgebra`). They are
public only for those crates: **internal, no stability promise**, they may change or disappear in
any release, and the facade never re-exports them.

## License

MIT.
