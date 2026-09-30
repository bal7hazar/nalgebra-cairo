# nalgebra_norm

The norm markers (EuclideanNorm, LpNorm, OneNorm, UniformNorm) and every Norm impl (upstream base::norm).

Part of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics, split per dimension
(`docs/SPLIT.md` §18: a number in a crate name is exactly that dimension; the dimension of a rectangular
shape is max(rows, columns)).

## What it holds

- `base::norm`: the norm markers `EuclideanNorm`, `LpNorm`, `OneNorm`, `UniformNorm` and every `Norm` impl of the static shapes.

Upstream's module paths are kept: `nalgebra_norm::<module path>` is `nalgebra::<module path>` (docs/SPLIT.md §3.1).
It depends on [`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md), [`nalgebra_static_core`](../static_core/README.md), [`nalgebra_static2`](../static2/README.md), [`nalgebra_static3`](../static3/README.md), [`nalgebra_static4`](../static4/README.md), [`nalgebra_static5`](../static5/README.md), [`nalgebra_static6_tall`](../static6_tall/README.md), [`nalgebra_static6_wide`](../static6_wide/README.md) and `simba`.

## When to depend on it

`nalgebra_norm` is a member of the declared closure `static4_norm` (18.7 s / 4.04 GB, documented, not gated).

Advanced use: the `Norm` markers and generic impls. The everyday `norm()`, `normalize()`, `apply_norm`, `lp_norm` are methods of the `static` crates. This crate depends on every method crate, so "static 2-4 + norm" is a dimension 5-6 closure (docs/SPLIT.md §18.4, §18.7).

For upstream's whole API at upstream's paths, depend on the facade [`nalgebra`](../../README.md), which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the crates above it.

## License

MIT.
