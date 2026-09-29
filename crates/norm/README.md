# nalgebra_norm

The norm markers and every `Norm` impl of the static shapes of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::norm`: the markers `EuclideanNorm`, `LpNorm`, `OneNorm`, `UniformNorm` (upstream
  `base/norm.rs`) and the 144 `Norm<Marker, Shape, T>` impls (in the marker's module), which
  `apply_norm` / `apply_metric_distance` dispatch to. The `Norm` trait itself is declared in
  `nalgebra_core`.

Upstream's module paths are kept: `nalgebra_norm::base::norm::EuclideanNorm` is
`nalgebra::base::norm::EuclideanNorm` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md),
[`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md),
[`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md),
[`nalgebra_static_core`](../static_core/README.md), [`nalgebra_static2`](../static2/README.md),
[`nalgebra_static3`](../static3/README.md), [`nalgebra_static4`](../static4/README.md),
[`nalgebra_static5`](../static5/README.md), [`nalgebra_static6_tall`](../static6_tall/README.md) and
[`nalgebra_static6_wide`](../static6_wide/README.md) (the impls call the shapes' own `norm`,
`lp_norm`, `one_norm` and `amax`: this crate sits above every method crate of the static shapes).

## When to depend on it

Depend on `nalgebra_norm` (and the crates it depends on) when you pass a norm marker to `apply_norm`
/ `apply_metric_distance` without the rest of the library. For upstream's whole API at upstream's
paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0 paths.

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
