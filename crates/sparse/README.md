# nalgebra_sparse

Upstream's legacy sparse matrices (`nalgebra::sparse`: `CsMatrix`, `CsCholesky`) and the Matrix
Market parser (`nalgebra::io`) of [nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo),
the Cairo port of the Rust `nalgebra` crate on `fixed::Fixed` (Q32.32), built for provable game
physics.

## What it holds

- `sparse`: `CsMatrix` / `CsVector` (compressed sparse column storage), its operators and its
  conversions to and from the static shapes and the dynamic matrices, `AxpyCs`, the triangular
  solves (`CsMatrixSolveTrait`), the sparse Cholesky factorisation `CsCholesky`, `cumsum`, and
  its error messages;
- `io`: `cs_matrix_from_matrix_market_str` (the Matrix Market coordinate format) and its error
  messages.

Upstream's module paths are kept: `nalgebra_sparse::sparse::CsMatrix` is
`nalgebra::sparse::CsMatrix` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), the types of every dimension
([`nalgebra_types2`](../types2/README.md), [`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md), [`nalgebra_types5`](../types5/README.md), [`nalgebra_types6`](../types6/README.md)) and
[`nalgebra_dynamic`](../dynamic/README.md) (which it pulls with the dynamic shapes it builds on).

## When to depend on it

Depend on `nalgebra_sparse` (and the crates it depends on) when you need compressed sparse column
matrices or to read Matrix Market files. For upstream's whole API at upstream's paths, depend on
the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0 paths (behind its features
`sparse` and `io`, on by default: `nalgebra` always depends on this crate, the features gate its
re-exports only).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
