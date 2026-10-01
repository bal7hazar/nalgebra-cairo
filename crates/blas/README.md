# nalgebra_blas

The BLAS-like kernels of the static shapes (upstream `nalgebra::base::blas`) of
[nalgebra-cairo](https://github.com/bal7hazar/nalgebra-cairo), the Cairo port of the Rust `nalgebra`
crate on `fixed::Fixed` (Q32.32), built for provable game physics.

## What it holds

- `base::blas`: one `...BlasTrait` per static shape (`dotc`, `tr_dot`, `axpy`, `axcpy`, `gemv`,
  `gemv_tr`, `sygemv`, `ger`, `syger`...) and the generic `MatrixGemm`, `MatrixGemmTr`, `MatrixGemv`,
  `MatrixGemvTr`, `MatrixQuadform`, `MatrixQuadformTr` with their impls.

Upstream's module paths are kept: `nalgebra_blas::base::blas::MatrixGemm` is
`nalgebra::base::blas::MatrixGemm` (docs/SPLIT.md §3.1). It depends on
[`nalgebra_core`](../core/README.md), [`nalgebra_types2`](../types2/README.md),
[`nalgebra_types3`](../types3/README.md), [`nalgebra_types4`](../types4/README.md),
[`nalgebra_types5`](../types5/README.md) and [`nalgebra_types6`](../types6/README.md).

## When to depend on it

Depend on `nalgebra_blas` (and the crates it depends on) when you need `gemm`, `gemv`, `axpy` or the
rank updates of the static shapes without the rest of the library. It belongs to the declared
closure `static4_blas` (static 2-4 + blas: 12.4 s / 2.88 GB on a GitHub runner over an empty
consumer, budget 15 s / 3 GB); its own marginal cost over its direct dependencies is 2.3 s / 0.36 GB
(12,105 lines). For upstream's whole API at
upstream's paths, depend on the facade `nalgebra`, which re-exports every sub-crate at the 0.1.0
paths (behind its feature `blas`, on by default: `nalgebra` always depends on this crate, the
feature gates its re-exports only).

## `internal`

This crate has no `internal` module: it shares no crate-private item of `nalgebra` 0.1.0 with the
crates above it.

## License

MIT.
