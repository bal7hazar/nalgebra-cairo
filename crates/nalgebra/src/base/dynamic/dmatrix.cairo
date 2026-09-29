//! `DMatrix<T>`: a matrix whose dimensions are runtime values (upstream `DMatrix`, `OMatrix<T,
//! Dyn, Dyn>`), and the ten partially dynamic aliases `Matrix2xX..Matrix6xX`,
//! `MatrixXx2..MatrixXx6` (DESIGN D5).
//!
//! Storage: the components in COLUMN-major order (upstream's `VecStorage`) in one `Span<T>`, with
//! the two dimensions. A `Span` is a view of a write-once array, so `DMatrix` is `Copy` (upstream
//! `Clone`, like every Cairo value type here) and every operation builds a new span; the `_mut`
//! forms (`resize_mut`...) take `ref self` and replace it. Loops are allowed (the sizes are
//! runtime values); every sum of products is ONE exact accumulation floored once
//! (`Real::Wide`), and the products of square matrices up to 6x6 dispatch to the static kernels
//! (bit-identical, measured cheaper).

pub use nalgebra_dynamic::base::dynamic::dmatrix::*;
