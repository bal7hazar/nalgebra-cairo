//! `Reflection1`: a reflection with respect to a hyperplane of the 1-dimensional space (upstream
//! `nalgebra::Reflection1`, i.e. `Reflection<T, Const<1>, ArrayStorage<T, 1, 1>>`), WP 8.4-P10.
//!
//! The hyperplane is orthogonal to `axis` (a unit vector) and lies at `bias` along it: the points
//! `p` with `axis . p = bias`. Written from one template for the sizes 1 to 6.
//!
//! Upstream's `reflect` / `reflect_rows` are generic over the shape of the matrix they update in
//! place (`&mut Matrix<T, R2, C2, S2>`); the Cairo counterparts are the generic traits
//! `Reflection1Columns<M, T>` (`reflect`, `reflect_with_sign`: `M` is any of the six shapes with 1
//! rows, `Matrix1` to `RowVector6`) and `Reflection1Rows<L, W, T>` (`reflect_rows`,
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 1 columns and `W` the scratch vector
//! of its row count), both taking the matrix by `ref` like Rust's `&mut`. The scratch vector `work`
//! of `reflect_rows` holds `lhs * axis - bias` on return, as upstream's does.
//!
//! Numeric contract (AGENTS.md): every dot product is accumulated exactly and rounded ONCE, the
//! factor `-2 (axis . x - bias)` is rounded once (`wide_mul_scalar`), and every updated entry is
//! one fused `mul_add` / `sum_prod2` (one floor rounding and one overflow check). Overflow panics;
//! nothing wraps silently. Unlike upstream, the bias is subtracted unconditionally (upstream skips
//! it when zero: subtracting zero is exact, so the results are identical).



pub use nalgebra_core::geometry::reflection1::*;

pub use nalgebra_types2::base::row_vector2::Reflection1ColumnsRowVector2;
pub use nalgebra_types2::base::vector2::Reflection1RowsVector2;
pub use nalgebra_types3::base::row_vector3::Reflection1ColumnsRowVector3;
pub use nalgebra_types3::base::vector3::Reflection1RowsVector3;
pub use nalgebra_types4::base::row_vector4::Reflection1ColumnsRowVector4;
pub use nalgebra_types4::base::vector4::Reflection1RowsVector4;
pub use nalgebra_types5::base::row_vector5::Reflection1ColumnsRowVector5;
pub use nalgebra_types5::base::vector5::Reflection1RowsVector5;
pub use nalgebra_types6::base::row_vector6::Reflection1ColumnsRowVector6;
pub use nalgebra_types6::base::vector6::Reflection1RowsVector6;
