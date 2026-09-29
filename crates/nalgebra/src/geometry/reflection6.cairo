//! `Reflection6`: a reflection with respect to a hyperplane of the 6-dimensional space (upstream
//! `nalgebra::Reflection6`, i.e. `Reflection<T, Const<6>, ArrayStorage<T, 6, 1>>`), WP 8.4-P10.
//!
//! The hyperplane is orthogonal to `axis` (a unit vector) and lies at `bias` along it: the points
//! `p` with `axis . p = bias`. Written from one template for the sizes 1 to 6.
//!
//! Upstream's `reflect` / `reflect_rows` are generic over the shape of the matrix they update in
//! place (`&mut Matrix<T, R2, C2, S2>`); the Cairo counterparts are the generic traits
//! `Reflection6Columns<M, T>` (`reflect`, `reflect_with_sign`: `M` is any of the six shapes with 6
//! rows, `Vector6` to `Matrix6`) and `Reflection6Rows<L, W, T>` (`reflect_rows`,
//! `reflect_rows_with_sign`: `L` is any of the six shapes with 6 columns and `W` the scratch vector
//! of its row count), both taking the matrix by `ref` like Rust's `&mut`. The scratch vector `work`
//! of `reflect_rows` holds `lhs * axis - bias` on return, as upstream's does.
//!
//! Numeric contract (AGENTS.md): every dot product is accumulated exactly and rounded ONCE, the
//! factor `-2 (axis . x - bias)` is rounded once (`wide_mul_scalar`), and every updated entry is
//! one fused `mul_add` / `sum_prod2` (one floor rounding and one overflow check). Overflow panics;
//! nothing wraps silently. Unlike upstream, the bias is subtracted unconditionally (upstream skips
//! it when zero: subtracting zero is exact, so the results are identical).

pub use nalgebra_geometry6::geometry::reflection6::*;
