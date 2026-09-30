//! `Unit<V>`: a wrapper guaranteeing (by contract) that a vector has unit norm (upstream
//! `nalgebra::Unit`).
//!
//! - `Normed<V, T>`: the few operations `Unit` needs from a vector type `V` over the scalar
//!   `T` (upstream `Normed`), implemented for the column vectors `Vector1<T>` to `Vector6<T>`;
//! - `UnitTrait` / `UnitImpl`: construction (`new_normalize`, `try_new`, `new_unchecked`, ...),
//!   in-place renormalization (upstream's `&mut self` methods) and the products upstream reaches
//!   through `Deref`, generic over any `Normed` vector;
//! - `Unit2Trait` / `Unit3Trait` / `Unit4Trait`: the axes (`x_axis`, ...);
//! - `-u` (exact, a negated unit vector is a unit vector).
//!
//! The `value` field is public: unlike upstream there is no `Deref`, so vector operations are
//! reached through `u.value` (`u.value.cross(v)`), and `new_unchecked` is just `Unit { value }`.
//! Unit vectors built by `new_normalize` have a norm of `1` within a few ulp, not exactly `1`
//! (fixed point cannot do better); `renormalize` / `renormalize_fast` bring them back.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;

pub use nalgebra_core::base::unit::*;

// the crate-private by-value helpers the in-crate tests use (`nalgebra_core::internal`)
#[cfg(test)]
use nalgebra_core::internal::base::unit::UnitInternalTrait;
pub use nalgebra_static2::base::unit::*;
pub use nalgebra_static3::base::unit::*;
pub use nalgebra_static4::base::unit::*;

pub use nalgebra_types2::base::vector2::Vector2Normed;
pub use nalgebra_types3::base::vector3::Vector3Normed;
pub use nalgebra_types4::base::vector4::Vector4Normed;
// `Normed` of `Vector5` / `Vector6`: in the modules of their types (docs/SPLIT.md §3.2)
pub use nalgebra_types5::base::vector5::Vector5Normed;
pub use nalgebra_types6::base::vector6::Vector6Normed;
