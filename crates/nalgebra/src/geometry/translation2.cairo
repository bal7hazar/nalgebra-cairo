//! `Translation2`: a 2D translation (upstream `nalgebra::Translation2`, which is
//! `Translation<T, 2>`).
//!
//! - `Translation2Trait` / `Translation2Impl`: construction, accessors, composition, point
//!   transforms, the homogeneous matrix and comparison — everything a translation can do is
//!   algebraic, so the whole type lives over `simba::scalar::Real` and needs no `*AngleTrait`;
//! - `a * b` (composition) and the conversions from / to `Vector2<T>`: their impls live in this
//!   module, where the compiler finds them without any import.
//!
//! A translation acts on POINTS only: a `Vector2` is a displacement, which a translation leaves
//! unchanged (upstream has no `Translation::transform_vector` either). Every operation of this
//! type is an exact addition or negation: nothing here rounds, and the oracle tolerance of the
//! whole `translation` suite is 0.
//!
//! Numeric contract (AGENTS.md): additions and negations panic instead of wrapping; no sum of
//! products is formed here, so no fused kernel is needed.

pub use nalgebra_geometry2::geometry::similarity2::Similarity2FromTranslation2;
pub use nalgebra_geometry2::geometry::translation2::*;
pub use nalgebra_types2::geometry::translation2::*;
