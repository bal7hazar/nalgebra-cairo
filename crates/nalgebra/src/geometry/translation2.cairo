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

// an explicit list: the conversions and products of the shapes into this module's type (0.1.0:
// `base::matrix*`, `base::row_vector*`) sit in this module of `nalgebra_static3`, the module of
// their geometry type (docs/SPLIT.md §3.2, §12.6)
pub use nalgebra_static3::geometry::translation2::{
    Similarity2FromTranslation2, Translation2, Translation2Div, Translation2DivAssign,
    Translation2FromArray, Translation2FromPoint, Translation2FromVector, Translation2Impl,
    Translation2IntoArray, Translation2Mul, Translation2MulAssign, Translation2One,
    Translation2Trait,
};
