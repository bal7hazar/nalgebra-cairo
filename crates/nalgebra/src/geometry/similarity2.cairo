//! `Similarity2`: a 2D uniform scaling, followed by a rotation, followed by a translation
//! (upstream `nalgebra::Similarity2`, which is `Similarity<T, UnitComplex<T>, 2>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The scaling factor must be nonzero:
//! constructors and scaling mutators panic with `nalgebra: zero scale` on zero.
//!
//! Numeric contract: rotations go through the `UnitComplex` fused kernels; the final
//! scale-plus-translation is fused into one wide accumulation per component. This preserves
//! upstream's fixed-point-observable order (`rotate`, then `scale`, then `translate`) while
//! avoiding a checked scalar addition after the scale.

// an explicit list: the conversions and products of the shapes into this module's type (0.1.0:
// `base::matrix*`, `base::row_vector*`) sit in this module of `nalgebra_static3`, the module of
// their geometry type (docs/SPLIT.md §3.2, §12.6)
pub use nalgebra_static3::geometry::similarity2::{
    Similarity2, Similarity2AngleImpl, Similarity2AngleTrait, Similarity2Default, Similarity2Div,
    Similarity2DivAssign, Similarity2DivAssignIsometry2, Similarity2Impl, Similarity2Mul,
    Similarity2MulAssign, Similarity2MulAssignIsometry2, Similarity2MulAssignTranslation2,
    Similarity2One, Similarity2Trait, errors,
};
