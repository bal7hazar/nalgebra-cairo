//! `SimilarityMatrix2`: a 2D uniform scaling, followed by a rotation stored as a `Rotation2`
//! MATRIX, followed by a translation (upstream `nalgebra::SimilarityMatrix2`, which is
//! `Similarity<T, Rotation2<T>, 2>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The rotation-matrix instance of upstream's
//! generic `Similarity` is a struct of its own, with the method set, field names and semantics of
//! `Similarity2` (whose rotation is a `UnitComplex`); the two convert into each other exactly with
//! `.into()`. The scaling factor must be nonzero: constructors and scaling mutators panic with
//! `nalgebra: zero scale` on zero (`similarity2::errors::ZERO_SCALING`).
//!
//! Numeric contract: the rotations go through the `Rotation2` fused kernels, the final
//! scale-plus-translation is one wide accumulation per component (`scale_translate`, shared with
//! `Similarity2`), preserving upstream's order (rotate, then scale, then translate).

pub use nalgebra_static3::geometry::similarity_matrix2::*;
