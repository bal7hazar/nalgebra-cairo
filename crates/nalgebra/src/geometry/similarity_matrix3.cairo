//! `SimilarityMatrix3`: a 3D uniform scaling, followed by a rotation stored as a `Rotation3`
//! MATRIX, followed by a translation (upstream `nalgebra::SimilarityMatrix3`, which is
//! `Similarity<T, Rotation3<T>, 3>`).
//!
//! `sim * p = translation + scaling * (rotation * p)`. The rotation-matrix instance of upstream's
//! generic `Similarity` is a struct of its own, with the method set, field names and semantics of
//! `Similarity3` (whose rotation is a `UnitQuaternion`); the two convert into each other with
//! `.into()` (Shepperd's method one way, upstream's quaternion-to-matrix form the other). The
//! scaling factor must be nonzero: constructors and scaling mutators panic with `nalgebra: zero
//! scale` on zero (`similarity3::errors::ZERO_SCALING`).
//!
//! Numeric contract: the rotations go through the `Rotation3` fused kernels, the final
//! scale-plus-translation is one wide accumulation per component (`scale_translate`, shared with
//! `Similarity3`), preserving upstream's order (rotate, then scale, then translate).

pub use nalgebra_static3::geometry::similarity_matrix3::*;
