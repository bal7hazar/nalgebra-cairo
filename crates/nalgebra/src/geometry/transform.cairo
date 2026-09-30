//! Upstream's `Transform<T, C: TCategory, D>` (`geometry/transform*.rs`), WP 8.4-P11a: the shared
//! part of the six transform types, i.e. the panic messages, the generic operator traits whose
//! output category depends on BOTH operands, and the crate-internal kernels.
//!
//! **Design.** Upstream has one struct generic over the dimension `D` (2 or 3 through its
//! aliases) and a phantom category `C` (`TGeneral`, `TProjective`, `TAffine`), and six aliases:
//! `Transform2/3` (`TGeneral`), `Projective2/3` (`TProjective`), `Affine2/3` (`TAffine`). Cairo has
//! no const generics and a phantom category would not select the methods of a category, so each
//! alias is a Cairo struct of its own (`transform2.cairo` ... `affine3.cairo`), wrapping the
//! homogeneous `Matrix3` / `Matrix4` in a private `matrix` field (upstream's field is private
//! too: read it with `matrix` / `into_inner`, build with `from_matrix_unchecked`). Every type has
//! upstream's method names and semantics, and upstream's category rules are kept exactly:
//! - `try_inverse` / `try_inverse_mut` everywhere; `inverse`, `inverse_mut`,
//!   `inverse_transform_point`, `inverse_transform_vector` and the division BY a transform only
//!   for the projective and affine categories (upstream's `C: SubTCategoryOf<TProjective>`);
//! - the category of a product follows upstream's `TCategoryMul` (`TGeneral` absorbs everything,
//!   then `TProjective`; `TAffine * TAffine` is affine). Same-category products are the `*` / `/`
//!   operators; every pair of categories is `TransformMul::mul_transform` /
//!   `TransformDiv::div_transform`, which also carry the products of the other geometry types by
//!   a transform (`Isometry3 * Affine3`...). The products by a rotation, a translation, an
//!   isometry or a similarity keep the category of the transform (upstream's
//!   `TCategoryMul<TAffine>`) and are the `mul_<rhs>` / `div_<rhs>` methods of each type;
//! - `set_category` (`TransformSetCategory`) and `Into` widen a category without check
//!   (upstream's `SuperTCategoryOf`); `TryInto` narrows it with upstream's
//!   `check_homogeneous_invariants` (`is_in_subset`), from a transform or from a raw matrix;
//! - the affine category has no normalizer: its `transform_point` / `transform_vector` ignore the
//!   last row, like upstream's `TAffine::has_normalizer() == false`; the general and projective
//!   ones divide by the homogeneous coordinate (`Matrix3/4::transform_point`, upstream's formula).
//!
//! **Numerics** (AGENTS.md). Every sum of products is ONE fused `Real` kernel (one floor per
//! output scalar). The products by a rotation, a translation, an isometry or a similarity skip
//! the exact zeros and ones of its homogeneous matrix, which gives bit for bit the full product
//! for less gas; the products of two transforms are full matrix products, like upstream. The
//! general and projective categories invert the whole homogeneous matrix (`Matrix3/4::
//! try_inverse`, upstream's formula); the affine one inverts by blocks with one step of iterative
//! refinement of the translation, which is cheaper, keeps the affine last row exact and is the
//! only measured candidate within the oracle tolerance on every distribution (see
//! `Affine3::try_inverse`). Overflow panics.

pub use nalgebra_core::geometry::transform::*;
pub use nalgebra_transform2::geometry::affine2::{
    Affine2DivAffine2, Affine2DivProjective2, Affine2MulAffine2, Affine2MulProjective2,
    Affine2MulTransform2, Affine2SetCategoryAffine2, Affine2SetCategoryProjective2,
    Affine2SetCategoryTransform2, Isometry2MulAffine2, IsometryMatrix2MulAffine2,
    Projective2DivAffine2, Projective2MulAffine2, Rotation2DivAffine2, Rotation2MulAffine2,
    Similarity2MulAffine2, SimilarityMatrix2MulAffine2, Transform2DivAffine2, Transform2MulAffine2,
    Translation2DivAffine2, Translation2MulAffine2, UnitComplexMulAffine2,
};
pub use nalgebra_transform2::geometry::projective2::{
    Isometry2MulProjective2, IsometryMatrix2MulProjective2, Projective2DivProjective2,
    Projective2MulProjective2, Projective2MulTransform2, Projective2SetCategoryProjective2,
    Projective2SetCategoryTransform2, Rotation2DivProjective2, Rotation2MulProjective2,
    Similarity2MulProjective2, SimilarityMatrix2MulProjective2, Transform2DivProjective2,
    Transform2MulProjective2, Translation2DivProjective2, Translation2MulProjective2,
    UnitComplexMulProjective2,
};
pub use nalgebra_transform2::geometry::transform2::{
    Isometry2MulTransform2, IsometryMatrix2MulTransform2, Rotation2DivTransform2,
    Rotation2MulTransform2, Similarity2MulTransform2, SimilarityMatrix2MulTransform2,
    Transform2MulTransform2, Transform2SetCategoryTransform2, Translation2DivTransform2,
    Translation2MulTransform2, UnitComplexMulTransform2,
};
pub use nalgebra_transform3::geometry::affine3::{
    Affine3DivAffine3, Affine3DivProjective3, Affine3MulAffine3, Affine3MulProjective3,
    Affine3MulTransform3, Affine3SetCategoryAffine3, Affine3SetCategoryProjective3,
    Affine3SetCategoryTransform3, Isometry3MulAffine3, IsometryMatrix3MulAffine3,
    Projective3DivAffine3, Projective3MulAffine3, Rotation3DivAffine3, Rotation3MulAffine3,
    Similarity3MulAffine3, SimilarityMatrix3MulAffine3, Transform3DivAffine3, Transform3MulAffine3,
    Translation3DivAffine3, Translation3MulAffine3, UnitQuaternionDivAffine3,
    UnitQuaternionMulAffine3,
};
pub use nalgebra_transform3::geometry::projective3::{
    Isometry3MulProjective3, IsometryMatrix3MulProjective3, Projective3DivProjective3,
    Projective3MulProjective3, Projective3MulTransform3, Projective3SetCategoryProjective3,
    Projective3SetCategoryTransform3, Rotation3DivProjective3, Rotation3MulProjective3,
    Similarity3MulProjective3, SimilarityMatrix3MulProjective3, Transform3DivProjective3,
    Transform3MulProjective3, Translation3DivProjective3, Translation3MulProjective3,
    UnitQuaternionDivProjective3, UnitQuaternionMulProjective3,
};
pub use nalgebra_transform3::geometry::transform3::{
    Isometry3MulTransform3, IsometryMatrix3MulTransform3, Rotation3DivTransform3,
    Rotation3MulTransform3, Similarity3MulTransform3, SimilarityMatrix3MulTransform3,
    Transform3MulTransform3, Transform3SetCategoryTransform3, Translation3DivTransform3,
    Translation3MulTransform3, UnitQuaternionDivTransform3, UnitQuaternionMulTransform3,
};
