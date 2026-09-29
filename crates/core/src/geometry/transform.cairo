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

/// `lhs * rhs` when the category of the product depends on both operands (upstream's `Mul` impls
/// with `TCategoryMul`): transform by transform, and isometry / similarity / rotation /
/// translation / unit complex / unit quaternion by transform. Upstream: `Mul<Transform>`.
pub trait TransformMul<Lhs, Rhs> {
    /// The transform type of the product.
    type Output;
    /// `self * rhs`.
    fn mul_transform(self: Lhs, rhs: Rhs) -> Self::Output;
}

/// `lhs / rhs` when the category of the quotient depends on both operands. Upstream:
/// `Div<Transform>`.
pub trait TransformDiv<Lhs, Rhs> {
    /// The transform type of the quotient.
    type Output;
    /// `self / rhs`.
    fn div_transform(self: Lhs, rhs: Rhs) -> Self::Output;
}

/// `t.set_category()`: the same matrix in a super-category (the target type is inferred, e.g.
/// `let p: Projective3<Fixed> = a.set_category();`). Upstream: `Transform::set_category::<CNew>`
/// with `CNew: SuperTCategoryOf<C>`.
pub trait TransformSetCategory<From, To> {
    /// The same matrix, unchecked. Exact.
    fn set_category(self: From) -> To;
}

/// Panic messages of the transform types (stable API).
pub mod errors {
    /// `inverse`, `inverse_transform_*` and the divisions by a transform, on a singular transform
    /// (upstream: `Option::unwrap` of `try_inverse`'s `None`).
    pub const NOT_INVERTIBLE: felt252 = 'nalgebra: not invertible';
}
