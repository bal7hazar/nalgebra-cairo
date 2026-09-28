//! Internal, no stability promise: the crate-private items of `base::solve` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

/// The unrolled substitutions of one (square, right-hand side) pair (crate-private: `LU` and
/// `Cholesky` use them directly).
pub trait SolveKernel<M, B> {
    /// The scalar.
    type Scalar;
    /// `self x = b`, `self` lower triangular: forward substitution, one floor per sum and one
    /// correctly rounded division per component.
    fn lower(self: M, b: B) -> B;
    /// `self x = b`, `self` upper triangular: back substitution.
    fn upper(self: M, b: B) -> B;
    /// `self x = b` with an implicit unit diagonal (the `L` of `LU`): no division, bit-identical
    /// to `lower_with_diag(b, 1)`.
    fn lower_unit(self: M, b: B) -> B;
    /// Upstream's `solve_lower_triangular_with_diag_mut` (see `MatrixSolve`).
    fn lower_with_diag(self: M, b: B, diag: Self::Scalar) -> B;
    /// Whether every diagonal entry is nonzero.
    fn nonzero_diagonal(self: M) -> bool;
    /// `x == 0`.
    fn is_zero(x: Self::Scalar) -> bool;
    /// `selfᵀ * b` (`MatrixTrMul::tr_mul`, whose output has `b`'s shape since `self` is square):
    /// the `Qᵀ b` of `QR`.
    fn tr_mul_rhs(self: M, b: B) -> B;
}
