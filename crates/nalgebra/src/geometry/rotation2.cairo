//! `Rotation2`: a 2D rotation stored as a 2x2 orthogonal matrix (upstream
//! `nalgebra::Rotation2`, which is `Rotation<T, 2>`).
//!
//! - `Rotation2Trait` / `Rotation2Impl`: construction, accessors, composition, transforms,
//!   conversions and renormalization — everything that is algebraic, hence available for any
//!   `simba::scalar::Real` scalar;
//! - `Rotation2AngleTrait` / `Rotation2AngleImpl`: the operations that go through an angle
//!   (`new`, `angle`, `powf`, `scaled_rotation_between`), which additionally need
//!   `simba::scalar::Transcendental`;
//! - `a * b` (composition of two rotations) and the conversions from / to `UnitComplex<T>`: their
//!   impls live in this module, where the compiler finds them without any import.
//!
//! `Rotation2` and `UnitComplex` hold the same information: the matrix is
//! `[[cos θ, -sin θ], [sin θ, cos θ]]`, whose first column IS the complex `(re, im)`. Measured
//! (Sierra gas, net of the group baseline), the matrix form costs **2.6x more per composition**
//! (10 260 against 4 000) and twice the storage, for transforms that are bit-identical and cost
//! exactly the same (4 000 either way, `bench_rotation2_transform_point__alt_unit_complex`). It
//! exists because upstream has it, because it composes with the `Matrix2` world without a
//! conversion, and because a `Rotation2` is what `to_homogeneous` and the matrix decompositions
//! consume. **Prefer `UnitComplex` in hot code**; converting costs 800 gas one way and 2 400 the
//! other, so even a single composition pays for the round trip.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod tests;

pub use nalgebra_geometry2::geometry::isometry2::Isometry2FromRotation2;
pub use nalgebra_geometry2::geometry::rotation2::*;
pub use nalgebra_geometry2::geometry::similarity2::Similarity2FromRotation2;
pub use nalgebra_geometry2::geometry::unit_complex::{
    Rotation2DivAssignUnitComplex, Rotation2FromUnitComplex, Rotation2IntoUnitComplex,
    Rotation2MulAssignUnitComplex,
};

// the crate-private helpers the in-crate tests use (`nalgebra_static3::internal`)
#[cfg(test)]
use nalgebra_geometry2::internal::geometry::rotation2::Rotation2InternalTrait;
pub use nalgebra_types2::geometry::rotation2::*;
