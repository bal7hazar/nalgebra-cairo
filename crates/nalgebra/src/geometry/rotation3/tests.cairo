//! Unit tests of `Rotation3`: exact rotations (identity, half turns, the axis permutation), the
//! rotation identities (`R · Rᵀ = I`, `det R = 1`, `R → q → R`), the degenerate cases
//! (`Option` /
//! `#[should_panic]`), and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same
//! raw inputs, tolerance in ulp).
//!
//! Oracle rotation matrices are orthonormalised in f64 then quantised component-wise, so they are
//! orthonormal only within about 2 ulp — exactly what this library produces.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 8 cases per distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo rotation3 --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/rotation3/oracle.cairo
//! ```
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/rotation3/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix3::{Matrix3InternalTrait, Matrix3Trait};
use crate::base::matrix_test_utils::{ONE_RAW, fx, r3i, v3i};
use crate::base::unit::Unit3Trait;
use crate::base::vector3::Vector3Trait;
use crate::geometry::rotation3::Rotation3InternalTrait;
use super::{Rotation3, Rotation3AngleTrait, Rotation3Trait};

/// The half turn about `x`: `diag(1, -1, -1)`, exact.
fn half_x() -> Rotation3<Fixed> {
    r3i([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
}

/// The 120° rotation about `(1, 1, 1)`: the cyclic permutation `(x, y, z) -> (z, x, y)`, exact.
fn third() -> Rotation3<Fixed> {
    r3i([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
}

#[test]
fn test_face_towards_geometry() {
    // dir becomes the third column (the local z axis), up leans towards the second.
    let dir = v3i(0, 0, 2);
    let up = v3i(0, 1, 0);
    let r = Rotation3Trait::face_towards(dir, up);
    assert!(r.matrix.column3().abs_diff_eq(v3i(0, 0, 1), 2));
    assert!(r.matrix.column2().abs_diff_eq(v3i(0, 1, 0), 2));
    assert!((r.matrix * r.matrix.transpose()).is_identity(8));
    // look_at_rh maps dir onto -z.
    let l = Rotation3Trait::look_at_rh(dir, up);
    assert!(l.transform_vector(v3i(0, 0, 1)).abs_diff_eq(v3i(0, 0, -1), 4));
}

// --- renormalization

#[test]
fn test_renormalize_keeps_a_rotation_matrix() {
    assert!(Rotation3Trait::<Fixed>::identity().renormalized() == Rotation3Trait::identity());
    assert!(third().renormalized() == third());
    assert!(half_x().renormalized() == half_x());
}

/// 64 compositions of a quarter turn: the matrix drifts, `renormalize` brings it back to
/// orthonormal.
#[test]
fn test_renormalize_fixes_a_drifted_matrix() {
    let step = Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), fx(0x123456789));
    let mut r = Rotation3Trait::<Fixed>::identity();
    for _ in 0_u32..64 {
        r = r * step;
    }
    let fixed = r.renormalized();
    assert!((fixed.matrix * fixed.matrix.transpose()).is_identity(4));
    assert!(fixed.matrix.determinant().abs_diff_eq(Real::one(), 4));
    // The correction is small: the drift of 64 products is a few ulp.
    assert!(fixed.abs_diff_eq(r, 64));
}

/// A matrix scaled by 1 + 1e-3 is brought back to orthonormal in one pass (the closed-form polar
/// factor ignores the scale).
#[test]
fn test_renormalize_handles_a_scaled_matrix() {
    let s = fx(ONE_RAW + 4294967);
    let scaled = Rotation3 { matrix: third().matrix.scale(s) };
    let fixed = scaled.renormalized();
    assert!((fixed.matrix * fixed.matrix.transpose()).is_identity(8));
    assert!(fixed.abs_diff_eq(third(), 8));
}

/// WP 8.4-P10 (upstream's formula, `from_matrix_eps(m, eps, 0, guess)`): the singular zero matrix
/// no longer panics (Gram-Schmidt divided by zero); the closed-form limit of the polar factor is
/// the identity.
#[test]
fn test_renormalize_of_the_zero_matrix_is_the_identity() {
    let zero = Rotation3 { matrix: Matrix3Trait::<Fixed>::zeros() };
    assert!(black_box(zero).renormalized() == Rotation3Trait::identity());
}
