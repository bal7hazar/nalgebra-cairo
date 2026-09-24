//! Unit tests of `Rotation2`: exact cases (identity, quarter turns, axes), the identities a
//! rotation matrix must satisfy (`R · Rᵀ = I`, `det R = 1`, `R → c → R`), panics on
//! degenerate inputs, and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same
//! raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution,
//! the `rotation2_*` ops of the `rotation2` suite) with
//! `cargo run --release -- emit-cairo rotation2 --from vectors --max-per-dist 6 --out
//! <oracle.cairo> --ops rotation2_new,rotation2_angle,rotation2_mul,rotation2_inverse,
//! rotation2_transform_vector`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/rotation2/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix2::Matrix2;
use crate::base::matrix_test_utils::{ONE_RAW, fx, r2};
use crate::geometry::rotation2::Rotation2InternalTrait;
use crate::geometry::unit_complex::{UnitComplexAngleTrait, UnitComplexTrait};
use super::{Rotation2, Rotation2Trait};

/// The raw value of 1.
fn id() -> Rotation2<Fixed> {
    Rotation2Trait::<Fixed>::identity()
}

/// The quarter turn `[[0, -1], [1, 0]]`, an exactly representable rotation.
fn quarter() -> Rotation2<Fixed> {
    r2([[0, -ONE_RAW], [ONE_RAW, 0]])
}

#[test]
fn test_from_matrix_normalizes_the_first_column() {
    // A first column 4 times too long (exact scaling): `from_matrix` divides it back and rebuilds
    // the second column from it, whatever the garbage that column held.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let four = Real::<Fixed>::from_int(4);
    let m = Matrix2 { m11: c.re * four, m21: c.im * four, m12: Real::zero(), m22: Real::zero() };
    let r = Rotation2Trait::from_matrix(m);
    assert!(r.abs_diff_eq(c.to_rotation_matrix(), 2));
    assert!(r.renormalized().abs_diff_eq(r, 2));
    // An exact rotation is a fixed point of `from_matrix`.
    assert!(Rotation2Trait::from_matrix(quarter().matrix) == quarter());
}

// --- renormalization

#[test]
fn test_renormalize_is_exact_on_a_rotation() {
    assert!(quarter().renormalized() == quarter());
    assert!(id().renormalized() == id());
}

#[test]
fn test_renormalize_recovers_orthogonality() {
    // A matrix whose columns drifted apart: renormalizing the first column and rebuilding makes
    // `R · Rᵀ = I` hold again within 2 ulp.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let drifted = Rotation2 {
        matrix: Matrix2 {
            m11: c.re + fx(0x68db8), m21: c.im + fx(0x68db8), m12: -c.im, m22: c.re - fx(0x68db8),
        },
    };
    let p = drifted * drifted.inverse();
    assert!(!p.abs_diff_eq(id(), 0x1000));
    let r = drifted.renormalized();
    assert!((r * r.inverse()).abs_diff_eq(id(), 2));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_renormalize_of_a_zero_first_column_panics() {
    let m = Matrix2 {
        m11: Real::<Fixed>::zero(), m21: Real::zero(), m12: Real::one(), m22: Real::one(),
    };
    let _ = black_box(Rotation2Trait::from_matrix_unchecked(m)).renormalized();
}
