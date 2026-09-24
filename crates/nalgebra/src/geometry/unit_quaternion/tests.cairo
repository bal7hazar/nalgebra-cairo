//! Unit tests of `UnitQuaternion`: exact rotations (identity, half turns, the 120° rotation about
//! `(1, 1, 1)`), the rotation identities (`q · q⁻¹ = 1`, `R · Rᵀ = I`, `q → R → q`), the
//! degenerate cases (`Option` / `#[should_panic]`), and the oracle vectors of `tools/oracle`
//! (upstream nalgebra 0.35 on the same raw inputs, tolerance in ulp).
//!
//! Oracle unit quaternions and rotation matrices are normalised in f64 then quantised, so their
//! norm is 1 within about 2 ulp, not exactly 1 — exactly what this library produces.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 8 cases per distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo quaternion --from vectors --max-per-dist 8 \
//!     --ops unit_quaternion_mul,unit_quaternion_inverse,unit_quaternion_from_axis_angle,\
//! unit_quaternion_from_scaled_axis,unit_quaternion_scaled_axis,unit_quaternion_angle,\
//! unit_quaternion_angle_to,unit_quaternion_to_rotation_matrix,\
//! unit_quaternion_from_rotation_matrix,unit_quaternion_transform_vector,\
//! unit_quaternion_inverse_transform_vector,unit_quaternion_slerp,unit_quaternion_nlerp,\
//! unit_quaternion_rotation_between,unit_quaternion_euler_angles,\
//! unit_quaternion_from_euler_angles,unit_quaternion_append_axisangle_linearized \
//!     --out crates/nalgebra/src/geometry/unit_quaternion/oracle.cairo
//! ```
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/unit_quaternion/tests.cairo`.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{ONE_RAW, fx, uqt};
use crate::geometry::quaternion::QuaternionTrait;
use crate::geometry::unit_quaternion::UnitQuaternionInternalTrait;
use super::{UnitQuaternion, UnitQuaternionTrait, oracle};

/// 1/2 in raw units.
const HALF_RAW: i64 = 0x80000000;

/// The half turn about `x`: `(0, 1, 0, 0)`, an exact unit quaternion.
fn half_x() -> UnitQuaternion<Fixed> {
    uqt((0, ONE_RAW, 0, 0))
}

/// The 120° rotation about `(1, 1, 1) / sqrt(3)`: `(1, 1, 1, 1) / 2`, exact, and its matrix is the
/// cyclic permutation `(x, y, z) -> (z, x, y)`.
fn third() -> UnitQuaternion<Fixed> {
    uqt((HALF_RAW, HALF_RAW, HALF_RAW, HALF_RAW))
}

/// `conj_mul` is `inverse() * other` bit for bit on the oracle inputs, and `q.conj_mul(q)` is the
/// identity up to the norm defect of `q` itself: exactly `(|q|², 0, 0, 0)` (the imaginary products
/// cancel exactly in the accumulator), and `|q|²` of these rounded unit inputs floors up to 3 ulp
/// away from 1.
#[test]
fn test_conj_mul_oracle_matches_inverse_then_mul() {
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(third().conj_mul(third()) == id);
    assert!(half_x().conj_mul(third()) == half_x().inverse() * third());
    let mut cases = oracle::unit_quaternion_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, _expected, _tol) = *case;
        assert!(uqt(ra).conj_mul(uqt(rb)) == uqt(ra).inverse() * uqt(rb));
        assert!(uqt(rb).conj_mul(uqt(ra)) == uqt(rb).inverse() * uqt(ra));
        let n2 = uqt(ra).quaternion.norm_squared();
        assert!(uqt(ra).conj_mul(uqt(ra)).quaternion == QuaternionTrait::from_real(n2));
        assert!(uqt(ra).conj_mul(uqt(ra)).quaternion.abs_diff_eq(id.quaternion, 3));
    }
}

// --- renormalization (from `UnitTrait`, on quaternions)

#[test]
fn test_renormalize_fast_on_a_drifted_rotation() {
    // 64 compositions of a 120° rotation: the norm drifts by a few ulp only.
    let mut q = third();
    for _ in 0_u32..64 {
        q = q * third();
    }
    let n = q.quaternion.norm();
    assert!(n.abs_diff_eq(Real::one(), 64));
    let fixed = q.renormalized_fast();
    assert!(fixed.quaternion.norm().abs_diff_eq(Real::one(), 2));
    assert!(q.renormalized().quaternion.norm().abs_diff_eq(Real::one(), 2));
    // Both renormalisations agree to 1 ulp on such a small drift.
    assert!(fixed.quaternion.abs_diff_eq(q.renormalized().quaternion, 2));
}

#[test]
fn test_renormalize_fast_fixes_a_scaled_quaternion() {
    // The 120° rotation scaled by 1 + 1e-3.
    let s = fx(HALF_RAW + 2147483);
    let drifted = uqt((s.raw, s.raw, s.raw, s.raw));
    assert!(!drifted.quaternion.norm().abs_diff_eq(Real::one(), 1000));
    let once = drifted.renormalized_fast();
    assert!(once.quaternion.norm().abs_diff_eq(Real::one(), 8000));
    assert!(once.renormalized_fast().quaternion.norm().abs_diff_eq(Real::one(), 4));
}
