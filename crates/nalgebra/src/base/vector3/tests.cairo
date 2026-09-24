//! Unit tests of `Vector3`: exact cases (expected values from a bit-exact integer model of the
//! Q32.32 kernels, floor rounding), panics, robustness at both ends of the range, identities, and
//! the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw inputs).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution,
//! every op of the suite) with
//! `cargo run --release -- emit-cairo vector3 --from vectors --max-per-dist 4 --out
//! <oracle.cairo>`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/vector3/tests.cairo`.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::v3;
use super::{Vector3, Vector3InternalTrait, Vector3Trait};

#[test]
fn test_orthonormal_basis_of_axes() {
    assert!(
        Vector3Trait::<Fixed>::x()
            .orthonormal_basis() == (v3(0, 0, -0x100000000), v3(0, 0x100000000, 0)),
    );
    assert!(
        Vector3Trait::<Fixed>::y()
            .orthonormal_basis() == (v3(0x100000000, 0, 0), v3(0, 0, -0x100000000)),
    );
    assert!(
        Vector3Trait::<Fixed>::z()
            .orthonormal_basis() == (v3(0x100000000, 0, 0), v3(0, 0x100000000, 0)),
    );
    assert!(
        (-Vector3Trait::<Fixed>::z())
            .orthonormal_basis() == (v3(0x100000000, 0, 0), v3(0, -0x100000000, 0)),
    );
}

/// `(v, u, w)` is orthonormal and right-handed (`u x w = v`) within `tol` ulp.
fn assert_orthonormal_basis(v: Vector3<Fixed>, tol: u64) {
    let (u, w) = v.orthonormal_basis();
    assert!(u.norm().abs_diff_eq(Real::one(), tol) && w.norm().abs_diff_eq(Real::one(), tol));
    assert!(u.dot(w).abs_diff_eq(Real::zero(), tol));
    assert!(u.dot(v).abs_diff_eq(Real::zero(), tol) && w.dot(v).abs_diff_eq(Real::zero(), tol));
    assert!(u.cross(w).abs_diff_eq(v, tol));
}

#[test]
fn test_orthonormal_basis_is_orthonormal_and_right_handed() {
    // Every octant, directions close to the poles `z = +-1` and to the equator, axes. Tolerance
    // 3 ulp: input-dependent rounding (the nearest-normalized `(0.25, -0.5, 0.125)` reaches 3 ulp
    // on `u.v` and `u x w - v`, the other directions stay within 2).
    let mut dirs = array![
        v3(0x100000000, 0x200000000, 0x300000000), // (1, 2, 3)
        v3(-0x100000000, 0x200000000, 0x300000000), // (-1, 2, 3)
        v3(0x100000000, -0x200000000, 0x300000000), // (1, -2, 3)
        v3(0x100000000, 0x200000000, -0x300000000), // (1, 2, -3)
        v3(-0x100000000, -0x200000000, 0x300000000), // (-1, -2, 3)
        v3(-0x100000000, 0x200000000, -0x300000000), // (-1, 2, -3)
        v3(0x100000000, -0x200000000, -0x300000000), // (1, -2, -3)
        v3(-0x100000000, -0x200000000, -0x300000000), // (-1, -2, -3)
        v3(0x100000000, 0x100000000, 0x186a000000000), // (1, 1, 100000)
        v3(-0x100000000, 0x100000000, -0x186a000000000), // (-1, 1, -100000)
        v3(0x186a000000000, 0x300000000, 0x100000000), // (100000, 3, 1)
        v3(0x300000000, -0x186a000000000, -0x100000000), // (3, -100000, -1)
        v3(0, 0x500000000, 0), // (0, 5, 0)
        v3(0x700000000, 0, 0), // (7, 0, 0)
        v3(0x100000000, 0x100000000, 0), // (1, 1, 0)
        v3(0x40000000, -0x80000000, 0x20000000) // (0.25, -0.5, 0.125)
    ]
        .span();
    while let Some(dir) = dirs.pop_front() {
        assert_orthonormal_basis((*dir).normalize(), 3);
    }
}

#[test]
fn test_orthonormal_basis_model_values() {
    // Unit vector of (1.5, -2.25, 3.75), then with z negated (other branch).
    assert!(
        v3(1393471396, -2090207096, 3483678492)
            .orthonormal_basis() == (
                v3(4045339972, 374440987, -1393471396), v3(374440987, 3733305816, 2090207096),
            ),
    );
    assert!(
        v3(1393471396, -2090207096, -3483678493)
            .orthonormal_basis() == (
                v3(4045339972, 374440987, 1393471396), v3(-374440987, -3733305816, 2090207096),
            ),
    );
}
