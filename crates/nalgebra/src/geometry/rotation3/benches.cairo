//! Gas benchmarks of `Rotation3` (`bench_rotation3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together with
//! the tests showing why (AGENTS.md rule 8).
//!
//! The inputs are the rotation matrices of the two unit quaternions of the first
//! `unit_quaternion_mul` oracle case (`a`, `b`), the vector `(1.5, -2.25, 3.75)` and the rotation
//! vector `(0.25, -0.1875, 0.125)`. Expected values are the results of the kernels themselves, all
//! of which are checked against upstream nalgebra in `tests.cairo`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/rotation3/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::MatrixTrMul;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix_test_utils::{fx, r3};
use crate::base::unit::Unit3Trait;
use crate::geometry::rotation3::Rotation3InternalTrait;
use crate::geometry::unit_quaternion::{UnitQuaternionInternalTrait, UnitQuaternionTrait};
use super::{Rotation3, Rotation3AngleTrait, Rotation3Trait};

/// The rotation matrix of the unit quaternion `q` below.
fn a() -> Rotation3<Fixed> {
    r3(
        [
            [-173945351, 4188633151, -933723411], [1215906026, -848092610, -4031011726],
            [-4115587397, -427592469, -1151455223],
        ],
    )
}

/// `renormalize` through the quaternion: `from_rotation_matrix`, one Newton step, back to a matrix.
fn alt_renormalize_quaternion(r: Rotation3<Fixed>) -> Rotation3<Fixed> {
    UnitQuaternionTrait::from_rotation_matrix(r).renormalized_fast().to_rotation_matrix()
}

/// `renormalize` by one Newton step of the polar decomposition, `R · (3I - RᵀR) / 2`: two 3x3
/// products (54 products) against two norms and six divisions for Gram-Schmidt.
fn alt_renormalize_newton(r: Rotation3<Fixed>) -> Rotation3<Fixed> {
    let m = r.matrix;
    let s = m.tr_mul(m);
    let three = Real::<Fixed>::TWO + Real::one();
    let c = Matrix3 {
        m11: Real::mul_add(-s.m11, Real::HALF, three * Real::HALF),
        m21: -s.m21 * Real::HALF,
        m31: -s.m31 * Real::HALF,
        m12: -s.m12 * Real::HALF,
        m22: Real::mul_add(-s.m22, Real::HALF, three * Real::HALF),
        m32: -s.m32 * Real::HALF,
        m13: -s.m13 * Real::HALF,
        m23: -s.m23 * Real::HALF,
        m33: Real::mul_add(-s.m33, Real::HALF, three * Real::HALF),
    };
    Rotation3 { matrix: m * c }
}

/// All three renormalizations restore orthonormality on a drifted matrix; Gram-Schmidt is the
/// cheapest (see the benchmarks) and the most accurate of the three here.
#[test]
fn test_renormalize_alts_restore_orthonormality() {
    let step = Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), fx(0x123456789));
    let mut r = Rotation3Trait::<Fixed>::identity();
    for _ in 0_u32..64 {
        r = r * step;
    }
    let gs = r.renormalized();
    let qn = alt_renormalize_quaternion(r);
    let nw = alt_renormalize_newton(r);
    assert!((gs.matrix * gs.matrix.transpose()).is_identity(4));
    assert!((qn.matrix * qn.matrix.transpose()).is_identity(16));
    assert!((nw.matrix * nw.matrix.transpose()).is_identity(16));
    assert!(gs.abs_diff_eq(qn, 64));
    assert!(gs.abs_diff_eq(nw, 64));
}

/// A Newton step of the polar decomposition only halves the error, so it does not fix a matrix
/// scaled by 1 + 1e-3, where Gram-Schmidt does in one pass.
#[test]
fn test_renormalize_alt_newton_only_converges() {
    let scaled = Rotation3 { matrix: a().matrix.scale(Real::one() + fx(4294967)) };
    assert!(
        (scaled.renormalized().matrix * scaled.renormalized().matrix.transpose()).is_identity(4),
    );
    let once = alt_renormalize_newton(scaled);
    assert!(!(once.matrix * once.matrix.transpose()).is_identity(0x1000));
}

#[test]
#[inline(never)]
fn bench_rotation3_renormalize__alt_quaternion() {
    let r = black_box(a());
    let e = black_box(a());
    assert!(alt_renormalize_quaternion(r).abs_diff_eq(e, 8));
}
