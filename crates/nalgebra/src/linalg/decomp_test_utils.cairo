//! Test-only helpers shared by the `linalg::qr` and `linalg::svd{2,3}` tests: builders from the
//! raw ROW-major layout of the oracle (`tools/oracle`), component-wise comparisons in ulp, the
//! orthonormality measure of a computed orthonormal factor and the tolerance predicate of the
//! oracle assertions of both decompositions.
//!
//! `linalg::lu::test_utils` is the same idea for `LU` but private to that module, and
//! `linalg::eigen_test_utils` carries only the 2x2 / 3x3 symmetric builders: the three files
//! overlap on purpose rather than making one of them public API.
//!
//! The oracle data of the two modules is regenerated from `tools/oracle` with:
//!
//! ```text
//! oracle emit-cairo qr --from vectors --ops qr2_q_r,qr2_solve \
//!     --out crates/nalgebra/src/linalg/qr/oracle_qr2.cairo
//! oracle emit-cairo qr --from vectors --ops qr3_q_r,qr3_solve \
//!     --out crates/nalgebra/src/linalg/qr/oracle_qr3.cairo
//! oracle emit-cairo qr --from vectors --ops qr4_q_r,qr4_solve \
//!     --out crates/nalgebra/src/linalg/qr/oracle_qr4.cairo
//! oracle emit-cairo symmetric_eigen_svd --from vectors \
//!     --ops svd2_singular_values,svd3_singular_values \
//!     --out crates/nalgebra/src/linalg/oracle_svd.cairo
//! ```

use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix2::{Matrix2, Matrix2Trait};
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix4::{Matrix4, Matrix4Trait};
use crate::base::vector2::{Vector2, Vector2Trait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::base::vector4::{Vector4, Vector4Trait};

/// 2^32: the raw value of 1.
const ONE_RAW: i64 = 0x100000000;

/// The `Fixed` of raw value `raw`.
pub fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

/// The integer `v` as a `Fixed`.
pub fn int(v: i64) -> Fixed {
    Fixed { raw: v * ONE_RAW }
}

/// `|a|` in raw units, as a `u128` (so that `MIN` does not overflow).
pub fn abs_raw(a: Fixed) -> u128 {
    let d: i128 = a.raw.into();
    let d = if d < 0 {
        -d
    } else {
        d
    };
    d.try_into().unwrap()
}

/// `|a - b|` in raw units.
pub fn ulp_diff(a: Fixed, b: Fixed) -> u128 {
    let d: i128 = a.raw.into() - b.raw.into();
    let d = if d < 0 {
        -d
    } else {
        d
    };
    d.try_into().unwrap()
}

/// The tolerance of a QR / SVD oracle assertion: the oracle's absolute `tol` PLUS one relative ulp
/// of the expected result, `max|expected| / 2^32`.
///
/// Same reasoning as `linalg::lu::test_utils::lu_tol`: the oracle derives `tol` from a sensitivity
/// analysis that assumes every intermediate carries a ONE-ulp error, and a fixed-point
/// orthogonalisation does not — the rounded unit column `q_i` multiplies entries of magnitude
/// `max |a_ij|`, so an intermediate of a `medium` matrix carries that many ulp. One relative ulp
/// covers the whole `qr` suite at all three sizes (measured: the worst case of `qr4_solve` sits at
/// 0.6 relative ulp above `tol`, every other op stays under `tol` alone). The `svd` assertions
/// need no relative term at all — their singular values stay well inside the oracle's own `tol`
/// —
/// and use this predicate only for the derived `solve` / `pseudo_inverse` comparisons.
pub fn decomp_tol(max_abs_expected: u128, tol: u64) -> u128 {
    tol.into() + max_abs_expected / 0x100000000
}

/// `err - allowed` when the error exceeds what is allowed, `0` otherwise: the quantity the oracle
/// tests report so that a regression shows BY HOW MUCH a case leaves the tolerance, not just that
/// it does.
pub fn excess(err: u128, allowed: u128) -> u128 {
    if err > allowed {
        err - allowed
    } else {
        0
    }
}

/// Largest `|m_ij|` in RAW units over the 4 entries (the argument of `decomp_tol`).
pub fn max_abs_m2(m: Matrix2<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m12));
    core::cmp::max(e, abs_raw(m.m22))
}

/// Largest `|m_ij|` in RAW units over the 9 entries (the argument of `decomp_tol`).
pub fn max_abs_m3(m: Matrix3<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m31));
    e = core::cmp::max(e, abs_raw(m.m12));
    e = core::cmp::max(e, abs_raw(m.m22));
    e = core::cmp::max(e, abs_raw(m.m32));
    e = core::cmp::max(e, abs_raw(m.m13));
    e = core::cmp::max(e, abs_raw(m.m23));
    core::cmp::max(e, abs_raw(m.m33))
}

/// Largest `|m_ij|` in RAW units over the 16 entries (the argument of `decomp_tol`).
pub fn max_abs_m4(m: Matrix4<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m31));
    e = core::cmp::max(e, abs_raw(m.m41));
    e = core::cmp::max(e, abs_raw(m.m12));
    e = core::cmp::max(e, abs_raw(m.m22));
    e = core::cmp::max(e, abs_raw(m.m32));
    e = core::cmp::max(e, abs_raw(m.m42));
    e = core::cmp::max(e, abs_raw(m.m13));
    e = core::cmp::max(e, abs_raw(m.m23));
    e = core::cmp::max(e, abs_raw(m.m33));
    e = core::cmp::max(e, abs_raw(m.m43));
    e = core::cmp::max(e, abs_raw(m.m14));
    e = core::cmp::max(e, abs_raw(m.m24));
    e = core::cmp::max(e, abs_raw(m.m34));
    core::cmp::max(e, abs_raw(m.m44))
}

/// Largest `|v_i|` in RAW units (the argument of `decomp_tol`).
pub fn max_abs_v2(v: Vector2<Fixed>) -> u128 {
    core::cmp::max(abs_raw(v.x), abs_raw(v.y))
}

/// Largest `|v_i|` in RAW units (the argument of `decomp_tol`).
pub fn max_abs_v3(v: Vector3<Fixed>) -> u128 {
    core::cmp::max(abs_raw(v.x), core::cmp::max(abs_raw(v.y), abs_raw(v.z)))
}

/// Largest `|v_i|` in RAW units (the argument of `decomp_tol`).
pub fn max_abs_v4(v: Vector4<Fixed>) -> u128 {
    let e = core::cmp::max(abs_raw(v.x), abs_raw(v.y));
    core::cmp::max(e, core::cmp::max(abs_raw(v.z), abs_raw(v.w)))
}

/// `ceil(max(1, max |m_ij|))` as an integer: the scale in which the reconstruction bounds of a
/// factorisation are expressed (a rounded unit column multiplies an entry of the input, so the
/// residual grows with the magnitude of the input).
pub fn amax_m2(m: Matrix2<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m12));
    e = core::cmp::max(e, abs_raw(m.m22));
    core::cmp::max(1, (e + 0xffffffff) / 0x100000000)
}

/// `ceil(max(1, max |m_ij|))` as an integer, see `amax_m2`.
pub fn amax_m3(m: Matrix3<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m31));
    e = core::cmp::max(e, abs_raw(m.m12));
    e = core::cmp::max(e, abs_raw(m.m22));
    e = core::cmp::max(e, abs_raw(m.m32));
    e = core::cmp::max(e, abs_raw(m.m13));
    e = core::cmp::max(e, abs_raw(m.m23));
    e = core::cmp::max(e, abs_raw(m.m33));
    core::cmp::max(1, (e + 0xffffffff) / 0x100000000)
}

/// `ceil(max(1, max |m_ij|))` as an integer, see `amax_m2`.
pub fn amax_m4(m: Matrix4<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m31));
    e = core::cmp::max(e, abs_raw(m.m41));
    e = core::cmp::max(e, abs_raw(m.m12));
    e = core::cmp::max(e, abs_raw(m.m22));
    e = core::cmp::max(e, abs_raw(m.m32));
    e = core::cmp::max(e, abs_raw(m.m42));
    e = core::cmp::max(e, abs_raw(m.m13));
    e = core::cmp::max(e, abs_raw(m.m23));
    e = core::cmp::max(e, abs_raw(m.m33));
    e = core::cmp::max(e, abs_raw(m.m43));
    e = core::cmp::max(e, abs_raw(m.m14));
    e = core::cmp::max(e, abs_raw(m.m24));
    e = core::cmp::max(e, abs_raw(m.m34));
    e = core::cmp::max(e, abs_raw(m.m44));
    core::cmp::max(1, (e + 0xffffffff) / 0x100000000)
}

/// `Matrix2` from raw ROW-major rows (oracle layout).
pub fn m2(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [r1, r2] = rows;
    let [a11, a12] = r1;
    let [a21, a22] = r2;
    Matrix2 { m11: fx(a11), m21: fx(a21), m12: fx(a12), m22: fx(a22) }
}

/// `Matrix3` from raw ROW-major rows (oracle layout).
pub fn m3(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [r1, r2, r3] = rows;
    let [a11, a12, a13] = r1;
    let [a21, a22, a23] = r2;
    let [a31, a32, a33] = r3;
    Matrix3 {
        m11: fx(a11),
        m21: fx(a21),
        m31: fx(a31),
        m12: fx(a12),
        m22: fx(a22),
        m32: fx(a32),
        m13: fx(a13),
        m23: fx(a23),
        m33: fx(a33),
    }
}

/// `Matrix4` from raw ROW-major rows (oracle layout).
pub fn m4(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [r1, r2, r3, r4] = rows;
    let [a11, a12, a13, a14] = r1;
    let [a21, a22, a23, a24] = r2;
    let [a31, a32, a33, a34] = r3;
    let [a41, a42, a43, a44] = r4;
    Matrix4 {
        m11: fx(a11),
        m21: fx(a21),
        m31: fx(a31),
        m41: fx(a41),
        m12: fx(a12),
        m22: fx(a22),
        m32: fx(a32),
        m42: fx(a42),
        m13: fx(a13),
        m23: fx(a23),
        m33: fx(a33),
        m43: fx(a43),
        m14: fx(a14),
        m24: fx(a24),
        m34: fx(a34),
        m44: fx(a44),
    }
}

/// `Vector2` from raw components (oracle layout).
pub fn v2(t: (i64, i64)) -> Vector2<Fixed> {
    let (c1, c2) = t;
    Vector2 { x: fx(c1), y: fx(c2) }
}

/// `Vector3` from raw components (oracle layout).
pub fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (c1, c2, c3) = t;
    Vector3 { x: fx(c1), y: fx(c2), z: fx(c3) }
}

/// `Vector4` from raw components (oracle layout).
pub fn v4(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (c1, c2, c3, c4) = t;
    Vector4 { x: fx(c1), y: fx(c2), z: fx(c3), w: fx(c4) }
}

/// Largest component-wise `|a - b|` in raw units over the 4 entries.
pub fn max_ulp_diff_m2(a: Matrix2<Fixed>, b: Matrix2<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    core::cmp::max(e, ulp_diff(a.m22, b.m22))
}

/// Largest component-wise `|a - b|` in raw units over the 9 entries.
pub fn max_ulp_diff_m3(a: Matrix3<Fixed>, b: Matrix3<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    core::cmp::max(e, ulp_diff(a.m33, b.m33))
}

/// Largest component-wise `|a - b|` in raw units over the 16 entries.
pub fn max_ulp_diff_m4(a: Matrix4<Fixed>, b: Matrix4<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    core::cmp::max(e, ulp_diff(a.m44, b.m44))
}

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff_v2(a: Vector2<Fixed>, b: Vector2<Fixed>) -> u128 {
    core::cmp::max(ulp_diff(a.x, b.x), ulp_diff(a.y, b.y))
}

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff_v3(a: Vector3<Fixed>, b: Vector3<Fixed>) -> u128 {
    let e = core::cmp::max(ulp_diff(a.x, b.x), ulp_diff(a.y, b.y));
    core::cmp::max(e, ulp_diff(a.z, b.z))
}

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff_v4(a: Vector4<Fixed>, b: Vector4<Fixed>) -> u128 {
    let e = core::cmp::max(ulp_diff(a.x, b.x), ulp_diff(a.y, b.y));
    core::cmp::max(e, core::cmp::max(ulp_diff(a.z, b.z), ulp_diff(a.w, b.w)))
}

/// `max_ij |(QᵀQ - I)_ij|` in raw units: how far the columns of `q` are from orthonormal.
pub fn orthonormality_error_m2(q: Matrix2<Fixed>) -> u128 {
    let (c1, c2) = (q.column1(), q.column2());
    let mut e = ulp_diff(c1.norm_squared(), Real::ONE);
    e = core::cmp::max(e, ulp_diff(c2.norm_squared(), Real::ONE));
    core::cmp::max(e, ulp_diff(c1.dot(c2), Real::ZERO))
}

/// `max_ij |(QᵀQ - I)_ij|` in raw units, see `orthonormality_error_m2`.
pub fn orthonormality_error_m3(q: Matrix3<Fixed>) -> u128 {
    let (c1, c2, c3) = (q.column1(), q.column2(), q.column3());
    let mut e = ulp_diff(c1.norm_squared(), Real::ONE);
    e = core::cmp::max(e, ulp_diff(c2.norm_squared(), Real::ONE));
    e = core::cmp::max(e, ulp_diff(c3.norm_squared(), Real::ONE));
    e = core::cmp::max(e, ulp_diff(c1.dot(c2), Real::ZERO));
    e = core::cmp::max(e, ulp_diff(c1.dot(c3), Real::ZERO));
    core::cmp::max(e, ulp_diff(c2.dot(c3), Real::ZERO))
}

/// `max_ij |(QᵀQ - I)_ij|` in raw units, see `orthonormality_error_m2`.
pub fn orthonormality_error_m4(q: Matrix4<Fixed>) -> u128 {
    let (c1, c2, c3, c4) = (q.column1(), q.column2(), q.column3(), q.column4());
    let mut e = ulp_diff(c1.norm_squared(), Real::ONE);
    e = core::cmp::max(e, ulp_diff(c2.norm_squared(), Real::ONE));
    e = core::cmp::max(e, ulp_diff(c3.norm_squared(), Real::ONE));
    e = core::cmp::max(e, ulp_diff(c4.norm_squared(), Real::ONE));
    e = core::cmp::max(e, ulp_diff(c1.dot(c2), Real::ZERO));
    e = core::cmp::max(e, ulp_diff(c1.dot(c3), Real::ZERO));
    e = core::cmp::max(e, ulp_diff(c1.dot(c4), Real::ZERO));
    e = core::cmp::max(e, ulp_diff(c2.dot(c3), Real::ZERO));
    e = core::cmp::max(e, ulp_diff(c2.dot(c4), Real::ZERO));
    core::cmp::max(e, ulp_diff(c3.dot(c4), Real::ZERO))
}
