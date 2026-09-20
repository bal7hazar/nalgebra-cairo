//! Test-only helpers of the `linalg` tests: builders from the raw row-major layout of the oracle
//! (`tools/oracle`), integer-valued builders and comparisons in ulp.
//!
//! `base::matrix_test_utils` is private to `base`, so the handful of helpers the decompositions
//! need are repeated here rather than widening that module's visibility.

use simba::fixed::Fixed;
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::sym_matrix2::SymMatrix2;
use crate::base::sym_matrix3::SymMatrix3;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;

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

/// `Vector2` from raw components.
pub fn v2(t: (i64, i64)) -> Vector2<Fixed> {
    let (x, y) = t;
    Vector2 { x: fx(x), y: fx(y) }
}

/// `Vector3` from raw components.
pub fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// `Vector2` from integers.
pub fn v2i(x: i64, y: i64) -> Vector2<Fixed> {
    Vector2 { x: int(x), y: int(y) }
}

/// `Vector3` from integers.
pub fn v3i(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: int(x), y: int(y), z: int(z) }
}

/// `Matrix2` from raw ROW-major rows (oracle layout).
pub fn m2(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [[m11, m12], [m21, m22]] = rows;
    Matrix2 { m11: fx(m11), m21: fx(m21), m12: fx(m12), m22: fx(m22) }
}

/// `Matrix3` from raw ROW-major rows (oracle layout).
pub fn m3(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = rows;
    Matrix3 {
        m11: fx(m11),
        m21: fx(m21),
        m31: fx(m31),
        m12: fx(m12),
        m22: fx(m22),
        m32: fx(m32),
        m13: fx(m13),
        m23: fx(m23),
        m33: fx(m33),
    }
}

/// The upper triangle of the raw ROW-major matrix `rows`, as a `SymMatrix2` (oracle layout).
pub fn s2(rows: [[i64; 2]; 2]) -> SymMatrix2<Fixed> {
    let [[m11, m12], [_, m22]] = rows;
    SymMatrix2 { m11: fx(m11), m12: fx(m12), m22: fx(m22) }
}

/// The upper triangle of the raw ROW-major matrix `rows`, as a `SymMatrix3` (oracle layout).
pub fn s3(rows: [[i64; 3]; 3]) -> SymMatrix3<Fixed> {
    let [[m11, m12, m13], [_, m22, m23], [_, _, m33]] = rows;
    SymMatrix3 {
        m11: fx(m11), m12: fx(m12), m13: fx(m13), m22: fx(m22), m23: fx(m23), m33: fx(m33),
    }
}

/// `SymMatrix2` from its integer upper triangle `(m11, m12, m22)`.
pub fn s2i(m11: i64, m12: i64, m22: i64) -> SymMatrix2<Fixed> {
    SymMatrix2 { m11: int(m11), m12: int(m12), m22: int(m22) }
}

/// `SymMatrix3` from its integer upper triangle `(m11, m12, m13, m22, m23, m33)`.
pub fn s3i(m11: i64, m12: i64, m13: i64, m22: i64, m23: i64, m33: i64) -> SymMatrix3<Fixed> {
    SymMatrix3 {
        m11: int(m11), m12: int(m12), m13: int(m13), m22: int(m22), m23: int(m23), m33: int(m33),
    }
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

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff_v2(a: Vector2<Fixed>, b: Vector2<Fixed>) -> u128 {
    core::cmp::max(ulp_diff(a.x, b.x), ulp_diff(a.y, b.y))
}

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff_v3(a: Vector3<Fixed>, b: Vector3<Fixed>) -> u128 {
    let e = core::cmp::max(ulp_diff(a.x, b.x), ulp_diff(a.y, b.y));
    core::cmp::max(e, ulp_diff(a.z, b.z))
}

/// Largest component-wise `|a - b|` in raw units, over the 3 stored components.
pub fn max_ulp_diff_s2(a: SymMatrix2<Fixed>, b: SymMatrix2<Fixed>) -> u128 {
    let e = core::cmp::max(ulp_diff(a.m11, b.m11), ulp_diff(a.m12, b.m12));
    core::cmp::max(e, ulp_diff(a.m22, b.m22))
}

/// Largest component-wise `|a - b|` in raw units, over the 6 stored components.
pub fn max_ulp_diff_s3(a: SymMatrix3<Fixed>, b: SymMatrix3<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    core::cmp::max(e, ulp_diff(a.m33, b.m33))
}

/// `|raw|` as a `u128`.
fn abs_raw(a: Fixed) -> u128 {
    let d: i128 = a.raw.into();
    let d = if d < 0 {
        -d
    } else {
        d
    };
    d.try_into().unwrap()
}

/// `ceil(max(1, max |m_ij|))` as an integer: the scale in which the reconstruction and
/// orthonormality bounds of the eigen decompositions are expressed (a rounded eigenvector
/// component multiplies an eigenvalue, so the residual grows with the magnitude of the input).
pub fn amax_s3(m: SymMatrix3<Fixed>) -> u128 {
    let mut e = core::cmp::max(abs_raw(m.m11), abs_raw(m.m12));
    e = core::cmp::max(e, abs_raw(m.m13));
    e = core::cmp::max(e, abs_raw(m.m22));
    e = core::cmp::max(e, abs_raw(m.m23));
    e = core::cmp::max(e, abs_raw(m.m33));
    core::cmp::max(1, (e + 0xffffffff) / 0x100000000)
}

/// `ceil(max(1, max |m_ij|))` as an integer, see `amax_s3`.
pub fn amax_s2(m: SymMatrix2<Fixed>) -> u128 {
    let e = core::cmp::max(abs_raw(m.m11), core::cmp::max(abs_raw(m.m12), abs_raw(m.m22)));
    core::cmp::max(1, (e + 0xffffffff) / 0x100000000)
}
