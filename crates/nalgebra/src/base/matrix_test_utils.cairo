//! Test-only helpers shared by the matrix tests: builders from the raw row-major layout of the
//! oracle (`tools/oracle`), integer-valued builders and component-wise comparisons in ulp.
//!
//! Oracle data (`oracle_matrix{2,3,4}.cairo`, `oracle_matrix{2,3,4}_inverse.cairo`) is regenerated
//! from `tools/oracle` with:
//!
//! ```text
//! oracle emit-cairo matrixN --from vectors --max-per-dist 4 --ops <all but the four below> \
//!     --out crates/nalgebra/src/base/oracle_matrixN.cairo
//! oracle emit-cairo matrixN --from vectors --max-per-dist 10 --ops matrixN_determinant,\
//!     matrixN_try_inverse,matrixN_try_inverse_singular,matrixN_try_inverse_near_singular \
//!     --out crates/nalgebra/src/base/oracle_matrixN_inverse.cairo
//! ```
//!
//! `oracle_sym_matrix.cairo` holds the symmetric-matrix vectors. The oracle has no `SymMatrixN`
//! suite (the type is rapier's, not upstream nalgebra's); its `udu` suite is used instead, whose
//! inputs are EXACTLY symmetric positive-definite in raw units and whose `_inverse` outputs are
//! upstream inverses:
//!
//! ```text
//! oracle emit-cairo udu --from vectors --max-per-dist 8 --ops udu2_inverse,udu3_inverse \
//!     --out crates/nalgebra/src/base/oracle_sym_matrix.cairo
//! ```

use simba::fixed::Fixed;
use super::matrix2::Matrix2;
use super::matrix3::Matrix3;
use super::matrix4::Matrix4;
use super::sym_matrix2::SymMatrix2;
use super::sym_matrix3::SymMatrix3;
use super::vector2::Vector2;
use super::vector3::Vector3;
use super::vector4::Vector4;

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

/// `Vector4` from raw components.
pub fn v4(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (x, y, z, w) = t;
    Vector4 { x: fx(x), y: fx(y), z: fx(z), w: fx(w) }
}

/// `Vector2` from integers.
pub fn v2i(x: i64, y: i64) -> Vector2<Fixed> {
    Vector2 { x: int(x), y: int(y) }
}

/// `Vector3` from integers.
pub fn v3i(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: int(x), y: int(y), z: int(z) }
}

/// `Vector4` from integers.
pub fn v4i(x: i64, y: i64, z: i64, w: i64) -> Vector4<Fixed> {
    Vector4 { x: int(x), y: int(y), z: int(z), w: int(w) }
}

/// `Matrix2` from raw ROW-major rows (oracle layout).
pub fn m2(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [[m11, m12], [m21, m22]] = rows;
    Matrix2 { m11: fx(m11), m21: fx(m21), m12: fx(m12), m22: fx(m22) }
}

/// `Matrix2` from integer ROW-major rows.
pub fn m2i(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [[m11, m12], [m21, m22]] = rows;
    Matrix2 { m11: int(m11), m21: int(m21), m12: int(m12), m22: int(m22) }
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

/// `Matrix3` from integer ROW-major rows.
pub fn m3i(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = rows;
    Matrix3 {
        m11: int(m11),
        m21: int(m21),
        m31: int(m31),
        m12: int(m12),
        m22: int(m22),
        m32: int(m32),
        m13: int(m13),
        m23: int(m23),
        m33: int(m33),
    }
}

/// `Matrix4` from raw ROW-major rows (oracle layout).
pub fn m4(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [r1, r2, r3, r4] = rows;
    let [m11, m12, m13, m14] = r1;
    let [m21, m22, m23, m24] = r2;
    let [m31, m32, m33, m34] = r3;
    let [m41, m42, m43, m44] = r4;
    Matrix4 {
        m11: fx(m11),
        m21: fx(m21),
        m31: fx(m31),
        m41: fx(m41),
        m12: fx(m12),
        m22: fx(m22),
        m32: fx(m32),
        m42: fx(m42),
        m13: fx(m13),
        m23: fx(m23),
        m33: fx(m33),
        m43: fx(m43),
        m14: fx(m14),
        m24: fx(m24),
        m34: fx(m34),
        m44: fx(m44),
    }
}

/// `Matrix4` from integer ROW-major rows.
pub fn m4i(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [r1, r2, r3, r4] = rows;
    let [m11, m12, m13, m14] = r1;
    let [m21, m22, m23, m24] = r2;
    let [m31, m32, m33, m34] = r3;
    let [m41, m42, m43, m44] = r4;
    Matrix4 {
        m11: int(m11),
        m21: int(m21),
        m31: int(m31),
        m41: int(m41),
        m12: int(m12),
        m22: int(m22),
        m32: int(m32),
        m42: int(m42),
        m13: int(m13),
        m23: int(m23),
        m33: int(m33),
        m43: int(m43),
        m14: int(m14),
        m24: int(m24),
        m34: int(m34),
        m44: int(m44),
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
pub fn max_ulp_diff2(a: Matrix2<Fixed>, b: Matrix2<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    core::cmp::max(e, ulp_diff(a.m22, b.m22))
}

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff3(a: Matrix3<Fixed>, b: Matrix3<Fixed>) -> u128 {
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

/// Largest component-wise `|a - b|` in raw units.
pub fn max_ulp_diff4(a: Matrix4<Fixed>, b: Matrix4<Fixed>) -> u128 {
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

/// `SymMatrix2` from its raw upper triangle `(m11, m12, m22)`.
pub fn s2(t: (i64, i64, i64)) -> SymMatrix2<Fixed> {
    let (m11, m12, m22) = t;
    SymMatrix2 { m11: fx(m11), m12: fx(m12), m22: fx(m22) }
}

/// `SymMatrix3` from its raw upper triangle `(m11, m12, m13, m22, m23, m33)`.
pub fn s3(t: (i64, i64, i64, i64, i64, i64)) -> SymMatrix3<Fixed> {
    let (m11, m12, m13, m22, m23, m33) = t;
    SymMatrix3 {
        m11: fx(m11), m12: fx(m12), m13: fx(m13), m22: fx(m22), m23: fx(m23), m33: fx(m33),
    }
}

/// `SymMatrix2` from its integer upper triangle `(m11, m12, m22)`.
pub fn s2i(t: (i64, i64, i64)) -> SymMatrix2<Fixed> {
    let (m11, m12, m22) = t;
    SymMatrix2 { m11: int(m11), m12: int(m12), m22: int(m22) }
}

/// `SymMatrix3` from its integer upper triangle `(m11, m12, m13, m22, m23, m33)`.
pub fn s3i(t: (i64, i64, i64, i64, i64, i64)) -> SymMatrix3<Fixed> {
    let (m11, m12, m13, m22, m23, m33) = t;
    SymMatrix3 {
        m11: int(m11), m12: int(m12), m13: int(m13), m22: int(m22), m23: int(m23), m33: int(m33),
    }
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
