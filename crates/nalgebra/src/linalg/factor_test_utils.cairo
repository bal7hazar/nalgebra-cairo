//! Test-only helpers shared by the factorisation tests and benchmarks: builders from the raw
//! ROW-major layout of the oracle (`tools/oracle`), integer-valued builders and component-wise
//! comparisons in raw units (ulp).
//!
//! `base::matrix_test_utils` has the same role for `base` but is private to that module, so the
//! builders the decompositions need (`m6`, the `SymMatrix` view of a full symmetric matrix, the
//! vector comparisons) are restated here rather than reached for across modules.
//!
//! Oracle data (`oracle_cholesky.cairo`, `oracle_udu.cairo`) is regenerated from `tools/oracle`
//! with:
//!
//! ```text
//! oracle emit-cairo cholesky --from vectors --max-per-dist 4 \
//!     --ops cholesky2_l,cholesky2_solve,cholesky2_inverse,cholesky3_l,cholesky3_solve,\
//! cholesky3_inverse,cholesky4_l,cholesky4_solve,cholesky4_inverse,cholesky6_l,cholesky6_solve,\
//! cholesky6_inverse --out crates/nalgebra/src/linalg/oracle_cholesky.cairo
//! oracle emit-cairo udu --from vectors --max-per-dist 4 \
//!     --ops ldlt2_l_d,ldlt3_l_d,ldlt4_l_d,ldlt6_l_d,udu2_u_d,udu3_u_d,udu2_solve,udu2_inverse,\
//! udu3_solve,udu3_inverse,udu4_solve,udu4_inverse,udu6_solve,udu6_inverse \
//!     --out crates/nalgebra/src/linalg/oracle_udu.cairo
//! ```
//!
//! Sizes 5 are emitted by neither: there is no `Vector5` / `Matrix5` in `base` (DESIGN D4 ships
//! the sizes physics uses: 2, 3, 4 and the 6 of spatial algebra).

use simba::fixed::Fixed;
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::matrix6::Matrix6;
use crate::base::sym_matrix2::SymMatrix2;
use crate::base::sym_matrix3::SymMatrix3;
use crate::base::vector2::Vector2;
use crate::base::vector3::Vector3;
use crate::base::vector4::Vector4;
use crate::base::vector6::Vector6;

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

// --- vectors ------------------------------------------------------------------------------------

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

/// `Vector6` from raw components, in the upstream order `(x, y, z, w, a, b)`.
pub fn v6(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Vector6 {
        a: Vector3 { x: fx(x), y: fx(y), z: fx(z) }, b: Vector3 { x: fx(w), y: fx(a), z: fx(b) },
    }
}

/// `Vector2` from integers.
pub fn v2i(t: (i64, i64)) -> Vector2<Fixed> {
    let (x, y) = t;
    Vector2 { x: int(x), y: int(y) }
}

/// `Vector3` from integers.
pub fn v3i(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: int(x), y: int(y), z: int(z) }
}

/// `Vector4` from integers.
pub fn v4i(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (x, y, z, w) = t;
    Vector4 { x: int(x), y: int(y), z: int(z), w: int(w) }
}

/// `Vector6` from integers, in the upstream order `(x, y, z, w, a, b)`.
pub fn v6i(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Vector6 {
        a: Vector3 { x: int(x), y: int(y), z: int(z) },
        b: Vector3 { x: int(w), y: int(a), z: int(b) },
    }
}

// --- matrices -----------------------------------------------------------------------------------

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

/// `Matrix6` from raw ROW-major rows (oracle layout), laid out as the four 3x3 blocks.
pub fn m6(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    let [a11, a12, a13, a14, a15, a16] = r1;
    let [a21, a22, a23, a24, a25, a26] = r2;
    let [a31, a32, a33, a34, a35, a36] = r3;
    let [a41, a42, a43, a44, a45, a46] = r4;
    let [a51, a52, a53, a54, a55, a56] = r5;
    let [a61, a62, a63, a64, a65, a66] = r6;
    Matrix6 {
        m11: Matrix3 {
            m11: fx(a11),
            m21: fx(a21),
            m31: fx(a31),
            m12: fx(a12),
            m22: fx(a22),
            m32: fx(a32),
            m13: fx(a13),
            m23: fx(a23),
            m33: fx(a33),
        },
        m21: Matrix3 {
            m11: fx(a41),
            m21: fx(a51),
            m31: fx(a61),
            m12: fx(a42),
            m22: fx(a52),
            m32: fx(a62),
            m13: fx(a43),
            m23: fx(a53),
            m33: fx(a63),
        },
        m12: Matrix3 {
            m11: fx(a14),
            m21: fx(a24),
            m31: fx(a34),
            m12: fx(a15),
            m22: fx(a25),
            m32: fx(a35),
            m13: fx(a16),
            m23: fx(a26),
            m33: fx(a36),
        },
        m22: Matrix3 {
            m11: fx(a44),
            m21: fx(a54),
            m31: fx(a64),
            m12: fx(a45),
            m22: fx(a55),
            m32: fx(a65),
            m13: fx(a46),
            m23: fx(a56),
            m33: fx(a66),
        },
    }
}

/// Every raw entry of `rows` scaled by 2^32: the integer matrix builders route through here.
fn scaled2(rows: [[i64; 2]; 2]) -> [[i64; 2]; 2] {
    let [[a, b], [c, d]] = rows;
    [[a * ONE_RAW, b * ONE_RAW], [c * ONE_RAW, d * ONE_RAW]]
}

/// `Matrix2` from integer ROW-major rows.
pub fn m2i(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    m2(scaled2(rows))
}

/// `Matrix3` from integer ROW-major rows.
pub fn m3i(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [[a, b, c], [d, e, f], [g, h, i]] = rows;
    m3(
        [
            [a * ONE_RAW, b * ONE_RAW, c * ONE_RAW], [d * ONE_RAW, e * ONE_RAW, f * ONE_RAW],
            [g * ONE_RAW, h * ONE_RAW, i * ONE_RAW],
        ],
    )
}

/// `Matrix4` from integer ROW-major rows.
pub fn m4i(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [r1, r2, r3, r4] = rows;
    let [a, b, c, d] = r1;
    let [e, f, g, h] = r2;
    let [i, j, k, l] = r3;
    let [m, n, o, p] = r4;
    m4(
        [
            [a * ONE_RAW, b * ONE_RAW, c * ONE_RAW, d * ONE_RAW],
            [e * ONE_RAW, f * ONE_RAW, g * ONE_RAW, h * ONE_RAW],
            [i * ONE_RAW, j * ONE_RAW, k * ONE_RAW, l * ONE_RAW],
            [m * ONE_RAW, n * ONE_RAW, o * ONE_RAW, p * ONE_RAW],
        ],
    )
}

/// `Matrix6` from integer ROW-major rows.
pub fn m6i(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    m6([row6(r1), row6(r2), row6(r3), row6(r4), row6(r5), row6(r6)])
}

/// One integer row of `m6i`, scaled by 2^32.
fn row6(r: [i64; 6]) -> [i64; 6] {
    let [a, b, c, d, e, f] = r;
    [a * ONE_RAW, b * ONE_RAW, c * ONE_RAW, d * ONE_RAW, e * ONE_RAW, f * ONE_RAW]
}

/// The `SymMatrix2` view of a raw ROW-major matrix: its UPPER triangle. The oracle's SPD inputs
/// are exactly symmetric in raw units, so this loses nothing.
pub fn s2(rows: [[i64; 2]; 2]) -> SymMatrix2<Fixed> {
    let [[m11, m12], [_, m22]] = rows;
    SymMatrix2 { m11: fx(m11), m12: fx(m12), m22: fx(m22) }
}

/// The `SymMatrix3` view of a raw ROW-major matrix: its UPPER triangle.
pub fn s3(rows: [[i64; 3]; 3]) -> SymMatrix3<Fixed> {
    let [[m11, m12, m13], [_, m22, m23], [_, _, m33]] = rows;
    SymMatrix3 {
        m11: fx(m11), m12: fx(m12), m13: fx(m13), m22: fx(m22), m23: fx(m23), m33: fx(m33),
    }
}

/// `SymMatrix2` from integer ROW-major rows.
pub fn s2i(rows: [[i64; 2]; 2]) -> SymMatrix2<Fixed> {
    s2(scaled2(rows))
}

/// `SymMatrix3` from integer ROW-major rows.
pub fn s3i(rows: [[i64; 3]; 3]) -> SymMatrix3<Fixed> {
    let [[a, b, c], [d, e, f], [g, h, i]] = rows;
    s3(
        [
            [a * ONE_RAW, b * ONE_RAW, c * ONE_RAW], [d * ONE_RAW, e * ONE_RAW, f * ONE_RAW],
            [g * ONE_RAW, h * ONE_RAW, i * ONE_RAW],
        ],
    )
}

// --- comparisons in raw units
// ---------------------------------------------------------------------

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

/// Largest component-wise `|a - b|` in raw units, over the four 3x3 blocks.
pub fn max_ulp_diff6(a: Matrix6<Fixed>, b: Matrix6<Fixed>) -> u128 {
    let mut e = max_ulp_diff3(a.m11, b.m11);
    e = core::cmp::max(e, max_ulp_diff3(a.m21, b.m21));
    e = core::cmp::max(e, max_ulp_diff3(a.m12, b.m12));
    core::cmp::max(e, max_ulp_diff3(a.m22, b.m22))
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
    let e = core::cmp::max(e, ulp_diff(a.z, b.z));
    core::cmp::max(e, ulp_diff(a.w, b.w))
}

/// Largest component-wise `|a - b|` in raw units, over the two 3-blocks.
pub fn max_ulp_diff_v6(a: Vector6<Fixed>, b: Vector6<Fixed>) -> u128 {
    core::cmp::max(max_ulp_diff_v3(a.a, b.a), max_ulp_diff_v3(a.b, b.b))
}
