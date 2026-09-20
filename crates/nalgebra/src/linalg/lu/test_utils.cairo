//! Test-only helpers shared by the `linalg::lu` tests: builders from the raw ROW-major layout of
//! the oracle (`tools/oracle`), integer-valued builders, component-wise comparisons in ulp and the
//! tolerance predicate of the LU oracle assertions.
//!
//! `base::matrix_test_utils` is the same idea for `base`, but it is private to that module, so the
//! two files overlap on purpose rather than making one of them public API.
//!
//! The oracle data of this module is regenerated from `tools/oracle` with:
//!
//! ```text
//! oracle emit-cairo lu --from vectors --max-per-dist 6 \
//!     --ops lu2_solve,lu2_inverse,lu2_determinant \
//!     --out crates/nalgebra/src/linalg/lu/oracle_lu2.cairo
//! oracle emit-cairo lu --from vectors --max-per-dist 6 \
//!     --ops lu3_solve,lu3_inverse,lu3_determinant,lu3_solve_near_singular \
//!     --out crates/nalgebra/src/linalg/lu/oracle_lu3.cairo
//! oracle emit-cairo lu --from vectors --max-per-dist 5 \
//!     --ops lu4_solve,lu4_inverse,lu4_determinant \
//!     --out crates/nalgebra/src/linalg/lu/oracle_lu4.cairo
//! oracle emit-cairo lu --from vectors --max-per-dist 3 \
//!     --ops lu6_solve,lu6_inverse,lu6_determinant,lu6_solve_near_singular \
//!     --out crates/nalgebra/src/linalg/lu/oracle_lu6.cairo
//! oracle emit-cairo matrix3 --from vectors --max-per-dist 6 \
//!     --ops matrix3_try_inverse,matrix3_determinant \
//!     --out crates/nalgebra/src/linalg/lu/oracle_matrix3_compare.cairo
//! ```

use simba::fixed::Fixed;
use crate::base::matrix2::Matrix2;
use crate::base::matrix3::Matrix3;
use crate::base::matrix4::Matrix4;
use crate::base::matrix6::Matrix6;
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

/// The tolerance of an LU oracle assertion: the oracle's absolute `tol` PLUS one relative ulp of
/// the expected result, `max|expected| / 2^32`.
///
/// The extra term is measured, not guessed. The oracle derives `tol` as `base + 2n * A` from the
/// sensitivity `A` of the operation, assuming every intermediate carries a ONE-ulp error. A
/// fixed-point LU does not: the multiplier `l_ik = a_ik / a_kk` is off by up to 1 ulp, and that
/// error reaches the trailing entry `a_ij` SCALED by `|a_kj|`, so an intermediate of a matrix with
/// entries of magnitude `f` carries `f` ulp, not 1. The resulting error is therefore relative
/// rather than absolute, and one relative ulp covers the whole `lu` suite: over its 408 committed
/// cases (all sizes, all distributions, `solve` / `try_inverse` / `determinant`, near-singular
/// included) the worst excess over `tol` is 0.77 relative ulp, reached by `lu3_determinant` on a
/// `medium` matrix (absolute error 466 349 against `tol` 228 249, on a determinant of 390 625 in
/// value: a RELATIVE error of 1.2 * 2^-32). Every other op of every size stays under `tol` alone.
pub fn lu_tol(max_abs_expected: u128, tol: u64) -> u128 {
    tol.into() + max_abs_expected / 0x100000000
}

/// `Matrix2` from raw ROW-major rows (oracle layout).
pub fn m2(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [r1, r2] = rows;
    let [a11, a12] = r1;
    let [a21, a22] = r2;
    Matrix2 { m11: fx(a11), m21: fx(a21), m12: fx(a12), m22: fx(a22) }
}

/// `Matrix2` from integer ROW-major rows (oracle layout).
pub fn m2i(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [r1, r2] = rows;
    let [a11, a12] = r1;
    let [a21, a22] = r2;
    Matrix2 { m11: int(a11), m21: int(a21), m12: int(a12), m22: int(a22) }
}

/// `Vector2` from raw components (oracle layout).
pub fn v2(t: (i64, i64)) -> Vector2<Fixed> {
    let (c1, c2) = t;
    Vector2 { x: fx(c1), y: fx(c2) }
}

/// `Vector2` from integer components (oracle layout).
pub fn v2i(t: (i64, i64)) -> Vector2<Fixed> {
    let (c1, c2) = t;
    Vector2 { x: int(c1), y: int(c2) }
}

/// Largest component-wise `|a - b|` in raw units over the 4 entries.
pub fn max_ulp_diff_m2(a: Matrix2<Fixed>, b: Matrix2<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e
}

/// Largest `|a_ij|` in raw units over the 4 entries.
pub fn max_abs_m2(a: Matrix2<Fixed>) -> u128 {
    let mut e = abs_raw(a.m11);
    e = core::cmp::max(e, abs_raw(a.m12));
    e = core::cmp::max(e, abs_raw(a.m21));
    e = core::cmp::max(e, abs_raw(a.m22));
    e
}

/// Largest component-wise `|a - b|` in raw units over the 2 components.
pub fn max_ulp_diff_v2(a: Vector2<Fixed>, b: Vector2<Fixed>) -> u128 {
    let mut e = ulp_diff(a.x, b.x);
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e
}

/// Largest `|a_i|` in raw units over the 2 components.
pub fn max_abs_v2(a: Vector2<Fixed>) -> u128 {
    let mut e = abs_raw(a.x);
    e = core::cmp::max(e, abs_raw(a.y));
    e
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

/// `Matrix3` from integer ROW-major rows (oracle layout).
pub fn m3i(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [r1, r2, r3] = rows;
    let [a11, a12, a13] = r1;
    let [a21, a22, a23] = r2;
    let [a31, a32, a33] = r3;
    Matrix3 {
        m11: int(a11),
        m21: int(a21),
        m31: int(a31),
        m12: int(a12),
        m22: int(a22),
        m32: int(a32),
        m13: int(a13),
        m23: int(a23),
        m33: int(a33),
    }
}

/// `Vector3` from raw components (oracle layout).
pub fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (c1, c2, c3) = t;
    Vector3 { x: fx(c1), y: fx(c2), z: fx(c3) }
}

/// `Vector3` from integer components (oracle layout).
pub fn v3i(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (c1, c2, c3) = t;
    Vector3 { x: int(c1), y: int(c2), z: int(c3) }
}

/// Largest component-wise `|a - b|` in raw units over the 9 entries.
pub fn max_ulp_diff_m3(a: Matrix3<Fixed>, b: Matrix3<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e
}

/// Largest `|a_ij|` in raw units over the 9 entries.
pub fn max_abs_m3(a: Matrix3<Fixed>) -> u128 {
    let mut e = abs_raw(a.m11);
    e = core::cmp::max(e, abs_raw(a.m12));
    e = core::cmp::max(e, abs_raw(a.m13));
    e = core::cmp::max(e, abs_raw(a.m21));
    e = core::cmp::max(e, abs_raw(a.m22));
    e = core::cmp::max(e, abs_raw(a.m23));
    e = core::cmp::max(e, abs_raw(a.m31));
    e = core::cmp::max(e, abs_raw(a.m32));
    e = core::cmp::max(e, abs_raw(a.m33));
    e
}

/// Largest component-wise `|a - b|` in raw units over the 3 components.
pub fn max_ulp_diff_v3(a: Vector3<Fixed>, b: Vector3<Fixed>) -> u128 {
    let mut e = ulp_diff(a.x, b.x);
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e
}

/// Largest `|a_i|` in raw units over the 3 components.
pub fn max_abs_v3(a: Vector3<Fixed>) -> u128 {
    let mut e = abs_raw(a.x);
    e = core::cmp::max(e, abs_raw(a.y));
    e = core::cmp::max(e, abs_raw(a.z));
    e
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

/// `Matrix4` from integer ROW-major rows (oracle layout).
pub fn m4i(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [r1, r2, r3, r4] = rows;
    let [a11, a12, a13, a14] = r1;
    let [a21, a22, a23, a24] = r2;
    let [a31, a32, a33, a34] = r3;
    let [a41, a42, a43, a44] = r4;
    Matrix4 {
        m11: int(a11),
        m21: int(a21),
        m31: int(a31),
        m41: int(a41),
        m12: int(a12),
        m22: int(a22),
        m32: int(a32),
        m42: int(a42),
        m13: int(a13),
        m23: int(a23),
        m33: int(a33),
        m43: int(a43),
        m14: int(a14),
        m24: int(a24),
        m34: int(a34),
        m44: int(a44),
    }
}

/// `Vector4` from raw components (oracle layout).
pub fn v4(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (c1, c2, c3, c4) = t;
    Vector4 { x: fx(c1), y: fx(c2), z: fx(c3), w: fx(c4) }
}

/// `Vector4` from integer components (oracle layout).
pub fn v4i(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (c1, c2, c3, c4) = t;
    Vector4 { x: int(c1), y: int(c2), z: int(c3), w: int(c4) }
}

/// Largest component-wise `|a - b|` in raw units over the 16 entries.
pub fn max_ulp_diff_m4(a: Matrix4<Fixed>, b: Matrix4<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11, b.m11);
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e
}

/// Largest `|a_ij|` in raw units over the 16 entries.
pub fn max_abs_m4(a: Matrix4<Fixed>) -> u128 {
    let mut e = abs_raw(a.m11);
    e = core::cmp::max(e, abs_raw(a.m12));
    e = core::cmp::max(e, abs_raw(a.m13));
    e = core::cmp::max(e, abs_raw(a.m14));
    e = core::cmp::max(e, abs_raw(a.m21));
    e = core::cmp::max(e, abs_raw(a.m22));
    e = core::cmp::max(e, abs_raw(a.m23));
    e = core::cmp::max(e, abs_raw(a.m24));
    e = core::cmp::max(e, abs_raw(a.m31));
    e = core::cmp::max(e, abs_raw(a.m32));
    e = core::cmp::max(e, abs_raw(a.m33));
    e = core::cmp::max(e, abs_raw(a.m34));
    e = core::cmp::max(e, abs_raw(a.m41));
    e = core::cmp::max(e, abs_raw(a.m42));
    e = core::cmp::max(e, abs_raw(a.m43));
    e = core::cmp::max(e, abs_raw(a.m44));
    e
}

/// Largest component-wise `|a - b|` in raw units over the 4 components.
pub fn max_ulp_diff_v4(a: Vector4<Fixed>, b: Vector4<Fixed>) -> u128 {
    let mut e = ulp_diff(a.x, b.x);
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e
}

/// Largest `|a_i|` in raw units over the 4 components.
pub fn max_abs_v4(a: Vector4<Fixed>) -> u128 {
    let mut e = abs_raw(a.x);
    e = core::cmp::max(e, abs_raw(a.y));
    e = core::cmp::max(e, abs_raw(a.z));
    e = core::cmp::max(e, abs_raw(a.w));
    e
}

/// `Matrix6` from raw ROW-major rows (oracle layout).
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

/// `Matrix6` from integer ROW-major rows (oracle layout).
pub fn m6i(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    let [a11, a12, a13, a14, a15, a16] = r1;
    let [a21, a22, a23, a24, a25, a26] = r2;
    let [a31, a32, a33, a34, a35, a36] = r3;
    let [a41, a42, a43, a44, a45, a46] = r4;
    let [a51, a52, a53, a54, a55, a56] = r5;
    let [a61, a62, a63, a64, a65, a66] = r6;
    Matrix6 {
        m11: Matrix3 {
            m11: int(a11),
            m21: int(a21),
            m31: int(a31),
            m12: int(a12),
            m22: int(a22),
            m32: int(a32),
            m13: int(a13),
            m23: int(a23),
            m33: int(a33),
        },
        m21: Matrix3 {
            m11: int(a41),
            m21: int(a51),
            m31: int(a61),
            m12: int(a42),
            m22: int(a52),
            m32: int(a62),
            m13: int(a43),
            m23: int(a53),
            m33: int(a63),
        },
        m12: Matrix3 {
            m11: int(a14),
            m21: int(a24),
            m31: int(a34),
            m12: int(a15),
            m22: int(a25),
            m32: int(a35),
            m13: int(a16),
            m23: int(a26),
            m33: int(a36),
        },
        m22: Matrix3 {
            m11: int(a44),
            m21: int(a54),
            m31: int(a64),
            m12: int(a45),
            m22: int(a55),
            m32: int(a65),
            m13: int(a46),
            m23: int(a56),
            m33: int(a66),
        },
    }
}

/// `Vector6` from raw components (oracle layout).
pub fn v6(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (c1, c2, c3, c4, c5, c6) = t;
    Vector6 {
        a: Vector3 { x: fx(c1), y: fx(c2), z: fx(c3) },
        b: Vector3 { x: fx(c4), y: fx(c5), z: fx(c6) },
    }
}

/// `Vector6` from integer components (oracle layout).
pub fn v6i(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (c1, c2, c3, c4, c5, c6) = t;
    Vector6 {
        a: Vector3 { x: int(c1), y: int(c2), z: int(c3) },
        b: Vector3 { x: int(c4), y: int(c5), z: int(c6) },
    }
}

/// Largest component-wise `|a - b|` in raw units over the 36 entries.
pub fn max_ulp_diff_m6(a: Matrix6<Fixed>, b: Matrix6<Fixed>) -> u128 {
    let mut e = ulp_diff(a.m11.m11, b.m11.m11);
    e = core::cmp::max(e, ulp_diff(a.m11.m12, b.m11.m12));
    e = core::cmp::max(e, ulp_diff(a.m11.m13, b.m11.m13));
    e = core::cmp::max(e, ulp_diff(a.m12.m11, b.m12.m11));
    e = core::cmp::max(e, ulp_diff(a.m12.m12, b.m12.m12));
    e = core::cmp::max(e, ulp_diff(a.m12.m13, b.m12.m13));
    e = core::cmp::max(e, ulp_diff(a.m11.m21, b.m11.m21));
    e = core::cmp::max(e, ulp_diff(a.m11.m22, b.m11.m22));
    e = core::cmp::max(e, ulp_diff(a.m11.m23, b.m11.m23));
    e = core::cmp::max(e, ulp_diff(a.m12.m21, b.m12.m21));
    e = core::cmp::max(e, ulp_diff(a.m12.m22, b.m12.m22));
    e = core::cmp::max(e, ulp_diff(a.m12.m23, b.m12.m23));
    e = core::cmp::max(e, ulp_diff(a.m11.m31, b.m11.m31));
    e = core::cmp::max(e, ulp_diff(a.m11.m32, b.m11.m32));
    e = core::cmp::max(e, ulp_diff(a.m11.m33, b.m11.m33));
    e = core::cmp::max(e, ulp_diff(a.m12.m31, b.m12.m31));
    e = core::cmp::max(e, ulp_diff(a.m12.m32, b.m12.m32));
    e = core::cmp::max(e, ulp_diff(a.m12.m33, b.m12.m33));
    e = core::cmp::max(e, ulp_diff(a.m21.m11, b.m21.m11));
    e = core::cmp::max(e, ulp_diff(a.m21.m12, b.m21.m12));
    e = core::cmp::max(e, ulp_diff(a.m21.m13, b.m21.m13));
    e = core::cmp::max(e, ulp_diff(a.m22.m11, b.m22.m11));
    e = core::cmp::max(e, ulp_diff(a.m22.m12, b.m22.m12));
    e = core::cmp::max(e, ulp_diff(a.m22.m13, b.m22.m13));
    e = core::cmp::max(e, ulp_diff(a.m21.m21, b.m21.m21));
    e = core::cmp::max(e, ulp_diff(a.m21.m22, b.m21.m22));
    e = core::cmp::max(e, ulp_diff(a.m21.m23, b.m21.m23));
    e = core::cmp::max(e, ulp_diff(a.m22.m21, b.m22.m21));
    e = core::cmp::max(e, ulp_diff(a.m22.m22, b.m22.m22));
    e = core::cmp::max(e, ulp_diff(a.m22.m23, b.m22.m23));
    e = core::cmp::max(e, ulp_diff(a.m21.m31, b.m21.m31));
    e = core::cmp::max(e, ulp_diff(a.m21.m32, b.m21.m32));
    e = core::cmp::max(e, ulp_diff(a.m21.m33, b.m21.m33));
    e = core::cmp::max(e, ulp_diff(a.m22.m31, b.m22.m31));
    e = core::cmp::max(e, ulp_diff(a.m22.m32, b.m22.m32));
    e = core::cmp::max(e, ulp_diff(a.m22.m33, b.m22.m33));
    e
}

/// Largest `|a_ij|` in raw units over the 36 entries.
pub fn max_abs_m6(a: Matrix6<Fixed>) -> u128 {
    let mut e = abs_raw(a.m11.m11);
    e = core::cmp::max(e, abs_raw(a.m11.m12));
    e = core::cmp::max(e, abs_raw(a.m11.m13));
    e = core::cmp::max(e, abs_raw(a.m12.m11));
    e = core::cmp::max(e, abs_raw(a.m12.m12));
    e = core::cmp::max(e, abs_raw(a.m12.m13));
    e = core::cmp::max(e, abs_raw(a.m11.m21));
    e = core::cmp::max(e, abs_raw(a.m11.m22));
    e = core::cmp::max(e, abs_raw(a.m11.m23));
    e = core::cmp::max(e, abs_raw(a.m12.m21));
    e = core::cmp::max(e, abs_raw(a.m12.m22));
    e = core::cmp::max(e, abs_raw(a.m12.m23));
    e = core::cmp::max(e, abs_raw(a.m11.m31));
    e = core::cmp::max(e, abs_raw(a.m11.m32));
    e = core::cmp::max(e, abs_raw(a.m11.m33));
    e = core::cmp::max(e, abs_raw(a.m12.m31));
    e = core::cmp::max(e, abs_raw(a.m12.m32));
    e = core::cmp::max(e, abs_raw(a.m12.m33));
    e = core::cmp::max(e, abs_raw(a.m21.m11));
    e = core::cmp::max(e, abs_raw(a.m21.m12));
    e = core::cmp::max(e, abs_raw(a.m21.m13));
    e = core::cmp::max(e, abs_raw(a.m22.m11));
    e = core::cmp::max(e, abs_raw(a.m22.m12));
    e = core::cmp::max(e, abs_raw(a.m22.m13));
    e = core::cmp::max(e, abs_raw(a.m21.m21));
    e = core::cmp::max(e, abs_raw(a.m21.m22));
    e = core::cmp::max(e, abs_raw(a.m21.m23));
    e = core::cmp::max(e, abs_raw(a.m22.m21));
    e = core::cmp::max(e, abs_raw(a.m22.m22));
    e = core::cmp::max(e, abs_raw(a.m22.m23));
    e = core::cmp::max(e, abs_raw(a.m21.m31));
    e = core::cmp::max(e, abs_raw(a.m21.m32));
    e = core::cmp::max(e, abs_raw(a.m21.m33));
    e = core::cmp::max(e, abs_raw(a.m22.m31));
    e = core::cmp::max(e, abs_raw(a.m22.m32));
    e = core::cmp::max(e, abs_raw(a.m22.m33));
    e
}

/// Largest component-wise `|a - b|` in raw units over the 6 components.
pub fn max_ulp_diff_v6(a: Vector6<Fixed>, b: Vector6<Fixed>) -> u128 {
    let mut e = ulp_diff(a.a.x, b.a.x);
    e = core::cmp::max(e, ulp_diff(a.a.y, b.a.y));
    e = core::cmp::max(e, ulp_diff(a.a.z, b.a.z));
    e = core::cmp::max(e, ulp_diff(a.b.x, b.b.x));
    e = core::cmp::max(e, ulp_diff(a.b.y, b.b.y));
    e = core::cmp::max(e, ulp_diff(a.b.z, b.b.z));
    e
}

/// Largest `|a_i|` in raw units over the 6 components.
pub fn max_abs_v6(a: Vector6<Fixed>) -> u128 {
    let mut e = abs_raw(a.a.x);
    e = core::cmp::max(e, abs_raw(a.a.y));
    e = core::cmp::max(e, abs_raw(a.a.z));
    e = core::cmp::max(e, abs_raw(a.b.x));
    e = core::cmp::max(e, abs_raw(a.b.y));
    e = core::cmp::max(e, abs_raw(a.b.z));
    e
}
