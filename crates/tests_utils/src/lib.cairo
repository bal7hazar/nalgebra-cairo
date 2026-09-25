//! Package `nalgebra_tests_utils` (WP 8.1c): the public-API part of the former
//! `crates/nalgebra/src/base/matrix_test_utils.cairo`, shared by the test-only packages
//! (`nalgebra_tests_base`, `nalgebra_tests_geometry`, `nalgebra_tests_linalg`), plus the
//! test-only `PartialEq` of the decompositions. Not a public API of `nalgebra`; the in-crate
//! tests keep their own copy of the helpers they need (`matrix_test_utils`).
//!
//! Test-only helpers of the whole crate: the builders of every `base` and `geometry` type from raw
//! Q32.32 values (the layout of the oracle, `tools/oracle`) or from integers, the component-wise
//! comparisons in raw units (ulp) and the tolerance predicates of the oracle assertions. It is the
//! SINGLE home of these helpers: the tests and benches of `base`, `geometry` and `linalg` import
//! them from here instead of restating them.
//!
//! # Naming
//!
//! - `fx(raw)` / `int(v)`: one `Fixed` from its raw value / from an integer.
//! - `isom2t` / `isom3t` / `simm2t` / `simm3t` (WP 8.4-P09b): the rotation-matrix poses from the
//!   oracle layout (translation tuple, ROW-major rotation rows, scaling).
//! - `vN`, `pN`, `uN`, `q`, `uq`, `uc`, `tN`, `iso2`, `iso3`, `sim2`, `sim3` take the raw
//!   components as SEPARATE arguments (`v3(x, y, z)`, `q(w, i, j, k)`, quaternions in the
//!   `(w, i, j, k)` order of the oracle); the suffix `i` builds from integers (`v3i(1, 2, 3)`).
//! - The suffix `t` is the same builder taking ONE tuple, the shape of the oracle tables:
//!   `v3t((x, y, z))`, `v3it((1, 2, 3))`, `uqt((w, i, j, k))`, `iso3t(((tx, ty, tz), (w, i, j,
//!   k)))`. Both shapes were in use before the builders were shared (hundreds of call sites
//!   each); they are kept under two explicit names rather than one name with two signatures.
//! - `mN(rows)` / `mNi(rows)`, `r2(rows)` / `r3(rows)` / `r3i(rows)`: square matrices from
//!   ROW-major rows (upstream `MatrixN::new` order), the oracle layout.
//! - `s2((m11, m12, m22))` / `s3((m11, m12, m13, m22, m23, m33))` and their `i` forms: a
//!   `SymMatrixN` from its upper triangle. `s2r(rows)` / `s3r(rows)` and `s2ir` / `s3ir`: the
//!   `SymMatrixN` view (upper triangle) of a full ROW-major matrix, the shape of the oracle's
//!   symmetric inputs, which are exactly symmetric in raw units so that nothing is lost.
//! - `pers4t((m11, m22, m33, m34))` / `ortho6t((m11, m14, m22, m24, m33, m34))` (WP 8.4-P11b):
//!   a `Perspective3` / `Orthographic3` from the raw structural entries of its matrix (oracle
//!   layout, `from_matrix_unchecked`).
//! - `ulp_diff`, `max_ulp_diffN` (matrices), `max_ulp_diff_vN`, `max_ulp_diff_sN`,
//!   `max_ulp_diff_q`, `max_ulp_diff_uc`: largest component-wise `|a - b|` in raw units.
//!   `abs_raw`, `max_abs_*`, `amax_*`, `oracle_tol`, `excess`, `orthonormality_error_mN`: the
//!   magnitudes and predicates the `linalg` oracle assertions are expressed with.
//!
//! The builders feed `black_box` in the benches: their bodies are part of the measured gas, so
//! changing one moves the snapshots in `gas/`.
//!
//! # Oracle data
//!
//! `base/oracle_matrix{2,3,4}.cairo`, `base/oracle_matrix{2,3,4}_inverse.cairo`:
//!
//! ```text
//! oracle emit-cairo matrixN --from vectors --max-per-dist 4 --ops <all but the four below> \
//!     --out crates/nalgebra/src/base/oracle_matrixN.cairo
//! oracle emit-cairo matrixN --from vectors --max-per-dist 10 --ops matrixN_determinant,\
//!     matrixN_try_inverse,matrixN_try_inverse_singular,matrixN_try_inverse_near_singular \
//!     --out crates/nalgebra/src/base/oracle_matrixN_inverse.cairo
//! ```
//!
//! `base/oracle_sym_matrix.cairo` holds the symmetric-matrix vectors. The oracle has no
//! `SymMatrixN` suite (the type is rapier's, not upstream nalgebra's); its `udu` suite is used
//! instead, whose inputs are EXACTLY symmetric positive-definite in raw units and whose `_inverse`
//! outputs are upstream inverses:
//!
//! ```text
//! oracle emit-cairo udu --from vectors --max-per-dist 8 --ops udu2_inverse,udu3_inverse \
//!     --out crates/nalgebra/src/base/oracle_sym_matrix.cairo
//! ```
//!
//! `linalg/oracle_cholesky.cairo`, `linalg/oracle_udu.cairo` (sizes 5 are emitted by neither:
//! there is no `Vector5` / `Matrix5` in `base`, DESIGN D4 ships the sizes physics uses: 2, 3, 4
//! and the 6 of spatial algebra):
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
//! `linalg/lu/oracle_*.cairo`:
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
//!
//! `linalg/qr/oracle_qr*.cairo`, `linalg/oracle_svd.cairo`:
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

use fixed::Fixed;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::matrix3::Matrix3;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::matrix6::Matrix6;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::base::vector4::Vector4;
use nalgebra::base::vector6::Vector6;
use nalgebra::geometry::isometry2::Isometry2;
use nalgebra::geometry::isometry3::Isometry3;
use nalgebra::geometry::isometry_matrix2::IsometryMatrix2;
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3;
use nalgebra::geometry::orthographic3::{Orthographic3, Orthographic3Trait};
use nalgebra::geometry::perspective3::{Perspective3, Perspective3Trait};
use nalgebra::geometry::quaternion::Quaternion;
use nalgebra::geometry::rotation2::Rotation2;
use nalgebra::geometry::rotation3::Rotation3;
use nalgebra::geometry::similarity2::Similarity2;
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::similarity_matrix2::SimilarityMatrix2;
use nalgebra::geometry::similarity_matrix3::SimilarityMatrix3;
use nalgebra::geometry::translation2::Translation2;
use nalgebra::geometry::translation3::Translation3;
use nalgebra::geometry::unit_complex::UnitComplex;
use nalgebra::geometry::unit_quaternion::UnitQuaternion;
use nalgebra::linalg::cholesky::{Cholesky2, Cholesky3, Cholesky4, Cholesky6};
use nalgebra::linalg::lu::lu2::Lu2;
use nalgebra::linalg::lu::lu3::Lu3;
use nalgebra::linalg::lu::lu4::Lu4;
use nalgebra::linalg::lu::lu6::Lu6;
use nalgebra::linalg::lu::{Perm2, Perm3, Perm4, Perm6};
use nalgebra::linalg::qr::qr2::Qr2;
use nalgebra::linalg::qr::qr3::Qr3;
use nalgebra::linalg::qr::qr4::Qr4;
use nalgebra::linalg::svd2::Svd2;
use nalgebra::linalg::svd3::Svd3;

/// 2^32: the raw value of 1.
pub const ONE_RAW: i64 = 0x100000000;

// --- scalars ------------------------------------------------------------------------------------

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
pub fn v2(x: i64, y: i64) -> Vector2<Fixed> {
    Vector2 { x: fx(x), y: fx(y) }
}

/// `Vector3` from raw components.
pub fn v3(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// `Vector4` from raw components.
pub fn v4(x: i64, y: i64, z: i64, w: i64) -> Vector4<Fixed> {
    Vector4 { x: fx(x), y: fx(y), z: fx(z), w: fx(w) }
}

/// `Vector6` from raw components, in the upstream order `(x, y, z, w, a, b)`.
pub fn v6(x: i64, y: i64, z: i64, w: i64, a: i64, b: i64) -> Vector6<Fixed> {
    Vector6 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a), b: fx(b) }
}

/// `Vector2` from a tuple of raw components (oracle layout).
pub fn v2t(t: (i64, i64)) -> Vector2<Fixed> {
    let (x, y) = t;
    Vector2 { x: fx(x), y: fx(y) }
}

/// `Vector3` from a tuple of raw components (oracle layout).
pub fn v3t(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// `Vector4` from a tuple of raw components (oracle layout).
pub fn v4t(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (x, y, z, w) = t;
    Vector4 { x: fx(x), y: fx(y), z: fx(z), w: fx(w) }
}

/// `Vector6` from a tuple of raw components (oracle layout), in the upstream order
/// `(x, y, z, w, a, b)`.
pub fn v6t(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Vector6 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a), b: fx(b) }
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

/// `Vector6` from integers, in the upstream order `(x, y, z, w, a, b)`.
pub fn v6i(x: i64, y: i64, z: i64, w: i64, a: i64, b: i64) -> Vector6<Fixed> {
    Vector6 { x: int(x), y: int(y), z: int(z), w: int(w), a: int(a), b: int(b) }
}

/// `Vector2` from a tuple of integers.
pub fn v2it(t: (i64, i64)) -> Vector2<Fixed> {
    let (x, y) = t;
    Vector2 { x: int(x), y: int(y) }
}

/// `Vector3` from a tuple of integers.
pub fn v3it(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: int(x), y: int(y), z: int(z) }
}

/// `Vector4` from a tuple of integers.
pub fn v4it(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (x, y, z, w) = t;
    Vector4 { x: int(x), y: int(y), z: int(z), w: int(w) }
}

/// `Vector6` from a tuple of integers, in the upstream order `(x, y, z, w, a, b)`.
pub fn v6it(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Vector6 { x: int(x), y: int(y), z: int(z), w: int(w), a: int(a), b: int(b) }
}

/// `Unit<Vector2>` from raw components, WITHOUT normalisation.
pub fn u2(x: i64, y: i64) -> Unit<Vector2<Fixed>> {
    Unit { value: v2(x, y) }
}

/// `Unit<Vector3>` from raw components, WITHOUT normalisation.
pub fn u3(x: i64, y: i64, z: i64) -> Unit<Vector3<Fixed>> {
    Unit { value: v3(x, y, z) }
}

/// `Unit<Vector4>` from raw components, WITHOUT normalisation.
pub fn u4(x: i64, y: i64, z: i64, w: i64) -> Unit<Vector4<Fixed>> {
    Unit { value: v4(x, y, z, w) }
}

/// `Unit<Vector3>` from a tuple of raw components, WITHOUT normalisation.
pub fn u3t(t: (i64, i64, i64)) -> Unit<Vector3<Fixed>> {
    Unit { value: v3t(t) }
}

// --- points -------------------------------------------------------------------------------------

/// `Point2` from raw coordinates.
pub fn p2(x: i64, y: i64) -> Point2<Fixed> {
    Point2 { x: fx(x), y: fx(y) }
}

/// `Point3` from raw coordinates.
pub fn p3(x: i64, y: i64, z: i64) -> Point3<Fixed> {
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// `Point2` from a tuple of raw coordinates (oracle layout).
pub fn p2t(t: (i64, i64)) -> Point2<Fixed> {
    let (x, y) = t;
    Point2 { x: fx(x), y: fx(y) }
}

/// `Point3` from a tuple of raw coordinates (oracle layout).
pub fn p3t(t: (i64, i64, i64)) -> Point3<Fixed> {
    let (x, y, z) = t;
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

// --- matrices -----------------------------------------------------------------------------------

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

/// The 3x3 block of rows 1-3, columns 1-3 of a `Matrix6` (upstream `fixed_view::<3, 3>(0, 0)`):
/// test-only, the 3x3-block formulations kept as evidence (and the block-wise tolerance helpers)
/// read the flat `Matrix6` through these (moves only).
#[inline(always)]
pub fn m6_block11(m: Matrix6<Fixed>) -> Matrix3<Fixed> {
    Matrix3 {
        m11: m.m11,
        m21: m.m21,
        m31: m.m31,
        m12: m.m12,
        m22: m.m22,
        m32: m.m32,
        m13: m.m13,
        m23: m.m23,
        m33: m.m33,
    }
}

/// The 3x3 block of rows 4-6, columns 1-3 of a `Matrix6` (upstream `fixed_view::<3, 3>(3, 0)`).
#[inline(always)]
pub fn m6_block21(m: Matrix6<Fixed>) -> Matrix3<Fixed> {
    Matrix3 {
        m11: m.m41,
        m21: m.m51,
        m31: m.m61,
        m12: m.m42,
        m22: m.m52,
        m32: m.m62,
        m13: m.m43,
        m23: m.m53,
        m33: m.m63,
    }
}

/// The 3x3 block of rows 1-3, columns 4-6 of a `Matrix6` (upstream `fixed_view::<3, 3>(0, 3)`).
#[inline(always)]
pub fn m6_block12(m: Matrix6<Fixed>) -> Matrix3<Fixed> {
    Matrix3 {
        m11: m.m14,
        m21: m.m24,
        m31: m.m34,
        m12: m.m15,
        m22: m.m25,
        m32: m.m35,
        m13: m.m16,
        m23: m.m26,
        m33: m.m36,
    }
}

/// The 3x3 block of rows 4-6, columns 4-6 of a `Matrix6` (upstream `fixed_view::<3, 3>(3, 3)`).
#[inline(always)]
pub fn m6_block22(m: Matrix6<Fixed>) -> Matrix3<Fixed> {
    Matrix3 {
        m11: m.m44,
        m21: m.m54,
        m31: m.m64,
        m12: m.m45,
        m22: m.m55,
        m32: m.m65,
        m13: m.m46,
        m23: m.m56,
        m33: m.m66,
    }
}

/// The `Matrix6` made of four 3x3 blocks (`bIJ`: block-row `I`, block-column `J`).
#[inline(always)]
pub fn m6_from_blocks(
    b11: Matrix3<Fixed>, b21: Matrix3<Fixed>, b12: Matrix3<Fixed>, b22: Matrix3<Fixed>,
) -> Matrix6<Fixed> {
    Matrix6 {
        m11: b11.m11,
        m21: b11.m21,
        m31: b11.m31,
        m41: b21.m11,
        m51: b21.m21,
        m61: b21.m31,
        m12: b11.m12,
        m22: b11.m22,
        m32: b11.m32,
        m42: b21.m12,
        m52: b21.m22,
        m62: b21.m32,
        m13: b11.m13,
        m23: b11.m23,
        m33: b11.m33,
        m43: b21.m13,
        m53: b21.m23,
        m63: b21.m33,
        m14: b12.m11,
        m24: b12.m21,
        m34: b12.m31,
        m44: b22.m11,
        m54: b22.m21,
        m64: b22.m31,
        m15: b12.m12,
        m25: b12.m22,
        m35: b12.m32,
        m45: b22.m12,
        m55: b22.m22,
        m65: b22.m32,
        m16: b12.m13,
        m26: b12.m23,
        m36: b12.m33,
        m46: b22.m13,
        m56: b22.m23,
        m66: b22.m33,
    }
}

/// Components `x, y, z` of a `Vector6` (upstream `fixed_rows::<3>(0)`).
#[inline(always)]
pub fn v6_head(v: Vector6<Fixed>) -> Vector3<Fixed> {
    Vector3 { x: v.x, y: v.y, z: v.z }
}

/// Components `w, a, b` of a `Vector6` (upstream `fixed_rows::<3>(3)`).
#[inline(always)]
pub fn v6_tail(v: Vector6<Fixed>) -> Vector3<Fixed> {
    Vector3 { x: v.w, y: v.a, z: v.b }
}

/// The `Vector6` `(head.x, head.y, head.z, tail.x, tail.y, tail.z)`.
#[inline(always)]
pub fn v6_from_halves(head: Vector3<Fixed>, tail: Vector3<Fixed>) -> Vector6<Fixed> {
    Vector6 { x: head.x, y: head.y, z: head.z, w: tail.x, a: tail.y, b: tail.z }
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
        m11: fx(a11),
        m21: fx(a21),
        m31: fx(a31),
        m12: fx(a12),
        m22: fx(a22),
        m32: fx(a32),
        m13: fx(a13),
        m23: fx(a23),
        m33: fx(a33),
        m41: fx(a41),
        m51: fx(a51),
        m61: fx(a61),
        m42: fx(a42),
        m52: fx(a52),
        m62: fx(a62),
        m43: fx(a43),
        m53: fx(a53),
        m63: fx(a63),
        m14: fx(a14),
        m24: fx(a24),
        m34: fx(a34),
        m15: fx(a15),
        m25: fx(a25),
        m35: fx(a35),
        m16: fx(a16),
        m26: fx(a26),
        m36: fx(a36),
        m44: fx(a44),
        m54: fx(a54),
        m64: fx(a64),
        m45: fx(a45),
        m55: fx(a55),
        m65: fx(a65),
        m46: fx(a46),
        m56: fx(a56),
        m66: fx(a66),
    }
}

/// `Matrix6` from integer ROW-major rows.
pub fn m6i(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    let [a11, a12, a13, a14, a15, a16] = r1;
    let [a21, a22, a23, a24, a25, a26] = r2;
    let [a31, a32, a33, a34, a35, a36] = r3;
    let [a41, a42, a43, a44, a45, a46] = r4;
    let [a51, a52, a53, a54, a55, a56] = r5;
    let [a61, a62, a63, a64, a65, a66] = r6;
    Matrix6 {
        m11: int(a11),
        m21: int(a21),
        m31: int(a31),
        m12: int(a12),
        m22: int(a22),
        m32: int(a32),
        m13: int(a13),
        m23: int(a23),
        m33: int(a33),
        m41: int(a41),
        m51: int(a51),
        m61: int(a61),
        m42: int(a42),
        m52: int(a52),
        m62: int(a62),
        m43: int(a43),
        m53: int(a53),
        m63: int(a63),
        m14: int(a14),
        m24: int(a24),
        m34: int(a34),
        m15: int(a15),
        m25: int(a25),
        m35: int(a35),
        m16: int(a16),
        m26: int(a26),
        m36: int(a36),
        m44: int(a44),
        m54: int(a54),
        m64: int(a64),
        m45: int(a45),
        m55: int(a55),
        m65: int(a65),
        m46: int(a46),
        m56: int(a56),
        m66: int(a66),
    }
}

// --- geometry -----------------------------------------------------------------------------------

/// `Quaternion` from raw components, in the `(w, i, j, k)` order of the oracle.
pub fn q(w: i64, i: i64, j: i64, k: i64) -> Quaternion<Fixed> {
    Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) }
}

/// `Quaternion` from a tuple of raw components `(w, i, j, k)` (oracle layout).
pub fn qt(t: (i64, i64, i64, i64)) -> Quaternion<Fixed> {
    let (w, i, j, k) = t;
    Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) }
}

/// `Quaternion` from integers, in the `(w, i, j, k)` order.
pub fn qi(w: i64, i: i64, j: i64, k: i64) -> Quaternion<Fixed> {
    Quaternion { i: int(i), j: int(j), k: int(k), w: int(w) }
}

/// `UnitQuaternion` from raw components `(w, i, j, k)`, WITHOUT normalisation.
pub fn uq(w: i64, i: i64, j: i64, k: i64) -> UnitQuaternion<Fixed> {
    UnitQuaternion { quaternion: Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) } }
}

/// `UnitQuaternion` from a tuple of raw components `(w, i, j, k)` (oracle layout), WITHOUT
/// normalisation.
pub fn uqt(t: (i64, i64, i64, i64)) -> UnitQuaternion<Fixed> {
    let (w, i, j, k) = t;
    UnitQuaternion { quaternion: Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) } }
}

/// `UnitComplex` from raw components `(re, im)`, WITHOUT normalisation.
pub fn uc(re: i64, im: i64) -> UnitComplex<Fixed> {
    UnitComplex { re: fx(re), im: fx(im) }
}

/// `UnitComplex` from a tuple of raw components `(re, im)` (oracle layout), WITHOUT
/// normalisation.
pub fn uct(t: (i64, i64)) -> UnitComplex<Fixed> {
    let (re, im) = t;
    UnitComplex { re: fx(re), im: fx(im) }
}

/// `Rotation2` from raw ROW-major rows (oracle layout), WITHOUT orthonormalisation.
pub fn r2(rows: [[i64; 2]; 2]) -> Rotation2<Fixed> {
    let [[m11, m12], [m21, m22]] = rows;
    Rotation2 { matrix: Matrix2 { m11: fx(m11), m21: fx(m21), m12: fx(m12), m22: fx(m22) } }
}

/// `Rotation3` from raw ROW-major rows (oracle layout), WITHOUT orthonormalisation.
pub fn r3(rows: [[i64; 3]; 3]) -> Rotation3<Fixed> {
    Rotation3 { matrix: m3(rows) }
}

/// `Rotation3` from integer ROW-major rows.
pub fn r3i(rows: [[i64; 3]; 3]) -> Rotation3<Fixed> {
    Rotation3 { matrix: m3i(rows) }
}

/// `Translation2` from raw components.
pub fn t2(x: i64, y: i64) -> Translation2<Fixed> {
    Translation2 { vector: Vector2 { x: fx(x), y: fx(y) } }
}

/// `Translation3` from raw components.
pub fn t3(x: i64, y: i64, z: i64) -> Translation3<Fixed> {
    Translation3 { vector: Vector3 { x: fx(x), y: fx(y), z: fx(z) } }
}

/// `Translation2` from a tuple of raw components (oracle layout).
pub fn t2t(t: (i64, i64)) -> Translation2<Fixed> {
    Translation2 { vector: v2t(t) }
}

/// `Translation3` from a tuple of raw components (oracle layout).
pub fn t3t(t: (i64, i64, i64)) -> Translation3<Fixed> {
    Translation3 { vector: v3t(t) }
}

/// `Isometry2` from the raw translation `(tx, ty)` and rotation `(re, im)`.
pub fn iso2(tx: i64, ty: i64, re: i64, im: i64) -> Isometry2<Fixed> {
    Isometry2 { rotation: uc(re, im), translation: Translation2 { vector: v2(tx, ty) } }
}

/// `Isometry2` from `((tx, ty), (re, im))` in raw units (oracle layout).
pub fn iso2t(t: ((i64, i64), (i64, i64))) -> Isometry2<Fixed> {
    let (tr, rot) = t;
    Isometry2 { rotation: uct(rot), translation: Translation2 { vector: v2t(tr) } }
}

/// `Isometry3` from the raw translation `(tx, ty, tz)` and rotation `(w, i, j, k)`.
pub fn iso3(t: (i64, i64, i64), r: (i64, i64, i64, i64)) -> Isometry3<Fixed> {
    let (tx, ty, tz) = t;
    let (w, i, j, k) = r;
    Isometry3 { rotation: uq(w, i, j, k), translation: Translation3 { vector: v3(tx, ty, tz) } }
}

/// `Isometry3` from `((tx, ty, tz), (w, i, j, k))` in raw units (oracle layout).
pub fn iso3t(t: ((i64, i64, i64), (i64, i64, i64, i64))) -> Isometry3<Fixed> {
    let (tr, rot) = t;
    Isometry3 { rotation: uqt(rot), translation: Translation3 { vector: v3t(tr) } }
}

/// `Similarity2` from the raw translation `(tx, ty)`, rotation `(re, im)` and scaling.
pub fn sim2(tx: i64, ty: i64, re: i64, im: i64, scaling: i64) -> Similarity2<Fixed> {
    Similarity2 {
        isometry: Isometry2 {
            rotation: uc(re, im), translation: Translation2 { vector: v2(tx, ty) },
        },
        scaling: fx(scaling),
    }
}

/// `Similarity2` from `((tx, ty), (re, im), scaling)` in raw units (oracle layout).
pub fn sim2t(t: ((i64, i64), (i64, i64), i64)) -> Similarity2<Fixed> {
    let (tr, rot, scaling) = t;
    Similarity2 {
        isometry: Isometry2 { rotation: uct(rot), translation: Translation2 { vector: v2t(tr) } },
        scaling: fx(scaling),
    }
}

/// `Similarity3` from the raw translation `(tx, ty, tz)`, rotation `(w, i, j, k)` and scaling.
pub fn sim3(t: (i64, i64, i64), r: (i64, i64, i64, i64), scaling: i64) -> Similarity3<Fixed> {
    let (tx, ty, tz) = t;
    let (w, i, j, k) = r;
    Similarity3 {
        isometry: Isometry3 {
            rotation: uq(w, i, j, k), translation: Translation3 { vector: v3(tx, ty, tz) },
        },
        scaling: fx(scaling),
    }
}

/// `Similarity3` from `((tx, ty, tz), (w, i, j, k), scaling)` in raw units (oracle layout).
pub fn sim3t(t: ((i64, i64, i64), (i64, i64, i64, i64), i64)) -> Similarity3<Fixed> {
    let (tr, rot, scaling) = t;
    Similarity3 {
        isometry: Isometry3 { rotation: uqt(rot), translation: Translation3 { vector: v3t(tr) } },
        scaling: fx(scaling),
    }
}

/// `IsometryMatrix2` from the raw translation `(tx, ty)` and ROW-major rotation rows (oracle
/// layout), WITHOUT orthonormalisation.
pub fn isom2t(t: (i64, i64), r: [[i64; 2]; 2]) -> IsometryMatrix2<Fixed> {
    IsometryMatrix2 { rotation: r2(r), translation: Translation2 { vector: v2t(t) } }
}

/// `IsometryMatrix3` from the raw translation `(tx, ty, tz)` and ROW-major rotation rows.
pub fn isom3t(t: (i64, i64, i64), r: [[i64; 3]; 3]) -> IsometryMatrix3<Fixed> {
    IsometryMatrix3 { rotation: r3(r), translation: Translation3 { vector: v3t(t) } }
}

/// `SimilarityMatrix2` from the raw translation, ROW-major rotation rows and scaling.
pub fn simm2t(t: (i64, i64), r: [[i64; 2]; 2], scaling: i64) -> SimilarityMatrix2<Fixed> {
    SimilarityMatrix2 { isometry: isom2t(t, r), scaling: fx(scaling) }
}

/// `SimilarityMatrix3` from the raw translation, ROW-major rotation rows and scaling.
pub fn simm3t(t: (i64, i64, i64), r: [[i64; 3]; 3], scaling: i64) -> SimilarityMatrix3<Fixed> {
    SimilarityMatrix3 { isometry: isom3t(t, r), scaling: fx(scaling) }
}

/// `Perspective3` from the raw entries `(m11, m22, m33, m34)` of its matrix (oracle layout; the
/// other entries are those of a perspective: `m43 = -1`, zeros elsewhere).
pub fn pers4t(t: (i64, i64, i64, i64)) -> Perspective3<Fixed> {
    let (m11, m22, m33, m34) = t;
    Perspective3Trait::from_matrix_unchecked(
        m4([[m11, 0, 0, 0], [0, m22, 0, 0], [0, 0, m33, m34], [0, 0, -ONE_RAW, 0]]),
    )
}

/// `Orthographic3` from the raw entries `(m11, m14, m22, m24, m33, m34)` of its matrix (oracle
/// layout; `m44 = 1`, zeros elsewhere).
pub fn ortho6t(t: (i64, i64, i64, i64, i64, i64)) -> Orthographic3<Fixed> {
    let (m11, m14, m22, m24, m33, m34) = t;
    Orthographic3Trait::from_matrix_unchecked(
        m4([[m11, 0, 0, m14], [0, m22, 0, m24], [0, 0, m33, m34], [0, 0, 0, ONE_RAW]]),
    )
}

// --- comparisons in raw units -------------------------------------------------------------------

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
    let mut e = max_ulp_diff3(m6_block11(a), m6_block11(b));
    e = core::cmp::max(e, max_ulp_diff3(m6_block21(a), m6_block21(b)));
    e = core::cmp::max(e, max_ulp_diff3(m6_block12(a), m6_block12(b)));
    core::cmp::max(e, max_ulp_diff3(m6_block22(a), m6_block22(b)))
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
    core::cmp::max(max_ulp_diff_v3(v6_head(a), v6_head(b)), max_ulp_diff_v3(v6_tail(a), v6_tail(b)))
}

/// Largest component-wise `|a - b|` in raw units, over `(i, j, k, w)`.
pub fn max_ulp_diff_q(a: Quaternion<Fixed>, b: Quaternion<Fixed>) -> u128 {
    let mut e = ulp_diff(a.i, b.i);
    e = core::cmp::max(e, ulp_diff(a.j, b.j));
    e = core::cmp::max(e, ulp_diff(a.k, b.k));
    core::cmp::max(e, ulp_diff(a.w, b.w))
}

/// Largest component-wise `|a - b|` in raw units, over `(re, im)`.
pub fn max_ulp_diff_uc(a: UnitComplex<Fixed>, b: UnitComplex<Fixed>) -> u128 {
    core::cmp::max(ulp_diff(a.re, b.re), ulp_diff(a.im, b.im))
}

// --- magnitudes and tolerances of the `linalg` oracle assertions -------------------------------

/// The tolerance of an LU / QR / SVD oracle assertion: the oracle's absolute `tol` PLUS one
/// relative ulp of the expected result, `max|expected| / 2^32`.
///
/// The extra term is measured, not guessed. The oracle derives `tol` as `base + 2n * A` from the
/// sensitivity `A` of the operation, assuming every intermediate carries a ONE-ulp error. A
/// fixed-point factorisation does not.
///
/// - LU: the multiplier `l_ik = a_ik / a_kk` is off by up to 1 ulp, and that error reaches the
///   trailing entry `a_ij` SCALED by `|a_kj|`, so an intermediate of a matrix with entries of
///   magnitude `f` carries `f` ulp, not 1. The resulting error is therefore relative rather than
///   absolute, and one relative ulp covers the whole `lu` suite: over its 408 committed cases
///   (all sizes, all distributions, `solve` / `try_inverse` / `determinant`, near-singular
///   included) the worst excess over `tol` is 0.77 relative ulp, reached by `lu3_determinant` on
///   a `medium` matrix (absolute error 466 349 against `tol` 228 249, on a determinant of
///   390 625 in value: a RELATIVE error of 1.2 * 2^-32). Every other op of every size stays under
///   `tol` alone.
/// - QR: the rounded unit column `q_i` multiplies entries of magnitude `max |a_ij|`, so an
///   intermediate of a `medium` matrix carries that many ulp. One relative ulp covers the whole
///   `qr` suite at all three sizes (measured: the worst case of `qr4_solve` sits at 0.6 relative
///   ulp above `tol`, every other op stays under `tol` alone).
/// - SVD: the singular values stay well inside the oracle's own `tol`; the predicate is used only
///   for the derived `solve` / `pseudo_inverse` comparisons.
pub fn oracle_tol(max_abs_expected: u128, tol: u64) -> u128 {
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

/// Largest `|m_ij|` in RAW units over the 4 entries (the argument of `oracle_tol`).
pub fn max_abs_m2(m: Matrix2<Fixed>) -> u128 {
    let mut e = abs_raw(m.m11);
    e = core::cmp::max(e, abs_raw(m.m21));
    e = core::cmp::max(e, abs_raw(m.m12));
    core::cmp::max(e, abs_raw(m.m22))
}

/// Largest `|m_ij|` in RAW units over the 9 entries (the argument of `oracle_tol`).
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

/// Largest `|m_ij|` in RAW units over the 16 entries (the argument of `oracle_tol`).
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

/// Largest `|m_ij|` in RAW units over the 36 entries (the argument of `oracle_tol`).
pub fn max_abs_m6(m: Matrix6<Fixed>) -> u128 {
    let mut e = max_abs_m3(m6_block11(m));
    e = core::cmp::max(e, max_abs_m3(m6_block21(m)));
    e = core::cmp::max(e, max_abs_m3(m6_block12(m)));
    core::cmp::max(e, max_abs_m3(m6_block22(m)))
}

/// Largest `|v_i|` in RAW units (the argument of `oracle_tol`).
pub fn max_abs_v2(v: Vector2<Fixed>) -> u128 {
    core::cmp::max(abs_raw(v.x), abs_raw(v.y))
}

/// Largest `|v_i|` in RAW units (the argument of `oracle_tol`).
pub fn max_abs_v3(v: Vector3<Fixed>) -> u128 {
    core::cmp::max(abs_raw(v.x), core::cmp::max(abs_raw(v.y), abs_raw(v.z)))
}

/// Largest `|v_i|` in RAW units (the argument of `oracle_tol`).
pub fn max_abs_v4(v: Vector4<Fixed>) -> u128 {
    let e = core::cmp::max(abs_raw(v.x), abs_raw(v.y));
    core::cmp::max(e, core::cmp::max(abs_raw(v.z), abs_raw(v.w)))
}

/// Largest `|v_i|` in RAW units (the argument of `oracle_tol`).
pub fn max_abs_v6(v: Vector6<Fixed>) -> u128 {
    core::cmp::max(max_abs_v3(v6_head(v)), max_abs_v3(v6_tail(v)))
}

// --- test-only field-wise equality of the decompositions (the library defines it under
// `#[cfg(test)]` only, which a dependent package does not see)

/// Test-only field-wise equality (upstream `Cholesky2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Cholesky2PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky2<T>> {
    fn eq(lhs: @Cholesky2<T>, rhs: @Cholesky2<T>) -> bool {
        lhs.l11 == rhs.l11 && lhs.l21 == rhs.l21 && lhs.l22 == rhs.l22
    }
}

/// Test-only field-wise equality (upstream `Cholesky3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Cholesky3PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky3<T>> {
    fn eq(lhs: @Cholesky3<T>, rhs: @Cholesky3<T>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l33 == rhs.l33
    }
}

/// Test-only field-wise equality (upstream `Cholesky4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Cholesky4PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky4<T>> {
    fn eq(lhs: @Cholesky4<T>, rhs: @Cholesky4<T>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l41 == rhs.l41
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l42 == rhs.l42
            && lhs.l33 == rhs.l33
            && lhs.l43 == rhs.l43
            && lhs.l44 == rhs.l44
    }
}

/// Test-only field-wise equality (upstream `Cholesky6` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Cholesky6PartialEq<T, +PartialEq<T>> of PartialEq<Cholesky6<T>> {
    fn eq(lhs: @Cholesky6<T>, rhs: @Cholesky6<T>) -> bool {
        lhs.l11 == rhs.l11
            && lhs.l21 == rhs.l21
            && lhs.l31 == rhs.l31
            && lhs.l41 == rhs.l41
            && lhs.l51 == rhs.l51
            && lhs.l61 == rhs.l61
            && lhs.l22 == rhs.l22
            && lhs.l32 == rhs.l32
            && lhs.l42 == rhs.l42
            && lhs.l52 == rhs.l52
            && lhs.l62 == rhs.l62
            && lhs.l33 == rhs.l33
            && lhs.l43 == rhs.l43
            && lhs.l53 == rhs.l53
            && lhs.l63 == rhs.l63
            && lhs.l44 == rhs.l44
            && lhs.l54 == rhs.l54
            && lhs.l64 == rhs.l64
            && lhs.l55 == rhs.l55
            && lhs.l65 == rhs.l65
            && lhs.l66 == rhs.l66
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
pub impl Perm2PartialEq of PartialEq<Perm2> {
    fn eq(lhs: @Perm2, rhs: @Perm2) -> bool {
        lhs.p1 == rhs.p1
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
pub impl Perm3PartialEq of PartialEq<Perm3> {
    fn eq(lhs: @Perm3, rhs: @Perm3) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
pub impl Perm4PartialEq of PartialEq<Perm4> {
    fn eq(lhs: @Perm4, rhs: @Perm4) -> bool {
        lhs.p1 == rhs.p1 && lhs.p2 == rhs.p2 && lhs.p3 == rhs.p3
    }
}

/// Test-only field-wise equality (upstream `PermutationSequence` has no `PartialEq`): the tests
/// and the benchmarks compare permutations through it.
pub impl Perm6PartialEq of PartialEq<Perm6> {
    fn eq(lhs: @Perm6, rhs: @Perm6) -> bool {
        lhs.p1 == rhs.p1
            && lhs.p2 == rhs.p2
            && lhs.p3 == rhs.p3
            && lhs.p4 == rhs.p4
            && lhs.p5 == rhs.p5
    }
}

/// Test-only field-wise equality (upstream `Lu2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Lu2PartialEq<T, +PartialEq<T>> of PartialEq<Lu2<T>> {
    fn eq(lhs: @Lu2<T>, rhs: @Lu2<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

/// Test-only field-wise equality (upstream `Lu3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Lu3PartialEq<T, +PartialEq<T>> of PartialEq<Lu3<T>> {
    fn eq(lhs: @Lu3<T>, rhs: @Lu3<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

/// Test-only field-wise equality (upstream `Lu4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Lu4PartialEq<T, +PartialEq<T>> of PartialEq<Lu4<T>> {
    fn eq(lhs: @Lu4<T>, rhs: @Lu4<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

/// Test-only field-wise equality (upstream `Lu6` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Lu6PartialEq<T, +PartialEq<T>> of PartialEq<Lu6<T>> {
    fn eq(lhs: @Lu6<T>, rhs: @Lu6<T>) -> bool {
        lhs.lu == rhs.lu && lhs.p == rhs.p
    }
}

/// Test-only field-wise equality (upstream `Qr2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Qr2PartialEq<T, +PartialEq<T>> of PartialEq<Qr2<T>> {
    fn eq(lhs: @Qr2<T>, rhs: @Qr2<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

/// Test-only field-wise equality (upstream `Qr3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Qr3PartialEq<T, +PartialEq<T>> of PartialEq<Qr3<T>> {
    fn eq(lhs: @Qr3<T>, rhs: @Qr3<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

/// Test-only field-wise equality (upstream `Qr4` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Qr4PartialEq<T, +PartialEq<T>> of PartialEq<Qr4<T>> {
    fn eq(lhs: @Qr4<T>, rhs: @Qr4<T>) -> bool {
        lhs.q == rhs.q && lhs.r == rhs.r
    }
}

/// Test-only field-wise equality (upstream `Svd2` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Svd2PartialEq<T, +PartialEq<T>> of PartialEq<Svd2<T>> {
    fn eq(lhs: @Svd2<T>, rhs: @Svd2<T>) -> bool {
        lhs.u == rhs.u && lhs.singular_values == rhs.singular_values && lhs.v_t == rhs.v_t
    }
}

/// Test-only field-wise equality (upstream `Svd3` has no `PartialEq`): the tests and the
/// benchmarks compare factors through it.
pub impl Svd3PartialEq<T, +PartialEq<T>> of PartialEq<Svd3<T>> {
    fn eq(lhs: @Svd3<T>, rhs: @Svd3<T>) -> bool {
        lhs.u == rhs.u && lhs.singular_values == rhs.singular_values && lhs.v_t == rhs.v_t
    }
}
