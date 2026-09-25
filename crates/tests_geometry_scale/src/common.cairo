//! Builders of the shapes, points and scales of the tests of WP 8.4-P10 from raw Q32.32
//! values, in the layout of the oracle (`tools/oracle`): vectors and points as tuples, matrices as
//! ROW-major arrays.

use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::matrix2x3::Matrix2x3;
use nalgebra::base::matrix2x4::Matrix2x4;
use nalgebra::base::matrix2x5::Matrix2x5;
use nalgebra::base::matrix2x6::Matrix2x6;
use nalgebra::base::matrix3::Matrix3;
use nalgebra::base::matrix3x2::Matrix3x2;
use nalgebra::base::matrix3x4::Matrix3x4;
use nalgebra::base::matrix3x5::Matrix3x5;
use nalgebra::base::matrix3x6::Matrix3x6;
use nalgebra::base::matrix4::Matrix4;
use nalgebra::base::matrix4x2::Matrix4x2;
use nalgebra::base::matrix4x3::Matrix4x3;
use nalgebra::base::matrix4x5::Matrix4x5;
use nalgebra::base::matrix4x6::Matrix4x6;
use nalgebra::base::matrix5::Matrix5;
use nalgebra::base::matrix5x2::Matrix5x2;
use nalgebra::base::matrix5x3::Matrix5x3;
use nalgebra::base::matrix5x4::Matrix5x4;
use nalgebra::base::matrix5x6::Matrix5x6;
use nalgebra::base::matrix6::Matrix6;
use nalgebra::base::matrix6x2::Matrix6x2;
use nalgebra::base::matrix6x3::Matrix6x3;
use nalgebra::base::matrix6x4::Matrix6x4;
use nalgebra::base::matrix6x5::Matrix6x5;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::row_vector2::RowVector2;
use nalgebra::base::row_vector3::RowVector3;
use nalgebra::base::row_vector4::RowVector4;
use nalgebra::base::row_vector5::RowVector5;
use nalgebra::base::row_vector6::RowVector6;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::base::vector4::Vector4;
use nalgebra::base::vector5::Vector5;
use nalgebra::base::vector6::Vector6;
use nalgebra::geometry::point1::Point1;
use nalgebra::geometry::point4::Point4;
use nalgebra::geometry::point5::Point5;
use nalgebra::geometry::point6::Point6;
use nalgebra::geometry::scale1::Scale1;
use nalgebra::geometry::scale2::Scale2;
use nalgebra::geometry::scale3::Scale3;
use nalgebra::geometry::scale4::Scale4;
use nalgebra::geometry::scale5::Scale5;
use nalgebra::geometry::scale6::Scale6;
use nalgebra_tests_utils::{fx, ulp_diff};

/// `Matrix1` from a tuple of raw components.
pub fn vec1(t: (i64,)) -> Matrix1<Fixed> {
    let (x,) = t;
    Matrix1 { x: fx(x) }
}

/// `Point1` from a tuple of raw coordinates.
pub fn pt1(t: (i64,)) -> Point1<Fixed> {
    let (x,) = t;
    Point1 { x: fx(x) }
}

/// `Scale1` from a tuple of raw factors.
pub fn sc1(t: (i64,)) -> Scale1<Fixed> {
    Scale1 { vector: vec1(t) }
}

/// Largest `|a - b|` over the coordinates, in raw units.
pub fn err_pt1(a: Point1<Fixed>, b: Point1<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e
}

/// `Vector2` from a tuple of raw components.
pub fn vec2(t: (i64, i64)) -> Vector2<Fixed> {
    let (x, y) = t;
    Vector2 { x: fx(x), y: fx(y) }
}

/// `Point2` from a tuple of raw coordinates.
pub fn pt2(t: (i64, i64)) -> Point2<Fixed> {
    let (x, y) = t;
    Point2 { x: fx(x), y: fx(y) }
}

/// `Scale2` from a tuple of raw factors.
pub fn sc2(t: (i64, i64)) -> Scale2<Fixed> {
    Scale2 { vector: vec2(t) }
}

/// Largest `|a - b|` over the coordinates, in raw units.
pub fn err_pt2(a: Point2<Fixed>, b: Point2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e
}

/// `Vector3` from a tuple of raw components.
pub fn vec3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// `Point3` from a tuple of raw coordinates.
pub fn pt3(t: (i64, i64, i64)) -> Point3<Fixed> {
    let (x, y, z) = t;
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// `Scale3` from a tuple of raw factors.
pub fn sc3(t: (i64, i64, i64)) -> Scale3<Fixed> {
    Scale3 { vector: vec3(t) }
}

/// Largest `|a - b|` over the coordinates, in raw units.
pub fn err_pt3(a: Point3<Fixed>, b: Point3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e
}

/// `Vector4` from a tuple of raw components.
pub fn vec4(t: (i64, i64, i64, i64)) -> Vector4<Fixed> {
    let (x, y, z, w) = t;
    Vector4 { x: fx(x), y: fx(y), z: fx(z), w: fx(w) }
}

/// `Point4` from a tuple of raw coordinates.
pub fn pt4(t: (i64, i64, i64, i64)) -> Point4<Fixed> {
    let (x, y, z, w) = t;
    Point4 { x: fx(x), y: fx(y), z: fx(z), w: fx(w) }
}

/// `Scale4` from a tuple of raw factors.
pub fn sc4(t: (i64, i64, i64, i64)) -> Scale4<Fixed> {
    Scale4 { vector: vec4(t) }
}

/// Largest `|a - b|` over the coordinates, in raw units.
pub fn err_pt4(a: Point4<Fixed>, b: Point4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e
}

/// `Vector5` from a tuple of raw components.
pub fn vec5(t: (i64, i64, i64, i64, i64)) -> Vector5<Fixed> {
    let (x, y, z, w, a) = t;
    Vector5 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a) }
}

/// `Point5` from a tuple of raw coordinates.
pub fn pt5(t: (i64, i64, i64, i64, i64)) -> Point5<Fixed> {
    let (x, y, z, w, a) = t;
    Point5 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a) }
}

/// `Scale5` from a tuple of raw factors.
pub fn sc5(t: (i64, i64, i64, i64, i64)) -> Scale5<Fixed> {
    Scale5 { vector: vec5(t) }
}

/// Largest `|a - b|` over the coordinates, in raw units.
pub fn err_pt5(a: Point5<Fixed>, b: Point5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e = core::cmp::max(e, ulp_diff(a.a, b.a));
    e
}

/// `Vector6` from a tuple of raw components.
pub fn vec6(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Vector6 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a), b: fx(b) }
}

/// `Point6` from a tuple of raw coordinates.
pub fn pt6(t: (i64, i64, i64, i64, i64, i64)) -> Point6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Point6 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a), b: fx(b) }
}

/// `Scale6` from a tuple of raw factors.
pub fn sc6(t: (i64, i64, i64, i64, i64, i64)) -> Scale6<Fixed> {
    Scale6 { vector: vec6(t) }
}

/// Largest `|a - b|` over the coordinates, in raw units.
pub fn err_pt6(a: Point6<Fixed>, b: Point6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e = core::cmp::max(e, ulp_diff(a.a, b.a));
    e = core::cmp::max(e, ulp_diff(a.b, b.b));
    e
}

/// `Matrix1` from ROW-major rows.
pub fn mat1x1(rows: [[i64; 1]; 1]) -> Matrix1<Fixed> {
    let [[r11]] = rows;
    Matrix1 { x: fx(r11) }
}

/// `RowVector2` from ROW-major rows.
pub fn mat1x2(rows: [[i64; 2]; 1]) -> RowVector2<Fixed> {
    let [[r11, r12]] = rows;
    RowVector2 { x: fx(r11), y: fx(r12) }
}

/// `RowVector3` from ROW-major rows.
pub fn mat1x3(rows: [[i64; 3]; 1]) -> RowVector3<Fixed> {
    let [[r11, r12, r13]] = rows;
    RowVector3 { x: fx(r11), y: fx(r12), z: fx(r13) }
}

/// `RowVector4` from ROW-major rows.
pub fn mat1x4(rows: [[i64; 4]; 1]) -> RowVector4<Fixed> {
    let [[r11, r12, r13, r14]] = rows;
    RowVector4 { x: fx(r11), y: fx(r12), z: fx(r13), w: fx(r14) }
}

/// `RowVector5` from ROW-major rows.
pub fn mat1x5(rows: [[i64; 5]; 1]) -> RowVector5<Fixed> {
    let [[r11, r12, r13, r14, r15]] = rows;
    RowVector5 { x: fx(r11), y: fx(r12), z: fx(r13), w: fx(r14), a: fx(r15) }
}

/// `RowVector6` from ROW-major rows.
pub fn mat1x6(rows: [[i64; 6]; 1]) -> RowVector6<Fixed> {
    let [[r11, r12, r13, r14, r15, r16]] = rows;
    RowVector6 { x: fx(r11), y: fx(r12), z: fx(r13), w: fx(r14), a: fx(r15), b: fx(r16) }
}

/// `Vector2` from ROW-major rows.
pub fn mat2x1(rows: [[i64; 1]; 2]) -> Vector2<Fixed> {
    let [[r11], [r21]] = rows;
    Vector2 { x: fx(r11), y: fx(r21) }
}

/// `Matrix2` from ROW-major rows.
pub fn mat2x2(rows: [[i64; 2]; 2]) -> Matrix2<Fixed> {
    let [[r11, r12], [r21, r22]] = rows;
    Matrix2 { m11: fx(r11), m21: fx(r21), m12: fx(r12), m22: fx(r22) }
}

/// `Matrix2x3` from ROW-major rows.
pub fn mat2x3(rows: [[i64; 3]; 2]) -> Matrix2x3<Fixed> {
    let [[r11, r12, r13], [r21, r22, r23]] = rows;
    Matrix2x3 { m11: fx(r11), m21: fx(r21), m12: fx(r12), m22: fx(r22), m13: fx(r13), m23: fx(r23) }
}

/// `Matrix2x4` from ROW-major rows.
pub fn mat2x4(rows: [[i64; 4]; 2]) -> Matrix2x4<Fixed> {
    let [[r11, r12, r13, r14], [r21, r22, r23, r24]] = rows;
    Matrix2x4 {
        m11: fx(r11),
        m21: fx(r21),
        m12: fx(r12),
        m22: fx(r22),
        m13: fx(r13),
        m23: fx(r23),
        m14: fx(r14),
        m24: fx(r24),
    }
}

/// `Matrix2x5` from ROW-major rows.
pub fn mat2x5(rows: [[i64; 5]; 2]) -> Matrix2x5<Fixed> {
    let [[r11, r12, r13, r14, r15], [r21, r22, r23, r24, r25]] = rows;
    Matrix2x5 {
        m11: fx(r11),
        m21: fx(r21),
        m12: fx(r12),
        m22: fx(r22),
        m13: fx(r13),
        m23: fx(r23),
        m14: fx(r14),
        m24: fx(r24),
        m15: fx(r15),
        m25: fx(r25),
    }
}

/// `Matrix2x6` from ROW-major rows.
pub fn mat2x6(rows: [[i64; 6]; 2]) -> Matrix2x6<Fixed> {
    let [[r11, r12, r13, r14, r15, r16], [r21, r22, r23, r24, r25, r26]] = rows;
    Matrix2x6 {
        m11: fx(r11),
        m21: fx(r21),
        m12: fx(r12),
        m22: fx(r22),
        m13: fx(r13),
        m23: fx(r23),
        m14: fx(r14),
        m24: fx(r24),
        m15: fx(r15),
        m25: fx(r25),
        m16: fx(r16),
        m26: fx(r26),
    }
}

/// `Vector3` from ROW-major rows.
pub fn mat3x1(rows: [[i64; 1]; 3]) -> Vector3<Fixed> {
    let [[r11], [r21], [r31]] = rows;
    Vector3 { x: fx(r11), y: fx(r21), z: fx(r31) }
}

/// `Matrix3x2` from ROW-major rows.
pub fn mat3x2(rows: [[i64; 2]; 3]) -> Matrix3x2<Fixed> {
    let [[r11, r12], [r21, r22], [r31, r32]] = rows;
    Matrix3x2 { m11: fx(r11), m21: fx(r21), m31: fx(r31), m12: fx(r12), m22: fx(r22), m32: fx(r32) }
}

/// `Matrix3` from ROW-major rows.
pub fn mat3x3(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
    let [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]] = rows;
    Matrix3 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
    }
}

/// `Matrix3x4` from ROW-major rows.
pub fn mat3x4(rows: [[i64; 4]; 3]) -> Matrix3x4<Fixed> {
    let [[r11, r12, r13, r14], [r21, r22, r23, r24], [r31, r32, r33, r34]] = rows;
    Matrix3x4 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
    }
}

/// `Matrix3x5` from ROW-major rows.
pub fn mat3x5(rows: [[i64; 5]; 3]) -> Matrix3x5<Fixed> {
    let [[r11, r12, r13, r14, r15], [r21, r22, r23, r24, r25], [r31, r32, r33, r34, r35]] = rows;
    Matrix3x5 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
    }
}

/// `Matrix3x6` from ROW-major rows.
pub fn mat3x6(rows: [[i64; 6]; 3]) -> Matrix3x6<Fixed> {
    let [
        [r11, r12, r13, r14, r15, r16],
        [r21, r22, r23, r24, r25, r26],
        [r31, r32, r33, r34, r35, r36],
    ] =
        rows;
    Matrix3x6 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m16: fx(r16),
        m26: fx(r26),
        m36: fx(r36),
    }
}

/// `Vector4` from ROW-major rows.
pub fn mat4x1(rows: [[i64; 1]; 4]) -> Vector4<Fixed> {
    let [[r11], [r21], [r31], [r41]] = rows;
    Vector4 { x: fx(r11), y: fx(r21), z: fx(r31), w: fx(r41) }
}

/// `Matrix4x2` from ROW-major rows.
pub fn mat4x2(rows: [[i64; 2]; 4]) -> Matrix4x2<Fixed> {
    let [[r11, r12], [r21, r22], [r31, r32], [r41, r42]] = rows;
    Matrix4x2 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
    }
}

/// `Matrix4x3` from ROW-major rows.
pub fn mat4x3(rows: [[i64; 3]; 4]) -> Matrix4x3<Fixed> {
    let [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33], [r41, r42, r43]] = rows;
    Matrix4x3 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
    }
}

/// `Matrix4` from ROW-major rows.
pub fn mat4x4(rows: [[i64; 4]; 4]) -> Matrix4<Fixed> {
    let [[r11, r12, r13, r14], [r21, r22, r23, r24], [r31, r32, r33, r34], [r41, r42, r43, r44]] =
        rows;
    Matrix4 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
    }
}

/// `Matrix4x5` from ROW-major rows.
pub fn mat4x5(rows: [[i64; 5]; 4]) -> Matrix4x5<Fixed> {
    let [
        [r11, r12, r13, r14, r15],
        [r21, r22, r23, r24, r25],
        [r31, r32, r33, r34, r35],
        [r41, r42, r43, r44, r45],
    ] =
        rows;
    Matrix4x5 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m45: fx(r45),
    }
}

/// `Matrix4x6` from ROW-major rows.
pub fn mat4x6(rows: [[i64; 6]; 4]) -> Matrix4x6<Fixed> {
    let [
        [r11, r12, r13, r14, r15, r16],
        [r21, r22, r23, r24, r25, r26],
        [r31, r32, r33, r34, r35, r36],
        [r41, r42, r43, r44, r45, r46],
    ] =
        rows;
    Matrix4x6 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m45: fx(r45),
        m16: fx(r16),
        m26: fx(r26),
        m36: fx(r36),
        m46: fx(r46),
    }
}

/// `Vector5` from ROW-major rows.
pub fn mat5x1(rows: [[i64; 1]; 5]) -> Vector5<Fixed> {
    let [[r11], [r21], [r31], [r41], [r51]] = rows;
    Vector5 { x: fx(r11), y: fx(r21), z: fx(r31), w: fx(r41), a: fx(r51) }
}

/// `Matrix5x2` from ROW-major rows.
pub fn mat5x2(rows: [[i64; 2]; 5]) -> Matrix5x2<Fixed> {
    let [[r11, r12], [r21, r22], [r31, r32], [r41, r42], [r51, r52]] = rows;
    Matrix5x2 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
    }
}

/// `Matrix5x3` from ROW-major rows.
pub fn mat5x3(rows: [[i64; 3]; 5]) -> Matrix5x3<Fixed> {
    let [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33], [r41, r42, r43], [r51, r52, r53]] =
        rows;
    Matrix5x3 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
    }
}

/// `Matrix5x4` from ROW-major rows.
pub fn mat5x4(rows: [[i64; 4]; 5]) -> Matrix5x4<Fixed> {
    let [
        [r11, r12, r13, r14],
        [r21, r22, r23, r24],
        [r31, r32, r33, r34],
        [r41, r42, r43, r44],
        [r51, r52, r53, r54],
    ] =
        rows;
    Matrix5x4 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m54: fx(r54),
    }
}

/// `Matrix5` from ROW-major rows.
pub fn mat5x5(rows: [[i64; 5]; 5]) -> Matrix5<Fixed> {
    let [
        [r11, r12, r13, r14, r15],
        [r21, r22, r23, r24, r25],
        [r31, r32, r33, r34, r35],
        [r41, r42, r43, r44, r45],
        [r51, r52, r53, r54, r55],
    ] =
        rows;
    Matrix5 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m54: fx(r54),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m45: fx(r45),
        m55: fx(r55),
    }
}

/// `Matrix5x6` from ROW-major rows.
pub fn mat5x6(rows: [[i64; 6]; 5]) -> Matrix5x6<Fixed> {
    let [
        [r11, r12, r13, r14, r15, r16],
        [r21, r22, r23, r24, r25, r26],
        [r31, r32, r33, r34, r35, r36],
        [r41, r42, r43, r44, r45, r46],
        [r51, r52, r53, r54, r55, r56],
    ] =
        rows;
    Matrix5x6 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m54: fx(r54),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m45: fx(r45),
        m55: fx(r55),
        m16: fx(r16),
        m26: fx(r26),
        m36: fx(r36),
        m46: fx(r46),
        m56: fx(r56),
    }
}

/// `Vector6` from ROW-major rows.
pub fn mat6x1(rows: [[i64; 1]; 6]) -> Vector6<Fixed> {
    let [[r11], [r21], [r31], [r41], [r51], [r61]] = rows;
    Vector6 { x: fx(r11), y: fx(r21), z: fx(r31), w: fx(r41), a: fx(r51), b: fx(r61) }
}

/// `Matrix6x2` from ROW-major rows.
pub fn mat6x2(rows: [[i64; 2]; 6]) -> Matrix6x2<Fixed> {
    let [[r11, r12], [r21, r22], [r31, r32], [r41, r42], [r51, r52], [r61, r62]] = rows;
    Matrix6x2 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m61: fx(r61),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m62: fx(r62),
    }
}

/// `Matrix6x3` from ROW-major rows.
pub fn mat6x3(rows: [[i64; 3]; 6]) -> Matrix6x3<Fixed> {
    let [
        [r11, r12, r13],
        [r21, r22, r23],
        [r31, r32, r33],
        [r41, r42, r43],
        [r51, r52, r53],
        [r61, r62, r63],
    ] =
        rows;
    Matrix6x3 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m61: fx(r61),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m62: fx(r62),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m63: fx(r63),
    }
}

/// `Matrix6x4` from ROW-major rows.
pub fn mat6x4(rows: [[i64; 4]; 6]) -> Matrix6x4<Fixed> {
    let [
        [r11, r12, r13, r14],
        [r21, r22, r23, r24],
        [r31, r32, r33, r34],
        [r41, r42, r43, r44],
        [r51, r52, r53, r54],
        [r61, r62, r63, r64],
    ] =
        rows;
    Matrix6x4 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m61: fx(r61),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m62: fx(r62),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m63: fx(r63),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m54: fx(r54),
        m64: fx(r64),
    }
}

/// `Matrix6x5` from ROW-major rows.
pub fn mat6x5(rows: [[i64; 5]; 6]) -> Matrix6x5<Fixed> {
    let [
        [r11, r12, r13, r14, r15],
        [r21, r22, r23, r24, r25],
        [r31, r32, r33, r34, r35],
        [r41, r42, r43, r44, r45],
        [r51, r52, r53, r54, r55],
        [r61, r62, r63, r64, r65],
    ] =
        rows;
    Matrix6x5 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m61: fx(r61),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m62: fx(r62),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m63: fx(r63),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m54: fx(r54),
        m64: fx(r64),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m45: fx(r45),
        m55: fx(r55),
        m65: fx(r65),
    }
}

/// `Matrix6` from ROW-major rows.
pub fn mat6x6(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [
        [r11, r12, r13, r14, r15, r16],
        [r21, r22, r23, r24, r25, r26],
        [r31, r32, r33, r34, r35, r36],
        [r41, r42, r43, r44, r45, r46],
        [r51, r52, r53, r54, r55, r56],
        [r61, r62, r63, r64, r65, r66],
    ] =
        rows;
    Matrix6 {
        m11: fx(r11),
        m21: fx(r21),
        m31: fx(r31),
        m41: fx(r41),
        m51: fx(r51),
        m61: fx(r61),
        m12: fx(r12),
        m22: fx(r22),
        m32: fx(r32),
        m42: fx(r42),
        m52: fx(r52),
        m62: fx(r62),
        m13: fx(r13),
        m23: fx(r23),
        m33: fx(r33),
        m43: fx(r43),
        m53: fx(r53),
        m63: fx(r63),
        m14: fx(r14),
        m24: fx(r24),
        m34: fx(r34),
        m44: fx(r44),
        m54: fx(r54),
        m64: fx(r64),
        m15: fx(r15),
        m25: fx(r25),
        m35: fx(r35),
        m45: fx(r45),
        m55: fx(r55),
        m65: fx(r65),
        m16: fx(r16),
        m26: fx(r26),
        m36: fx(r36),
        m46: fx(r46),
        m56: fx(r56),
        m66: fx(r66),
    }
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat1x1(a: Matrix1<Fixed>, b: Matrix1<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat1x2(a: RowVector2<Fixed>, b: RowVector2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat1x3(a: RowVector3<Fixed>, b: RowVector3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat1x4(a: RowVector4<Fixed>, b: RowVector4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat1x5(a: RowVector5<Fixed>, b: RowVector5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e = core::cmp::max(e, ulp_diff(a.a, b.a));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat1x6(a: RowVector6<Fixed>, b: RowVector6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e = core::cmp::max(e, ulp_diff(a.a, b.a));
    e = core::cmp::max(e, ulp_diff(a.b, b.b));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat2x1(a: Vector2<Fixed>, b: Vector2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat2x2(a: Matrix2<Fixed>, b: Matrix2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat2x3(a: Matrix2x3<Fixed>, b: Matrix2x3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat2x4(a: Matrix2x4<Fixed>, b: Matrix2x4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat2x5(a: Matrix2x5<Fixed>, b: Matrix2x5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat2x6(a: Matrix2x6<Fixed>, b: Matrix2x6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m16, b.m16));
    e = core::cmp::max(e, ulp_diff(a.m26, b.m26));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat3x1(a: Vector3<Fixed>, b: Vector3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat3x2(a: Matrix3x2<Fixed>, b: Matrix3x2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat3x3(a: Matrix3<Fixed>, b: Matrix3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat3x4(a: Matrix3x4<Fixed>, b: Matrix3x4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat3x5(a: Matrix3x5<Fixed>, b: Matrix3x5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat3x6(a: Matrix3x6<Fixed>, b: Matrix3x6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m16, b.m16));
    e = core::cmp::max(e, ulp_diff(a.m26, b.m26));
    e = core::cmp::max(e, ulp_diff(a.m36, b.m36));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat4x1(a: Vector4<Fixed>, b: Vector4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat4x2(a: Matrix4x2<Fixed>, b: Matrix4x2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat4x3(a: Matrix4x3<Fixed>, b: Matrix4x3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
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
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat4x4(a: Matrix4<Fixed>, b: Matrix4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
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
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat4x5(a: Matrix4x5<Fixed>, b: Matrix4x5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
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
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m45, b.m45));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat4x6(a: Matrix4x6<Fixed>, b: Matrix4x6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
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
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m45, b.m45));
    e = core::cmp::max(e, ulp_diff(a.m16, b.m16));
    e = core::cmp::max(e, ulp_diff(a.m26, b.m26));
    e = core::cmp::max(e, ulp_diff(a.m36, b.m36));
    e = core::cmp::max(e, ulp_diff(a.m46, b.m46));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat5x1(a: Vector5<Fixed>, b: Vector5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e = core::cmp::max(e, ulp_diff(a.a, b.a));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat5x2(a: Matrix5x2<Fixed>, b: Matrix5x2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat5x3(a: Matrix5x3<Fixed>, b: Matrix5x3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat5x4(a: Matrix5x4<Fixed>, b: Matrix5x4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m54, b.m54));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat5x5(a: Matrix5<Fixed>, b: Matrix5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m54, b.m54));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m45, b.m45));
    e = core::cmp::max(e, ulp_diff(a.m55, b.m55));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat5x6(a: Matrix5x6<Fixed>, b: Matrix5x6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m54, b.m54));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m45, b.m45));
    e = core::cmp::max(e, ulp_diff(a.m55, b.m55));
    e = core::cmp::max(e, ulp_diff(a.m16, b.m16));
    e = core::cmp::max(e, ulp_diff(a.m26, b.m26));
    e = core::cmp::max(e, ulp_diff(a.m36, b.m36));
    e = core::cmp::max(e, ulp_diff(a.m46, b.m46));
    e = core::cmp::max(e, ulp_diff(a.m56, b.m56));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat6x1(a: Vector6<Fixed>, b: Vector6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.x, b.x));
    e = core::cmp::max(e, ulp_diff(a.y, b.y));
    e = core::cmp::max(e, ulp_diff(a.z, b.z));
    e = core::cmp::max(e, ulp_diff(a.w, b.w));
    e = core::cmp::max(e, ulp_diff(a.a, b.a));
    e = core::cmp::max(e, ulp_diff(a.b, b.b));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat6x2(a: Matrix6x2<Fixed>, b: Matrix6x2<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m61, b.m61));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m62, b.m62));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat6x3(a: Matrix6x3<Fixed>, b: Matrix6x3<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m61, b.m61));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m62, b.m62));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m63, b.m63));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat6x4(a: Matrix6x4<Fixed>, b: Matrix6x4<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m61, b.m61));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m62, b.m62));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m63, b.m63));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m54, b.m54));
    e = core::cmp::max(e, ulp_diff(a.m64, b.m64));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat6x5(a: Matrix6x5<Fixed>, b: Matrix6x5<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m61, b.m61));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m62, b.m62));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m63, b.m63));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m54, b.m54));
    e = core::cmp::max(e, ulp_diff(a.m64, b.m64));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m45, b.m45));
    e = core::cmp::max(e, ulp_diff(a.m55, b.m55));
    e = core::cmp::max(e, ulp_diff(a.m65, b.m65));
    e
}

/// Largest `|a - b|` over the entries, in raw units.
pub fn err_mat6x6(a: Matrix6<Fixed>, b: Matrix6<Fixed>) -> u128 {
    let mut e = 0;
    e = core::cmp::max(e, ulp_diff(a.m11, b.m11));
    e = core::cmp::max(e, ulp_diff(a.m21, b.m21));
    e = core::cmp::max(e, ulp_diff(a.m31, b.m31));
    e = core::cmp::max(e, ulp_diff(a.m41, b.m41));
    e = core::cmp::max(e, ulp_diff(a.m51, b.m51));
    e = core::cmp::max(e, ulp_diff(a.m61, b.m61));
    e = core::cmp::max(e, ulp_diff(a.m12, b.m12));
    e = core::cmp::max(e, ulp_diff(a.m22, b.m22));
    e = core::cmp::max(e, ulp_diff(a.m32, b.m32));
    e = core::cmp::max(e, ulp_diff(a.m42, b.m42));
    e = core::cmp::max(e, ulp_diff(a.m52, b.m52));
    e = core::cmp::max(e, ulp_diff(a.m62, b.m62));
    e = core::cmp::max(e, ulp_diff(a.m13, b.m13));
    e = core::cmp::max(e, ulp_diff(a.m23, b.m23));
    e = core::cmp::max(e, ulp_diff(a.m33, b.m33));
    e = core::cmp::max(e, ulp_diff(a.m43, b.m43));
    e = core::cmp::max(e, ulp_diff(a.m53, b.m53));
    e = core::cmp::max(e, ulp_diff(a.m63, b.m63));
    e = core::cmp::max(e, ulp_diff(a.m14, b.m14));
    e = core::cmp::max(e, ulp_diff(a.m24, b.m24));
    e = core::cmp::max(e, ulp_diff(a.m34, b.m34));
    e = core::cmp::max(e, ulp_diff(a.m44, b.m44));
    e = core::cmp::max(e, ulp_diff(a.m54, b.m54));
    e = core::cmp::max(e, ulp_diff(a.m64, b.m64));
    e = core::cmp::max(e, ulp_diff(a.m15, b.m15));
    e = core::cmp::max(e, ulp_diff(a.m25, b.m25));
    e = core::cmp::max(e, ulp_diff(a.m35, b.m35));
    e = core::cmp::max(e, ulp_diff(a.m45, b.m45));
    e = core::cmp::max(e, ulp_diff(a.m55, b.m55));
    e = core::cmp::max(e, ulp_diff(a.m65, b.m65));
    e = core::cmp::max(e, ulp_diff(a.m16, b.m16));
    e = core::cmp::max(e, ulp_diff(a.m26, b.m26));
    e = core::cmp::max(e, ulp_diff(a.m36, b.m36));
    e = core::cmp::max(e, ulp_diff(a.m46, b.m46));
    e = core::cmp::max(e, ulp_diff(a.m56, b.m56));
    e = core::cmp::max(e, ulp_diff(a.m66, b.m66));
    e
}
