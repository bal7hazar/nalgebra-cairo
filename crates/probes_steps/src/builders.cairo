//! Builders of the probe inputs from raw Q32.32 values (`value = raw / 2^32`). The builders run in
//! the baseline as in the operation, so they never show in a net figure.

use fixed::Fixed;
use nalgebra_geometry3::geometry::isometry3::Isometry3;
use nalgebra_geometry3::geometry::quaternion::Quaternion;
use nalgebra_geometry3::geometry::unit_quaternion::UnitQuaternion;
use nalgebra_types3::base::matrix3::Matrix3;
use nalgebra_types3::base::point3::Point3;
use nalgebra_types3::base::vector3::Vector3;
use nalgebra_types3::geometry::translation3::Translation3;
use nalgebra_types6::base::matrix6::Matrix6;
use nalgebra_types6::base::vector6::Vector6;

pub fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

pub fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

pub fn p3(t: (i64, i64, i64)) -> Point3<Fixed> {
    let (x, y, z) = t;
    Point3 { x: fx(x), y: fx(y), z: fx(z) }
}

/// A `Matrix3` from ROW-major rows.
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

pub fn v6(t: (i64, i64, i64, i64, i64, i64)) -> Vector6<Fixed> {
    let (x, y, z, w, a, b) = t;
    Vector6 { x: fx(x), y: fx(y), z: fx(z), w: fx(w), a: fx(a), b: fx(b) }
}

/// A `Matrix6` from ROW-major rows.
pub fn m6(rows: [[i64; 6]; 6]) -> Matrix6<Fixed> {
    let [r1, r2, r3, r4, r5, r6] = rows;
    let [m11, m12, m13, m14, m15, m16] = r1;
    let [m21, m22, m23, m24, m25, m26] = r2;
    let [m31, m32, m33, m34, m35, m36] = r3;
    let [m41, m42, m43, m44, m45, m46] = r4;
    let [m51, m52, m53, m54, m55, m56] = r5;
    let [m61, m62, m63, m64, m65, m66] = r6;
    Matrix6 {
        m11: fx(m11),
        m21: fx(m21),
        m31: fx(m31),
        m41: fx(m41),
        m51: fx(m51),
        m61: fx(m61),
        m12: fx(m12),
        m22: fx(m22),
        m32: fx(m32),
        m42: fx(m42),
        m52: fx(m52),
        m62: fx(m62),
        m13: fx(m13),
        m23: fx(m23),
        m33: fx(m33),
        m43: fx(m43),
        m53: fx(m53),
        m63: fx(m63),
        m14: fx(m14),
        m24: fx(m24),
        m34: fx(m34),
        m44: fx(m44),
        m54: fx(m54),
        m64: fx(m64),
        m15: fx(m15),
        m25: fx(m25),
        m35: fx(m35),
        m45: fx(m45),
        m55: fx(m55),
        m65: fx(m65),
        m16: fx(m16),
        m26: fx(m26),
        m36: fx(m36),
        m46: fx(m46),
        m56: fx(m56),
        m66: fx(m66),
    }
}

/// A `UnitQuaternion` from raw `(w, i, j, k)`, WITHOUT normalisation.
pub fn uq(t: (i64, i64, i64, i64)) -> UnitQuaternion<Fixed> {
    let (w, i, j, k) = t;
    UnitQuaternion { quaternion: Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) } }
}

/// An `Isometry3` from the raw translation and the raw rotation `(w, i, j, k)`.
pub fn iso3(t: (i64, i64, i64), r: (i64, i64, i64, i64)) -> Isometry3<Fixed> {
    Isometry3 { rotation: uq(r), translation: Translation3 { vector: v3(t) } }
}
