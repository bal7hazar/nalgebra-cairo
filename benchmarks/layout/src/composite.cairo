//! Question 4 — composite patterns typical of a physics engine, Q32.32 only.
//! Every kernel is written as an explicit scalar formula so that the comparison is between
//! algorithms / representations, not between call overheads.

use crate::fixed::{Fixed, FixedTrait};
use crate::mat3::Mat3F;
use crate::vec3::Vec3;

pub type V3 = Vec3<Fixed>;
pub type M3 = Mat3F<Fixed>;

#[derive(Copy, Drop, Debug, PartialEq)]
pub struct Quat {
    pub x: Fixed,
    pub y: Fixed,
    pub z: Fixed,
    pub w: Fixed,
}

/// Rigid transform, rotation stored as a unit quaternion (nalgebra `Isometry3`): 7 scalars.
#[derive(Copy, Drop)]
pub struct IsoQ {
    pub rot: Quat,
    pub trans: V3,
}

/// Rigid transform, rotation stored as a matrix: 12 scalars.
#[derive(Copy, Drop)]
pub struct IsoM {
    pub rot: M3,
    pub trans: V3,
}

/// Symmetric 3x3 (rapier `SdpMatrix3`): 6 scalars.
#[derive(Copy, Drop, Debug, PartialEq)]
pub struct Sym3 {
    pub m11: Fixed,
    pub m12: Fixed,
    pub m13: Fixed,
    pub m22: Fixed,
    pub m23: Fixed,
    pub m33: Fixed,
}

// ---- vector / matrix kernels -----------------------------------------------------------------

#[inline(always)]
pub fn v3(x: Fixed, y: Fixed, z: Fixed) -> V3 {
    Vec3 { x, y, z }
}

#[inline(always)]
pub fn v_add(a: V3, b: V3) -> V3 {
    Vec3 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

pub fn v_cross(a: V3, b: V3) -> V3 {
    Vec3 { x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x }
}

pub fn v_dot(a: V3, b: V3) -> Fixed {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn m_mul_vec(m: M3, v: V3) -> V3 {
    Vec3 {
        x: m.m11 * v.x + m.m12 * v.y + m.m13 * v.z,
        y: m.m21 * v.x + m.m22 * v.y + m.m23 * v.z,
        z: m.m31 * v.x + m.m32 * v.y + m.m33 * v.z,
    }
}

pub fn m_mul_mat(a: M3, b: M3) -> M3 {
    Mat3F {
        m11: a.m11 * b.m11 + a.m12 * b.m21 + a.m13 * b.m31,
        m12: a.m11 * b.m12 + a.m12 * b.m22 + a.m13 * b.m32,
        m13: a.m11 * b.m13 + a.m12 * b.m23 + a.m13 * b.m33,
        m21: a.m21 * b.m11 + a.m22 * b.m21 + a.m23 * b.m31,
        m22: a.m21 * b.m12 + a.m22 * b.m22 + a.m23 * b.m32,
        m23: a.m21 * b.m13 + a.m22 * b.m23 + a.m23 * b.m33,
        m31: a.m31 * b.m11 + a.m32 * b.m21 + a.m33 * b.m31,
        m32: a.m31 * b.m12 + a.m32 * b.m22 + a.m33 * b.m32,
        m33: a.m31 * b.m13 + a.m32 * b.m23 + a.m33 * b.m33,
    }
}

#[inline(always)]
pub fn m_transpose(m: M3) -> M3 {
    Mat3F {
        m11: m.m11,
        m12: m.m21,
        m13: m.m31,
        m21: m.m12,
        m22: m.m22,
        m23: m.m32,
        m31: m.m13,
        m32: m.m23,
        m33: m.m33,
    }
}

/// Cross-product matrix [v]x : [v]x * w == v x w. Construction is free (3 negations).
#[inline(always)]
pub fn skew(v: V3) -> M3 {
    let zero = FixedTrait::from_raw(0);
    Mat3F {
        m11: zero,
        m12: -v.z,
        m13: v.y,
        m21: v.z,
        m22: zero,
        m23: -v.x,
        m31: -v.y,
        m32: v.x,
        m33: zero,
    }
}

/// [v]x * M computed column by column with cross products (18 muls instead of 27).
pub fn skew_mul_mat(v: V3, m: M3) -> M3 {
    let c1 = v_cross(v, v3(m.m11, m.m21, m.m31));
    let c2 = v_cross(v, v3(m.m12, m.m22, m.m32));
    let c3 = v_cross(v, v3(m.m13, m.m23, m.m33));
    Mat3F {
        m11: c1.x,
        m12: c2.x,
        m13: c3.x,
        m21: c1.y,
        m22: c2.y,
        m23: c3.y,
        m31: c1.z,
        m32: c2.z,
        m33: c3.z,
    }
}

/// M * M^T exploiting symmetry: 6 dot products (18 muls) instead of 27 muls.
pub fn m_mul_own_transpose(m: M3) -> Sym3 {
    let (r1, r2, r3) = (v3(m.m11, m.m12, m.m13), v3(m.m21, m.m22, m.m23), v3(m.m31, m.m32, m.m33));
    Sym3 {
        m11: v_dot(r1, r1),
        m12: v_dot(r1, r2),
        m13: v_dot(r1, r3),
        m22: v_dot(r2, r2),
        m23: v_dot(r2, r3),
        m33: v_dot(r3, r3),
    }
}

/// World-space inertia R * diag(d) * R^T, naive: two full products (54 muls).
pub fn inertia_world_naive(r: M3, d: V3) -> M3 {
    let zero = FixedTrait::from_raw(0);
    let diag = Mat3F {
        m11: d.x,
        m12: zero,
        m13: zero,
        m21: zero,
        m22: d.y,
        m23: zero,
        m31: zero,
        m32: zero,
        m33: d.z,
    };
    m_mul_mat(m_mul_mat(r, diag), m_transpose(r))
}

/// World-space inertia, structured: scale columns (9 muls) then 6 symmetric dots (18 muls).
pub fn inertia_world_sym(r: M3, d: V3) -> Sym3 {
    let (s11, s12, s13) = (r.m11 * d.x, r.m12 * d.y, r.m13 * d.z);
    let (s21, s22, s23) = (r.m21 * d.x, r.m22 * d.y, r.m23 * d.z);
    let (s31, s32, s33) = (r.m31 * d.x, r.m32 * d.y, r.m33 * d.z);
    Sym3 {
        m11: s11 * r.m11 + s12 * r.m12 + s13 * r.m13,
        m12: s11 * r.m21 + s12 * r.m22 + s13 * r.m23,
        m13: s11 * r.m31 + s12 * r.m32 + s13 * r.m33,
        m22: s21 * r.m21 + s22 * r.m22 + s23 * r.m23,
        m23: s21 * r.m31 + s22 * r.m32 + s23 * r.m33,
        m33: s31 * r.m31 + s32 * r.m32 + s33 * r.m33,
    }
}

pub fn sym_mul_vec(m: Sym3, v: V3) -> V3 {
    Vec3 {
        x: m.m11 * v.x + m.m12 * v.y + m.m13 * v.z,
        y: m.m12 * v.x + m.m22 * v.y + m.m23 * v.z,
        z: m.m13 * v.x + m.m23 * v.y + m.m33 * v.z,
    }
}

#[inline(always)]
pub fn sym_to_full(m: Sym3) -> M3 {
    Mat3F {
        m11: m.m11,
        m12: m.m12,
        m13: m.m13,
        m21: m.m12,
        m22: m.m22,
        m23: m.m23,
        m31: m.m13,
        m32: m.m23,
        m33: m.m33,
    }
}

/// Symmetric inverse: 6 cofactors (12 muls) + det (3 muls) + 6 divs.
pub fn sym_inverse(m: Sym3) -> Sym3 {
    let c11 = m.m22 * m.m33 - m.m23 * m.m23;
    let c12 = m.m13 * m.m23 - m.m12 * m.m33;
    let c13 = m.m12 * m.m23 - m.m13 * m.m22;
    let det = m.m11 * c11 + m.m12 * c12 + m.m13 * c13;
    Sym3 {
        m11: c11 / det,
        m12: c12 / det,
        m13: c13 / det,
        m22: (m.m11 * m.m33 - m.m13 * m.m13) / det,
        m23: (m.m12 * m.m13 - m.m11 * m.m23) / det,
        m33: (m.m11 * m.m22 - m.m12 * m.m12) / det,
    }
}

/// Full inverse (adjugate): 9 cofactors (18 muls) + det (3 muls) + 9 divs.
pub fn m_inverse(a: M3) -> M3 {
    let c11 = a.m22 * a.m33 - a.m23 * a.m32;
    let c12 = a.m23 * a.m31 - a.m21 * a.m33;
    let c13 = a.m21 * a.m32 - a.m22 * a.m31;
    let det = a.m11 * c11 + a.m12 * c12 + a.m13 * c13;
    Mat3F {
        m11: c11 / det,
        m12: (a.m13 * a.m32 - a.m12 * a.m33) / det,
        m13: (a.m12 * a.m23 - a.m13 * a.m22) / det,
        m21: c12 / det,
        m22: (a.m11 * a.m33 - a.m13 * a.m31) / det,
        m23: (a.m13 * a.m21 - a.m11 * a.m23) / det,
        m31: c13 / det,
        m32: (a.m12 * a.m31 - a.m11 * a.m32) / det,
        m33: (a.m11 * a.m22 - a.m12 * a.m21) / det,
    }
}

// ---- quaternion ------------------------------------------------------------------------------

/// Hamilton product: 16 muls.
pub fn q_mul(a: Quat, b: Quat) -> Quat {
    Quat {
        x: a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y: a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z: a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        w: a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    }
}

#[inline(always)]
pub fn q_conjugate(q: Quat) -> Quat {
    Quat { x: -q.x, y: -q.y, z: -q.z, w: q.w }
}

/// q * (v, 0) * q^-1 with two generic Hamilton products: 32 muls.
pub fn q_rotate_sandwich(q: Quat, v: V3) -> V3 {
    let p = Quat { x: v.x, y: v.y, z: v.z, w: FixedTrait::from_raw(0) };
    let r = q_mul(q_mul(q, p), q_conjugate(q));
    v3(r.x, r.y, r.z)
}

/// Expanded formula v + 2w(u x v) + 2 u x (u x v): 15 muls, doublings done with additions.
pub fn q_rotate_expanded(q: Quat, v: V3) -> V3 {
    let u = v3(q.x, q.y, q.z);
    let c = v_cross(u, v);
    let t = v3(c.x + c.x, c.y + c.y, c.z + c.z);
    let uxt = v_cross(u, t);
    Vec3 { x: v.x + q.w * t.x + uxt.x, y: v.y + q.w * t.y + uxt.y, z: v.z + q.w * t.z + uxt.z }
}

/// Unit quaternion -> rotation matrix: 9 muls.
pub fn q_to_mat3(q: Quat) -> M3 {
    let one = FixedTrait::from_int(1);
    let (x2, y2, z2) = (q.x + q.x, q.y + q.y, q.z + q.z);
    let (xx, yy, zz) = (q.x * x2, q.y * y2, q.z * z2);
    let (xy, xz, yz) = (q.x * y2, q.x * z2, q.y * z2);
    let (wx, wy, wz) = (q.w * x2, q.w * y2, q.w * z2);
    Mat3F {
        m11: one - yy - zz,
        m12: xy - wz,
        m13: xz + wy,
        m21: xy + wz,
        m22: one - xx - zz,
        m23: yz - wx,
        m31: xz - wy,
        m32: yz + wx,
        m33: one - xx - yy,
    }
}

/// Convert to a matrix, then multiply: 9 + 9 muls (the matrix can be reused for more points).
pub fn q_rotate_via_mat3(q: Quat, v: V3) -> V3 {
    m_mul_vec(q_to_mat3(q), v)
}

// ---- isometries ------------------------------------------------------------------------------

pub fn isoq_transform_point(iso: IsoQ, p: V3) -> V3 {
    v_add(q_rotate_expanded(iso.rot, p), iso.trans)
}

pub fn isom_transform_point(iso: IsoM, p: V3) -> V3 {
    v_add(m_mul_vec(iso.rot, p), iso.trans)
}

/// (q1, t1) * (q2, t2) = (q1 q2, t1 + q1 t2): 16 + 15 muls.
pub fn isoq_compose(a: IsoQ, b: IsoQ) -> IsoQ {
    IsoQ { rot: q_mul(a.rot, b.rot), trans: v_add(q_rotate_expanded(a.rot, b.trans), a.trans) }
}

/// (R1, t1) * (R2, t2) = (R1 R2, t1 + R1 t2): 27 + 9 muls.
pub fn isom_compose(a: IsoM, b: IsoM) -> IsoM {
    IsoM { rot: m_mul_mat(a.rot, b.rot), trans: v_add(m_mul_vec(a.rot, b.trans), a.trans) }
}

// ---- normalisation ---------------------------------------------------------------------------

pub fn v_norm(v: V3) -> Fixed {
    v_dot(v, v).sqrt()
}

/// 3 muls + sqrt + 3 divs.
pub fn v_normalize_div(v: V3) -> V3 {
    let n = v_norm(v);
    Vec3 { x: v.x / n, y: v.y / n, z: v.z / n }
}

/// 3 muls + sqrt + 1 div + 3 muls (one extra truncation on the inverse).
pub fn v_normalize_inv(v: V3) -> V3 {
    let inv = FixedTrait::from_int(1) / v_norm(v);
    Vec3 { x: v.x * inv, y: v.y * inv, z: v.z * inv }
}

#[cfg(test)]
mod tests {
    use crate::fixed::{Fixed, FixedTrait, HALF, Scalar, input};
    use crate::mat3::Mat3F;
    use super::*;

    fn lit(n: i64) -> Fixed {
        Scalar::from_int(n)
    }

    fn half() -> Fixed {
        harness::black_box(FixedTrait::from_raw(HALF))
    }

    /// 120 degrees about (1, 1, 1): (x, y, z) -> (z, x, y). Exact in Q32.32.
    fn quat() -> Quat {
        Quat { x: half(), y: half(), z: half(), w: half() }
    }

    fn point() -> V3 {
        v3(input(1), input(-2), input(3))
    }

    fn mat_a() -> M3 {
        Mat3F {
            m11: input(1),
            m12: input(2),
            m13: input(3),
            m21: input(0),
            m22: input(1),
            m23: input(4),
            m31: input(5),
            m32: input(6),
            m33: input(0),
        }
    }

    fn v_is(v: V3, x: i64, y: i64, z: i64) -> bool {
        v.x == lit(x) && v.y == lit(y) && v.z == lit(z)
    }

    // ---- quaternion product ----
    #[test]
    #[inline(never)]
    fn bench_quat_mul__baseline() {
        let (a, _b) = (quat(), quat());
        assert!(a.x == half() && a.y == half() && a.z == half() && a.w == half());
    }

    #[test]
    #[inline(never)]
    fn bench_quat_mul__hamilton() {
        let (a, b) = (quat(), quat());
        let _ = half();
        // q * q = 240 degrees about (1,1,1) = (0.5, 0.5, 0.5, -0.5)
        let r = q_mul(a, b);
        assert!(r.x == a.x && r.y == a.y && r.z == a.z && r.w == -a.w);
    }

    // ---- rotate one vector ----
    #[test]
    #[inline(never)]
    fn bench_quat_rotate__baseline() {
        let (_q, v) = (quat(), point());
        assert!(v_is(v, 1, -2, 3));
    }

    #[test]
    #[inline(never)]
    fn bench_quat_rotate__expanded() {
        let (q, v) = (quat(), point());
        assert!(v_is(q_rotate_expanded(q, v), 3, 1, -2));
    }

    #[test]
    #[inline(never)]
    fn bench_quat_rotate__via_mat3() {
        let (q, v) = (quat(), point());
        assert!(v_is(q_rotate_via_mat3(q, v), 3, 1, -2));
    }

    #[test]
    #[inline(never)]
    fn bench_quat_rotate__sandwich() {
        let (q, v) = (quat(), point());
        assert!(v_is(q_rotate_sandwich(q, v), 3, 1, -2));
    }

    // ---- rotate four vectors with the same rotation ----
    #[test]
    #[inline(never)]
    fn bench_quat_rotate_x4__baseline() {
        let (_q, v) = (quat(), point());
        let (_a, _b, _c) = (point(), point(), point());
        assert!(v_is(v, 1, -2, 3) && v_is(v, 1, -2, 3) && v_is(v, 1, -2, 3) && v_is(v, 1, -2, 3));
    }

    #[test]
    #[inline(never)]
    fn bench_quat_rotate_x4__expanded() {
        let (q, v) = (quat(), point());
        let (a, b, c) = (point(), point(), point());
        assert!(
            v_is(q_rotate_expanded(q, v), 3, 1, -2)
                && v_is(q_rotate_expanded(q, a), 3, 1, -2)
                && v_is(q_rotate_expanded(q, b), 3, 1, -2)
                && v_is(q_rotate_expanded(q, c), 3, 1, -2),
        );
    }

    #[test]
    #[inline(never)]
    fn bench_quat_rotate_x4__via_mat3_once() {
        let (q, v) = (quat(), point());
        let (a, b, c) = (point(), point(), point());
        let m = q_to_mat3(q);
        assert!(
            v_is(m_mul_vec(m, v), 3, 1, -2)
                && v_is(m_mul_vec(m, a), 3, 1, -2)
                && v_is(m_mul_vec(m, b), 3, 1, -2)
                && v_is(m_mul_vec(m, c), 3, 1, -2),
        );
    }

    // ---- quaternion -> matrix ----
    #[test]
    #[inline(never)]
    fn bench_quat_to_mat3__baseline() {
        let q = quat();
        assert!(q.x == half() && q.y == half() && q.z == half());
    }

    #[test]
    #[inline(never)]
    fn bench_quat_to_mat3__convert() {
        let q = quat();
        let _ = half();
        let m = q_to_mat3(q);
        assert!(m.m13 == lit(1) && m.m21 == lit(1) && m.m32 == lit(1));
    }

    // ---- isometry: transform point ----
    fn translation() -> V3 {
        v3(input(10), input(20), input(30))
    }

    #[test]
    #[inline(never)]
    fn bench_iso_transform_point__baseline() {
        let (_q, _t, p) = (quat(), translation(), point());
        assert!(v_is(p, 1, -2, 3));
    }

    #[test]
    #[inline(never)]
    fn bench_iso_transform_point__quat() {
        let (q, t, p) = (quat(), translation(), point());
        let iso = IsoQ { rot: q, trans: t };
        assert!(v_is(isoq_transform_point(iso, p), 13, 21, 28));
    }

    /// Rotation already stored as a matrix (conversion paid elsewhere, e.g. once per step).
    #[test]
    #[inline(never)]
    fn bench_iso_transform_point__mat3() {
        let (q, t, p) = (quat(), translation(), point());
        let zero = q.x - q.y;
        let one = q.x + q.y;
        let rot = Mat3F {
            m11: zero,
            m12: zero,
            m13: one,
            m21: one,
            m22: zero,
            m23: zero,
            m31: zero,
            m32: one,
            m33: zero,
        };
        let iso = IsoM { rot, trans: t };
        assert!(v_is(isom_transform_point(iso, p), 13, 21, 28));
    }

    // ---- isometry: composition ----
    #[test]
    #[inline(never)]
    fn bench_iso_compose__baseline() {
        let (_q, t) = (quat(), translation());
        let (_q2, _t2) = (quat(), translation());
        assert!(v_is(t, 10, 20, 30));
    }

    #[test]
    #[inline(never)]
    fn bench_iso_compose__quat() {
        let a = IsoQ { rot: quat(), trans: translation() };
        let b = IsoQ { rot: quat(), trans: translation() };
        let c = isoq_compose(a, b);
        // t = t1 + R t2 = (10, 20, 30) + (30, 10, 20)
        assert!(v_is(c.trans, 40, 30, 50));
    }

    #[test]
    #[inline(never)]
    fn bench_iso_compose__mat3() {
        let (q, t) = (quat(), translation());
        let (q2, t2) = (quat(), translation());
        let (zero, one) = (q.x - q.y, q.x + q.y);
        let (zero2, one2) = (q2.x - q2.y, q2.x + q2.y);
        let a = IsoM {
            rot: Mat3F {
                m11: zero,
                m12: zero,
                m13: one,
                m21: one,
                m22: zero,
                m23: zero,
                m31: zero,
                m32: one,
                m33: zero,
            },
            trans: t,
        };
        let b = IsoM {
            rot: Mat3F {
                m11: zero2,
                m12: zero2,
                m13: one2,
                m21: one2,
                m22: zero2,
                m23: zero2,
                m31: zero2,
                m32: one2,
                m33: zero2,
            },
            trans: t2,
        };
        let c = isom_compose(a, b);
        assert!(v_is(c.trans, 40, 30, 50));
    }

    // ---- normalize ----
    const THREE_FIFTHS: i64 = 2576980377; // floor(0.6 * 2^32)
    const FOUR_FIFTHS: i64 = 3435973836; // floor(0.8 * 2^32)

    #[test]
    #[inline(never)]
    fn bench_normalize__baseline() {
        let v = v3(input(3), input(0), input(4));
        assert!(v_is(v, 3, 0, 4));
    }

    #[test]
    #[inline(never)]
    fn bench_normalize__norm_only() {
        let v = v3(input(3), input(0), input(4));
        let n = v_norm(v);
        assert!(n == lit(5) && n == lit(5) && n == lit(5));
    }

    #[test]
    #[inline(never)]
    fn bench_normalize__three_divisions() {
        let v = v3(input(3), input(0), input(4));
        let n = v_normalize_div(v);
        assert!(n.x.raw == THREE_FIFTHS && n.y.raw == 0 && n.z.raw == FOUR_FIFTHS);
    }

    #[test]
    #[inline(never)]
    fn bench_normalize__one_division_three_muls() {
        let v = v3(input(3), input(0), input(4));
        let n = v_normalize_inv(v);
        assert!(n.x.raw == THREE_FIFTHS && n.y.raw == 0 && n.z.raw == FOUR_FIFTHS);
    }

    // ---- cross-product matrix ----
    #[test]
    #[inline(never)]
    fn bench_skew_mul_vec__baseline() {
        let (a, _b) = (point(), v3(input(4), input(5), input(-6)));
        assert!(v_is(a, 1, -2, 3));
    }

    #[test]
    #[inline(never)]
    fn bench_skew_mul_vec__cross() {
        let (a, b) = (point(), v3(input(4), input(5), input(-6)));
        assert!(v_is(v_cross(a, b), -3, 18, 13));
    }

    #[test]
    #[inline(never)]
    fn bench_skew_mul_vec__skew_then_mat_vec() {
        let (a, b) = (point(), v3(input(4), input(5), input(-6)));
        assert!(v_is(m_mul_vec(skew(a), b), -3, 18, 13));
    }

    #[test]
    #[inline(never)]
    fn bench_skew_mul_mat__baseline() {
        let (v, _m) = (point(), mat_a());
        assert!(v_is(v, 1, -2, 3));
    }

    #[test]
    #[inline(never)]
    fn bench_skew_mul_mat__skew_then_mat_mat() {
        let (v, m) = (point(), mat_a());
        let r = m_mul_mat(skew(v), m);
        // [v]x * A, first row = (-3*0 + -2*5, -3*1 + -2*6, -3*4 + 0) = (-10, -15, -12)
        assert!(r.m11 == lit(-10) && r.m12 == lit(-15) && r.m13 == lit(-12));
    }

    #[test]
    #[inline(never)]
    fn bench_skew_mul_mat__cross_per_column() {
        let (v, m) = (point(), mat_a());
        let r = skew_mul_mat(v, m);
        assert!(r.m11 == lit(-10) && r.m12 == lit(-15) && r.m13 == lit(-12));
    }

    // ---- M * M^T ----
    #[test]
    #[inline(never)]
    fn bench_m_mt__baseline() {
        let m = mat_a();
        assert!(m.m11 == lit(1) && m.m12 == lit(2) && m.m13 == lit(3));
    }

    #[test]
    #[inline(never)]
    fn bench_m_mt__transpose_then_mat_mat() {
        let m = mat_a();
        let r = m_mul_mat(m, m_transpose(m));
        assert!(r.m11 == lit(14) && r.m12 == lit(14) && r.m33 == lit(61));
    }

    #[test]
    #[inline(never)]
    fn bench_m_mt__symmetric_6_dots() {
        let m = mat_a();
        let r = m_mul_own_transpose(m);
        assert!(r.m11 == lit(14) && r.m12 == lit(14) && r.m33 == lit(61));
    }

    // ---- world inertia R diag(d) R^T ----
    #[test]
    #[inline(never)]
    fn bench_inertia_world__baseline() {
        let (m, _d) = (mat_a(), point());
        assert!(m.m11 == lit(1) && m.m12 == lit(2) && m.m13 == lit(3));
    }

    #[test]
    #[inline(never)]
    fn bench_inertia_world__two_mat_mat() {
        let (m, d) = (mat_a(), point());
        let r = inertia_world_naive(m, d);
        // row1 . (d * row1) = 1 - 8 + 27 = 20 ; row1 . (d * row2) = -4 + 36 = 32 ; row3: 25 - 72
        assert!(r.m11 == lit(20) && r.m12 == lit(32) && r.m33 == lit(-47));
    }

    #[test]
    #[inline(never)]
    fn bench_inertia_world__scaled_symmetric() {
        let (m, d) = (mat_a(), point());
        let r = inertia_world_sym(m, d);
        assert!(r.m11 == lit(20) && r.m12 == lit(32) && r.m33 == lit(-47));
    }

    // ---- symmetric 3x3 ----
    fn sym() -> Sym3 {
        // [2 -1 0; -1 2 -1; 0 -1 1], det = 1, inverse = [1 1 1; 1 2 2; 1 2 3]
        Sym3 {
            m11: input(2),
            m12: input(-1),
            m13: input(0),
            m22: input(2),
            m23: input(-1),
            m33: input(1),
        }
    }

    #[test]
    #[inline(never)]
    fn bench_sym3_mul_vec__baseline() {
        let (_s, v) = (sym(), point());
        assert!(v_is(v, 1, -2, 3));
    }

    #[test]
    #[inline(never)]
    fn bench_sym3_mul_vec__sym6() {
        let (s, v) = (sym(), point());
        assert!(v_is(sym_mul_vec(s, v), 4, -8, 5));
    }

    #[test]
    #[inline(never)]
    fn bench_sym3_mul_vec__full9() {
        let (s, v) = (sym(), point());
        assert!(v_is(m_mul_vec(sym_to_full(s), v), 4, -8, 5));
    }

    #[test]
    #[inline(never)]
    fn bench_sym3_inverse__baseline() {
        let s = sym();
        assert!(s.m11 == lit(2) && s.m22 == lit(2) && s.m33 == lit(1));
    }

    #[test]
    #[inline(never)]
    fn bench_sym3_inverse__sym6() {
        let r = sym_inverse(sym());
        assert!(r.m11 == lit(1) && r.m22 == lit(2) && r.m33 == lit(3));
    }

    #[test]
    #[inline(never)]
    fn bench_sym3_inverse__full9() {
        let r = m_inverse(sym_to_full(sym()));
        assert!(r.m11 == lit(1) && r.m22 == lit(2) && r.m33 == lit(3));
    }
}
