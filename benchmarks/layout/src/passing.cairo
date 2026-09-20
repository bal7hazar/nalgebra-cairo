//! Question 3 (part 2) — by value (Copy) vs snapshot `@T` vs `Box<T>` for 3 / 7 / 9 / 16-felt
//! structs. The callee is `#[inline(never)]` so the calling convention is actually exercised, and
//! the scalar is `felt252` so the body is nearly free and the argument passing dominates.

use crate::fixed::Fixed;
use crate::mat3::Mat3F;
use crate::static_mats_gen::Mat4F;
use crate::vec3::Vec3;

/// 7-felt rigid transform (quaternion + translation), felt-only stand-in for the passing test.
#[derive(Copy, Drop)]
pub struct Iso7 {
    pub qx: felt252,
    pub qy: felt252,
    pub qz: felt252,
    pub qw: felt252,
    pub tx: felt252,
    pub ty: felt252,
    pub tz: felt252,
}

#[inline(never)]
pub fn vec3_sum_val(v: Vec3<felt252>) -> felt252 {
    v.x + v.z
}
#[inline(never)]
pub fn vec3_sum_snap(v: @Vec3<felt252>) -> felt252 {
    *v.x + *v.z
}
#[inline(never)]
pub fn vec3_sum_box(v: Box<Vec3<felt252>>) -> felt252 {
    let v = v.unbox();
    v.x + v.z
}

#[inline(never)]
pub fn iso_sum_val(i: Iso7) -> felt252 {
    i.qx + i.tz
}
#[inline(never)]
pub fn iso_sum_snap(i: @Iso7) -> felt252 {
    *i.qx + *i.tz
}
#[inline(never)]
pub fn iso_sum_box(i: Box<Iso7>) -> felt252 {
    let i = i.unbox();
    i.qx + i.tz
}

#[inline(never)]
pub fn mat3_trace_val(m: Mat3F<felt252>) -> felt252 {
    m.m11 + m.m33
}
#[inline(never)]
pub fn mat3_trace_snap(m: @Mat3F<felt252>) -> felt252 {
    *m.m11 + *m.m33
}
#[inline(never)]
pub fn mat3_trace_box(m: Box<Mat3F<felt252>>) -> felt252 {
    let m = m.unbox();
    m.m11 + m.m33
}

#[inline(never)]
pub fn mat4_trace_val(m: Mat4F<felt252>) -> felt252 {
    m.m11 + m.m44
}
#[inline(never)]
pub fn mat4_trace_snap(m: @Mat4F<felt252>) -> felt252 {
    *m.m11 + *m.m44
}
#[inline(never)]
pub fn mat4_trace_box(m: Box<Mat4F<felt252>>) -> felt252 {
    let m = m.unbox();
    m.m11 + m.m44
}

/// Default inlining (no attribute): does the difference survive when the compiler decides?
pub fn mat4_trace_val_default(m: Mat4F<felt252>) -> felt252 {
    m.m11 + m.m44
}
pub fn mat4_trace_snap_default(m: @Mat4F<felt252>) -> felt252 {
    *m.m11 + *m.m44
}

/// Realistic body: Q32.32 mat * vec, by value vs by snapshot, never inlined.
#[inline(never)]
pub fn mul_vec_val(m: Mat3F<Fixed>, v: Vec3<Fixed>) -> Vec3<Fixed> {
    Vec3 {
        x: m.m11 * v.x + m.m12 * v.y + m.m13 * v.z,
        y: m.m21 * v.x + m.m22 * v.y + m.m23 * v.z,
        z: m.m31 * v.x + m.m32 * v.y + m.m33 * v.z,
    }
}
#[inline(never)]
pub fn mul_vec_snap(m: @Mat3F<Fixed>, v: @Vec3<Fixed>) -> Vec3<Fixed> {
    Vec3 {
        x: *m.m11 * *v.x + *m.m12 * *v.y + *m.m13 * *v.z,
        y: *m.m21 * *v.x + *m.m22 * *v.y + *m.m23 * *v.z,
        z: *m.m31 * *v.x + *m.m32 * *v.y + *m.m33 * *v.z,
    }
}

/// Returning a large struct: by value vs boxed.
#[inline(never)]
pub fn mat4_identity_like_val(k: felt252) -> Mat4F<felt252> {
    Mat4F {
        m11: k,
        m12: 0,
        m13: 0,
        m14: 0,
        m21: 0,
        m22: k,
        m23: 0,
        m24: 0,
        m31: 0,
        m32: 0,
        m33: k,
        m34: 0,
        m41: 0,
        m42: 0,
        m43: 0,
        m44: k,
    }
}
#[inline(never)]
pub fn mat4_identity_like_box(k: felt252) -> Box<Mat4F<felt252>> {
    BoxTrait::new(
        Mat4F {
            m11: k,
            m12: 0,
            m13: 0,
            m14: 0,
            m21: 0,
            m22: k,
            m23: 0,
            m24: 0,
            m31: 0,
            m32: 0,
            m33: k,
            m34: 0,
            m41: 0,
            m42: 0,
            m43: 0,
            m44: k,
        },
    )
}

#[cfg(test)]
mod tests {
    use crate::fixed::{Fixed, Scalar, input};
    use crate::mat3::Mat3F;
    use crate::static_mats_gen::Mat4F;
    use crate::vec3::Vec3;
    use super::*;

    fn vec3() -> Vec3<felt252> {
        Vec3 { x: input(1), y: input(2), z: input(3) }
    }

    fn iso() -> Iso7 {
        Iso7 {
            qx: input(1),
            qy: input(2),
            qz: input(3),
            qw: input(4),
            tx: input(5),
            ty: input(6),
            tz: input(7),
        }
    }

    fn mat3() -> Mat3F<felt252> {
        Mat3F {
            m11: input(1),
            m12: input(2),
            m13: input(3),
            m21: input(4),
            m22: input(5),
            m23: input(6),
            m31: input(7),
            m32: input(8),
            m33: input(9),
        }
    }

    fn mat4() -> Mat4F<felt252> {
        Mat4F {
            m11: input(1),
            m12: input(2),
            m13: input(3),
            m14: input(4),
            m21: input(5),
            m22: input(6),
            m23: input(7),
            m24: input(8),
            m31: input(9),
            m32: input(10),
            m33: input(11),
            m34: input(12),
            m41: input(13),
            m42: input(14),
            m43: input(15),
            m44: input(16),
        }
    }

    // ---- Vec3 (3 felts) ----
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__baseline() {
        let v = vec3();
        assert!(v.x + v.x + v.x == 3);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__value_x1() {
        let v = vec3();
        assert!(vec3_sum_val(v) + v.x + v.x == 6);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__value_x3() {
        let v = vec3();
        assert!(vec3_sum_val(v) + vec3_sum_val(v) + vec3_sum_val(v) == 12);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__snapshot_x1() {
        let v = vec3();
        assert!(vec3_sum_snap(@v) + v.x + v.x == 6);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__snapshot_x3() {
        let v = vec3();
        assert!(vec3_sum_snap(@v) + vec3_sum_snap(@v) + vec3_sum_snap(@v) == 12);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__box_x1() {
        let v = vec3();
        assert!(vec3_sum_box(BoxTrait::new(v)) + v.x + v.x == 6);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_vec3__box_x3() {
        let v = vec3();
        let b = BoxTrait::new(v);
        assert!(vec3_sum_box(b) + vec3_sum_box(b) + vec3_sum_box(b) == 12);
    }

    // ---- Isometry-like (7 felts) ----
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__baseline() {
        let i = iso();
        assert!(i.qx + i.qx + i.qx == 3);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__value_x1() {
        let i = iso();
        assert!(iso_sum_val(i) + i.qx + i.qx == 10);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__value_x3() {
        let i = iso();
        assert!(iso_sum_val(i) + iso_sum_val(i) + iso_sum_val(i) == 24);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__snapshot_x1() {
        let i = iso();
        assert!(iso_sum_snap(@i) + i.qx + i.qx == 10);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__snapshot_x3() {
        let i = iso();
        assert!(iso_sum_snap(@i) + iso_sum_snap(@i) + iso_sum_snap(@i) == 24);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__box_x1() {
        let i = iso();
        assert!(iso_sum_box(BoxTrait::new(i)) + i.qx + i.qx == 10);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_iso7__box_x3() {
        let i = iso();
        let b = BoxTrait::new(i);
        assert!(iso_sum_box(b) + iso_sum_box(b) + iso_sum_box(b) == 24);
    }

    // ---- Mat3 (9 felts) ----
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__baseline() {
        let m = mat3();
        assert!(m.m11 + m.m11 + m.m11 == 3);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__value_x1() {
        let m = mat3();
        assert!(mat3_trace_val(m) + m.m11 + m.m11 == 12);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__value_x3() {
        let m = mat3();
        assert!(mat3_trace_val(m) + mat3_trace_val(m) + mat3_trace_val(m) == 30);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__snapshot_x1() {
        let m = mat3();
        assert!(mat3_trace_snap(@m) + m.m11 + m.m11 == 12);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__snapshot_x3() {
        let m = mat3();
        assert!(mat3_trace_snap(@m) + mat3_trace_snap(@m) + mat3_trace_snap(@m) == 30);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__box_x1() {
        let m = mat3();
        assert!(mat3_trace_box(BoxTrait::new(m)) + m.m11 + m.m11 == 12);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat3__box_x3() {
        let m = mat3();
        let b = BoxTrait::new(m);
        assert!(mat3_trace_box(b) + mat3_trace_box(b) + mat3_trace_box(b) == 30);
    }

    // ---- Mat4 (16 felts) ----
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__baseline() {
        let m = mat4();
        assert!(m.m11 + m.m11 + m.m11 == 3);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__value_x1() {
        let m = mat4();
        assert!(mat4_trace_val(m) + m.m11 + m.m11 == 19);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__value_x3() {
        let m = mat4();
        assert!(mat4_trace_val(m) + mat4_trace_val(m) + mat4_trace_val(m) == 51);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__snapshot_x1() {
        let m = mat4();
        assert!(mat4_trace_snap(@m) + m.m11 + m.m11 == 19);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__snapshot_x3() {
        let m = mat4();
        assert!(mat4_trace_snap(@m) + mat4_trace_snap(@m) + mat4_trace_snap(@m) == 51);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__box_x1() {
        let m = mat4();
        assert!(mat4_trace_box(BoxTrait::new(m)) + m.m11 + m.m11 == 19);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__box_x3() {
        let m = mat4();
        let b = BoxTrait::new(m);
        assert!(mat4_trace_box(b) + mat4_trace_box(b) + mat4_trace_box(b) == 51);
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__value_default_inline_x3() {
        let m = mat4();
        assert!(
            mat4_trace_val_default(m) + mat4_trace_val_default(m) + mat4_trace_val_default(m) == 51,
        );
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mat4__snapshot_default_inline_x3() {
        let m = mat4();
        assert!(
            mat4_trace_snap_default(@m)
                + mat4_trace_snap_default(@m)
                + mat4_trace_snap_default(@m) == 51,
        );
    }

    // ---- Returning 16 felts ----
    #[test]
    #[inline(never)]
    fn bench_return_mat4__baseline() {
        let k: felt252 = input(3);
        assert!(k + k == 6);
    }
    #[test]
    #[inline(never)]
    fn bench_return_mat4__value() {
        let m = mat4_identity_like_val(input(3));
        assert!(m.m11 + m.m44 == 6);
    }
    #[test]
    #[inline(never)]
    fn bench_return_mat4__boxed() {
        let m = mat4_identity_like_box(input(3)).unbox();
        assert!(m.m11 + m.m44 == 6);
    }

    // ---- Realistic: Q32.32 mat3 * vec3 ----
    fn mat3_fixed() -> (Mat3F<Fixed>, Vec3<Fixed>) {
        (
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
            },
            Vec3 { x: input(1), y: input(-2), z: input(3) },
        )
    }

    fn lit(n: i64) -> Fixed {
        Scalar::from_int(n)
    }

    #[test]
    #[inline(never)]
    fn bench_pass_mulvec__baseline() {
        let (m, _) = mat3_fixed();
        assert!(m.m11 == lit(1) && m.m12 == lit(2) && m.m13 == lit(3));
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mulvec__value() {
        let (m, v) = mat3_fixed();
        let r = mul_vec_val(m, v);
        assert!(r.x == lit(6) && r.y == lit(10) && r.z == lit(-7));
    }
    #[test]
    #[inline(never)]
    fn bench_pass_mulvec__snapshot() {
        let (m, v) = mat3_fixed();
        let r = mul_vec_snap(@m, @v);
        assert!(r.x == lit(6) && r.y == lit(10) && r.z == lit(-7));
    }
}
