use fixed::Fixed;
use glam::{
    BVec2, BVec3, BVec4, IVec2, IVec3, IVec4, Mat2, Mat3, Mat4, UVec2, UVec3, UVec4, Vec2, Vec3,
    Vec4,
};
use nalgebra::{
    Matrix2, Matrix3, Matrix4, UnitVector2, UnitVector3, UnitVector4, Vector2, Vector3, Vector4,
};
use nalgebra_testing::black_box;
use crate::glam_matrix::*;
use super::{int, near};

#[test]
fn test_vec_into_vector_order() {
    let v2: Vector2<Fixed> = black_box(Vec2 { x: int(1), y: int(2) }).into();
    assert!(v2 == Vector2 { x: int(1), y: int(2) });
    let v3: Vector3<Fixed> = black_box(Vec3 { x: int(1), y: int(2), z: int(3) }).into();
    assert!(v3 == Vector3 { x: int(1), y: int(2), z: int(3) });
    let v4: Vector4<Fixed> = black_box(Vec4 { x: int(1), y: int(2), z: int(3), w: int(4) }).into();
    assert!(v4 == Vector4 { x: int(1), y: int(2), z: int(3), w: int(4) });
}

#[test]
fn test_vector_into_vec_round_trip() {
    let v2 = black_box(Vector2 { x: int(-1), y: int(2) });
    let g2: Vec2 = v2.into();
    assert!(g2 == Vec2 { x: int(-1), y: int(2) });
    let v3 = black_box(Vector3 { x: int(-1), y: int(2), z: int(-3) });
    let g3: Vec3 = v3.into();
    assert!(g3 == Vec3 { x: int(-1), y: int(2), z: int(-3) });
    let back: Vector3<Fixed> = g3.into();
    assert!(back == v3);
    let v4 = black_box(Vector4 { x: int(1), y: int(-2), z: int(3), w: int(-4) });
    let g4: Vec4 = v4.into();
    assert!(g4 == Vec4 { x: int(1), y: int(-2), z: int(3), w: int(-4) });
    let back: Vector4<Fixed> = g4.into();
    assert!(back == v4);
}

#[test]
fn test_ivec_vector_order_and_round_trip() {
    let v2: Vector2<i32> = black_box(IVec2 { x: -1, y: 2 }).into();
    assert!(v2 == Vector2 { x: -1, y: 2 });
    let v3: Vector3<i32> = black_box(IVec3 { x: -1, y: 2, z: -3 }).into();
    assert!(v3 == Vector3 { x: -1, y: 2, z: -3 });
    let v4: Vector4<i32> = black_box(IVec4 { x: -1, y: 2, z: -3, w: 4 }).into();
    assert!(v4 == Vector4 { x: -1, y: 2, z: -3, w: 4 });
    let g2: IVec2 = v2.into();
    let g3: IVec3 = v3.into();
    let g4: IVec4 = v4.into();
    assert!(g2 == IVec2 { x: -1, y: 2 });
    assert!(g3 == IVec3 { x: -1, y: 2, z: -3 });
    assert!(g4 == IVec4 { x: -1, y: 2, z: -3, w: 4 });
}

#[test]
fn test_uvec_vector_order_and_round_trip() {
    let v2: Vector2<u32> = black_box(UVec2 { x: 1, y: 4_000_000_000 }).into();
    assert!(v2 == Vector2 { x: 1, y: 4_000_000_000 });
    let v3: Vector3<u32> = black_box(UVec3 { x: 1, y: 2, z: 3 }).into();
    assert!(v3 == Vector3 { x: 1, y: 2, z: 3 });
    let v4: Vector4<u32> = black_box(UVec4 { x: 1, y: 2, z: 3, w: 4 }).into();
    assert!(v4 == Vector4 { x: 1, y: 2, z: 3, w: 4 });
    let g2: UVec2 = v2.into();
    let g3: UVec3 = v3.into();
    let g4: UVec4 = v4.into();
    assert!(g2 == UVec2 { x: 1, y: 4_000_000_000 });
    assert!(g3 == UVec3 { x: 1, y: 2, z: 3 });
    assert!(g4 == UVec4 { x: 1, y: 2, z: 3, w: 4 });
}

#[test]
fn test_bvec_vector_order_and_round_trip() {
    let v2: Vector2<bool> = black_box(BVec2 { x: true, y: false }).into();
    assert!(v2 == Vector2 { x: true, y: false });
    let v3: Vector3<bool> = black_box(BVec3 { x: false, y: true, z: false }).into();
    assert!(v3 == Vector3 { x: false, y: true, z: false });
    let v4: Vector4<bool> = black_box(BVec4 { x: false, y: false, z: true, w: true }).into();
    assert!(v4 == Vector4 { x: false, y: false, z: true, w: true });
    let g2: BVec2 = v2.into();
    let g3: BVec3 = v3.into();
    let g4: BVec4 = v4.into();
    assert!(g2 == BVec2 { x: true, y: false });
    assert!(g3 == BVec3 { x: false, y: true, z: false });
    assert!(g4 == BVec4 { x: false, y: false, z: true, w: true });
}

#[test]
fn test_unit_vector_try_from_vec_normalizes() {
    // (0, 3, 4) / 5 = (0, 0.6, 0.8): one correctly rounded division per component.
    let u3: UnitVector3<Fixed> = black_box(Vec3 { x: int(0), y: int(3), z: int(4) })
        .try_into()
        .unwrap();
    assert!(u3.value.x == int(0));
    assert!(near(u3.value.y, Fixed { raw: 2576980378 }, 1));
    assert!(near(u3.value.z, Fixed { raw: 3435973837 }, 1));
    let u2: UnitVector2<Fixed> = black_box(Vec2 { x: int(3), y: int(4) }).try_into().unwrap();
    assert!(near(u2.value.x, Fixed { raw: 2576980378 }, 1));
    assert!(near(u2.value.y, Fixed { raw: 3435973837 }, 1));
    let u4: UnitVector4<Fixed> = black_box(Vec4 { x: int(0), y: int(0), z: int(0), w: int(-2) })
        .try_into()
        .unwrap();
    assert!(u4.value == Vector4 { x: int(0), y: int(0), z: int(0), w: int(-1) });
}

#[test]
fn test_unit_vector_try_from_zero_is_none() {
    let z2: Option<UnitVector2<Fixed>> = black_box(Vec2 { x: int(0), y: int(0) }).try_into();
    assert!(z2.is_none());
    let z3: Option<UnitVector3<Fixed>> = black_box(Vec3 { x: int(0), y: int(0), z: int(0) })
        .try_into();
    assert!(z3.is_none());
    let z4: Option<UnitVector4<Fixed>> = black_box(
        Vec4 { x: int(0), y: int(0), z: int(0), w: int(0) },
    )
        .try_into();
    assert!(z4.is_none());
}

#[test]
fn test_unit_vector_into_vec_keeps_components() {
    let u2 = black_box(UnitVector2 { value: Vector2 { x: int(1), y: int(0) } });
    let g2: Vec2 = u2.into();
    assert!(g2 == Vec2 { x: int(1), y: int(0) });
    let u3 = black_box(UnitVector3 { value: Vector3 { x: int(0), y: int(1), z: int(0) } });
    let g3: Vec3 = u3.into();
    assert!(g3 == Vec3 { x: int(0), y: int(1), z: int(0) });
    let u4 = black_box(
        UnitVector4 { value: Vector4 { x: int(0), y: int(0), z: int(0), w: int(1) } },
    );
    let g4: Vec4 = u4.into();
    assert!(g4 == Vec4 { x: int(0), y: int(0), z: int(0), w: int(1) });
}

#[test]
fn test_mat2_is_column_major() {
    // Mat2 axes are COLUMNS: x_axis = (1, 2) is the first column (m11, m21).
    let m: Matrix2<Fixed> = black_box(
        Mat2 { x_axis: Vec2 { x: int(1), y: int(2) }, y_axis: Vec2 { x: int(3), y: int(4) } },
    )
        .into();
    assert!(m == Matrix2 { m11: int(1), m21: int(2), m12: int(3), m22: int(4) });
    let g: Mat2 = m.into();
    assert!(
        g == Mat2 { x_axis: Vec2 { x: int(1), y: int(2) }, y_axis: Vec2 { x: int(3), y: int(4) } },
    );
}

#[test]
fn test_mat3_is_column_major() {
    let g = black_box(
        Mat3 {
            x_axis: Vec3 { x: int(11), y: int(21), z: int(31) },
            y_axis: Vec3 { x: int(12), y: int(22), z: int(32) },
            z_axis: Vec3 { x: int(13), y: int(23), z: int(33) },
        },
    );
    let m: Matrix3<Fixed> = g.into();
    assert!(
        m == Matrix3 {
            m11: int(11),
            m21: int(21),
            m31: int(31),
            m12: int(12),
            m22: int(22),
            m32: int(32),
            m13: int(13),
            m23: int(23),
            m33: int(33),
        },
    );
    let back: Mat3 = m.into();
    assert!(back == g);
}

#[test]
fn test_mat4_is_column_major() {
    let g = black_box(
        Mat4 {
            x_axis: Vec4 { x: int(11), y: int(21), z: int(31), w: int(41) },
            y_axis: Vec4 { x: int(12), y: int(22), z: int(32), w: int(42) },
            z_axis: Vec4 { x: int(13), y: int(23), z: int(33), w: int(43) },
            w_axis: Vec4 { x: int(14), y: int(24), z: int(34), w: int(44) },
        },
    );
    let m: Matrix4<Fixed> = g.into();
    assert!(m.m11 == int(11) && m.m21 == int(21) && m.m31 == int(31) && m.m41 == int(41));
    assert!(m.m12 == int(12) && m.m22 == int(22) && m.m32 == int(32) && m.m42 == int(42));
    assert!(m.m13 == int(13) && m.m23 == int(23) && m.m33 == int(33) && m.m43 == int(43));
    assert!(m.m14 == int(14) && m.m24 == int(24) && m.m34 == int(34) && m.m44 == int(44));
    let back: Mat4 = m.into();
    assert!(back == g);
}
