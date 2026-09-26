use fixed::Fixed;
use glam::{Mat3, Mat4, Vec3, Vec4};
use nalgebra::{
    Isometry2, Isometry3, Quaternion, Similarity2, Similarity3, Translation2, Translation3,
    UnitComplex, UnitQuaternion, Vector2, Vector3,
};
use nalgebra_testing::black_box;
use crate::glam_similarity::*;
use super::{FRAC_1_SQRT_2_RAW, int, near};

fn h() -> Fixed {
    Fixed { raw: FRAC_1_SQRT_2_RAW }
}

/// Rotation by 90 degrees, scaling by 3, translation by (3, 4), column-major.
fn sim3() -> Mat3 {
    Mat3 {
        x_axis: Vec3 { x: int(0), y: int(3), z: int(0) },
        y_axis: Vec3 { x: int(-3), y: int(0), z: int(0) },
        z_axis: Vec3 { x: int(3), y: int(4), z: int(1) },
    }
}

/// Quarter turn about z, scaling by 2, translation by (1, 2, 3), column-major.
fn sim4() -> Mat4 {
    Mat4 {
        x_axis: Vec4 { x: int(0), y: int(2), z: int(0), w: int(0) },
        y_axis: Vec4 { x: int(-2), y: int(0), z: int(0), w: int(0) },
        z_axis: Vec4 { x: int(0), y: int(0), z: int(2), w: int(0) },
        w_axis: Vec4 { x: int(1), y: int(2), z: int(3), w: int(1) },
    }
}

#[test]
fn test_similarity2_into_mat3_folds_the_scaling() {
    let s = black_box(
        Similarity2 {
            isometry: Isometry2 {
                rotation: UnitComplex { re: int(0), im: int(1) },
                translation: Translation2 { vector: Vector2 { x: int(3), y: int(4) } },
            },
            scaling: int(3),
        },
    );
    let m: Mat3 = s.into();
    assert!(m == sim3());
}

#[test]
fn test_mat3_try_into_similarity2() {
    let s: Similarity2<Fixed> = black_box(sim3()).try_into().unwrap();
    assert!(s.scaling == int(3));
    assert!(s.isometry.rotation == UnitComplex { re: int(0), im: int(1) });
    assert!(s.isometry.translation.vector == Vector2 { x: int(3), y: int(4) });
    let back: Mat3 = s.into();
    assert!(back == sim3());
}

#[test]
fn test_mat3_try_into_similarity2_mean_of_the_column_norms() {
    // Columns of norm 2 and 4: the scaling is their mean, 3 (upstream: mean of the norms).
    let mut m = sim3();
    m.x_axis = Vec3 { x: int(2), y: int(0), z: int(0) };
    m.y_axis = Vec3 { x: int(0), y: int(4), z: int(0) };
    let s: Similarity2<Fixed> = black_box(m).try_into().unwrap();
    assert!(s.scaling == int(3));
    assert!(s.isometry.rotation == UnitComplex { re: int(1), im: int(0) });
}

#[test]
fn test_mat3_try_into_similarity2_rejects() {
    // A zero column.
    let mut m = sim3();
    m.y_axis = Vec3 { x: int(0), y: int(0), z: int(0) };
    let s: Option<Similarity2<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    let mut m = sim3();
    m.x_axis = Vec3 { x: int(0), y: int(0), z: int(0) };
    let s: Option<Similarity2<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    // A nonzero bottom row.
    let mut m = sim3();
    m.x_axis.z = int(1);
    let s: Option<Similarity2<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    // The corner is not one.
    let mut m = sim3();
    m.z_axis.z = int(2);
    let s: Option<Similarity2<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    // A reflection (negative determinant): no rotation and scaling represent it in 2D.
    let mut m = sim3();
    m.x_axis = Vec3 { x: int(2), y: int(0), z: int(0) };
    m.y_axis = Vec3 { x: int(0), y: int(-2), z: int(0) };
    let s: Option<Similarity2<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
}

#[test]
fn test_similarity3_into_mat4_folds_the_scaling() {
    let s = black_box(
        Similarity3 {
            isometry: Isometry3 {
                rotation: UnitQuaternion {
                    quaternion: Quaternion { i: int(0), j: int(0), k: h(), w: h() },
                },
                translation: Translation3 { vector: Vector3 { x: int(1), y: int(2), z: int(3) } },
            },
            scaling: int(2),
        },
    );
    let m: Mat4 = s.into();
    let e = sim4();
    assert!(near(m.x_axis.x, e.x_axis.x, 8) && near(m.x_axis.y, e.x_axis.y, 8));
    assert!(near(m.y_axis.x, e.y_axis.x, 8) && near(m.y_axis.y, e.y_axis.y, 8));
    assert!(near(m.z_axis.z, e.z_axis.z, 8));
    assert!(m.w_axis == e.w_axis);
}

#[test]
fn test_mat4_try_into_similarity3() {
    let s: Similarity3<Fixed> = black_box(sim4()).try_into().unwrap();
    assert!(s.scaling == int(2));
    let q = s.isometry.rotation.quaternion;
    assert!(near(q.i, int(0), 4) && near(q.j, int(0), 4));
    assert!(near(q.k, h(), 4) && near(q.w, h(), 4));
    assert!(s.isometry.translation.vector == Vector3 { x: int(1), y: int(2), z: int(3) });
}

#[test]
fn test_mat4_try_into_similarity3_reflection_gives_a_negative_scaling() {
    // diag(-2, 2, 2): negative determinant, so the columns are negated and the scaling is -2:
    // the rotation is diag(1, -1, -1) (a half turn about x), (i, j, k, w) = (±1, 0, 0, 0).
    let mut m = sim4();
    m.x_axis = Vec4 { x: int(-2), y: int(0), z: int(0), w: int(0) };
    m.y_axis = Vec4 { x: int(0), y: int(2), z: int(0), w: int(0) };
    let s: Similarity3<Fixed> = black_box(m).try_into().unwrap();
    assert!(s.scaling == int(-2));
    let q = s.isometry.rotation.quaternion;
    assert!(q.i.raw == int(1).raw || q.i.raw == int(-1).raw);
    assert!(q.j == int(0) && q.k == int(0) && q.w == int(0));
}

#[test]
fn test_mat4_try_into_similarity3_accepts_a_shear() {
    // Upstream comments out its orthogonality test: only nonzero columns are required.
    let mut m = sim4();
    m.x_axis = Vec4 { x: int(1), y: int(0), z: int(0), w: int(0) };
    m.y_axis = Vec4 { x: int(1), y: int(1), z: int(0), w: int(0) };
    m.z_axis = Vec4 { x: int(0), y: int(0), z: int(1), w: int(0) };
    let s: Option<Similarity3<Fixed>> = black_box(m).try_into();
    assert!(s.is_some());
}

#[test]
fn test_mat4_try_into_similarity3_rejects() {
    let mut m = sim4();
    m.z_axis = Vec4 { x: int(0), y: int(0), z: int(0), w: int(0) };
    let s: Option<Similarity3<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    let mut m = sim4();
    m.x_axis = Vec4 { x: int(0), y: int(0), z: int(0), w: int(0) };
    let s: Option<Similarity3<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    let mut m = sim4();
    m.y_axis.w = Fixed { raw: 1 };
    let s: Option<Similarity3<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
    let mut m = sim4();
    m.w_axis.w = int(2);
    let s: Option<Similarity3<Fixed>> = black_box(m).try_into();
    assert!(s.is_none());
}
