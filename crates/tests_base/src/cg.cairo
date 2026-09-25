//! Oracle tests of the homogeneous / computer-graphics helpers (`nalgebra::base::cg`, WP 8.3-P07)
//! against upstream nalgebra (`oracle_cg`, suite `cg` of `tools/oracle`): the operations whose
//! result is not an exact composition of floored products (rotations, observer / view frames,
//! the homogeneous transforms that divide by the normaliser). The exact operations (scalings,
//! translations, `append_*` / `prepend_*`) are modelled bit for bit by the generated package
//! `crates/shapes_tests_cg`.

use nalgebra::base::point2::Point2Trait;
use nalgebra::base::point3::Point3Trait;
use nalgebra::{Matrix3CgAngleTrait, Matrix3CgTrait, Matrix4CgAngleTrait, Matrix4CgTrait};
use nalgebra_tests_utils::{
    fx, m3, m4, max_ulp_diff3, max_ulp_diff4, max_ulp_diff_v2, max_ulp_diff_v3, p2t, p3t, u3t, v2t,
    v3t,
};
use crate::oracle_cg;

#[test]
fn test_matrix3_new_rotation_oracle() {
    let mut cases = oracle_cg::matrix3_new_rotation_cases();
    while let Some(case) = cases.pop_front() {
        let (angle, expected, tol) = *case;
        let r = Matrix3CgAngleTrait::new_rotation(fx(angle));
        assert!(max_ulp_diff3(r, m3(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_new_rotation_oracle() {
    let mut cases = oracle_cg::matrix4_new_rotation_cases();
    while let Some(case) = cases.pop_front() {
        let (axisangle, expected, tol) = *case;
        let r = Matrix4CgAngleTrait::new_rotation(v3t(axisangle));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_from_scaled_axis_oracle() {
    let mut cases = oracle_cg::matrix4_from_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (axisangle, expected, tol) = *case;
        let r = Matrix4CgAngleTrait::from_scaled_axis(v3t(axisangle));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_from_axis_angle_oracle() {
    let mut cases = oracle_cg::matrix4_from_axis_angle_cases();
    while let Some(case) = cases.pop_front() {
        let (axis, angle, expected, tol) = *case;
        let r = Matrix4CgAngleTrait::from_axis_angle(u3t(axis), fx(angle));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_from_euler_angles_oracle() {
    let mut cases = oracle_cg::matrix4_from_euler_angles_cases();
    while let Some(case) = cases.pop_front() {
        let ((roll, pitch, yaw), expected, tol) = *case;
        let r = Matrix4CgAngleTrait::from_euler_angles(fx(roll), fx(pitch), fx(yaw));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_new_rotation_wrt_point_oracle() {
    let mut cases = oracle_cg::matrix4_new_rotation_wrt_point_cases();
    while let Some(case) = cases.pop_front() {
        let (axisangle, pt, expected, tol) = *case;
        let r = Matrix4CgAngleTrait::new_rotation_wrt_point(v3t(axisangle), p3t(pt));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_face_towards_oracle() {
    let mut cases = oracle_cg::matrix4_face_towards_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, expected, tol) = *case;
        let r = Matrix4CgTrait::face_towards(p3t(eye), p3t(target), v3t(up));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
        assert!(Matrix4CgTrait::new_observer_frame(p3t(eye), p3t(target), v3t(up)) == r);
    }
}

#[test]
fn test_matrix4_look_at_rh_oracle() {
    let mut cases = oracle_cg::matrix4_look_at_rh_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, expected, tol) = *case;
        let r = Matrix4CgTrait::look_at_rh(p3t(eye), p3t(target), v3t(up));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_look_at_lh_oracle() {
    let mut cases = oracle_cg::matrix4_look_at_lh_cases();
    while let Some(case) = cases.pop_front() {
        let (eye, target, up, expected, tol) = *case;
        let r = Matrix4CgTrait::look_at_lh(p3t(eye), p3t(target), v3t(up));
        assert!(max_ulp_diff4(r, m4(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix3_transform_point_oracle() {
    let mut cases = oracle_cg::matrix3_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (m, p, expected, tol) = *case;
        let r = m3(m).transform_point(p2t(p));
        let e = p2t(expected);
        assert!(max_ulp_diff_v2(r.coords(), e.coords()) <= tol.into());
    }
}

#[test]
fn test_matrix4_transform_point_oracle() {
    let mut cases = oracle_cg::matrix4_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (m, p, expected, tol) = *case;
        let r = m4(m).transform_point(p3t(p));
        let e = p3t(expected);
        assert!(max_ulp_diff_v3(r.coords(), e.coords()) <= tol.into());
    }
}

#[test]
fn test_matrix3_transform_vector_oracle() {
    let mut cases = oracle_cg::matrix3_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (m, v, expected, tol) = *case;
        let r = m3(m).transform_vector(v2t(v));
        assert!(max_ulp_diff_v2(r, v2t(expected)) <= tol.into());
    }
}

#[test]
fn test_matrix4_transform_vector_oracle() {
    let mut cases = oracle_cg::matrix4_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (m, v, expected, tol) = *case;
        let r = m4(m).transform_vector(v3t(v));
        assert!(max_ulp_diff_v3(r, v3t(expected)) <= tol.into());
    }
}
