//! Unit tests of `Rotation3`: exact rotations (identity, half turns, the axis permutation), the
//! rotation identities (`R · Rᵀ = I`, `det R = 1`, `R → q → R`), the degenerate cases
//! (`Option` /
//! `#[should_panic]`), and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same
//! raw inputs, tolerance in ulp).
//!
//! Oracle rotation matrices are orthonormalised in f64 then quantised component-wise, so they are
//! orthonormal only within about 2 ulp — exactly what this library produces.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 8 cases per distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo rotation3 --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/rotation3/oracle.cairo
//! ```

use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::matrix_test_utils::{ONE_RAW, fx, int, m3, r3, r3i, u3t, uqt, v3i, v3t};
use crate::base::point3::Point3;
use crate::base::unit::Unit3Trait;
use crate::base::vector3::Vector3Trait;
use crate::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use super::{Rotation3, Rotation3AngleTrait, Rotation3Trait, oracle};

const HALF_RAW: i64 = 0x80000000;

/// The half turn about `x`: `diag(1, -1, -1)`, exact.
fn half_x() -> Rotation3<Fixed> {
    r3i([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
}

/// The 120° rotation about `(1, 1, 1)`: the cyclic permutation `(x, y, z) -> (z, x, y)`, exact.
fn third() -> Rotation3<Fixed> {
    r3i([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
}

// --- constructors and parts

#[test]
fn test_identity_is_exact() {
    let id = Rotation3Trait::<Fixed>::identity();
    assert!(id.matrix == Matrix3Trait::identity());
    assert!(id.matrix() == Matrix3Trait::identity());
    assert!(Rotation3Trait::from_matrix_unchecked(Matrix3Trait::<Fixed>::identity()) == id);
    assert!(id.transform_vector(v3i(1, -2, 3)) == v3i(1, -2, 3));
}

#[test]
fn test_inverse_is_the_transpose() {
    assert!(third().inverse() == r3i([[0, 1, 0], [0, 0, 1], [1, 0, 0]]));
    assert!(third().inverse() == third().transpose());
    assert!(third().inverse().inverse() == third());
    assert!(half_x().inverse() == half_x());
    // R · Rᵀ = I exactly for these.
    assert!(third() * third().inverse() == Rotation3Trait::identity());
}

#[test]
fn test_inverse_oracle_is_exact() {
    let mut cases = oracle::rotation3_inverse_cases();
    assert!(cases.len() == 8);
    while let Some(case) = cases.pop_front() {
        let (rr, expected, tol) = *case;
        assert!(tol == 0);
        assert!(r3(rr).inverse() == r3(expected));
    }
}

#[test]
fn test_to_homogeneous() {
    let h = third().to_homogeneous();
    assert!(h.m13 == Real::ONE && h.m21 == Real::ONE && h.m32 == Real::ONE);
    assert!(h.m11 == Real::ZERO && h.m44 == Real::ONE);
    assert!(h.m41 == Real::ZERO && h.m42 == Real::ZERO && h.m43 == Real::ZERO);
    assert!(h.m14 == Real::ZERO && h.m24 == Real::ZERO && h.m34 == Real::ZERO);
}

// --- composition

#[test]
fn test_mul_exact_and_oracle() {
    let id = Rotation3Trait::<Fixed>::identity();
    assert!(id * third() == third());
    assert!(third() * id == third());
    // Three cyclic permutations are the identity.
    assert!(third() * third() * third() == id);
    let mut cases = oracle::rotation3_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, expected, tol) = *case;
        // One rescale per entry reproduces the exact floor bit for bit.
        assert!(tol == 0);
        assert!((r3(ra) * r3(rb)).matrix == m3(expected));
    }
}

#[test]
fn test_mul_stays_orthonormal() {
    let mut cases = oracle::rotation3_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, _, _) = *case;
        let m = (r3(ra) * r3(rb)).matrix;
        assert!((m * m.transpose()).is_identity(8));
    }
}

// --- transforms

#[test]
fn test_transform_vector_exact_rotations() {
    assert!(half_x().transform_vector(v3i(1, 2, 3)) == v3i(1, -2, -3));
    assert!(third().transform_vector(v3i(1, 2, 3)) == v3i(3, 1, 2));
    assert!(third().inverse_transform_vector(v3i(3, 1, 2)) == v3i(1, 2, 3));
    assert!(
        third()
            .transform_point(
                Point3 { x: int(1), y: int(2), z: int(3) },
            ) == Point3 { x: int(3), y: int(1), z: int(2) },
    );
    assert!(
        third()
            .inverse_transform_point(
                Point3 { x: int(3), y: int(1), z: int(2) },
            ) == Point3 { x: int(1), y: int(2), z: int(3) },
    );
}

#[test]
fn test_transform_vector_oracle_is_bit_exact() {
    let mut cases = oracle::rotation3_transform_vector_cases();
    assert!(cases.len() == 32);
    while let Some(case) = cases.pop_front() {
        let (rr, rv, expected, tol) = *case;
        // The expectation is the exact floor of the matrix-vector product.
        assert!(tol == 0);
        assert!(r3(rr).transform_vector(v3t(rv)) == v3t(expected));
    }
}

#[test]
fn test_inverse_transform_vector_round_trip() {
    let mut cases = oracle::rotation3_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (rr, rv, _, _) = *case;
        let r = r3(rr);
        let v = v3t(rv);
        // The inverse transform undoes it to within the rounding of both products, which is
        // relative: the oracle matrices are orthonormal to about 2 ulp only, so the error grows
        // with |v| (up to 4e4 in the `large` distribution).
        let scale: u64 = v.amax().raw.try_into().unwrap();
        let tol = 8 + 16 * (scale / 0x100000000 + 1);
        assert!(r.inverse_transform_vector(r.transform_vector(v)).abs_diff_eq(v, tol));
        assert!(r.transform_vector(v).norm().abs_diff_eq(v.norm(), tol));
    }
}

// --- conversions with `UnitQuaternion`

#[test]
fn test_unit_quaternion_conversions_exact() {
    let q = uqt((HALF_RAW, HALF_RAW, HALF_RAW, HALF_RAW));
    assert!(Rotation3Trait::from_unit_quaternion(q) == third());
    assert!(third().to_unit_quaternion() == q);
    // `Into` both ways.
    let r: Rotation3<Fixed> = q.into();
    assert!(r == third());
    let back: UnitQuaternion<Fixed> = third().into();
    assert!(back == q);
    assert!(
        Rotation3Trait::<Fixed>::identity().to_unit_quaternion() == UnitQuaternionTrait::identity(),
    );
}

/// `R → q → R` within 4 ulp per entry (measured).
#[test]
fn test_quaternion_round_trip() {
    let mut cases = oracle::rotation3_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, _, _, _) = *case;
        let r = r3(ra);
        let back = Rotation3Trait::from_unit_quaternion(r.to_unit_quaternion());
        assert!(back.abs_diff_eq(r, 4));
    }
}

// --- axis-angle

#[test]
fn test_from_axis_angle_exact_cases() {
    let z = Unit3Trait::<Fixed>::z_axis();
    // A zero angle is the identity exactly (the branch of upstream).
    assert!(
        Rotation3AngleTrait::from_axis_angle(z, Real::<Fixed>::ZERO) == Rotation3Trait::identity(),
    );
    // A quarter turn about z maps x to y.
    let r = Rotation3AngleTrait::from_axis_angle(z, Real::<Fixed>::FRAC_PI_2);
    assert!(r.transform_vector(v3i(1, 0, 0)).abs_diff_eq(v3i(0, 1, 0), 8));
    assert!(r.transform_vector(v3i(0, 1, 0)).abs_diff_eq(v3i(-1, 0, 0), 8));
    assert!(r.transform_vector(v3i(0, 0, 1)).abs_diff_eq(v3i(0, 0, 1), 8));
    // A half turn about x is diag(1, -1, -1) within a few ulp.
    let h = Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::x_axis(), Real::<Fixed>::PI);
    assert!(h.abs_diff_eq(half_x(), 8));
}

#[test]
fn test_from_axis_angle_oracle() {
    let mut cases = oracle::rotation3_from_axis_angle_cases();
    while let Some(case) = cases.pop_front() {
        let (raxis, rangle, expected, tol) = *case;
        let got = Rotation3AngleTrait::from_axis_angle(u3t(raxis), fx(rangle));
        assert!(got.matrix.abs_diff_eq(m3(expected), tol));
        assert!((got.matrix * got.matrix.transpose()).is_identity(16));
    }
}

#[test]
fn test_from_scaled_axis_oracle_and_zero() {
    // A zero rotation vector is the identity (no division by zero).
    assert!(
        Rotation3AngleTrait::from_scaled_axis(
            Vector3Trait::<Fixed>::zeros(),
        ) == Rotation3Trait::identity(),
    );
    let mut cases = oracle::rotation3_from_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (rv, expected, tol) = *case;
        assert!(
            Rotation3AngleTrait::from_scaled_axis(v3t(rv)).matrix.abs_diff_eq(m3(expected), tol),
        );
    }
}

#[test]
fn test_angle_oracle_and_exact_cases() {
    assert!(Rotation3Trait::<Fixed>::identity().angle() == Real::ZERO);
    assert!(half_x().angle().abs_diff_eq(Real::PI, 16));
    let mut cases = oracle::rotation3_angle_cases();
    while let Some(case) = cases.pop_front() {
        let (rr, expected, tol) = *case;
        assert!(Real::abs_diff_eq(r3(rr).angle(), fx(expected), tol));
    }
}

/// The trace of a rounded rotation matrix can exceed 3: the argument of `acos` is clamped instead
/// of panicking with `Fixed: acos domain`.
#[test]
fn test_angle_clamps_the_trace() {
    let over = r3([[ONE_RAW + 4, 0, 0], [0, ONE_RAW + 4, 0], [0, 0, ONE_RAW + 4]]);
    assert!(over.angle() == Real::ZERO);
    let under = r3([[-ONE_RAW - 4, 0, 0], [0, -ONE_RAW - 4, 0], [0, 0, ONE_RAW + 4]]);
    assert!(under.angle().abs_diff_eq(Real::PI, 4));
}

#[test]
fn test_axis_and_scaled_axis() {
    // The identity and a half turn have no determined axis (the antisymmetric part vanishes).
    assert!(Rotation3Trait::<Fixed>::identity().axis() == None);
    assert!(half_x().axis() == None);
    assert!(Rotation3Trait::<Fixed>::identity().scaled_axis() == Vector3Trait::zeros());
    assert!(half_x().scaled_axis() == Vector3Trait::zeros());
    // The 120° rotation about (1, 1, 1): the axis is exact here (the antisymmetric part is (1, 1,
    // 1)).
    let axis = third().axis().unwrap();
    let inv_sqrt3 = 2479700524;
    assert!(axis.value.abs_diff_eq(v3t((inv_sqrt3, inv_sqrt3, inv_sqrt3)), 2));
    // axis · angle = 1.2091995 per component (5 193 472 632 raw); `acos` near the middle of its
    // range is accurate to a few ulp, and the axis is exact here.
    let c = 5193472632;
    assert!(third().scaled_axis().abs_diff_eq(v3t((c, c, c)), 64));
}

#[test]
fn test_scaled_axis_oracle() {
    let mut cases = oracle::rotation3_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (rr, expected, tol) = *case;
        assert!(r3(rr).scaled_axis().abs_diff_eq(v3t(expected), tol));
    }
}

/// `from_scaled_axis` and `scaled_axis` are inverse to each other.
#[test]
fn test_scaled_axis_round_trip() {
    let mut cases = oracle::rotation3_from_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (rv, _, _) = *case;
        let back = Rotation3AngleTrait::from_scaled_axis(v3t(rv)).scaled_axis();
        assert!(back.abs_diff_eq(v3t(rv), 2048));
    }
}

// --- Euler angles

#[test]
fn test_from_euler_angles_oracle() {
    let mut cases = oracle::rotation3_from_euler_angles_cases();
    while let Some(case) = cases.pop_front() {
        let (reuler, expected, tol) = *case;
        let (roll, pitch, yaw) = reuler;
        let got = Rotation3AngleTrait::from_euler_angles(fx(roll), fx(pitch), fx(yaw));
        assert!(got.matrix.abs_diff_eq(m3(expected), tol));
        assert!((got.matrix * got.matrix.transpose()).is_identity(16));
    }
}

#[test]
fn test_euler_angles_oracle() {
    let mut cases = oracle::rotation3_euler_angles_cases();
    while let Some(case) = cases.pop_front() {
        let (rr, expected, tol) = *case;
        let (roll, pitch, yaw) = r3(rr).euler_angles();
        let (eroll, epitch, eyaw) = expected;
        assert!(Real::abs_diff_eq(roll, fx(eroll), tol));
        assert!(Real::abs_diff_eq(pitch, fx(epitch), tol));
        assert!(Real::abs_diff_eq(yaw, fx(eyaw), tol));
    }
}

/// The convention is `Rz(yaw) · Ry(pitch) · Rx(roll)`, the same as `UnitQuaternion`'s.
#[test]
fn test_euler_angles_convention() {
    let (r, p, y) = (fx(0x20000000), fx(0x40000000), fx(0x60000000));
    let composed = Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::z_axis(), y)
        * Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), p)
        * Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::x_axis(), r);
    let direct = Rotation3AngleTrait::from_euler_angles(r, p, y);
    assert!(direct.abs_diff_eq(composed, 16));
    // Same rotation as the quaternion constructor.
    let q = UnitQuaternionAngleTrait::from_euler_angles(r, p, y);
    assert!(direct.abs_diff_eq(q.to_rotation_matrix(), 32));
    // And the angles come back.
    let (broll, bpitch, byaw) = direct.euler_angles();
    assert!(broll.abs_diff_eq(r, 64) && bpitch.abs_diff_eq(p, 64) && byaw.abs_diff_eq(y, 64));
}

#[test]
fn test_euler_angles_gimbal_lock() {
    // pitch = +pi/2 (m31 = -1): yaw is set to 0 and roll carries the rotation.
    let r = Rotation3AngleTrait::from_euler_angles(
        Real::ZERO, Real::<Fixed>::FRAC_PI_2, Real::ZERO,
    );
    let (roll, pitch, yaw) = r.euler_angles();
    assert!(pitch.abs_diff_eq(Real::FRAC_PI_2, 64));
    assert!(yaw == Real::ZERO || roll.abs_diff_eq(Real::ZERO, 64));
    let back = Rotation3AngleTrait::from_euler_angles(roll, pitch, yaw);
    assert!(back.abs_diff_eq(r, 1024));
}

// --- rotation between vectors, look-at

#[test]
fn test_rotation_between_exact_cases() {
    let x = v3i(1, 0, 0);
    let y = v3i(0, 1, 0);
    let r = Rotation3Trait::rotation_between(x, y).unwrap();
    assert!(r.transform_vector(x).abs_diff_eq(y, 8));
    assert!(Rotation3Trait::rotation_between(x, v3i(3, 0, 0)) == Some(Rotation3Trait::identity()));
    assert!(Rotation3Trait::rotation_between(x, v3i(-2, 0, 0)) == None);
    assert!(
        Rotation3Trait::rotation_between(
            Vector3Trait::zeros(), y,
        ) == Some(Rotation3Trait::identity()),
    );
}

#[test]
fn test_rotation_between_oracle() {
    let mut cases = oracle::rotation3_rotation_between_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (ra, rb, expected, tol) = *case;
        let got = Rotation3Trait::rotation_between(v3t(ra), v3t(rb)).unwrap();
        assert!(got.matrix.abs_diff_eq(m3(expected), tol));
        // It maps the direction of a onto the direction of b.
        assert!(
            got.transform_vector(v3t(ra).normalize()).abs_diff_eq(v3t(rb).normalize(), tol * 4 + 8),
        );
    }
}

#[test]
fn test_scaled_rotation_between() {
    let x = v3i(1, 0, 0);
    let y = v3i(0, 1, 0);
    let full = Rotation3AngleTrait::scaled_rotation_between(x, y, Real::ONE).unwrap();
    assert!(full.abs_diff_eq(Rotation3Trait::rotation_between(x, y).unwrap(), 16));
    let half = Rotation3AngleTrait::scaled_rotation_between(x, y, Real::HALF).unwrap();
    let eighth = Rotation3AngleTrait::from_axis_angle(
        Unit3Trait::<Fixed>::z_axis(), Real::<Fixed>::FRAC_PI_4,
    );
    assert!(half.abs_diff_eq(eighth, 16));
    assert!(Rotation3AngleTrait::scaled_rotation_between(x, v3i(-1, 0, 0), Real::ONE) == None);
}

#[test]
fn test_face_towards_and_look_at_rh_oracle() {
    let mut cases = oracle::rotation3_face_towards_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (rdir, rup, expected, tol) = *case;
        let got = Rotation3Trait::face_towards(v3t(rdir), v3t(rup));
        assert!(got.matrix.abs_diff_eq(m3(expected), tol));
    }
    let mut cases = oracle::rotation3_look_at_rh_cases();
    while let Some(case) = cases.pop_front() {
        let (rdir, rup, expected, tol) = *case;
        let got = Rotation3Trait::look_at_rh(v3t(rdir), v3t(rup));
        assert!(got.matrix.abs_diff_eq(m3(expected), tol));
    }
}

#[test]
fn test_face_towards_geometry() {
    // dir becomes the third column (the local z axis), up leans towards the second.
    let dir = v3i(0, 0, 2);
    let up = v3i(0, 1, 0);
    let r = Rotation3Trait::face_towards(dir, up);
    assert!(r.matrix.column3().abs_diff_eq(v3i(0, 0, 1), 2));
    assert!(r.matrix.column2().abs_diff_eq(v3i(0, 1, 0), 2));
    assert!((r.matrix * r.matrix.transpose()).is_identity(8));
    // look_at_rh maps dir onto -z.
    let l = Rotation3Trait::look_at_rh(dir, up);
    assert!(l.transform_vector(v3i(0, 0, 1)).abs_diff_eq(v3i(0, 0, -1), 4));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_face_towards_with_a_parallel_up_panics() {
    let _ = Rotation3Trait::face_towards(black_box(v3i(0, 0, 1)), black_box(v3i(0, 0, 2)));
}

// --- renormalization

#[test]
fn test_renormalize_keeps_a_rotation_matrix() {
    assert!(Rotation3Trait::<Fixed>::identity().renormalize() == Rotation3Trait::identity());
    assert!(third().renormalize() == third());
    assert!(half_x().renormalize() == half_x());
}

/// 64 compositions of a quarter turn: the matrix drifts, `renormalize` brings it back to
/// orthonormal.
#[test]
fn test_renormalize_fixes_a_drifted_matrix() {
    let step = Rotation3AngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), fx(0x123456789));
    let mut r = Rotation3Trait::<Fixed>::identity();
    for _ in 0_u32..64 {
        r = r * step;
    }
    let fixed = r.renormalize();
    assert!((fixed.matrix * fixed.matrix.transpose()).is_identity(4));
    assert!(fixed.matrix.determinant().abs_diff_eq(Real::ONE, 4));
    // The correction is small: the drift of 64 products is a few ulp.
    assert!(fixed.abs_diff_eq(r, 64));
}

/// A matrix scaled by 1 + 1e-3 is brought back to orthonormal in one pass (Gram-Schmidt is not a
/// Newton step: it renormalizes exactly, whatever the scale).
#[test]
fn test_renormalize_handles_a_scaled_matrix() {
    let s = fx(ONE_RAW + 4294967);
    let scaled = Rotation3 { matrix: third().matrix.scale(s) };
    let fixed = scaled.renormalize();
    assert!((fixed.matrix * fixed.matrix.transpose()).is_identity(4));
    assert!(fixed.abs_diff_eq(third(), 4));
}

#[test]
#[should_panic(expected: 'Fixed: division by zero')]
fn test_renormalize_of_a_singular_matrix_panics() {
    let _ = black_box(Rotation3 { matrix: Matrix3Trait::<Fixed>::zeros() }).renormalize();
}

// --- comparison

#[test]
fn test_abs_diff_eq_counts_ulps() {
    assert!(third().abs_diff_eq(third(), 0));
    let drifted = Rotation3 {
        matrix: Matrix3 {
            m11: fx(2),
            m21: fx(ONE_RAW - 2),
            m31: fx(0),
            m12: fx(0),
            m22: fx(1),
            m32: fx(ONE_RAW),
            m13: fx(ONE_RAW),
            m23: fx(0),
            m33: fx(0),
        },
    };
    assert!(third().abs_diff_eq(drifted, 2));
    assert!(!third().abs_diff_eq(drifted, 1));
}
