//! Unit tests of `UnitQuaternion`: exact rotations (identity, half turns, the 120° rotation about
//! `(1, 1, 1)`), the rotation identities (`q · q⁻¹ = 1`, `R · Rᵀ = I`, `q → R → q`), the
//! degenerate cases (`Option` / `#[should_panic]`), and the oracle vectors of `tools/oracle`
//! (upstream nalgebra 0.35 on the same raw inputs, tolerance in ulp).
//!
//! Oracle unit quaternions and rotation matrices are normalised in f64 then quantised, so their
//! norm is 1 within about 2 ulp, not exactly 1 — exactly what this library produces.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 8 cases per distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo quaternion --from vectors --max-per-dist 8 \
//!     --ops unit_quaternion_mul,unit_quaternion_inverse,unit_quaternion_from_axis_angle,\
//! unit_quaternion_from_scaled_axis,unit_quaternion_scaled_axis,unit_quaternion_angle,\
//! unit_quaternion_angle_to,unit_quaternion_to_rotation_matrix,\
//! unit_quaternion_from_rotation_matrix,unit_quaternion_transform_vector,\
//! unit_quaternion_inverse_transform_vector,unit_quaternion_slerp,unit_quaternion_nlerp,\
//! unit_quaternion_rotation_between,unit_quaternion_euler_angles,\
//! unit_quaternion_from_euler_angles,unit_quaternion_append_axisangle_linearized \
//!     --out crates/nalgebra/src/geometry/unit_quaternion/oracle.cairo
//! ```

use nalgebra_testing::black_box;
use simba::fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix3::{Matrix3, Matrix3Trait};
use crate::base::point3::Point3;
use crate::base::unit::{Unit, Unit3Trait};
use crate::base::vector3::{Vector3, Vector3Trait};
use crate::geometry::quaternion::{Quaternion, QuaternionTrait};
use crate::geometry::rotation3::{Rotation3, Rotation3Trait};
use super::{UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait, oracle};

const ONE_RAW: i64 = 0x100000000;
/// 1/2 in raw units.
const HALF_RAW: i64 = 0x80000000;

fn fx(raw: i64) -> Fixed {
    Fixed { raw }
}

fn int(v: i64) -> Fixed {
    Fixed { raw: v * ONE_RAW }
}

fn v3(t: (i64, i64, i64)) -> Vector3<Fixed> {
    let (x, y, z) = t;
    Vector3 { x: fx(x), y: fx(y), z: fx(z) }
}

fn v3i(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    Vector3 { x: int(x), y: int(y), z: int(z) }
}

fn u3(t: (i64, i64, i64)) -> Unit<Vector3<Fixed>> {
    Unit { value: v3(t) }
}

/// Unit quaternion from raw components, in the `(w, i, j, k)` order of the oracle.
fn uq(t: (i64, i64, i64, i64)) -> UnitQuaternion<Fixed> {
    let (w, i, j, k) = t;
    UnitQuaternion { quaternion: Quaternion { i: fx(i), j: fx(j), k: fx(k), w: fx(w) } }
}

/// `Matrix3` from raw ROW-major rows (oracle layout).
fn m3(rows: [[i64; 3]; 3]) -> Matrix3<Fixed> {
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

/// `Rotation3` from raw ROW-major rows (oracle layout).
fn r3(rows: [[i64; 3]; 3]) -> Rotation3<Fixed> {
    Rotation3 { matrix: m3(rows) }
}

/// `Rotation3` from integer ROW-major rows.
fn r3i(rows: [[i64; 3]; 3]) -> Rotation3<Fixed> {
    let [[m11, m12, m13], [m21, m22, m23], [m31, m32, m33]] = rows;
    Rotation3 {
        matrix: Matrix3 {
            m11: int(m11),
            m21: int(m21),
            m31: int(m31),
            m12: int(m12),
            m22: int(m22),
            m32: int(m32),
            m13: int(m13),
            m23: int(m23),
            m33: int(m33),
        },
    }
}

/// The half turn about `x`: `(0, 1, 0, 0)`, an exact unit quaternion.
fn half_x() -> UnitQuaternion<Fixed> {
    uq((0, ONE_RAW, 0, 0))
}

/// The 120° rotation about `(1, 1, 1) / sqrt(3)`: `(1, 1, 1, 1) / 2`, exact, and its matrix is the
/// cyclic permutation `(x, y, z) -> (z, x, y)`.
fn third() -> UnitQuaternion<Fixed> {
    uq((HALF_RAW, HALF_RAW, HALF_RAW, HALF_RAW))
}

/// `|a - b|` in raw units.
fn ulp(a: Fixed, b: Fixed) -> u128 {
    let d: i128 = a.raw.into() - b.raw.into();
    let d = if d < 0 {
        -d
    } else {
        d
    };
    d.try_into().unwrap()
}

/// Largest component-wise distance between two quaternions, in raw units.
fn max_ulp_q(a: Quaternion<Fixed>, b: Quaternion<Fixed>) -> u128 {
    let mut e = ulp(a.i, b.i);
    e = core::cmp::max(e, ulp(a.j, b.j));
    e = core::cmp::max(e, ulp(a.k, b.k));
    core::cmp::max(e, ulp(a.w, b.w))
}

// --- constructors, parts, identity

#[test]
fn test_identity_is_exact() {
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(id.quaternion == QuaternionTrait::identity());
    assert!(id.quaternion.norm() == Real::ONE);
    assert!(id.quaternion() == QuaternionTrait::identity());
    assert!(id.imag() == Vector3Trait::zeros());
    // The constructors of upstream's `Unit<Quaternion>`.
    assert!(UnitQuaternionTrait::new_unchecked(QuaternionTrait::<Fixed>::identity()) == id);
    assert!(UnitQuaternionTrait::new_normalize(QuaternionTrait::<Fixed>::identity()) == id);
}

#[test]
fn test_new_normalize_makes_a_unit_quaternion() {
    // (1, 2, -3, 4) / sqrt(30), the same values as `Quaternion::normalize`.
    let q = QuaternionTrait::new(int(1), int(2), int(-3), int(4));
    let u: UnitQuaternion<Fixed> = UnitQuaternionTrait::new_normalize(q);
    assert!(u.quaternion == q.normalize());
    assert!(u.quaternion.norm().abs_diff_eq(Real::ONE, 2));
}

#[test]
fn test_imag_and_quaternion_accessors() {
    assert!(third().imag() == v3((HALF_RAW, HALF_RAW, HALF_RAW)));
    assert!(third().quaternion().scalar() == fx(HALF_RAW));
    assert!(third().into_inner() == third().quaternion);
    assert!(UnitQuaternionTrait::try_new(third().quaternion, Real::ZERO) == Some(third()));
    assert!(UnitQuaternionTrait::try_new(QuaternionTrait::<Fixed>::zero(), Real::ZERO) == None);
}

// --- conjugate, inverse, composition

#[test]
fn test_inverse_is_the_conjugate_and_exact() {
    assert!(third().inverse() == uq((HALF_RAW, -HALF_RAW, -HALF_RAW, -HALF_RAW)));
    assert!(third().inverse() == third().conjugate());
    assert!(third().inverse().inverse() == third());
    assert!(half_x().inverse() == uq((0, -ONE_RAW, 0, 0)));
}

#[test]
fn test_inverse_oracle_is_exact() {
    let mut cases = oracle::unit_quaternion_inverse_cases();
    assert!(cases.len() == 8);
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(tol == 0);
        assert!(uq(rq).inverse() == uq(expected));
    }
}

#[test]
fn test_mul_identity_and_inverse() {
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(id * third() == third());
    assert!(third() * id == third());
    // q · q⁻¹ = 1 exactly here: every product is a multiple of 1/4.
    assert!(third() * third().inverse() == id);
    assert!(half_x() * half_x().inverse() == id);
    // Three 120° turns about (1, 1, 1) are the identity.
    assert!(third() * third() * third() == -id);
}

/// The oracle expectation is the EXACT floor of the Hamilton product, which one rescale per
/// component reproduces bit for bit.
#[test]
fn test_mul_oracle_is_bit_exact() {
    let mut cases = oracle::unit_quaternion_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, expected, _tol) = *case;
        assert!(uq(ra) * uq(rb) == uq(expected));
        // The product of two unit quaternions is a unit quaternion.
        assert!((uq(ra) * uq(rb)).quaternion.norm().abs_diff_eq(Real::ONE, 2));
    }
}

#[test]
fn test_rotation_to_and_composition() {
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(id.rotation_to(third()) == third());
    assert!(third().rotation_to(third()) == id);
    // rotation_to(b) · a = b.
    let r = half_x().rotation_to(third());
    assert!((r * half_x()).quaternion.abs_diff_eq(third().quaternion, 2));
}

// --- transforms

#[test]
fn test_transform_vector_exact_rotations() {
    // The half turn about x flips y and z; the 120° turn about (1, 1, 1) permutes the axes.
    assert!(half_x().transform_vector(v3i(1, 2, 3)) == v3i(1, -2, -3));
    assert!(third().transform_vector(v3i(1, 2, 3)) == v3i(3, 1, 2));
    assert!(third().inverse_transform_vector(v3i(3, 1, 2)) == v3i(1, 2, 3));
    assert!(
        UnitQuaternionTrait::<Fixed>::identity().transform_vector(v3i(1, 2, 3)) == v3i(1, 2, 3),
    );
    // Points transform like their coordinates (a rotation fixes the origin).
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
fn test_transform_vector_oracle() {
    let mut cases = oracle::unit_quaternion_transform_vector_cases();
    assert!(cases.len() == 32);
    while let Some(case) = cases.pop_front() {
        let (rq, rv, expected, tol) = *case;
        assert!(uq(rq).transform_vector(v3(rv)).abs_diff_eq(v3(expected), tol));
    }
}

#[test]
fn test_inverse_transform_vector_oracle() {
    let mut cases = oracle::unit_quaternion_inverse_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, rv, expected, tol) = *case;
        assert!(uq(rq).inverse_transform_vector(v3(rv)).abs_diff_eq(v3(expected), tol));
    }
}

/// A rotation preserves lengths, and the inverse transform undoes it.
#[test]
fn test_transform_vector_round_trip() {
    let mut cases = oracle::unit_quaternion_to_rotation_matrix_cases();
    let v = v3i(1, -2, 3);
    while let Some(case) = cases.pop_front() {
        let (rq, _, _) = *case;
        let q = uq(rq);
        let r = q.transform_vector(v);
        assert!(r.norm().abs_diff_eq(v.norm(), 8));
        assert!(q.inverse_transform_vector(r).abs_diff_eq(v, 16));
    }
}

#[test]
fn test_transform_vector_matches_the_rotation_matrix() {
    let mut cases = oracle::unit_quaternion_transform_vector_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, rv, _, tol) = *case;
        let q = uq(rq);
        let direct = q.transform_vector(v3(rv));
        let through_matrix = q.to_rotation_matrix().transform_vector(v3(rv));
        assert!(direct.abs_diff_eq(through_matrix, tol));
    }
}

// --- rotation matrix conversions

#[test]
fn test_to_rotation_matrix_exact() {
    assert!(
        UnitQuaternionTrait::<Fixed>::identity().to_rotation_matrix() == Rotation3Trait::identity(),
    );
    assert!(half_x().to_rotation_matrix() == r3i([[1, 0, 0], [0, -1, 0], [0, 0, -1]]));
    assert!(third().to_rotation_matrix() == r3i([[0, 0, 1], [1, 0, 0], [0, 1, 0]]));
    // -q is the same rotation.
    assert!((-third()).to_rotation_matrix() == third().to_rotation_matrix());
}

#[test]
fn test_to_rotation_matrix_oracle() {
    let mut cases = oracle::unit_quaternion_to_rotation_matrix_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(uq(rq).to_rotation_matrix().matrix.abs_diff_eq(m3(expected), tol));
    }
}

/// `R · Rᵀ = I` for every matrix produced by `to_rotation_matrix`.
#[test]
fn test_to_rotation_matrix_is_orthonormal() {
    let mut cases = oracle::unit_quaternion_to_rotation_matrix_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, _, _) = *case;
        let m = uq(rq).to_rotation_matrix().matrix;
        assert!((m * m.transpose()).is_identity(8));
        assert!(m.determinant().abs_diff_eq(Real::ONE, 8));
    }
}

#[test]
fn test_from_rotation_matrix_exact() {
    assert!(
        UnitQuaternionTrait::from_rotation_matrix(
            Rotation3Trait::<Fixed>::identity(),
        ) == UnitQuaternionTrait::identity(),
    );
    assert!(
        UnitQuaternionTrait::from_rotation_matrix(
            r3i([[0, 0, 1], [1, 0, 0], [0, 1, 0]]),
        ) == third(),
    );
    assert!(
        UnitQuaternionTrait::from_rotation_matrix(
            r3i([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
        ) == half_x(),
    );
    // The half turns about y and z exercise the third and fourth branches.
    assert!(
        UnitQuaternionTrait::from_rotation_matrix(
            r3i([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]),
        ) == uq((0, 0, ONE_RAW, 0)),
    );
    assert!(
        UnitQuaternionTrait::from_rotation_matrix(
            r3i([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]),
        ) == uq((0, 0, 0, ONE_RAW)),
    );
}

#[test]
fn test_from_rotation_matrix_oracle() {
    let mut cases = oracle::unit_quaternion_from_rotation_matrix_cases();
    while let Some(case) = cases.pop_front() {
        let (rr, expected, tol) = *case;
        let got = UnitQuaternionTrait::from_rotation_matrix(r3(rr));
        // The sign is upstream's: the branch selection matches, so the bits do too.
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    }
}

/// `q → R → q` within 3 ulp per component (measured), up to the sign (`q` and `-q` are the same
/// rotation and Shepperd's branch does not preserve the sign of the input).
#[test]
fn test_rotation_matrix_round_trip() {
    let mut cases = oracle::unit_quaternion_to_rotation_matrix_cases();
    let mut worst: u128 = 0;
    while let Some(case) = cases.pop_front() {
        let (rq, _, _) = *case;
        let q = uq(rq);
        let back = UnitQuaternionTrait::from_rotation_matrix(q.to_rotation_matrix());
        let e = core::cmp::min(
            max_ulp_q(back.quaternion, q.quaternion), max_ulp_q(back.quaternion, (-q).quaternion),
        );
        worst = core::cmp::max(worst, e);
    }
    assert!(worst <= 3, "q -> R -> q round trip off by {} ulp", worst);
}

/// `R → q → R` within 4 ulp per entry (measured).
#[test]
fn test_quaternion_round_trip_from_a_matrix() {
    let mut cases = oracle::unit_quaternion_from_rotation_matrix_cases();
    while let Some(case) = cases.pop_front() {
        let (rr, _, _) = *case;
        let r = r3(rr);
        let back = UnitQuaternionTrait::from_rotation_matrix(r).to_rotation_matrix();
        assert!(back.matrix.abs_diff_eq(r.matrix, 4));
    }
}

#[test]
fn test_to_homogeneous() {
    let h = third().to_homogeneous();
    assert!(h.m11 == Real::ZERO && h.m13 == Real::ONE && h.m21 == Real::ONE);
    assert!(h.m44 == Real::ONE);
    assert!(h.m41 == Real::ZERO && h.m42 == Real::ZERO && h.m43 == Real::ZERO);
    assert!(h.m14 == Real::ZERO && h.m24 == Real::ZERO && h.m34 == Real::ZERO);
}

// --- axis and angle

#[test]
fn test_axis_angle_exact_cases() {
    // 120° about (1, 1, 1) / sqrt(3): the axis components are 1 / sqrt(3) = 0.5773502691.
    let (axis, angle) = third().axis_angle().unwrap();
    let inv_sqrt3 = 2479700524;
    assert!(axis.value.abs_diff_eq(v3((inv_sqrt3, inv_sqrt3, inv_sqrt3)), 2));
    // 2 pi / 3 = 2.0943951023 (8 995 358 469 raw); `atan2` is accurate to about 12 ulp.
    assert!(angle.abs_diff_eq(fx(8995358469), 16));
    // Half turn about x: the axis is exact and the angle is pi.
    assert!(half_x().axis().unwrap() == Unit3Trait::<Fixed>::x_axis());
    assert!(half_x().angle().abs_diff_eq(Real::PI, 16));
}

#[test]
fn test_axis_of_the_identity_is_none() {
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(id.axis() == None);
    assert!(id.axis_angle() == None);
    assert!(id.angle() == Real::ZERO);
    assert!(id.scaled_axis() == Vector3Trait::zeros());
    // -identity is the identity too (a rotation by 2 pi).
    assert!((-id).axis() == None);
    assert!((-id).angle() == Real::ZERO);
}

/// The axis is flipped so that the angle lies in `[0, pi]` (upstream's convention).
#[test]
fn test_axis_sign_convention() {
    let q = -third();
    assert!(q.quaternion.w < Real::ZERO);
    assert!(q.axis().unwrap() == third().axis().unwrap());
    assert!(q.angle() == third().angle());
    assert!(q.scaled_axis() == third().scaled_axis());
}

#[test]
fn test_angle_oracle() {
    let mut cases = oracle::unit_quaternion_angle_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(Real::abs_diff_eq(uq(rq).angle(), fx(expected), tol));
    }
}

#[test]
fn test_angle_to_oracle() {
    let mut cases = oracle::unit_quaternion_angle_to_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, expected, tol) = *case;
        assert!(Real::abs_diff_eq(uq(ra).angle_to(uq(rb)), fx(expected), tol));
    }
}

#[test]
fn test_angle_to_is_a_rotation_metric() {
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(id.angle_to(id) == Real::ZERO);
    assert!(third().angle_to(third()) == Real::ZERO);
    // q and -q are the same rotation: zero angle, although the components are far apart.
    assert!(third().angle_to(-third()) == Real::ZERO);
    assert!(third().angle_to(id).abs_diff_eq(third().angle(), 1));
}

#[test]
fn test_scaled_axis_oracle() {
    let mut cases = oracle::unit_quaternion_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        assert!(uq(rq).scaled_axis().abs_diff_eq(v3(expected), tol));
    }
}

// --- axis-angle and Euler-angle constructors

#[test]
fn test_from_axis_angle_exact_cases() {
    let z = Unit3Trait::<Fixed>::z_axis();
    // A zero angle is the identity exactly (sin 0 = 0, cos 0 = 1).
    assert!(
        UnitQuaternionAngleTrait::from_axis_angle(
            z, Real::<Fixed>::ZERO,
        ) == UnitQuaternionTrait::identity(),
    );
    // A quarter turn about z: (cos(pi/4), 0, 0, sin(pi/4)) = 0.7071067811 both.
    let q = UnitQuaternionAngleTrait::from_axis_angle(z, Real::<Fixed>::FRAC_PI_2);
    assert!(q.quaternion.w.abs_diff_eq(Real::FRAC_1_SQRT_2, 4));
    assert!(q.quaternion.k.abs_diff_eq(Real::FRAC_1_SQRT_2, 4));
    assert!(q.quaternion.i == Real::ZERO && q.quaternion.j == Real::ZERO);
    // It maps x to y within a few ulp.
    assert!(q.transform_vector(v3i(1, 0, 0)).abs_diff_eq(v3i(0, 1, 0), 8));
    // A half turn about x: cos(pi/2) is 0 within 1 ulp, sin(pi/2) is 1.
    let h = UnitQuaternionAngleTrait::from_axis_angle(
        Unit3Trait::<Fixed>::x_axis(), Real::<Fixed>::PI,
    );
    assert!(h.quaternion.abs_diff_eq(half_x().quaternion, 2));
}

#[test]
fn test_from_axis_angle_oracle() {
    let mut cases = oracle::unit_quaternion_from_axis_angle_cases();
    while let Some(case) = cases.pop_front() {
        let (raxis, rangle, expected, tol) = *case;
        let got = UnitQuaternionAngleTrait::from_axis_angle(u3(raxis), fx(rangle));
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    }
}

#[test]
fn test_from_scaled_axis_exact_and_oracle() {
    // A zero rotation vector is the identity (no division by zero).
    assert!(
        UnitQuaternionAngleTrait::from_scaled_axis(
            Vector3Trait::<Fixed>::zeros(),
        ) == UnitQuaternionTrait::identity(),
    );
    // Same rotation as from_axis_angle(axis, |v|).
    let axis = Unit3Trait::<Fixed>::y_axis();
    let scaled = UnitQuaternionAngleTrait::from_scaled_axis(v3((0, ONE_RAW, 0)));
    let direct = UnitQuaternionAngleTrait::from_axis_angle(axis, Real::ONE);
    assert!(scaled.quaternion.abs_diff_eq(direct.quaternion, 2));
    let mut cases = oracle::unit_quaternion_from_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (rv, expected, tol) = *case;
        let got = UnitQuaternionAngleTrait::from_scaled_axis(v3(rv));
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    }
}

/// `scaled_axis` is the inverse of `from_scaled_axis` (the exponential and logarithmic maps).
#[test]
fn test_scaled_axis_round_trip() {
    let mut cases = oracle::unit_quaternion_from_scaled_axis_cases();
    while let Some(case) = cases.pop_front() {
        let (rv, _, _) = *case;
        let back = UnitQuaternionAngleTrait::from_scaled_axis(v3(rv)).scaled_axis();
        assert!(back.abs_diff_eq(v3(rv), 64));
    }
}

#[test]
fn test_from_euler_angles_oracle() {
    let mut cases = oracle::unit_quaternion_from_euler_angles_cases();
    while let Some(case) = cases.pop_front() {
        let (reuler, expected, tol) = *case;
        let (roll, pitch, yaw) = reuler;
        let got = UnitQuaternionAngleTrait::from_euler_angles(fx(roll), fx(pitch), fx(yaw));
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    }
}

#[test]
fn test_euler_angles_oracle() {
    let mut cases = oracle::unit_quaternion_euler_angles_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, expected, tol) = *case;
        let (roll, pitch, yaw) = uq(rq).euler_angles();
        let (eroll, epitch, eyaw) = expected;
        assert!(Real::abs_diff_eq(roll, fx(eroll), tol));
        assert!(Real::abs_diff_eq(pitch, fx(epitch), tol));
        assert!(Real::abs_diff_eq(yaw, fx(eyaw), tol));
    }
}

/// The Euler angles rebuild the same rotation (the oracle excludes the gimbal lock).
#[test]
fn test_euler_angles_round_trip() {
    let mut cases = oracle::unit_quaternion_euler_angles_cases();
    while let Some(case) = cases.pop_front() {
        let (rq, _, _) = *case;
        let q = uq(rq);
        let (roll, pitch, yaw) = q.euler_angles();
        let back = UnitQuaternionAngleTrait::from_euler_angles(roll, pitch, yaw);
        assert!(back.angle_to(q).abs_diff_eq(Real::ZERO, 256));
    }
}

/// Single-axis Euler angles are the rotations about the axes, and the convention is
/// `Rz(yaw)·Ry(pitch)·Rx(roll)`.
#[test]
fn test_euler_angles_convention() {
    let half = fx(HALF_RAW);
    let q = UnitQuaternionAngleTrait::from_euler_angles(half, Real::ZERO, Real::ZERO);
    let x = UnitQuaternionAngleTrait::from_axis_angle(Unit3Trait::<Fixed>::x_axis(), half);
    assert!(q.quaternion.abs_diff_eq(x.quaternion, 2));
    let q = UnitQuaternionAngleTrait::from_euler_angles(Real::ZERO, half, Real::ZERO);
    let y = UnitQuaternionAngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), half);
    assert!(q.quaternion.abs_diff_eq(y.quaternion, 2));
    let q = UnitQuaternionAngleTrait::from_euler_angles(Real::ZERO, Real::ZERO, half);
    let z = UnitQuaternionAngleTrait::from_axis_angle(Unit3Trait::<Fixed>::z_axis(), half);
    assert!(q.quaternion.abs_diff_eq(z.quaternion, 2));
    // The composition order: Rz · Ry · Rx.
    let (r, p, y2) = (fx(0x20000000), fx(0x40000000), fx(0x60000000));
    let composed = UnitQuaternionAngleTrait::from_axis_angle(Unit3Trait::<Fixed>::z_axis(), y2)
        * UnitQuaternionAngleTrait::from_axis_angle(Unit3Trait::<Fixed>::y_axis(), p)
        * UnitQuaternionAngleTrait::from_axis_angle(Unit3Trait::<Fixed>::x_axis(), r);
    assert!(
        UnitQuaternionAngleTrait::from_euler_angles(r, p, y2)
            .quaternion
            .abs_diff_eq(composed.quaternion, 8),
    );
}

#[test]
fn test_euler_angles_gimbal_lock() {
    // m31 = -1: pitch = +pi/2, yaw = 0 (upstream's choice).
    let q = UnitQuaternionAngleTrait::from_euler_angles(
        Real::ZERO, Real::<Fixed>::FRAC_PI_2, Real::ZERO,
    );
    let (roll, pitch, yaw) = q.euler_angles();
    assert!(pitch.abs_diff_eq(Real::FRAC_PI_2, 64));
    assert!(yaw == Real::ZERO || roll.abs_diff_eq(Real::ZERO, 64));
    // The rotation is still rebuilt.
    let back = UnitQuaternionAngleTrait::from_euler_angles(roll, pitch, yaw);
    assert!(back.angle_to(q).abs_diff_eq(Real::ZERO, 1024));
}

// --- rotation between vectors

#[test]
fn test_rotation_between_exact_cases() {
    let x = v3i(1, 0, 0);
    let y = v3i(0, 1, 0);
    // A quarter turn about z: (cos 45°, 0, 0, sin 45°), algebraic so within 1 ulp.
    let q = UnitQuaternionTrait::rotation_between(x, y).unwrap();
    assert!(q.quaternion.w.abs_diff_eq(Real::FRAC_1_SQRT_2, 2));
    assert!(q.quaternion.k.abs_diff_eq(Real::FRAC_1_SQRT_2, 2));
    assert!(q.transform_vector(x).abs_diff_eq(y, 4));
    // Parallel vectors (any length) give the identity.
    assert!(
        UnitQuaternionTrait::rotation_between(
            x, v3i(3, 0, 0),
        ) == Some(UnitQuaternionTrait::identity()),
    );
    // Antiparallel vectors have no defined axis.
    assert!(UnitQuaternionTrait::rotation_between(x, v3i(-2, 0, 0)) == None);
    // A zero-length input has no direction: the identity, like upstream.
    assert!(
        UnitQuaternionTrait::rotation_between(
            Vector3Trait::zeros(), y,
        ) == Some(UnitQuaternionTrait::identity()),
    );
    assert!(
        UnitQuaternionTrait::rotation_between(
            x, Vector3Trait::zeros(),
        ) == Some(UnitQuaternionTrait::identity()),
    );
}

#[test]
fn test_rotation_between_oracle() {
    let mut cases = oracle::unit_quaternion_rotation_between_cases();
    assert!(cases.len() >= 20);
    while let Some(case) = cases.pop_front() {
        let (ra, rb, expected, tol) = *case;
        let got = UnitQuaternionTrait::rotation_between(v3(ra), v3(rb)).unwrap();
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 4));
    }
}

/// The rotation takes the direction of `a` to the direction of `b`.
#[test]
fn test_rotation_between_maps_a_to_b() {
    let mut cases = oracle::unit_quaternion_rotation_between_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, _, tol) = *case;
        let q = UnitQuaternionTrait::rotation_between(v3(ra), v3(rb)).unwrap();
        let ua = v3(ra).normalize();
        let ub = v3(rb).normalize();
        assert!(q.transform_vector(ua).abs_diff_eq(ub, tol * 4 + 8));
    }
}

/// `scaled_rotation_between(a, b, 1)` is `rotation_between(a, b)`, through upstream's trigonometric
/// path: the two agree within the accuracy of `acos` and `sin_cos`.
#[test]
fn test_scaled_rotation_between() {
    let x = v3i(1, 0, 0);
    let y = v3i(0, 1, 0);
    let full = UnitQuaternionAngleTrait::scaled_rotation_between(x, y, Real::ONE).unwrap();
    assert!(
        full
            .quaternion
            .abs_diff_eq(UnitQuaternionTrait::rotation_between(x, y).unwrap().quaternion, 8),
    );
    // Half of the rotation: an eighth of a turn about z.
    let half = UnitQuaternionAngleTrait::scaled_rotation_between(x, y, Real::HALF).unwrap();
    assert!(
        half
            .quaternion
            .abs_diff_eq(
                UnitQuaternionAngleTrait::from_axis_angle(
                    Unit3Trait::<Fixed>::z_axis(), Real::FRAC_PI_4,
                )
                    .quaternion,
                8,
            ),
    );
    assert!(UnitQuaternionAngleTrait::scaled_rotation_between(x, v3i(-1, 0, 0), Real::ONE) == None);
    assert!(
        UnitQuaternionAngleTrait::scaled_rotation_between(
            x, v3i(2, 0, 0), Real::ONE,
        ) == Some(UnitQuaternionTrait::identity()),
    );
}

// --- interpolation

#[test]
fn test_nlerp_endpoints_and_oracle() {
    // t = 0 and t = 1 give the (renormalized) endpoints.
    let a = third();
    let b = half_x();
    assert!(a.nlerp(b, Real::ZERO).quaternion.abs_diff_eq(a.quaternion, 2));
    assert!(a.nlerp(b, Real::ONE).quaternion.abs_diff_eq(b.quaternion, 2));
    let mut cases = oracle::unit_quaternion_nlerp_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, t, expected, tol) = *case;
        let got = uq(ra).nlerp(uq(rb), fx(t));
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 2));
    }
}

#[test]
#[should_panic(expected: 'simba: division by zero')]
fn test_nlerp_of_opposite_quaternions_panics() {
    let a = black_box(third());
    let _ = a.nlerp(black_box(-a), Real::HALF);
}

#[test]
fn test_slerp_endpoints_and_oracle() {
    let a = third();
    let b = half_x();
    assert!(a.slerp(b, Real::ZERO).quaternion.abs_diff_eq(a.quaternion, 4));
    assert!(a.slerp(b, Real::ONE).quaternion.abs_diff_eq(b.quaternion, 4));
    // Interpolating a rotation with itself is that rotation (the early exit).
    assert!(a.slerp(a, Real::HALF) == a);
    let mut cases = oracle::unit_quaternion_slerp_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, t, expected, tol) = *case;
        let got = uq(ra).slerp(uq(rb), fx(t));
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 2));
    }
}

/// `slerp` takes the shortest arc: interpolating towards `-b` gives the same rotation as towards
/// `b` (upstream negates `b` when the dot product is negative). `nlerp` does not.
#[test]
fn test_slerp_takes_the_shortest_arc() {
    let a = third();
    let b = half_x();
    let t = fx(0x40000000);
    assert!(a.slerp(b, t).quaternion.abs_diff_eq(a.slerp(-b, t).quaternion, 4));
    assert!(!a.nlerp(b, t).quaternion.abs_diff_eq(a.nlerp(-b, t).quaternion, 4));
}

/// Halfway along the arc, `slerp` is at half the angle from both ends, unlike `nlerp`.
#[test]
fn test_slerp_has_constant_angular_velocity() {
    let a = UnitQuaternionTrait::<Fixed>::identity();
    let b = UnitQuaternionAngleTrait::from_axis_angle(
        Unit3Trait::<Fixed>::z_axis(), Real::<Fixed>::FRAC_PI_2,
    );
    let mid = a.slerp(b, Real::HALF);
    assert!(a.angle_to(mid).abs_diff_eq(mid.angle_to(b), 16));
    assert!(a.angle_to(mid).abs_diff_eq(Real::FRAC_PI_4, 32));
}

/// The shortest-arc flip makes `slerp` TOTAL on `[-1, 1]`: `-a` is the same rotation as `a`, so the
/// arc is empty and the early return gives `a` back instead of the undefined interpolation upstream
/// guards against. With `epsilon = 0` `try_slerp` therefore never returns `None` in Q32.32 (that
/// would need `sqrt(1 - c²) = 0` with `c < 1`, impossible in raw units), and `slerp` never panics.
#[test]
fn test_try_slerp_is_total_with_a_zero_epsilon() {
    let a = third();
    assert!(a.try_slerp(-a, Real::HALF, Real::ZERO) == Some(a));
    assert!(a.slerp(-a, Real::HALF) == a);
    assert!(a.try_slerp(a, Real::HALF, Real::ZERO) == Some(a));
    assert!(a.try_slerp(half_x(), Real::HALF, Real::ZERO) != None);
}

/// A positive `epsilon` rejects nearly aligned rotations (`sin` of the half angle below it), where
/// the interpolation direction is ill-conditioned.
#[test]
fn test_try_slerp_rejects_nearly_aligned_rotations() {
    let a = UnitQuaternionTrait::<Fixed>::identity();
    // 1/256 rad apart: sin of the half angle is about 1/512.
    let b = UnitQuaternionAngleTrait::from_axis_angle(
        Unit3Trait::<Fixed>::z_axis(), fx(ONE_RAW / 256),
    );
    assert!(a.try_slerp(b, Real::HALF, Real::HALF) == None);
    assert!(a.try_slerp(b, Real::HALF, fx(ONE_RAW / 4096)) != None);
}

#[test]
#[should_panic(expected: 'nalgebra: ambiguous slerp')]
fn test_slerp_panics_when_try_slerp_rejects() {
    // Only reachable through a `try_slerp` epsilon; `slerp` itself passes 0 (see above).
    let a = black_box(UnitQuaternionTrait::<Fixed>::identity());
    let b = black_box(third());
    match a.try_slerp(b, Real::HALF, Real::ONE) {
        Some(_) => (),
        None => core::panic_with_felt252('nalgebra: ambiguous slerp'),
    }
}

// --- powf

#[test]
fn test_powf() {
    let q = third();
    let id = UnitQuaternionTrait::<Fixed>::identity();
    assert!(q.powf(Real::ZERO).quaternion.abs_diff_eq(id.quaternion, 2));
    assert!(q.powf(Real::ONE).quaternion.abs_diff_eq(q.quaternion, 8));
    // Three times 120° is a full turn: -identity or identity depending on the sign convention.
    let cubed = q.powf(Real::from_int(3));
    assert!(cubed.angle().abs_diff_eq(Real::ZERO, 64) || cubed.angle().abs_diff_eq(Real::TAU, 64));
    // Twice the angle is the square of the rotation.
    assert!(q.powf(Real::TWO).quaternion.abs_diff_eq((q * q).quaternion, 16));
    // The identity to any power is the identity.
    assert!(id.powf(Real::from_int(7)) == id);
}

// --- integration (rapier's hot path)

#[test]
fn test_append_axisangle_linearized_oracle() {
    let mut cases = oracle::unit_quaternion_append_axisangle_linearized_cases();
    assert!(cases.len() == 8);
    while let Some(case) = cases.pop_front() {
        let (rq, rv, expected, tol) = *case;
        let got = uq(rq).append_axisangle_linearized(v3(rv));
        assert!(got.quaternion.abs_diff_eq(uq(expected).quaternion, tol));
        assert!(got.quaternion.norm().abs_diff_eq(Real::ONE, 2));
    }
}

#[test]
fn test_append_axisangle_linearized_zero_is_a_no_op() {
    // The exact normalisation of an already unit quaternion moves it by at most 2 ulp.
    let q = third();
    assert!(
        q
            .append_axisangle_linearized(Vector3Trait::zeros())
            .quaternion
            .abs_diff_eq(q.quaternion, 2),
    );
}

/// For a small increment the linearised update agrees with the exact one to second order.
#[test]
fn test_append_axisangle_linearized_matches_the_exact_update() {
    let q = third();
    // |omega| = 2^-8: the second-order error is about |omega|² / 8 = 2e-6 (8 400 ulp).
    let w = v3((0x1000000, -0x800000, 0x400000));
    let got = q.append_axisangle_linearized(w);
    let exact = UnitQuaternionAngleTrait::from_scaled_axis(w) * q;
    assert!(got.quaternion.abs_diff_eq(exact.quaternion, 8400));
    // At 2^-16 the difference is under 1 ulp per component.
    let w = v3((0x10000, -0x8000, 0x4000));
    let got = q.append_axisangle_linearized(w);
    let exact = UnitQuaternionAngleTrait::from_scaled_axis(w) * q;
    assert!(got.quaternion.abs_diff_eq(exact.quaternion, 4));
}

/// 1 000 integration steps: the quaternion stays a unit quaternion (that is what the normalisation
/// inside `append_axisangle_linearized` is for) and the rotation stays exact to a few ulp.
#[test]
fn test_append_axisangle_linearized_is_stable() {
    // 1 024 steps of 2 pi / 1 024 about z: a full turn.
    let step = v3((0, 0, 26353589));
    let mut q = UnitQuaternionTrait::<Fixed>::identity();
    for _ in 0_u32..1024 {
        q = q.append_axisangle_linearized(step);
    }
    assert!(q.quaternion.norm().abs_diff_eq(Real::ONE, 2));
    // The linearised update over-rotates by |step|²/24 per step (tan(x/2) vs x/2): about 6e-4 rad
    // in total here, plus the rounding of 1 024 steps.
    assert!(
        q.angle().abs_diff_eq(Real::ZERO, 0x300000) || q.angle().abs_diff_eq(Real::TAU, 0x300000),
    );
}

// --- renormalization (from `UnitTrait`, on quaternions)

#[test]
fn test_renormalize_fast_on_a_drifted_rotation() {
    // 64 compositions of a 120° rotation: the norm drifts by a few ulp only.
    let mut q = third();
    for _ in 0_u32..64 {
        q = q * third();
    }
    let n = q.quaternion.norm();
    assert!(n.abs_diff_eq(Real::ONE, 64));
    let fixed = q.renormalize_fast();
    assert!(fixed.quaternion.norm().abs_diff_eq(Real::ONE, 2));
    assert!(q.renormalize().quaternion.norm().abs_diff_eq(Real::ONE, 2));
    // Both renormalisations agree to 1 ulp on such a small drift.
    assert!(fixed.quaternion.abs_diff_eq(q.renormalize().quaternion, 2));
}

#[test]
fn test_renormalize_fast_fixes_a_scaled_quaternion() {
    // The 120° rotation scaled by 1 + 1e-3.
    let s = fx(HALF_RAW + 2147483);
    let drifted = uq((s.raw, s.raw, s.raw, s.raw));
    assert!(!drifted.quaternion.norm().abs_diff_eq(Real::ONE, 1000));
    let once = drifted.renormalize_fast();
    assert!(once.quaternion.norm().abs_diff_eq(Real::ONE, 8000));
    assert!(once.renormalize_fast().quaternion.norm().abs_diff_eq(Real::ONE, 4));
}
