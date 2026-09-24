//! Unit tests of the WP 8.4-P09a completion of `Rotation3`: the oracle vectors of `tools/oracle`
//! (suite `rotation_matrix_completion`, upstream nalgebra 0.35 on the same raw inputs, tolerance
//! in ulp), exact cases, identities with the existing API, upstream's quirks and the panics.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix3::{Matrix3, Matrix3Trait};
use nalgebra::base::unit::Unit;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::isometry3::Isometry3;
use nalgebra::geometry::rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::unit_quaternion::{
    UnitQuaternion, UnitQuaternionAngleTrait, UnitQuaternionTrait,
};
use nalgebra_tests_utils::{
    ONE_RAW, excess, fx, int, m3, max_ulp_diff3, max_ulp_diff_q, max_ulp_diff_v3, qt, r3, r3i, u3,
    ulp_diff, uqt, v3i, v3t,
};
use simba::scalar::Real;
use crate::oracle;
use super::benches::{alt_angle_to_rotation_to_angle, alt_from_matrix_eps_matrix};

/// The excess of `err` over `tol`, printed when positive.
fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}

fn rot_err(a: Rotation3<Fixed>, b: Rotation3<Fixed>) -> u128 {
    max_ulp_diff3(a.matrix, b.matrix)
}

/// A rotation of about 1.2 rad about `(1, -2, 3) / |..|` (the first `rotation3_new` oracle input
/// family), built exactly from a raw matrix of the oracle.
fn a() -> Rotation3<Fixed> {
    Rotation3AngleTrait::from_euler_angles(fx(0x40000000), fx(-0x60000000), fx(0x90000000))
}

fn b() -> Rotation3<Fixed> {
    Rotation3AngleTrait::from_euler_angles(fx(-0x70000000), fx(0x20000000), fx(0x28000000))
}

fn x_axis() -> Unit<Vector3<Fixed>> {
    u3(ONE_RAW, 0, 0)
}

fn y_axis() -> Unit<Vector3<Fixed>> {
    u3(0, ONE_RAW, 0)
}

fn z_axis() -> Unit<Vector3<Fixed>> {
    u3(0, 0, ONE_RAW)
}

// --- oracle vectors

#[test]
fn test_new_oracle() {
    let mut cases = oracle::rotation3_new_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (v, e, tol) = *case;
        let got = Rotation3AngleTrait::new(v3t(v));
        worst = core::cmp::max(worst, report("new", n, rot_err(got, r3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_powf_oracle() {
    let mut cases = oracle::rotation3_powf_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (r, p, e, tol) = *case;
        let got = r3(r).powf(fx(p));
        worst = core::cmp::max(worst, report("powf", n, rot_err(got, r3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_angle_to_oracle() {
    let mut cases = oracle::rotation3_angle_to_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (x, y, e, tol) = *case;
        let got = r3(x).angle_to(r3(y));
        worst = core::cmp::max(worst, report("angle_to", n, ulp_diff(got, fx(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_axis_angle_oracle() {
    let mut cases = oracle::rotation3_axis_angle_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (r, axis, angle, tol) = *case;
        let (got_axis, got_angle) = r3(r).axis_angle().unwrap();
        let err = core::cmp::max(
            max_ulp_diff_v3(got_axis.value, v3t(axis)), ulp_diff(got_angle, fx(angle)),
        );
        worst = core::cmp::max(worst, report("axis_angle", n, err, tol));
        n += 1;
    }
    assert!(worst == 0);
}

/// `rotation_to` and `/` are exact: one floor per entry of the exact product.
#[test]
fn test_rotation_to_and_div_oracle_are_bit_exact() {
    let mut cases = oracle::rotation3_rotation_to_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, _) = *case;
        assert!(r3(x).rotation_to(r3(y)) == r3(e));
        // `b * a⁻¹` is `b / a`.
        assert!(r3(y) / r3(x) == r3(e));
    }
    let mut cases = oracle::rotation3_div_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, _) = *case;
        assert!(r3(x) / r3(y) == r3(e));
    }
}

fn check_euler(
    op: ByteArray,
    mut cases: Span<([[i64; 3]; 3], (i64, i64, i64), u64)>,
    seq: [Unit<Vector3<Fixed>>; 3],
    extrinsic: bool,
) -> u128 {
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (r, e, tol) = *case;
        let (angles, observable) = r3(r).euler_angles_ordered(seq, extrinsic);
        assert!(observable);
        let [a0, a1, a2] = angles;
        let err = max_ulp_diff_v3(Vector3 { x: a0, y: a1, z: a2 }, v3t(e));
        worst = core::cmp::max(worst, report(op.clone(), n, err, tol));
        n += 1;
    }
    worst
}

#[test]
fn test_euler_angles_ordered_oracle() {
    let (x, y, z) = (x_axis(), y_axis(), z_axis());
    let w = check_euler(
        "zyx", oracle::rotation3_euler_angles_ordered_zyx_cases(), [z, y, x], false,
    );
    let w = core::cmp::max(
        w,
        check_euler(
            "xyz extrinsic",
            oracle::rotation3_euler_angles_ordered_xyz_extrinsic_cases(),
            [x, y, z],
            true,
        ),
    );
    let w = core::cmp::max(
        w, check_euler("xzy", oracle::rotation3_euler_angles_ordered_xzy_cases(), [x, z, y], false),
    );
    assert!(w == 0);
}

#[test]
fn test_from_matrix_oracle() {
    let mut cases = oracle::rotation3_from_matrix_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let got = Rotation3AngleTrait::from_matrix(m3(m));
        worst = core::cmp::max(worst, report("from_matrix", n, rot_err(got, r3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_slerp_oracle() {
    let mut cases = oracle::rotation3_slerp_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (x, y, t, e, tol) = *case;
        let got = r3(x).slerp(r3(y), fx(t));
        worst = core::cmp::max(worst, report("slerp", n, rot_err(got, r3(e)), tol));
        // `try_slerp` with `epsilon = 0` is `Some(slerp)`.
        assert!(r3(x).try_slerp(r3(y), fx(t), Real::zero()) == Some(got));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_look_at_lh_oracle() {
    let mut cases = oracle::rotation3_look_at_lh_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (d, u, e, tol) = *case;
        let got = Rotation3Trait::look_at_lh(v3t(d), v3t(u));
        worst = core::cmp::max(worst, report("look_at_lh", n, rot_err(got, r3(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

#[test]
fn test_mul_div_unit_quaternion_oracle() {
    let mut cases = oracle::rotation3_mul_unit_quaternion_cases();
    let (mut worst, mut n) = (0, 0);
    while let Some(case) = cases.pop_front() {
        let (r, q, e, tol) = *case;
        let got = r3(r).mul_unit_quaternion(uqt(q));
        worst =
            core::cmp::max(worst, report("mul_uq", n, max_ulp_diff_q(got.quaternion, qt(e)), tol));
        // Upstream's formula, bit for bit.
        assert!(got == UnitQuaternionTrait::from_rotation_matrix(r3(r)) * uqt(q));
        n += 1;
    }
    let mut cases = oracle::rotation3_div_unit_quaternion_cases();
    while let Some(case) = cases.pop_front() {
        let (r, q, e, tol) = *case;
        let got = r3(r).div_unit_quaternion(uqt(q));
        worst =
            core::cmp::max(worst, report("div_uq", n, max_ulp_diff_q(got.quaternion, qt(e)), tol));
        n += 1;
    }
    assert!(worst == 0);
}

// --- parts, frames, unit vectors

#[test]
fn test_parts_and_basis() {
    let r = a();
    assert!(r.into_inner() == r.matrix && r.unwrap() == r.matrix);
    let m = r.matrix;
    let basis = [
        Vector3 { x: m.m11, y: m.m21, z: m.m31 }, Vector3 { x: m.m12, y: m.m22, z: m.m32 },
        Vector3 { x: m.m13, y: m.m23, z: m.m33 },
    ];
    assert!(Rotation3Trait::from_basis_unchecked(basis) == r);
}

#[test]
fn test_unit_vector_transforms() {
    let u = u3(0, ONE_RAW, 0);
    let r = a();
    assert!(r.transform_unit_vector(u).value == r.transform_vector(u.value));
    assert!(r.inverse_transform_unit_vector(u).value == r.inverse_transform_vector(u.value));
}

#[test]
fn test_observer_frames() {
    let (d, u) = (v3i(1, 2, -3), v3i(0, 1, 0));
    let f = Rotation3Trait::face_towards(d, u);
    assert!(Rotation3Trait::look_at_lh(d, u) == f.inverse());
    assert!(Rotation3Trait::new_observer_frames(d, u) == f);
    // `look_at_lh` maps `dir` onto the positive z axis.
    let z = Rotation3Trait::look_at_lh(d, u).transform_vector(d);
    assert!(Real::abs_diff_eq(z.x, fx(0), 8) && Real::abs_diff_eq(z.y, fx(0), 8));
}

// --- angle functions

#[test]
fn test_new_is_from_scaled_axis() {
    let v = v3t((0x40000000, -0x30000000, 0x20000000));
    assert!(Rotation3AngleTrait::new(v) == Rotation3AngleTrait::from_scaled_axis(v));
    assert!(Rotation3AngleTrait::new(v3i(0, 0, 0)) == Rotation3Trait::identity());
}

#[test]
fn test_axis_angle_cases() {
    assert!(Rotation3Trait::<Fixed>::identity().axis_angle() == None);
    let (axis, angle) = a().axis_angle().unwrap();
    assert!(Some(axis) == a().axis());
    assert!(angle == a().angle());
}

/// `angle_to` is `rotation_to(other).angle()` with one rounding instead of four: the two agree to
/// a few ulp on the oracle set, and the fused one is closer to upstream.
#[test]
fn test_angle_to_alt_rotation_to_angle_agrees() {
    let mut cases = oracle::rotation3_angle_to_cases();
    while let Some(case) = cases.pop_front() {
        let (x, y, e, tol) = *case;
        let alt = alt_angle_to_rotation_to_angle(r3(x), r3(y));
        assert!(ulp_diff(alt, fx(e)) <= tol.into() + 64);
    }
    assert!(a().angle_to(a()) < fx(0x10000));
}

#[test]
fn test_powf_cases() {
    let r = a();
    assert!(r.powf(Real::zero()).matrix.abs_diff_eq(Matrix3Trait::identity(), 4));
    assert!(rot_err(r.powf(Real::one()), r) <= 64);
    assert!(rot_err(r.powf(int(2)), r * r) <= 128);
    let id: Rotation3<Fixed> = Rotation3Trait::identity();
    assert!(id.powf(int(3)) == id);
}

/// Upstream's quirk, kept: a half turn (undetermined axis) to any power is `-I` when `m11 < 0`
/// (not a rotation), the identity otherwise.
#[test]
fn test_powf_of_a_half_turn_is_upstreams() {
    let half_z = r3i([[-1, 0, 0], [0, -1, 0], [0, 0, 1]]);
    assert!(half_z.powf(Real::HALF) == r3i([[-1, 0, 0], [0, -1, 0], [0, 0, -1]]));
    let half_x = r3i([[1, 0, 0], [0, -1, 0], [0, 0, -1]]);
    assert!(half_x.powf(Real::HALF) == Rotation3Trait::identity());
}

#[test]
fn test_to_euler_angles_is_euler_angles() {
    assert!(a().to_euler_angles() == a().euler_angles());
}

/// The intrinsic `zyx` decomposition of `from_euler_angles(roll, pitch, yaw)` is
/// `[yaw, pitch, roll]`, and the extrinsic `xyz` one `[roll, pitch, yaw]` (upstream's example).
#[test]
fn test_euler_angles_ordered_round_trips() {
    let (x, y, z) = (x_axis(), y_axis(), z_axis());
    let (roll, pitch, yaw) = (fx(0x40000000), fx(-0x60000000), fx(0x90000000));
    let r = Rotation3AngleTrait::from_euler_angles(roll, pitch, yaw);
    let ([a0, a1, a2], ok) = r.euler_angles_ordered([z, y, x], false);
    assert!(ok);
    let e = Vector3 { x: yaw, y: pitch, z: roll };
    assert!(max_ulp_diff_v3(Vector3 { x: a0, y: a1, z: a2 }, e) <= 64);
    let ([b0, b1, b2], ok) = r.euler_angles_ordered([x, y, z], true);
    assert!(ok);
    let e = Vector3 { x: roll, y: pitch, z: yaw };
    assert!(max_ulp_diff_v3(Vector3 { x: b0, y: b1, z: b2 }, e) <= 64);
    // The identity decomposes into zeros.
    let id: Rotation3<Fixed> = Rotation3Trait::identity();
    let ([c0, c1, c2], ok) = id.euler_angles_ordered([z, y, x], false);
    assert!(ok && c0 == fx(0) && c2 == fx(0) && Real::abs_diff_eq(c1, fx(0), 4));
}

/// At the gimbal lock (`pitch = π/2`) the angles are not observable: the first one (intrinsic)
/// carries the combined rotation, the last is zero — upstream: `[0.2, π/2, 0]` for
/// `from_euler_angles(0.1, π/2, 0.3)`.
#[test]
fn test_euler_angles_ordered_gimbal_lock() {
    let (x, y, z) = (x_axis(), y_axis(), z_axis());
    let r = Rotation3AngleTrait::from_euler_angles(
        fx(429496730), Real::frac_pi_2(), fx(1288490189),
    );
    let ([a0, a1, a2], ok) = r.euler_angles_ordered([z, y, x], false);
    assert!(!ok);
    assert!(Real::abs_diff_eq(a0, fx(858993459), 1024));
    assert!(Real::abs_diff_eq(a1, Real::frac_pi_2(), 1024));
    assert!(a2 == fx(0));
}

/// Upstream asserts `n3 ⟂ n1`: a symmetric (proper Euler) sequence panics upstream, and here.
#[test]
#[should_panic(expected: ('nalgebra: axes not orthogonal',))]
fn test_euler_angles_ordered_symmetric_sequence_panics() {
    let (x, z) = (x_axis(), z_axis());
    let _ = a().euler_angles_ordered([z, x, z], false);
}

#[test]
#[should_panic(expected: ('nalgebra: axes not orthogonal',))]
fn test_euler_angles_ordered_non_orthogonal_axes_panics() {
    let (x, y) = (x_axis(), y_axis());
    let _ = a().euler_angles_ordered([x, x, y], false);
}

// --- closest rotation, interpolation

#[test]
fn test_from_matrix_cases() {
    // A rotation is its own closest rotation; a uniform scaling does not change it.
    // Measured: 29 ulp through the closed form (the normalised squarings of Horn's matrix, then
    // `to_rotation_matrix`), 3 ulp through 64 Müller iterations.
    let r = a();
    assert!(rot_err(Rotation3AngleTrait::from_matrix(r.matrix), r) <= 32);
    let two = int(2);
    assert!(
        rot_err(
            Rotation3AngleTrait::from_matrix(r.matrix * Matrix3Trait::from_diagonal_element(two)),
            r,
        ) <= 32,
    );
    // `max_iter = 0` is the closed form; a bounded iteration from the identity gets close.
    let id = Rotation3Trait::identity();
    assert!(
        Rotation3AngleTrait::from_matrix_eps(
            r.matrix, Real::default_epsilon(), 0, id,
        ) == Rotation3AngleTrait::from_matrix(r.matrix),
    );
    let it = Rotation3AngleTrait::from_matrix_eps(r.matrix, Real::default_epsilon(), 64, id);
    assert!(rot_err(it, r) <= 4);
}

/// The loser: upstream's iteration on matrices follows the same path as the quaternion form (the
/// same rotation after the same number of steps from the identity), at a higher cost per step
/// (`bench_rotation3_from_matrix__alt_matrix_iterate_8`).
#[test]
fn test_from_matrix_alt_matrix_iteration_agrees() {
    let m = super::benches::skewed();
    let id = Rotation3Trait::identity();
    let kept = Rotation3AngleTrait::from_matrix_eps(m, Real::default_epsilon(), 8, id);
    let alt = alt_from_matrix_eps_matrix(m, 8);
    assert!(rot_err(kept, alt) <= 64);
}

#[test]
fn test_slerp_cases() {
    let (x, y) = (a(), b());
    assert!(rot_err(x.slerp(y, Real::zero()), x) <= 8);
    assert!(rot_err(x.slerp(y, Real::one()), y) <= 8);
    // The midpoint is halfway: angle_to(mid) = angle_to(other) / 2.
    let mid = x.slerp(y, Real::HALF);
    assert!(Real::abs_diff_eq(x.angle_to(mid) + x.angle_to(mid), x.angle_to(y), 0x40000));
    // Nearly aligned rotations are rejected with a positive epsilon.
    let near = x * Rotation3AngleTrait::from_scaled_axis(v3t((0x100000, 0, 0)));
    assert!(x.try_slerp(near, Real::HALF, fx(0x800000)) == None);
    assert!(x.try_slerp(near, Real::HALF, Real::zero()).is_some());
}

// --- heterogeneous operators, comparisons, casts, trait impls

#[test]
fn test_mul_div_unit_quaternion_identities() {
    let q = UnitQuaternionTrait::from_rotation_matrix(b());
    let r = a();
    assert!(r.div_unit_quaternion(q) == UnitQuaternionTrait::from_rotation_matrix(r) / q);
    // `(r * q) / q` is `r` again (as a quaternion, up to the sign).
    let back = r.mul_unit_quaternion(q) / q;
    assert!(back.abs_diff_eq(UnitQuaternionTrait::from_rotation_matrix(r), 8));
}

#[test]
fn test_relative_ulps_eq() {
    let r = a();
    let m = r.matrix;
    let drifted = Rotation3 { matrix: Matrix3 { m11: m.m11 + fx(3), ..m } };
    assert!(r.relative_eq(r, 0, Real::zero()) && r.ulps_eq(r, 0, 0));
    assert!(!r.relative_eq(drifted, 2, Real::zero()) && r.relative_eq(drifted, 3, Real::zero()));
    assert!(!r.ulps_eq(drifted, 2, 2) && r.ulps_eq(drifted, 0, 3));
    assert!(r.relative_eq(drifted, 0, fx(0x10000)));
}

#[test]
fn test_cast_is_the_identity_on_fixed() {
    assert!(a().cast::<Fixed>() == a());
}

#[test]
fn test_trait_impls() {
    let id: Rotation3<Fixed> = Rotation3Trait::identity();
    assert!(Default::default() == id);
    assert!(One::one() == id && id.is_one() && !a().is_one() && a().is_non_one());
    let mut r = a();
    let m = r.matrix;
    assert!(r[(0, 0)] == m.m11 && r[(0, 2)] == m.m13 && r[(1, 0)] == m.m21);
    assert!(r[(2, 1)] == m.m32 && r[(2, 2)] == m.m33);
    assert!(a() / a() == a().rotation_to(a()));
    let iso: Isometry3<Fixed> = a().into();
    assert!(iso.rotation == UnitQuaternionTrait::from_rotation_matrix(a()));
    assert!(iso.translation.vector == v3i(0, 0, 0));
    let sim: Similarity3<Fixed> = a().into();
    assert!(sim.isometry == iso && sim.scaling == Real::one());
    let q: UnitQuaternion<Fixed> = a().into();
    assert!(q == UnitQuaternionTrait::from_rotation_matrix(a()));
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_index_out_of_bounds_panics() {
    let mut r = a();
    let _ = r[(3, 0)];
}

#[test]
#[should_panic(expected: ('nalgebra: index out of bounds',))]
fn test_index_column_out_of_bounds_panics() {
    let mut r = a();
    let _ = r[(0, 3)];
}

/// `UnitQuaternionAngleTrait` is used through `slerp`; keep the import meaningful.
#[test]
fn test_slerp_is_the_quaternion_slerp() {
    let (x, y) = (a(), b());
    let q = UnitQuaternionAngleTrait::slerp(
        UnitQuaternionTrait::from_rotation_matrix(x),
        UnitQuaternionTrait::from_rotation_matrix(y),
        Real::HALF,
    );
    assert!(x.slerp(y, Real::HALF) == q.to_rotation_matrix());
}
