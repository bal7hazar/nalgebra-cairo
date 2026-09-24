//! Gas benchmarks of the WP 8.4-P09a completion of `Rotation3` (`bench_rotation3_<op>__<variant>`,
//! net = raw - `baseline` of the group), and the alternative implementations that lost (`alt_*`),
//! kept as evidence together with the tests showing why (AGENTS.md rule 8).
//!
//! The inputs are two rotations built by `from_euler_angles` (`a`, `b`), a unit quaternion and a
//! rotation vector. Expected values are the results of the kernels themselves, all of which are
//! checked against upstream nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::matrix3::Matrix3;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::quaternion::Quaternion;
use nalgebra::geometry::rotation3::{Rotation3, Rotation3AngleTrait, Rotation3Trait};
use nalgebra::geometry::unit_quaternion::UnitQuaternion;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, fx, r3, u3, v3t};
use simba::scalar::Real;

/// `from_euler_angles(0.25, -0.375, 0.5625)`, as raw entries.
fn a() -> Rotation3<Fixed> {
    Rotation3AngleTrait::from_euler_angles(fx(0x40000000), fx(-0x60000000), fx(0x90000000))
}

/// `from_euler_angles(-0.4375, 0.125, 0.15625)`.
fn b() -> Rotation3<Fixed> {
    Rotation3AngleTrait::from_euler_angles(fx(-0x70000000), fx(0x20000000), fx(0x28000000))
}

/// A unit quaternion (the `a` of the `unit_quaternion_mul` oracle).
fn q() -> UnitQuaternion<Fixed> {
    UnitQuaternion {
        quaternion: Quaternion {
            i: fx(-2563574020), j: fx(-2263667719), k: fx(2114881862), w: fx(-1509276477),
        },
    }
}

/// A rotation vector `(0.25, -0.1875, 0.125)`.
fn w() -> Vector3<Fixed> {
    v3t((0x40000000, -0x30000000, 0x20000000))
}

fn axes() -> [Unit<Vector3<Fixed>>; 3] {
    [u3(0, 0, ONE_RAW), u3(0, ONE_RAW, 0), u3(ONE_RAW, 0, 0)]
}

// --- alternative implementations (losers)

/// `angle_to` as upstream writes it: the full product `other · selfᵀ` (27 products, 9 roundings)
/// then `angle` (the trace, a rounded halving and the `acos`), where the kept version forms only
/// the trace, as ONE fused kernel of 9 products.
pub fn alt_angle_to_rotation_to_angle(x: Rotation3<Fixed>, y: Rotation3<Fixed>) -> Fixed {
    x.rotation_to(y).angle()
}

// --- new

#[test]
#[inline(never)]
fn bench_rotation3_new__baseline() {
    let v = black_box(w());
    let e: Rotation3<Fixed> = black_box(Rotation3AngleTrait::from_scaled_axis(w()));
    assert!(v == v && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_new__rodrigues() {
    let v = black_box(w());
    let e: Rotation3<Fixed> = black_box(Rotation3AngleTrait::from_scaled_axis(w()));
    assert!(Rotation3AngleTrait::new(v) == e);
}

// --- powf

#[test]
#[inline(never)]
fn bench_rotation3_powf__baseline() {
    let r = black_box(a());
    let e = black_box(a().powf(Real::HALF));
    assert!(r == r && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_powf__axis_angle() {
    let r = black_box(a());
    let e = black_box(a().powf(Real::HALF));
    assert!(r.powf(Real::HALF) == e);
}

// --- angle_to

#[test]
#[inline(never)]
fn bench_rotation3_angle_to__baseline() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().angle_to(b()));
    assert!(x == x && y == y && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_angle_to__fused_trace() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().angle_to(b()));
    assert!(x.angle_to(y) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_angle_to__alt_rotation_to_angle() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(alt_angle_to_rotation_to_angle(a(), b()));
    assert!(alt_angle_to_rotation_to_angle(x, y) == e);
}

// --- axis_angle

#[test]
#[inline(never)]
fn bench_rotation3_axis_angle__baseline() {
    let r = black_box(a());
    let e = black_box(a().axis_angle());
    assert!(r == r && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_axis_angle__axis_then_angle() {
    let r = black_box(a());
    let e = black_box(a().axis_angle());
    assert!(r.axis_angle() == e);
}

// --- euler_angles_ordered

#[test]
#[inline(never)]
fn bench_rotation3_euler_angles_ordered__baseline() {
    let r = black_box(a());
    let s = black_box(axes());
    let e = black_box(a().euler_angles_ordered(axes(), false));
    assert!(r == r && s == s && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_euler_angles_ordered__shuster_markley() {
    let r = black_box(a());
    let s = black_box(axes());
    let e = black_box(a().euler_angles_ordered(axes(), false));
    assert!(r.euler_angles_ordered(s, false) == e);
}

// --- from_matrix

fn skewed() -> Matrix3<Fixed> {
    r3(
        [
            [4294967296, 429496729, -214748364], [0, 3865470566, 858993459],
            [214748364, -429496729, 5153960755],
        ],
    )
        .matrix
}

#[test]
#[inline(never)]
fn bench_rotation3_from_matrix__baseline() {
    let m = black_box(skewed());
    let e: Rotation3<Fixed> = black_box(Rotation3AngleTrait::from_matrix(skewed()));
    assert!(m == m && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_matrix__closed_form() {
    let m = black_box(skewed());
    let e: Rotation3<Fixed> = black_box(Rotation3AngleTrait::from_matrix(skewed()));
    assert!(Rotation3AngleTrait::from_matrix(m) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_from_matrix__iterate_8() {
    let m = black_box(skewed());
    let id = Rotation3Trait::identity();
    let e: Rotation3<Fixed> = black_box(
        Rotation3AngleTrait::from_matrix_eps(skewed(), Real::default_epsilon(), 8, id),
    );
    assert!(Rotation3AngleTrait::from_matrix_eps(m, Real::default_epsilon(), 8, id) == e);
}

// --- slerp

#[test]
#[inline(never)]
fn bench_rotation3_slerp__baseline() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().slerp(b(), Real::HALF));
    assert!(x == x && y == y && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_slerp__through_quaternions() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().slerp(b(), Real::HALF));
    assert!(x.slerp(y, Real::HALF) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_slerp__try_slerp() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().slerp(b(), Real::HALF));
    assert!(x.try_slerp(y, Real::HALF, Real::zero()) == Some(e));
}

// --- rotation_to, division

#[test]
#[inline(never)]
fn bench_rotation3_rotation_to__baseline() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(x == x && y == y && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_rotation_to__product_with_transpose() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(x.rotation_to(y) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_rotation_to__div() {
    let (x, y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(y / x == e);
}

// --- products with unit quaternions

#[test]
#[inline(never)]
fn bench_rotation3_mul_unit_quaternion__baseline() {
    let (r, u) = (black_box(a()), black_box(q()));
    let e = black_box(a().mul_unit_quaternion(q()));
    assert!(r == r && u == u && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_mul_unit_quaternion__shepperd_then_hamilton() {
    let (r, u) = (black_box(a()), black_box(q()));
    let e = black_box(a().mul_unit_quaternion(q()));
    assert!(r.mul_unit_quaternion(u) == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_mul_unit_quaternion__div() {
    let (r, u) = (black_box(a()), black_box(q()));
    let e = black_box(a().div_unit_quaternion(q()));
    assert!(r.div_unit_quaternion(u) == e);
}

// --- look_at_lh

#[test]
#[inline(never)]
fn bench_rotation3_look_at_lh__baseline() {
    let (d, u) = (black_box(w()), black_box(v3t((0, ONE_RAW, 0))));
    let e = black_box(Rotation3Trait::look_at_lh(w(), v3t((0, ONE_RAW, 0))));
    assert!(d == d && u == u && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_look_at_lh__face_towards_transposed() {
    let (d, u) = (black_box(w()), black_box(v3t((0, ONE_RAW, 0))));
    let e = black_box(Rotation3Trait::look_at_lh(w(), v3t((0, ONE_RAW, 0))));
    assert!(Rotation3Trait::look_at_lh(d, u) == e);
}

// --- comparison, index

#[test]
#[inline(never)]
fn bench_rotation3_relative_eq__baseline() {
    let (x, y) = (black_box(a()), black_box(a()));
    assert!(x == y);
}

#[test]
#[inline(never)]
fn bench_rotation3_relative_eq__component_wise() {
    let (x, y) = (black_box(a()), black_box(a()));
    assert!(x.relative_eq(y, 1, Real::zero()));
}

#[test]
#[inline(never)]
fn bench_rotation3_relative_eq__ulps_eq() {
    let (x, y) = (black_box(a()), black_box(a()));
    assert!(x.ulps_eq(y, 1, 4));
}

#[test]
#[inline(never)]
fn bench_rotation3_index__baseline() {
    let r = black_box(a());
    let e = black_box(a().matrix.m23);
    assert!(r == r && e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_index__match() {
    let mut r = black_box(a());
    let e = black_box(a().matrix.m23);
    assert!(r[(1, 2)] == e);
}
