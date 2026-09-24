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

/// Upstream's `Rotation3::from_matrix_eps` literally, on MATRICES: `ω = Σ_c r_c × m_c / (|Σ_c
/// r_c ·
/// m_c| + ε)`, `R ← from_axis_angle(ω/|ω|, |ω|) · R` (Rodrigues' formula, then a 27-product
/// composition) for `iters` iterations (no stationary-point perturbation: the benchmark and the
/// test stop before convergence). The kept version runs the same iteration on a unit quaternion
/// (one `to_rotation_matrix`, a half-angle `sin_cos` and a 16-product composition per step).
pub fn alt_from_matrix_eps_matrix(m: Matrix3<Fixed>, iters: usize) -> Rotation3<Fixed> {
    let mut r: Rotation3<Fixed> = Rotation3Trait::identity();
    let mut k: usize = 0;
    while k < iters {
        k += 1;
        let a = r.matrix;
        let x = Real::sum_prod3(a.m21, m.m31, a.m22, m.m32, a.m23, m.m33)
            - Real::sum_prod3(a.m31, m.m21, a.m32, m.m22, a.m33, m.m23);
        let y = Real::sum_prod3(a.m31, m.m11, a.m32, m.m12, a.m33, m.m13)
            - Real::sum_prod3(a.m11, m.m31, a.m12, m.m32, a.m13, m.m33);
        let z = Real::sum_prod3(a.m11, m.m21, a.m12, m.m22, a.m13, m.m23)
            - Real::sum_prod3(a.m21, m.m11, a.m22, m.m12, a.m23, m.m13);
        let d = Real::sum_prod3(a.m11, m.m11, a.m21, m.m21, a.m31, m.m31)
            + Real::sum_prod3(a.m12, m.m12, a.m22, m.m22, a.m32, m.m32)
            + Real::sum_prod3(a.m13, m.m13, a.m23, m.m23, a.m33, m.m33);
        let (x, y, z) = Real::div3(x, y, z, Real::abs(d) + Real::default_epsilon());
        let n = Real::norm3(x, y, z);
        if n <= Real::default_epsilon() {
            break;
        }
        let (ux, uy, uz) = Real::div3(x, y, z, n);
        let axis = Unit { value: Vector3 { x: ux, y: uy, z: uz } };
        r = Rotation3AngleTrait::from_axis_angle(axis, n) * r;
    }
    r
}

// --- new

#[test]
#[inline(never)]
fn bench_rotation3_new__baseline() {
    let _v = black_box(w());
    let e: Rotation3<Fixed> = black_box(Rotation3AngleTrait::from_scaled_axis(w()));
    assert!(e == e);
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
    let _r = black_box(a());
    let e = black_box(a().powf(Real::HALF));
    assert!(e == e);
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
    let (_x, _y) = (black_box(a()), black_box(b()));
    let e = black_box(a().angle_to(b()));
    assert!(e == e);
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
    let _r = black_box(a());
    let e = black_box(a().axis_angle());
    assert!(e == e);
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
    let _r = black_box(a());
    let _s = black_box(axes());
    let e = black_box(a().euler_angles_ordered(axes(), false));
    assert!(e == e);
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

pub fn skewed() -> Matrix3<Fixed> {
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
    let _m = black_box(skewed());
    let e: Rotation3<Fixed> = black_box(Rotation3AngleTrait::from_matrix(skewed()));
    assert!(e == e);
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

#[test]
#[inline(never)]
fn bench_rotation3_from_matrix__alt_matrix_iterate_8() {
    let m = black_box(skewed());
    let e: Rotation3<Fixed> = black_box(alt_from_matrix_eps_matrix(skewed(), 8));
    assert!(alt_from_matrix_eps_matrix(m, 8) == e);
}

// --- slerp

#[test]
#[inline(never)]
fn bench_rotation3_slerp__baseline() {
    let (_x, _y) = (black_box(a()), black_box(b()));
    let e = black_box(a().slerp(b(), Real::HALF));
    assert!(e == e);
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
    let (_x, _y) = (black_box(a()), black_box(b()));
    let e = black_box(a().rotation_to(b()));
    assert!(e == e);
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
    let (_r, _u) = (black_box(a()), black_box(q()));
    let e = black_box(a().mul_unit_quaternion(q()));
    assert!(e == e);
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
    let (_d, _u) = (black_box(w()), black_box(v3t((0, ONE_RAW, 0))));
    let e = black_box(Rotation3Trait::look_at_lh(w(), v3t((0, ONE_RAW, 0))));
    assert!(e == e);
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
    let (_x, _y) = (black_box(a()), black_box(a()));
    assert!(black_box(true));
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
    let _r = black_box(a());
    let e = black_box(a().matrix.m23);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_index__match() {
    let mut r = black_box(a());
    let e = black_box(a().matrix.m23);
    assert!(r[(1, 2)] == e);
}
