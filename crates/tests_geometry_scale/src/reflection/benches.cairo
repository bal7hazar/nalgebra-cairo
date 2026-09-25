//! Gas benchmarks of `Reflection2` / `Reflection3` / `Reflection6` (WP 8.4-P10),
//! `bench_reflection<D>_<op>__<variant>`, net = raw - `baseline` of the group, and the alternative
//! that lost (`alt_*`), kept as evidence together with the test showing that both agree (AGENTS.md
//! rule 8).
//!
//! `reflect` accumulates the dot product with the axis EXACTLY (`wide_add_prod`), subtracts the
//! bias and multiplies by -2 in the accumulator (`wide_mul_scalar`): one rounding for the factor;
//! `alt_sum_prod` is the fused `sum_prod3` kernel followed by a subtraction and an exact doubling
//! (one rounding for the dot product, exact factor).
//!
//! Inputs: the reflection about the plane `x = 3` (axis `(1, 0, 0)`), so that every result is
//! exact.

use fixed::Fixed;
use nalgebra::base::matrix3x2::Matrix3x2;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::reflection2::{Reflection2, Reflection2Columns, Reflection2Trait};
use nalgebra::geometry::reflection3::{
    Reflection3, Reflection3Columns, Reflection3Rows, Reflection3Trait,
};
use nalgebra::geometry::reflection6::{Reflection6, Reflection6Columns, Reflection6Trait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int, ulp_diff};
use simba::scalar::Real;
use crate::common::{err_mat2x3, mat2x3, pt3, vec2, vec3, vec6};

const ONE: i64 = 0x100000000;

fn r2() -> Reflection2<Fixed> {
    Reflection2Trait::new(Unit { value: vec2((ONE, 0)) }, int(3))
}

fn r3() -> Reflection3<Fixed> {
    Reflection3Trait::new(Unit { value: vec3((ONE, 0, 0)) }, int(3))
}

fn r6() -> Reflection6<Fixed> {
    Reflection6Trait::new(Unit { value: vec6((ONE, 0, 0, 0, 0, 0)) }, int(3))
}

/// `reflect` of a vector with the fused `sum_prod3` dot product and an exact factor.
fn alt_reflect_sum_prod(r: Reflection3<Fixed>, v: Vector3<Fixed>) -> Vector3<Fixed> {
    let a = r.axis();
    let d = Real::sum_prod3(a.x, v.x, a.y, v.y, a.z, v.z) - r.bias();
    let f = -(d + d);
    Vector3 {
        x: Real::mul_add(f, a.x, v.x), y: Real::mul_add(f, a.y, v.y), z: Real::mul_add(f, a.z, v.z),
    }
}

#[test]
fn test_reflect_alt_sum_prod_agrees_within_an_ulp() {
    let axis = nalgebra::base::unit::UnitTrait::new_normalize(vec3((3 * ONE, -4 * ONE, 12 * ONE)));
    let r = Reflection3Trait::new(axis, fx(0x12345678));
    let v = vec3((0x123456789, -0x3456789abc, 0x56789abcd));
    let mut got = v;
    r.reflect(ref got);
    let alt = alt_reflect_sum_prod(r, v);
    assert!(
        ulp_diff(got.x, alt.x) <= 2 && ulp_diff(got.y, alt.y) <= 2 && ulp_diff(got.z, alt.z) <= 2,
    );
    // Exact on the benchmark input.
    assert!(alt_reflect_sum_prod(r3(), vec3((7 * ONE, ONE, ONE))) == vec3((-ONE, ONE, ONE)));
}

// --- reflect on a vector

#[test]
#[inline(never)]
fn bench_reflection3_reflect__baseline() {
    let _r = black_box(r3());
    let _v = black_box(vec3((7 * ONE, ONE, ONE)));
    let e = black_box(vec3((-ONE, ONE, ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_reflection3_reflect__vector3() {
    let r = black_box(r3());
    let mut v = black_box(vec3((7 * ONE, ONE, ONE)));
    let e = black_box(vec3((-ONE, ONE, ONE)));
    r.reflect(ref v);
    assert!(v == e);
}

#[test]
#[inline(never)]
fn bench_reflection3_reflect__alt_sum_prod() {
    let r = black_box(r3());
    let v = black_box(vec3((7 * ONE, ONE, ONE)));
    let e = black_box(vec3((-ONE, ONE, ONE)));
    assert!(alt_reflect_sum_prod(r, v) == e);
}

#[test]
#[inline(never)]
fn bench_reflection3_reflect__matrix3x2() {
    let r = black_box(r3());
    let mut m = black_box(
        Matrix3x2 { m11: int(7), m21: int(1), m31: int(1), m12: int(5), m22: int(2), m32: int(2) },
    );
    let e = black_box(
        Matrix3x2 { m11: int(-1), m21: int(1), m31: int(1), m12: int(1), m22: int(2), m32: int(2) },
    );
    r.reflect(ref m);
    assert!(m == e);
}

#[test]
#[inline(never)]
fn bench_reflection3_reflect_with_sign__vector3() {
    let r = black_box(r3());
    let mut v = black_box(vec3((7 * ONE, ONE, ONE)));
    let e = black_box(vec3((ONE, -ONE, -ONE)));
    r.reflect_with_sign(ref v, int(-1));
    assert!(v == e);
}

// --- reflect_rows

#[test]
#[inline(never)]
fn bench_reflection3_reflect_rows__baseline() {
    let _r = black_box(r3());
    let _m = black_box(mat2x3([[7 * ONE, ONE, ONE], [5 * ONE, 2 * ONE, 2 * ONE]]));
    let e = black_box(mat2x3([[-ONE, ONE, ONE], [ONE, 2 * ONE, 2 * ONE]]));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_reflection3_reflect_rows__matrix2x3() {
    let r = black_box(r3());
    let mut m = black_box(mat2x3([[7 * ONE, ONE, ONE], [5 * ONE, 2 * ONE, 2 * ONE]]));
    let e = black_box(mat2x3([[-ONE, ONE, ONE], [ONE, 2 * ONE, 2 * ONE]]));
    let mut work = black_box(vec2((0, 0)));
    r.reflect_rows(ref m, ref work);
    assert!(err_mat2x3(m, e) == 0 && work == Vector2 { x: int(4), y: int(2) });
}

// --- other sizes

#[test]
#[inline(never)]
fn bench_reflection2_reflect__baseline() {
    let _r = black_box(r2());
    let _v = black_box(vec2((7 * ONE, ONE)));
    let e = black_box(vec2((-ONE, ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_reflection2_reflect__vector2() {
    let r = black_box(r2());
    let mut v = black_box(vec2((7 * ONE, ONE)));
    let e = black_box(vec2((-ONE, ONE)));
    r.reflect(ref v);
    assert!(v == e);
}

#[test]
#[inline(never)]
fn bench_reflection6_reflect__baseline() {
    let _r = black_box(r6());
    let _v = black_box(vec6((7 * ONE, ONE, ONE, ONE, ONE, ONE)));
    let e = black_box(vec6((-ONE, ONE, ONE, ONE, ONE, ONE)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_reflection6_reflect__vector6() {
    let r = black_box(r6());
    let mut v = black_box(vec6((7 * ONE, ONE, ONE, ONE, ONE, ONE)));
    let e = black_box(vec6((-ONE, ONE, ONE, ONE, ONE, ONE)));
    r.reflect(ref v);
    assert!(v == e);
}

// --- construction

#[test]
#[inline(never)]
fn bench_reflection3_new_containing_point__baseline() {
    let _a = black_box(Unit { value: vec3((ONE, 0, 0)) });
    let _p = black_box(pt3((5 * ONE, 2 * ONE, ONE)));
    let e = black_box(int(5));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_reflection3_new_containing_point__exact_dot() {
    let a = black_box(Unit { value: vec3((ONE, 0, 0)) });
    let p = black_box(pt3((5 * ONE, 2 * ONE, ONE)));
    let e = black_box(int(5));
    assert!(Reflection3Trait::new_containing_point(a, p).bias() == e);
}
