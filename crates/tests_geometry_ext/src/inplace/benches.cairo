//! Gas benchmarks of the WP 8.4-R in-place forms that have a body of their own
//! (`bench_<group>__<variant>`, net = raw - `baseline` of the group): `Rotation2 *= UnitComplex`
//! against the by-value product with the rotation matrix, and `Point3::apply` against
//! `Point3::map`. The other in-place forms of the package (`inverse_mut`, `transpose_mut`,
//! `Translation *= t`) are `self = self.op(..)` and cost what the by-value form costs.

use fixed::Fixed;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::geometry::point::Point3ExtTrait;
use nalgebra::geometry::rotation2::{Rotation2, Rotation2AngleTrait};
use nalgebra::geometry::unit_complex::UnitComplex;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, p3};

fn a() -> Rotation2<Fixed> {
    Rotation2AngleTrait::new(fx(0xc0000000))
}

fn c() -> UnitComplex<Fixed> {
    Rotation2AngleTrait::new(fx(-0x200000000)).into()
}

// --- Rotation2 *= UnitComplex

#[test]
#[inline(never)]
fn bench_rotation2_mul_assign_unit_complex__baseline() {
    let (_a, _c) = (black_box(a()), black_box(c()));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_rotation2_mul_assign_unit_complex__mul_assign() {
    let (mut a, c) = (black_box(a()), black_box(c()));
    a *= c;
    assert!(a.matrix.m11 == a.matrix.m22);
}

#[test]
#[inline(never)]
fn bench_rotation2_mul_assign_unit_complex__alt_by_value() {
    let (a, c) = (black_box(a()), black_box(c()));
    let rc = Rotation2 { matrix: Matrix2 { m11: c.re, m21: c.im, m12: -c.im, m22: c.re } };
    let r = a * rc;
    assert!(r.matrix.m11 == r.matrix.m22);
}

// --- Point3::apply

#[test]
#[inline(never)]
fn bench_point3_apply__baseline() {
    let _p = black_box(p3(0x100000000, -0x200000000, 0x300000000));
    assert!(black_box(true));
}

#[test]
#[inline(never)]
fn bench_point3_apply__apply() {
    let mut p = black_box(p3(0x100000000, -0x200000000, 0x300000000));
    p.apply(|c: Fixed| -c);
    assert!(p.x == fx(-0x100000000));
}

#[test]
#[inline(never)]
fn bench_point3_apply__alt_map() {
    let p = black_box(p3(0x100000000, -0x200000000, 0x300000000));
    assert!(p.map(|c: Fixed| -c).x == fx(-0x100000000));
}
