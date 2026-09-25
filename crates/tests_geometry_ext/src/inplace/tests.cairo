//! WP 8.4-R: the in-place forms of the points (`map`, `apply`), the translations (`inverse_mut`,
//! `*=`, `/=`) and the rotation matrices (`transpose_mut`, `Rotation2 *= UnitComplex`), each
//! checked bit for bit against its by-value form. Everything but the rotation products is an exact
//! addition or negation, so the expected values are exact.

use fixed::Fixed;
use nalgebra::geometry::point1::{Point1, Point1Trait};
use nalgebra::geometry::point4::{Point4, Point4Trait};
use nalgebra::geometry::point5::{Point5, Point5Trait};
use nalgebra::geometry::point6::{Point6, Point6Trait};
use nalgebra::geometry::point::Point2ExtTrait;
use nalgebra::geometry::point::Point3ExtTrait;
use nalgebra::geometry::rotation2::{Rotation2, Rotation2Trait};
use nalgebra::geometry::rotation3::Rotation3Trait;
use nalgebra::geometry::translation1::Translation1Trait;
use nalgebra::geometry::translation2::Translation2Trait;
use nalgebra::geometry::translation3::Translation3Trait;
use nalgebra::geometry::translation4::Translation4Trait;
use nalgebra::geometry::translation5::Translation5Trait;
use nalgebra::geometry::translation6::Translation6Trait;
use nalgebra_tests_utils::{fx, int, p2, p3, r2, r3, t2, t3, uc};

// --- Point::map / Point::apply

#[test]
fn test_point1_map_apply() {
    let p: Point1<Fixed> = Point1Trait::new(int(3));
    let k = int(2);
    assert!(p.map(|c: Fixed| c * k) == Point1Trait::new(int(6)));
    let raw: Point1<i64> = p.map(|c: Fixed| c.raw);
    assert!(raw.x == 3 * 0x100000000);
    let mut q = p;
    q.apply(|c: Fixed| c * k);
    assert!(q == p.map(|c: Fixed| c * k));
}

#[test]
fn test_point2_map_apply() {
    let p = p2(0x180000000, -0x80000001);
    let k = fx(0x2AAAAAAB);
    let by_value = p.map(|c: Fixed| c * k);
    assert!(by_value.x == p.x * k && by_value.y == p.y * k);
    let raw = p.map(|c: Fixed| c.raw);
    assert!(raw.x == 0x180000000 && raw.y == -0x80000001);
    let mut q = p;
    q.apply(|c: Fixed| c * k);
    assert!(q == by_value);
}

#[test]
fn test_point3_map_apply() {
    let p = p3(0x100000000, -0x200000000, 0x123456789);
    let k = fx(-0x3FFFFFFF);
    let by_value = p.map(|c: Fixed| c * k);
    assert!(by_value.x == p.x * k && by_value.y == p.y * k && by_value.z == p.z * k);
    let mut q = p;
    q.apply(|c: Fixed| -c);
    assert!(q == p3(-0x100000000, 0x200000000, -0x123456789));
    let mut q = p;
    q.apply(|c: Fixed| c * k);
    assert!(q == by_value);
}

#[test]
fn test_point4_map_apply() {
    let p: Point4<Fixed> = Point4Trait::new(int(1), int(-2), int(3), int(4));
    let mapped = p.map(|c: Fixed| c + c);
    assert!(mapped == Point4Trait::new(int(2), int(-4), int(6), int(8)));
    let raw: Point4<i64> = p.map(|c: Fixed| c.raw);
    assert!(raw.w == 4 * 0x100000000 && raw.y == -2 * 0x100000000);
    let mut q = p;
    q.apply(|c: Fixed| c + c);
    assert!(q == mapped);
}

#[test]
fn test_point5_map_apply() {
    let p: Point5<Fixed> = Point5Trait::new(int(1), int(-2), int(3), int(4), int(-5));
    let mapped = p.map(|c: Fixed| c + int(1));
    assert!(mapped == Point5Trait::new(int(2), int(-1), int(4), int(5), int(-4)));
    let mut q = p;
    q.apply(|c: Fixed| c + int(1));
    assert!(q == mapped);
}

#[test]
fn test_point6_map_apply() {
    let p: Point6<Fixed> = Point6Trait::new(int(1), int(-2), int(3), int(4), int(-5), int(6));
    let mapped = p.map(|c: Fixed| c * c);
    assert!(mapped == Point6Trait::new(int(1), int(4), int(9), int(16), int(25), int(36)));
    let raw: Point6<i64> = p.map(|c: Fixed| c.raw);
    assert!(raw.b == 6 * 0x100000000 && raw.a == -5 * 0x100000000);
    let mut q = p;
    q.apply(|c: Fixed| c * c);
    assert!(q == mapped);
}

// --- Translation: inverse_mut, *=, /=

#[test]
fn test_translation_inverse_mut_matches_inverse() {
    let mut a = Translation1Trait::new(int(3));
    a.inverse_mut();
    assert!(a == Translation1Trait::new(int(3)).inverse());
    assert!(a == Translation1Trait::new(int(-3)));

    let t = t2(0x100000001, -0x7FFFFFFF);
    let mut b = t;
    b.inverse_mut();
    assert!(b == t.inverse() && b == t2(-0x100000001, 0x7FFFFFFF));

    let t = t3(0x100000001, -0x7FFFFFFF, 0x2468ACE);
    let mut c = t;
    c.inverse_mut();
    assert!(c == t.inverse() && c == t3(-0x100000001, 0x7FFFFFFF, -0x2468ACE));

    let t4 = Translation4Trait::new(int(1), int(-2), int(3), int(4));
    let mut d = t4;
    d.inverse_mut();
    assert!(d == t4.inverse() && d == Translation4Trait::new(int(-1), int(2), int(-3), int(-4)));

    let t5 = Translation5Trait::new(int(1), int(-2), int(3), int(4), int(5));
    let mut e = t5;
    e.inverse_mut();
    assert!(e == t5.inverse());

    let t6 = Translation6Trait::new(int(1), int(-2), int(3), int(4), int(5), int(-6));
    let mut f = t6;
    f.inverse_mut();
    assert!(f == t6.inverse());
}

#[test]
fn test_translation_mul_div_assign_match_operators() {
    let (a, b) = (t3(0x100000001, -0x7FFFFFFF, 5), t3(-3, 0x300000000, 0x2468ACE));
    let mut m = a;
    m *= b;
    assert!(m == a * b);
    let mut d = a;
    d /= b;
    assert!(d == a / b);

    let (a, b) = (t2(7, -9), t2(0x100000000, 0x100000001));
    let mut m = a;
    m *= b;
    assert!(m == a * b);
    let mut d = a;
    d /= b;
    assert!(d == a / b);

    let (a, b) = (Translation1Trait::new(int(2)), Translation1Trait::new(fx(5)));
    let mut m = a;
    m *= b;
    assert!(m == a * b);
    let mut d = a;
    d /= b;
    assert!(d == a / b);

    let a = Translation4Trait::new(int(1), int(-2), int(3), int(4));
    let b = Translation4Trait::new(int(4), int(3), int(-2), int(1));
    let mut m = a;
    m *= b;
    assert!(m == a * b);
    let mut d = a;
    d /= b;
    assert!(d == a / b && d == Translation4Trait::new(int(-3), int(-5), int(5), int(3)));

    let a = Translation5Trait::new(int(1), int(-2), int(3), int(4), int(5));
    let mut m = a;
    m *= a;
    assert!(m == a * a);
    let mut d = a;
    d /= a;
    assert!(d == Translation5Trait::new(int(0), int(0), int(0), int(0), int(0)));

    let a = Translation6Trait::new(int(1), int(-2), int(3), int(4), int(5), int(-6));
    let mut m = a;
    m *= a;
    assert!(m == a * a);
    let mut d = a;
    d /= a;
    assert!(d == Translation6Trait::new(int(0), int(0), int(0), int(0), int(0), int(0)));
}

// --- Rotation: transpose_mut, Rotation2 *= UnitComplex

#[test]
fn test_rotation_transpose_mut_matches_transpose() {
    let r = r2([[0x100000000, -0x20000000], [0x20000000, 0x100000000]]);
    let mut m = r;
    m.transpose_mut();
    assert!(m == r.transpose());
    assert!(m.matrix.m12 == r.matrix.m21 && m.matrix.m21 == r.matrix.m12);

    let r = r3(
        [
            [0x100000000, 0x1234, 0x5678], [0x9ABC, 0x100000000, 0xDEF0], [0x1111, 0x2222, 0x3333],
        ],
    );
    let mut m = r;
    m.transpose_mut();
    assert!(m == r.transpose());
    assert!(m.matrix.m12 == r.matrix.m21 && m.matrix.m31 == r.matrix.m13);
    m.transpose_mut();
    assert!(m == r);
}

#[test]
fn test_rotation2_mul_div_assign_unit_complex() {
    // A rotation by 3/5, 4/5 turned by (0.8, 0.6) and by (0.6, -0.8), off the exact grid.
    let r = r2([[0x99999999, -0x66666666], [0x66666666, 0x99999999]]);
    let cs = [uc(0xCCCCCCCD, 0x99999999), uc(0x99999999, -0xCCCCCCCD), uc(0x100000000, 0)];
    for c in cs.span() {
        let c = *c;
        let rc: Rotation2<Fixed> = Rotation2 {
            matrix: nalgebra::base::matrix2::Matrix2 {
                m11: c.re, m21: c.im, m12: -c.im, m22: c.re,
            },
        };
        let mut m = r;
        m *= c;
        assert!(m == r * rc);
        let mut d = r;
        d /= c;
        assert!(d == r / rc);
    }
}
