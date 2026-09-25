//! Gas benchmarks of `UnitDualQuaternion` (WP 8.4-P12) (`bench_<type>_<op>__<variant>`, net =
//! raw - `baseline` of the group), and the alternative implementations that lost (`alt_*`,
//! AGENTS.md rule 8):
//!
//! - `from_parts` / `translation`: upstream's literal product, then the halving / doubling (two
//!   roundings) against the halving / doubling folded into the fused accumulation;
//! - `inverse`: the quaternion conjugate of the dual part (free, and equal for an EXACTLY unit
//!   dual quaternion) against upstream's `(-real*) · dual · real*` — cheaper but off by up to
//!   `|dual| · (|real|² - 1)` on the rounded unit dual quaternions of fixed point;
//! - `transform_point` / `inverse_transform_point`: the fused isometry kernels
//!   (`to_isometry().transform_point(p)`) against upstream's literal sandwich — cheaper but off
//!   by up to `|p| · (|real|² - 1)`;
//! - `mul_translation` / `mul_unit_quaternion`: the general product by `from_parts(t, identity)` /
//!   `from_rotation(q)` against the reduced kernels.
//!
//! The fixtures are the `Isometry3` bench poses as unit dual quaternions (hardcoded). Expected
//! values are the results of the kernels themselves, all of which are checked against upstream
//! nalgebra in `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::dual_quaternion::{DualQuaternion, DualQuaternionTrait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::quaternion::QuaternionTrait;
use nalgebra::geometry::translation3::Translation3;
use nalgebra::geometry::unit_dual_quaternion::{
    UnitDualQuaternion, UnitDualQuaternionAngleTrait, UnitDualQuaternionTrait,
};
use nalgebra::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, fx, max_ulp_diff_v3, p3, p3t, t3, uq, v3};
use crate::common::{dq_err, udqt};
use crate::oracle;

fn ra() -> UnitQuaternion<Fixed> {
    uq(4234293283, 534340439, -400755330, 267170219)
}

fn rb() -> UnitQuaternion<Fixed> {
    uq(3689020097, -1022754606, 767065954, 1789820560)
}

/// `from_parts((1.5, -2.25, 3.75), ra)`.
fn a() -> UnitDualQuaternion<Fixed> {
    udqt(
        (
            (4234293283, 534340439, -400755330, 267170219),
            (-1352549237, 3626569709, -3962069285, 8239866402),
        ),
    )
}

/// `from_parts((-0.75, 0.5, 1.25), rb)`.
fn b() -> UnitDualQuaternion<Fixed> {
    udqt(
        (
            (3689020097, -1022754606, 767065954, 1789820560),
            (-1693937316, -1415343618, 954216105, 2273676479),
        ),
    )
}

fn pt() -> Point3<Fixed> {
    p3(0x280000000, -0x100000000, 0x40000000)
}

fn tr() -> Translation3<Fixed> {
    t3(ONE_RAW, -2 * ONE_RAW, ONE_RAW / 2)
}

// --- alternative implementations (losers)

/// The LOSER of `from_parts`: upstream's literal `(0, t) · r` rounded, then halved (nearest).
#[inline(always)]
fn alt_from_parts(t: Translation3<Fixed>, r: UnitQuaternion<Fixed>) -> UnitDualQuaternion<Fixed> {
    UnitDualQuaternion {
        dual_quaternion: DualQuaternion {
            real: r.quaternion,
            dual: (QuaternionTrait::from_parts(fx(0), t.vector) * r.quaternion).half(),
        },
    }
}

/// The LOSER of `translation`: upstream's literal `dual · real*` rounded, then doubled.
#[inline(always)]
fn alt_translation(x: UnitDualQuaternion<Fixed>) -> Vector3<Fixed> {
    let v = (x.dual_quaternion.dual * x.dual_quaternion.real.conjugate()).vector();
    Vector3 { x: v.x + v.x, y: v.y + v.y, z: v.z + v.z }
}

/// The LOSER of `inverse`: the conjugate of both parts (exact for an exactly unit input only).
#[inline(always)]
fn alt_inverse(x: UnitDualQuaternion<Fixed>) -> UnitDualQuaternion<Fixed> {
    UnitDualQuaternion { dual_quaternion: x.dual_quaternion.conjugate() }
}

/// The LOSER of `transform_point`: the isometry's unit-quaternion sandwich plus `translation()`.
#[inline(always)]
fn alt_transform_point(x: UnitDualQuaternion<Fixed>, p: Point3<Fixed>) -> Point3<Fixed> {
    x.to_isometry().transform_point(p)
}

/// The LOSER of `inverse_transform_point`: the isometry's inverse transform.
#[inline(always)]
fn alt_inverse_transform_point(x: UnitDualQuaternion<Fixed>, p: Point3<Fixed>) -> Point3<Fixed> {
    x.to_isometry().inverse_transform_point(p)
}

#[test]
fn test_from_parts_translation_alt_agree() {
    let (f, l) = (
        UnitDualQuaternionTrait::from_parts(t3(0x180000000, -0x240000000, 0x3c0000000), ra()),
        alt_from_parts(t3(0x180000000, -0x240000000, 0x3c0000000), ra()),
    );
    assert!(f.abs_diff_eq(l, 1) && f != l);
    assert!(max_ulp_diff_v3(a().translation().vector, alt_translation(a())) <= 2);
}

#[test]
fn test_mul_translation_mul_unit_quaternion_alt_agree() {
    let ft: UnitDualQuaternion<Fixed> = tr().into();
    assert!(a().mul_translation(tr()).abs_diff_eq(a() * ft, 1));
    assert!(a().mul_unit_quaternion(rb()) == a() * UnitDualQuaternionTrait::from_rotation(rb()));
}

/// The conjugate shortcut agrees on a small transform, but misses upstream's inverse by far more
/// than the oracle tolerance once the translation is large (`|dual| · (|real|² - 1)`).
#[test]
fn test_inverse_alt_conjugate_loses_on_large_translations() {
    assert!(alt_inverse(a()).abs_diff_eq(a().inverse(), 4));
    let mut worst_alt = 0;
    let mut cases = oracle::unit_dual_quaternion_inverse_cases();
    while let Some(case) = cases.pop_front() {
        let (x, e, tol) = *case;
        assert!(dq_err(udqt(x).inverse().dual_quaternion, e) <= tol.into());
        worst_alt = core::cmp::max(worst_alt, dq_err(alt_inverse(udqt(x)).dual_quaternion, e));
    }
    assert!(worst_alt > 1000);
}

/// Likewise for the point transforms (`|p| · (|real|² - 1)`).
#[test]
fn test_transform_point_alt_to_isometry_loses_on_large_points() {
    assert!(
        max_ulp_diff_v3(
            alt_transform_point(a(), pt()).coords(), a().transform_point(pt()).coords(),
        ) <= 4,
    );
    let (mut worst_alt, mut worst_alt_inv) = (0, 0);
    let mut cases = oracle::unit_dual_quaternion_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, p, e, _tol) = *case;
        let err = max_ulp_diff_v3(alt_transform_point(udqt(x), p3t(p)).coords(), p3t(e).coords());
        worst_alt = core::cmp::max(worst_alt, err);
    }
    let mut cases = oracle::unit_dual_quaternion_inverse_transform_point_cases();
    while let Some(case) = cases.pop_front() {
        let (x, p, e, _tol) = *case;
        let got = alt_inverse_transform_point(udqt(x), p3t(p));
        worst_alt_inv =
            core::cmp::max(worst_alt_inv, max_ulp_diff_v3(got.coords(), p3t(e).coords()));
    }
    assert!(worst_alt > 1000 && worst_alt_inv > 1000);
}

// --- benchmarks: construction and parts

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_from_parts__baseline() {
    let _t: Translation3<Fixed> = black_box(t3(0x180000000, -0x240000000, 0x3c0000000));
    let _r: UnitQuaternion<Fixed> = black_box(ra());
    let e: UnitDualQuaternion<Fixed> = black_box(a());
    assert!(e == e);
}

/// The halving folded into the accumulation (winner).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_from_parts__fused() {
    let t: Translation3<Fixed> = black_box(t3(0x180000000, -0x240000000, 0x3c0000000));
    let r: UnitQuaternion<Fixed> = black_box(ra());
    let e: UnitDualQuaternion<Fixed> = black_box(a());
    assert!(UnitDualQuaternionTrait::from_parts(t, r) == e);
}

/// Upstream's literal product then `half` (loser).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_from_parts__alt_upstream() {
    let t: Translation3<Fixed> = black_box(t3(0x180000000, -0x240000000, 0x3c0000000));
    let r: UnitQuaternion<Fixed> = black_box(ra());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, 534340439, -400755330, 267170219),
                (-1352549236, 3626569710, -3962069284, 8239866402),
            ),
        ),
    );
    assert!(alt_from_parts(t, r) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_translation__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: Vector3<Fixed> = black_box(v3(6442450942, -9663676417, 16106127358));
    assert!(e == e);
}

/// The doubling folded into the accumulation (winner).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_translation__fused() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: Vector3<Fixed> = black_box(v3(6442450942, -9663676417, 16106127358));
    assert!(x.translation().vector == e);
}

/// Upstream's literal product then the doubling (loser).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_translation__alt_upstream() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: Vector3<Fixed> = black_box(v3(6442450942, -9663676418, 16106127358));
    assert!(alt_translation(x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_to_isometry__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: Isometry3<Fixed> = black_box(
        Isometry3Trait::from_parts(t3(6442450942, -9663676417, 16106127358), ra()),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_to_isometry__to_isometry() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: Isometry3<Fixed> = black_box(
        Isometry3Trait::from_parts(t3(6442450942, -9663676417, 16106127358), ra()),
    );
    assert!(x.to_isometry() == e);
}

// --- benchmarks: algebra

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _y: UnitDualQuaternion<Fixed> = black_box(b());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-4570607052, -1563305228, -6390197141, 8340766433),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul__mul() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let y: UnitDualQuaternion<Fixed> = black_box(b());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-4570607052, -1563305228, -6390197141, 8340766433),
            ),
        ),
    );
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_inverse__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, -534340439, 400755330, -267170219),
                (-1352549237, -3626569710, 3962069283, -8239866402),
            ),
        ),
    );
    assert!(e == e);
}

/// Upstream's `(-real*) · dual · real*` (winner: fidelity).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_inverse__literal() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, -534340439, 400755330, -267170219),
                (-1352549237, -3626569710, 3962069283, -8239866402),
            ),
        ),
    );
    assert!(x.inverse() == e);
}

/// The conjugate shortcut (loser: off by `|dual| · (|real|² - 1)`).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_inverse__alt_conjugate() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, -534340439, 400755330, -267170219),
                (-1352549237, -3626569709, 3962069285, -8239866402),
            ),
        ),
    );
    assert!(alt_inverse(x) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_div__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _y: UnitDualQuaternion<Fixed> = black_box(b());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3549427458, 1681980882, -814151394, -1535059155),
                (-1092862717, 7371661133, -99863367, 5603210823),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_div__div() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let y: UnitDualQuaternion<Fixed> = black_box(b());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3549427458, 1681980882, -814151394, -1535059155),
                (-1092862717, 7371661133, -99863367, 5603210823),
            ),
        ),
    );
    assert!(x / y == e);
}

// --- benchmarks: transforms

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_transform_point__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3(17355384373, -13000621593, 18265871982));
    assert!(e == e);
}

/// Upstream's literal sandwich (winner: fidelity).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_transform_point__literal() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3(17355384373, -13000621593, 18265871982));
    assert!(x.transform_point(p) == e);
}

/// `to_isometry().transform_point(p)` (loser: off by `|p| · (|real|² - 1)`).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_transform_point__alt_to_isometry() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3(17355384371, -13000621596, 18265871982));
    assert!(alt_transform_point(x, p) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_inverse_transform_point__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3(1722455392, 1021418165, -16408298172));
    assert!(e == e);
}

/// `inverse()` then the literal sandwich, upstream's composition (winner: fidelity).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_inverse_transform_point__literal() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3(1722455392, 1021418165, -16408298172));
    assert!(x.inverse_transform_point(p) == e);
}

/// `to_isometry().inverse_transform_point(p)` (loser: off by `|p| · (|real|² - 1)`).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_inverse_transform_point__alt_to_isometry() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let p: Point3<Fixed> = black_box(pt());
    let e: Point3<Fixed> = black_box(p3(1722455393, 1021418168, -16408298174));
    assert!(alt_inverse_transform_point(x, p) == e);
}

// --- benchmarks: interpolation

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_nlerp__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _y: UnitDualQuaternion<Fixed> = black_box(b());
    let _t: Fixed = black_box(fx(ONE_RAW / 3));
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4218528271, 15935807, -11951857, 806453041),
                (-1526407077, 2025637718, -2418470657, 6507184303),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_nlerp__nlerp() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let y: UnitDualQuaternion<Fixed> = black_box(b());
    let t: Fixed = black_box(fx(ONE_RAW / 3));
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4218528271, 15935807, -11951857, 806453041),
                (-1526407077, 2025637718, -2418470657, 6507184303),
            ),
        ),
    );
    assert!(x.nlerp(y, t) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_sclerp__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _y: UnitDualQuaternion<Fixed> = black_box(b());
    let _t: Fixed = black_box(fx(ONE_RAW / 3));
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4217160265, 8708451, -6531341, 813747428),
                (-1270283618, 1992283232, -2388432328, 6542620026),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_sclerp__sclerp() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let y: UnitDualQuaternion<Fixed> = black_box(b());
    let t: Fixed = black_box(fx(ONE_RAW / 3));
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4217160265, 8708451, -6531341, 813747428),
                (-1270283618, 1992283232, -2388432328, 6542620026),
            ),
        ),
    );
    assert!(x.sclerp(y, t) == e);
}

// --- benchmarks: heterogeneous products

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_translation__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _t: Translation3<Fixed> = black_box(tr());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, 534340439, -400755330, 267170219),
                (-2087267342, 5910697737, -8196362569, 8964476948),
            ),
        ),
    );
    assert!(e == e);
}

/// The reduced kernel (winner).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_translation__fused() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let t: Translation3<Fixed> = black_box(tr());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, 534340439, -400755330, 267170219),
                (-2087267342, 5910697737, -8196362569, 8964476948),
            ),
        ),
    );
    assert!(x.mul_translation(t) == e);
}

/// `self * from_parts(t, identity)`, upstream's formula (loser).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_translation__alt_from_parts() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let t: Translation3<Fixed> = black_box(tr());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (4234293283, 534340439, -400755330, 267170219),
                (-2087267342, 5910697737, -8196362569, 8964476948),
            ),
        ),
    );
    assert!(x * UnitDualQuaternionTrait::from_parts(t, UnitQuaternionTrait::identity()) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_unit_quaternion__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _r: UnitQuaternion<Fixed> = black_box(rb());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-3024284795, 314298191, -7118079354, 6217929968),
            ),
        ),
    );
    assert!(e == e);
}

/// The reduced kernel (winner, the same bits).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_unit_quaternion__fused() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let r: UnitQuaternion<Fixed> = black_box(rb());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-3024284795, 314298191, -7118079354, 6217929968),
            ),
        ),
    );
    assert!(x.mul_unit_quaternion(r) == e);
}

/// `self * from_rotation(r)`, upstream's formula (loser, the same bits).
#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_unit_quaternion__alt_from_rotation() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let r: UnitQuaternion<Fixed> = black_box(rb());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-3024284795, 314298191, -7118079354, 6217929968),
            ),
        ),
    );
    assert!(x * UnitDualQuaternionTrait::from_rotation(r) == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_isometry__baseline() {
    let _x: UnitDualQuaternion<Fixed> = black_box(a());
    let _i: Isometry3<Fixed> = black_box(b().to_isometry());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-4570607052, -1563305229, -6390197142, 8340766433),
            ),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_dual_quaternion_mul_isometry__mul_isometry() {
    let x: UnitDualQuaternion<Fixed> = black_box(a());
    let i: Isometry3<Fixed> = black_box(b().to_isometry());
    let e: UnitDualQuaternion<Fixed> = black_box(
        udqt(
            (
                (3724384862, -764072791, 125720322, 1994013199),
                (-4570607052, -1563305229, -6390197142, 8340766433),
            ),
        ),
    );
    assert!(x.mul_isometry(i) == e);
}

#[test]
fn test_bench_fixtures_are_the_bench_poses() {
    assert!(
        a() == UnitDualQuaternionTrait::from_parts(
            t3(0x180000000, -0x240000000, 0x3c0000000), ra(),
        ),
    );
    assert!(
        b() == UnitDualQuaternionTrait::from_parts(t3(-0xc0000000, 0x80000000, 0x140000000), rb()),
    );
}
