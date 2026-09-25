//! Gas benchmarks of the WP 8.4-P09b completion of `Isometry2/3` / `Similarity2/3` and of the
//! rotation operators with rotation-matrix poses (`bench_<type>_<op>__<variant>`, net = raw -
//! `baseline` of the group), and the alternative implementations that lost (`alt_*`, AGENTS.md
//! rule 8): upstream's literal `r.transform_vector(-p) + p` against the fused `rotate_translate`
//! of `rotation_wrt_point`.
//!
//! The poses are the `Isometry2` / `Isometry3` bench poses (hardcoded). Expected values are the
//! results of the kernels themselves, all of which are checked against upstream nalgebra in
//! `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::isometry2::{Isometry2, Isometry2Trait};
use nalgebra::geometry::isometry3::{Isometry3, Isometry3Trait};
use nalgebra::geometry::isometry_matrix3::IsometryMatrix3;
use nalgebra::geometry::quaternion::{Quaternion, QuaternionTranscendentalTrait};
use nalgebra::geometry::rotation3::{Rotation3, Rotation3Trait};
use nalgebra::geometry::similarity2::{Similarity2, Similarity2Trait};
use nalgebra::geometry::similarity3::Similarity3;
use nalgebra::geometry::translation2::Translation2;
use nalgebra::geometry::translation3::Translation3;
use nalgebra::geometry::unit_complex::{UnitComplex, UnitComplexTrait};
use nalgebra::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{iso2, iso3, isom3t, p2, p3, q, sim2, sim3, uc, uq, v3};

fn a2() -> Isometry2<Fixed> {
    iso2(6442450944, -9663676416, 3955926847, 1672539044)
}

fn b2() -> Isometry2<Fixed> {
    iso2(-3221225472, 2147483648, 4235452929, -712518464)
}

fn a3() -> Isometry3<Fixed> {
    iso3((6442450944, -9663676416, 16106127360), (4234293283, 534340439, -400755330, 267170219))
}

fn b3() -> Isometry3<Fixed> {
    iso3((-3221225472, 2147483648, 5368709120), (3689020097, -1022754606, 767065954, 1789820560))
}

// --- alternative implementations (losers)

/// The LOSER of `rotation_wrt_point`: upstream's literal `r.transform_vector(-p) + p` (two fused
/// kernels, then two checked additions). Bit for bit the same result.
#[inline(always)]
fn alt_rotation_wrt_point2(r: UnitComplex<Fixed>, p: Point2<Fixed>) -> Isometry2<Fixed> {
    let s = r.transform_vector(Vector2 { x: -p.x, y: -p.y });
    Isometry2 {
        rotation: r, translation: Translation2 { vector: Vector2 { x: s.x + p.x, y: s.y + p.y } },
    }
}

/// The LOSER of `rotation_wrt_point` in 3D.
#[inline(always)]
fn alt_rotation_wrt_point3(r: UnitQuaternion<Fixed>, p: Point3<Fixed>) -> Isometry3<Fixed> {
    let s = r.transform_vector(Vector3 { x: -p.x, y: -p.y, z: -p.z });
    Isometry3 {
        rotation: r,
        translation: Translation3 { vector: Vector3 { x: s.x + p.x, y: s.y + p.y, z: s.z + p.z } },
    }
}

#[test]
fn test_rotation_wrt_point_alt_agrees_bit_for_bit() {
    let (r, p) = (b2().rotation, p2(0x180000000, -0x240000000));
    assert!(alt_rotation_wrt_point2(r, p) == Isometry2Trait::rotation_wrt_point(r, p));
    let (r3, p3v) = (b3().rotation, p3(0x180000000, -0x240000000, 0x80000000));
    assert!(alt_rotation_wrt_point3(r3, p3v) == Isometry3Trait::rotation_wrt_point(r3, p3v));
}


#[test]
#[inline(never)]
fn bench_isometry2_div__baseline() {
    let _x: Isometry2<Fixed> = black_box(a2());
    let _y: Isometry2<Fixed> = black_box(b2());
    let e: Isometry2<Fixed> = black_box(iso2(10313000999, -9746270763, 3623642725, 2305636022));
    assert!(e == e);
}

/// Upstream's formula, `a * b.inverse()`.
#[test]
#[inline(never)]
fn bench_isometry2_div__inverse_then_mul() {
    let x: Isometry2<Fixed> = black_box(a2());
    let y: Isometry2<Fixed> = black_box(b2());
    let e: Isometry2<Fixed> = black_box(iso2(10313000999, -9746270763, 3623642725, 2305636022));
    assert!(x / y == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_rotation_wrt_point__baseline() {
    let _r: UnitComplex<Fixed> = black_box(uc(4235452929, -712518464));
    let _c: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Isometry2<Fixed> = black_box(iso2(1692438094, 934870370, 4235452929, -712518464));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_rotation_wrt_point__fused() {
    let r: UnitComplex<Fixed> = black_box(uc(4235452929, -712518464));
    let c: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Isometry2<Fixed> = black_box(iso2(1692438094, 934870370, 4235452929, -712518464));
    assert!(Isometry2Trait::rotation_wrt_point(r, c) == e);
}

/// The loser: the same bits, two checked additions more.
#[test]
#[inline(never)]
fn bench_isometry2_rotation_wrt_point__alt_rotate_then_add() {
    let r: UnitComplex<Fixed> = black_box(uc(4235452929, -712518464));
    let c: Point2<Fixed> = black_box(p2(0x180000000, -0x240000000));
    let e: Isometry2<Fixed> = black_box(iso2(1692438094, 934870370, 4235452929, -712518464));
    assert!(alt_rotation_wrt_point2(r, c) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_div__baseline() {
    let _x: Isometry3<Fixed> = black_box(a3());
    let _y: Isometry3<Fixed> = black_box(b3());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (10844409728, -10237397104, 11196481686),
            (3549427458, 1681980882, -814151394, -1535059155),
        ),
    );
    assert!(e == e);
}

/// Upstream's formula, `a * b.inverse()`.
#[test]
#[inline(never)]
fn bench_isometry3_div__inverse_then_mul() {
    let x: Isometry3<Fixed> = black_box(a3());
    let y: Isometry3<Fixed> = black_box(b3());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (10844409728, -10237397104, 11196481686),
            (3549427458, 1681980882, -814151394, -1535059155),
        ),
    );
    assert!(x / y == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_rotation_wrt_point__baseline() {
    let _r: UnitQuaternion<Fixed> = black_box(uq(3689020097, -1022754606, 767065954, 1789820560));
    let _c: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x80000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-5323917186, -9714403387, 1121077341),
            (3689020097, -1022754606, 767065954, 1789820560),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_rotation_wrt_point__fused() {
    let r: UnitQuaternion<Fixed> = black_box(uq(3689020097, -1022754606, 767065954, 1789820560));
    let c: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x80000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-5323917186, -9714403387, 1121077341),
            (3689020097, -1022754606, 767065954, 1789820560),
        ),
    );
    assert!(Isometry3Trait::rotation_wrt_point(r, c) == e);
}

/// The loser: the same bits, three checked additions more.
#[test]
#[inline(never)]
fn bench_isometry3_rotation_wrt_point__alt_rotate_then_add() {
    let r: UnitQuaternion<Fixed> = black_box(uq(3689020097, -1022754606, 767065954, 1789820560));
    let c: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x80000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-5323917186, -9714403387, 1121077341),
            (3689020097, -1022754606, 767065954, 1789820560),
        ),
    );
    assert!(alt_rotation_wrt_point3(r, c) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_look_at_lh__baseline() {
    let _eye: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x80000000));
    let _t: Point3<Fixed> = black_box(p3(0x100000000, 0, -0x200000000));
    let _u: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (5896186702, 5016171100, 8920566786),
            (-395856695, -149696793, -3997764486, -1511790835),
        ),
    );
    assert!(e == e);
}

/// `UnitQuaternion::look_at_lh` (the matrix frame, Shepperd's method, a conjugation), then one
/// quaternion rotation.
#[test]
#[inline(never)]
fn bench_isometry3_look_at_lh__unit_quaternion_frame() {
    let eye: Point3<Fixed> = black_box(p3(0x180000000, -0x240000000, 0x80000000));
    let t: Point3<Fixed> = black_box(p3(0x100000000, 0, -0x200000000));
    let u: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (5896186702, 5016171100, 8920566786),
            (-395856695, -149696793, -3997764486, -1511790835),
        ),
    );
    assert!(Isometry3Trait::look_at_lh(eye, t, u) == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_mul_isometry__baseline() {
    let _s: Similarity2<Fixed> = black_box(
        sim2(6442450944, -9663676416, 3955926847, 1672539044, 0x180000000),
    );
    let _y: Isometry2<Fixed> = black_box(b2());
    let e: Similarity2<Fixed> = black_box(
        sim2(737628957, -8578337706, 4178578243, 993090093, 6442450944),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity2_mul_isometry__fused() {
    let s: Similarity2<Fixed> = black_box(
        sim2(6442450944, -9663676416, 3955926847, 1672539044, 0x180000000),
    );
    let y: Isometry2<Fixed> = black_box(b2());
    let e: Similarity2<Fixed> = black_box(
        sim2(737628957, -8578337706, 4178578243, 993090093, 6442450944),
    );
    assert!(s.mul_isometry(y) == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_div__baseline() {
    let _s: Similarity3<Fixed> = black_box(
        sim3(
            (6442450944, -9663676416, 16106127360),
            (4234293283, 534340439, -400755330, 267170219),
            0x200000000,
        ),
    );
    let _y: Similarity3<Fixed> = black_box(
        sim3(
            (-3221225472, 2147483648, 5368709120),
            (3689020097, -1022754606, 767065954, 1789820560),
            0x80000000,
        ),
    );
    let e: Similarity3<Fixed> = black_box(
        sim3(
            (24050286082, -11958559160, -3532455330),
            (3549427458, 1681980882, -814151394, -1535059155),
            17179869184,
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_similarity3_div__inverse_then_mul() {
    let s: Similarity3<Fixed> = black_box(
        sim3(
            (6442450944, -9663676416, 16106127360),
            (4234293283, 534340439, -400755330, 267170219),
            0x200000000,
        ),
    );
    let y: Similarity3<Fixed> = black_box(
        sim3(
            (-3221225472, 2147483648, 5368709120),
            (3689020097, -1022754606, 767065954, 1789820560),
            0x80000000,
        ),
    );
    let e: Similarity3<Fixed> = black_box(
        sim3(
            (24050286082, -11958559160, -3532455330),
            (3549427458, 1681980882, -814151394, -1535059155),
            17179869184,
        ),
    );
    assert!(s / y == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_mul_isometry__baseline() {
    let _r: Rotation3<Fixed> = black_box(b3().rotation.to_rotation_matrix());
    let _y: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (6442450944, -9663676416, 16106127360),
            [[0x100000000, 0, 0], [0, 0, -0x100000000], [0, 0x100000000, 0]],
        ),
    );
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (13278517999, 7838486529, 12511524417),
            [
                [2529250561, 465276883, 3439935043], [2709293634, 2396233711, -2316146817],
                [-2170106835, 3533882496, 1117611249],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_rotation3_mul_isometry__matrix() {
    let r: Rotation3<Fixed> = black_box(b3().rotation.to_rotation_matrix());
    let y: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (6442450944, -9663676416, 16106127360),
            [[0x100000000, 0, 0], [0, 0, -0x100000000], [0, 0x100000000, 0]],
        ),
    );
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (13278517999, 7838486529, 12511524417),
            [
                [2529250561, 465276883, 3439935043], [2709293634, 2396233711, -2316146817],
                [-2170106835, 3533882496, 1117611249],
            ],
        ),
    );
    assert!(r.mul_isometry(y) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_exp_real__baseline() {
    let _x: Quaternion<Fixed> = black_box(q(0x180000000, 0, 0, 0));
    let e: Quaternion<Fixed> = black_box(q(4294967296, 0, 0, 0));
    assert!(e == e);
}

/// Upstream's early exit: the identity, no scalar `exp` evaluated. The figure is nevertheless the
/// one of the full path (`quaternion_exp__sin_cos`, 67 920): Sierra's gas accounting charges the
/// costliest branch of the `if` and does not refund the untaken one (measured: `exp` of a real and
/// of a non-real quaternion cost exactly the same).
#[test]
#[inline(never)]
fn bench_quaternion_exp_real__identity() {
    let x: Quaternion<Fixed> = black_box(q(0x180000000, 0, 0, 0));
    let e: Quaternion<Fixed> = black_box(q(4294967296, 0, 0, 0));
    assert!(x.exp() == e);
}
