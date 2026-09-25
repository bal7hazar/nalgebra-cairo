//! Gas benchmarks of `IsometryMatrix2` / `IsometryMatrix3` (`bench_<type>_<op>__<variant>`, net =
//! raw - `baseline` of the group), and the alternative implementations that lost (`alt_*`), kept
//! as evidence (AGENTS.md rule 8): upstream's literal "rotate, then add the translation" against
//! the fused `rotate_translate`, and `inverse() * other` against the fused `inv_mul`.
//!
//! The 2D inputs are the matrix forms of the `Isometry2` bench poses (`a = new((1.5, -2.25), 0.4
//! rad)`, `b = new((-0.75, 0.5), -1/6 rad)`, hardcoded), the 3D ones the conversions of the
//! `Isometry3` bench poses (the conversion is paid by the baseline too). Expected values are the
//! results of the kernels themselves, all of which are checked against upstream nalgebra in
//! `tests.cairo`.

use fixed::Fixed;
use nalgebra::base::point2::Point2;
use nalgebra::base::point3::Point3;
use nalgebra::base::vector3::Vector3;
use nalgebra::geometry::isometry3::Isometry3;
use nalgebra::geometry::isometry_matrix2::{IsometryMatrix2, IsometryMatrix2Trait};
use nalgebra::geometry::isometry_matrix3::{IsometryMatrix3, IsometryMatrix3Trait};
use nalgebra::geometry::rotation2::Rotation2Trait;
use nalgebra::geometry::rotation3::Rotation3Trait;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{iso3, isom2t, isom3t, p2, p3, v3};

/// `new((1.5, -2.25), 0.4 rad)` as a matrix pose.
fn a2() -> IsometryMatrix2<Fixed> {
    isom2t((6442450944, -9663676416), [[3955926847, -1672539044], [1672539044, 3955926847]])
}

/// `new((-0.75, 0.5), -1/6 rad)` as a matrix pose.
fn b2() -> IsometryMatrix2<Fixed> {
    isom2t((-3221225472, 2147483648), [[4235452929, 712518464], [-712518464, 4235452929]])
}

/// The `a` of the `Isometry3` benches, converted.
fn a3() -> IsometryMatrix3<Fixed> {
    let q: Isometry3<Fixed> = iso3(
        (6442450944, -9663676416, 16106127360), (4234293283, 534340439, -400755330, 267170219),
    );
    q.into()
}

/// The `b` of the `Isometry3` benches, converted.
fn b3() -> IsometryMatrix3<Fixed> {
    let q: Isometry3<Fixed> = iso3(
        (-3221225472, 2147483648, 5368709120), (3689020097, -1022754606, 767065954, 1789820560),
    );
    q.into()
}

// --- alternative implementations (losers)

/// The LOSER of `transform_point`: upstream's literal `rotation * p + translation`, two fused
/// kernels then two checked `Fixed` additions. Bit for bit the same result (`tests.cairo`).
#[inline(always)]
fn alt_transform_point2_rotate_then_add(
    i: IsometryMatrix2<Fixed>, q: Point2<Fixed>,
) -> Point2<Fixed> {
    let c = i.rotation.transform_point(q);
    Point2 { x: c.x + i.translation.vector.x, y: c.y + i.translation.vector.y }
}

/// The LOSER of `transform_point` in 3D (three checked additions).
#[inline(always)]
fn alt_transform_point3_rotate_then_add(
    i: IsometryMatrix3<Fixed>, q: Point3<Fixed>,
) -> Point3<Fixed> {
    let c = i.rotation.transform_point(q);
    let t = i.translation.vector;
    Point3 { x: c.x + t.x, y: c.y + t.y, z: c.z + t.z }
}


#[test]
#[inline(never)]
fn bench_isometry_matrix2_mul__baseline() {
    let _x: IsometryMatrix2<Fixed> = black_box(a2());
    let _y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((2639236286, -8940117276), [[4178578243, -993090094], [993090093, 4178578243]]),
    );
    assert!(e == e);
}

/// Four fused kernels for the rotation, one `rotate_translate`.
#[test]
#[inline(never)]
fn bench_isometry_matrix2_mul__composition() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((2639236286, -8940117276), [[4178578243, -993090094], [993090093, 4178578243]]),
    );
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix2_inverse__baseline() {
    let _x: IsometryMatrix2<Fixed> = black_box(a2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((-2170677422, 11409643971), [[3955926847, 1672539044], [-1672539044, 3955926847]]),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix2_inverse__transpose() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((-2170677422, 11409643971), [[3955926847, 1672539044], [-1672539044, 3955926847]]),
    );
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix2_inv_mul__baseline() {
    let _x: IsometryMatrix2<Fixed> = black_box(a2());
    let _y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((-4301353035, 14642011678), [[3623642725, 2305636022], [-2305636023, 3623642725]]),
    );
    assert!(e == e);
}

/// Upstream's formula: `rotationᵀ · (b.t - a.t)`, `rotationᵀ · b.rotation`.
#[test]
#[inline(never)]
fn bench_isometry_matrix2_inv_mul__fused() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((-4301353035, 14642011678), [[3623642725, 2305636022], [-2305636023, 3623642725]]),
    );
    assert!(x.inv_mul(y) == e);
}

/// The loser: the inverse is materialised, then composed (one more rounding).
#[test]
#[inline(never)]
fn bench_isometry_matrix2_inv_mul__alt_inverse_then_mul() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((-4301353036, 14642011677), [[3623642725, 2305636022], [-2305636023, 3623642725]]),
    );
    assert!(x.inverse() * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix2_div__baseline() {
    let _x: IsometryMatrix2<Fixed> = black_box(a2());
    let _y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((10313000999, -9746270763), [[3623642725, -2305636023], [2305636022, 3623642725]]),
    );
    assert!(e == e);
}

/// Upstream's formula, `a * b.inverse()`.
#[test]
#[inline(never)]
fn bench_isometry_matrix2_div__inverse_then_mul() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let y: IsometryMatrix2<Fixed> = black_box(b2());
    let e: IsometryMatrix2<Fixed> = black_box(
        isom2t((10313000999, -9746270763), [[3623642725, -2305636023], [2305636022, 3623642725]]),
    );
    assert!(x / y == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix2_transform_point__baseline() {
    let _x: IsometryMatrix2<Fixed> = black_box(a2());
    let _q: Point2<Fixed> = black_box(p2(-0x280000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(-9719387589, 989701650));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix2_transform_point__fused() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let q: Point2<Fixed> = black_box(p2(-0x280000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(-9719387589, 989701650));
    assert!(x.transform_point(q) == e);
}

/// The loser: the same bits, two checked additions more.
#[test]
#[inline(never)]
fn bench_isometry_matrix2_transform_point__alt_rotate_then_add() {
    let x: IsometryMatrix2<Fixed> = black_box(a2());
    let q: Point2<Fixed> = black_box(p2(-0x280000000, 0x3c0000000));
    let e: Point2<Fixed> = black_box(p2(-9719387589, 989701650));
    assert!(alt_transform_point2_rotate_then_add(x, q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_mul__baseline() {
    let _x: IsometryMatrix3<Fixed> = black_box(a3());
    let _y: IsometryMatrix3<Fixed> = black_box(b3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (2084353234, -9298899152, 21074521373),
            [
                [2436096684, -3502951961, -491431154], [3413489599, 2171600315, 1441868542],
                [-927505130, -1208397167, 4015750812],
            ],
        ),
    );
    assert!(e == e);
}

/// 27 products for the rotation (one rounding per entry), one `rotate_translate`.
#[test]
#[inline(never)]
fn bench_isometry_matrix3_mul__composition() {
    let x: IsometryMatrix3<Fixed> = black_box(a3());
    let y: IsometryMatrix3<Fixed> = black_box(b3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (2084353234, -9298899152, 21074521373),
            [
                [2436096684, -3502951961, -491431154], [3413489599, 2171600315, 1441868542],
                [-927505130, -1208397167, 4015750812],
            ],
        ),
    );
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_inverse__baseline() {
    let _x: IsometryMatrix3<Fixed> = black_box(a3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (-8531988122, 6465531078, -16724271018),
            [
                [4186940973, 427075328, 856665640], [-626508539, 4128772954, 1003725567],
                [-723710168, -1103442173, 4087224368],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_inverse__transpose() {
    let x: IsometryMatrix3<Fixed> = black_box(a3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (-8531988122, 6465531078, -16724271018),
            [
                [4186940973, 427075328, 856665640], [-626508539, 4128772954, 1003725567],
                [-723710168, -1103442173, 4087224368],
            ],
        ),
    );
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_inv_mul__baseline() {
    let _x: IsometryMatrix3<Fixed> = black_box(a3());
    let _y: IsometryMatrix3<Fixed> = black_box(b3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (-10387824138, 10254455918, -11624179018),
            [
                [2302191991, -3346021979, 1396707923], [1728364614, 2467124263, 3061502522],
                [-3187383232, -1078971206, 2668923630],
            ],
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_inv_mul__fused() {
    let x: IsometryMatrix3<Fixed> = black_box(a3());
    let y: IsometryMatrix3<Fixed> = black_box(b3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (-10387824138, 10254455918, -11624179018),
            [
                [2302191991, -3346021979, 1396707923], [1728364614, 2467124263, 3061502522],
                [-3187383232, -1078971206, 2668923630],
            ],
        ),
    );
    assert!(x.inv_mul(y) == e);
}

/// The loser: the inverse is materialised, then composed.
#[test]
#[inline(never)]
fn bench_isometry_matrix3_inv_mul__alt_inverse_then_mul() {
    let x: IsometryMatrix3<Fixed> = black_box(a3());
    let y: IsometryMatrix3<Fixed> = black_box(b3());
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (-10387824138, 10254455918, -11624179019),
            [
                [2302191991, -3346021979, 1396707923], [1728364614, 2467124263, 3061502522],
                [-3187383232, -1078971206, 2668923630],
            ],
        ),
    );
    assert!(x.inverse() * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_transform_point__baseline() {
    let _x: IsometryMatrix3<Fixed> = black_box(a3());
    let _q: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let e: Point3<Fixed> = black_box(p3(-6917091136, 3923952211, 20793852412));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_transform_point__fused() {
    let x: IsometryMatrix3<Fixed> = black_box(a3());
    let q: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let e: Point3<Fixed> = black_box(p3(-6917091136, 3923952211, 20793852412));
    assert!(x.transform_point(q) == e);
}

/// The loser: the same bits, three checked additions more.
#[test]
#[inline(never)]
fn bench_isometry_matrix3_transform_point__alt_rotate_then_add() {
    let x: IsometryMatrix3<Fixed> = black_box(a3());
    let q: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let e: Point3<Fixed> = black_box(p3(-6917091136, 3923952211, 20793852412));
    assert!(alt_transform_point3_rotate_then_add(x, q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry_matrix3_look_at_rh__baseline() {
    let _eye: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let _t: Point3<Fixed> = black_box(p3(0x100000000, 0, -0x200000000));
    let _u: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (4100899740, -5595312497, -18356221886),
            [
                [2653523361, 0, 3377211550], [2175954859, 3284655669, -1709678819],
                [-2582784989, 2767269631, 2029331063],
            ],
        ),
    );
    assert!(e == e);
}

/// `Rotation3::look_at_rh` (three normalisations) and one `Matrix3 * Vector3`.
#[test]
#[inline(never)]
fn bench_isometry_matrix3_look_at_rh__rotation3_frame() {
    let eye: Point3<Fixed> = black_box(p3(-0x280000000, 0x3c0000000, 0xc0000000));
    let t: Point3<Fixed> = black_box(p3(0x100000000, 0, -0x200000000));
    let u: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: IsometryMatrix3<Fixed> = black_box(
        isom3t(
            (4100899740, -5595312497, -18356221886),
            [
                [2653523361, 0, 3377211550], [2175954859, 3284655669, -1709678819],
                [-2582784989, 2767269631, 2029331063],
            ],
        ),
    );
    assert!(IsometryMatrix3Trait::look_at_rh(eye, t, u) == e);
}
