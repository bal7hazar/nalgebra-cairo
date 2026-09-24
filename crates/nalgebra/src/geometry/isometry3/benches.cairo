//! Gas benchmarks of `Isometry3` (`bench_isometry3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together with
//! the tests showing why (AGENTS.md rule 8).
//!
//! The isometries used are `a = new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))` and
//! `b = new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))` (hardcoded, so that the `sin_cos` of their
//! construction is not measured), the point / vector `(-2.5, 3.75, 0.75)`, the translation
//! `(1.25, -0.375, 2.5)` and the half turn about `y` as the extra rotation. Expected values are the
//! results of the kernels themselves, all of which are checked against upstream nalgebra in
//! `tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix3::Matrix3Trait;
use crate::base::matrix_test_utils::{fx, iso3, p3, uq, v3};
use crate::base::point3::{Point3, Point3Trait};
use crate::base::vector3::Vector3;
use crate::geometry::isometry3::Isometry3InternalTrait;
use crate::geometry::translation3::Translation3;
use crate::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use super::{Isometry3, Isometry3AngleTrait, Isometry3Trait};

/// `new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))`.
fn a() -> Isometry3<Fixed> {
    iso3((6442450944, -9663676416, 16106127360), (4234293283, 534340439, -400755330, 267170219))
}

/// `new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))`.
fn b() -> Isometry3<Fixed> {
    iso3((-3221225472, 2147483648, 5368709120), (3689020097, -1022754606, 767065954, 1789820560))
}

/// `(-2.5, 3.75, 0.75)`, as a point and as a vector.
fn p() -> Point3<Fixed> {
    p3(-0x280000000, 0x3c0000000, 0xc0000000)
}

fn v() -> Vector3<Fixed> {
    v3(-0x280000000, 0x3c0000000, 0xc0000000)
}

/// `(1.25, -0.375, 2.5)`.
fn t() -> Translation3<Fixed> {
    Translation3 { vector: v3(0x140000000, -0x60000000, 0x280000000) }
}

/// The half turn about `y`, exactly representable.
fn r() -> UnitQuaternion<Fixed> {
    uq(0, 0, 0x100000000, 0)
}

// --- alternative implementations (losers)

/// The LOSER of `transform_point`: upstream's literal `rotation * p + translation`, i.e. the
/// quaternion sandwich and then three `Fixed` additions, instead of the single wide accumulation
/// of `rotate_translate`. Bit for bit the same result (adding an integral number of raw units
/// commutes with the floor), 5 % dearer: every `Fixed` addition pays an overflow check.
fn alt_transform_point_rotate_then_add(i: Isometry3<Fixed>, q: Point3<Fixed>) -> Point3<Fixed> {
    let c = i.rotation.transform_point(q);
    let tr = i.translation.vector;
    Point3 { x: c.x + tr.x, y: c.y + tr.y, z: c.z + tr.z }
}

/// `inv_mul` as it was before `UnitQuaternion::conj_mul`: the rotation is
/// `self.rotation.conjugate() * other.rotation` and the translation goes through
/// `transform_vector` of the conjugate, i.e. six negations more for the same bits.
fn alt_inv_mul_conjugate_then_mul(s: Isometry3<Fixed>, o: Isometry3<Fixed>) -> Isometry3<Fixed> {
    let d = Vector3 {
        x: o.translation.vector.x - s.translation.vector.x,
        y: o.translation.vector.y - s.translation.vector.y,
        z: o.translation.vector.z - s.translation.vector.z,
    };
    let c = s.rotation.conjugate();
    Isometry3 {
        rotation: UnitQuaternion { quaternion: c.quaternion * o.rotation.quaternion },
        translation: Translation3 { vector: c.transform_vector(d) },
    }
}

// --- why the alternatives lost

/// The fused `inv_mul` (`conj_mul` and the sign-folded `inverse_transform_vector`) gives exactly
/// the bits of the conjugate-first formulation.
#[test]
fn test_inv_mul_fused_matches_conjugate_then_mul() {
    assert!(a().inv_mul(b()) == alt_inv_mul_conjugate_then_mul(a(), b()));
    assert!(b().inv_mul(a()) == alt_inv_mul_conjugate_then_mul(b(), a()));
}

/// `inv_mul` and `self.inverse() * other` are the same transform, but the second rounds the
/// intermediate `rotation⁻¹ · (-translation)` before adding the rotated translation of `other`:
/// 2 ulp apart here, and 1.55x the gas.
#[test]
fn test_inv_mul_alt_inverse_then_mul_differs_by_rounding() {
    let (x, y) = (a(), b());
    let got = x.inv_mul(y);
    let alt = x.inverse() * y;
    assert!(got.rotation == alt.rotation);
    assert!(got.translation != alt.translation);
    assert!(got.abs_diff_eq(alt, 2));
}

/// The fully fused `transform_point` gives exactly the same bits as rotating and then adding,
/// because the translation is an integral number of raw units: `floor(x + t) = floor(x) + t`.
#[test]
fn test_transform_point_fused_and_composed_agree_bit_for_bit() {
    let (x, q) = (a(), p());
    assert!(x.transform_point(q) == alt_transform_point_rotate_then_add(x, q));
}

// --- gas benchmarks

#[test]
#[inline(never)]
fn bench_isometry3_identity__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let e: Isometry3<Fixed> = black_box(iso3((0, 0, 0), (0x100000000, 0, 0, 0)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_identity__const() {
    let _x: Isometry3<Fixed> = black_box(a());
    let e: Isometry3<Fixed> = black_box(iso3((0, 0, 0), (0x100000000, 0, 0, 0)));
    assert!(Isometry3Trait::<Fixed>::identity() == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_from_parts__baseline() {
    let _t: Translation3<Fixed> = black_box(a().translation);
    let _r: UnitQuaternion<Fixed> = black_box(a().rotation);
    let e: Isometry3<Fixed> = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_from_parts__wrap() {
    let u: Translation3<Fixed> = black_box(a().translation);
    let q: UnitQuaternion<Fixed> = black_box(a().rotation);
    let e: Isometry3<Fixed> = black_box(a());
    assert!(Isometry3Trait::from_parts(u, q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_translation__baseline() {
    let _u: Vector3<Fixed> = black_box(t().vector);
    let e: Isometry3<Fixed> = black_box(
        iso3((0x140000000, -0x60000000, 0x280000000), (0x100000000, 0, 0, 0)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_translation__pure() {
    let u: Vector3<Fixed> = black_box(t().vector);
    let e: Isometry3<Fixed> = black_box(
        iso3((0x140000000, -0x60000000, 0x280000000), (0x100000000, 0, 0, 0)),
    );
    assert!(Isometry3Trait::translation(u.x, u.y, u.z) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_from_translation__baseline() {
    let _u: Translation3<Fixed> = black_box(t());
    let e: Isometry3<Fixed> = black_box(
        iso3((0x140000000, -0x60000000, 0x280000000), (0x100000000, 0, 0, 0)),
    );
    assert!(e == e);
}

/// `t.into()` (upstream `From<Translation3> for Isometry3`).
#[test]
#[inline(never)]
fn bench_isometry3_from_translation__pure() {
    let u: Translation3<Fixed> = black_box(t());
    let e: Isometry3<Fixed> = black_box(
        iso3((0x140000000, -0x60000000, 0x280000000), (0x100000000, 0, 0, 0)),
    );
    let r: Isometry3<Fixed> = u.into();
    assert!(r == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_new__baseline() {
    let _u: Vector3<Fixed> = black_box(v3(6442450944, -9663676416, 16106127360));
    let _w: Vector3<Fixed> = black_box(v3(0x40000000, -0x30000000, 0x20000000));
    let e: Isometry3<Fixed> = black_box(a());
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_new__from_scaled_axis() {
    let u: Vector3<Fixed> = black_box(v3(6442450944, -9663676416, 16106127360));
    let w: Vector3<Fixed> = black_box(v3(0x40000000, -0x30000000, 0x20000000));
    let e: Isometry3<Fixed> = black_box(a());
    assert!(Isometry3AngleTrait::new(u, w) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_rotation__baseline() {
    let _w: Vector3<Fixed> = black_box(v3(0x40000000, -0x30000000, 0x20000000));
    let e: Isometry3<Fixed> = black_box(
        iso3((0, 0, 0), (4234293283, 534340439, -400755330, 267170219)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_rotation__from_scaled_axis() {
    let w: Vector3<Fixed> = black_box(v3(0x40000000, -0x30000000, 0x20000000));
    let e: Isometry3<Fixed> = black_box(
        iso3((0, 0, 0), (4234293283, 534340439, -400755330, 267170219)),
    );
    assert!(Isometry3AngleTrait::rotation(w) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inverse__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-8531988124, 6465531078, -16724271021),
            (4234293283, -534340439, 400755330, -267170219),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inverse__conjugate_rotate() {
    let x: Isometry3<Fixed> = black_box(a());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-8531988124, 6465531078, -16724271021),
            (4234293283, -534340439, 400755330, -267170219),
        ),
    );
    assert!(x.inverse() == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inv_mul__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _y: Isometry3<Fixed> = black_box(b());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-10387824139, 10254455918, -11624179020),
            (3549427458, -1252539985, 1386739257, 1535059155),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inv_mul__direct() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(b());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-10387824139, 10254455918, -11624179020),
            (3549427458, -1252539985, 1386739257, 1535059155),
        ),
    );
    assert!(x.inv_mul(y) == e);
}

/// The previous formulation: conjugate first (six negations), same bits as `__direct`.
#[test]
#[inline(never)]
fn bench_isometry3_inv_mul__alt_conjugate_then_mul() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(b());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-10387824139, 10254455918, -11624179020),
            (3549427458, -1252539985, 1386739257, 1535059155),
        ),
    );
    assert!(alt_inv_mul_conjugate_then_mul(x, y) == e);
}

/// The loser: the inverse is materialised, so its translation is rotated once for nothing and
/// rounded before the composition (2 ulp apart, see the test above).
#[test]
#[inline(never)]
fn bench_isometry3_inv_mul__alt_inverse_then_mul() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(b());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-10387824141, 10254455918, -11624179022),
            (3549427458, -1252539985, 1386739257, 1535059155),
        ),
    );
    assert!(x.inverse() * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_mul__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _y: Isometry3<Fixed> = black_box(b());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (2084353234, -9298899151, 21074521374), (3724384862, -764072791, 125720322, 1994013199),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_mul__compose() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(b());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (2084353234, -9298899151, 21074521374), (3724384862, -764072791, 125720322, 1994013199),
        ),
    );
    assert!(x * y == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_transform_point__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-6917091138, 3923952211, 20793852411));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_transform_point__fused() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-6917091138, 3923952211, 20793852411));
    assert!(x.transform_point(q) == e);
}

/// The loser: `rotation * p` and then three `Fixed` additions (upstream's literal form). The same
/// bits (see the test above), 5 % dearer.
#[test]
#[inline(never)]
fn bench_isometry3_transform_point__alt_rotate_then_add() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-6917091138, 3923952211, 20793852411));
    assert!(alt_transform_point_rotate_then_add(x, q) == e);
}

/// The matrix path: converting to a `Rotation3` and multiplying. It loses on ONE point and wins
/// from two on, like `UnitQuaternion::transform_vector` (see the module documentation).
#[test]
#[inline(never)]
fn bench_isometry3_transform_point__alt_rotation_matrix() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-6917091138, 3923952211, 20793852411));
    let m = x.rotation.to_rotation_matrix().matrix;
    let c = m.mul_vec(Vector3 { x: q.x, y: q.y, z: q.z });
    let tr = x.translation.vector;
    let got = Point3 { x: c.x + tr.x, y: c.y + tr.y, z: c.z + tr.z };
    assert!(got.abs_diff_eq(e, 16));
}

#[test]
#[inline(never)]
fn bench_isometry3_transform_vector__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _w: Vector3<Fixed> = black_box(v());
    let e: Vector3<Fixed> = black_box(v3(-13359542082, 13587628627, 4687725051));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_transform_vector__rotate() {
    let x: Isometry3<Fixed> = black_box(a());
    let w: Vector3<Fixed> = black_box(v());
    let e: Vector3<Fixed> = black_box(v3(-13359542082, 13587628627, 4687725051));
    assert!(x.transform_vector(w) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inverse_transform_point__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-16755308845, 24267495181, -15987485471));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inverse_transform_point__subtract_rotate() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: Point3<Fixed> = black_box(p());
    let e: Point3<Fixed> = black_box(p3(-16755308845, 24267495181, -15987485471));
    assert!(x.inverse_transform_point(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inverse_transform_vector__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _w: Vector3<Fixed> = black_box(v());
    let e: Vector3<Fixed> = black_box(v3(-8223320724, 17801964100, 736785547));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_inverse_transform_vector__conjugate_rotate() {
    let x: Isometry3<Fixed> = black_box(a());
    let w: Vector3<Fixed> = black_box(v());
    let e: Vector3<Fixed> = black_box(v3(-8223320724, 17801964100, 736785547));
    assert!(x.inverse_transform_vector(w) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_translation__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _u: Translation3<Fixed> = black_box(t());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (11811160064, -11274289152, 26843545600),
            (4234293283, 534340439, -400755330, 267170219),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_translation__add() {
    let x: Isometry3<Fixed> = black_box(a());
    let u: Translation3<Fixed> = black_box(t());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (11811160064, -11274289152, 26843545600),
            (4234293283, 534340439, -400755330, 267170219),
        ),
    );
    let mut m = x;
    m.append_translation_mut(u);
    assert!(m == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_prepend_translation__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _u: Translation3<Fixed> = black_box(t());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (10101792444, -13436727545, 27018623241),
            (4234293283, 534340439, -400755330, 267170219),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_prepend_translation__rotate_add() {
    let x: Isometry3<Fixed> = black_box(a());
    let u: Translation3<Fixed> = black_box(t());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (10101792444, -13436727545, 27018623241),
            (4234293283, 534340439, -400755330, 267170219),
        ),
    );
    assert!(x.mul_translation(u) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_rotation__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _q: UnitQuaternion<Fixed> = black_box(r());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-6442450944, -9663676416, -16106127360),
            (400755330, 267170219, 4234293283, -534340439),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_rotation__compose_rotate() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: UnitQuaternion<Fixed> = black_box(r());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-6442450944, -9663676416, -16106127360),
            (400755330, 267170219, 4234293283, -534340439),
        ),
    );
    let mut m = x;
    m.append_rotation_mut(q);
    assert!(m == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_prepend_rotation__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _q: UnitQuaternion<Fixed> = black_box(r());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (400755330, -267170219, 4234293283, 534340439),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_prepend_rotation__compose() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: UnitQuaternion<Fixed> = black_box(r());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (400755330, -267170219, 4234293283, 534340439),
        ),
    );
    assert!(x.mul_unit_quaternion(q) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_rotation_wrt_point__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _q: UnitQuaternion<Fixed> = black_box(r());
    let _c: Point3<Fixed> = black_box(p());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-27917287424, -9663676416, -9663676416),
            (400755330, 267170219, 4234293283, -534340439),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_rotation_wrt_point__shift_rotate() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: UnitQuaternion<Fixed> = black_box(r());
    let c: Point3<Fixed> = black_box(p());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (-27917287424, -9663676416, -9663676416),
            (400755330, 267170219, 4234293283, -534340439),
        ),
    );
    let mut m = x;
    m.append_rotation_wrt_point_mut(q, c);
    assert!(m == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_rotation_wrt_center__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _q: UnitQuaternion<Fixed> = black_box(r());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (400755330, 267170219, 4234293283, -534340439),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_append_rotation_wrt_center__compose() {
    let x: Isometry3<Fixed> = black_box(a());
    let q: UnitQuaternion<Fixed> = black_box(r());
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (400755330, 267170219, 4234293283, -534340439),
        ),
    );
    let mut m = x;
    m.append_rotation_wrt_center_mut(q);
    assert!(m == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_to_homogeneous__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(4186940973));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_to_homogeneous__matrix4() {
    let x: Isometry3<Fixed> = black_box(a());
    let e: Fixed = black_box(fx(4186940973));
    let m = x.to_homogeneous();
    assert!(m.m11 == e && m.m14 == fx(6442450944));
}

#[test]
#[inline(never)]
fn bench_isometry3_renormalize__baseline() {
    let _x: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_renormalize__exact() {
    let x: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    assert!(x.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_renormalize_fast__baseline() {
    let _x: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    assert!(e == e);
}

/// One Newton step instead of a norm and four divisions: the same bits here (the drift is 3 ulp),
/// which is the case a physics step is in after composing a pose once.
#[test]
#[inline(never)]
fn bench_isometry3_renormalize_fast__newton() {
    let x: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (6442450944, -9663676416, 16106127360), (4234293283, 534340442, -400755332, 267170219),
        ),
    );
    assert!(x.renormalize_fast() == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_lerp_slerp__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _y: Isometry3<Fixed> = black_box(b());
    let _s: Fixed = black_box(fx(0x40000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (4026531840, -6710886400, 13421772800), (4237245848, 140968788, -105726593, 679294774),
        ),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_lerp_slerp__acos_sin() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(b());
    let s: Fixed = black_box(fx(0x40000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (4026531840, -6710886400, 13421772800), (4237245848, 140968788, -105726593, 679294774),
        ),
    );
    assert!(x.lerp_slerp(y, s) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_lerp_nlerp__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _y: Isometry3<Fixed> = black_box(b());
    let _s: Fixed = black_box(fx(0x40000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (4026531840, -6710886400, 13421772800), (4238238332, 150031943, -112523959, 670006486),
        ),
    );
    assert!(e == e);
}

/// The trigonometry-free interpolation: same path, cheaper parametrisation.
#[test]
#[inline(never)]
fn bench_isometry3_lerp_nlerp__lerp_normalize() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(b());
    let s: Fixed = black_box(fx(0x40000000));
    let e: Isometry3<Fixed> = black_box(
        iso3(
            (4026531840, -6710886400, 13421772800), (4238238332, 150031943, -112523959, 670006487),
        ),
    );
    assert!(x.lerp_nlerp(y, s) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_face_towards__baseline() {
    let _eye: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x300000000));
    let _target: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x400000000));
    let _up: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: Isometry3<Fixed> = black_box(
        iso3((0x100000000, 0x200000000, 0x300000000), (0x100000000, 0, 0, 0)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_face_towards__gram_schmidt() {
    let eye: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x300000000));
    let target: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x400000000));
    let up: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: Isometry3<Fixed> = black_box(
        iso3((0x100000000, 0x200000000, 0x300000000), (0x100000000, 0, 0, 0)),
    );
    assert!(Isometry3Trait::face_towards(eye, target, up) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_look_at_rh__baseline() {
    let _eye: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x300000000));
    let _target: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x400000000));
    let _up: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: Isometry3<Fixed> = black_box(
        iso3((0x100000000, -0x200000000, 0x300000000), (0, 0, 0x100000000, 0)),
    );
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_look_at_rh__gram_schmidt() {
    let eye: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x300000000));
    let target: Point3<Fixed> = black_box(p3(0x100000000, 0x200000000, 0x400000000));
    let up: Vector3<Fixed> = black_box(v3(0, 0x100000000, 0));
    let e: Isometry3<Fixed> = black_box(
        iso3((0x100000000, -0x200000000, 0x300000000), (0, 0, 0x100000000, 0)),
    );
    assert!(Isometry3Trait::look_at_rh(eye, target, up) == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_abs_diff_eq__baseline() {
    let _x: Isometry3<Fixed> = black_box(a());
    let _y: Isometry3<Fixed> = black_box(
        iso3(
            (6442450946, -9663676415, 16106127360), (4234293283, 534340440, -400755330, 267170219),
        ),
    );
    let e: bool = black_box(true);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_isometry3_abs_diff_eq__ulps() {
    let x: Isometry3<Fixed> = black_box(a());
    let y: Isometry3<Fixed> = black_box(
        iso3(
            (6442450946, -9663676415, 16106127360), (4234293283, 534340440, -400755330, 267170219),
        ),
    );
    let e: bool = black_box(true);
    assert!(x.abs_diff_eq(y, 2) == e);
}
