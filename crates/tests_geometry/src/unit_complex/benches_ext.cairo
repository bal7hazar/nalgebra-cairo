//! Gas benchmarks of the WP 8.4-P08 completion of `UnitComplex`
//! (`bench_unit_complex_<op>__<variant>`, net = raw - `baseline` of the group), and the
//! alternative implementations that lost (`alt_*` variants), kept as evidence together with the
//! tests showing why (AGENTS.md rule 8; `tests_ext.cairo`).
//!
//! The inputs are `c = new(0.4 rad)` and `d = new(-1/6 rad)` of `benches.cairo`, the vector
//! `(1.5, -2.25)`, the isometry `(v, d)`, the similarity `((v, d), 2)` and a well-conditioned
//! matrix. Expected values are the results of the kernels themselves, all of which are checked
//! against upstream nalgebra in `tests_ext.cairo`.
//!
//! Moved from `crates/nalgebra/src/geometry/unit_complex/benches_ext.cairo` (WP 8.1c, test-only
//! package).

use core::num::traits::One;
use fixed::Fixed;
use nalgebra::base::matrix2::Matrix2;
use nalgebra::base::unit::Unit;
use nalgebra::base::vector2::Vector2;
use nalgebra::geometry::isometry2::Isometry2;
use nalgebra::geometry::rotation2::Rotation2;
use nalgebra::geometry::similarity2::Similarity2;
use nalgebra::geometry::translation2::Translation2;
use nalgebra::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, iso2, m2i, sim2, uc, v2};

fn c() -> UnitComplex<Fixed> {
    uc(3955926847, 1672539044)
}

fn d() -> UnitComplex<Fixed> {
    uc(4235452929, -712518464)
}

fn rot_d() -> Rotation2<Fixed> {
    d().to_rotation_matrix()
}

/// `(1.5, -2.25)`.
fn v() -> Vector2<Fixed> {
    v2(0x180000000, -0x240000000)
}

/// `(0.6, 0.8)`.
fn u2a() -> Unit<Vector2<Fixed>> {
    Unit { value: v2(2576980378, 3435973837) }
}

/// `(-0.8, 0.6)`.
fn u2b() -> Unit<Vector2<Fixed>> {
    Unit { value: v2(-3435973837, 2576980378) }
}

fn iso() -> Isometry2<Fixed> {
    Isometry2 { rotation: d(), translation: Translation2 { vector: v() } }
}

fn sim() -> Similarity2<Fixed> {
    Similarity2 { isometry: iso(), scaling: fx(0x200000000) }
}

/// A well-conditioned matrix.
fn mat() -> Matrix2<Fixed> {
    m2i([[2, -1], [1, 3]])
}

/// The columns of `c`'s rotation matrix.
fn basis() -> [Vector2<Fixed>; 2] {
    [Vector2 { x: c().re, y: c().im }, Vector2 { x: -c().im, y: c().re }]
}

// --- div

#[test]
#[inline(never)]
fn bench_unit_complex_div__baseline() {
    let _c = black_box(c());
    let _d = black_box(d());
    let e = black_box(uc(3623642725, 2305636022));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_div__fused() {
    let c = black_box(c());
    let d = black_box(d());
    let e = black_box(uc(3623642725, 2305636022));
    assert!(c / d == e);
}

// --- mul_rotation

#[test]
#[inline(never)]
fn bench_unit_complex_mul_rotation__baseline() {
    let _c = black_box(c());
    let _r = black_box(rot_d());
    let e = black_box(uc(4178578243, 993090093));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_mul_rotation__kernels() {
    let c = black_box(c());
    let r = black_box(rot_d());
    let e = black_box(uc(4178578243, 993090093));
    assert!(c.mul_rotation(r) == e);
}

// --- div_rotation

#[test]
#[inline(never)]
fn bench_unit_complex_div_rotation__baseline() {
    let _c = black_box(c());
    let _r = black_box(rot_d());
    let e = black_box(uc(3623642725, 2305636022));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_div_rotation__kernels() {
    let c = black_box(c());
    let r = black_box(rot_d());
    let e = black_box(uc(3623642725, 2305636022));
    assert!(c.div_rotation(r) == e);
}

// --- mul_translation

#[test]
#[inline(never)]
fn bench_unit_complex_mul_translation__baseline() {
    let _c = black_box(c());
    let _t = black_box(Translation2 { vector: v() });
    let e = black_box(iso2(9697103119, -6392026840, 3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_mul_translation__rotate() {
    let c = black_box(c());
    let t = black_box(Translation2 { vector: v() });
    let e = black_box(iso2(9697103119, -6392026840, 3955926847, 1672539044));
    assert!(c.mul_translation(t) == e);
}

// --- mul_isometry

#[test]
#[inline(never)]
fn bench_unit_complex_mul_isometry__baseline() {
    let _c = black_box(c());
    let _i = black_box(iso());
    let e = black_box(iso2(9697103119, -6392026840, 4178578243, 993090093));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_mul_isometry__compose() {
    let c = black_box(c());
    let i = black_box(iso());
    let e = black_box(iso2(9697103119, -6392026840, 4178578243, 993090093));
    assert!(c.mul_isometry(i) == e);
}

// --- mul_similarity

#[test]
#[inline(never)]
fn bench_unit_complex_mul_similarity__baseline() {
    let _c = black_box(c());
    let _s = black_box(sim());
    let e = black_box(sim2(9697103119, -6392026840, 4178578243, 993090093, 8589934592));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_mul_similarity__compose() {
    let c = black_box(c());
    let s = black_box(sim());
    let e = black_box(sim2(9697103119, -6392026840, 4178578243, 993090093, 8589934592));
    assert!(c.mul_similarity(s) == e);
}

// --- from_complex

#[test]
#[inline(never)]
fn bench_unit_complex_from_complex__baseline() {
    let _q = black_box(v());
    let e = black_box(uc(2382419202, -3573628803));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_complex__normalize() {
    let q = black_box(v());
    let e = black_box(uc(2382419202, -3573628803));
    assert!(UnitComplexTrait::from_complex(q) == e);
}

// --- from_complex_and_get

#[test]
#[inline(never)]
fn bench_unit_complex_from_complex_and_get__baseline() {
    let _q = black_box(v());
    let e = black_box((uc(2382419202, -3573628803), fx(11614293609)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_complex_and_get__normalize() {
    let q = black_box(v());
    let e = black_box((uc(2382419202, -3573628803), fx(11614293609)));
    assert!(UnitComplexTrait::from_complex_and_get(q) == e);
}

// --- from_basis_unchecked

#[test]
#[inline(never)]
fn bench_unit_complex_from_basis_unchecked__baseline() {
    let _bs = black_box(basis());
    let e = black_box(uc(3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_basis_unchecked__first_column() {
    let bs = black_box(basis());
    let e = black_box(uc(3955926847, 1672539044));
    assert!(UnitComplexTrait::from_basis_unchecked(bs) == e);
}

// --- rotation_between_axis

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between_axis__baseline() {
    let _u = black_box(u2a());
    let _w = black_box(u2b());
    let e = black_box(uc(0, 4294967296));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_rotation_between_axis__algebraic() {
    let u = black_box(u2a());
    let w = black_box(u2b());
    let e = black_box(uc(0, 4294967296));
    assert!(UnitComplexTrait::rotation_between_axis(u, w) == e);
}

// --- scaled_rotation_between_axis

#[test]
#[inline(never)]
fn bench_unit_complex_scaled_rotation_between_axis__baseline() {
    let _u = black_box(u2a());
    let _w = black_box(u2b());
    let _s = black_box(fx(0x80000000));
    let e = black_box(uc(3037000500, 3037000500));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_scaled_rotation_between_axis__atan2() {
    let u = black_box(u2a());
    let w = black_box(u2b());
    let s = black_box(fx(0x80000000));
    let e = black_box(uc(3037000500, 3037000500));
    assert!(UnitComplexAngleTrait::scaled_rotation_between_axis(u, w, s) == e);
}

// --- from_matrix

#[test]
#[inline(never)]
fn bench_unit_complex_from_matrix__baseline() {
    let _m = black_box(mat());
    let e = black_box(uc(3987777022, 1595110809));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_matrix__closed_form() {
    let m = black_box(mat());
    let e = black_box(uc(3987777022, 1595110809));
    assert!(UnitComplexAngleTrait::from_matrix(m) == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_matrix__alt_iterate() {
    let m = black_box(mat());
    let e = black_box(uc(3987777021, 1595110808));
    assert!(
        UnitComplexAngleTrait::from_matrix_eps(m, fx(32), 16, UnitComplexTrait::identity()) == e,
    );
}

// --- from_matrix_eps

#[test]
#[inline(never)]
fn bench_unit_complex_from_matrix_eps__baseline() {
    let _m = black_box(mat());
    let e = black_box(uc(3987777021, 1595110808));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_matrix_eps__muller_4() {
    let m = black_box(mat());
    let e = black_box(uc(3987777021, 1595110808));
    assert!(
        UnitComplexAngleTrait::from_matrix_eps(m, fx(32), 4, UnitComplexTrait::identity()) == e,
    );
}

// --- from_scaled_axis

#[test]
#[inline(never)]
fn bench_unit_complex_from_scaled_axis__baseline() {
    let _x = black_box(fx(0x66666666));
    let e = black_box(uc(3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_from_scaled_axis__sin_cos() {
    let x = black_box(fx(0x66666666));
    let e = black_box(uc(3955926847, 1672539044));
    assert!(UnitComplexAngleTrait::from_scaled_axis(x) == e);
}

// --- scaled_axis

#[test]
#[inline(never)]
fn bench_unit_complex_scaled_axis__baseline() {
    let _c = black_box(c());
    let e = black_box(fx(1717986918));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_scaled_axis__atan2() {
    let c = black_box(c());
    let e = black_box(fx(1717986918));
    assert!(c.scaled_axis() == e);
}

// --- axis_angle

#[test]
#[inline(never)]
fn bench_unit_complex_axis_angle__baseline() {
    let _c = black_box(c());
    let e = black_box((fx(4294967296), fx(1717986918)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_axis_angle__atan2() {
    let c = black_box(c());
    let e = black_box((fx(4294967296), fx(1717986918)));
    assert!(c.axis_angle().unwrap() == e);
}

// --- transform_unit_vector

#[test]
#[inline(never)]
fn bench_unit_complex_transform_unit_vector__baseline() {
    let _c = black_box(c());
    let _u = black_box(u2a());
    let e = black_box(v2(1035524873, 4168264904));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_transform_unit_vector__transform_vector() {
    let c = black_box(c());
    let u = black_box(u2a());
    let e = black_box(v2(1035524873, 4168264904));
    assert!(c.transform_unit_vector(u).value == e);
}

// --- inverse_transform_unit_vector

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_unit_vector__baseline() {
    let _c = black_box(c());
    let _u = black_box(u2a());
    let e = black_box(v2(3711587343, 2161218051));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_inverse_transform_unit_vector__inverse_transform_vector() {
    let c = black_box(c());
    let u = black_box(u2a());
    let e = black_box(v2(3711587343, 2161218051));
    assert!(c.inverse_transform_unit_vector(u).value == e);
}

// --- cast

#[test]
#[inline(never)]
fn bench_unit_complex_cast__baseline() {
    let _c = black_box(c());
    let e = black_box(uc(3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_cast__into() {
    let c = black_box(c());
    let e = black_box(uc(3955926847, 1672539044));
    assert!({
        let r: UnitComplex<Fixed> = c.cast();
        r
    } == e);
}

// --- relative_eq

#[test]
#[inline(never)]
fn bench_unit_complex_relative_eq__baseline() {
    let _c = black_box(c());
    let _d = black_box(d());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_relative_eq__components() {
    let c = black_box(c());
    let d = black_box(d());
    let e = black_box(false);
    assert!(c.relative_eq(d, 4, fx(0x100)) == e);
}

// --- ulps_eq

#[test]
#[inline(never)]
fn bench_unit_complex_ulps_eq__baseline() {
    let _c = black_box(c());
    let _d = black_box(d());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_ulps_eq__components() {
    let c = black_box(c());
    let d = black_box(d());
    let e = black_box(false);
    assert!(c.ulps_eq(d, 4, 4) == e);
}

// --- default

#[test]
#[inline(never)]
fn bench_unit_complex_default__baseline() {
    let e = black_box(uc(4294967296, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_default__const() {
    let e = black_box(uc(4294967296, 0));
    assert!(Default::<UnitComplex<Fixed>>::default() == e);
}

// --- one

#[test]
#[inline(never)]
fn bench_unit_complex_one__baseline() {
    let e = black_box(uc(4294967296, 0));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_one__const() {
    let e = black_box(uc(4294967296, 0));
    assert!(One::<UnitComplex<Fixed>>::one() == e);
}

// --- into_isometry

#[test]
#[inline(never)]
fn bench_unit_complex_into_isometry__baseline() {
    let _c = black_box(c());
    let e = black_box(iso2(0, 0, 3955926847, 1672539044));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_into_isometry__struct() {
    let c = black_box(c());
    let e = black_box(iso2(0, 0, 3955926847, 1672539044));
    assert!({
        let r: Isometry2<Fixed> = c.into();
        r
    } == e);
}

// --- into_similarity

#[test]
#[inline(never)]
fn bench_unit_complex_into_similarity__baseline() {
    let _c = black_box(c());
    let e = black_box(sim2(0, 0, 3955926847, 1672539044, 4294967296));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_unit_complex_into_similarity__struct() {
    let c = black_box(c());
    let e = black_box(sim2(0, 0, 3955926847, 1672539044, 4294967296));
    assert!({
        let r: Similarity2<Fixed> = c.into();
        r
    } == e);
}
