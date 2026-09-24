//! Gas benchmarks of `Isometry2` (`bench_isometry2_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together with
//! the tests showing why (AGENTS.md rule 8).
//!
//! The isometries used are `a = new((1.5, -2.25), 0.4 rad)` and `b = new((-0.75, 0.5), -1/6 rad)`
//! (hardcoded, so that the `sin_cos` of their construction is not measured), the point / vector
//! `(-2.5, 3.75)`, the translation `(1.25, -0.375)` and the rotation `UnitComplex::new(0.1)`.
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `tests.cairo`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/isometry2/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{fx, iso2};
use crate::geometry::isometry2::Isometry2InternalTrait;
use super::Isometry2;

/// `new((1.5, -2.25), 0.4 rad)`.
fn a() -> Isometry2<Fixed> {
    iso2(6442450944, -9663676416, 3955926847, 1672539044)
}

/// `new((-0.75, 0.5), -1/6 rad)`.
fn b() -> Isometry2<Fixed> {
    iso2(-3221225472, 2147483648, 4235452929, -712518464)
}

#[test]
#[inline(never)]
fn bench_isometry2_renormalize__exact() {
    let x: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926850, 1672539042));
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926849, 1672539042));
    assert!(x.renormalize() == e);
}

#[test]
#[inline(never)]
fn bench_isometry2_renormalize_fast__newton() {
    let x: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926850, 1672539042));
    let e: Isometry2<Fixed> = black_box(iso2(6442450944, -9663676416, 3955926848, 1672539041));
    assert!(x.renormalize_fast() == e);
}

/// The trigonometry-free interpolation: same path, cheaper parametrisation.
#[test]
#[inline(never)]
fn bench_isometry2_lerp_nlerp__lerp_normalize() {
    let x: Isometry2<Fixed> = black_box(a());
    let y: Isometry2<Fixed> = black_box(b());
    let s: Fixed = black_box(fx(0x40000000));
    let e: Isometry2<Fixed> = black_box(iso2(4026531840, -6710886400, 4149247215, 1109275270));
    assert!(x.lerp_nlerp(y, s) == e);
}
