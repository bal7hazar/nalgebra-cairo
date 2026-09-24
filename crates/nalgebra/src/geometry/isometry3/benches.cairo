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
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/isometry3/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{fx, iso3};
use crate::geometry::isometry3::Isometry3InternalTrait;
use super::Isometry3;

/// `new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))`.
fn a() -> Isometry3<Fixed> {
    iso3((6442450944, -9663676416, 16106127360), (4234293283, 534340439, -400755330, 267170219))
}

/// `new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))`.
fn b() -> Isometry3<Fixed> {
    iso3((-3221225472, 2147483648, 5368709120), (3689020097, -1022754606, 767065954, 1789820560))
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
