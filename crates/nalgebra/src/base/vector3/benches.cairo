//! Gas benchmarks of `Vector3` (`bench_vector3_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/vector3/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::v3;
use super::{Vector3, Vector3InternalTrait};

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zpos__duff() {
    let u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, 3483678492));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(4045339972, 374440987, -1393471396), v3(374440987, 3733305816, 2090207096)),
    );
    assert!(u.orthonormal_basis() == e);
}

#[test]
#[inline(never)]
fn bench_vector3_orthonormal_basis_zneg__duff() {
    let u: Vector3<Fixed> = black_box(v3(1393471396, -2090207096, -3483678493));
    let e: (Vector3<Fixed>, Vector3<Fixed>) = black_box(
        (v3(4045339972, 374440987, 1393471396), v3(-374440987, -3733305816, 2090207096)),
    );
    assert!(u.orthonormal_basis() == e);
}
