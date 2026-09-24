//! Gas benchmarks of `Quaternion` (`bench_quaternion_<op>__<variant>`, net = raw - `baseline` of
//! the group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/quaternion/benches.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{ONE_RAW, q};
use crate::geometry::quaternion::QuaternionInternalTrait;
use super::Quaternion;

/// `1 + 2i - 3j + 4k`, of squared norm 30 (norm 5.477).
fn a() -> Quaternion<Fixed> {
    q(ONE_RAW, 2 * ONE_RAW, -3 * ONE_RAW, 4 * ONE_RAW)
}

/// `-2 + i + 5j - k`, of squared norm 31.
fn b() -> Quaternion<Fixed> {
    q(-2 * ONE_RAW, ONE_RAW, 5 * ONE_RAW, -ONE_RAW)
}

#[test]
#[inline(never)]
fn bench_quaternion_conj_mul__fused() {
    let x = black_box(a());
    let y = black_box(b());
    let e = black_box(q(-19 * ONE_RAW, 22 * ONE_RAW, -7 * ONE_RAW, -6 * ONE_RAW));
    assert!(x.conj_mul(y) == e);
}
