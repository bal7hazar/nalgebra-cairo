//! Unit tests of `Quaternion`: the argument / storage conventions, exact integer cases, the
//! algebra identities (`i·j = k`, `q · q⁻¹ = 1`), panics, and the oracle vectors of
//! `tools/oracle`
//! (upstream nalgebra 0.35 on the same raw inputs, tolerance in ulp).
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 4 cases per distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo quaternion --from vectors --max-per-dist 4 \
//!     --ops quaternion_mul,quaternion_conjugate,quaternion_norm,quaternion_normalize,\
//! quaternion_try_inverse --out crates/nalgebra/src/geometry/quaternion/oracle.cairo
//! ```
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/quaternion/tests.cairo`.

use fixed::Fixed;
use nalgebra_testing::black_box;
use crate::base::matrix_test_utils::{ONE_RAW, qi, qt};
use crate::geometry::quaternion::QuaternionInternalTrait;
use super::{Quaternion, QuaternionTrait, oracle};

const MIN: i64 = -0x8000000000000000;

/// `1 + 2i - 3j + 4k`, of squared norm 30.
fn a() -> Quaternion<Fixed> {
    qi(1, 2, -3, 4)
}

/// `-2 + i + 5j - k`, of squared norm 31.
fn b() -> Quaternion<Fixed> {
    qi(-2, 1, 5, -1)
}

// --- conj_mul

#[test]
fn test_conj_mul_exact_integer_case() {
    assert!(a().conj_mul(b()) == qi(-19, 22, -7, -6));
    assert!(a().conj_mul(b()) == a().conjugate() * b());
    assert!(b().conj_mul(a()) == b().conjugate() * a());
}

/// `conj(q) · q = |q|²`, a real quaternion, exactly for integer components.
#[test]
fn test_conj_mul_self_is_the_squared_norm() {
    assert!(a().conj_mul(a()) == qi(30, 0, 0, 0));
    assert!(b().conj_mul(b()) == qi(31, 0, 0, 0));
    let one = QuaternionTrait::<Fixed>::identity();
    assert!(one.conj_mul(a()) == a());
    assert!(a().conj_mul(one) == a().conjugate());
}

/// Bit-identical to `conjugate() * other` on every oracle input of the Hamilton product, in both
/// orders.
#[test]
fn test_conj_mul_oracle_matches_conjugate_then_mul() {
    let mut cases = oracle::quaternion_mul_cases();
    while let Some(case) = cases.pop_front() {
        let (ra, rb, _expected, _tol) = *case;
        assert!(qt(ra).conj_mul(qt(rb)) == qt(ra).conjugate() * qt(rb));
        assert!(qt(rb).conj_mul(qt(ra)) == qt(rb).conjugate() * qt(ra));
    }
}

/// The one behavioural difference: a component equal to `MIN` is never negated, so it does not
/// panic (`conjugate()` does, see `test_neg_min_panics`). `-(MIN · 2^-32)` floors to `2^31` raw.
#[test]
fn test_conj_mul_min_component_does_not_panic() {
    let m = black_box(qt((0, MIN, 0, 0)));
    let r = black_box(qt((1, 0, 0, 0)));
    assert!(m.conj_mul(r) == qt((0, 0x80000000, 0, 0)));
}

#[test]
#[should_panic(expected: 'Fixed: overflow')]
fn test_conj_mul_overflow_panics() {
    // 50 000² > 2^31: the real part does not fit.
    let big = qt((50000 * ONE_RAW, 0, 0, 0));
    let _ = black_box(big).conj_mul(black_box(big));
}
