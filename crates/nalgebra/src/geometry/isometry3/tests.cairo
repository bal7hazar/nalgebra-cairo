//! Unit tests of `Isometry3`: exact cases (identity, pure translations, half turns, the observer
//! frames), the identities a rigid-body transform must satisfy (`iso · iso⁻¹ = id`,
//! `inv_mul(a, b) = a⁻¹ · b`, `to_homogeneous` acts like `transform_point`, append / prepend
//! against the composition), and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on
//! the same raw inputs).
//!
//! Stated tolerances: a pure translation is carried BIT FOR BIT (`floor(x + t) = floor(x) + t` for
//! an integral `t` in raw units); everything that goes through the quaternion is floored once per
//! component and inherits the tolerance of the oracle (`ceil(2 + 3·A)`, the sensitivity policy of
//! `tools/oracle`, which for the `large` distribution reaches a few 10^5 ulp because the inputs
//! themselves are of the order of 10^4). `inv_mul(a, b)` and `a.inverse() * b` agree within a few
//! ulp but not bit for bit (the second rounds one more intermediate): see `benches.cairo`.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 5 cases per distribution, the
//! whole `isometry3` suite) with
//! `cargo run --release -- emit-cairo isometry3 --from vectors --max-per-dist 5 --out
//! <oracle.cairo>`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/isometry3/tests.cairo`.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, int, v3t};
use crate::geometry::isometry3::Isometry3InternalTrait;
use crate::geometry::quaternion::Quaternion;
use crate::geometry::unit_quaternion::{UnitQuaternion, UnitQuaternionTrait};
use super::{Isometry3, Isometry3AngleTrait, Isometry3Trait};

/// `new((1.5, -2.25, 3.75), (0.25, -0.1875, 0.125))`.
fn a() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3t((0x180000000, -0x240000000, 0x3c0000000)), v3t((0x40000000, -0x30000000, 0x20000000)),
    )
}

/// `new((-0.75, 0.5, 1.25), (-0.5, 0.375, 0.875))`.
fn b() -> Isometry3<Fixed> {
    Isometry3AngleTrait::new(
        v3t((-0xc0000000, 0x80000000, 0x140000000)), v3t((-0x80000000, 0x60000000, 0xe0000000)),
    )
}

// --- renormalisation, comparison, interpolation

#[test]
fn test_renormalize_restores_the_rotation_and_keeps_the_translation() {
    let q = a().rotation;
    let drifted = Isometry3Trait::from_parts(
        a().translation,
        UnitQuaternion {
            quaternion: Quaternion {
                i: q.quaternion.i * int(4),
                j: q.quaternion.j * int(4),
                k: q.quaternion.k * int(4),
                w: q.quaternion.w * int(4),
            },
        },
    );
    let r = drifted.renormalize();
    assert!(r.translation == drifted.translation);
    assert!(r.rotation.abs_diff_eq(q, 4));
    // `renormalize_fast` is a single Newton step: valid near 1, so apply it to a small drift.
    let small = Isometry3Trait::from_parts(
        a().translation,
        UnitQuaternion {
            quaternion: Quaternion {
                i: q.quaternion.i + fx(3),
                j: q.quaternion.j - fx(2),
                k: q.quaternion.k,
                w: q.quaternion.w,
            },
        },
    );
    assert!(small.renormalize_fast().rotation.abs_diff_eq(small.renormalize().rotation, 4));
    assert!(small.renormalize_fast().translation == small.translation);
}

#[test]
fn test_lerp_nlerp_endpoints_and_midpoint() {
    let (x, y) = (a(), b());
    assert!(x.lerp_nlerp(y, Real::zero()).abs_diff_eq(x, 4));
    assert!(x.lerp_nlerp(y, Real::one()).abs_diff_eq(y, 4));
    let h = x.lerp_nlerp(y, Real::HALF);
    assert!(
        h
            .translation
            .vector
            .x == Real::lerp(x.translation.vector.x, y.translation.vector.x, Real::HALF),
    );
    let q = h.rotation.quaternion;
    assert!(Real::abs_diff_eq(Real::norm_squared4(q.i, q.j, q.k, q.w), Real::one(), 4));
}

#[test]
fn test_lerp_slerp_endpoints_and_agreement_with_lerp_nlerp() {
    let (x, y) = (a(), b());
    assert!(x.lerp_slerp(y, Real::zero()).abs_diff_eq(x, 4));
    assert!(x.lerp_slerp(y, Real::one()).abs_diff_eq(y, 16));
    let (s, n) = (x.lerp_slerp(y, Real::HALF), x.lerp_nlerp(y, Real::HALF));
    assert!(s.translation == n.translation);
    // Same path, different parametrisation: at the midpoint both are the half-way rotation.
    assert!(s.rotation.abs_diff_eq(n.rotation, 4096));
}
