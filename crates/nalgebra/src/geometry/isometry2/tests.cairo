//! Unit tests of `Isometry2`: exact cases (identity, pure translations, quarter turns), the
//! identities a rigid-body transform must satisfy (`iso · iso⁻¹ = id`, `inv_mul(a, b) =
//! a⁻¹ · b`, `to_homogeneous` acts like `transform_point`, append / prepend against the
//! composition), and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs).
//!
//! Stated tolerances: the translation part of a pure translation is carried BIT FOR BIT
//! (`floor(x + t) = floor(x) + t` for an integral `t` in raw units); everything that goes through
//! a rotation is floored once per component, which the oracle bounds at 2 ulp for the whole
//! `isometry2` suite. `inv_mul(a, b)` and `a.inverse() * b` agree within 2 ulp but not bit for bit
//! (the second rounds one more intermediate): see `benches.cairo`.
//!
//! `oracle.cairo` is emitted from `tools/oracle` (committed vectors, 6 cases per distribution, the
//! whole `isometry2` suite) with
//! `cargo run --release -- emit-cairo isometry2 --from vectors --max-per-dist 6 --out
//! <oracle.cairo>`.
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/isometry2/tests.cairo`.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, v2t};
use crate::geometry::isometry2::Isometry2InternalTrait;
use crate::geometry::translation2::Translation2Trait;
use crate::geometry::unit_complex::{UnitComplex, UnitComplexAngleTrait, UnitComplexTrait};
use super::{Isometry2, Isometry2AngleTrait, Isometry2Trait};

/// `new((1.5, -2.25), 0.4 rad)`.
fn a() -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(v2t((0x180000000, -0x240000000)), fx(0x66666666))
}

/// `new((-0.75, 0.5), -1/6 rad)`.
fn b() -> Isometry2<Fixed> {
    Isometry2AngleTrait::new(v2t((-0xc0000000, 0x80000000)), fx(-0x2aaaaaaa))
}

// --- renormalisation, comparison, interpolation

#[test]
fn test_renormalize_restores_the_rotation_and_keeps_the_translation() {
    // A rotation scaled by 4 (exact): both renormalisations bring it back, the translation is
    // untouched.
    let c = UnitComplexAngleTrait::<Fixed>::new(fx(0x1f0a3d70a));
    let four = Real::<Fixed>::from_int(4);
    let drifted = Isometry2Trait::from_parts(
        Translation2Trait::new(fx(0x180000000), fx(-0x240000000)),
        UnitComplex { re: c.re * four, im: c.im * four },
    );
    let r = drifted.renormalize();
    assert!(r.translation == drifted.translation);
    assert!(r.rotation.abs_diff_eq(c, 2));
    // `renormalize_fast` is a single Newton step: valid near 1, so apply it to a small drift.
    let small = Isometry2Trait::from_parts(
        drifted.translation, UnitComplex { re: c.re + fx(3), im: c.im - fx(2) },
    );
    assert!(small.renormalize_fast().rotation.abs_diff_eq(small.renormalize().rotation, 2));
    assert!(small.renormalize_fast().translation == small.translation);
}

#[test]
fn test_lerp_nlerp_endpoints_and_midpoint() {
    let (x, y) = (a(), b());
    // `t = 0` and `t = 1` give the endpoints back, up to the renormalisation of the rotation.
    assert!(x.lerp_nlerp(y, Real::zero()).abs_diff_eq(x, 2));
    assert!(x.lerp_nlerp(y, Real::one()).abs_diff_eq(y, 2));
    // The translation is the exact `lerp`, whatever the rotation does.
    let h = x.lerp_nlerp(y, Real::HALF);
    assert!(
        h
            .translation
            .vector
            .x == Real::lerp(x.translation.vector.x, y.translation.vector.x, Real::HALF),
    );
    // The interpolated rotation is unit and lies between the two (half the angle to `y`).
    assert!(Real::abs_diff_eq(Real::norm_squared2(h.rotation.re, h.rotation.im), Real::one(), 2));
    let half_angle = x.rotation.angle_to(y.rotation) * Real::HALF;
    assert!(Real::abs_diff_eq(x.rotation.angle_to(h.rotation), half_angle, 1048576));
}

#[test]
fn test_lerp_slerp_endpoints_and_agreement_with_lerp_nlerp() {
    let (x, y) = (a(), b());
    assert!(x.lerp_slerp(y, Real::zero()).abs_diff_eq(x, 2));
    assert!(x.lerp_slerp(y, Real::one()).abs_diff_eq(y, 8));
    // The two interpolations differ by the parametrisation only: the angle to `x` is the same
    // at the midpoint (both walk half the arc there).
    let (s, n) = (x.lerp_slerp(y, Real::HALF), x.lerp_nlerp(y, Real::HALF));
    assert!(s.translation == n.translation);
    assert!(s.rotation.abs_diff_eq(n.rotation, 65536));
}
