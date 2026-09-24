//! Gas benchmarks of `Unit` (`bench_unit<n>_<op>__<variant>`, net = raw - `baseline` of the
//! group), and the alternative implementations that lost (`alt_*`), kept as evidence together
//! with the tests showing why (AGENTS.md rule 8).
//!
//! Expected values come from a bit-exact integer model of the Q32.32 kernels (floor rounding).
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_base/src/unit/benches.cairo`.

use fixed::Fixed;
use simba::scalar::Real;
use crate::base::matrix_test_utils::u3;
use crate::base::unit::UnitInternalTrait;
use crate::base::vector3::{Vector3, Vector3Trait};
use super::Unit;

// --- alternative implementations (losers)

/// `renormalize_fast` as upstream writes it: `v * (1/2 * (3 - |v|²))`, one product per component.
/// Bit-identical to `renormalize_fast` (whose `mul_add(s, -1/2, 3/2)` floors the same value as
/// `1/2 * (3 - s)`), only the gas differs: one subtraction and one rescaled product more.
#[inline(always)]
fn alt_renormalize_fast_upstream(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let three: Fixed = Real::TWO + Real::one();
    let f = Real::HALF * (three - v.norm_squared());
    Unit { value: Vector3 { x: v.x * f, y: v.y * f, z: v.z * f } }
}

/// `renormalize_fast` as `v + v * (1/2 * (1 - s))`: one fused `mul_add` per component (dearer than
/// one product per component).
#[inline(always)]
fn alt_renormalize_fast_mul_add(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let f = Real::HALF * (Real::one() - v.norm_squared());
    Unit {
        value: Vector3 {
            x: Real::mul_add(v.x, f, v.x),
            y: Real::mul_add(v.y, f, v.y),
            z: Real::mul_add(v.z, f, v.z),
        },
    }
}

/// `renormalize_fast` as `v * (1 + 1/2 * (1 - s))`: the same factor without the constant 3.
#[inline(always)]
fn alt_renormalize_fast_scale(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let f = Real::one() + Real::HALF * (Real::one() - v.norm_squared());
    Unit { value: Vector3 { x: v.x * f, y: v.y * f, z: v.z * f } }
}

/// `renormalize_fast` through `lerp`: `v + (0 - v) * -f` per component (an extra exact
/// difference per component).
#[inline(always)]
fn alt_renormalize_fast_lerp(u: Unit<Vector3<Fixed>>) -> Unit<Vector3<Fixed>> {
    let v = u.value;
    let f = Real::HALF * (Real::one() - v.norm_squared());
    Unit {
        value: Vector3 {
            x: Real::lerp(v.x, Real::zero(), -f),
            y: Real::lerp(v.y, Real::zero(), -f),
            z: Real::lerp(v.z, Real::zero(), -f),
        },
    }
}

// --- why the alternatives lost

#[test]
fn test_renormalize_fast_alts_are_bit_identical() {
    // `floor((3 - s) / 2) = 1 + floor((1 - s) / 2)` in raw units, and `v + floor(v * f)` floors
    // once like `floor(v * (1 + f))`: the variants only differ in gas.
    let mut units = array![
        u3(1393475576, -2090213367, 3483688943), u3(1393464429, -2090196645, 3483661074),
        u3(1394864867, -2092297303, 3487162170), u3(1189375559, -1585834079, 4757502235),
        u3(0, 0, 0), u3(-0x100000000, 5, 7),
    ]
        .span();
    while let Some(u) = units.pop_front() {
        let r = (*u).renormalized_fast();
        assert!(alt_renormalize_fast_upstream(*u) == r);
        assert!(alt_renormalize_fast_mul_add(*u) == r);
        assert!(alt_renormalize_fast_scale(*u) == r);
        assert!(alt_renormalize_fast_lerp(*u) == r);
    }
}
