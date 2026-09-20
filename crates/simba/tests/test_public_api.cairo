//! The public API as a downstream crate sees it: only `simba::` paths, no `crate::` access.

use simba::fixed::types::{HALF, ONE, PI, TWO};
use simba::fixed::{Fixed, WideTrait};
use simba::scalar::Real;

/// Operators and conversions resolve with `Fixed` alone in scope (their impls are re-exported by
/// the module of the type); methods need `Real`.
#[test]
fn test_api_operators_need_no_impl_import() {
    let three: Fixed = 3_u8.into();
    let x = (three * HALF + ONE - TWO) / TWO; // 0.25
    assert!(x == Fixed { raw: 0x40000000 });
    assert!(-x < x);
    assert!(three % TWO == ONE);
    let mut y = x;
    y += x;
    assert!(y == HALF);
    let big: Option<Fixed> = 0x80000000_u32.try_into();
    assert!(big.is_none());
    assert!(x.recip() == Fixed { raw: 0x400000000 });
    assert!(PI.floor().to_int() == 3);
}

#[test]
fn test_api_wide_accumulator() {
    let mut w = WideTrait::zero();
    for i in 1..7_i32 {
        let x = Real::<Fixed>::from_int(i);
        w = w.add_prod(x, x);
    }
    assert!(w.rescale() == Real::from_int(91));
}

mod with_prelude {
    use simba::prelude::*;

    #[test]
    fn test_api_prelude_is_enough() {
        let a = Real::<Fixed>::from_ratio(3, 2);
        let b: Fixed = (-2_i8).into();
        assert!(Real::mul_add(a, b, Real::ONE) == Real::from_int(-2));
        assert!(Real::norm2(Real::from_int(3), Real::from_int(-4)) == Real::<Fixed>::from_int(5));
        assert!(WideTrait::from_prod(a, a).sqrt() == a);
        assert!(format!("{}", a) == "1.5000000000");
    }

    #[test]
    #[should_panic(expected: 'simba: overflow')]
    fn test_api_stable_panic_message() {
        let _ = nalgebra_testing::black_box(Real::<Fixed>::MAX) + Real::EPSILON;
    }

    /// `Transcendental` gives the method syntax and is usable next to `Real`.
    #[test]
    fn test_api_transcendental_methods() {
        let quarter_turn = Real::<Fixed>::FRAC_PI_2;
        assert!(quarter_turn.sin() == Real::ONE);
        assert!(quarter_turn.cos() == Real::ZERO);
        assert!(quarter_turn.sin_cos() == (Real::<Fixed>::ONE, Real::ZERO));
        assert!(Real::<Fixed>::ZERO.tan() == Real::ZERO);
        assert!(Real::<Fixed>::ONE.asin() == quarter_turn);
        assert!(Real::<Fixed>::ONE.acos() == Real::ZERO);
        assert!(Real::<Fixed>::ONE.atan() == Real::FRAC_PI_4);
        assert!(Transcendental::atan2(Real::<Fixed>::ONE, Real::ZERO) == quarter_turn);
        assert!(Real::<Fixed>::ZERO.exp() == Real::ONE);
        assert!(Real::<Fixed>::ONE.ln() == Real::ZERO);
        assert!(Real::<Fixed>::ONE.exp() == Real::E);
    }

    #[test]
    #[should_panic(expected: 'simba: out of domain')]
    fn test_api_transcendental_domain_panic() {
        let _ = nalgebra_testing::black_box(Real::<Fixed>::TWO).acos();
    }
}
