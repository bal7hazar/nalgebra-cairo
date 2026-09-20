//! `simba::scalar::Transcendental` for glam.cairo's `fixed::Fixed`.

use fixed::Fixed as Glam;
use fixed::trig::TrigTrait;
use simba::fixed::transcendental as stranscendental;
use simba::scalar::Transcendental;
use crate::convert::{from_simba, to_simba};

/// `Transcendental` for glam.cairo's Q32.32 scalar.
///
/// **Split on purpose** (the only place in this package where the implementation is not simba's):
///
/// - `sin`, `cos`, `sin_cos`, `tan`, `asin`, `acos`, `atan`, `atan2` forward to glam.cairo's
///   `fixed::trig::TrigTrait`, so a glam.cairo / rapier.cairo user gets from nalgebra exactly the
///   angles their own code already produces. A rotation built by `glam::Quat` and one built by
///   `nalgebra::UnitQuaternion` then agree bit for bit, which matters far more downstream than
///   agreeing with simba.
/// - `exp` and `ln` forward to `simba::fixed::transcendental`: glam.cairo has no exponential or
///   logarithm at all, so there is nothing to be consistent with.
///
/// Consequence, measured case by case in `crate::conformance`: the trigonometric results may
/// differ from `simba::scalar::Transcendental<simba::fixed::Fixed>` by a few ulp (both
/// implementations are generated minimax polynomials, but not the same ones, and glam's `asin` /
/// `acos` clamp where simba's panic). Both stay inside the oracle tolerances. Code that needs
/// bit-identical trigonometry across the two scalars must call one of them explicitly.
pub impl FixedTranscendental of Transcendental<Glam> {
    #[inline(always)]
    fn sin(self: Glam) -> Glam {
        TrigTrait::sin(self)
    }
    #[inline(always)]
    fn cos(self: Glam) -> Glam {
        TrigTrait::cos(self)
    }
    #[inline(always)]
    fn sin_cos(self: Glam) -> (Glam, Glam) {
        TrigTrait::sin_cos(self)
    }
    #[inline(always)]
    fn tan(self: Glam) -> Glam {
        TrigTrait::tan(self)
    }
    #[inline(always)]
    fn asin(self: Glam) -> Glam {
        TrigTrait::asin(self)
    }
    #[inline(always)]
    fn acos(self: Glam) -> Glam {
        TrigTrait::acos(self)
    }
    #[inline(always)]
    fn atan(self: Glam) -> Glam {
        TrigTrait::atan(self)
    }
    #[inline(always)]
    fn atan2(y: Glam, x: Glam) -> Glam {
        TrigTrait::atan2(y, x)
    }
    #[inline(always)]
    fn exp(self: Glam) -> Glam {
        from_simba(stranscendental::exp(to_simba(self)))
    }
    #[inline(always)]
    fn ln(self: Glam) -> Glam {
        from_simba(stranscendental::ln(to_simba(self)))
    }
}
