//! Internal, no stability promise: the crate-private items of `geometry::unit_complex` that the
//! packages above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix2::Matrix2;
use simba::scalar::{Real, Transcendental};
use crate::geometry::unit_complex::{FROM_MATRIX_MAX_ITER, UnitComplex, UnitComplexTrait};

/// Crate-internal by-value forms of the in-place `renormalize` / `renormalize_fast` (WP 8.0: the
/// public methods are upstream's `&mut self` ones), for the tests and the value-style call sites.
#[generate_trait]
pub impl UnitComplexInternalImpl<
    T, impl R: Real<T>, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of UnitComplexInternalTrait<T> {
    /// `self` renormalized exactly (`UnitComplexTrait::renormalize`), by value.
    #[inline(always)]
    fn renormalized(self: UnitComplex<T>) -> UnitComplex<T> {
        let mut r = self;
        let _ = UnitComplexTrait::renormalize(ref r);
        r
    }

    /// `self` renormalized by one Newton step (`UnitComplexTrait::renormalize_fast`), by value.
    #[inline(always)]
    fn renormalized_fast(self: UnitComplex<T>) -> UnitComplex<T> {
        let mut r = self;
        UnitComplexTrait::renormalize_fast(ref r);
        r
    }
}

/// Crate-internal kernel of `UnitComplexAngleTrait::from_matrix_eps`.
#[generate_trait]
pub impl UnitComplexAngleInternalImpl<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +Copy<T>,
    +Drop<T>,
> of UnitComplexAngleInternalTrait<T> {
    /// `from_matrix_eps` and the number of iterations it ran (for the convergence tests).
    fn from_matrix_eps_count(
        m: Matrix2<T>, eps: T, max_iter: usize, guess: UnitComplex<T>,
    ) -> (UnitComplex<T>, usize) {
        let cap = if max_iter == 0 || max_iter > FROM_MATRIX_MAX_ITER {
            FROM_MATRIX_MAX_ITER
        } else {
            max_iter
        };
        let mut r = guess;
        let mut iter: usize = 0;
        while iter < cap {
            iter += 1;
            // With columns (re, im) and (-im, re), a = m21 - m12 and b = m11 + m22 (exact):
            // axis = re·a - im·b, denom = re·b + im·a, one fused kernel each.
            let (a, b) = (m.m21 - m.m12, m.m11 + m.m22);
            let axis = R::diff_prod(r.re, a, r.im, b);
            let denom = R::sum_prod2(r.re, b, r.im, a);
            let angle = R::div(axis, R::abs(denom) + R::default_epsilon());
            // `|angle| <= eps`, with `Real` comparisons only (the trait carries no `PartialOrd`).
            if R::max(R::abs(angle), eps) == eps {
                break;
            }
            let (sin, cos) = Tr::sin_cos(angle);
            r =
                UnitComplex {
                    re: R::diff_prod(cos, r.re, sin, r.im), im: R::sum_prod2(cos, r.im, sin, r.re),
                };
        }
        (r, iter)
    }
}
