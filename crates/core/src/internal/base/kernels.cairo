//! Internal, no stability promise: the crate-private items of `base::kernels` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::Real;

/// Sums of 5 and 6 products with one rounding.
#[generate_trait]
pub impl Fused<T, impl R: Real<T>, +Drop<T>, +Drop<R::Wide>> of FusedTrait<T> {
    /// `a0 * b0 + .. + a4 * b4`: the exact sum in `Real::Wide`, floored once.
    #[inline(always)]
    fn sum_prod5(a0: T, b0: T, a1: T, b1: T, a2: T, b2: T, a3: T, b3: T, a4: T, b4: T) -> T {
        R::wide_rescale(
            R::wide_add_prod(
                R::wide_add_prod(
                    R::wide_add_prod(
                        R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a0, b0), a1, b1), a2, b2,
                    ),
                    a3,
                    b3,
                ),
                a4,
                b4,
            ),
        )
    }

    /// `a0 * b0 + .. + a5 * b5`: the exact sum in `Real::Wide`, floored once.
    #[inline(always)]
    fn sum_prod6(
        a0: T, b0: T, a1: T, b1: T, a2: T, b2: T, a3: T, b3: T, a4: T, b4: T, a5: T, b5: T,
    ) -> T {
        R::wide_rescale(
            R::wide_add_prod(
                R::wide_add_prod(
                    R::wide_add_prod(
                        R::wide_add_prod(
                            R::wide_add_prod(R::wide_add_prod(R::wide_zero(), a0, b0), a1, b1),
                            a2,
                            b2,
                        ),
                        a3,
                        b3,
                    ),
                    a4,
                    b4,
                ),
                a5,
                b5,
            ),
        )
    }
}

/// `x^p` for a runtime exponent (`lp_norm`): exponentiation by squaring, each product floored.
#[generate_trait]
pub impl Powi<T, impl R: Real<T>, +Mul<T>, +Copy<T>, +Drop<T>> of PowiTrait<T> {
    /// `x^p`, `x^0 = 1`: `⌊log2 p⌋` squarings and one product per set bit of `p` (the loop runs
    /// on the bits of the runtime `p`), each floored. Panics on overflow.
    fn powi(x: T, p: u32) -> T {
        let mut acc = R::one();
        let mut base = x;
        let mut e = p;
        while e != 0 {
            let (q, r) = DivRem::div_rem(e, 2);
            if r == 1 {
                acc = acc * base;
            }
            e = q;
            if e != 0 {
                base = base * base;
            }
        }
        acc
    }
}
