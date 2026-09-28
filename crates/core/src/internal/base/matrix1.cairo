//! Internal, no stability promise: the crate-private items of `base::matrix1` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::{Real, Transcendental};
use crate::base::errors;
use crate::base::matrix1::Matrix1;

/// `Unit::try_slerp` on the values `a`, `b` of two unit vectors (upstream `interpolation.rs`): `a`
/// when they are equal, `None` when `sin θ <= epsilon`, otherwise each component is ONE
/// `sum_prod2` of the weights `sin((1 - t) θ) / sin θ` and `sin(t θ) / sin θ`.
///
/// The angle comes from the Kahan half-angle form of `angle`: `θ = 2 atan2(|a - b|, |a + b|)` and
/// `sin θ = |a - b| |a + b| / 2` (unit vectors), two fused norms of exact differences / sums.
/// Upstream's `acos(a · b)` and `sqrt(1 - (a · b)²)` lose the last bit of `1 - c²` near `c = 1`
/// (a unit vector interpolated with itself came out √2 too long); that form is kept as a
/// benchmark (`bench_vector4_slerp__alt_acos`: 141 180 gas against 155 000 for this one).
pub fn slerp_unit<
    T,
    impl R: Real<T>,
    impl Tr: Transcendental<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
>(
    a: Matrix1<T>, b: Matrix1<T>, t: T, epsilon: T,
) -> Option<Matrix1<T>> {
    let d = R::abs(a.x - b.x);
    if d == R::zero() {
        return Option::Some(a);
    }
    let s = R::abs(a.x + b.x);
    let half = Tr::atan2(d, s);
    let hang = half + half;
    let shang = (d * s) * R::HALF;
    if shang <= epsilon {
        return Option::None;
    }
    let ta = R::div(Tr::sin((R::one() - t) * hang), shang);
    let tb = R::div(Tr::sin(t * hang), shang);
    Option::Some(Matrix1 { x: R::sum_prod2(a.x, ta, b.x, tb) })
}

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl Matrix1EditImpl<T, +Copy<T>, +Drop<T>> of Matrix1EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: Matrix1<T>, index: (usize, usize), v: T) -> Matrix1<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => Matrix1 { x: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: Matrix1<T>, i: usize) -> Matrix1<T> {
        match i {
            0 => Matrix1 { x: self.x },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: Matrix1<T>, j: usize) -> Matrix1<T> {
        match j {
            0 => Matrix1 { x: self.x },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
