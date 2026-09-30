//! Internal, no stability promise: the crate-private items of `base::vector5` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::errors;
use nalgebra_core::base::matrix1::Matrix1;
use simba::scalar::{Real, Transcendental};
use crate::base::vector5::Vector5;

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
    a: Vector5<T>, b: Vector5<T>, t: T, epsilon: T,
) -> Option<Vector5<T>> {
    let d = {
        let w = R::wide_add_prod(R::wide_zero(), a.x - b.x, a.x - b.x);
        let w = R::wide_add_prod(w, a.y - b.y, a.y - b.y);
        let w = R::wide_add_prod(w, a.z - b.z, a.z - b.z);
        let w = R::wide_add_prod(w, a.w - b.w, a.w - b.w);
        R::wide_sqrt(R::wide_add_prod(w, a.a - b.a, a.a - b.a))
    };
    if d == R::zero() {
        return Option::Some(a);
    }
    let s = {
        let w = R::wide_add_prod(R::wide_zero(), a.x + b.x, a.x + b.x);
        let w = R::wide_add_prod(w, a.y + b.y, a.y + b.y);
        let w = R::wide_add_prod(w, a.z + b.z, a.z + b.z);
        let w = R::wide_add_prod(w, a.w + b.w, a.w + b.w);
        R::wide_sqrt(R::wide_add_prod(w, a.a + b.a, a.a + b.a))
    };
    let half = Tr::atan2(d, s);
    let hang = half + half;
    let shang = (d * s) * R::HALF;
    if shang <= epsilon {
        return Option::None;
    }
    let ta = R::div(Tr::sin((R::one() - t) * hang), shang);
    let tb = R::div(Tr::sin(t * hang), shang);
    Option::Some(
        Vector5 {
            x: R::sum_prod2(a.x, ta, b.x, tb),
            y: R::sum_prod2(a.y, ta, b.y, tb),
            z: R::sum_prod2(a.z, ta, b.z, tb),
            w: R::sum_prod2(a.w, ta, b.w, tb),
            a: R::sum_prod2(a.a, ta, b.a, tb),
        },
    )
}

/// Private helpers of the `swap*` methods (runtime positions: one `match` each).
#[generate_trait]
pub impl Vector5EditImpl<T, +Copy<T>, +Drop<T>> of Vector5EditTrait<T> {
    /// `self` with the component at `index` (`(row, column)`) replaced by `v`; panics out of
    /// bounds.
    #[inline(always)]
    fn replace(self: Vector5<T>, index: (usize, usize), v: T) -> Vector5<T> {
        let (i, j) = index;
        match j {
            0 => match i {
                0 => Vector5 { x: v, y: self.y, z: self.z, w: self.w, a: self.a },
                1 => Vector5 { x: self.x, y: v, z: self.z, w: self.w, a: self.a },
                2 => Vector5 { x: self.x, y: self.y, z: v, w: self.w, a: self.a },
                3 => Vector5 { x: self.x, y: self.y, z: self.z, w: v, a: self.a },
                4 => Vector5 { x: self.x, y: self.y, z: self.z, w: self.w, a: v },
                _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
            },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Row `i`, a `Matrix1`; panics out of bounds.
    #[inline(always)]
    fn row_at(self: Vector5<T>, i: usize) -> Matrix1<T> {
        match i {
            0 => Matrix1 { x: self.x },
            1 => Matrix1 { x: self.y },
            2 => Matrix1 { x: self.z },
            3 => Matrix1 { x: self.w },
            4 => Matrix1 { x: self.a },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }

    /// Column `j`, a `Vector5`; panics out of bounds.
    #[inline(always)]
    fn column_at(self: Vector5<T>, j: usize) -> Vector5<T> {
        match j {
            0 => Vector5 { x: self.x, y: self.y, z: self.z, w: self.w, a: self.a },
            _ => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
        }
    }
}
