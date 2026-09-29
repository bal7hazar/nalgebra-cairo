//! Internal, no stability promise: the crate-private items of `base::vector6` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use simba::scalar::{Real, Transcendental};
use crate::base::vector6::Vector6;

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
    a: Vector6<T>, b: Vector6<T>, t: T, epsilon: T,
) -> Option<Vector6<T>> {
    let d = {
        let w = R::wide_add_prod(R::wide_zero(), a.x - b.x, a.x - b.x);
        let w = R::wide_add_prod(w, a.y - b.y, a.y - b.y);
        let w = R::wide_add_prod(w, a.z - b.z, a.z - b.z);
        let w = R::wide_add_prod(w, a.w - b.w, a.w - b.w);
        let w = R::wide_add_prod(w, a.a - b.a, a.a - b.a);
        R::wide_sqrt(R::wide_add_prod(w, a.b - b.b, a.b - b.b))
    };
    if d == R::zero() {
        return Option::Some(a);
    }
    let s = {
        let w = R::wide_add_prod(R::wide_zero(), a.x + b.x, a.x + b.x);
        let w = R::wide_add_prod(w, a.y + b.y, a.y + b.y);
        let w = R::wide_add_prod(w, a.z + b.z, a.z + b.z);
        let w = R::wide_add_prod(w, a.w + b.w, a.w + b.w);
        let w = R::wide_add_prod(w, a.a + b.a, a.a + b.a);
        R::wide_sqrt(R::wide_add_prod(w, a.b + b.b, a.b + b.b))
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
        Vector6 {
            x: R::sum_prod2(a.x, ta, b.x, tb),
            y: R::sum_prod2(a.y, ta, b.y, tb),
            z: R::sum_prod2(a.z, ta, b.z, tb),
            w: R::sum_prod2(a.w, ta, b.w, tb),
            a: R::sum_prod2(a.a, ta, b.a, tb),
            b: R::sum_prod2(a.b, ta, b.b, tb),
        },
    )
}
