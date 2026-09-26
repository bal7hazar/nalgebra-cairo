//! The free functions of upstream's crate root (`nalgebra-rs/src/lib.rs`), re-exported by
//! `lib.cairo` as `nalgebra::{zero, one, wrap, clamp, min, max, abs, inf, sup, inf_sup,
//! partial_*, center, distance, distance_squared, convert*, try_convert*, is_convertible}`.
//!
//! Upstream names, argument order and semantics. Cairo forms:
//! - Upstream's `&T` arguments are values (Cairo values are `Copy`), and the `Option<&T>` results
//!   of `partial_min` / `partial_max` / `partial_clamp` / `partial_sort2` are `Option<T>`; except
//!   `is_convertible` and the `*_ref*` conversions, which take a snapshot `@From` (the `&From` of
//!   their upstream signature: what tells them apart from `convert` / `try_convert`).
//! - The upstream bounds are Cairo traits: `num::Zero` / `num::One` are corelib's
//!   `core::num::traits::Zero` / `One` (`Fixed`, the integers, `One` on the square matrices), the
//!   orderings corelib's `PartialOrd` (Cairo has no `Ord`: `min` / `max` take `PartialOrd`, like
//!   corelib's `core::cmp::min` / `max`), `Signed` simba's `Real`, `SimdPartialOrd` on matrices
//!   the kernel trait `MatrixInfSup` (one impl per static shape, in its module), the points the
//!   kernel trait `PointMetric` (`Point1` .. `Point6`), and simba's `SupersetOf` the corelib
//!   conversions: `convert` / `convert_ref` are `Into`, `try_convert` / `try_convert_ref` /
//!   `is_convertible` are `TryInto` (which corelib derives from every `Into`), and
//!   `convert_unchecked` / `convert_ref_unchecked` the kernel trait `ConvertUnchecked` (the
//!   checked conversions of the crate, without their check).
//! - `partial_cmp` returns this module's `Ordering` (upstream `core::cmp::Ordering`, which Cairo's
//!   corelib does not have).

use core::num::traits::{One, Zero};
use simba::scalar::Real;
use crate::base::point2::{Point2, Point2InternalTrait};
use crate::base::point3::{Point3, Point3InternalTrait};
use crate::base::{Matrix3, Matrix4};
use crate::geometry::{
    Affine2, Affine2Trait, Affine3, Affine3Trait, Point1, Point4, Point5, Point6, Projective2,
    Projective2Trait, Projective3, Projective3Trait, Transform2, Transform2Trait, Transform3,
    Transform3Trait,
};

/// Panic messages of the crate-root functions (stable API).
pub mod errors {
    /// `wrap` with `min >= max` (upstream: "Invalid wrapping bounds.").
    pub const INVALID_WRAPPING_BOUNDS: felt252 = 'nalgebra: invalid wrap bounds';
}

/// The result of `partial_cmp`: `a` is less than, equal to or greater than `b`. Upstream:
/// `core::cmp::Ordering` (Rust's standard library; Cairo's corelib has no ordering type).
#[derive(Copy, Drop, PartialEq, Debug)]
pub enum Ordering {
    Less,
    Equal,
    Greater,
}

// --- identities --------------------------------------------------------------------------------

/// The multiplicative identity `T::one()` (a scalar, or the identity of a square matrix).
/// Upstream: `nalgebra::one`.
pub fn one<T, +One<T>>() -> T {
    One::one()
}

/// The additive identity `T::zero()`. Upstream: `nalgebra::zero` (the static matrices have
/// `zeros()` instead of `num::Zero`: their `Sum` folds without it).
pub fn zero<T, +Zero<T>>() -> T {
    Zero::zero()
}

// --- orderings ----------------------------------------------------------------------------------

/// `val` wrapped into `[min, max]` by adding or subtracting `max - min` until it enters the
/// interval (upstream's loop, the same results: `val < min` ends in `[min, max)`, `val > max` in
/// `(min, max]`). Exact; panics with `nalgebra: invalid wrap bounds` unless `min < max`, and on
/// overflow. Upstream: `nalgebra::wrap`.
pub fn wrap<T, +PartialOrd<T>, +Add<T>, +Sub<T>, +Copy<T>, +Drop<T>>(val: T, min: T, max: T) -> T {
    if !(min < max) {
        core::panic_with_felt252(errors::INVALID_WRAPPING_BOUNDS)
    }
    let width = max - min;
    let mut val = val;
    if val < min {
        val = val + width;
        while val < min {
            val = val + width;
        }
    } else if val > max {
        val = val - width;
        while val > max {
            val = val - width;
        }
    }
    val
}

/// `val` clamped to `[min, max]`: `val` when `min < val < max`, else `max` when `val > min`, else
/// `min` (upstream's exact tests, so a partial order gives upstream's result too). Upstream:
/// `nalgebra::clamp`.
pub fn clamp<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(val: T, min: T, max: T) -> T {
    if val > min {
        if val < max {
            val
        } else {
            max
        }
    } else {
        min
    }
}

/// The larger of `a` and `b` (`b` when equal, like `std::cmp::max`). Upstream: `nalgebra::max`.
pub fn max<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> T {
    core::cmp::max(a, b)
}

/// The smaller of `a` and `b` (`a` when equal, like `std::cmp::min`). Upstream: `nalgebra::min`.
pub fn min<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> T {
    core::cmp::min(a, b)
}

/// `|a|` (`Real::abs`). Exact; panics on overflow (`|MIN|`). Upstream: `nalgebra::abs`
/// (deprecated upstream in favour of the methods).
pub fn abs<T, impl R: Real<T>>(a: T) -> T {
    R::abs(a)
}

/// The kernel of `inf` / `sup` / `inf_sup` (upstream's `SimdPartialOrd` bound on the
/// components): implemented by each static shape in its module, as its methods of the same
/// names. Static functions (no `self`), so they never compete with the shapes' methods.
pub trait MatrixInfSup<M> {
    /// The component-wise minimum.
    fn inf(a: M, b: M) -> M;
    /// The component-wise maximum.
    fn sup(a: M, b: M) -> M;
    /// `(inf(a, b), sup(a, b))`.
    fn inf_sup(a: M, b: M) -> (M, M);
}

/// The component-wise minimum of two matrices of the same shape. Exact. Upstream:
/// `nalgebra::inf` (deprecated upstream in favour of `Matrix::inf`).
pub fn inf<M, +MatrixInfSup<M>>(a: M, b: M) -> M {
    MatrixInfSup::inf(a, b)
}

/// The component-wise maximum of two matrices of the same shape. Exact. Upstream:
/// `nalgebra::sup` (deprecated upstream in favour of `Matrix::sup`).
pub fn sup<M, +MatrixInfSup<M>>(a: M, b: M) -> M {
    MatrixInfSup::sup(a, b)
}

/// `(inf(a, b), sup(a, b))`. Exact. Upstream: `nalgebra::inf_sup` (deprecated upstream in
/// favour of `Matrix::inf_sup`).
pub fn inf_sup<M, +MatrixInfSup<M>>(a: M, b: M) -> (M, M) {
    MatrixInfSup::inf_sup(a, b)
}

/// `a` compared with `b` by a partial order: `Less` when `a <= b` but not `a >= b`, `Greater`
/// when `a >= b` but not `a <= b`, `Equal` when both, `None` when neither (two matrices whose
/// components compare differently: upstream's `PartialOrd for Matrix` gives the same answers).
/// Two comparisons. Upstream: `nalgebra::partial_cmp`.
pub fn partial_cmp<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> Option<Ordering> {
    let le = a <= b;
    let ge = a >= b;
    if le {
        if ge {
            Option::Some(Ordering::Equal)
        } else {
            Option::Some(Ordering::Less)
        }
    } else if ge {
        Option::Some(Ordering::Greater)
    } else {
        Option::None
    }
}

/// `a < b` (`false` when not comparable). Upstream: `nalgebra::partial_lt`.
pub fn partial_lt<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> bool {
    a < b
}

/// `a <= b` (`false` when not comparable). Upstream: `nalgebra::partial_le`.
pub fn partial_le<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> bool {
    a <= b
}

/// `a > b` (`false` when not comparable). Upstream: `nalgebra::partial_gt`.
pub fn partial_gt<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> bool {
    a > b
}

/// `a >= b` (`false` when not comparable). Upstream: `nalgebra::partial_ge`.
pub fn partial_ge<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> bool {
    a >= b
}

/// The smaller of `a` and `b` if they are comparable (`a` when equal), `None` otherwise. One
/// comparison when `a <= b`, two otherwise. Upstream: `nalgebra::partial_min` (`Option<&T>`).
pub fn partial_min<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> Option<T> {
    if a <= b {
        Option::Some(a)
    } else if a >= b {
        Option::Some(b)
    } else {
        Option::None
    }
}

/// The larger of `a` and `b` if they are comparable (`a` when equal), `None` otherwise. One
/// comparison when `a >= b`, two otherwise. Upstream: `nalgebra::partial_max` (`Option<&T>`).
pub fn partial_max<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> Option<T> {
    if a >= b {
        Option::Some(a)
    } else if a <= b {
        Option::Some(b)
    } else {
        Option::None
    }
}

/// `value` clamped to `[min, max]` if it is comparable with both bounds, `None` otherwise: `min`
/// when `value` is less than `min`, else `max` when it is greater than `max`, else `value`.
/// Upstream: `nalgebra::partial_clamp` (`Option<&T>`).
pub fn partial_clamp<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(value: T, min: T, max: T) -> Option<T> {
    let lo_le = value <= min;
    let lo_ge = value >= min;
    let hi_le = value <= max;
    let hi_ge = value >= max;
    if !(lo_le || lo_ge) || !(hi_le || hi_ge) {
        return Option::None;
    }
    if lo_le && !lo_ge {
        Option::Some(min)
    } else if hi_ge && !hi_le {
        Option::Some(max)
    } else {
        Option::Some(value)
    }
}

/// `(a, b)` sorted in increasing order if they are comparable, `None` otherwise. Upstream:
/// `nalgebra::partial_sort2` (`Option<(&T, &T)>`, `(b, a)` when equal: the same values, since
/// equal components are identical, so `(a, b)` here, which costs one comparison when `a <= b`).
pub fn partial_sort2<T, +PartialOrd<T>, +Copy<T>, +Drop<T>>(a: T, b: T) -> Option<(T, T)> {
    if a <= b {
        Option::Some((a, b))
    } else if a >= b {
        Option::Some((b, a))
    } else {
        Option::None
    }
}

// --- points -------------------------------------------------------------------------------------

/// The kernel of `center` / `distance` / `distance_squared` on the points `Point1` .. `Point6`
/// (upstream: generic over `Point<T, D>`). Static functions (no `self`).
pub trait PointMetric<P, T> {
    /// The midpoint of `p1` and `p2`.
    fn center(p1: P, p2: P) -> P;
    /// The Euclidean distance between `p1` and `p2`.
    fn distance(p1: P, p2: P) -> T;
    /// The squared Euclidean distance between `p1` and `p2`.
    fn distance_squared(p1: P, p2: P) -> T;
}

/// The midpoint of `p1` and `p2`: per coordinate the exact floor of `(a + b) / 2` (a fused
/// `sum_prod2` with the constant `1/2`, so the sum cannot overflow; measured cheaper than
/// upstream's `(p1 + p2) * 0.5`, `Point2::center` kernel). Upstream: `nalgebra::center`.
pub fn center<P, T, +PointMetric<P, T>>(p1: P, p2: P) -> P {
    PointMetric::center(p1, p2)
}

/// The distance `|p2 - p1|`: the square root of the UNSCALED exact sum of the squared coordinate
/// differences, floored once (no intermediate overflow). Panics when a difference overflows.
/// Upstream: `nalgebra::distance`.
pub fn distance<P, T, +PointMetric<P, T>>(p1: P, p2: P) -> T {
    PointMetric::distance(p1, p2)
}

/// The squared distance `|p2 - p1|^2`, fused (floored once). Panics when a difference or the
/// result overflows. Upstream: `nalgebra::distance_squared`.
pub fn distance_squared<P, T, +PointMetric<P, T>>(p1: P, p2: P) -> T {
    PointMetric::distance_squared(p1, p2)
}

/// `PointMetric` on `Point1`: `|dx|` and `dx * dx` (the norms of `Matrix1`).
pub impl MetricPoint1<
    T, impl R: Real<T>, +Sub<T>, +Mul<T>, +Copy<T>, +Drop<T>,
> of PointMetric<Point1<T>, T> {
    #[inline(always)]
    fn center(p1: Point1<T>, p2: Point1<T>) -> Point1<T> {
        Point1 { x: R::sum_prod2(p1.x, R::HALF, p2.x, R::HALF) }
    }
    #[inline(always)]
    fn distance(p1: Point1<T>, p2: Point1<T>) -> T {
        R::abs(p2.x - p1.x)
    }
    #[inline(always)]
    fn distance_squared(p1: Point1<T>, p2: Point1<T>) -> T {
        let d = p2.x - p1.x;
        d * d
    }
}

/// `PointMetric` on `Point2`: the kernels of `Point2InternalTrait`.
pub impl MetricPoint2<
    T,
    impl R: Real<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<T>,
    +Drop<T>,
> of PointMetric<Point2<T>, T> {
    #[inline(always)]
    fn center(p1: Point2<T>, p2: Point2<T>) -> Point2<T> {
        Point2InternalTrait::center(p1, p2)
    }
    #[inline(always)]
    fn distance(p1: Point2<T>, p2: Point2<T>) -> T {
        Point2InternalTrait::distance(p2, p1)
    }
    #[inline(always)]
    fn distance_squared(p1: Point2<T>, p2: Point2<T>) -> T {
        Point2InternalTrait::distance_squared(p2, p1)
    }
}

/// `PointMetric` on `Point3`: the kernels of `Point3InternalTrait`.
pub impl MetricPoint3<
    T,
    impl R: Real<T>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
    +Copy<T>,
    +Drop<T>,
> of PointMetric<Point3<T>, T> {
    #[inline(always)]
    fn center(p1: Point3<T>, p2: Point3<T>) -> Point3<T> {
        Point3InternalTrait::center(p1, p2)
    }
    #[inline(always)]
    fn distance(p1: Point3<T>, p2: Point3<T>) -> T {
        Point3InternalTrait::distance(p2, p1)
    }
    #[inline(always)]
    fn distance_squared(p1: Point3<T>, p2: Point3<T>) -> T {
        Point3InternalTrait::distance_squared(p2, p1)
    }
}

/// `PointMetric` on `Point4`: `Real::norm4` / `norm_squared4` of the differences.
pub impl MetricPoint4<
    T, impl R: Real<T>, +Sub<T>, +Copy<T>, +Drop<T>,
> of PointMetric<Point4<T>, T> {
    #[inline(always)]
    fn center(p1: Point4<T>, p2: Point4<T>) -> Point4<T> {
        Point4 {
            x: R::sum_prod2(p1.x, R::HALF, p2.x, R::HALF),
            y: R::sum_prod2(p1.y, R::HALF, p2.y, R::HALF),
            z: R::sum_prod2(p1.z, R::HALF, p2.z, R::HALF),
            w: R::sum_prod2(p1.w, R::HALF, p2.w, R::HALF),
        }
    }
    #[inline(always)]
    fn distance(p1: Point4<T>, p2: Point4<T>) -> T {
        R::norm4(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, p2.w - p1.w)
    }
    #[inline(always)]
    fn distance_squared(p1: Point4<T>, p2: Point4<T>) -> T {
        R::norm_squared4(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, p2.w - p1.w)
    }
}

/// `PointMetric` on `Point5`: one `Real::Wide` chain of the squared differences (the norms of
/// `Vector5`).
pub impl MetricPoint5<
    T, impl R: Real<T>, +Sub<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of PointMetric<Point5<T>, T> {
    #[inline(always)]
    fn center(p1: Point5<T>, p2: Point5<T>) -> Point5<T> {
        Point5 {
            x: R::sum_prod2(p1.x, R::HALF, p2.x, R::HALF),
            y: R::sum_prod2(p1.y, R::HALF, p2.y, R::HALF),
            z: R::sum_prod2(p1.z, R::HALF, p2.z, R::HALF),
            w: R::sum_prod2(p1.w, R::HALF, p2.w, R::HALF),
            a: R::sum_prod2(p1.a, R::HALF, p2.a, R::HALF),
        }
    }
    fn distance(p1: Point5<T>, p2: Point5<T>) -> T {
        R::wide_sqrt(squares5::<T, R>(p1, p2))
    }
    fn distance_squared(p1: Point5<T>, p2: Point5<T>) -> T {
        R::wide_rescale(squares5::<T, R>(p1, p2))
    }
}

/// The unscaled exact sum of the squared coordinate differences of two `Point5`s.
fn squares5<T, impl R: Real<T>, +Sub<T>, +Copy<T>, +Drop<T>>(
    p1: Point5<T>, p2: Point5<T>,
) -> R::Wide {
    let (dx, dy, dz, dw, da) = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z, p2.w - p1.w, p2.a - p1.a);
    let w = R::wide_add_prod(R::wide_zero(), dx, dx);
    let w = R::wide_add_prod(w, dy, dy);
    let w = R::wide_add_prod(w, dz, dz);
    let w = R::wide_add_prod(w, dw, dw);
    R::wide_add_prod(w, da, da)
}

/// `PointMetric` on `Point6`: one `Real::Wide` chain of the squared differences (the norms of
/// `Vector6`).
pub impl MetricPoint6<
    T, impl R: Real<T>, +Sub<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>,
> of PointMetric<Point6<T>, T> {
    #[inline(always)]
    fn center(p1: Point6<T>, p2: Point6<T>) -> Point6<T> {
        Point6 {
            x: R::sum_prod2(p1.x, R::HALF, p2.x, R::HALF),
            y: R::sum_prod2(p1.y, R::HALF, p2.y, R::HALF),
            z: R::sum_prod2(p1.z, R::HALF, p2.z, R::HALF),
            w: R::sum_prod2(p1.w, R::HALF, p2.w, R::HALF),
            a: R::sum_prod2(p1.a, R::HALF, p2.a, R::HALF),
            b: R::sum_prod2(p1.b, R::HALF, p2.b, R::HALF),
        }
    }
    fn distance(p1: Point6<T>, p2: Point6<T>) -> T {
        R::wide_sqrt(squares6::<T, R>(p1, p2))
    }
    fn distance_squared(p1: Point6<T>, p2: Point6<T>) -> T {
        R::wide_rescale(squares6::<T, R>(p1, p2))
    }
}

/// The unscaled exact sum of the squared coordinate differences of two `Point6`s.
fn squares6<T, impl R: Real<T>, +Sub<T>, +Copy<T>, +Drop<T>>(
    p1: Point6<T>, p2: Point6<T>,
) -> R::Wide {
    let (dx, dy, dz) = (p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
    let (dw, da, db) = (p2.w - p1.w, p2.a - p1.a, p2.b - p1.b);
    let w = R::wide_add_prod(R::wide_zero(), dx, dx);
    let w = R::wide_add_prod(w, dy, dy);
    let w = R::wide_add_prod(w, dz, dz);
    let w = R::wide_add_prod(w, dw, dw);
    let w = R::wide_add_prod(w, da, da);
    R::wide_add_prod(w, db, db)
}

// --- conversions --------------------------------------------------------------------------------

/// `t.into()`: `t` converted into the type `To`. Upstream: `nalgebra::convert` (`SupersetOf`,
/// which is `Into` here: every widening conversion of the crate).
pub fn convert<From, To, +Into<From, To>>(t: From) -> To {
    t.into()
}

/// `(*t).into()`. Upstream: `nalgebra::convert_ref`.
pub fn convert_ref<From, To, +Into<From, To>, +Copy<From>>(t: @From) -> To {
    (*t).into()
}

/// `t.try_into()`: `Some` when `t` is in the subset `To` (a transform category, ...), `None`
/// otherwise. Upstream: `nalgebra::try_convert` (`SupersetOf::to_subset`).
pub fn try_convert<From, To, +TryInto<From, To>>(t: From) -> Option<To> {
    t.try_into()
}

/// `(*t).try_into()`. Upstream: `nalgebra::try_convert_ref`.
pub fn try_convert_ref<From, To, +TryInto<From, To>, +Copy<From>>(t: @From) -> Option<To> {
    (*t).try_into()
}

/// `true` when `*t` converts into `To` (`try_into` succeeds; always for an `Into` pair). The
/// target is named by the call: `is_convertible::<_, Affine2<Fixed>>(@m)`. Upstream:
/// `nalgebra::is_convertible` (`SupersetOf::is_in_subset`).
pub fn is_convertible<From, To, +TryInto<From, To>, +Copy<From>, +Drop<To>>(t: @From) -> bool {
    let converted: Option<To> = (*t).try_into();
    converted.is_some()
}

/// The conversion of the superset `From` into its subset `To` WITHOUT the check of `try_convert`
/// (the value is assumed to be in the subset: nothing is verified, like upstream). Implemented for
/// every checked conversion of the crate (`TryInto`): the homogeneous matrices and the transform
/// categories into `Transform2/3` / `Projective2/3` / `Affine2/3`.
pub trait ConvertUnchecked<From, To> {
    /// `t` as a `To`, unchecked.
    fn convert_unchecked(t: From) -> To;
}

/// `t` converted into its subset `To` without any check (`ConvertUnchecked`: the checked
/// conversions of the crate, minus their check). Upstream: `nalgebra::convert_unchecked`
/// (`SupersetOf::to_subset_unchecked`).
pub fn convert_unchecked<From, To, +ConvertUnchecked<From, To>>(t: From) -> To {
    ConvertUnchecked::convert_unchecked(t)
}

/// `convert_unchecked(*t)`. Upstream: `nalgebra::convert_ref_unchecked`.
pub fn convert_ref_unchecked<From, To, +ConvertUnchecked<From, To>, +Copy<From>>(t: @From) -> To {
    ConvertUnchecked::convert_unchecked(*t)
}

/// `ConvertUnchecked`: the matrix `t` as a `Transform2`, unchecked
/// (`Transform2Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Matrix3> for Transform2` (`from_superset_unchecked`).
pub impl UncheckedMatrix3Transform2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Matrix3<T>, Transform2<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Matrix3<T>) -> Transform2<T> {
        Transform2Trait::from_matrix_unchecked(t)
    }
}

/// `ConvertUnchecked`: the matrix `t` as a `Projective2`, unchecked
/// (`Projective2Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Matrix3> for Projective2` (`from_superset_unchecked`).
pub impl UncheckedMatrix3Projective2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Matrix3<T>, Projective2<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Matrix3<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(t)
    }
}

/// `ConvertUnchecked`: the matrix of `t` as a `Projective2`, unchecked
/// (`Projective2Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Transform2> for Projective2` (`from_superset_unchecked`).
pub impl UncheckedTransform2Projective2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Transform2<T>, Projective2<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Transform2<T>) -> Projective2<T> {
        Projective2Trait::from_matrix_unchecked(t.into_inner())
    }
}

/// `ConvertUnchecked`: the matrix `t` as a `Affine2`, unchecked
/// (`Affine2Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Matrix3> for Affine2` (`from_superset_unchecked`).
pub impl UncheckedMatrix3Affine2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Matrix3<T>, Affine2<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Matrix3<T>) -> Affine2<T> {
        Affine2Trait::from_matrix_unchecked(t)
    }
}

/// `ConvertUnchecked`: the matrix of `t` as a `Affine2`, unchecked
/// (`Affine2Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Transform2> for Affine2` (`from_superset_unchecked`).
pub impl UncheckedTransform2Affine2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Transform2<T>, Affine2<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Transform2<T>) -> Affine2<T> {
        Affine2Trait::from_matrix_unchecked(t.into_inner())
    }
}

/// `ConvertUnchecked`: the matrix of `t` as a `Affine2`, unchecked
/// (`Affine2Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Projective2> for Affine2` (`from_superset_unchecked`).
pub impl UncheckedProjective2Affine2<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Projective2<T>, Affine2<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Projective2<T>) -> Affine2<T> {
        Affine2Trait::from_matrix_unchecked(t.into_inner())
    }
}

/// `ConvertUnchecked`: the matrix `t` as a `Transform3`, unchecked
/// (`Transform3Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Matrix4> for Transform3` (`from_superset_unchecked`).
pub impl UncheckedMatrix4Transform3<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Matrix4<T>, Transform3<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Matrix4<T>) -> Transform3<T> {
        Transform3Trait::from_matrix_unchecked(t)
    }
}

/// `ConvertUnchecked`: the matrix `t` as a `Projective3`, unchecked
/// (`Projective3Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Matrix4> for Projective3` (`from_superset_unchecked`).
pub impl UncheckedMatrix4Projective3<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Matrix4<T>, Projective3<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Matrix4<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(t)
    }
}

/// `ConvertUnchecked`: the matrix of `t` as a `Projective3`, unchecked
/// (`Projective3Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Transform3> for Projective3` (`from_superset_unchecked`).
pub impl UncheckedTransform3Projective3<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Transform3<T>, Projective3<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Transform3<T>) -> Projective3<T> {
        Projective3Trait::from_matrix_unchecked(t.into_inner())
    }
}

/// `ConvertUnchecked`: the matrix `t` as a `Affine3`, unchecked
/// (`Affine3Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Matrix4> for Affine3` (`from_superset_unchecked`).
pub impl UncheckedMatrix4Affine3<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Matrix4<T>, Affine3<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Matrix4<T>) -> Affine3<T> {
        Affine3Trait::from_matrix_unchecked(t)
    }
}

/// `ConvertUnchecked`: the matrix of `t` as a `Affine3`, unchecked
/// (`Affine3Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Transform3> for Affine3` (`from_superset_unchecked`).
pub impl UncheckedTransform3Affine3<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Transform3<T>, Affine3<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Transform3<T>) -> Affine3<T> {
        Affine3Trait::from_matrix_unchecked(t.into_inner())
    }
}

/// `ConvertUnchecked`: the matrix of `t` as a `Affine3`, unchecked
/// (`Affine3Trait::from_matrix_unchecked`).
/// Upstream: `SubsetOf<Projective3> for Affine3` (`from_superset_unchecked`).
pub impl UncheckedProjective3Affine3<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of ConvertUnchecked<Projective3<T>, Affine3<T>> {
    #[inline(always)]
    fn convert_unchecked(t: Projective3<T>) -> Affine3<T> {
        Affine3Trait::from_matrix_unchecked(t.into_inner())
    }
}
