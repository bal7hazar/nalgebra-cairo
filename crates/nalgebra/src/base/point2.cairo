//! `Point2`: a 2-dimensional point (upstream `nalgebra::Point2`, which lives in `geometry`).
//!
//! A point is an affine position, as opposed to a `Vector2` displacement: points cannot be added
//! together, and `p - q` is a vector. Corelib's `Add` / `Sub` are homogeneous (`T + T -> T`), so
//! the heterogeneous operations are named methods:
//!
//! | upstream | here |
//! |---|---|
//! | `p - q` (a vector) | `p.sub_point(q)` |
//! | `p + v`, `p - v` | `p.add_vector(v)`, `p.sub_vector(v)` (and `p += v`, `p -= v`) |
//! | `p * k`, `p / k` | `p.scale(k)`, `p.unscale(k)` (and `p *= k`, `p /= k`) |
//! | `-p`, `Point::from(v)`, `p.coords` | `-p`, `v.into()` / `from_coordinates`, `p.coords()` |
//!
//! - `Point2Trait` / `Point2Impl`: constructors, conversions, affine operations and
//!   interpolation, generic over a `simba::scalar::Real` scalar (the distances and the midpoint
//!   are upstream's crate-root free functions, crate-internal kernels here until they are ported);
//! - operators and conversions from / to `[T; 2]` and `Vector2<T>`: their impls live in
//!   this module, where the compiler finds them without any import.
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently.

#[cfg(test)]
mod benches;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod tests;
// an explicit list: `Point2Index` / `Point2PartialOrd` (0.1.0: `geometry::point`) now sit in
// `nalgebra_core::base::point2`, the module of their type (docs/SPLIT.md §3.2, §12.6)
pub use nalgebra_core::base::point2::{
    Point2, Point2AddAssign, Point2DivAssign, Point2FromArray, Point2FromVector, Point2Impl,
    Point2IntoArray, Point2IntoVector, Point2MulAssign, Point2Neg, Point2SubAssign, Point2Trait,
};
// the crate-private by-value helpers the in-crate tests use (`nalgebra_core::internal`)
#[cfg(test)]
use nalgebra_core::internal::base::point2::Point2InternalTrait;
