//! `DualQuaternion`: a dual quaternion `real + ε·dual` (upstream `nalgebra::DualQuaternion`,
//! WP 8.4-P12).
//!
//! - `DualQuaternionTrait` / `DualQuaternionImpl`: construction, conjugation, normalisation,
//!   inverse, interpolation, the scalar products (`scale` / `unscale`, upstream's `* k` / `/ k`),
//!   the approximate comparisons and the products by a `UnitDualQuaternion`
//!   (`mul_unit_dual_quaternion` / `div_unit_dual_quaternion`: Cairo's `Mul` / `Div` are
//!   homogeneous);
//! - the operators `+`, `-`, unary `-`, `*` (the dual-quaternion product), `*=` / `/=` by a scalar,
//!   `Zero`, `One` and `Index`: their impls live in this module, where the compiler finds them
//!   without any import.
//!
//! The unit dual quaternion of a rigid-body transform is `UnitDualQuaternion`
//! (`geometry::unit_dual_quaternion`); this type is the general algebra it is built on.
//!
//! Conventions: the fields are upstream's (`real`, `dual`), and `Serde` / `Index` follow
//! upstream's `[T; 8]` view `(real.i, real.j, real.k, real.w, dual.i, dual.j, dual.k, dual.w)`
//! (the storage order of each `Quaternion`, DESIGN D8).
//!
//! The approximate comparisons (`abs_diff_eq`, `relative_eq`, `ulps_eq`) count their tolerances
//! in ulp (DESIGN D3) and, like upstream's `approx` impls, accept `other` OR `-other` taken as a
//! WHOLE (the eight components compared with the same sign: `(r, d)` and `(-r, -d)` are the same
//! rigid transform, `(r, -d)` is not).
//!
//! Numeric contract (AGENTS.md): every sum of products goes through a fused `Real` kernel (one
//! floor rounding and one overflow check per output scalar); nothing wraps silently. In
//! particular each component of the dual part of a product, `a.real · b.dual + a.dual · b.real`,
//! is ONE accumulation of eight products floored once, where upstream rounds two Hamilton
//! products and their sum.

pub use nalgebra_static3::geometry::dual_quaternion::*;
