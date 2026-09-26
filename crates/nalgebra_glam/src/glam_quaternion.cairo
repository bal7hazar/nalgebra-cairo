//! `Quaternion` and `UnitQuaternion` <-> `Quat` (upstream
//! `third_party/glam/common/glam_quaternion.rs`).
//!
//! # Order of the components
//!
//! glam's `Quat::from_xyzw(x, y, z, w)` stores the vector part first; nalgebra's
//! `Quaternion::new(w, i, j, k)` takes the scalar part first, and the Cairo struct fields are
//! `i, j, k, w`. The conversions map `x -> i`, `y -> j`, `z -> k`, `w -> w` (DESIGN D8), exact.
//!
//! `Quat` -> `UnitQuaternion` normalizes (`UnitQuaternion::new_normalize`, upstream's
//! `From<Quat> for UnitQuaternion<f32>`): the quaternion divided by its norm, one norm (floored
//! once) and four correctly rounded divisions, and it panics with `Fixed: division by zero` on the
//! zero quaternion (upstream: NaN components). `UnitQuaternion` -> `Quat` returns the stored
//! components. glam-cairo has no `DQuat` (`f64`): those impls stay excluded (`interop`).
//!
//! Impls of `Into` are found by the compiler only when in scope: `use
//! nalgebra_glam::glam_quaternion::QuaternionFromQuat;` (or `use nalgebra_glam::prelude::*;`).

use fixed::Fixed;
use glam::Quat;
use nalgebra::{Quaternion, UnitQuaternion, UnitQuaternionTrait};

/// The `Quaternion<Fixed>` of a `Quat`: `Quaternion { i: x, j: y, k: z, w }`, exact. Upstream:
/// `From<Quat> for Quaternion<f32>` (`Quaternion::new(e.w, e.x, e.y, e.z)`).
pub impl QuaternionFromQuat of Into<Quat, Quaternion<Fixed>> {
    #[inline(always)]
    fn into(self: Quat) -> Quaternion<Fixed> {
        Quaternion { i: self.x, j: self.y, k: self.z, w: self.w }
    }
}

/// The `Quat` of a `Quaternion<Fixed>`: `Quat { x: i, y: j, z: k, w }`, exact. Upstream:
/// `From<Quaternion<f32>> for Quat` (`Quat::from_xyzw(e.i, e.j, e.k, e.w)`).
pub impl QuaternionIntoQuat of Into<Quaternion<Fixed>, Quat> {
    #[inline(always)]
    fn into(self: Quaternion<Fixed>) -> Quat {
        Quat { x: self.i, y: self.j, z: self.k, w: self.w }
    }
}

/// The unit quaternion of the direction of a `Quat`: `UnitQuaternion::new_normalize` of the
/// `Quaternion` of the quat (see the module documentation for the rounding and the panic).
/// Upstream: `From<Quat> for UnitQuaternion<f32>`.
pub impl UnitQuaternionFromQuat of Into<Quat, UnitQuaternion<Fixed>> {
    #[inline(always)]
    fn into(self: Quat) -> UnitQuaternion<Fixed> {
        UnitQuaternionTrait::new_normalize(
            Quaternion { i: self.x, j: self.y, k: self.z, w: self.w },
        )
    }
}

/// The `Quat` of the stored components of a `UnitQuaternion<Fixed>` (`Quat { x: i, y: j, z: k,
/// w }`), exact. Upstream: `From<UnitQuaternion<f32>> for Quat`.
pub impl UnitQuaternionIntoQuat of Into<UnitQuaternion<Fixed>, Quat> {
    #[inline(always)]
    fn into(self: UnitQuaternion<Fixed>) -> Quat {
        let q = self.quaternion;
        Quat { x: q.i, y: q.j, z: q.k, w: q.w }
    }
}
