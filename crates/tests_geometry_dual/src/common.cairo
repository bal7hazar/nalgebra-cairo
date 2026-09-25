//! Package-local helpers: builders of dual quaternions from the oracle's raw tuples
//! `((w, i, j, k), (w, i, j, k))` and their component-wise errors in raw units.

use core::cmp::max;
use fixed::Fixed;
use nalgebra::geometry::dual_quaternion::DualQuaternion;
use nalgebra::geometry::unit_dual_quaternion::UnitDualQuaternion;
use nalgebra_tests_utils::{excess, max_ulp_diff_q, qt};

/// A dual quaternion as the oracle's raw tuple `(real (w, i, j, k), dual (w, i, j, k))`.
pub type Dq = ((i64, i64, i64, i64), (i64, i64, i64, i64));

/// `DualQuaternion` from raw components (oracle layout).
pub fn dqt(t: Dq) -> DualQuaternion<Fixed> {
    let (r, d) = t;
    DualQuaternion { real: qt(r), dual: qt(d) }
}

/// `UnitDualQuaternion` from raw components (oracle layout), WITHOUT normalisation.
pub fn udqt(t: Dq) -> UnitDualQuaternion<Fixed> {
    UnitDualQuaternion { dual_quaternion: dqt(t) }
}

/// Largest component-wise error in raw units.
pub fn dq_err(got: DualQuaternion<Fixed>, e: Dq) -> u128 {
    let e = dqt(e);
    max(max_ulp_diff_q(got.real, e.real), max_ulp_diff_q(got.dual, e.dual))
}

/// The excess of `err` over `tol`, printed when positive.
pub fn report(op: ByteArray, index: usize, err: u128, tol: u64) -> u128 {
    let e = excess(err, tol.into());
    if e > 0 {
        println!("{op} case {index}: error {err} ulp, tolerance {tol}");
    }
    e
}
