//! Unit tests of the WP 8.4-P08 completion of `UnitComplex`: the division, the heterogeneous
//! operators (rotations, translations, isometries, similarities), `from_complex`,
//! `rotation_between_axis`, `from_matrix_eps`, the `Vector1` forms (`from_scaled_axis`,
//! `scaled_axis`, `axis_angle`) and the trait impls: exact cases, identities, the iteration
//! measurement and the oracle vectors of `tools/oracle` (upstream nalgebra 0.35 on the same raw
//! inputs, tolerance in ulp).
//!
//! `oracle_ext.cairo` is emitted from `tools/oracle` (committed vectors, at most 8 cases per
//! distribution):
//!
//! ```text
//! cargo run --release -- emit-cairo unit_complex_completion --from vectors --max-per-dist 8 \
//!     --out crates/nalgebra/src/geometry/unit_complex/oracle_ext.cairo
//! ```
//!
//! In-crate part (WP 8.1c): the tests of crate-internal items only; the tests of the public API are
//! in `crates/tests_geometry/src/unit_complex/tests_ext.cairo`.

use crate::base::matrix_test_utils::{excess, fx, m2, max_ulp_diff_uc, uct};
use super::{FROM_MATRIX_MAX_ITER, UnitComplexAngleInternalTrait, UnitComplexTrait, oracle_ext};

/// The 2D Müller iteration (`max_iter > 0`) reaches the closed form: from the identity, with
/// `eps` = 32 ulp, 23 of the 24 oracle cases stop in 4 to 12 iterations (the step `tan(φ - θ)`
/// converges cubically once the error is below π/2), and the last one oscillates at the noise
/// floor until the bound, 3 ulp from the closed form.
#[test]
fn test_from_matrix_eps_iterations_on_the_oracle_set() {
    let mut cases = oracle_ext::unit_complex_from_matrix_cases();
    let (mut worst, mut most, mut stopped) = (0, 0, 0);
    while let Some(case) = cases.pop_front() {
        let (m, e, tol) = *case;
        let (got, iterations) = UnitComplexAngleInternalTrait::from_matrix_eps_count(
            m2(m), fx(32), 0, UnitComplexTrait::identity(),
        );
        if iterations < FROM_MATRIX_MAX_ITER {
            stopped += 1;
            most = core::cmp::max(most, iterations);
        }
        worst = core::cmp::max(worst, excess(max_ulp_diff_uc(got, uct(e)), tol.into()));
    }
    println!("unit_complex from_matrix_eps: {stopped} of 24 stop, in at most {most} iterations");
    assert!(stopped == 23 && most == 12);
    assert!(worst == 0);
}
