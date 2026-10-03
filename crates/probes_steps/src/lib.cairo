//! Cairo-steps probes of the hot primitives (WP 11-OPT-0, docs/STEPS.md): one probe per primitive,
//! each a pair of tests `probe_<group>__baseline` / `probe_<group>__op` over the same fixed inputs
//! and the same exact assertion, the baseline without the operation. Run with
//! `snforge test -p nalgebra_probes_steps --tracked-resource cairo-steps --detailed-resources`:
//! `scripts/steps_report.py` subtracts the baseline from the operation (net Cairo steps).
//!
//! Test-only: nothing here is compiled outside `snforge test`.
#[cfg(test)]
mod builders;
#[cfg(test)]
mod cholesky;
#[cfg(test)]
mod eq;
#[cfg(test)]
mod isometry3;
#[cfg(test)]
mod lu;
#[cfg(test)]
mod matrix3;
#[cfg(test)]
mod matrix6;
#[cfg(test)]
mod scalar;
#[cfg(test)]
mod svd3;
#[cfg(test)]
mod symmetric_eigen3;
#[cfg(test)]
mod unit_quaternion;
#[cfg(test)]
mod vector3;
