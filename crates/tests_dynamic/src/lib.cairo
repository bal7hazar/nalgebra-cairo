//! Package `nalgebra_tests_dynamic` (WP 8.5-P13): tests and gas benchmarks of the dynamic
//! matrices `DMatrix` / `DVector` / `RowDVector` (DESIGN D5) and of the static-shape forms that
//! come with them (`base/dynamic/shapes.cairo`), through the public API, and the benchmarks of
//! the storage layouts D5 was decided on (`layout`).

#[cfg(test)]
mod benches;
#[cfg(test)]
mod construction;
#[cfg(test)]
mod edition;
#[cfg(test)]
mod helpers;
#[cfg(test)]
mod layout;
#[cfg(test)]
mod ops;
#[cfg(test)]
mod oracle;
#[cfg(test)]
mod oracle_dynamic;
#[cfg(test)]
mod static_forms;
