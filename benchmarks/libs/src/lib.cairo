//! Gas benchmarks of EXISTING Cairo libraries (cubit, orion, alexandria, origami) against
//! hand-written baselines. See README.md.

#[cfg(test)]
mod bench_alexandria;
#[cfg(test)]
mod bench_mat;
#[cfg(test)]
mod bench_q16;
#[cfg(test)]
mod bench_q32;
#[cfg(test)]
mod bench_q64;
#[cfg(test)]
mod bench_trig;
#[cfg(test)]
mod bench_vec;
pub mod reference;
