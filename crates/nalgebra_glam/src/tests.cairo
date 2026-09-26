//! Tests of the conversions (WP 8.6-P19): exact component order, round trips, every `TryInto`
//! rejection. Operands go through `black_box`, so the compiler does not specialize a function per
//! constant argument (compile budget, AGENTS.md).

use fixed::{Fixed, ONE_RAW};

mod glam_isometry;
mod glam_matrix;
mod glam_point;
mod glam_quaternion;
mod glam_rotation;
mod glam_similarity;
mod glam_translation;
mod glam_unit_complex;

/// The integer `v` as a `Fixed`.
fn int(v: i64) -> Fixed {
    Fixed { raw: v * ONE_RAW }
}

/// `1 / sqrt(2)` rounded to the nearest raw unit: the sine and cosine of `π / 4`.
const FRAC_1_SQRT_2_RAW: i64 = 3037000500;

/// Whether `a` and `b` differ by at most `ulps` raw units.
fn near(a: Fixed, b: Fixed, ulps: i64) -> bool {
    let d = a.raw - b.raw;
    -ulps <= d && d <= ulps
}
