//! Transcendental functions. x = 0.75 rad for the trigonometric ones, 3.5 for ln, 0.75 for exp.
//!
//! Results are asserted against the true value with a tolerance (1e-3, the accuracy actually
//! reached by each implementation is printed by `report_accuracy` and quoted in the README).
//! Same fixed overhead in every test of a group (see bench_q32).

mod q32 {
    use cubit::f64::types::fixed::{Fixed, FixedTrait};
    use harness::black_box;
    use crate::reference::q32;

    const X_MAG: u64 = 3221225472; // 0.75
    const X: i64 = 3221225472;
    const TOL: u64 = 4294967; // 1e-3

    #[derive(Copy, Drop)]
    struct Inputs {
        cx: Fixed,
        cy: Fixed,
        rx: i64,
    }

    #[inline(never)]
    fn inputs() -> Inputs {
        Inputs {
            cx: black_box(FixedTrait::new(X_MAG, false)),
            cy: black_box(FixedTrait::new(15032385536, false)), // 3.5
            rx: black_box(X),
        }
    }

    fn near(value: u64, expected: u64) -> bool {
        if value > expected {
            value - expected < TOL
        } else {
            expected - value < TOL
        }
    }

    #[inline(never)]
    fn check(c: Fixed, c_expected: u64, r: i64, r_expected: u64) {
        assert!(!c.sign && near(c.mag, c_expected));
        assert!(near(r.try_into().unwrap(), r_expected));
    }

    #[test]
    fn report_accuracy() {
        let i = inputs();
        println!("q32 sin       {} / 2927616182", FixedTrait::sin(i.cx).mag);
        println!("q32 sin_fast  {} / 2927616182", FixedTrait::sin_fast(i.cx).mag);
        println!("q32 sin_ref   {} / 2927616182", q32::sin(i.rx));
        println!("q32 cos       {} / 3142579763", FixedTrait::cos(i.cx).mag);
        println!("q32 cos_fast  {} / 3142579763", FixedTrait::cos_fast(i.cx).mag);
        println!("q32 tan       {} / 4001176329", FixedTrait::tan(i.cx).mag);
        println!("q32 tan_fast  {} / 4001176329", FixedTrait::tan_fast(i.cx).mag);
        println!("q32 atan      {} / 2763816217", FixedTrait::atan(i.cx).mag);
        println!("q32 atan_fast {} / 2763816217", FixedTrait::atan_fast(i.cx).mag);
        println!("q32 acos      {} / 3104119958", FixedTrait::acos(i.cx).mag);
        println!("q32 acos_fast {} / 3104119958", FixedTrait::acos_fast(i.cx).mag);
        println!("q32 exp       {} / 9092445837", FixedTrait::exp(i.cx).mag);
        println!("q32 ln        {} / 5380575979", FixedTrait::ln(i.cy).mag);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q32__cubit() {
        let i = inputs();
        check(FixedTrait::sin(i.cx), 2927616182, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q32__cubit_fast() {
        let i = inputs();
        check(FixedTrait::sin_fast(i.cx), 2927616182, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q32__reference_poly() {
        let i = inputs();
        check(i.cx, X_MAG, q32::sin(i.rx), 2927616182);
    }

    #[test]
    #[inline(never)]
    fn bench_cos_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_cos_q32__cubit() {
        let i = inputs();
        check(FixedTrait::cos(i.cx), 3142579763, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_cos_q32__cubit_fast() {
        let i = inputs();
        check(FixedTrait::cos_fast(i.cx), 3142579763, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_tan_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_tan_q32__cubit() {
        let i = inputs();
        check(FixedTrait::tan(i.cx), 4001176329, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_tan_q32__cubit_fast() {
        let i = inputs();
        check(FixedTrait::tan_fast(i.cx), 4001176329, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_atan_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_atan_q32__cubit() {
        let i = inputs();
        check(FixedTrait::atan(i.cx), 2763816217, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_atan_q32__cubit_fast() {
        let i = inputs();
        check(FixedTrait::atan_fast(i.cx), 2763816217, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_acos_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_acos_q32__cubit() {
        let i = inputs();
        check(FixedTrait::acos(i.cx), 3104119958, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_acos_q32__cubit_fast() {
        let i = inputs();
        check(FixedTrait::acos_fast(i.cx), 3104119958, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_exp_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_exp_q32__cubit() {
        let i = inputs();
        check(FixedTrait::exp(i.cx), 9092445837, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_ln_q32__baseline() {
        let i = inputs();
        check(i.cx, X_MAG, i.rx, X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_ln_q32__cubit() {
        let i = inputs();
        check(FixedTrait::ln(i.cy), 5380575979, i.rx, X_MAG);
    }
}

mod q64 {
    use cubit::f128::types::fixed::{Fixed, FixedTrait};
    use harness::black_box;

    const X_MAG: u128 = 13835058055282163712; // 0.75
    const TOL: u128 = 18446744073709551; // 1e-3

    #[inline(never)]
    fn inputs() -> Fixed {
        black_box(FixedTrait::new(X_MAG, false))
    }

    #[inline(never)]
    fn check(c: Fixed, expected: u128) {
        let error = if c.mag > expected {
            c.mag - expected
        } else {
            expected - c.mag
        };
        assert!(!c.sign && error < TOL);
    }

    #[test]
    fn report_accuracy() {
        println!("q64 sin       {} / 12574015756871165952", FixedTrait::sin(inputs()).mag);
        println!("q64 sin_fast  {} / 12574015756871165952", FixedTrait::sin_fast(inputs()).mag);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q64__baseline() {
        check(inputs(), X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q64__cubit() {
        check(FixedTrait::sin(inputs()), 12574015756871165952);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q64__cubit_fast() {
        check(FixedTrait::sin_fast(inputs()), 12574015756871165952);
    }
}

mod q16 {
    use harness::black_box;
    use orion::numbers::fixed_point::core::FixedTrait;
    use orion::numbers::fixed_point::implementations::fp16x16::math::trig;
    use orion::numbers::{FP16x16, FP16x16Impl};

    const X_MAG: u32 = 49152; // 0.75
    const TOL: u32 = 66; // 1e-3

    #[inline(never)]
    fn inputs() -> FP16x16 {
        black_box(FixedTrait::new(X_MAG, false))
    }

    #[inline(never)]
    fn check(c: FP16x16, expected: u32) {
        let error = if c.mag > expected {
            c.mag - expected
        } else {
            expected - c.mag
        };
        assert!(!c.sign && error < TOL);
    }

    #[test]
    fn report_accuracy() {
        println!("q16 sin       {} / 44672", inputs().sin().mag);
        println!("q16 sin_fast  {} / 44672", inputs().sin_fast().mag);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q16__baseline() {
        check(inputs(), X_MAG);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q16__orion_fp16x16() {
        check(inputs().sin(), 44672);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_q16__orion_fp16x16_fast() {
        check(inputs().sin_fast(), 44672);
    }

    /// `FixedTrait::sin` silently dispatches to the LUT version; this is the Taylor loop it hides.
    #[test]
    #[inline(never)]
    fn bench_sin_q16__orion_fp16x16_taylor() {
        check(trig::sin(inputs()), 44672);
    }
}

/// alexandria_math::trigonometry works in degrees scaled by 1e8 (decimal fixed point).
mod dec8 {
    use alexandria_math::trigonometry::fast_sin;
    use harness::black_box;

    const X: i64 = 4297183463; // 42.97183463 deg = 0.75 rad
    const TOL: i64 = 100000; // 1e-3

    #[inline(never)]
    fn inputs() -> i64 {
        black_box(X)
    }

    #[inline(never)]
    fn check(value: i64, expected: i64) {
        let error = if value > expected {
            value - expected
        } else {
            expected - value
        };
        assert!(error < TOL);
    }

    #[test]
    fn report_accuracy() {
        println!("dec8 fast_sin {} / 68163876", fast_sin(inputs()));
    }

    #[test]
    #[inline(never)]
    fn bench_sin_dec8__baseline() {
        check(inputs(), X);
    }

    #[test]
    #[inline(never)]
    fn bench_sin_dec8__alexandria_fast_sin() {
        check(fast_sin(inputs()), 68163876);
    }
}
