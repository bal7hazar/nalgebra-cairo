//! Scalar functions for the transcendental work package, evaluated by the pure-Rust `libm`
//! (the same functions upstream nalgebra calls through `simba` with `libm-force`).

use crate::engine::{fangle, fs, is, with, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use std::f64::consts::{FRAC_PI_2, PI};

/// `4 + |x| / 8` ulp: 2 ulp of polynomial error (DESIGN D6) with margin, plus a range reduction
/// by a Q32.32 `2*pi` constant (0.5 ulp of error per turn).
fn trig_budget(x: f64) -> f64 {
    4.0 + x.abs() / 8.0
}

pub fn suite() -> Suite {
    let pos = |name: &str| with(fs(name), Gen::Pos);
    let ops = vec![
        Op::new("scalar_sqrt", "x.sqrt()")
            .input(pos("x"))
            .out(fs("sqrt"))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(vec![libm::sqrt(x[0])]))
            .special(&[0.0], 0)
            .special(&[1.0], 0)
            .special(&[4.0], 0)
            .special(&[0.25], 0)
            .special(&[2.0], 1),
        Op::new("scalar_inv_sqrt", "1 / x.sqrt()")
            .input(pos("x"))
            .out(fs("inv_sqrt"))
            .tol(Tol::Model(
                Box::new(|x, _| 2.0 + 1.0 / x[0]),
                "ceil(2 + 1 / x): sqrt rounded to 1 ulp, then one division",
            ))
            .eval(|x| Some(vec![1.0 / libm::sqrt(x[0])]))
            .special(&[1.0], 0)
            .special(&[4.0], 0),
        Op::new("scalar_recip", "x.recip()")
            .input(is("x"))
            .out(fs("recip"))
            .tol(Tol::Ulp(1))
            .eval(|x| Some(vec![1.0 / x[0]]))
            .special(&[1.0], 0)
            .special(&[-2.0], 0),
        Op::new("scalar_sin", "x.sin()")
            .input(is("x"))
            .out(fs("sin"))
            .tol(Tol::Model(
                Box::new(|x, _| trig_budget(x[0])),
                "ceil(4 + |x| / 8): polynomial error + range reduction by a Q32.32 2*pi",
            ))
            .eval(|x| Some(vec![libm::sin(x[0])]))
            .special(&[0.0], 0)
            .special(&[FRAC_PI_2], 4)
            .special(&[-FRAC_PI_2], 4)
            .special(&[PI], 4),
        Op::new("scalar_cos", "x.cos()")
            .input(is("x"))
            .out(fs("cos"))
            .tol(Tol::Model(
                Box::new(|x, _| trig_budget(x[0])),
                "ceil(4 + |x| / 8): polynomial error + range reduction by a Q32.32 2*pi",
            ))
            .eval(|x| Some(vec![libm::cos(x[0])]))
            .special(&[0.0], 0)
            .special(&[FRAC_PI_2], 4)
            .special(&[PI], 4)
            .special(&[-PI], 4),
        Op::new("scalar_tan", "x.tan()")
            .input(is("x"))
            .out(fs("tan"))
            .dists(&Dist::NO_LARGE)
            .tol(Tol::Model(
                Box::new(|x, y| 2.0 * trig_budget(x[0]) * (1.0 + y[0] * y[0])),
                "ceil(2 * (4 + |x| / 8) * (1 + tan^2)): sin / cos error amplified by the pole",
            ))
            .eval(|x| Some(vec![libm::tan(x[0])]))
            .special(&[0.0], 0),
        Op::new("scalar_atan2", "y.atan2(x)")
            .input(is("y"))
            .input(is("x"))
            .out(fangle("atan2"))
            .tol(Tol::Ulp(12))
            .eval(|x| Some(vec![libm::atan2(x[0], x[1])]))
            .special(&[0.0, 1.0], 0)
            .special(&[1.0, 0.0], 12)
            .special(&[0.0, -1.0], 12)
            .special(&[-1.0, 0.0], 12)
            .special(&[1.0, 1.0], 12)
            .special(&[-1.0, -1.0], 12)
            .special(&[1.0, -1.0], 12),
        Op::new("scalar_atan", "x.atan()")
            .input(is("x"))
            .out(fangle("atan"))
            .tol(Tol::Ulp(12))
            .eval(|x| Some(vec![libm::atan(x[0])]))
            .special(&[0.0], 0)
            .special(&[1.0], 12)
            .special(&[-1.0], 12),
        Op::new("scalar_asin", "x.asin()")
            .input(with(fs("x"), Gen::Range(-1.0, 1.0)))
            .out(fangle("asin"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(8))
            .eval(|x| Some(vec![libm::asin(x[0])]))
            .special(&[0.0], 0)
            .special(&[1.0], 8)
            .special(&[-1.0], 8)
            .special(&[0.5], 8),
        Op::new("scalar_acos", "x.acos()")
            .input(with(fs("x"), Gen::Range(-1.0, 1.0)))
            .out(fangle("acos"))
            .dists(&Dist::UNIT)
            .tol(Tol::Ulp(8))
            .eval(|x| Some(vec![libm::acos(x[0])]))
            .special(&[1.0], 0)
            .special(&[0.0], 8)
            .special(&[-1.0], 8)
            .special(&[-0.5], 8),
        Op::new("scalar_exp", "x.exp()")
            .input(with(fs("x"), Gen::Range(-20.0, 21.0)))
            .out(fs("exp"))
            .dists(&Dist::UNIT)
            .cap(1 << 36)
            .tol(Tol::Model(
                Box::new(|_, y| 4.0 + 4.0 * y[0]),
                "ceil(4 + 4 * exp(x)): relative accuracy of 4 ulp",
            ))
            .eval(|x| Some(vec![libm::exp(x[0])]))
            .special(&[0.0], 0)
            .special(&[1.0], 15)
            .special(&[-1.0], 6),
        Op::new("scalar_ln", "x.ln()")
            .input(pos("x"))
            .out(fs("ln"))
            .tol(Tol::Ulp(8))
            .eval(|x| Some(vec![libm::log(x[0])]))
            .special(&[1.0], 0)
            .special(&[2.0], 8)
            .special(&[0.5], 8),
    ];
    Suite {
        name: "scalar",
        description: "Scalar functions (libm, f64): sqrt, inv_sqrt, recip, sin, cos, tan, atan2, \
                      atan, asin, acos, exp, ln",
        ops,
    }
}
