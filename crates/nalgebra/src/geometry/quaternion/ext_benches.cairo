//! Gas benchmarks of the WP 8.4-P08 completion of `Quaternion` (`bench_quaternion_<op>__<variant>`,
//! net = raw - `baseline` of the group), and the alternative implementations that lost (`alt_*`),
//! kept as evidence together with the tests showing why (AGENTS.md rule 8; `ext_tests.cairo`).
//!
//! The inputs are `s = 0.5 + 0.25i - 0.5j + 0.75k` and `t = -0.25 + 0.5i + 0.125j - 0.375k`.
//! Expected values are the results of the kernels themselves, all of which are checked against
//! upstream nalgebra in `ext_tests.cairo`.

use core::num::traits::One;
use fixed::Fixed;
use nalgebra_testing::black_box;
use simba::scalar::Real;
use crate::base::matrix_test_utils::{fx, qt, u3t, v3t};
use super::ext_tests::alt_ln_acos;
use super::{Quaternion, QuaternionTrait, QuaternionTranscendentalTrait};

/// `0.5 + 0.25i - 0.5j + 0.75k`.
fn s() -> Quaternion<Fixed> {
    qt((0x80000000, 0x40000000, -0x80000000, 0xc0000000))
}

/// `-0.25 + 0.5i + 0.125j - 0.375k`.
fn t() -> Quaternion<Fixed> {
    qt((-0x40000000, 0x80000000, 0x20000000, -0x60000000))
}

// --- alternative implementations (losers)

/// `inner` as upstream writes it, `(a * b + b * a) / 2`: two Hamilton products and a halving,
/// where the reduced form is one fused kernel per component (`test_inner_alt_products_agrees`).
pub(crate) fn alt_inner_products(a: Quaternion<Fixed>, b: Quaternion<Fixed>) -> Quaternion<Fixed> {
    (a * b + b * a).half()
}

/// `sinh` as upstream writes it, `(exp(q) - exp(-q)) / 2`: two full quaternion exponentials
/// (`test_sinh_alt_exp_difference_agrees`).
pub(crate) fn alt_sinh_exp_difference(q: Quaternion<Fixed>) -> Quaternion<Fixed> {
    (q.exp() - (-q).exp()).half()
}


// --- magnitude

#[test]
#[inline(never)]
fn bench_quaternion_magnitude__baseline() {
    let _x = black_box(s());
    let e = black_box(fx(4555500749));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_magnitude__norm4() {
    let x = black_box(s());
    let e = black_box(fx(4555500749));
    assert!(x.magnitude() == e);
}

// --- magnitude_squared

#[test]
#[inline(never)]
fn bench_quaternion_magnitude_squared__baseline() {
    let _x = black_box(s());
    let e = black_box(fx(4831838208));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_magnitude_squared__fused() {
    let x = black_box(s());
    let e = black_box(fx(4831838208));
    assert!(x.magnitude_squared() == e);
}

// --- is_pure

#[test]
#[inline(never)]
fn bench_quaternion_is_pure__baseline() {
    let _x = black_box(s());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_is_pure__compare() {
    let x = black_box(s());
    let e = black_box(false);
    assert!(x.is_pure() == e);
}

// --- pure

#[test]
#[inline(never)]
fn bench_quaternion_pure__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((0, 1073741824, -2147483648, 3221225472)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_pure__struct() {
    let x = black_box(s());
    let e = black_box(qt((0, 1073741824, -2147483648, 3221225472)));
    assert!(x.pure() == e);
}

// --- cast

#[test]
#[inline(never)]
fn bench_quaternion_cast__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((2147483648, 1073741824, -2147483648, 3221225472)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_cast__into() {
    let x = black_box(s());
    let e = black_box(qt((2147483648, 1073741824, -2147483648, 3221225472)));
    assert!({
        let r: Quaternion<Fixed> = x.cast();
        r
    } == e);
}

// --- half

#[test]
#[inline(never)]
fn bench_quaternion_half__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((1073741824, 536870912, -1073741824, 1610612736)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_half__div4() {
    let x = black_box(s());
    let e = black_box(qt((1073741824, 536870912, -1073741824, 1610612736)));
    assert!(x.half() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_half__alt_scale_half() {
    let x = black_box(s());
    let e = black_box(qt((1073741824, 536870912, -1073741824, 1610612736)));
    assert!(x.scale(Real::HALF) == e);
}

// --- squared

#[test]
#[inline(never)]
fn bench_quaternion_squared__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((-2684354560, 1073741824, -2147483648, 3221225472)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_squared__reduced() {
    let x = black_box(s());
    let e = black_box(qt((-2684354560, 1073741824, -2147483648, 3221225472)));
    assert!(x.squared() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_squared__alt_mul() {
    let x = black_box(s());
    let e = black_box(qt((-2684354560, 1073741824, -2147483648, 3221225472)));
    assert!(x * x == e);
}

// --- inner

#[test]
#[inline(never)]
fn bench_quaternion_inner__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(qt((402653184, 805306368, 805306368, -1610612736)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_inner__reduced() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((402653184, 805306368, 805306368, -1610612736)));
    assert!(x.inner(y) == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_inner__alt_products() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((402653184, 805306368, 805306368, -1610612736)));
    assert!(alt_inner_products(x, y) == e);
}

// --- outer

#[test]
#[inline(never)]
fn bench_quaternion_outer__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(qt((0, 402653184, 2013265920, 1207959552)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_outer__cross() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((0, 402653184, 2013265920, 1207959552)));
    assert!(x.outer(y) == e);
}

// --- right_div

#[test]
#[inline(never)]
fn bench_quaternion_right_div__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(qt((-3149642684, -3722304990, -3722304990, -2576980378)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_right_div__fused() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((-3149642684, -3722304990, -3722304990, -2576980378)));
    assert!(x.right_div(y).unwrap() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_right_div__alt_inverse_then_mul() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((-3149642684, -3722304991, -3722304990, -2576980378)));
    assert!(x * y.try_inverse().unwrap() == e);
}

// --- left_div

#[test]
#[inline(never)]
fn bench_quaternion_left_div__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(qt((-3149642684, -2004318071, 4867629602, 2576980378)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_left_div__fused() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((-3149642684, -2004318071, 4867629602, 2576980378)));
    assert!(x.left_div(y).unwrap() == e);
}

// --- project

#[test]
#[inline(never)]
fn bench_quaternion_project__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(qt((2147483648, -644245094, 536870912, 1825361101)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_project__right_div() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((2147483648, -644245094, 536870912, 1825361101)));
    assert!(x.project(y).unwrap() == e);
}

// --- reject

#[test]
#[inline(never)]
fn bench_quaternion_reject__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(qt((0, 1717986918, -2684354560, 1395864371)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_reject__right_div() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(qt((0, 1717986918, -2684354560, 1395864371)));
    assert!(x.reject(y).unwrap() == e);
}

// --- sqrt

#[test]
#[inline(never)]
fn bench_quaternion_sqrt__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((3794014942, 607758020, -1215516040, 1823274060)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sqrt__algebraic() {
    let x = black_box(s());
    let e = black_box(qt((3794014942, 607758020, -1215516040, 1823274060)));
    assert!(x.sqrt() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sqrt__alt_powf() {
    let x = black_box(s());
    let e = black_box(qt((3794014941, 607758019, -1215516041, 1823274058)));
    assert!(x.powf(Real::HALF) == e);
}

// --- relative_eq

#[test]
#[inline(never)]
fn bench_quaternion_relative_eq__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_relative_eq__components() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(false);
    assert!(x.relative_eq(y, 4, fx(0x100)) == e);
}

// --- ulps_eq

#[test]
#[inline(never)]
fn bench_quaternion_ulps_eq__baseline() {
    let _x = black_box(s());
    let _y = black_box(t());
    let e = black_box(false);
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_ulps_eq__components() {
    let x = black_box(s());
    let y = black_box(t());
    let e = black_box(false);
    assert!(x.ulps_eq(y, 4, 4) == e);
}

// --- index

#[test]
#[inline(never)]
fn bench_quaternion_index__baseline() {
    let _x = black_box(s());
    let e = black_box(fx(3221225472));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_index__match() {
    let mut x = black_box(s());
    let e = black_box(fx(3221225472));
    assert!(x[2] == e);
}

// --- one

#[test]
#[inline(never)]
fn bench_quaternion_one__baseline() {
    let e = black_box(qt((4294967296, 0, 0, 0)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_one__const() {
    let e = black_box(qt((4294967296, 0, 0, 0)));
    assert!(One::<Quaternion<Fixed>>::one() == e);
}

// --- from_array

#[test]
#[inline(never)]
fn bench_quaternion_from_array__baseline() {
    let _a = black_box([fx(1), fx(2), fx(3), fx(4)]);
    let e = black_box(qt((4, 1, 2, 3)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_from_array__into() {
    let a = black_box([fx(1), fx(2), fx(3), fx(4)]);
    let e = black_box(qt((4, 1, 2, 3)));
    assert!({
        let r: Quaternion<Fixed> = a.into();
        r
    } == e);
}

// --- exp

#[test]
#[inline(never)]
fn bench_quaternion_exp__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((4202588264, 1523194506, -3046389014, 4569583520)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_exp__sin_cos() {
    let x = black_box(s());
    let e = black_box(qt((4202588264, 1523194506, -3046389014, 4569583520)));
    assert!(x.exp() == e);
}

// --- exp_eps

#[test]
#[inline(never)]
fn bench_quaternion_exp_eps__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((4202588264, 1523194506, -3046389014, 4569583520)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_exp_eps__sin_cos() {
    let x = black_box(s());
    let e = black_box(qt((4202588264, 1523194506, -3046389014, 4569583520)));
    assert!(x.exp_eps(fx(0x1000)) == e);
}

// --- ln

#[test]
#[inline(never)]
fn bench_quaternion_ln__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((252937142, 1239609436, -2479218873, 3718828308)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_ln__atan2() {
    let x = black_box(s());
    let e = black_box(qt((252937142, 1239609436, -2479218873, 3718828308)));
    assert!(x.ln() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_ln__alt_acos() {
    let x = black_box(s());
    let e = black_box(qt((252937142, 1239609436, -2479218871, 3718828306)));
    assert!(alt_ln_acos(x) == e);
}

// --- powf

#[test]
#[inline(never)]
fn bench_quaternion_powf__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((-230145600, 1252382744, -2504765491, 3757148234)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_powf__exp_ln() {
    let x = black_box(s());
    let e = black_box(qt((-230145600, 1252382744, -2504765491, 3757148234)));
    assert!(x.powf(fx(0x180000000)) == e);
}

// --- polar_decomposition

#[test]
#[inline(never)]
fn bench_quaternion_polar_decomposition__baseline() {
    let _x = black_box(s());
    let e = black_box((fx(4555500749), fx(4638193803), v3t((1147878294, -2295756587, 3443634881))));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_polar_decomposition__atan2() {
    let x = black_box(s());
    let e = black_box((fx(4555500749), fx(4638193803), v3t((1147878294, -2295756587, 3443634881))));
    assert!({
        let (n, a, v) = x.polar_decomposition();
        (n, a, v.unwrap().value)
    } == e);
}

// --- from_polar_decomposition

#[test]
#[inline(never)]
fn bench_quaternion_from_polar_decomposition__baseline() {
    let _n = black_box(fx(0x200000000));
    let _a = black_box(fx(0x80000000));
    let _u = black_box(u3t((1393471396, -2090207096, 3483678492)));
    let e = black_box(qt((7538376806, 1336131549, -2004197326, 3340328874)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_from_polar_decomposition__sin_cos() {
    let n = black_box(fx(0x200000000));
    let a = black_box(fx(0x80000000));
    let u = black_box(u3t((1393471396, -2090207096, 3483678492)));
    let e = black_box(qt((7538376806, 1336131549, -2004197326, 3340328874)));
    assert!(QuaternionTranscendentalTrait::from_polar_decomposition(n, a, u) == e);
}

// --- cos

#[test]
#[inline(never)]
fn bench_quaternion_cos__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((5542011728, -593204889, 1186409777, -1779614666)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_cos__upstream() {
    let x = black_box(s());
    let e = black_box(qt((5542011728, -593204889, 1186409777, -1779614666)));
    assert!(x.cos() == e);
}

// --- sin

#[test]
#[inline(never)]
fn bench_quaternion_sin__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((3027614805, 1085854265, -2171708531, 3257562795)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sin__upstream() {
    let x = black_box(s());
    let e = black_box(qt((3027614805, 1085854265, -2171708531, 3257562795)));
    assert!(x.sin() == e);
}

// --- tan

#[test]
#[inline(never)]
fn bench_quaternion_tan__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((935290183, 941630498, -1883260998, 2824891495)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_tan__upstream() {
    let x = black_box(s());
    let e = black_box(qt((935290183, 941630498, -1883260998, 2824891495)));
    assert!(x.tan() == e);
}

// --- sinh

#[test]
#[inline(never)]
fn bench_quaternion_sinh__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((1328271221, 1041773225, -2083546451, 3125319675)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sinh__closed_form() {
    let x = black_box(s());
    let e = black_box(qt((1328271221, 1041773225, -2083546451, 3125319675)));
    assert!(x.sinh() == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_sinh__alt_exp_difference() {
    let x = black_box(s());
    let e = black_box(qt((1328271222, 1041773225, -2083546450, 3125319676)));
    assert!(alt_sinh_exp_difference(x) == e);
}

// --- cosh

#[test]
#[inline(never)]
fn bench_quaternion_cosh__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((2874317042, 481421281, -962842563, 1444263844)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_cosh__closed_form() {
    let x = black_box(s());
    let e = black_box(qt((2874317042, 481421281, -962842563, 1444263844)));
    assert!(x.cosh() == e);
}

// --- tanh

#[test]
#[inline(never)]
fn bench_quaternion_tanh__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((4045956961, 879016513, -1758033029, 2637049542)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_tanh__upstream() {
    let x = black_box(s());
    let e = black_box(qt((4045956961, 879016513, -1758033029, 2637049542)));
    assert!(x.tanh() == e);
}

// --- acos

#[test]
#[inline(never)]
fn bench_quaternion_acos__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((5194458874, -1011709611, 2023419218, -3035128826)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_acos__upstream() {
    let x = black_box(s());
    let e = black_box(qt((5194458874, -1011709611, 2023419218, -3035128826)));
    assert!(x.acos() == e);
}

// --- asin

#[test]
#[inline(never)]
fn bench_quaternion_asin__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((1552059980, 1011709607, -2023419219, 3035128831)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_asin__upstream() {
    let x = black_box(s());
    let e = black_box(qt((1552059980, 1011709607, -2023419219, 3035128831)));
    assert!(x.asin() == e);
}

// --- atan

#[test]
#[inline(never)]
fn bench_quaternion_atan__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((3640309744, 790600544, -1581201085, 2371801624)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_atan__upstream() {
    let x = black_box(s());
    let e = black_box(qt((3640309744, 790600544, -1581201085, 2371801624)));
    assert!(x.atan() == e);
}

// --- asinh

#[test]
#[inline(never)]
fn bench_quaternion_asinh__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((2985936747, 968972398, -1937944799, 2906917197)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_asinh__upstream() {
    let x = black_box(s());
    let e = black_box(qt((2985936747, 968972398, -1937944799, 2906917197)));
    assert!(x.asinh() == e);
}

// --- acosh

#[test]
#[inline(never)]
fn bench_quaternion_acosh__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((3785470731, 1388277528, -2776555061, 4164832587)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_acosh__upstream() {
    let x = black_box(s());
    let e = black_box(qt((3785470731, 1388277528, -2776555061, 4164832587)));
    assert!(x.acosh() == e);
}

// --- atanh

#[test]
#[inline(never)]
fn bench_quaternion_atanh__baseline() {
    let _x = black_box(s());
    let e = black_box(qt((1096989674, 939832513, -1879665026, 2819497539)));
    assert!(e == e);
}

#[test]
#[inline(never)]
fn bench_quaternion_atanh__upstream() {
    let x = black_box(s());
    let e = black_box(qt((1096989674, 939832513, -1879665026, 2819497539)));
    assert!(x.atanh() == e);
}
