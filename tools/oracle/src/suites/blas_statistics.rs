//! WP 8.3-P06: upstream `base/statistics.rs` (suite `statistics`) and `base/blas.rs` (suite
//! `blas`) on a sample of the static shapes (one per kind and kernel size, `tools/shapegen`
//! DESIGN §3.3 Tier B). Every Cairo kernel is also modelled bit for bit on integer raws in
//! `tools/shapegen/tests_stats.py` / `tests_blas.py` (all 36 shapes); these vectors check the
//! agreement with upstream's f64 results:
//!
//! - means: the exact sum divided by the count, correctly rounded (1 ulp from the floor of the
//!   f64 result);
//! - variances: the squares of the exact differences summed exactly and floored once, then
//!   divided by the count correctly rounded (and upstream's own f64 roundings);
//! - products: one floored product per step;
//! - `alpha * sum + beta * c` (`gemv`, `gemm`, `ger`, `axcpy`, `quadform*`...): the exact sum times
//!   `alpha` floored once plus `beta * c` floored once; `axpy` is one fused sum of two products;
//! - `tr_dot` is one fused sum of products: the exact floor (i128 path).

use super::{flat, ring, sm, sv, Ring};
use crate::engine::{fm, fs, fv, im, is, iv, Op, Suite, Tol};
use crate::gen::Dist;
use nalgebra::{Matrix2, Matrix2x3, Matrix2x4, Matrix3, Matrix3x2, Matrix4, Matrix4x3};

/// Means: the exact sum divided by the count, correctly rounded (half an ulp), against the floor
/// of upstream's once-rounded f64 quotient.
const MEAN: Tol = Tol::Ulp(2);
/// Variances: the mean and the quotient correctly rounded, the sum of squares floored once (below
/// 2 ulp here); upstream's f64 rounds every square and every partial sum (a few ulp of Q32.32 on
/// `medium` inputs, whose squares reach 1e6).
const VARIANCE: Tol = Tol::Ulp(6);
/// Sequential floored products: one rounding per factor, amplified by the remaining factors.
const PRODUCT: Tol = Tol::Sens { k: 2.0, base: 2.0 };
/// `alpha * sum + beta * c`: two floors here (below 2 ulp), against the floor of the f64 result.
/// On `small` / `unit` inputs only: on `medium` ones the f64 triple products (1e9) carry hundreds
/// of Q32.32 ulp of their own rounding.
const BLAS: Tol = Tol::Ulp(3);
/// `quadform*`: the intermediate product floored once per entry (an error amplified by `alpha`
/// and the other factor), then `BLAS`.
const QUADFORM: Tol = Tol::Sens { k: 1.0, base: 3.0 };
const SMALL: [Dist; 2] = [Dist::Small, Dist::Unit];

fn tr_dot<T: Ring>(x: &[T]) -> Vec<T> {
    vec![sm::<T, 2, 3>(x).tr_dot(&sm::<T, 3, 2>(&x[6..]))]
}

fn statistics_ops() -> Vec<Op> {
    let m23 = |x: &[f64]| -> Matrix2x3<f64> { sm(x) };
    vec![
        Op::new("matrix2x3_mean", "m.mean()")
            .input(im("m", 2, 3))
            .out(fs("mean"))
            .tol(MEAN)
            .eval(move |x| Some(vec![m23(x).mean()])),
        Op::new("matrix2x3_variance", "m.variance()")
            .input(im("m", 2, 3))
            .out(fs("variance"))
            .dists(&Dist::NO_LARGE)
            .tol(VARIANCE)
            .eval(move |x| Some(vec![m23(x).variance()])),
        Op::new("vector5_variance", "v.variance()")
            .input(iv("v", 5))
            .out(fs("variance"))
            .dists(&Dist::NO_LARGE)
            .tol(VARIANCE)
            .eval(|x| Some(vec![sv::<f64, 5>(x).variance()])),
        Op::new("matrix6_mean", "m.mean()")
            .input(im("m", 6, 6))
            .out(fs("mean"))
            .tol(MEAN)
            .eval(|x| Some(vec![sm::<f64, 6, 6>(x).mean()])),
        Op::new("matrix6_variance", "m.variance()")
            .input(im("m", 6, 6))
            .out(fs("variance"))
            .dists(&Dist::NO_LARGE)
            .tol(VARIANCE)
            .eval(|x| Some(vec![sm::<f64, 6, 6>(x).variance()])),
        Op::new("matrix2x3_product", "m.product()")
            .input(im("m", 2, 3))
            .out(fs("product"))
            .dists(&SMALL)
            .tol(PRODUCT)
            .eval(move |x| Some(vec![m23(x).product()])),
        Op::new("matrix3_row_mean", "m.row_mean()")
            .input(im("m", 3, 3))
            .out(fv("row_mean", 3))
            .tol(MEAN)
            .eval(|x| Some(flat(&sm::<f64, 3, 3>(x).row_mean()))),
        Op::new("matrix3x2_row_variance", "m.row_variance()")
            .input(im("m", 3, 2))
            .out(fv("row_variance", 2))
            .dists(&Dist::NO_LARGE)
            .tol(VARIANCE)
            .eval(|x| Some(flat(&sm::<f64, 3, 2>(x).row_variance()))),
        Op::new("matrix2x3_column_mean", "m.column_mean()")
            .input(im("m", 2, 3))
            .out(fv("column_mean", 2))
            .tol(MEAN)
            .eval(move |x| Some(flat(&m23(x).column_mean()))),
        Op::new("matrix2x3_column_variance", "m.column_variance()")
            .input(im("m", 2, 3))
            .out(fv("column_variance", 2))
            .dists(&Dist::NO_LARGE)
            .tol(VARIANCE)
            .eval(move |x| Some(flat(&m23(x).column_variance()))),
        Op::new("matrix4_column_variance", "m.column_variance()")
            .input(im("m", 4, 4))
            .out(fv("column_variance", 4))
            .dists(&Dist::NO_LARGE)
            .tol(VARIANCE)
            .eval(|x| Some(flat(&sm::<f64, 4, 4>(x).column_variance()))),
    ]
}

fn blas_ops() -> Vec<Op> {
    vec![
        Op::new("matrix2x3_tr_dot", "a.tr_dot(&b)")
            .input(im("a", 2, 3))
            .input(im("b", 3, 2))
            .out(fs("tr_dot"))
            .ring(ring!(tr_dot)),
        Op::new("vector3_axpy", "y.axpy(a, &x, b)")
            .input(is("a"))
            .input(iv("x", 3))
            .input(is("b"))
            .input(iv("y", 3))
            .out(fv("y", 3))
            .dists(&SMALL)
            .tol(BLAS)
            .eval(|x| {
                let mut y = sv::<f64, 3>(&x[5..]);
                y.axpy(x[0], &sv::<f64, 3>(&x[1..]), x[4]);
                Some(flat(&y))
            }),
        Op::new("vector4_axcpy", "y.axcpy(a, &x, c, b)")
            .input(is("a"))
            .input(iv("x", 4))
            .input(is("c"))
            .input(is("b"))
            .input(iv("y", 4))
            .out(fv("y", 4))
            .dists(&SMALL)
            .tol(BLAS)
            .eval(|x| {
                let mut y = sv::<f64, 4>(&x[7..]);
                y.axcpy(x[0], &sv::<f64, 4>(&x[1..]), x[5], x[6]);
                Some(flat(&y))
            }),
        Op::new("vector2_gemv", "y.gemv(alpha, &a, &x, beta), a: 2x4")
            .input(is("alpha"))
            .input(im("a", 2, 4))
            .input(iv("x", 4))
            .input(is("beta"))
            .input(iv("y", 2))
            .out(fv("y", 2))
            .dists(&SMALL)
            .tol(BLAS)
            .eval(|x| {
                let mut y = sv::<f64, 2>(&x[14..]);
                let a: Matrix2x4<f64> = sm(&x[1..]);
                y.gemv(x[0], &a, &sv::<f64, 4>(&x[9..]), x[13]);
                Some(flat(&y))
            }),
        Op::new("vector2_gemv_tr", "y.gemv_tr(alpha, &a, &x, beta), a: 3x2")
            .input(is("alpha"))
            .input(im("a", 3, 2))
            .input(iv("x", 3))
            .input(is("beta"))
            .input(iv("y", 2))
            .out(fv("y", 2))
            .dists(&SMALL)
            .tol(BLAS)
            .eval(|x| {
                let mut y = sv::<f64, 2>(&x[11..]);
                let a: Matrix3x2<f64> = sm(&x[1..]);
                y.gemv_tr(x[0], &a, &sv::<f64, 3>(&x[7..]), x[10]);
                Some(flat(&y))
            }),
        Op::new(
            "vector3_sygemv",
            "y.sygemv(alpha, &a, &x, beta) (lower triangle of a)",
        )
        .input(is("alpha"))
        .input(im("a", 3, 3))
        .input(iv("x", 3))
        .input(is("beta"))
        .input(iv("y", 3))
        .out(fv("y", 3))
        .dists(&SMALL)
        .tol(BLAS)
        .eval(|x| {
            let mut y = sv::<f64, 3>(&x[14..]);
            let a: Matrix3<f64> = sm(&x[1..]);
            y.sygemv(x[0], &a, &sv::<f64, 3>(&x[10..]), x[13]);
            Some(flat(&y))
        }),
        Op::new("matrix2x3_ger", "m.ger(alpha, &x, &y, beta)")
            .input(is("alpha"))
            .input(iv("x", 2))
            .input(iv("y", 3))
            .input(is("beta"))
            .input(im("m", 2, 3))
            .out(fm("m", 2, 3))
            .dists(&SMALL)
            .tol(BLAS)
            .eval(|x| {
                let mut m: Matrix2x3<f64> = sm(&x[7..]);
                m.ger(x[0], &sv::<f64, 2>(&x[1..]), &sv::<f64, 3>(&x[3..]), x[6]);
                Some(flat(&m))
            }),
        Op::new(
            "matrix3_syger",
            "m.syger(alpha, &x, &y, beta) (lower triangle)",
        )
        .input(is("alpha"))
        .input(iv("x", 3))
        .input(iv("y", 3))
        .input(is("beta"))
        .input(im("m", 3, 3))
        .out(fm("m", 3, 3))
        .dists(&SMALL)
        .tol(BLAS)
        .eval(|x| {
            let mut m: Matrix3<f64> = sm(&x[8..]);
            m.syger(x[0], &sv::<f64, 3>(&x[1..]), &sv::<f64, 3>(&x[4..]), x[7]);
            Some(flat(&m))
        }),
        Op::new(
            "matrix2x3_gemm",
            "c.gemm(alpha, &a, &b, beta), a: 2x4, b: 4x3",
        )
        .input(is("alpha"))
        .input(im("a", 2, 4))
        .input(im("b", 4, 3))
        .input(is("beta"))
        .input(im("c", 2, 3))
        .out(fm("c", 2, 3))
        .dists(&SMALL)
        .tol(BLAS)
        .eval(|x| {
            let mut c: Matrix2x3<f64> = sm(&x[22..]);
            let a: Matrix2x4<f64> = sm(&x[1..]);
            let b: Matrix4x3<f64> = sm(&x[9..]);
            c.gemm(x[0], &a, &b, x[21]);
            Some(flat(&c))
        }),
        Op::new(
            "matrix3_gemm_tr",
            "c.gemm_tr(alpha, &a, &b, beta), a: 2x3, b: 2x3",
        )
        .input(is("alpha"))
        .input(im("a", 2, 3))
        .input(im("b", 2, 3))
        .input(is("beta"))
        .input(im("c", 3, 3))
        .out(fm("c", 3, 3))
        .dists(&SMALL)
        .tol(BLAS)
        .eval(|x| {
            let mut c: Matrix3<f64> = sm(&x[14..]);
            let a: Matrix2x3<f64> = sm(&x[1..]);
            let b: Matrix2x3<f64> = sm(&x[7..]);
            c.gemm_tr(x[0], &a, &b, x[13]);
            Some(flat(&c))
        }),
        Op::new("matrix4_gemm", "c.gemm(alpha, &a, &b, beta), a, b: 4x4")
            .input(is("alpha"))
            .input(im("a", 4, 4))
            .input(im("b", 4, 4))
            .input(is("beta"))
            .input(im("c", 4, 4))
            .out(fm("c", 4, 4))
            .dists(&SMALL)
            .tol(BLAS)
            .eval(|x| {
                let mut c: Matrix4<f64> = sm(&x[34..]);
                let a: Matrix4<f64> = sm(&x[1..]);
                let b: Matrix4<f64> = sm(&x[17..]);
                c.gemm(x[0], &a, &b, x[33]);
                Some(flat(&c))
            }),
        Op::new(
            "matrix2_quadform",
            "m.quadform(alpha, &mid, &rhs, beta), mid: 3x3, rhs: 3x2",
        )
        .input(is("alpha"))
        .input(im("mid", 3, 3))
        .input(im("rhs", 3, 2))
        .input(is("beta"))
        .input(im("m", 2, 2))
        .out(fm("m", 2, 2))
        .dists(&SMALL)
        .tol(QUADFORM)
        .eval(|x| {
            let mut m: Matrix2<f64> = sm(&x[17..]);
            let mid: Matrix3<f64> = sm(&x[1..]);
            let rhs: Matrix3x2<f64> = sm(&x[10..]);
            m.quadform(x[0], &mid, &rhs, x[16]);
            Some(flat(&m))
        }),
        Op::new(
            "matrix3_quadform_tr",
            "m.quadform_tr(alpha, &lhs, &mid, beta), lhs: 3x2, mid: 2x2",
        )
        .input(is("alpha"))
        .input(im("lhs", 3, 2))
        .input(im("mid", 2, 2))
        .input(is("beta"))
        .input(im("m", 3, 3))
        .out(fm("m", 3, 3))
        .dists(&SMALL)
        .tol(QUADFORM)
        .eval(|x| {
            let mut m: Matrix3<f64> = sm(&x[12..]);
            let lhs: Matrix3x2<f64> = sm(&x[1..]);
            let mid: Matrix2<f64> = sm(&x[7..]);
            m.quadform_tr(x[0], &lhs, &mid, x[11]);
            Some(flat(&m))
        }),
    ]
}

pub fn suites() -> Vec<Suite> {
    vec![
        Suite {
            name: "statistics",
            description: "Statistics (base/statistics.rs): mean, variance, product and their \
                          row_* / column_* forms on a sample of the static shapes",
            ops: statistics_ops(),
        },
        Suite {
            name: "blas",
            description: "BLAS-like kernels (base/blas.rs): tr_dot, axpy, axcpy, gemv, gemv_tr, \
                          sygemv, ger, syger, gemm, gemm_tr, quadform, quadform_tr on a sample \
                          of the static shapes",
            ops: blas_ops(),
        },
    ]
}
