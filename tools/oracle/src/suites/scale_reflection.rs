//! WP 8.4-P10: `Scale1..6` (products, inverses, point transforms) and `Reflection1..6`
//! (`new_containing_point`, `reflect`, `reflect_with_sign`, `reflect_rows`,
//! `reflect_rows_with_sign`), suite `scale_reflection`.
//!
//! The scale products are exact (one floored product per component, evaluated on `i128` raws);
//! the inverses and the inverse point transform are one correctly rounded division each; the
//! reflections are evaluated by upstream in f64 and their tolerance follows the kernels of the
//! Cairo side: one exactly accumulated dot product (rounded once), then one fused update per entry.

use super::{flat, ring, sm, sv, Ring};
use crate::engine::{fm, fs, fv, im, iu, iv, with, Input, Op, Suite, Tol};
use crate::gen::Gen;
use nalgebra::{ArrayStorage, Const, Point, Reflection, SMatrix, SVector, Scale, Unit};

/// One dot product rounded once (1 ulp, times the axis entries <= 1), a factor of at most 2 and
/// one fused update per entry (1 ulp): a constant budget, whatever the magnitude of the matrix
/// (the sign, in [-2, 2], doubles it).
const REFLECT: Tol = Tol::Ulp(8);

// Inputs ---------------------------------------------------------------------------------------------

/// The `n` factors of a scale, each in [0.25, 4] (invertible, and their inverses fit).
fn ifactors(name: &str, n: usize) -> Input {
    with(fv(name, n), Gen::Group(vec![Gen::Range(0.25, 4.0); n]))
}

/// A reflection bias in [-2, 2].
fn ibias(name: &str) -> Input {
    with(fs(name), Gen::Range(-2.0, 2.0))
}

/// The sign of `reflect_with_sign`, in [-2, 2].
fn isign(name: &str) -> Input {
    with(fs(name), Gen::Range(-2.0, 2.0))
}

// Scale ----------------------------------------------------------------------------------------------

fn scale_vec<T: Ring, const N: usize>(x: &[T]) -> Scale<T, N> {
    Scale::from(sv::<T, N>(x))
}

fn scale_mul<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(scale_vec::<T, N>(x) * scale_vec::<T, N>(&x[N..])).vector)
}

fn scale_transform_point<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(scale_vec::<T, N>(x) * Point::<T, N>::from(sv::<T, N>(&x[N..]))).coords)
}

fn scale_mul_vector<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(scale_vec::<T, N>(x) * sv::<T, N>(&x[N..])))
}

fn scale_scale<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(scale_vec::<T, N>(x) * x[N]).vector)
}

fn scale_ops<const N: usize>() -> Vec<Op> {
    let p = format!("scale{N}");
    let s = |x: &[f64]| Scale::<f64, N>::from(sv::<f64, N>(x));
    let pt = |x: &[f64]| Point::<f64, N>::from(sv::<f64, N>(x));
    vec![
        Op::new(format!("{p}_mul"), "a * b (component-wise product)")
            .input(iv("a", N))
            .input(iv("b", N))
            .out(fv("product", N))
            .ring(ring!(scale_mul, N)),
        Op::new(
            format!("{p}_transform_point"),
            "s.transform_point(&p) = s * p",
        )
        .input(iv("s", N))
        .input(iv("p", N))
        .out(fv("point", N))
        .ring(ring!(scale_transform_point, N)),
        Op::new(format!("{p}_mul_vector"), "s * v")
            .input(iv("s", N))
            .input(iv("v", N))
            .out(fv("vector", N))
            .ring(ring!(scale_mul_vector, N)),
        Op::new(
            format!("{p}_scale"),
            "s * k (every factor times the scalar)",
        )
        .input(iv("s", N))
        .input(with(fs("k"), Gen::S))
        .out(fv("product", N))
        .ring(ring!(scale_scale, N)),
        Op::new(format!("{p}_try_inverse"), "s.try_inverse().unwrap()")
            .input(ifactors("s", N))
            .out(fv("inverse", N))
            .tol(Tol::Ulp(1))
            .special(&[-2.0; N][..], 1)
            .eval(move |x| s(x).try_inverse().map(|i| flat(&i.vector))),
        Op::new(
            format!("{p}_pseudo_inverse"),
            "s.pseudo_inverse() (zero factors stay zero)",
        )
        .input(ifactors("s", N))
        .out(fv("inverse", N))
        .tol(Tol::Ulp(1))
        .special(
            &{
                let mut zeros = [0.5; N];
                zeros[0] = 0.0;
                zeros
            }[..],
            1,
        )
        .eval(move |x| Some(flat(&s(x).pseudo_inverse().vector))),
        Op::new(
            format!("{p}_try_inverse_transform_point"),
            "s.try_inverse_transform_point(&p).unwrap()",
        )
        .input(ifactors("s", N))
        .input(iv("p", N))
        .out(fv("point", N))
        .tol(Tol::Ulp(2))
        .eval(move |x| {
            s(x).try_inverse_transform_point(&pt(&x[N..]))
                .map(|q| flat(&q.coords))
        }),
    ]
}

// Reflection -----------------------------------------------------------------------------------------

type Refl<const D: usize> = Reflection<f64, Const<D>, ArrayStorage<f64, D, 1>>;

/// `(axis, bias)`, the axis taken as unit without renormalisation.
fn refl<const D: usize>(x: &[f64]) -> Refl<D> {
    Reflection::new(Unit::new_unchecked(sv::<f64, D>(x)), x[D])
}

/// `refl.reflect(&mut m)`, `m` a `D x C` matrix: `(axis, bias, m)` -> `m`.
fn reflect_op<const D: usize, const C: usize>() -> Op {
    Op::new(
        format!("reflection{D}_reflect_cols{C}"),
        "r.reflect(&mut m): every column of the D x C matrix m reflected",
    )
    .input(iu("axis", D))
    .input(ibias("bias"))
    .input(im("m", D, C))
    .out(fm("reflected", D, C))
    .tol(REFLECT)
    .eval(|x| {
        let mut m: SMatrix<f64, D, C> = sm(&x[D + 1..]);
        refl::<D>(x).reflect(&mut m);
        Some(flat(&m))
    })
}

/// `refl.reflect_with_sign(&mut m, sign)`: `(axis, bias, m, sign)` -> `m`.
fn reflect_sign_op<const D: usize, const C: usize>() -> Op {
    Op::new(
        format!("reflection{D}_reflect_with_sign_cols{C}"),
        "r.reflect_with_sign(&mut m, sign)",
    )
    .input(iu("axis", D))
    .input(ibias("bias"))
    .input(im("m", D, C))
    .input(isign("sign"))
    .out(fm("reflected", D, C))
    .tol(REFLECT)
    .eval(|x| {
        let mut m: SMatrix<f64, D, C> = sm(&x[D + 1..]);
        refl::<D>(x).reflect_with_sign(&mut m, x[D + 1 + D * C]);
        Some(flat(&m))
    })
}

/// `refl.reflect_rows(&mut lhs, &mut work)`, `lhs` an `R x D` matrix: `(axis, bias, lhs)` ->
/// `(lhs, work)`.
fn reflect_rows_op<const D: usize, const R: usize>() -> Op {
    Op::new(
        format!("reflection{D}_reflect_rows{R}"),
        "r.reflect_rows(&mut lhs, &mut work): every row of the R x D matrix lhs reflected, \
         work = lhs * axis - bias",
    )
    .input(iu("axis", D))
    .input(ibias("bias"))
    .input(im("lhs", R, D))
    .out(fm("reflected", R, D))
    .out(fv("work", R))
    .tol(REFLECT)
    .eval(|x| {
        let mut lhs: SMatrix<f64, R, D> = sm(&x[D + 1..]);
        let mut work = SVector::<f64, R>::zeros();
        refl::<D>(x).reflect_rows(&mut lhs, &mut work);
        let mut out = flat(&lhs);
        out.extend(flat(&work));
        Some(out)
    })
}

/// `refl.reflect_rows_with_sign(&mut lhs, &mut work, sign)`.
fn reflect_rows_sign_op<const D: usize, const R: usize>() -> Op {
    Op::new(
        format!("reflection{D}_reflect_rows_with_sign{R}"),
        "r.reflect_rows_with_sign(&mut lhs, &mut work, sign)",
    )
    .input(iu("axis", D))
    .input(ibias("bias"))
    .input(im("lhs", R, D))
    .input(isign("sign"))
    .out(fm("reflected", R, D))
    .out(fv("work", R))
    .tol(REFLECT)
    .eval(|x| {
        let mut lhs: SMatrix<f64, R, D> = sm(&x[D + 1..]);
        let mut work = SVector::<f64, R>::zeros();
        refl::<D>(x).reflect_rows_with_sign(&mut lhs, &mut work, x[D + 1 + R * D]);
        let mut out = flat(&lhs);
        out.extend(flat(&work));
        Some(out)
    })
}

/// `Reflection::new_containing_point(axis, &pt).bias()`.
fn containing_point_op<const D: usize>() -> Op {
    Op::new(
        format!("reflection{D}_new_containing_point"),
        "Reflection::new_containing_point(axis, &pt).bias() = axis . pt",
    )
    .input(iu("axis", D))
    .input(iv("pt", D))
    .out(fs("bias"))
    .tol(Tol::Ulp(4))
    .eval(|x| {
        let r: Refl<D> = Reflection::new_containing_point(
            Unit::new_unchecked(sv::<f64, D>(x)),
            &Point::<f64, D>::from(sv::<f64, D>(&x[D..])),
        );
        Some(vec![r.bias()])
    })
}

fn reflection_ops() -> Vec<Op> {
    vec![
        containing_point_op::<1>(),
        containing_point_op::<2>(),
        containing_point_op::<3>(),
        containing_point_op::<5>(),
        containing_point_op::<6>(),
        reflect_op::<1, 2>(),
        reflect_op::<2, 3>(),
        reflect_op::<3, 1>(),
        reflect_op::<3, 2>(),
        reflect_op::<4, 2>(),
        reflect_op::<5, 1>(),
        reflect_op::<6, 1>(),
        reflect_sign_op::<2, 2>(),
        reflect_sign_op::<3, 2>(),
        reflect_sign_op::<5, 2>(),
        reflect_rows_op::<1, 3>(),
        reflect_rows_op::<2, 3>(),
        reflect_rows_op::<3, 2>(),
        reflect_rows_op::<4, 1>(),
        reflect_rows_op::<5, 2>(),
        reflect_rows_op::<6, 2>(),
        reflect_rows_sign_op::<2, 2>(),
        reflect_rows_sign_op::<3, 3>(),
        reflect_rows_sign_op::<5, 1>(),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut ops = scale_ops::<2>();
    ops.extend(scale_ops::<3>());
    ops.extend(scale_ops::<5>());
    ops.extend(reflection_ops());
    vec![Suite {
        name: "scale_reflection",
        description: "Scale2/3/5 (products, scalar product, point / vector transforms, inverse, \
                      pseudo-inverse, inverse point transform) and Reflection1..6 \
                      (new_containing_point, reflect, reflect_with_sign, reflect_rows, \
                      reflect_rows_with_sign)",
        ops,
    }]
}
