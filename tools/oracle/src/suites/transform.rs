//! WP 8.4-P11a: `Transform2/3`, `Projective2/3`, `Affine2/3` (upstream `geometry/transform*.rs`),
//! suite `transform`: the inverses, `transform_point` / `transform_vector` and their inverse
//! forms, and the products with isometries, similarities, rotations and other transforms.
//!
//! An affine transform is given by its linear block and its translation (`(linear, translation)`,
//! the last row `(0, .., 0, 1)` implicit), a projective / general transform by its whole
//! homogeneous matrix. The linear blocks that get inverted are well-conditioned (condition number
//! <= 8, like the `try_inverse` ops of the base suites); the other ones have independent entries.

use super::{flat, iso2, iso3, sim3, sm, sv, uquat};
use crate::engine::{fm, fv, iiso2, iiso3, im, isim3, iuq, iv, with, Input, Op, Suite, Tol};
use crate::gen::{Dist, Gen};
use nalgebra::{
    Affine2, Affine3, Matrix3, Matrix4, Point2, Point3, Projective2, Projective3, Vector2, Vector3,
};

/// The fused kernels on exact inputs: the floor of the exact result (1 ulp for the f64 oracle).
const FUSED: Tol = Tol::Ulp(1);
/// The inverses of the homogeneous `n x n` matrices (every category, affine included): one
/// rounding per cofactor / quotient, amplified by the conditioning (the `inverse_tol` of the base
/// suites, `k = 2n`).
const INVERSE3: Tol = Tol::Sens { k: 6.0, base: 4.0 };
const INVERSE4: Tol = Tol::Sens { k: 8.0, base: 4.0 };
/// A product by a rounded homogeneous matrix (a quaternion's rotation matrix, a similarity's
/// scaled rotation): a few one-ulp roundings on the entries of the factor, amplified.
const PRODUCT: Tol = Tol::Sens { k: 4.0, base: 2.0 };
/// The homogeneous division of the general / projective `transform_point` (the `cg` suite's
/// `TRANSFORM` policy).
const TRANSFORM: Tol = Tol::Sens { k: 3.0, base: 2.0 };
/// The general / projective `transform_vector` divides BEFORE the product (the `cg` suite's
/// `TRANSFORM_VECTOR` policy).
const TRANSFORM_VECTOR: Tol = Tol::SensMag {
    k: 3.0,
    base: 2.0,
    mag: 2.0,
};
/// An inverse, then a transform of the point by it (upstream's formula): the one-ulp roundings of
/// the inverse's entries are multiplied by the coordinates of the point (up to `d` terms per
/// output), an absolute error proportional to `max |input|` that the first-order sensitivity does
/// not see (the `cg` suite's `TRANSFORM_VECTOR` reasoning).
const INVERSE_TRANSFORM: Tol = Tol::SensMag {
    k: 10.0,
    base: 6.0,
    mag: 4.0,
};

/// A product by an inverse (upstream's `a * b.inverse()`): the roundings of the inverse are
/// multiplied by the entries of `a` (the `INVERSE_TRANSFORM` reasoning).
const QUOTIENT: Tol = Tol::SensMag {
    k: 8.0,
    base: 4.0,
    mag: 4.0,
};

// Inputs ------------------------------------------------------------------------------------------

/// A well-conditioned linear block (invertible).
fn iwell(name: &str, n: usize) -> Input {
    with(fm(name, n, n), Gen::WellCond(n))
}

// Builders ----------------------------------------------------------------------------------------

/// `Affine2` from `(linear 2x2 row-major, translation (x, y))`.
#[rustfmt::skip]
fn aff2(x: &[f64]) -> Affine2<f64> {
    Affine2::from_matrix_unchecked(Matrix3::new(
        x[0], x[1], x[4],
        x[2], x[3], x[5],
        0.0, 0.0, 1.0,
    ))
}

/// `Affine3` from `(linear 3x3 row-major, translation (x, y, z))`.
#[rustfmt::skip]
fn aff3(x: &[f64]) -> Affine3<f64> {
    Affine3::from_matrix_unchecked(Matrix4::new(
        x[0], x[1], x[2], x[9],
        x[3], x[4], x[5], x[10],
        x[6], x[7], x[8], x[11],
        0.0, 0.0, 0.0, 1.0,
    ))
}

fn p2(x: &[f64]) -> Point2<f64> {
    Point2::new(x[0], x[1])
}

fn p3(x: &[f64]) -> Point3<f64> {
    Point3::new(x[0], x[1], x[2])
}

fn v2(x: &[f64]) -> Vector2<f64> {
    sv::<f64, 2>(x)
}

fn v3(x: &[f64]) -> Vector3<f64> {
    sv::<f64, 3>(x)
}

/// The homogeneous normaliser away from zero (`|n| >= 1/4`, the `cg` suite's rule).
fn normaliser_ok(n: f64) -> bool {
    n.abs() >= 0.25
}

// Ops ---------------------------------------------------------------------------------------------

fn inverse_ops() -> Vec<Op> {
    vec![
        Op::new(
            "affine2_try_inverse",
            "Affine2 (linear, translation): t.try_inverse().unwrap().into_inner()",
        )
        .input(iwell("linear", 2))
        .input(iv("translation", 2))
        .out(fm("inverse", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE3)
        .eval(|x| aff2(x).try_inverse().map(|t| flat(&t.into_inner()))),
        Op::new(
            "affine3_try_inverse",
            "Affine3 (linear, translation): t.try_inverse().unwrap().into_inner()",
        )
        .input(iwell("linear", 3))
        .input(iv("translation", 3))
        .out(fm("inverse", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE4)
        .eval(|x| aff3(x).try_inverse().map(|t| flat(&t.into_inner()))),
        Op::new(
            "projective2_try_inverse",
            "Projective2::from_matrix_unchecked(m).try_inverse().unwrap().into_inner()",
        )
        .input(iwell("m", 3))
        .out(fm("inverse", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE3)
        .eval(|x| {
            let m: Matrix3<f64> = sm(x);
            Projective2::from_matrix_unchecked(m)
                .try_inverse()
                .map(|t| flat(&t.into_inner()))
        }),
        Op::new(
            "projective3_try_inverse",
            "Projective3::from_matrix_unchecked(m).try_inverse().unwrap().into_inner()",
        )
        .input(iwell("m", 4))
        .out(fm("inverse", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE4)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            Projective3::from_matrix_unchecked(m)
                .try_inverse()
                .map(|t| flat(&t.into_inner()))
        }),
    ]
}

fn transform_ops() -> Vec<Op> {
    vec![
        Op::new(
            "affine2_transform_point",
            "Affine2 (linear, translation): t.transform_point(&p)",
        )
        .input(im("linear", 2, 2))
        .input(iv("translation", 2))
        .input(iv("p", 2))
        .out(fv("point", 2))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| Some(flat(&aff2(x).transform_point(&p2(&x[6..])).coords))),
        Op::new(
            "affine2_transform_vector",
            "Affine2 (linear, translation): t.transform_vector(&v)",
        )
        .input(im("linear", 2, 2))
        .input(iv("translation", 2))
        .input(iv("v", 2))
        .out(fv("vector", 2))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| Some(flat(&aff2(x).transform_vector(&v2(&x[6..]))))),
        Op::new(
            "affine3_transform_point",
            "Affine3 (linear, translation): t.transform_point(&p)",
        )
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| Some(flat(&aff3(x).transform_point(&p3(&x[12..])).coords))),
        Op::new(
            "affine3_transform_vector",
            "Affine3 (linear, translation): t.transform_vector(&v)",
        )
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| Some(flat(&aff3(x).transform_vector(&v3(&x[12..]))))),
        Op::new(
            "projective3_transform_point",
            "Projective3::from_matrix_unchecked(m).transform_point(&p), |m[3, :] . (p, 1)| >= 1/4",
        )
        .input(im("m", 4, 4))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .tol(TRANSFORM)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            let p = p3(&x[16..]);
            let n = m[(3, 0)] * p.x + m[(3, 1)] * p.y + m[(3, 2)] * p.z + m[(3, 3)];
            normaliser_ok(n).then(|| {
                flat(
                    &Projective3::from_matrix_unchecked(m)
                        .transform_point(&p)
                        .coords,
                )
            })
        }),
        Op::new(
            "projective3_transform_vector",
            "Projective3::from_matrix_unchecked(m).transform_vector(&v), |m[3, :3] . v| >= 1/4",
        )
        .input(im("m", 4, 4))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .tol(TRANSFORM_VECTOR)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            let v = v3(&x[16..]);
            let n = m[(3, 0)] * v.x + m[(3, 1)] * v.y + m[(3, 2)] * v.z;
            normaliser_ok(n)
                .then(|| flat(&Projective3::from_matrix_unchecked(m).transform_vector(&v)))
        }),
        Op::new(
            "affine3_inverse_transform_point",
            "Affine3 (linear, translation): t.inverse_transform_point(&p)",
        )
        .input(iwell("linear", 3))
        .input(iv("translation", 3))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE_TRANSFORM)
        .eval(|x| Some(flat(&aff3(x).inverse_transform_point(&p3(&x[12..])).coords))),
        Op::new(
            "affine3_inverse_transform_vector",
            "Affine3 (linear, translation): t.inverse_transform_vector(&v)",
        )
        .input(iwell("linear", 3))
        .input(iv("translation", 3))
        .input(iv("v", 3))
        .out(fv("vector", 3))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE_TRANSFORM)
        .eval(|x| Some(flat(&aff3(x).inverse_transform_vector(&v3(&x[12..]))))),
        Op::new(
            "projective3_inverse_transform_point",
            "Projective3::from_matrix_unchecked(m).inverse_transform_point(&p), the normaliser of \
             the inverse >= 1/4",
        )
        .input(iwell("m", 4))
        .input(iv("p", 3))
        .out(fv("point", 3))
        .dists(&Dist::NO_LARGE)
        .tol(INVERSE_TRANSFORM)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            let p = p3(&x[16..]);
            let inv = m.try_inverse()?;
            let n = inv[(3, 0)] * p.x + inv[(3, 1)] * p.y + inv[(3, 2)] * p.z + inv[(3, 3)];
            normaliser_ok(n).then(|| {
                flat(
                    &Projective3::from_matrix_unchecked(m)
                        .inverse_transform_point(&p)
                        .coords,
                )
            })
        }),
    ]
}

fn product_ops() -> Vec<Op> {
    vec![
        Op::new(
            "affine2_mul_isometry2",
            "(Affine2 (linear, translation) * iso).into_inner()",
        )
        .input(im("linear", 2, 2))
        .input(iv("translation", 2))
        .input(iiso2("iso"))
        .out(fm("product", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| Some(flat(&(aff2(x) * iso2(&x[6..])).into_inner()))),
        Op::new(
            "isometry2_mul_affine2",
            "(iso * Affine2 (linear, translation)).into_inner()",
        )
        .input(iiso2("iso"))
        .input(im("linear", 2, 2))
        .input(iv("translation", 2))
        .out(fm("product", 3, 3))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| Some(flat(&(iso2(x) * aff2(&x[4..])).into_inner()))),
        Op::new(
            "affine3_mul_isometry3",
            "(Affine3 (linear, translation) * iso).into_inner()",
        )
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .input(iiso3("iso"))
        .out(fm("product", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(PRODUCT)
        .eval(|x| Some(flat(&(aff3(x) * iso3(&x[12..])).into_inner()))),
        Op::new(
            "isometry3_mul_affine3",
            "(iso * Affine3 (linear, translation)).into_inner()",
        )
        .input(iiso3("iso"))
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .out(fm("product", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(PRODUCT)
        .eval(|x| Some(flat(&(iso3(x) * aff3(&x[7..])).into_inner()))),
        Op::new(
            "affine3_mul_unit_quaternion",
            "(Affine3 (linear, translation) * q).into_inner()",
        )
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .input(iuq("q"))
        .out(fm("product", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(PRODUCT)
        .eval(|x| Some(flat(&(aff3(x) * uquat(&x[12..])).into_inner()))),
        Op::new(
            "similarity3_mul_affine3",
            "(sim * Affine3 (linear, translation)).into_inner()",
        )
        .input(isim3("sim"))
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .out(fm("product", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(PRODUCT)
        .eval(|x| Some(flat(&(sim3(x) * aff3(&x[8..])).into_inner()))),
        Op::new(
            "projective3_mul_affine3",
            "(Projective3::from_matrix_unchecked(m) * Affine3 (linear, translation)).into_inner()",
        )
        .input(im("m", 4, 4))
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .out(fm("product", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(FUSED)
        .eval(|x| {
            let m: Matrix4<f64> = sm(x);
            Some(flat(
                &(Projective3::from_matrix_unchecked(m) * aff3(&x[16..])).into_inner(),
            ))
        }),
        Op::new(
            "affine3_div_projective3",
            "(Affine3 (linear, translation) / Projective3::from_matrix_unchecked(m)).into_inner()",
        )
        .input(im("linear", 3, 3))
        .input(iv("translation", 3))
        .input(iwell("m", 4))
        .out(fm("quotient", 4, 4))
        .dists(&Dist::NO_LARGE)
        .tol(QUOTIENT)
        .eval(|x| {
            let m: Matrix4<f64> = sm(&x[12..]);
            Some(flat(
                &(aff3(x) / Projective3::from_matrix_unchecked(m)).into_inner(),
            ))
        }),
    ]
}

pub fn suites() -> Vec<Suite> {
    let mut ops = inverse_ops();
    ops.extend(transform_ops());
    ops.extend(product_ops());
    vec![Suite {
        name: "transform",
        description: "Transform2/3, Projective2/3, Affine2/3 (geometry/transform*.rs): \
                      try_inverse, transform_point, transform_vector, inverse_transform_*, \
                      products with isometries, similarities, unit quaternions and transforms",
        ops,
    }]
}
