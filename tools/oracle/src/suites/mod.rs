//! Suite definitions: one suite per upstream feature, one op per upstream function.

use crate::engine::Suite;
use nalgebra::{
    ClosedAddAssign, ClosedMulAssign, ClosedSubAssign, Complex, DMatrix, Dim, Isometry2, Isometry3,
    Matrix, Quaternion, RawStorage, SMatrix, SVector, Scalar, Similarity2, Similarity3,
    Translation2, Translation3, Unit, UnitComplex, UnitQuaternion,
};
use num_traits::{One, Zero};

mod base;
mod cg;
mod dual_quaternion;
mod geometry;
mod linalg;
mod pose_completion;
mod projections;
mod rotation_completion;
mod scalar;
mod scale_reflection;

/// Every suite, in a stable order.
pub fn all() -> Vec<Suite> {
    let mut suites = vec![scalar::suite()];
    suites.extend(base::suites());
    suites.extend(geometry::suites());
    suites.extend(rotation_completion::suites());
    suites.extend(pose_completion::suites());
    suites.extend(scale_reflection::suites());
    suites.extend(dual_quaternion::suites());
    suites.extend(projections::suites());
    suites.extend(linalg::suites());
    suites.extend(cg::suites());
    suites
}

/// Scalars upstream nalgebra can run bilinear ops on: `f64` and exact `i128` raws.
pub trait Ring:
    Scalar
    + Copy
    + Zero
    + One
    + ClosedAddAssign
    + ClosedSubAssign
    + ClosedMulAssign
    + core::ops::Neg<Output = Self>
{
}
impl Ring for f64 {}
impl Ring for i128 {}

/// `(f64 evaluation, exact i128 evaluation)` of a generic bilinear op.
macro_rules! ring {
    ($f:ident $(, $n:expr)*) => {
        (
            Box::new(|x: &[f64]| Some($f::<f64 $(, { $n })*>(x))) as crate::engine::EvalFn,
            Box::new(|x: &[i128]| Some($f::<i128 $(, { $n })*>(x))) as crate::engine::ExactFn,
        )
    };
}
pub(crate) use ring;

// Flat slices <-> nalgebra values. Matrices are ROW-major on the oracle's interface. -----------

pub fn sv<T: Scalar, const N: usize>(x: &[T]) -> SVector<T, N> {
    SVector::<T, N>::from_column_slice(&x[..N])
}

pub fn sm<T: Scalar, const R: usize, const C: usize>(x: &[T]) -> SMatrix<T, R, C> {
    SMatrix::<T, R, C>::from_row_slice(&x[..R * C])
}

pub fn dm(x: &[f64], n: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(n, n, &x[..n * n])
}

/// Row-major flattening (a column vector flattens to its components).
pub fn flat<T: Scalar, R: Dim, C: Dim, S: RawStorage<T, R, C>>(m: &Matrix<T, R, C, S>) -> Vec<T> {
    let mut out = Vec::with_capacity(m.nrows() * m.ncols());
    for i in 0..m.nrows() {
        for j in 0..m.ncols() {
            out.push(m[(i, j)].clone());
        }
    }
    out
}

/// `(w, i, j, k)`.
pub fn quat(x: &[f64]) -> Quaternion<f64> {
    Quaternion::new(x[0], x[1], x[2], x[3])
}

/// `(w, i, j, k)`, taken as unit without renormalisation (what the Cairo side does).
pub fn uquat(x: &[f64]) -> UnitQuaternion<f64> {
    Unit::new_unchecked(quat(x))
}

pub fn flat_q(q: &Quaternion<f64>) -> Vec<f64> {
    vec![q.w, q.i, q.j, q.k]
}

/// `(re, im)`, taken as unit without renormalisation.
pub fn ucomplex(x: &[f64]) -> UnitComplex<f64> {
    Unit::new_unchecked(Complex::new(x[0], x[1]))
}

pub fn flat_c(c: &UnitComplex<f64>) -> Vec<f64> {
    vec![c.re, c.im]
}

/// `(tx, ty, re, im)`.
pub fn iso2(x: &[f64]) -> Isometry2<f64> {
    Isometry2::from_parts(Translation2::new(x[0], x[1]), ucomplex(&x[2..]))
}

pub fn flat_iso2(iso: &Isometry2<f64>) -> Vec<f64> {
    let mut out = flat(&iso.translation.vector);
    out.extend(flat_c(&iso.rotation));
    out
}

/// `(tx, ty, tz, w, i, j, k)`.
pub fn iso3(x: &[f64]) -> Isometry3<f64> {
    Isometry3::from_parts(Translation3::new(x[0], x[1], x[2]), uquat(&x[3..]))
}

pub fn flat_iso3(iso: &Isometry3<f64>) -> Vec<f64> {
    let mut out = flat(&iso.translation.vector);
    out.extend(flat_q(&iso.rotation));
    out
}

/// `(tx, ty, re, im, scaling)`.
pub fn sim2(x: &[f64]) -> Similarity2<f64> {
    Similarity2::from_isometry(iso2(x), x[4])
}

pub fn flat_sim2(sim: &Similarity2<f64>) -> Vec<f64> {
    let mut out = flat_iso2(&sim.isometry);
    out.push(sim.scaling());
    out
}

/// `(tx, ty, tz, w, i, j, k, scaling)`.
pub fn sim3(x: &[f64]) -> Similarity3<f64> {
    Similarity3::from_isometry(iso3(x), x[7])
}

pub fn flat_sim3(sim: &Similarity3<f64>) -> Vec<f64> {
    let mut out = flat_iso3(&sim.isometry);
    out.push(sim.scaling());
    out
}
