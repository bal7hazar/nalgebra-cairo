//! WP 8.5-P13: the dynamic matrices (`DMatrix`, `DVector`, upstream `base/ops.rs`,
//! `base/norm.rs`) at sizes 3, 6 and 16, plus a rectangular product. The Cairo side dispatches the
//! square products up to 6x6 to the static kernels and runs its loops above (DESIGN D5): both
//! accumulate every output exactly and floor it once, so the products, dot products and squared
//! norms are the exact floors (i128 path); the norm is the floor of the exact root. Vectors are
//! `n x 1` matrices on the oracle's interface (`[[i64; 1]; n]` in Cairo: one flattening helper
//! for every size).

use super::{flat, ring, Ring};
use crate::engine::{fm, fs, im, Op, Suite, Tol};
use crate::gen::Dist;
use nalgebra::{DMatrix, DVector};

fn dmat<T: Ring>(x: &[T], r: usize, c: usize) -> DMatrix<T> {
    DMatrix::from_row_slice(r, c, &x[..r * c])
}

fn mul<T: Ring, const M: usize, const K: usize, const N: usize>(x: &[T]) -> Vec<T> {
    flat(&(dmat(x, M, K) * dmat(&x[M * K..], K, N)))
}

fn mul_vec<T: Ring, const M: usize, const K: usize>(x: &[T]) -> Vec<T> {
    flat(&(dmat(x, M, K) * DVector::from_column_slice(&x[M * K..M * K + K])))
}

fn dot<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    vec![DVector::from_column_slice(&x[..N]).dot(&DVector::from_column_slice(&x[N..2 * N]))]
}

fn norm_squared<T: Ring, const N: usize>(x: &[T]) -> Vec<T> {
    let v = DVector::from_column_slice(&x[..N]);
    vec![v.dot(&v)]
}

fn mul_op<const M: usize, const K: usize, const N: usize>(name: String) -> Op {
    Op::new(name, "a * b (DMatrix)")
        .input(im("a", M, K))
        .input(im("b", K, N))
        .out(fm("ab", M, N))
        .dists(&Dist::NO_LARGE)
        .ring(ring!(mul, M, K, N))
}

fn mul_vec_op<const M: usize, const K: usize>(name: String) -> Op {
    Op::new(name, "a * v (DMatrix * DVector)")
        .input(im("a", M, K))
        .input(im("v", K, 1))
        .out(fm("av", M, 1))
        .dists(&Dist::NO_LARGE)
        .ring(ring!(mul_vec, M, K))
}

fn ops() -> Vec<Op> {
    vec![
        mul_op::<3, 3, 3>("dmatrix3_mul".into()),
        mul_op::<6, 6, 6>("dmatrix6_mul".into()),
        mul_op::<16, 16, 16>("dmatrix16_mul".into()),
        mul_op::<4, 7, 5>("dmatrix4x7_mul_7x5".into()),
        mul_vec_op::<6, 6>("dmatrix6_mul_vec".into()),
        mul_vec_op::<16, 16>("dmatrix16_mul_vec".into()),
        mul_vec_op::<3, 8>("dmatrix3x8_mul_vec".into()),
        Op::new("dvector16_dot", "a.dot(&b) (DVector)")
            .input(im("a", 16, 1))
            .input(im("b", 16, 1))
            .out(fs("dot"))
            .dists(&Dist::NO_LARGE)
            .ring(ring!(dot, 16)),
        Op::new("dvector16_norm_squared", "a.norm_squared() (DVector)")
            .input(im("a", 16, 1))
            .out(fs("norm_squared"))
            .dists(&Dist::NO_LARGE)
            .ring(ring!(norm_squared, 16)),
        Op::new("dvector16_norm", "a.norm() (DVector)")
            .input(im("a", 16, 1))
            .out(fs("norm"))
            .tol(Tol::Ulp(2))
            .eval(|x| Some(vec![DVector::from_column_slice(&x[..16]).norm()])),
    ]
}

pub fn suites() -> Vec<Suite> {
    vec![Suite {
        name: "dynamic",
        description: "Dynamic matrices (DMatrix, DVector): products, matrix-vector products, dot \
                      products and norms at sizes 3, 6 and 16",
        ops: ops(),
    }]
}
