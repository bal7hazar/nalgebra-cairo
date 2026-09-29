//! Internal, no stability promise: the crate-private items of `linalg::svd3` that the packages
//! above this one use (docs/SPLIT.md §12.3). Never re-exported by the facade `nalgebra`.

use nalgebra_core::base::matrix3::Matrix3;
use nalgebra_core::base::matrix_mul::MatrixMul;
use nalgebra_core::base::vector3::Vector3;
use nalgebra_core::internal::base::sym_matrix3::SymMatrix3;
use nalgebra_static3::base::matrix3::Matrix3Trait;
use nalgebra_static3::base::vector3::Vector3Trait;
use nalgebra_static3::internal::base::matrix3::Matrix3InternalTrait;
use nalgebra_static3::internal::base::vector3::Vector3InternalTrait;
use simba::scalar::Real;
use crate::linalg::svd3::Svd3;
use crate::linalg::symmetric_eigen3::SymmetricEigen3;

/// Crate-internal kernels of `Svd3<T>` (WP 8.0: the public API is strictly upstream's): the Gram
/// matrix `MᵀM` as a `SymMatrix3` (the input of the eigen decomposition), and the column scaling
/// and the guarded reciprocal / quotient shared by `recompose`, `pseudo_inverse` and `solve`.
#[generate_trait]
pub impl Svd3InternalImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
    +PartialEq<T>,
    +PartialOrd<T>,
> of Svd3InternalTrait<T> {
    /// The body of `new` from the eigen decomposition of `MᵀM` (shared with `try_new`),
    /// documented on `new`. `#[inline(always)]`: `new` compiles to the code it had before the
    /// split (WP 8.5-P14b), so its gas is unchanged.
    #[inline(always)]
    fn from_eigen(
        matrix: Matrix3<T>, eigen: SymmetricEigen3<T>, compute_u: bool, compute_v: bool,
    ) -> Svd3<T> {
        // RENORMALISE the eigenvectors. `SymmetricEigen3` divides each accumulated column by its
        // own FLOORED norm, and on a `small` matrix those columns are tiny in raw units, so the
        // floor costs `1 / |u|_raw` RELATIVE — up to 1.4e-7 on the oracle vectors. That bias
        // lands directly on `σ = |M v|`, on the orthonormality of `V` and on `recompose`. Here the
        // columns are of magnitude 2^32, so the same floor costs only 2.3e-10. Column 3 is the
        // cross product of the first two, exactly as `SymmetricEigen3` builds it.
        let ev = eigen.eigenvectors;
        let (c1, c2) = (ev.column1(), ev.column2());
        let n1 = R::norm3(c1.x, c1.y, c1.z);
        let n2 = R::norm3(c2.x, c2.y, c2.z);
        let v1 = {
            let (x, y, z) = R::div3(c1.x, c1.y, c1.z, n1);
            Vector3 { x, y, z }
        };
        let v2 = {
            let (x, y, z) = R::div3(c2.x, c2.y, c2.z, n2);
            Vector3 { x, y, z }
        };
        let v3 = v1.cross(v2);
        let (w1, w2, w3) = (matrix.mul_mat(v1), matrix.mul_mat(v2), matrix.mul_mat(v3));
        let s1 = R::norm3(w1.x, w1.y, w1.z);
        let s2 = R::norm3(w2.x, w2.y, w2.z);
        let s3 = R::norm3(w3.x, w3.y, w3.z);
        // `SymmetricEigen3` sorts the eigenvalues ASCENDING and the singular values are DESCENDING,
        // so the columns generally come out reversed — but reversing them unconditionally would
        // also reorder EQUAL singular values, and the decomposition of the identity would not be
        // the identity. The sorting network of 3 elements, run on the computed norms with a STRICT
        // comparison, reverses exactly when the order asks for it and leaves ties alone. Branches
        // and moves only.
        let (mut s1, mut s2, mut s3) = (s1, s2, s3);
        let (mut w1, mut w2, mut w3) = (w1, w2, w3);
        let (mut v1, mut v2, mut v3) = (v1, v2, v3);
        if s2 > s1 {
            let (ts, tw, tv) = (s1, w1, v1);
            s1 = s2;
            w1 = w2;
            v1 = v2;
            s2 = ts;
            w2 = tw;
            v2 = tv;
        }
        if s3 > s1 {
            let (ts, tw, tv) = (s1, w1, v1);
            s1 = s3;
            w1 = w3;
            v1 = v3;
            s3 = ts;
            w3 = tw;
            v3 = tv;
        }
        if s3 > s2 {
            let (ts, tw, tv) = (s2, w2, v2);
            s2 = s3;
            w2 = w3;
            v2 = v3;
            s3 = ts;
            w3 = tw;
            v3 = tv;
        }
        let u = if compute_u {
            Some(Svd3InternalTrait::left(w1, w2, w3, s1))
        } else {
            None
        };
        Svd3 {
            u,
            singular_values: Vector3 { x: s1, y: s2, z: s3 },
            v_t: if compute_v {
                Some(Matrix3Trait::from_rows(v1, v2, v3))
            } else {
                None
            },
        }
    }

    /// The left singular vectors of `new` from the sorted `w_i = M v_i` and `σ_1` (see `new`).
    /// Skipped (`compute_u = false`) they cost no Cairo step (measured: 5 235 → 4 892 steps net,
    /// `bench_svd3_new__without_u`); the Sierra gas of the snapshots charges a branch at its
    /// costliest path whatever is executed (docs/BENCHMARK.md).
    #[inline(always)]
    fn left(w1: Vector3<T>, w2: Vector3<T>, w3: Vector3<T>, s1: T) -> Matrix3<T> {
        let u1 = if s1 == R::zero() {
            Vector3 { x: R::one(), y: R::zero(), z: R::zero() }
        } else {
            {
                let (x, y, z) = R::div3(w1.x, w1.y, w1.z, s1);
                Vector3 { x, y, z }
            }
        };
        // `w2` stripped of its `u1` component: one fused dot and one fused `mul_add` per
        // component, so `u2` is orthogonal to `u1` to within the final normalisation alone.
        let p = R::sum_prod3(u1.x, w2.x, u1.y, w2.y, u1.z, w2.z);
        let g = Vector3 {
            x: R::mul_add(-p, u1.x, w2.x),
            y: R::mul_add(-p, u1.y, w2.y),
            z: R::mul_add(-p, u1.z, w2.z),
        };
        let n = R::norm3(g.x, g.y, g.z);
        let u2 = if n == R::zero() {
            let (basis, _) = u1.orthonormal_basis();
            basis
        } else {
            {
                let (x, y, z) = R::div3(g.x, g.y, g.z, n);
                Vector3 { x, y, z }
            }
        };
        let c = u1.cross(u2);
        let along = R::sum_prod3(c.x, w3.x, c.y, w3.y, c.z, w3.z);
        let u3 = if along.is_sign_negative() {
            Vector3 { x: -c.x, y: -c.y, z: -c.z }
        } else {
            c
        };
        Matrix3Trait::from_columns(u1, u2, u3)
    }

    /// `MᵀM` as a symmetric matrix: 6 fused kernels instead of 27 products, bit-identical to the
    /// upper triangle of `m.transpose() * m` and to `m.transpose().mul_transpose()`. The latter
    /// would reuse `base` instead of repeating the kernel, and it is NOT free: the transpose
    /// costs 2 760 gas of moves that Sierra does not elide (13 500 against 16 260, measured,
    /// `bench_svd3_gram__fused` against `bench_svd3_gram__transpose_mul_transpose`). Panics on
    /// overflow. Upstream: `m.tr_mul(&m)`.
    #[inline(always)]
    fn gram(m: Matrix3<T>) -> SymMatrix3<T> {
        SymMatrix3 {
            m11: R::norm_squared3(m.m11, m.m21, m.m31),
            m12: R::sum_prod3(m.m11, m.m12, m.m21, m.m22, m.m31, m.m32),
            m13: R::sum_prod3(m.m11, m.m13, m.m21, m.m23, m.m31, m.m33),
            m22: R::norm_squared3(m.m12, m.m22, m.m32),
            m23: R::sum_prod3(m.m12, m.m13, m.m22, m.m23, m.m32, m.m33),
            m33: R::norm_squared3(m.m13, m.m23, m.m33),
        }
    }
    /// `m * diag(d)`: each column of `m` scaled by the matching component of `d`, 9 floored
    /// products. No upstream equivalent (upstream materialises `Matrix::from_diagonal`).
    #[inline(always)]
    fn scale_columns(m: Matrix3<T>, d: Vector3<T>) -> Matrix3<T> {
        Matrix3 {
            m11: m.m11 * d.x,
            m21: m.m21 * d.x,
            m31: m.m31 * d.x,
            m12: m.m12 * d.y,
            m22: m.m22 * d.y,
            m32: m.m32 * d.y,
            m13: m.m13 * d.z,
            m23: m.m23 * d.z,
            m33: m.m33 * d.z,
        }
    }
    /// `1 / s` when `s > eps`, `0` otherwise: upstream's `pseudo_inverse` filter.
    #[inline(always)]
    fn inverted(s: T, eps: T) -> T {
        if s > eps {
            s.recip()
        } else {
            R::zero()
        }
    }
    /// `y / s` when `s > eps`, `0` otherwise: upstream's `solve` filter.
    #[inline(always)]
    fn divided(y: T, s: T, eps: T) -> T {
        if s > eps {
            R::div(y, s)
        } else {
            R::zero()
        }
    }
}
