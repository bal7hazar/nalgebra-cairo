//! The triangular solves of `CsMatrix` (upstream `sparse/cs_matrix_solve.rs`): `L x = b` and
//! `Lᵀ x = b` for a lower-triangular sparse `L` (its strict upper part is ignored, like
//! upstream's), dense or sparse `b`.
//!
//! Formulation: upstream solves column by column, scattering `b[i] -= x[j] * L[i, j]` into the
//! right-hand side; Cairo memory is write-once, so `L x = b` runs ROW by row on the transposed
//! pattern (`x[i] = (b[i] - sum_j L[i, j] x[j]) / L[i, i]`, the `x[j]` read back from the
//! solution built so far) and `Lᵀ x = b` column by column backwards (already a gather upstream).
//! Every `b[i] - sum` is ONE exact accumulation floored once, then divided (rounded to nearest):
//! two roundings per component, where upstream rounds every update.

use simba::scalar::Real;
use crate::base::dynamic::DMatrix;
use crate::base::errors;
use super::cs_matrix::CsMatrix;
use super::cs_utils::{gather, transpose_pattern};
use super::errors as sparse_errors;

/// Panics with `nalgebra: matrix not square` unless `m` is square, with `nalgebra: dimension
/// mismatch` unless `nrows` equals it; returns the dimension.
fn check_system<T>(m: @CsMatrix<T>, nrows: usize) -> usize {
    let n = *m.nrows;
    if n != *m.ncols {
        core::panic_with_felt252(sparse_errors::NOT_SQUARE);
    }
    if nrows != n {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH);
    }
    n
}

/// Triangular solves of `CsMatrix<T>` for any `Real` scalar.
#[generate_trait]
pub impl CsMatrixSolveImpl<
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
> of CsMatrixSolveTrait<T> {
    /// The solution `x` of `self * x = b` (`self` lower-triangular, square), `None` when a
    /// diagonal entry is missing or zero. `b` is any dense matrix with `self.nrows()` rows (a
    /// `DMatrix`, a `DVector`, a static shape: anything that converts `Into<DMatrix>`); the
    /// result is a `DMatrix` (Cairo has no generic `OMatrix<T, R2, C2>`). Panics with `nalgebra:
    /// matrix not square` / `nalgebra: dimension mismatch`. Upstream:
    /// `CsMatrix::solve_lower_triangular(&b)`. Deviation: upstream checks `nrows == b.len()`, which
    /// rejects every right-hand side with more than one column; the rows are compared here.
    fn solve_lower_triangular<B, +Into<B, DMatrix<T>>, +Drop<B>>(
        self: @CsMatrix<T>, b: B,
    ) -> Option<DMatrix<T>> {
        let mut x: DMatrix<T> = b.into();
        if Self::solve_lower_triangular_mut(self, ref x) {
            Some(x)
        } else {
            None
        }
    }

    /// The solution `x` of `selfᵀ * x = b` (`self` lower-triangular, square), `None` when a
    /// diagonal entry is missing or zero; `b` and the result as in `solve_lower_triangular`.
    /// Upstream: `CsMatrix::tr_solve_lower_triangular(&b)`.
    fn tr_solve_lower_triangular<B, +Into<B, DMatrix<T>>, +Drop<B>>(
        self: @CsMatrix<T>, b: B,
    ) -> Option<DMatrix<T>> {
        let mut x: DMatrix<T> = b.into();
        if Self::tr_solve_lower_triangular_mut(self, ref x) {
            Some(x)
        } else {
            None
        }
    }

    /// Replaces `b` by the solution of `self * x = b`; returns `false` (and leaves `b`
    /// unchanged; upstream leaves it partially overwritten) when a diagonal entry is missing or
    /// zero. Upstream: `CsMatrix::solve_lower_triangular_mut(&mut b)`.
    fn solve_lower_triangular_mut(self: @CsMatrix<T>, ref b: DMatrix<T>) -> bool {
        let n = check_system(self, b.nrows);
        let ncols = b.ncols;
        if ncols == 0 {
            return true;
        }
        let zero = R::zero();
        // The rows of `self`: `self[i, j]` for `j` in `tj[tp[i]..tp[i + 1]]`, ascending.
        let (tp, tj, src) = transpose_pattern(n, *self.p, *self.i);
        let tv = gather(*self.vals, src);
        let mut out: Array<T> = array![];
        let mut data = b.data;
        let mut ok = true;
        let mut c: usize = 0;
        while c != ncols {
            let col = data.slice(c * n, n);
            let mut x: Array<T> = array![];
            let mut i: usize = 0;
            while i != n {
                let mut w = R::wide_add(R::wide_zero(), *col[i]);
                let mut diag = zero;
                let mut q = *tp[i];
                let re = *tp[i + 1];
                while q != re {
                    let j = *tj[q];
                    if j < i {
                        w = R::wide_sub_prod(w, *tv[q], *x[j]);
                    } else {
                        if j == i {
                            diag = *tv[q];
                        }
                        break;
                    }
                    q += 1;
                }
                if diag == zero {
                    ok = false;
                    break;
                }
                x.append(R::div(R::wide_rescale(w), diag));
                i += 1;
            }
            if !ok {
                break;
            }
            out.append_span(x.span());
            c += 1;
        }
        if ok {
            b = DMatrix { data: out.span(), nrows: n, ncols };
        }
        ok
    }

    /// Replaces `b` by the solution of `selfᵀ * x = b`; returns `false` (and leaves `b`
    /// unchanged) when a diagonal entry is missing or zero. Upstream:
    /// `CsMatrix::tr_solve_lower_triangular_mut(&mut b)`.
    fn tr_solve_lower_triangular_mut(self: @CsMatrix<T>, ref b: DMatrix<T>) -> bool {
        let n = check_system(self, b.nrows);
        let ncols = b.ncols;
        if ncols == 0 {
            return true;
        }
        let zero = R::zero();
        let p = *self.p;
        let li = *self.i;
        let lv = *self.vals;
        let mut out: Array<T> = array![];
        let data = b.data;
        let mut ok = true;
        let mut c: usize = 0;
        while c != ncols {
            let col = data.slice(c * n, n);
            // `xr[t]` is `x[n - 1 - t]`: the solution is built backwards.
            let mut xr: Array<T> = array![];
            let mut j = n;
            while j != 0 {
                j -= 1;
                let mut q = *p[j];
                let e = *p[j + 1];
                // Skip the strict upper part, then expect the diagonal.
                while q != e && *li[q] < j {
                    q += 1;
                }
                if q == e || *li[q] != j || *lv[q] == zero {
                    ok = false;
                    break;
                }
                let diag = *lv[q];
                q += 1;
                let mut w = R::wide_add(R::wide_zero(), *col[j]);
                while q != e {
                    w = R::wide_sub_prod(w, *lv[q], *xr[n - 1 - *li[q]]);
                    q += 1;
                }
                xr.append(R::div(R::wide_rescale(w), diag));
            }
            if !ok {
                break;
            }
            let mut t = n;
            while t != 0 {
                t -= 1;
                out.append(*xr[t]);
            }
            c += 1;
        }
        if ok {
            b = DMatrix { data: out.span(), nrows: n, ncols };
        }
        ok
    }

    /// The sparse solution of `self * x = b` for a sparse one-column `b` (a `CsVector`): its
    /// pattern is the set of rows reachable from the pattern of `b` through the strictly lower
    /// part of `self` (ascending, explicit zeros kept, like upstream's sorted reach), `None` when
    /// a diagonal entry of a reached row is missing or zero. Panics with `nalgebra: matrix not
    /// square` / `nalgebra: dimension mismatch` (`b` must be `nrows x 1`). Upstream:
    /// `CsMatrix::solve_lower_triangular_cs(&b)` (a depth-first reach then a scatter; here one
    /// forward pass over the rows computes both, `O(n + nnz)`).
    fn solve_lower_triangular_cs(self: @CsMatrix<T>, b: CsMatrix<T>) -> Option<CsMatrix<T>> {
        let n = check_system(self, b.nrows);
        if b.ncols != 1 {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH);
        }
        let zero = R::zero();
        let (tp, tj, src) = transpose_pattern(n, *self.p, *self.i);
        let tv = gather(*self.vals, src);
        let mut bi = b.i;
        let mut bv = b.vals;
        let mut reach: Array<bool> = array![];
        let mut x: Array<T> = array![];
        let mut out_i: Array<usize> = array![];
        let mut out_v: Array<T> = array![];
        let mut ok = true;
        let mut i: usize = 0;
        while i != n {
            let mut w = R::wide_zero();
            let mut reached = false;
            if bi.len() != 0 && *bi[0] == i {
                bi.pop_front().unwrap();
                w = R::wide_add(w, *bv.pop_front().unwrap());
                reached = true;
            }
            let mut diag = zero;
            let mut q = *tp[i];
            let re = *tp[i + 1];
            while q != re {
                let j = *tj[q];
                if j < i {
                    if *reach[j] {
                        reached = true;
                        w = R::wide_sub_prod(w, *tv[q], *x[j]);
                    }
                } else {
                    if j == i {
                        diag = *tv[q];
                    }
                    break;
                }
                q += 1;
            }
            if reached {
                if diag == zero {
                    ok = false;
                    break;
                }
                let xi = R::div(R::wide_rescale(w), diag);
                out_i.append(i);
                out_v.append(xi);
                x.append(xi);
            } else {
                x.append(zero);
            }
            reach.append(reached);
            i += 1;
        }
        if !ok {
            return None;
        }
        let len = out_i.len();
        Some(
            CsMatrix {
                nrows: n, ncols: 1, p: array![0, len].span(), i: out_i.span(), vals: out_v.span(),
            },
        )
    }
}
