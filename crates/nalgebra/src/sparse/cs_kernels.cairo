//! The value kernels of `CsMatrix` (crate-internal: the public forms are its methods and
//! operators). Columns are sorted and deduplicated on input and output (the module invariant).

use simba::scalar::Real;
use crate::base::errors;
use super::cs_matrix::CsMatrix;
use super::cs_utils::{gather, transpose_pattern, union_sorted};

/// Appends `n` copies of `val` to `out`.
pub(crate) fn append_n<T, +Copy<T>, +Drop<T>>(ref out: Array<T>, n: usize, val: T) {
    let mut k = n;
    while k >= 4 {
        out.append(val);
        out.append(val);
        out.append(val);
        out.append(val);
        k -= 4;
    }
    while k != 0 {
        out.append(val);
        k -= 1;
    }
}

/// The kernels. Methods of a generic impl (AGENTS: no generic free functions), instantiated per
/// scalar.
pub(crate) trait CsKernels<T> {
    /// The column-major dense components of `m`.
    fn dense_data(m: CsMatrix<T>) -> Span<T>;
    /// The sparse matrix of the non-zero components of the column-major `data`.
    fn from_dense(data: Span<T>, nrows: usize, ncols: usize) -> CsMatrix<T>;
    /// `a + b`.
    fn add(a: CsMatrix<T>, b: CsMatrix<T>) -> CsMatrix<T>;
    /// `a * b`.
    fn mul(a: CsMatrix<T>, b: CsMatrix<T>) -> CsMatrix<T>;
    /// The column-major dense `n`-vector whose components `rows[k]` are `vals[k]`, zeros
    /// elsewhere (`rows` ascending).
    fn dense_column(rows: Span<usize>, vals: Span<T>, n: usize) -> Span<T>;
}

pub(crate) impl CsKernelsImpl<
    T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>, +Add<T>, +PartialEq<T>,
> of CsKernels<T> {
    fn dense_data(m: CsMatrix<T>) -> Span<T> {
        let zero = R::zero();
        let mut out: Array<T> = array![];
        let mut pp = m.p;
        let mut ii = m.i;
        let mut vv = m.vals;
        let mut start = *pp.pop_front().unwrap();
        while let Some(e) = pp.pop_front() {
            let e = *e;
            let mut next: usize = 0;
            let mut q = start;
            while q != e {
                let r = *ii.pop_front().unwrap();
                append_n(ref out, r - next, zero);
                out.append(*vv.pop_front().unwrap());
                next = r + 1;
                q += 1;
            }
            append_n(ref out, m.nrows - next, zero);
            start = e;
        }
        out.span()
    }

    fn from_dense(data: Span<T>, nrows: usize, ncols: usize) -> CsMatrix<T> {
        let zero = R::zero();
        let mut p: Array<usize> = array![0];
        let mut out_i: Array<usize> = array![];
        let mut out_v: Array<T> = array![];
        let mut dd = data;
        let mut count: usize = 0;
        let mut j: usize = 0;
        while j != ncols {
            let mut r: usize = 0;
            while r != nrows {
                let x = *dd.pop_front().unwrap();
                if x != zero {
                    out_i.append(r);
                    out_v.append(x);
                    count += 1;
                }
                r += 1;
            }
            p.append(count);
            j += 1;
        }
        CsMatrix { nrows, ncols, p: p.span(), i: out_i.span(), vals: out_v.span() }
    }

    fn add(a: CsMatrix<T>, b: CsMatrix<T>) -> CsMatrix<T> {
        if a.nrows != b.nrows || a.ncols != b.ncols {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH);
        }
        let mut p: Array<usize> = array![0];
        let mut out_i: Array<usize> = array![];
        let mut out_v: Array<T> = array![];
        let mut ap = a.p;
        let mut bp = b.p;
        let mut ai = a.i;
        let mut av = a.vals;
        let mut bi = b.i;
        let mut bv = b.vals;
        let mut a_start = *ap.pop_front().unwrap();
        let mut b_start = *bp.pop_front().unwrap();
        while let Some(a_end) = ap.pop_front() {
            let a_end = *a_end;
            let b_end = *bp.pop_front().unwrap();
            let mut na = a_end - a_start;
            let mut nb = b_end - b_start;
            // Merge the two sorted columns.
            while na != 0 && nb != 0 {
                let ra = *ai[0];
                let rb = *bi[0];
                if ra < rb {
                    out_i.append(ra);
                    out_v.append(*av.pop_front().unwrap());
                    ai.pop_front().unwrap();
                    na -= 1;
                } else if rb < ra {
                    out_i.append(rb);
                    out_v.append(*bv.pop_front().unwrap());
                    bi.pop_front().unwrap();
                    nb -= 1;
                } else {
                    out_i.append(ra);
                    out_v.append(*av.pop_front().unwrap() + *bv.pop_front().unwrap());
                    ai.pop_front().unwrap();
                    bi.pop_front().unwrap();
                    na -= 1;
                    nb -= 1;
                }
            }
            while na != 0 {
                out_i.append(*ai.pop_front().unwrap());
                out_v.append(*av.pop_front().unwrap());
                na -= 1;
            }
            while nb != 0 {
                out_i.append(*bi.pop_front().unwrap());
                out_v.append(*bv.pop_front().unwrap());
                nb -= 1;
            }
            p.append(out_v.len());
            a_start = a_end;
            b_start = b_end;
        }
        CsMatrix {
            nrows: a.nrows, ncols: a.ncols, p: p.span(), i: out_i.span(), vals: out_v.span(),
        }
    }

    fn mul(a: CsMatrix<T>, b: CsMatrix<T>) -> CsMatrix<T> {
        if a.ncols != b.nrows {
            core::panic_with_felt252(errors::DIMENSION_MISMATCH);
        }
        let zero = R::zero();
        // The rows of `a` (its transpose): `a[r, c]` for `c` in `tj[tp[r]..tp[r + 1]]`.
        let (tp, tj, src) = transpose_pattern(a.nrows, a.p, a.i);
        let tv = gather(a.vals, src);
        let mut p: Array<usize> = array![0];
        let mut out_i: Array<usize> = array![];
        let mut out_v: Array<T> = array![];
        let mut bp = b.p;
        let mut start = *bp.pop_front().unwrap();
        while let Some(end) = bp.pop_front() {
            let end = *end;
            let rows_j = b.i.slice(start, end - start);
            // The pattern of the output column: the union of the columns of `a` it combines.
            let mut pattern: Span<usize> = array![].span();
            let mut rr = rows_j;
            while let Some(k) = rr.pop_front() {
                let ks = *a.p[*k];
                pattern = union_sorted(pattern, a.i.slice(ks, *a.p[*k + 1] - ks));
            }
            if pattern.len() != 0 {
                let dense = Self::dense_column(rows_j, b.vals.slice(start, end - start), b.nrows);
                while let Some(r) = pattern.pop_front() {
                    let r = *r;
                    let rs = *tp[r];
                    let len = *tp[r + 1] - rs;
                    let mut cj = tj.slice(rs, len);
                    let mut cv = tv.slice(rs, len);
                    let mut w = R::wide_zero();
                    while let Some(c) = cj.pop_front() {
                        w = R::wide_add_prod(w, *cv.pop_front().unwrap(), *dense[*c]);
                    }
                    let x = R::wide_rescale(w);
                    if x != zero {
                        out_i.append(r);
                        out_v.append(x);
                    }
                }
            }
            p.append(out_v.len());
            start = end;
        }
        CsMatrix {
            nrows: a.nrows, ncols: b.ncols, p: p.span(), i: out_i.span(), vals: out_v.span(),
        }
    }

    fn dense_column(mut rows: Span<usize>, mut vals: Span<T>, n: usize) -> Span<T> {
        let zero = R::zero();
        let mut out: Array<T> = array![];
        let mut next: usize = 0;
        while let Some(r) = rows.pop_front() {
            let r = *r;
            append_n(ref out, r - next, zero);
            out.append(*vals.pop_front().unwrap());
            next = r + 1;
        }
        append_n(ref out, n - next, zero);
        out.span()
    }
}
