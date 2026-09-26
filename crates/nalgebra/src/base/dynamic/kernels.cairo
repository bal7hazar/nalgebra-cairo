//! Loops of the dynamic matrices on column-major `Span`s (DESIGN D5): the element-wise maps, the
//! dot product (straight-line up to 6 terms, `multi_pop_front::<4>` chunks beyond), the product
//! (the left factor transposed once, every output ONE fused sum of products of two contiguous
//! runs) and the copies of the edition operations. Crate-internal: the public forms are the
//! methods of `DMatrix`, `DVector` and `RowDVector`.
//!
//! Measured (`crates/tests_dynamic/src/layout.cairo`, `Fixed`): the dot product of 6 terms costs
//! 6 870 gas straight-line, 14 240 with 4-chunks, 15 670 with one `pop_front` per term; of 16
//! terms, 21 430 with 4-chunks, 19 950 with 8-chunks, 35 370 with `pop_front`, 58 370 indexed.

use simba::scalar::Real;
use super::super::errors;

/// The loops on column-major spans. Methods of a generic impl (AGENTS: no generic free
/// functions), instantiated per scalar.
pub(crate) trait DynKernels<T> {
    /// `n` copies of `val`.
    fn filled(n: usize, val: T) -> Array<T>;
    /// Appends `n` copies of `val` to `out`.
    fn append_n(ref out: Array<T>, n: usize, val: T);
    /// `a + b` component-wise (same length, checked by the caller).
    fn add(a: Span<T>, b: Span<T>) -> Span<T>;
    /// `a - b` component-wise.
    fn sub(a: Span<T>, b: Span<T>) -> Span<T>;
    /// `-a`.
    fn neg(a: Span<T>) -> Span<T>;
    /// `a * k` component-wise, one floor per component.
    fn scale(a: Span<T>, k: T) -> Span<T>;
    /// `a / k` component-wise, one correctly rounded division per component.
    fn unscale(a: Span<T>, k: T) -> Span<T>;
    /// `a .* b`, one floor per component.
    fn component_mul(a: Span<T>, b: Span<T>) -> Span<T>;
    /// `sum a[k] b[k]`, accumulated exactly and floored once.
    fn dot(a: Span<T>, b: Span<T>) -> T;
    /// The sum of squares, accumulated exactly and floored once.
    fn norm_squared(a: Span<T>) -> T;
    /// The square root of the exact (unscaled) sum of squares, floored once.
    fn norm(a: Span<T>) -> T;
    /// Whether every pair of components is within `ulps` raw units.
    fn abs_diff_eq(a: Span<T>, b: Span<T>, ulps: u64) -> bool;
    /// The transpose of the column-major `nrows x ncols` `a` (column-major).
    fn transpose(a: Span<T>, nrows: usize, ncols: usize) -> Span<T>;
    /// The column-major product of `a` (`m x k`) and `b` (`k x n`).
    fn mul(a: Span<T>, m: usize, k: usize, b: Span<T>, n: usize) -> Span<T>;
    /// `a` with `n` columns of `val` inserted before column `i` (`i <= ncols`).
    fn insert_columns(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize, val: T) -> Span<T>;
    /// `a` with `n` rows of `val` inserted before row `i` (`i <= nrows`).
    fn insert_rows(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize, val: T) -> Span<T>;
    /// `a` without the columns `i .. i + n` (`i + n <= ncols`).
    fn remove_columns(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize) -> Span<T>;
    /// `a` without the rows `i .. i + n` (`i + n <= nrows`).
    fn remove_rows(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize) -> Span<T>;
    /// `a` without the columns listed in `indices`; returns the data and the new column count.
    fn remove_columns_at(
        a: Span<T>, nrows: usize, ncols: usize, indices: Span<usize>,
    ) -> (Span<T>, usize);
    /// `a` without the rows listed in `indices`; returns the data and the new row count.
    fn remove_rows_at(
        a: Span<T>, nrows: usize, ncols: usize, indices: Span<usize>,
    ) -> (Span<T>, usize);
    /// `a` resized to `new_nrows x new_ncols`, the common top-left block kept, the rest `val`.
    fn resize(
        a: Span<T>, nrows: usize, ncols: usize, new_nrows: usize, new_ncols: usize, val: T,
    ) -> Span<T>;
    /// The `nrows x ncols` matrix with `elts` on its first diagonal components, zeros elsewhere.
    fn partial_diagonal(nrows: usize, ncols: usize, elts: Span<T>) -> Span<T>;
    /// The column-major data of the `nrows x ncols` matrix whose ROW-major data is `a`.
    fn from_row_major(a: Span<T>, nrows: usize, ncols: usize) -> Span<T>;
}

/// The component at the linear (column-major) index `k`, `None` out of bounds.
pub(crate) fn get_linear<T, +Copy<T>>(data: Span<T>, k: usize) -> Option<T> {
    match data.get(k) {
        Some(x) => Some(*x.unbox()),
        None => None,
    }
}

/// The component at the linear index `k`; panics with `nalgebra: index out of bounds`.
pub(crate) fn at_linear<T, +Copy<T>>(data: Span<T>, k: usize) -> T {
    match data.get(k) {
        Some(x) => *x.unbox(),
        None => core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS),
    }
}

/// Whether `indices` contains `k`.
fn contains(mut indices: Span<usize>, k: usize) -> bool {
    loop {
        match indices.pop_front() {
            Some(x) => { if *x == k {
                break true;
            } },
            None => { break false; },
        }
    }
}

/// Appends the whole of `run` to `out`.
fn append_run<T, +Copy<T>, +Drop<T>>(ref out: Array<T>, mut run: Span<T>) {
    while let Some(x) = run.multi_pop_front::<4>() {
        let [x0, x1, x2, x3] = x.unbox();
        out.append(x0);
        out.append(x1);
        out.append(x2);
        out.append(x3);
    }
    while let Some(x) = run.pop_front() {
        out.append(*x);
    }
}

/// The exact sum of squares of `a`.
fn sum_squares<T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>>(mut a: Span<T>) -> R::Wide {
    let mut w = R::wide_zero();
    while let Some(x) = a.multi_pop_front::<4>() {
        let [x0, x1, x2, x3] = x.unbox();
        w = R::wide_add_prod(R::wide_add_prod(w, x0, x0), x1, x1);
        w = R::wide_add_prod(R::wide_add_prod(w, x2, x2), x3, x3);
    }
    while let Some(x) = a.pop_front() {
        w = R::wide_add_prod(w, *x, *x);
    }
    w
}

pub(crate) impl DynKernelsImpl<
    T,
    impl R: Real<T>,
    +Copy<T>,
    +Drop<T>,
    +Drop<R::Wide>,
    +Add<T>,
    +Sub<T>,
    +Mul<T>,
    +Neg<T>,
> of DynKernels<T> {
    fn filled(n: usize, val: T) -> Array<T> {
        let mut out: Array<T> = array![];
        Self::append_n(ref out, n, val);
        out
    }

    fn append_n(ref out: Array<T>, n: usize, val: T) {
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

    fn add(mut a: Span<T>, mut b: Span<T>) -> Span<T> {
        let mut out: Array<T> = array![];
        while let Some(x) = a.multi_pop_front::<4>() {
            let [x0, x1, x2, x3] = x.unbox();
            let [y0, y1, y2, y3] = b.multi_pop_front::<4>().unwrap().unbox();
            out.append(x0 + y0);
            out.append(x1 + y1);
            out.append(x2 + y2);
            out.append(x3 + y3);
        }
        while let Some(x) = a.pop_front() {
            out.append(*x + *b.pop_front().unwrap());
        }
        out.span()
    }

    fn sub(mut a: Span<T>, mut b: Span<T>) -> Span<T> {
        let mut out: Array<T> = array![];
        while let Some(x) = a.multi_pop_front::<4>() {
            let [x0, x1, x2, x3] = x.unbox();
            let [y0, y1, y2, y3] = b.multi_pop_front::<4>().unwrap().unbox();
            out.append(x0 - y0);
            out.append(x1 - y1);
            out.append(x2 - y2);
            out.append(x3 - y3);
        }
        while let Some(x) = a.pop_front() {
            out.append(*x - *b.pop_front().unwrap());
        }
        out.span()
    }

    fn neg(mut a: Span<T>) -> Span<T> {
        let mut out: Array<T> = array![];
        while let Some(x) = a.multi_pop_front::<4>() {
            let [x0, x1, x2, x3] = x.unbox();
            out.append(-x0);
            out.append(-x1);
            out.append(-x2);
            out.append(-x3);
        }
        while let Some(x) = a.pop_front() {
            out.append(-*x);
        }
        out.span()
    }

    fn scale(mut a: Span<T>, k: T) -> Span<T> {
        let mut out: Array<T> = array![];
        while let Some(x) = a.multi_pop_front::<4>() {
            let [x0, x1, x2, x3] = x.unbox();
            out.append(x0 * k);
            out.append(x1 * k);
            out.append(x2 * k);
            out.append(x3 * k);
        }
        while let Some(x) = a.pop_front() {
            out.append(*x * k);
        }
        out.span()
    }

    fn unscale(mut a: Span<T>, k: T) -> Span<T> {
        let mut out: Array<T> = array![];
        while let Some(x) = a.pop_front() {
            out.append(R::div(*x, k));
        }
        out.span()
    }

    fn component_mul(mut a: Span<T>, mut b: Span<T>) -> Span<T> {
        let mut out: Array<T> = array![];
        while let Some(x) = a.multi_pop_front::<4>() {
            let [x0, x1, x2, x3] = x.unbox();
            let [y0, y1, y2, y3] = b.multi_pop_front::<4>().unwrap().unbox();
            out.append(x0 * y0);
            out.append(x1 * y1);
            out.append(x2 * y2);
            out.append(x3 * y3);
        }
        while let Some(x) = a.pop_front() {
            out.append(*x * *b.pop_front().unwrap());
        }
        out.span()
    }

    fn dot(mut a: Span<T>, mut b: Span<T>) -> T {
        let w = R::wide_zero();
        match a.len() {
            0 => R::wide_rescale(w),
            1 => R::wide_rescale(R::wide_add_prod(w, *a[0], *b[0])),
            2 => {
                let [a0, a1] = (*a.try_into().unwrap()).unbox();
                let [b0, b1] = (*b.try_into().unwrap()).unbox();
                R::wide_rescale(R::wide_add_prod(R::wide_add_prod(w, a0, b0), a1, b1))
            },
            3 => {
                let [a0, a1, a2] = (*a.try_into().unwrap()).unbox();
                let [b0, b1, b2] = (*b.try_into().unwrap()).unbox();
                let w = R::wide_add_prod(R::wide_add_prod(w, a0, b0), a1, b1);
                R::wide_rescale(R::wide_add_prod(w, a2, b2))
            },
            4 => {
                let [a0, a1, a2, a3] = (*a.try_into().unwrap()).unbox();
                let [b0, b1, b2, b3] = (*b.try_into().unwrap()).unbox();
                let w = R::wide_add_prod(R::wide_add_prod(w, a0, b0), a1, b1);
                R::wide_rescale(R::wide_add_prod(R::wide_add_prod(w, a2, b2), a3, b3))
            },
            5 => {
                let [a0, a1, a2, a3, a4] = (*a.try_into().unwrap()).unbox();
                let [b0, b1, b2, b3, b4] = (*b.try_into().unwrap()).unbox();
                let w = R::wide_add_prod(R::wide_add_prod(w, a0, b0), a1, b1);
                let w = R::wide_add_prod(R::wide_add_prod(w, a2, b2), a3, b3);
                R::wide_rescale(R::wide_add_prod(w, a4, b4))
            },
            6 => {
                let [a0, a1, a2, a3, a4, a5] = (*a.try_into().unwrap()).unbox();
                let [b0, b1, b2, b3, b4, b5] = (*b.try_into().unwrap()).unbox();
                let w = R::wide_add_prod(R::wide_add_prod(w, a0, b0), a1, b1);
                let w = R::wide_add_prod(R::wide_add_prod(w, a2, b2), a3, b3);
                R::wide_rescale(R::wide_add_prod(R::wide_add_prod(w, a4, b4), a5, b5))
            },
            _ => {
                let mut w = w;
                while let Some(x) = a.multi_pop_front::<4>() {
                    let [x0, x1, x2, x3] = x.unbox();
                    let [y0, y1, y2, y3] = b.multi_pop_front::<4>().unwrap().unbox();
                    w = R::wide_add_prod(R::wide_add_prod(w, x0, y0), x1, y1);
                    w = R::wide_add_prod(R::wide_add_prod(w, x2, y2), x3, y3);
                }
                while let Some(x) = a.pop_front() {
                    w = R::wide_add_prod(w, *x, *b.pop_front().unwrap());
                }
                R::wide_rescale(w)
            },
        }
    }

    fn norm_squared(a: Span<T>) -> T {
        R::wide_rescale(sum_squares::<T, R>(a))
    }

    fn norm(a: Span<T>) -> T {
        R::wide_sqrt(sum_squares::<T, R>(a))
    }

    fn abs_diff_eq(mut a: Span<T>, mut b: Span<T>, ulps: u64) -> bool {
        loop {
            match a.pop_front() {
                Some(x) => { if !R::abs_diff_eq(*x, *b.pop_front().unwrap(), ulps) {
                    break false;
                } },
                None => { break true; },
            }
        }
    }

    fn transpose(a: Span<T>, nrows: usize, ncols: usize) -> Span<T> {
        let mut out: Array<T> = array![];
        let len = nrows * ncols;
        let mut i: usize = 0;
        while i != nrows {
            let mut k = i;
            while k < len {
                out.append(*a[k]);
                k += nrows;
            }
            i += 1;
        }
        out.span()
    }

    fn mul(a: Span<T>, m: usize, k: usize, b: Span<T>, n: usize) -> Span<T> {
        let mut out: Array<T> = array![];
        if k == 0 {
            Self::append_n(ref out, m * n, R::zero());
            return out.span();
        }
        let at = Self::transpose(a, m, k);
        let mut j: usize = 0;
        while j != n {
            let bj = b.slice(j * k, k);
            let mut i: usize = 0;
            while i != m {
                out.append(Self::dot(at.slice(i * k, k), bj));
                i += 1;
            }
            j += 1;
        }
        out.span()
    }

    fn insert_columns(
        a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize, val: T,
    ) -> Span<T> {
        if i > ncols {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        let mut out: Array<T> = array![];
        append_run(ref out, a.slice(0, i * nrows));
        Self::append_n(ref out, n * nrows, val);
        append_run(ref out, a.slice(i * nrows, (ncols - i) * nrows));
        out.span()
    }

    fn insert_rows(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize, val: T) -> Span<T> {
        if i > nrows {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        let mut out: Array<T> = array![];
        let mut rest = a;
        let tail = nrows - i;
        let mut j: usize = 0;
        while j != ncols {
            append_run(ref out, rest.slice(0, i));
            Self::append_n(ref out, n, val);
            append_run(ref out, rest.slice(i, tail));
            rest = rest.slice(nrows, rest.len() - nrows);
            j += 1;
        }
        out.span()
    }

    fn remove_columns(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize) -> Span<T> {
        if i + n > ncols {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        let mut out: Array<T> = array![];
        append_run(ref out, a.slice(0, i * nrows));
        append_run(ref out, a.slice((i + n) * nrows, (ncols - i - n) * nrows));
        out.span()
    }

    fn remove_rows(a: Span<T>, nrows: usize, ncols: usize, i: usize, n: usize) -> Span<T> {
        if i + n > nrows {
            core::panic_with_felt252(errors::INDEX_OUT_OF_BOUNDS)
        }
        let mut out: Array<T> = array![];
        let mut rest = a;
        let tail = nrows - i - n;
        let mut j: usize = 0;
        while j != ncols {
            append_run(ref out, rest.slice(0, i));
            append_run(ref out, rest.slice(i + n, tail));
            rest = rest.slice(nrows, rest.len() - nrows);
            j += 1;
        }
        out.span()
    }

    fn remove_columns_at(
        a: Span<T>, nrows: usize, ncols: usize, indices: Span<usize>,
    ) -> (Span<T>, usize) {
        let mut out: Array<T> = array![];
        let mut kept: usize = 0;
        let mut j: usize = 0;
        while j != ncols {
            if !contains(indices, j) {
                append_run(ref out, a.slice(j * nrows, nrows));
                kept += 1;
            }
            j += 1;
        }
        (out.span(), kept)
    }

    fn remove_rows_at(
        a: Span<T>, nrows: usize, ncols: usize, indices: Span<usize>,
    ) -> (Span<T>, usize) {
        let mut keep: Array<bool> = array![];
        let mut kept: usize = 0;
        let mut i: usize = 0;
        while i != nrows {
            let k = !contains(indices, i);
            keep.append(k);
            if k {
                kept += 1;
            }
            i += 1;
        }
        let keep = keep.span();
        let mut out: Array<T> = array![];
        let mut rest = a;
        while !rest.is_empty() {
            let mut flags = keep;
            while let Some(x) = rest.pop_front() {
                if *flags.pop_front().unwrap() {
                    out.append(*x);
                }
                if flags.is_empty() {
                    break;
                }
            }
        }
        (out.span(), kept)
    }

    fn resize(
        a: Span<T>, nrows: usize, ncols: usize, new_nrows: usize, new_ncols: usize, val: T,
    ) -> Span<T> {
        let mut out: Array<T> = array![];
        let keep_r = if nrows < new_nrows {
            nrows
        } else {
            new_nrows
        };
        let keep_c = if ncols < new_ncols {
            ncols
        } else {
            new_ncols
        };
        let mut j: usize = 0;
        while j != keep_c {
            append_run(ref out, a.slice(j * nrows, keep_r));
            Self::append_n(ref out, new_nrows - keep_r, val);
            j += 1;
        }
        Self::append_n(ref out, (new_ncols - keep_c) * new_nrows, val);
        out.span()
    }

    fn partial_diagonal(nrows: usize, ncols: usize, mut elts: Span<T>) -> Span<T> {
        let min = if nrows < ncols {
            nrows
        } else {
            ncols
        };
        if elts.len() > min {
            core::panic_with_felt252(errors::TOO_MANY_DIAGONAL)
        }
        let mut out: Array<T> = array![];
        let zero = R::zero();
        let mut j: usize = 0;
        while let Some(x) = elts.pop_front() {
            Self::append_n(ref out, j, zero);
            out.append(*x);
            Self::append_n(ref out, nrows - j - 1, zero);
            j += 1;
        }
        Self::append_n(ref out, (ncols - j) * nrows, zero);
        out.span()
    }

    fn from_row_major(a: Span<T>, nrows: usize, ncols: usize) -> Span<T> {
        Self::transpose(a, ncols, nrows)
    }
}
