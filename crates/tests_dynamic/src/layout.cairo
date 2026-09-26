//! Benchmarks of the storage layouts of the dynamic matrices (DESIGN D5, WP 8.5-P13): the losing
//! candidates live here, next to the benchmarks that eliminated them.
//!
//! Candidates, on `Fixed` components:
//! - `col`: a column-major `Span<T>` (upstream's `VecStorage` order) read by index,
//!   `*data[i + j * nrows]`;
//! - `colseq`: the same storage, the products reading contiguous columns with `pop_front` (the
//!   left factor transposed once, so that each output is a dot product of two contiguous runs);
//! - `row`: a row-major `Span<T>` (the former D5), rows contiguous, columns strided;
//! - `dict`: a `Felt252Dict<Nullable<T>>` keyed by the linear index (mutable in place, every read
//!   through the dictionary, squashed when dropped).
//!
//! Kernels: one component read, the sum of one column, the matrix product (6x6, 16x16) and a
//! resize that keeps the top-left block. Inner loops of the dot product: indexed `while`,
//! `pop_front` on both runs, `multi_pop_front::<4>` chunks.

use core::dict::{Felt252Dict, Felt252DictTrait};
use core::nullable::{NullableTrait, match_nullable};
use fixed::Fixed;
use nalgebra_testing::black_box;
use nalgebra_tests_utils::int;
use simba::scalar::Real;

/// `0, 1, 2, ..` as integers, then `black_box`ed (column-major or row-major is the caller's
/// reading).
fn ramp(n: usize) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut k: usize = 0;
    while k != n {
        let v: i64 = k.into();
        out.append(int(v % 7 - 3));
        k += 1;
    }
    black_box(out.span())
}

fn dict_of(data: Span<Fixed>) -> Felt252Dict<Nullable<Fixed>> {
    let mut d: Felt252Dict<Nullable<Fixed>> = Default::default();
    let mut k: usize = 0;
    let n = data.len();
    while k != n {
        d.insert(k.into(), NullableTrait::new(*data[k]));
        k += 1;
    }
    d
}

fn dict_get(ref d: Felt252Dict<Nullable<Fixed>>, k: usize) -> Fixed {
    match match_nullable(d.get(k.into())) {
        core::nullable::FromNullableResult::Null => int(0),
        core::nullable::FromNullableResult::NotNull(v) => v.unbox(),
    }
}

// --- dot products of two contiguous runs --------------------------------------------------------

fn dot_index(a: Span<Fixed>, b: Span<Fixed>) -> Fixed {
    let n = a.len();
    let mut w = Real::<Fixed>::wide_zero();
    let mut k: usize = 0;
    while k != n {
        w = Real::wide_add_prod(w, *a[k], *b[k]);
        k += 1;
    }
    Real::wide_rescale(w)
}

fn dot_pop(mut a: Span<Fixed>, mut b: Span<Fixed>) -> Fixed {
    let mut w = Real::<Fixed>::wide_zero();
    while let Some(x) = a.pop_front() {
        let y = b.pop_front().unwrap();
        w = Real::wide_add_prod(w, *x, *y);
    }
    Real::wide_rescale(w)
}

fn dot_chunk4(mut a: Span<Fixed>, mut b: Span<Fixed>) -> Fixed {
    let mut w = Real::<Fixed>::wide_zero();
    while let Some(xs) = a.multi_pop_front::<4>() {
        let [x0, x1, x2, x3] = xs.unbox();
        let [y0, y1, y2, y3] = b.multi_pop_front::<4>().unwrap().unbox();
        w = Real::wide_add_prod(w, x0, y0);
        w = Real::wide_add_prod(w, x1, y1);
        w = Real::wide_add_prod(w, x2, y2);
        w = Real::wide_add_prod(w, x3, y3);
    }
    while let Some(x) = a.pop_front() {
        let y = b.pop_front().unwrap();
        w = Real::wide_add_prod(w, *x, *y);
    }
    Real::wide_rescale(w)
}

fn dot_chunk8(mut a: Span<Fixed>, mut b: Span<Fixed>) -> Fixed {
    let mut w = Real::<Fixed>::wide_zero();
    while let Some(xs) = a.multi_pop_front::<8>() {
        let [x0, x1, x2, x3, x4, x5, x6, x7] = xs.unbox();
        let [y0, y1, y2, y3, y4, y5, y6, y7] = b.multi_pop_front::<8>().unwrap().unbox();
        w = Real::wide_add_prod(w, x0, y0);
        w = Real::wide_add_prod(w, x1, y1);
        w = Real::wide_add_prod(w, x2, y2);
        w = Real::wide_add_prod(w, x3, y3);
        w = Real::wide_add_prod(w, x4, y4);
        w = Real::wide_add_prod(w, x5, y5);
        w = Real::wide_add_prod(w, x6, y6);
        w = Real::wide_add_prod(w, x7, y7);
    }
    while let Some(xs) = a.multi_pop_front::<4>() {
        let [x0, x1, x2, x3] = xs.unbox();
        let [y0, y1, y2, y3] = b.multi_pop_front::<4>().unwrap().unbox();
        w = Real::wide_add_prod(w, x0, y0);
        w = Real::wide_add_prod(w, x1, y1);
        w = Real::wide_add_prod(w, x2, y2);
        w = Real::wide_add_prod(w, x3, y3);
    }
    while let Some(x) = a.pop_front() {
        let y = b.pop_front().unwrap();
        w = Real::wide_add_prod(w, *x, *y);
    }
    Real::wide_rescale(w)
}

/// Straight-line dot products of runs of 1 to 8, the rest through `dot_chunk4`.
fn dot_dispatch(a: Span<Fixed>, b: Span<Fixed>) -> Fixed {
    let w = Real::<Fixed>::wide_zero();
    match a.len() {
        0 => Real::wide_rescale(w),
        1 => Real::wide_rescale(Real::wide_add_prod(w, *a[0], *b[0])),
        2 => {
            let [a0, a1] = (*a.try_into().unwrap()).unbox();
            let [b0, b1] = (*b.try_into().unwrap()).unbox();
            Real::wide_rescale(Real::wide_add_prod(Real::wide_add_prod(w, a0, b0), a1, b1))
        },
        3 => {
            let [a0, a1, a2] = (*a.try_into().unwrap()).unbox();
            let [b0, b1, b2] = (*b.try_into().unwrap()).unbox();
            let w = Real::wide_add_prod(Real::wide_add_prod(w, a0, b0), a1, b1);
            Real::wide_rescale(Real::wide_add_prod(w, a2, b2))
        },
        4 => {
            let [a0, a1, a2, a3] = (*a.try_into().unwrap()).unbox();
            let [b0, b1, b2, b3] = (*b.try_into().unwrap()).unbox();
            let w = Real::wide_add_prod(Real::wide_add_prod(w, a0, b0), a1, b1);
            Real::wide_rescale(Real::wide_add_prod(Real::wide_add_prod(w, a2, b2), a3, b3))
        },
        5 => {
            let [a0, a1, a2, a3, a4] = (*a.try_into().unwrap()).unbox();
            let [b0, b1, b2, b3, b4] = (*b.try_into().unwrap()).unbox();
            let w = Real::wide_add_prod(Real::wide_add_prod(w, a0, b0), a1, b1);
            let w = Real::wide_add_prod(Real::wide_add_prod(w, a2, b2), a3, b3);
            Real::wide_rescale(Real::wide_add_prod(w, a4, b4))
        },
        6 => {
            let [a0, a1, a2, a3, a4, a5] = (*a.try_into().unwrap()).unbox();
            let [b0, b1, b2, b3, b4, b5] = (*b.try_into().unwrap()).unbox();
            let w = Real::wide_add_prod(Real::wide_add_prod(w, a0, b0), a1, b1);
            let w = Real::wide_add_prod(Real::wide_add_prod(w, a2, b2), a3, b3);
            Real::wide_rescale(Real::wide_add_prod(Real::wide_add_prod(w, a4, b4), a5, b5))
        },
        _ => dot_chunk4(a, b),
    }
}

// --- products ------------------------------------------------------------------------------------

/// Column-major `a (m x k) * b (k x n)`, every factor read by index.
fn mul_col(a: Span<Fixed>, b: Span<Fixed>, m: usize, kk: usize, n: usize) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut j: usize = 0;
    while j != n {
        let mut i: usize = 0;
        while i != m {
            let mut w = Real::<Fixed>::wide_zero();
            let mut k: usize = 0;
            while k != kk {
                w = Real::wide_add_prod(w, *a[i + k * m], *b[k + j * kk]);
                k += 1;
            }
            out.append(Real::wide_rescale(w));
            i += 1;
        }
        j += 1;
    }
    out.span()
}

/// The transpose of the column-major `m x n` `a`, column-major.
fn transpose_col(a: Span<Fixed>, m: usize, n: usize) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut i: usize = 0;
    while i != m {
        let mut k = i;
        let end = i + m * n;
        while k != end {
            out.append(*a[k]);
            k += m;
        }
        i += 1;
    }
    out.span()
}

/// Column-major product, `a` transposed once, each output the `pop_front` dot product of a row
/// of `a` and a column of `b` (two contiguous runs).
fn mul_colseq(a: Span<Fixed>, b: Span<Fixed>, m: usize, kk: usize, n: usize) -> Span<Fixed> {
    let at = transpose_col(a, m, kk);
    let mut out: Array<Fixed> = array![];
    let mut bcols = b;
    while let Some(bj) = bcols.multi_pop_front_dyn(kk) {
        let mut arows = at;
        while let Some(ai) = arows.multi_pop_front_dyn(kk) {
            out.append(dot_pop(ai, bj));
        }
    }
    out.span()
}

/// Same with the `multi_pop_front::<4>` inner loop.
fn mul_colseq4(a: Span<Fixed>, b: Span<Fixed>, m: usize, kk: usize, n: usize) -> Span<Fixed> {
    let at = transpose_col(a, m, kk);
    let mut out: Array<Fixed> = array![];
    let mut bcols = b;
    while let Some(bj) = bcols.multi_pop_front_dyn(kk) {
        let mut arows = at;
        while let Some(ai) = arows.multi_pop_front_dyn(kk) {
            out.append(dot_chunk4(ai, bj));
        }
    }
    out.span()
}

/// Same with the dot products dispatched on the inner dimension (`dot_dispatch`).
fn mul_colseqd(a: Span<Fixed>, b: Span<Fixed>, m: usize, kk: usize, n: usize) -> Span<Fixed> {
    let at = transpose_col(a, m, kk);
    let mut out: Array<Fixed> = array![];
    let mut bcols = b;
    while let Some(bj) = bcols.multi_pop_front_dyn(kk) {
        let mut arows = at;
        while let Some(ai) = arows.multi_pop_front_dyn(kk) {
            out.append(dot_dispatch(ai, bj));
        }
    }
    out.span()
}

/// Row-major `a * b`: the rows of `a` contiguous (`pop_front`), the columns of `b` strided.
fn mul_row(a: Span<Fixed>, b: Span<Fixed>, m: usize, kk: usize, n: usize) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut arows = a;
    while let Some(ai) = arows.multi_pop_front_dyn(kk) {
        let mut j: usize = 0;
        while j != n {
            let mut row = ai;
            let mut w = Real::<Fixed>::wide_zero();
            let mut k = j;
            while let Some(x) = row.pop_front() {
                w = Real::wide_add_prod(w, *x, *b[k]);
                k += n;
            }
            out.append(Real::wide_rescale(w));
            j += 1;
        }
    }
    out.span()
}

/// Dictionary storage, column-major keys.
fn mul_dict(
    ref a: Felt252Dict<Nullable<Fixed>>,
    ref b: Felt252Dict<Nullable<Fixed>>,
    m: usize,
    kk: usize,
    n: usize,
) -> Felt252Dict<Nullable<Fixed>> {
    let mut out: Felt252Dict<Nullable<Fixed>> = Default::default();
    let mut j: usize = 0;
    while j != n {
        let mut i: usize = 0;
        while i != m {
            let mut w = Real::<Fixed>::wide_zero();
            let mut k: usize = 0;
            while k != kk {
                w = Real::wide_add_prod(w, dict_get(ref a, i + k * m), dict_get(ref b, k + j * kk));
                k += 1;
            }
            out.insert((i + j * m).into(), NullableTrait::new(Real::wide_rescale(w)));
            i += 1;
        }
        j += 1;
    }
    out
}

trait SpanDyn<T> {
    fn multi_pop_front_dyn(ref self: Span<T>, n: usize) -> Option<Span<T>>;
}

impl SpanDynImpl<T> of SpanDyn<T> {
    #[inline(always)]
    fn multi_pop_front_dyn(ref self: Span<T>, n: usize) -> Option<Span<T>> {
        if self.len() < n || n == 0 {
            return None;
        }
        let head = self.slice(0, n);
        self = self.slice(n, self.len() - n);
        Some(head)
    }
}

// --- resize (keep the top-left block, fill with `val`) --------------------------------------

fn resize_col(
    a: Span<Fixed>, m: usize, n: usize, m2: usize, n2: usize, val: Fixed,
) -> Span<Fixed> {
    let mut out: Array<Fixed> = array![];
    let keep_r = if m < m2 {
        m
    } else {
        m2
    };
    let mut j: usize = 0;
    while j != n2 {
        if j < n {
            let mut col = a.slice(j * m, keep_r);
            while let Some(x) = col.pop_front() {
                out.append(*x);
            }
            let mut i = keep_r;
            while i != m2 {
                out.append(val);
                i += 1;
            }
        } else {
            let mut i: usize = 0;
            while i != m2 {
                out.append(val);
                i += 1;
            }
        }
        j += 1;
    }
    out.span()
}

fn resize_dict(ref d: Felt252Dict<Nullable<Fixed>>, m: usize, n: usize, m2: usize, n2: usize, val: Fixed) {
    // Keys are `i * 64 + j` here (row, column): growing writes the new cells only.
    let mut j: usize = 0;
    while j != n2 {
        let mut i: usize = 0;
        while i != m2 {
            if i >= m || j >= n {
                d.insert((i * 64 + j).into(), NullableTrait::new(val));
            }
            i += 1;
        }
        j += 1;
    }
}

// --- benchmarks ------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_dyn_layout_index__baseline() {
    let a = ramp(36);
    let (i, j) = black_box((4_usize, 3_usize));
    assert!(a.len() == 36 && i + j == 7);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_index__col() {
    let a = ramp(36);
    let (i, j) = black_box((4_usize, 3_usize));
    assert!(i < 6 && j < 6);
    assert!(*a[i + j * 6] == int(-2));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_index__dict() {
    let a = ramp(36);
    let (i, j) = black_box((4_usize, 3_usize));
    let mut d = dict_of(a);
    assert!(i < 6 && j < 6);
    assert!(dict_get(ref d, i + j * 6) == int(-2));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_index__dict_build_only() {
    let a = ramp(36);
    let (i, j) = black_box((4_usize, 3_usize));
    let mut d = dict_of(a);
    assert!(i < 6 && j < 6);
    assert!(dict_get(ref d, 0) == int(-3));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_column16__baseline() {
    let a = ramp(256);
    let j = black_box(5_usize);
    assert!(a.len() == 256 && j == 5);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_column16__col() {
    let a = ramp(256);
    let j = black_box(5_usize);
    let mut col = a.slice(j * 16, 16);
    let mut w = Real::<Fixed>::wide_zero();
    while let Some(x) = col.pop_front() {
        w = Real::wide_add(w, *x);
    }
    assert!(Real::<Fixed>::wide_rescale(w) != int(1000000));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_column16__row() {
    let a = ramp(256);
    let j = black_box(5_usize);
    let mut w = Real::<Fixed>::wide_zero();
    let mut k = j;
    while k < 256 {
        w = Real::wide_add(w, *a[k]);
        k += 16;
    }
    assert!(Real::<Fixed>::wide_rescale(w) != int(1000000));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot16__baseline() {
    let a = ramp(16);
    let b = ramp(16);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot16__index() {
    let a = ramp(16);
    let b = ramp(16);
    assert!(dot_index(a, b) == int(69));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot16__pop() {
    let a = ramp(16);
    let b = ramp(16);
    assert!(dot_pop(a, b) == int(69));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot16__chunk4() {
    let a = ramp(16);
    let b = ramp(16);
    assert!(dot_chunk4(a, b) == int(69));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot16__chunk8() {
    let a = ramp(16);
    let b = ramp(16);
    assert!(dot_chunk8(a, b) == int(69));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot6__baseline() {
    let a = ramp(6);
    let b = ramp(6);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot6__pop() {
    let a = ramp(6);
    let b = ramp(6);
    assert!(dot_pop(a, b) == int(19));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot6__chunk4() {
    let a = ramp(6);
    let b = ramp(6);
    assert!(dot_chunk4(a, b) == int(19));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot6__chunk8() {
    let a = ramp(6);
    let b = ramp(6);
    assert!(dot_chunk8(a, b) == int(19));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__baseline() {
    let a = ramp(36);
    let b = ramp(36);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__col() {
    let a = ramp(36);
    let b = ramp(36);
    let c = mul_col(a, b, 6, 6, 6);
    assert!(c.len() == 36);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__colseq() {
    let a = ramp(36);
    let b = ramp(36);
    let c = mul_colseq(a, b, 6, 6, 6);
    assert!(c.len() == 36);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__colseq4() {
    let a = ramp(36);
    let b = ramp(36);
    let c = mul_colseq4(a, b, 6, 6, 6);
    assert!(c.len() == 36);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__colseqd() {
    let a = ramp(36);
    let b = ramp(36);
    let c = mul_colseqd(a, b, 6, 6, 6);
    assert!(c.len() == 36);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_dot6__dispatch() {
    let a = ramp(6);
    let b = ramp(6);
    assert!(dot_dispatch(a, b) == int(19));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__row() {
    let a = ramp(36);
    let b = ramp(36);
    let c = mul_row(a, b, 6, 6, 6);
    assert!(c.len() == 36);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul6__dict() {
    let a = ramp(36);
    let b = ramp(36);
    let mut da = dict_of(a);
    let mut db = dict_of(b);
    let mut c = mul_dict(ref da, ref db, 6, 6, 6);
    assert!(dict_get(ref c, 35) != int(123456));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul16__baseline() {
    let a = ramp(256);
    let b = ramp(256);
    assert!(a.len() == b.len());
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul16__col() {
    let a = ramp(256);
    let b = ramp(256);
    let c = mul_col(a, b, 16, 16, 16);
    assert!(c.len() == 256);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul16__colseq() {
    let a = ramp(256);
    let b = ramp(256);
    let c = mul_colseq(a, b, 16, 16, 16);
    assert!(c.len() == 256);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul16__colseq4() {
    let a = ramp(256);
    let b = ramp(256);
    let c = mul_colseq4(a, b, 16, 16, 16);
    assert!(c.len() == 256);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul16__row() {
    let a = ramp(256);
    let b = ramp(256);
    let c = mul_row(a, b, 16, 16, 16);
    assert!(c.len() == 256);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_mul16__dict() {
    let a = ramp(256);
    let b = ramp(256);
    let mut da = dict_of(a);
    let mut db = dict_of(b);
    let mut c = mul_dict(ref da, ref db, 16, 16, 16);
    assert!(dict_get(ref c, 255) != int(123456));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_resize6__baseline() {
    let a = ramp(36);
    let val = black_box(int(9));
    assert!(a.len() == 36 && val == int(9));
}

#[test]
#[inline(never)]
fn bench_dyn_layout_resize6__col() {
    let a = ramp(36);
    let val = black_box(int(9));
    let c = resize_col(a, 6, 6, 7, 8, val);
    assert!(c.len() == 56);
}

#[test]
#[inline(never)]
fn bench_dyn_layout_resize6__dict() {
    let a = ramp(36);
    let val = black_box(int(9));
    let mut d = dict_of(a);
    resize_dict(ref d, 6, 6, 7, 8, val);
    assert!(dict_get(ref d, 6 * 64 + 7) == val);
}

/// The candidates agree.
#[test]
fn test_dyn_layout_products_agree() {
    let a = ramp(36);
    let b = ramp(30);
    let c = mul_col(a, b, 6, 6, 5);
    let s = mul_colseq(a, b, 6, 6, 5);
    let s4 = mul_colseq4(a, b, 6, 6, 5);
    assert!(c == s && c == s4 && c == mul_colseqd(a, b, 6, 6, 5));
    // Row-major: `b` read as a 6x5 row-major matrix is the transpose layout; compare on squares.
    let b6 = ramp(36);
    let at = transpose_col(a, 6, 6);
    let bt = transpose_col(b6, 6, 6);
    // (A B)^T = B^T A^T; row-major A B is column-major (A B)^T.
    let r = mul_row(at, bt, 6, 6, 6);
    let ct = transpose_col(mul_col(a, b6, 6, 6, 6), 6, 6);
    assert!(r == ct);
    let mut da = dict_of(a);
    let mut db = dict_of(b6);
    let mut cd = mul_dict(ref da, ref db, 6, 6, 6);
    let cc = mul_col(a, b6, 6, 6, 6);
    let mut k: usize = 0;
    while k != 36 {
        assert!(dict_get(ref cd, k) == *cc[k]);
        k += 1;
    }
    assert!(dot_index(a, b6) == dot_pop(a, b6) && dot_pop(a, b6) == dot_chunk4(a, b6));
}
