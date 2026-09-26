//! Gas benchmarks of the sparse module, the Matrix Market parser and the convolutions. Operands
//! are built from `black_box`ed spans in every variant of a group, the baseline included, so
//! `net` is the operation alone. Sizes: the 2D Laplacian of a 4x4 grid (16 nodes, 64 entries)
//! and the 1D Laplacian of 16 nodes (46 entries).
//!
//! The measured alternatives to the library's transpose (a stable merge of the sorted columns as
//! `(row, col, position)` tuples, `sparse/cs_utils.cairo`): `bench_cs_transpose__alt_dict`,
//! upstream's counting sort, its scatters through `Felt252Dict` work arrays;
//! `bench_cs_transpose__alt_packed_keys`, the same merge on `u128` keys packing the row and the
//! position (a multiplication and a `DivRem` per entry).

use core::dict::Felt252Dict;
use fixed::Fixed;
use nalgebra::io::cs_matrix_from_matrix_market_str;
use nalgebra::sparse::{
    AxpyCs, CsCholesky, CsCholeskyTrait, CsMatrix, CsMatrixSolveTrait, CsMatrixTrait,
};
use nalgebra::{Convolution, DMatrix, DMatrixTrait, DVector, DVectorTrait, Vector6};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{int, v3i, v6i};

fn abs_diff(a: usize, b: usize) -> usize {
    if a > b {
        a - b
    } else {
        b - a
    }
}

/// The triplets of the SPD 2D Laplacian of a `g x g` grid (4 on the diagonal, -1 off it),
/// row-major (hence NOT sorted by column: `from_triplet` sorts them).
fn lap2d_triplets(g: usize) -> (Span<usize>, Span<usize>, Span<Fixed>) {
    let n = g * g;
    let mut rows: Array<usize> = array![];
    let mut cols: Array<usize> = array![];
    let mut vals: Array<Fixed> = array![];
    let mut i = 0;
    while i != n {
        let mut j = 0;
        while j != n {
            let d = abs_diff(i / g, j / g) + abs_diff(i % g, j % g);
            if d <= 1 {
                rows.append(i);
                cols.append(j);
                vals.append(if d == 0 {
                    int(4)
                } else {
                    int(-1)
                });
            }
            j += 1;
        }
        i += 1;
    }
    (black_box(rows.span()), black_box(cols.span()), black_box(vals.span()))
}

fn lap2d(g: usize) -> CsMatrix<Fixed> {
    let (r, c, v) = lap2d_triplets(g);
    CsMatrixTrait::from_triplet(g * g, g * g, r, c, v)
}

/// The SPD 1D Laplacian of `n` nodes (2 on the diagonal, -1 off it, plus 1/4 on the diagonal).
fn lap1d(n: usize) -> CsMatrix<Fixed> {
    let mut rows: Array<usize> = array![];
    let mut cols: Array<usize> = array![];
    let mut vals: Array<Fixed> = array![];
    let mut i = 0;
    while i != n {
        if i > 0 {
            rows.append(i);
            cols.append(i - 1);
            vals.append(int(-1));
        }
        rows.append(i);
        cols.append(i);
        vals.append(Fixed { raw: 0x240000000 });
        if i + 1 < n {
            rows.append(i);
            cols.append(i + 1);
            vals.append(int(-1));
        }
        i += 1;
    }
    CsMatrixTrait::from_triplet(
        n, n, black_box(rows.span()), black_box(cols.span()), black_box(vals.span()),
    )
}

fn ramp(n: usize) -> DVector<Fixed> {
    let mut out: Array<Fixed> = array![];
    let mut k: usize = 0;
    while k != n {
        let v: i64 = k.into();
        out.append(int(v % 7 - 3));
        k += 1;
    }
    DVectorTrait::from_column_slice(black_box(out.span()))
}

// --- construction ------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_cs_from_triplet__baseline() {
    let (r, c, v) = lap2d_triplets(4);
    assert!(r.len() == 64 && c.len() == 64 && v.len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_from_triplet__library() {
    let (r, c, v) = lap2d_triplets(4);
    assert!(CsMatrixTrait::from_triplet(16, 16, r, c, v).len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_into_dmatrix__baseline() {
    let m = lap2d(4);
    assert!(m.len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_into_dmatrix__library() {
    let m = lap2d(4);
    let d: DMatrix<Fixed> = m.into();
    assert!(d.len() == 256);
}

#[test]
#[inline(never)]
fn bench_cs_from_dmatrix__baseline() {
    let d: DMatrix<Fixed> = lap2d(4).into();
    assert!(d.len() == 256);
}

#[test]
#[inline(never)]
fn bench_cs_from_dmatrix__library() {
    let d: DMatrix<Fixed> = lap2d(4).into();
    let m: CsMatrix<Fixed> = black_box(d).into();
    assert!(m.len() == 64);
}

// --- transpose ---------------------------------------------------------------------------------

/// Upstream's counting sort by row, its scatters through `Felt252Dict`s: the transposed pattern
/// and values.
fn transpose_dict(m: CsMatrix<Fixed>) -> (Span<usize>, Span<usize>, Span<Fixed>) {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let len = m.len();
    let p = m.p();
    let i = m.i();
    let vals = m.values();
    let mut counts: Felt252Dict<usize> = Default::default();
    let mut ii = i;
    while let Some(r) = ii.pop_front() {
        let key: felt252 = (*r).into();
        counts.insert(key, counts.get(key) + 1);
    }
    let mut tp: Array<usize> = array![];
    let mut next: Felt252Dict<usize> = Default::default();
    let mut sum = 0;
    let mut r = 0;
    while r != nrows {
        tp.append(sum);
        next.insert(r.into(), sum);
        sum += counts.get(r.into());
        r += 1;
    }
    let mut perm: Felt252Dict<usize> = Default::default();
    let mut colof: Felt252Dict<usize> = Default::default();
    let mut j = 0;
    while j != ncols {
        let start = *p[j];
        let end = if j + 1 == ncols {
            len
        } else {
            *p[j + 1]
        };
        let mut q = start;
        while q != end {
            let key: felt252 = (*i[q]).into();
            let d = next.get(key);
            next.insert(key, d + 1);
            perm.insert(d.into(), q);
            colof.insert(d.into(), j);
            q += 1;
        }
        j += 1;
    }
    let mut ti: Array<usize> = array![];
    let mut tv: Array<Fixed> = array![];
    let mut d = 0;
    while d != len {
        ti.append(colof.get(d.into()));
        tv.append(*vals[perm.get(d.into())]);
        d += 1;
    }
    (tp.span(), ti.span(), tv.span())
}

/// The first formulation of the library's transpose: `(row << 32 | position)` keys packed into
/// one `u128`, merged by column runs, unpacked with a `DivRem`.
fn transpose_packed(m: CsMatrix<Fixed>) -> (Span<usize>, Span<usize>, Span<Fixed>) {
    let nrows = m.nrows();
    let ncols = m.ncols();
    let len = m.len();
    let p = m.p();
    let i = m.i();
    let mut keys: Array<u128> = array![];
    let mut majors: Array<usize> = array![];
    let mut ends: Array<usize> = array![];
    let mut j = 0;
    while j != ncols {
        let start = *p[j];
        let end = if j + 1 == ncols {
            len
        } else {
            *p[j + 1]
        };
        let mut q = start;
        while q != end {
            let r: u128 = (*i[q]).into();
            keys.append(r * 0x100000000 + q.into());
            majors.append(j);
            q += 1;
        }
        if end != start {
            ends.append(end);
        }
        j += 1;
    }
    let mut keys = keys.span();
    let mut ends = ends.span();
    while ends.len() > 1 {
        let mut out: Array<u128> = array![];
        let mut new_ends: Array<usize> = array![];
        let mut start = 0;
        loop {
            match ends.pop_front() {
                Some(e1) => {
                    let e1 = *e1;
                    match ends.pop_front() {
                        Some(e2) => {
                            let e2 = *e2;
                            let mut a = keys.slice(start, e1 - start);
                            let mut b = keys.slice(e1, e2 - e1);
                            loop {
                                if a.len() == 0 {
                                    out.append_span(b);
                                    break;
                                }
                                if b.len() == 0 {
                                    out.append_span(a);
                                    break;
                                }
                                if *a[0] < *b[0] {
                                    out.append(*a.pop_front().unwrap());
                                } else {
                                    out.append(*b.pop_front().unwrap());
                                }
                            }
                            new_ends.append(e2);
                            start = e2;
                        },
                        None => {
                            out.append_span(keys.slice(start, e1 - start));
                            new_ends.append(e1);
                            break;
                        },
                    }
                },
                None => { break; },
            }
        }
        keys = out.span();
        ends = new_ends.span();
    }
    let majors = majors.span();
    let vals = m.values();
    let mut tp: Array<usize> = array![0];
    let mut ti: Array<usize> = array![];
    let mut tv: Array<Fixed> = array![];
    let mut row: usize = 0;
    let mut count: usize = 0;
    while let Some(key) = keys.pop_front() {
        let (r, q) = DivRem::div_rem(*key, 0x100000000);
        let r: usize = r.try_into().unwrap();
        let q: usize = q.try_into().unwrap();
        while row != r {
            tp.append(count);
            row += 1;
        }
        ti.append(*majors[q]);
        tv.append(*vals[q]);
        count += 1;
    }
    while row != nrows {
        tp.append(count);
        row += 1;
    }
    (tp.span(), ti.span(), tv.span())
}

#[test]
#[inline(never)]
fn bench_cs_transpose__baseline() {
    let m = lap2d(4);
    assert!(m.len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_transpose__library() {
    let m = lap2d(4);
    assert!(m.transpose().len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_transpose__alt_dict() {
    let m = lap2d(4);
    let (_, ti, tv) = transpose_dict(m);
    assert!(ti.len() == 64 && tv.len() == 64);
    assert!(ti == m.transpose().i());
}

#[test]
#[inline(never)]
fn bench_cs_transpose__alt_packed_keys() {
    let m = lap2d(4);
    let (_, ti, tv) = transpose_packed(m);
    assert!(ti.len() == 64 && tv.len() == 64);
}

// --- arithmetic --------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_cs_add__baseline() {
    let a = lap2d(4);
    let b = lap1d(16);
    assert!(a.len() + b.len() == 110);
}

#[test]
#[inline(never)]
fn bench_cs_add__library() {
    let a = lap2d(4);
    let b = lap1d(16);
    assert!((a + b).len() == 70);
}

#[test]
#[inline(never)]
fn bench_cs_mul__baseline() {
    let a = lap1d(16);
    let b = lap1d(16);
    assert!(a.len() + b.len() == 92);
}

#[test]
#[inline(never)]
fn bench_cs_mul__library() {
    let a = lap1d(16);
    let b = lap1d(16);
    assert!((a * b).len() == 74);
}

#[test]
#[inline(never)]
fn bench_cs_mul_lap2d__baseline() {
    let a = lap2d(4);
    assert!(a.len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_mul_lap2d__library() {
    let a = lap2d(4);
    assert!((a * a).len() == 132);
}

#[test]
#[inline(never)]
fn bench_cs_scale__baseline() {
    let a = lap2d(4);
    let k = black_box(int(3));
    assert!(a.len() == 64 && k == int(3));
}

#[test]
#[inline(never)]
fn bench_cs_scale__library() {
    let a = lap2d(4);
    let k = black_box(int(3));
    assert!(a.scale(k).len() == 64);
}

// --- solves ------------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_cs_solve_lower__baseline() {
    let l = CsCholeskyTrait::new(@lap2d(4)).unwrap_l().unwrap();
    let b = ramp(16);
    assert!(l.len() == 67 && b.len() == 16);
}

#[test]
#[inline(never)]
fn bench_cs_solve_lower__library() {
    let l = CsCholeskyTrait::new(@lap2d(4)).unwrap_l().unwrap();
    let b = ramp(16);
    assert!(l.solve_lower_triangular(b).unwrap().len() == 16);
}

#[test]
#[inline(never)]
fn bench_cs_tr_solve_lower__baseline() {
    let l = CsCholeskyTrait::new(@lap2d(4)).unwrap_l().unwrap();
    let b = ramp(16);
    assert!(l.len() == 67 && b.len() == 16);
}

#[test]
#[inline(never)]
fn bench_cs_tr_solve_lower__library() {
    let l = CsCholeskyTrait::new(@lap2d(4)).unwrap_l().unwrap();
    let b = ramp(16);
    assert!(l.tr_solve_lower_triangular(b).unwrap().len() == 16);
}

#[test]
#[inline(never)]
fn bench_cs_solve_lower_cs__baseline() {
    let l = CsCholeskyTrait::new(@lap2d(4)).unwrap_l().unwrap();
    let b: CsMatrix<Fixed> = CsMatrixTrait::from_triplet(
        16, 1, black_box(array![5].span()), black_box(array![0].span()), array![int(1)].span(),
    );
    assert!(l.len() == 67 && b.len() == 1);
}

#[test]
#[inline(never)]
fn bench_cs_solve_lower_cs__library() {
    let l = CsCholeskyTrait::new(@lap2d(4)).unwrap_l().unwrap();
    let b: CsMatrix<Fixed> = CsMatrixTrait::from_triplet(
        16, 1, black_box(array![5].span()), black_box(array![0].span()), array![int(1)].span(),
    );
    assert!(l.solve_lower_triangular_cs(b).unwrap().len() == 11);
}

// --- Cholesky ----------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_cs_cholesky__baseline() {
    let a = lap2d(4);
    assert!(a.len() == 64);
}

#[test]
#[inline(never)]
fn bench_cs_cholesky__library() {
    let a = lap2d(4);
    let chol: CsCholesky<Fixed> = CsCholeskyTrait::new(@a);
    assert!(chol.unwrap_l().unwrap().len() == 67);
}

#[test]
#[inline(never)]
fn bench_cs_cholesky__alt_left_looking() {
    // Upstream's `new`: the lower triangle, hence a transposition of the pattern.
    let a = lap2d(4);
    let mut chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@a);
    assert!(chol.decompose_left_looking(a.values()));
    assert!(chol.unwrap_l().unwrap().len() == 67);
}

#[test]
#[inline(never)]
fn bench_cs_cholesky__symbolic() {
    let a = lap2d(4);
    let chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@a);
    assert!(chol.l().is_none());
}

#[test]
#[inline(never)]
fn bench_cs_cholesky_numeric__baseline() {
    let a = lap2d(4);
    let chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@a);
    assert!(chol.l().is_none());
}

#[test]
#[inline(never)]
fn bench_cs_cholesky_numeric__library() {
    let a = lap2d(4);
    let mut chol: CsCholesky<Fixed> = CsCholeskyTrait::new_symbolic(@a);
    assert!(chol.decompose_up_looking(a.values()));
}

#[test]
#[inline(never)]
fn bench_cs_cholesky_lap1d__baseline() {
    let a = lap1d(16);
    assert!(a.len() == 46);
}

#[test]
#[inline(never)]
fn bench_cs_cholesky_lap1d__library() {
    let a = lap1d(16);
    let chol: CsCholesky<Fixed> = CsCholeskyTrait::new(@a);
    assert!(chol.unwrap_l().unwrap().len() == 31);
}

// --- axpy_cs -----------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_axpy_cs__baseline() {
    let y = ramp(16);
    let x: CsMatrix<Fixed> = ramp(16).into();
    assert!(y.len() == 16 && x.len() == 14);
}

#[test]
#[inline(never)]
fn bench_axpy_cs__library() {
    let mut y = ramp(16);
    let x: CsMatrix<Fixed> = ramp(16).into();
    y.axpy_cs(int(2), x, int(3));
    assert!(y.len() == 16);
}

// --- convolutions ------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_convolve__baseline() {
    let v = ramp(16);
    let k = ramp(5);
    assert!(v.len() == 16 && k.len() == 5);
}

#[test]
#[inline(never)]
fn bench_convolve__full() {
    let v = ramp(16);
    let k = ramp(5);
    assert!(v.convolve_full(k).len() == 20);
}

#[test]
#[inline(never)]
fn bench_convolve__valid() {
    let v = ramp(16);
    let k = ramp(5);
    assert!(v.convolve_valid(k).len() == 12);
}

#[test]
#[inline(never)]
fn bench_convolve__same() {
    let v = ramp(16);
    let k = ramp(5);
    assert!(v.convolve_same(k).len() == 16);
}

#[test]
#[inline(never)]
fn bench_convolve_vector6__baseline() {
    let v = v6i(1, 2, 3, black_box(4), 5, 6);
    let k = v3i(1, black_box(-1), 2);
    assert!(v.x == int(1) && k.x == int(1));
}

#[test]
#[inline(never)]
fn bench_convolve_vector6__full() {
    let v = v6i(1, 2, 3, black_box(4), 5, 6);
    let k = v3i(1, black_box(-1), 2);
    assert!(v.convolve_full(k).len() == 8);
}

#[test]
#[inline(never)]
fn bench_convolve_vector6__same() {
    let v = v6i(1, 2, 3, black_box(4), 5, 6);
    let k = v3i(1, black_box(-1), 2);
    let r: Vector6<Fixed> = v.convolve_same(k);
    assert!(r.x == int(1));
}

// --- Matrix Market -----------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_matrix_market__baseline() {
    let text: ByteArray = black_box(
        "%%MatrixMarket matrix coordinate real general\n% 3x3\n3 3 4\n1 1 1.5\n2 1 -2\n3 3 .25\n1 3 2e1\n",
    );
    assert!(text.len() == 89);
}

#[test]
#[inline(never)]
fn bench_matrix_market__library() {
    let text: ByteArray = black_box(
        "%%MatrixMarket matrix coordinate real general\n% 3x3\n3 3 4\n1 1 1.5\n2 1 -2\n3 3 .25\n1 3 2e1\n",
    );
    let m: CsMatrix<Fixed> = cs_matrix_from_matrix_market_str(@text).unwrap();
    assert!(m.len() == 4);
}
