//! `CsCholesky<T>`: the Cholesky factorization `A = L Lᵀ` of a sparse symmetric positive-definite
//! matrix (upstream's legacy `nalgebra::sparse::CsCholesky`): a symbolic analysis of the
//! pattern (`new_symbolic`), then numeric factorizations of values with that pattern
//! (`decompose_left_looking` / `decompose_up_looking`), `L` read with `l` / `unwrap_l`.
//!
//! Symbolic analysis (upstream's `elimination_tree` / `reach` / `nonzero_pattern`): the pattern
//! of row `k` of `L` is the set of nodes reached from the entries `A[i, k]`, `i < k`, by walking
//! up the elimination tree until a node already reached for `k`; the tree is built during the
//! same pass (a node's parent is the first row that reaches it, Liu's row-subtree
//! characterisation), the `parent` and `mark` work arrays are `Felt252Dict`s (Cairo memory is
//! write-once; they replace upstream's `forest` / `ancestor` / `marks` vectors). Every walk goes
//! up the tree, so each row's pattern is a union of ascending chains: kept sorted by merges. The
//! pattern of `L` is upstream's (the column counts are its column lengths), with the diagonal
//! always stored.
//!
//! Numeric factorization: row by row (up-looking): `L[k, j] = (A[k, j] - sum_{c < j} L[k, c]
//! L[j, c]) / L[j, j]` over the pattern of row `k` in ascending `j`, then `L[k, k] = sqrt(A[k, k]
//! - sum_c L[k, c]^2)`, each sum ONE exact accumulation floored once (the sparse dot products of
//! two sorted rows by merge), the quotient rounded to nearest. Both entry points run this gather
//! kernel (upstream's left-looking variant scatters column updates into a work vector, which
//! write-once memory cannot do); as upstream, `decompose_left_looking` reads the LOWER triangle
//! of the values (`A[i, k]`, `i >= k`) and `decompose_up_looking` the UPPER one (`A[i, k]`,
//! `i <= k`): identical results on symmetric input. `L` is then gathered into its column
//! layout.

use core::dict::Felt252Dict;
use simba::scalar::Real;
use super::cs_matrix::CsMatrix;
use super::cs_utils::{gather, transpose_pattern, union_sorted};
use super::errors as sparse_errors;

/// Marks an entry of the pattern of `L` without a value in the input (a fill-in: zero).
const NONE: usize = 0xffffffff;

/// The sparse Cholesky factorization of a symmetric positive-definite `CsMatrix` (upstream
/// `CsCholesky<T, Dyn>`): the analysed pattern and, once a numeric factorization succeeded, the
/// factor `L`. Not `Copy` / `Clone` (upstream is neither).
#[derive(Drop)]
pub struct CsCholesky<T> {
    n: usize,
    /// The number of stored entries of the analysed matrix (`original_i.len()` upstream).
    nvals: usize,
    /// The pattern of `L` by rows (strict lower part ascending, then the diagonal).
    rp: Span<usize>,
    rj: Span<usize>,
    /// Per entry of `rj`: the position in `values` of `A[j, k]` (upper triangle, column `k`) /
    /// of `A[k, j]` (lower triangle, column `j`), `NONE` for a fill-in.
    upper_src: Span<usize>,
    lower_src: Span<usize>,
    /// The column layout of `L` (rows ascending) and the row-layout position of each entry.
    lp: Span<usize>,
    li: Span<usize>,
    lsrc: Span<usize>,
    /// The values of `L` in its column layout (meaningful when `ok`).
    lvals: Span<T>,
    ok: bool,
}

/// `values[pos]` in a wide accumulator, nothing for a fill-in.
fn start_with<T, impl R: Real<T>, +Copy<T>, +Drop<T>, +Drop<R::Wide>>(
    values: Span<T>, pos: usize,
) -> R::Wide {
    if pos == NONE {
        R::wide_zero()
    } else {
        R::wide_add(R::wide_zero(), *values[pos])
    }
}

/// For each index of `pattern` (ascending), the position (`base + k`) of the same index in the
/// ascending `keys`, `NONE` when absent.
fn match_positions(
    mut pattern: Span<usize>, mut keys: Span<usize>, mut positions: Span<usize>,
) -> Span<usize> {
    let mut out: Array<usize> = array![];
    while let Some(j) = pattern.pop_front() {
        let j = *j;
        while keys.len() != 0 && *keys[0] < j {
            keys.pop_front().unwrap();
            positions.pop_front().unwrap();
        }
        if keys.len() != 0 && *keys[0] == j {
            out.append(*positions[0]);
        } else {
            out.append(NONE);
        }
    }
    out.span()
}

/// Positions `start, start + 1, .., end - 1`.
fn range(start: usize, end: usize) -> Span<usize> {
    let mut out: Array<usize> = array![];
    let mut k = start;
    while k != end {
        out.append(k);
        k += 1;
    }
    out.span()
}

/// Methods of `CsCholesky<T>` for any `Real` scalar.
#[generate_trait]
pub impl CsCholeskyImpl<
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
> of CsCholeskyTrait<T> {
    /// The symbolic analysis of `m` then its numeric factorization
    /// (`decompose_left_looking(m.values())`); `l()` is `None` when `m` is not positive
    /// definite. Upstream: `CsCholesky::new(&m)`.
    fn new(m: @CsMatrix<T>) -> CsCholesky<T> {
        let mut me = Self::new_symbolic(m);
        let _ = Self::decompose_left_looking(ref me, *m.vals);
        me
    }

    /// The symbolic analysis of the pattern of `m` (its upper triangle, `A[i, k]` with `i < k`,
    /// drives the elimination tree, like upstream's): no numeric factorization yet (`l()` is
    /// `None`). Panics with `nalgebra: matrix not square`. Upstream:
    /// `CsCholesky::new_symbolic(&m)`.
    fn new_symbolic(m: @CsMatrix<T>) -> CsCholesky<T> {
        let n = *m.nrows;
        if n != *m.ncols {
            core::panic_with_felt252(sparse_errors::NOT_SQUARE);
        }
        let mp = *m.p;
        let mi = *m.i;
        // The rows of `m` (for the lower-triangle positions).
        let (tp, tj, tsrc) = transpose_pattern(n, mp, mi);
        // `parent[j] = parent + 1` (0: not yet known), `mark[j] = k + 1` when reached for row `k`.
        let mut parent: Felt252Dict<usize> = Default::default();
        let mut mark: Felt252Dict<usize> = Default::default();
        let mut rp: Array<usize> = array![0];
        let mut rj: Array<usize> = array![];
        let mut upper_src: Array<usize> = array![];
        let mut lower_src: Array<usize> = array![];
        let mut k: usize = 0;
        while k != n {
            let cs = *mp[k];
            let ce = *mp[k + 1];
            let mut pattern: Span<usize> = array![].span();
            let mut q = cs;
            while q != ce {
                let i = *mi[q];
                if i >= k {
                    // Rows ascending: the rest of the column is not above the diagonal.
                    break;
                }
                let mut chain: Array<usize> = array![];
                let mut j = i;
                loop {
                    if mark.get(j.into()) == k + 1 {
                        break;
                    }
                    mark.insert(j.into(), k + 1);
                    chain.append(j);
                    let pj = parent.get(j.into());
                    if pj == 0 {
                        parent.insert(j.into(), k + 1);
                        break;
                    }
                    j = pj - 1;
                    if j >= k {
                        break;
                    }
                }
                pattern = union_sorted(pattern, chain.span());
                q += 1;
            }
            let mut row: Array<usize> = array![];
            row.append_span(pattern);
            row.append(k);
            let row = row.span();
            rj.append_span(row);
            rp.append(rj.len());
            upper_src.append_span(match_positions(row, mi.slice(cs, ce - cs), range(cs, ce)));
            let rs = *tp[k];
            let rl = *tp[k + 1] - rs;
            lower_src.append_span(match_positions(row, tj.slice(rs, rl), tsrc.slice(rs, rl)));
            k += 1;
        }
        let rp = rp.span();
        let rj = rj.span();
        let (lp, li, lsrc) = transpose_pattern(n, rp, rj);
        CsCholesky {
            n,
            nvals: mi.len(),
            rp,
            rj,
            upper_src: upper_src.span(),
            lower_src: lower_src.span(),
            lp,
            li,
            lsrc,
            lvals: array![].span(),
            ok: false,
        }
    }

    /// The factor `L` (lower-triangular, the analysed pattern), `None` unless the last numeric
    /// factorization succeeded. Upstream: `CsCholesky::l` (`Option<&CsMatrix>`; a copy here).
    fn l(self: @CsCholesky<T>) -> Option<CsMatrix<T>> {
        if *self.ok {
            Some(
                CsMatrix {
                    nrows: *self.n, ncols: *self.n, p: *self.lp, i: *self.li, vals: *self.lvals,
                },
            )
        } else {
            None
        }
    }

    /// The factor `L`, consuming the factorization; `None` unless the last numeric factorization
    /// succeeded. Upstream: `CsCholesky::unwrap_l`.
    fn unwrap_l(self: CsCholesky<T>) -> Option<CsMatrix<T>> {
        Self::l(@self)
    }

    /// The numeric factorization of the values `values` (in the storage order of the analysed
    /// matrix, `m.values()`), reading its LOWER triangle; returns whether it is positive definite
    /// (a pivot whose floored value is not positive fails). Panics with `nalgebra: values too
    /// short` when `values` has fewer entries than the analysed matrix. Upstream:
    /// `CsCholesky::decompose_left_looking(&values)` (same factor; see the module doc for the
    /// formulation).
    fn decompose_left_looking(ref self: CsCholesky<T>, values: Span<T>) -> bool {
        let src = self.lower_src;
        CsCholeskyNumeric::decompose(ref self, values, src)
    }

    /// As `decompose_left_looking`, reading the UPPER triangle of `values`. Upstream:
    /// `CsCholesky::decompose_up_looking(&values)`.
    fn decompose_up_looking(ref self: CsCholesky<T>, values: Span<T>) -> bool {
        let src = self.upper_src;
        CsCholeskyNumeric::decompose(ref self, values, src)
    }
}

/// The numeric kernel (crate-private: the public forms are `decompose_left_looking` /
/// `decompose_up_looking`).
#[generate_trait]
impl CsCholeskyNumericImpl<
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
> of CsCholeskyNumeric<T> {
    /// The factorization (see the module doc): `src` maps every row-layout entry of `L` to its
    /// input value.
    fn decompose(ref self: CsCholesky<T>, values: Span<T>, src: Span<usize>) -> bool {
        if values.len() < self.nvals {
            core::panic_with_felt252(sparse_errors::VALUES_TOO_SHORT);
        }
        let zero = R::zero();
        let rp = self.rp;
        let rj = self.rj;
        let n = self.n;
        // `L` by rows, in the order of `rj`.
        let mut lv: Array<T> = array![];
        let mut ok = true;
        let mut k: usize = 0;
        while k != n {
            let rs = *rp[k];
            let diag_pos = *rp[k + 1] - 1;
            let mut w_diag = start_with(values, *src[diag_pos]);
            let mut q = rs;
            while q != diag_pos {
                let j = *rj[q];
                let mut w = start_with(values, *src[q]);
                // sum_{c < j} L[k, c] L[j, c]: merge row k (entries rs..q) with row j (strict
                // part).
                let mut a = rs;
                let mut b = *rp[j];
                let jd = *rp[j + 1] - 1;
                while a != q && b != jd {
                    let ca = *rj[a];
                    let cb = *rj[b];
                    if ca == cb {
                        w = R::wide_sub_prod(w, *lv[a], *lv[b]);
                        a += 1;
                        b += 1;
                    } else if ca < cb {
                        a += 1;
                    } else {
                        b += 1;
                    }
                }
                let lkj = R::div(R::wide_rescale(w), *lv[jd]);
                lv.append(lkj);
                w_diag = R::wide_sub_prod(w_diag, lkj, lkj);
                q += 1;
            }
            let d = R::wide_rescale(w_diag);
            if !(d > zero) {
                ok = false;
                break;
            }
            lv.append(R::sqrt(d));
            k += 1;
        }
        self.ok = ok;
        if ok {
            self.lvals = gather(lv.span(), self.lsrc);
        }
        ok
    }
}
