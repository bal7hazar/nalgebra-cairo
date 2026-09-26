//! `cumsum` (upstream `sparse::cs_utils::cumsum`) and the crate-internal index kernels of the
//! sparse module.
//!
//! Cairo memory is write-once: upstream's scatters (`workspace[row] += 1`, `res.data.i[shift] =
//! j`) have no direct counterpart. The kernels here reorder `(major, minor, position)` entries by
//! a stable bottom-up MERGE of the runs that are already sorted (the columns of a compressed
//! matrix, the ascending stretches of a triplet list), then GATHER the values through the
//! positions (span reads are random access). Every loop walks spans front to back.

use crate::base::dynamic::DVector;
use crate::base::errors;

/// The exclusive prefix sums of `a`, written to both `a` and `b` (`b[i] = a[0] + .. + a[i - 1]`,
/// then `a[i] = b[i]`); returns the total. Panics with `nalgebra: dimension mismatch` unless `a`
/// and `b` have the same length. Upstream: `nalgebra::sparse::cs_utils::cumsum(&mut a, &mut b)`
/// (`OVector<usize, D>`; a `DVector<usize>` here, both `ref`).
pub fn cumsum(ref a: DVector<usize>, ref b: DVector<usize>) -> usize {
    if a.data.len() != b.data.len() {
        core::panic_with_felt252(errors::DIMENSION_MISMATCH);
    }
    let mut out: Array<usize> = array![];
    let mut sum = 0;
    let mut data = a.data;
    while let Some(x) = data.pop_front() {
        out.append(sum);
        sum += *x;
    }
    let out = out.span();
    a = DVector { data: out };
    b = DVector { data: out };
    sum
}

/// An entry being reordered: `(major, minor, position)`, ordered by `(major, minor)`.
pub(crate) type Entry = (usize, usize, usize);

/// Whether `a` sorts before or with `b` (`(major, minor)` lexicographic; ties keep their order:
/// the merges are stable).
#[inline(always)]
fn le(a: Entry, b: Entry) -> bool {
    let (a0, a1, _) = a;
    let (b0, b1, _) = b;
    a0 < b0 || (a0 == b0 && a1 <= b1)
}

/// Appends the whole of `run` to `out`.
fn append_all(ref out: Array<Entry>, mut run: Span<Entry>) {
    while let Some(x) = run.pop_front() {
        out.append(*x);
    }
}

/// Appends the stable merge of the sorted runs `a` and `b` to `out`.
fn merge_runs(ref out: Array<Entry>, mut a: Span<Entry>, mut b: Span<Entry>) {
    let mut x = match a.pop_front() {
        Some(v) => *v,
        None => {
            append_all(ref out, b);
            return;
        },
    };
    let mut y = match b.pop_front() {
        Some(v) => *v,
        None => {
            out.append(x);
            append_all(ref out, a);
            return;
        },
    };
    loop {
        if le(x, y) {
            out.append(x);
            match a.pop_front() {
                Some(v) => { x = *v; },
                None => {
                    out.append(y);
                    append_all(ref out, b);
                    break;
                },
            }
        } else {
            out.append(y);
            match b.pop_front() {
                Some(v) => { y = *v; },
                None => {
                    out.append(x);
                    append_all(ref out, a);
                    break;
                },
            }
        }
    }
}

/// `entries` sorted by `(major, minor)` (stably), given the exclusive ends of its sorted runs
/// (`ends` strictly increasing, its last element `entries.len()`): pairs of adjacent runs are
/// merged until one is left (`ceil(log2(runs))` passes of `entries.len()` appends each).
///
/// Measured (`bench_cs_transpose__*`, 64 entries in 16 runs): 1 979 690 gas with keys packed
/// into one `u128` (`major << 32 | position`: a multiplication and a `DivRem` per entry) against
/// the tuples here.
pub(crate) fn sort_runs(mut entries: Span<Entry>, mut ends: Span<usize>) -> Span<Entry> {
    while ends.len() > 1 {
        let mut out: Array<Entry> = array![];
        let mut new_ends: Array<usize> = array![];
        let mut start = 0;
        loop {
            match ends.pop_front() {
                Some(e1) => {
                    let e1 = *e1;
                    match ends.pop_front() {
                        Some(e2) => {
                            let e2 = *e2;
                            merge_runs(
                                ref out,
                                entries.slice(start, e1 - start),
                                entries.slice(e1, e2 - e1),
                            );
                            new_ends.append(e2);
                            start = e2;
                        },
                        None => {
                            append_all(ref out, entries.slice(start, e1 - start));
                            new_ends.append(e1);
                            break;
                        },
                    }
                },
                None => { break; },
            }
        }
        entries = out.span();
        ends = new_ends.span();
    }
    entries
}

/// The exclusive ends of the maximal sorted runs of `entries` (empty for no entries).
pub(crate) fn natural_runs(entries: Span<Entry>) -> Span<usize> {
    let mut ends: Array<usize> = array![];
    let mut k = entries;
    let mut prev = match k.pop_front() {
        Some(x) => *x,
        None => { return ends.span(); },
    };
    let mut idx = 1;
    while let Some(x) = k.pop_front() {
        let x = *x;
        if !le(prev, x) {
            ends.append(idx);
        }
        prev = x;
        idx += 1;
    }
    ends.append(idx);
    ends.span()
}

/// The transposed pattern of the compressed pattern `(p, i)` (`p`: `nmajor + 1` pointers, `i`:
/// minor indices, ascending within each major slice) with `nminor` minor slots: returns
/// `(tp, ti, src)`, `tp` the `nminor + 1` pointers, `ti` the major index of each transposed
/// entry (ascending within each slice), `src` its position in `i`. The values follow by
/// `gather(vals, src)`. The major slices are the sorted runs of the merge.
pub(crate) fn transpose_pattern(
    nminor: usize, p: Span<usize>, i: Span<usize>,
) -> (Span<usize>, Span<usize>, Span<usize>) {
    let mut entries: Array<Entry> = array![];
    let mut ends: Array<usize> = array![];
    let mut pp = p;
    let mut start = *pp.pop_front().unwrap();
    let mut ii = i;
    let mut j: usize = 0;
    let mut q: usize = start;
    while let Some(e) = pp.pop_front() {
        let e = *e;
        if e != start {
            while q != e {
                entries.append((*ii.pop_front().unwrap(), j, q));
                q += 1;
            }
            ends.append(e);
        }
        start = e;
        j += 1;
    }
    let mut sorted = sort_runs(entries.span(), ends.span());
    let mut tp: Array<usize> = array![0];
    let mut ti: Array<usize> = array![];
    let mut src: Array<usize> = array![];
    let mut row: usize = 0;
    let mut count: usize = 0;
    while let Some(entry) = sorted.pop_front() {
        let (r, c, q) = *entry;
        while row != r {
            tp.append(count);
            row += 1;
        }
        ti.append(c);
        src.append(q);
        count += 1;
    }
    while row != nminor {
        tp.append(count);
        row += 1;
    }
    (tp.span(), ti.span(), src.span())
}

/// `data[idx[0]], data[idx[1]], ..`.
pub(crate) fn gather<T, +Copy<T>, +Drop<T>>(data: Span<T>, mut idx: Span<usize>) -> Span<T> {
    let mut out: Array<T> = array![];
    while let Some(k) = idx.pop_front() {
        out.append(*data.at(*k));
    }
    out.span()
}

/// The ascending union of two ascending index lists (common indices kept once).
pub(crate) fn union_sorted(mut a: Span<usize>, mut b: Span<usize>) -> Span<usize> {
    if a.len() == 0 {
        return b;
    }
    if b.len() == 0 {
        return a;
    }
    let mut out: Array<usize> = array![];
    let mut x = *a.pop_front().unwrap();
    let mut y = *b.pop_front().unwrap();
    loop {
        if x < y {
            out.append(x);
            match a.pop_front() {
                Some(v) => { x = *v; },
                None => {
                    out.append(y);
                    out.append_span(b);
                    break;
                },
            }
        } else if y < x {
            out.append(y);
            match b.pop_front() {
                Some(v) => { y = *v; },
                None => {
                    out.append(x);
                    out.append_span(a);
                    break;
                },
            }
        } else {
            out.append(x);
            match a.pop_front() {
                Some(v) => { x = *v; },
                None => {
                    out.append_span(b);
                    break;
                },
            }
            match b.pop_front() {
                Some(v) => { y = *v; },
                None => {
                    out.append(x);
                    out.append_span(a);
                    break;
                },
            }
        }
    }
    out.span()
}
