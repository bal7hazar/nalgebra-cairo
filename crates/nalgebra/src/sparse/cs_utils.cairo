//! `cumsum` (upstream `sparse::cs_utils::cumsum`) and the crate-internal index kernels of the
//! sparse module.
//!
//! Cairo memory is write-once: upstream's scatters (`workspace[row] += 1`, `res.data.i[shift] =
//! j`) have no direct counterpart. The kernels here reorder entries by SORTING packed `u128` keys
//! (`major << 32 | position`, unique, so no stability question) with a bottom-up merge of the
//! runs that are already sorted (the columns of a compressed matrix, the ascending stretches of a
//! triplet list), then GATHER the values through the recovered positions (span reads are random
//! access). Every loop walks spans front to back.

use crate::base::dynamic::DVector;
use crate::base::errors;

/// `2^32`: the packing factor of the sort keys (`major * 2^32 + position`).
const SHIFT: u128 = 0x100000000;
const SHIFT_NZ: NonZero<u128> = 0x100000000;

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

/// Appends the whole of `run` to `out`.
fn append_all(ref out: Array<u128>, mut run: Span<u128>) {
    while let Some(x) = run.pop_front() {
        out.append(*x);
    }
}

/// Appends the merge of the ascending runs `a` and `b` to `out` (keys are unique).
fn merge_runs(ref out: Array<u128>, mut a: Span<u128>, mut b: Span<u128>) {
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
        if x < y {
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

/// `keys` sorted ascending, given the exclusive ends of its ascending runs (`ends` strictly
/// increasing, its last element `keys.len()`): pairs of adjacent runs are merged until one is
/// left (`ceil(log2(runs))` passes of `keys.len()` appends each).
pub(crate) fn sort_runs(mut keys: Span<u128>, mut ends: Span<usize>) -> Span<u128> {
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
                            merge_runs(
                                ref out, keys.slice(start, e1 - start), keys.slice(e1, e2 - e1),
                            );
                            new_ends.append(e2);
                            start = e2;
                        },
                        None => {
                            append_all(ref out, keys.slice(start, e1 - start));
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
    keys
}

/// The exclusive ends of the maximal ascending runs of `keys` (empty for an empty `keys`).
pub(crate) fn natural_runs(keys: Span<u128>) -> Span<usize> {
    let mut ends: Array<usize> = array![];
    let mut k = keys;
    let mut prev = match k.pop_front() {
        Some(x) => *x,
        None => { return ends.span(); },
    };
    let mut idx = 1;
    while let Some(x) = k.pop_front() {
        if *x < prev {
            ends.append(idx);
        }
        prev = *x;
        idx += 1;
    }
    ends.append(idx);
    ends.span()
}

/// Unpacks a sort key into `(major, position)` (one `DivRem`).
#[inline(always)]
pub(crate) fn unpack(key: u128) -> (usize, usize) {
    let (major, pos) = DivRem::div_rem(key, SHIFT_NZ);
    (major.try_into().unwrap(), pos.try_into().unwrap())
}

/// Packs `(major, position)` into a sort key.
#[inline(always)]
pub(crate) fn pack(major: usize, pos: usize) -> u128 {
    major.into() * SHIFT + pos.into()
}

/// The transposed pattern of the compressed pattern `(p, i)` (`p`: `nmajor + 1` pointers, `i`:
/// minor indices, ascending within each major slice) with `nminor` minor slots: returns
/// `(tp, ti, src)`, `tp` the `nminor + 1` pointers, `ti` the major index of each transposed
/// entry (ascending within each slice), `src` its position in `i`. The values follow by
/// `gather(vals, src)`.
pub(crate) fn transpose_pattern(
    nminor: usize, p: Span<usize>, i: Span<usize>,
) -> (Span<usize>, Span<usize>, Span<usize>) {
    let mut keys: Array<u128> = array![];
    let mut majors: Array<usize> = array![];
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
                keys.append(pack(*ii.pop_front().unwrap(), q));
                majors.append(j);
                q += 1;
            }
            ends.append(e);
        }
        start = e;
        j += 1;
    }
    let mut sorted = sort_runs(keys.span(), ends.span());
    let majors = majors.span();
    let mut tp: Array<usize> = array![0];
    let mut ti: Array<usize> = array![];
    let mut src: Array<usize> = array![];
    let mut row: usize = 0;
    let mut count: usize = 0;
    while let Some(key) = sorted.pop_front() {
        let (r, q) = unpack(*key);
        while row != r {
            tp.append(count);
            row += 1;
        }
        ti.append(*majors.at(q));
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
