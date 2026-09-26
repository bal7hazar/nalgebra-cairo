//! Gas benchmarks (steps criterion, WP 8.6-P21): `Sum` / `Product` against the hand-written
//! chain and upstream's fold from the identity, the macros against the constructors they expand
//! to (identical by construction), and the crate-root functions against the method / the
//! formulation upstream writes. Operands through `black_box` in every variant of a group, the
//! baseline included, so `net` is the operation alone.

use fixed::Fixed;
use nalgebra::base::point3::{Point3, Point3Trait};
use nalgebra::{
    DMatrix, DMatrixTrait, Matrix2, Matrix3, Matrix3Trait, Matrix4, Matrix4Trait, Point5,
    Point5Trait, Vector3Trait, Vector5Trait, center, distance, dmatrix, inf, matrix, stack,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int, m3i};

fn quad() -> (Matrix3<Fixed>, Matrix3<Fixed>, Matrix3<Fixed>, Matrix3<Fixed>) {
    black_box(
        (
            m3i([[1, 2, 3], [4, 5, 6], [7, 8, 10]]),
            m3i([[0, 1, 0], [-1, 0, 2], [3, 1, 1]]),
            m3i([[2, 0, 0], [0, 3, 0], [1, 0, 1]]),
            m3i([[1, -1, 0], [0, 1, 1], [2, 0, -1]]),
        ),
    )
}

/// Upstream's `Sum` formulation, `iter.fold(Matrix::zero(), |acc, x| acc + x)` (the losing
/// variant: one addition more than the library's fold from the first item).
fn sum_from_zero(mut items: Span<Matrix3<Fixed>>) -> Matrix3<Fixed> {
    let mut acc: Matrix3<Fixed> = Matrix3Trait::zeros();
    while let Option::Some(x) = items.pop_front() {
        acc = acc + *x;
    }
    acc
}

/// Candidate: the library's loop unrolled twice (one loop iteration per two items).
fn sum_unrolled2(mut iter: core::array::ArrayIter<Matrix3<Fixed>>) -> Matrix3<Fixed> {
    let Option::Some(mut acc) = iter.next() else {
        return Matrix3Trait::zeros();
    };
    loop {
        match iter.next() {
            Option::Some(x) => {
                acc = acc + x;
                match iter.next() {
                    Option::Some(y) => { acc = acc + y; },
                    Option::None => { break; },
                }
            },
            Option::None => { break; },
        }
    }
    acc
}

/// Candidate: the library's loop unrolled 4 times.
fn sum_unrolled4(mut iter: core::array::ArrayIter<Matrix3<Fixed>>) -> Matrix3<Fixed> {
    let Option::Some(mut acc) = iter.next() else {
        return Matrix3Trait::zeros();
    };
    loop {
        let Option::Some(x) = iter.next() else {
            break;
        };
        acc = acc + x;
        let Option::Some(x) = iter.next() else {
            break;
        };
        acc = acc + x;
        let Option::Some(x) = iter.next() else {
            break;
        };
        acc = acc + x;
        let Option::Some(x) = iter.next() else {
            break;
        };
        acc = acc + x;
    }
    acc
}

/// Candidate: `Iterator::fold` with a closure, from the first item.
fn sum_fold(mut iter: core::array::ArrayIter<Matrix3<Fixed>>) -> Matrix3<Fixed> {
    let Option::Some(first) = iter.next() else {
        return Matrix3Trait::zeros();
    };
    iter.fold(first, |acc, x| acc + x)
}

// --- Sum of 4 Matrix3
// ------------------------------------------------------------------------------
// -----------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__baseline() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__chain() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s = a + b + c + d;
    assert!(s.m11 == int(4));
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__library() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s: Matrix3<Fixed> = items.into_iter().sum();
    assert!(s.m11 == int(4));
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__snapshots() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s: Matrix3<Fixed> = *items.span().into_iter().sum();
    assert!(s.m11 == int(4));
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__alt_unrolled2() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s = sum_unrolled2(items.into_iter());
    assert!(s.m11 == int(4));
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__alt_unrolled4() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s = sum_unrolled4(items.into_iter());
    assert!(s.m11 == int(4));
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__alt_fold() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s = sum_fold(items.into_iter());
    assert!(s.m11 == int(4));
}

#[test]
#[inline(never)]
fn bench_sum_matrix3_4__upstream_from_zero() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s = sum_from_zero(items.span());
    assert!(s.m11 == int(4));
}

// --- Product of 4 Matrix3
// -------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_product_matrix3_4__baseline() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
}

#[test]
#[inline(never)]
fn bench_product_matrix3_4__chain() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let p = a * b * c * d;
    assert!(p.m11 != int(12345));
}

/// The owned form: corelib's blanket `Product` (a fold from `one()`, like upstream).
#[test]
#[inline(never)]
fn bench_product_matrix3_4__owned_corelib() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let p: Matrix3<Fixed> = items.into_iter().product();
    assert!(p.m11 != int(12345));
}

/// The snapshot form: the library's fold from the first item.
#[test]
#[inline(never)]
fn bench_product_matrix3_4__snapshots() {
    let (a, b, c, d) = quad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let p: Matrix3<Fixed> = *items.span().into_iter().product();
    assert!(p.m11 != int(12345));
}

// --- Sum of 4 DMatrix 3x3
// -------------------------------------------------------------------------

fn dquad() -> (DMatrix<Fixed>, DMatrix<Fixed>, DMatrix<Fixed>, DMatrix<Fixed>) {
    let (a, b, c, d) = quad();
    (a.into(), b.into(), c.into(), d.into())
}

#[test]
#[inline(never)]
fn bench_sum_dmatrix3_4__baseline() {
    let (a, b, c, d) = dquad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
}

#[test]
#[inline(never)]
fn bench_sum_dmatrix3_4__chain() {
    let (a, b, c, d) = dquad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s = a + b + c + d;
    assert!(s.len() == 9);
}

#[test]
#[inline(never)]
fn bench_sum_dmatrix3_4__library() {
    let (a, b, c, d) = dquad();
    let items = array![a, b, c, d];
    assert!(items.len() == 4);
    let s: DMatrix<Fixed> = items.into_iter().sum();
    assert!(s.len() == 9);
}

// --- macros against their expansion
// ---------------------------------------------------------------

fn nine() -> Span<Fixed> {
    black_box(array![int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9)].span())
}

#[test]
#[inline(never)]
fn bench_matrix_macro3__baseline() {
    let v = nine();
    assert!(v.len() == 9);
}

#[test]
#[inline(never)]
fn bench_matrix_macro3__macro() {
    let v = nine();
    assert!(v.len() == 9);
    let m: Matrix3<Fixed> = matrix![
        * v[0],
        * v[1],
        * v[2];
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8]];
    assert!(black_box(m).m12 == int(2));
}

#[test]
#[inline(never)]
fn bench_matrix_macro3__new() {
    let v = nine();
    assert!(v.len() == 9);
    let m = Matrix3Trait::new(*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8]);
    assert!(black_box(m).m12 == int(2));
}

#[test]
#[inline(never)]
fn bench_dmatrix_macro3__baseline() {
    let v = nine();
    assert!(v.len() == 9);
}

#[test]
#[inline(never)]
fn bench_dmatrix_macro3__macro() {
    let v = nine();
    assert!(v.len() == 9);
    let m: DMatrix<Fixed> = dmatrix![
        * v[0],
        * v[1],
        * v[2];
        * v[3],
        * v[4],
        * v[5];
        * v[6],
        * v[7],
        * v[8]];
    assert!(black_box(m).len() == 9);
}

#[test]
#[inline(never)]
fn bench_dmatrix_macro3__from_row_slice() {
    let v = nine();
    assert!(v.len() == 9);
    let m = DMatrixTrait::from_row_slice(
        3, 3, array![*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8]].span(),
    );
    assert!(black_box(m).len() == 9);
}

fn blocks() -> (Matrix2<Fixed>, Matrix2<Fixed>, Matrix2<Fixed>, Matrix2<Fixed>) {
    black_box(
        (
            matrix![int (1), int (2); int (5), int (6)],
            matrix![int (3), int (4); int (7), int (8)],
            matrix![int (9), int (10); int (13), int (14)],
            matrix![int (11), int (12); int (15), int (16)],
        ),
    )
}

#[test]
#[inline(never)]
fn bench_stack_2x2_blocks__baseline() {
    let (a, b, c, d) = blocks();
    assert!(a != d && b != c);
}

#[test]
#[inline(never)]
fn bench_stack_2x2_blocks__macro() {
    let (a, b, c, d) = blocks();
    assert!(a != d && b != c);
    let m: Matrix4<Fixed> = stack![a, b; c, d];
    assert!(black_box(m).m23 == int(7));
}

#[test]
#[inline(never)]
fn bench_stack_2x2_blocks__new() {
    let (a, b, c, d) = blocks();
    assert!(a != d && b != c);
    let m = Matrix4Trait::new(
        a.m11,
        a.m12,
        b.m11,
        b.m12,
        a.m21,
        a.m22,
        b.m21,
        b.m22,
        c.m11,
        c.m12,
        d.m11,
        d.m12,
        c.m21,
        c.m22,
        d.m21,
        d.m22,
    );
    assert!(black_box(m).m23 == int(7));
}

// --- crate-root functions
// -------------------------------------------------------------------------

fn points3() -> (Point3<Fixed>, Point3<Fixed>) {
    black_box(
        (
            Point3Trait::new(fx(0x180000000), fx(-0x280000000), fx(0x40000000)),
            Point3Trait::new(fx(0x3c0000000), fx(0x1c0000000), fx(-0x100000000)),
        ),
    )
}

#[test]
#[inline(never)]
fn bench_distance_point3__baseline() {
    let (p, q) = points3();
    assert!(p != q);
}

#[test]
#[inline(never)]
fn bench_distance_point3__root() {
    let (p, q) = points3();
    assert!(p != q);
    assert!(distance(p, q) > int(0));
}

/// Upstream's formulation, `(p2.coords - p1.coords).norm()`.
#[test]
#[inline(never)]
fn bench_distance_point3__coords() {
    let (p, q) = points3();
    assert!(p != q);
    assert!((q.coords() - p.coords()).norm() > int(0));
}

#[test]
#[inline(never)]
fn bench_center_point3__baseline() {
    let (p, q) = points3();
    assert!(p != q);
}

#[test]
#[inline(never)]
fn bench_center_point3__root() {
    let (p, q) = points3();
    assert!(p != q);
    assert!(center(p, q).x > int(0));
}

/// Upstream's formulation, `(p1.coords + p2.coords) * 0.5` (`scale`).
#[test]
#[inline(never)]
fn bench_center_point3__coords() {
    let (p, q) = points3();
    assert!(p != q);
    assert!((p.coords() + q.coords()).scale(fx(0x80000000)).x > int(0));
}

fn points5() -> (Point5<Fixed>, Point5<Fixed>) {
    black_box(
        (
            Point5Trait::new(fx(0x180000000), fx(-0x280000000), fx(0x40000000), int(2), int(-1)),
            Point5Trait::new(fx(0x3c0000000), fx(0x1c0000000), fx(-0x100000000), int(3), int(4)),
        ),
    )
}

#[test]
#[inline(never)]
fn bench_distance_point5__baseline() {
    let (p, q) = points5();
    assert!(p != q);
}

#[test]
#[inline(never)]
fn bench_distance_point5__root() {
    let (p, q) = points5();
    assert!(p != q);
    assert!(distance(p, q) > int(0));
}

/// Upstream's formulation, `(p2.coords - p1.coords).norm()`.
#[test]
#[inline(never)]
fn bench_distance_point5__coords() {
    let (p, q) = points5();
    assert!(p != q);
    assert!((q.coords() - p.coords()).norm() > int(0));
}

#[test]
#[inline(never)]
fn bench_inf_matrix3__baseline() {
    let (a, b, _, _) = quad();
    assert!(a != b);
}

#[test]
#[inline(never)]
fn bench_inf_matrix3__root() {
    let (a, b, _, _) = quad();
    assert!(a != b);
    assert!(inf(a, b).m11 == int(0));
}

#[test]
#[inline(never)]
fn bench_inf_matrix3__method() {
    let (a, b, _, _) = quad();
    assert!(a != b);
    assert!(a.inf(b).m11 == int(0));
}

/// The losing formulation of `dmatrix!`'s row-length check: a `usize` `!=` chain, not folded.
#[test]
#[inline(never)]
fn bench_dmatrix_macro3__alt_usize_check() {
    let v = nine();
    assert!(v.len() == 9);
    let ncols = 3;
    if 3 != ncols || 3 != ncols {
        core::panic_with_felt252('x');
    }
    let m = DMatrixTrait::from_row_slice(
        3, ncols, array![*v[0], *v[1], *v[2], *v[3], *v[4], *v[5], *v[6], *v[7], *v[8]].span(),
    );
    assert!(black_box(m).len() == 9);
}
