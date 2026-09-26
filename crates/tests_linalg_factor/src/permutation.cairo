//! Tests and gas benchmarks of the permutation sequences `Perm2/3/4/6` (upstream
//! `PermutationSequence`, `src/linalg/permutation_sequence.rs`): `len`, `is_empty`,
//! `determinant`, `append_permutation` (and its step-order contract), and the row / column
//! permutations (`PermuteRows`, `PermuteColumns`) with their inverses.

use fixed::Fixed;
use nalgebra::{
    Lu3Trait, Matrix2x3, Matrix2x3Trait, Matrix3, Matrix3Trait, Matrix3x2Trait, Matrix6,
    Matrix6Trait, MatrixMul, Perm2Trait, Perm3, Perm3Trait, Perm4Trait, Perm6, Perm6Trait,
    PermuteColumns, PermuteRows, RowVector3Trait, Vector3, Vector3Trait, Vector6, Vector6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{Perm3PartialEq, Perm6PartialEq, int};

fn v3() -> Vector3<Fixed> {
    Vector3Trait::new(int(1), int(2), int(3))
}

fn m6() -> Matrix6<Fixed> {
    Matrix6Trait::new(
        int(11),
        int(12),
        int(13),
        int(14),
        int(15),
        int(16),
        int(21),
        int(22),
        int(23),
        int(24),
        int(25),
        int(26),
        int(31),
        int(32),
        int(33),
        int(34),
        int(35),
        int(36),
        int(41),
        int(42),
        int(43),
        int(44),
        int(45),
        int(46),
        int(51),
        int(52),
        int(53),
        int(54),
        int(55),
        int(56),
        int(61),
        int(62),
        int(63),
        int(64),
        int(65),
        int(66),
    )
}

#[test]
fn test_perm_identity() {
    let p = Perm3Trait::identity();
    assert!(p.len() == 0 && p.is_empty());
    let d: Fixed = p.determinant();
    assert!(d == int(1));
    assert!(Perm2Trait::identity().is_empty() && Perm4Trait::identity().len() == 0);
    assert!(Perm6Trait::identity().is_empty());
}

/// `append_permutation` records `(min, max)` at the step `min` (0-based indices like upstream),
/// `i == i2` records nothing; `len` counts the transpositions and `determinant` is their parity.
#[test]
fn test_perm_append_len_determinant() {
    let mut p = Perm3Trait::identity();
    p.append_permutation(2, 0);
    assert!(p == Perm3 { p1: 3, p2: 2 });
    assert!(p.len() == 1 && !p.is_empty());
    let d: Fixed = p.determinant();
    assert!(d == int(-1));
    p.append_permutation(1, 1);
    assert!(p.len() == 1);
    p.append_permutation(1, 2);
    assert!(p == Perm3 { p1: 3, p2: 3 });
    assert!(p.len() == 2);
    let d: Fixed = p.determinant();
    assert!(d == int(1));
    let mut q = Perm6Trait::identity();
    q.append_permutation(0, 5);
    q.append_permutation(2, 3);
    q.append_permutation(4, 5);
    assert!(q == Perm6 { p1: 6, p2: 2, p3: 4, p4: 4, p5: 6 });
    assert!(q.len() == 3);
    let d: Fixed = q.determinant();
    assert!(d == int(-1));
}

/// A transposition whose smaller index is not after every recorded step cannot be stored in the
/// compact sequence.
#[test]
#[should_panic(expected: 'nalgebra: permutation order')]
fn test_perm_append_out_of_order_panics() {
    let mut p = Perm3Trait::identity();
    p.append_permutation(1, 2);
    p.append_permutation(0, 1);
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_perm_append_out_of_bounds_panics() {
    let mut p = Perm3Trait::identity();
    p.append_permutation(0, 3);
}

/// Rows: `(0 2)` then `(1 2)` maps `(1, 2, 3)` to `(3, 1, 2)`; the inverse replays them backwards.
#[test]
fn test_perm_permute_rows() {
    let p = Perm3 { p1: 3, p2: 3 };
    let mut v = v3();
    p.permute_rows(ref v);
    assert!(v == Vector3Trait::new(int(3), int(1), int(2)));
    p.inv_permute_rows(ref v);
    assert!(v == v3());
    let mut m = Matrix3x2Trait::new(int(1), int(10), int(2), int(20), int(3), int(30));
    p.permute_rows(ref m);
    assert!(m == Matrix3x2Trait::new(int(3), int(30), int(1), int(10), int(2), int(20)));
    p.inv_permute_rows(ref m);
    assert!(m == Matrix3x2Trait::new(int(1), int(10), int(2), int(20), int(3), int(30)));
}

/// Columns, likewise, on a row vector and on a 2x3 matrix.
#[test]
fn test_perm_permute_columns() {
    let p = Perm3 { p1: 3, p2: 3 };
    let mut r = RowVector3Trait::new(int(1), int(2), int(3));
    p.permute_columns(ref r);
    assert!(r == RowVector3Trait::new(int(3), int(1), int(2)));
    p.inv_permute_columns(ref r);
    assert!(r == RowVector3Trait::new(int(1), int(2), int(3)));
    let mut m: Matrix2x3<Fixed> = Matrix2x3Trait::new(
        int(1), int(2), int(3), int(10), int(20), int(30),
    );
    p.permute_columns(ref m);
    assert!(m == Matrix2x3Trait::new(int(3), int(1), int(2), int(30), int(10), int(20)));
}

/// On 6x6: the row permutation of `p` equals the product by its permutation matrix (the
/// identity with the rows permuted), and the column permutation by `pᵀ` undoes it.
#[test]
fn test_perm6_rows_and_columns() {
    let p = Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 };
    let mut pm: Matrix6<Fixed> = Matrix6Trait::identity();
    p.permute_rows(ref pm);
    let mut m = m6();
    p.permute_rows(ref m);
    assert!(m == pm.mul_mat(m6()));
    let mut back = m;
    p.inv_permute_rows(ref back);
    assert!(back == m6());
    let mut c = m6();
    p.permute_columns(ref c);
    let mut c2 = m6().transpose();
    p.permute_rows(ref c2);
    assert!(c == c2.transpose());
    let mut v: Vector6<Fixed> = Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6));
    p.permute_rows(ref v);
    assert!(v == pm.mul_mat(Vector6Trait::new(int(1), int(2), int(3), int(4), int(5), int(6))));
}

/// The permutation of an LU factorisation: `P A = L U` with `P` applied by `permute_rows`.
#[test]
fn test_perm_of_lu() {
    let a: Matrix3<Fixed> = black_box(
        Matrix3Trait::new(int(1), int(2), int(3), int(8), int(4), int(0), int(4), int(4), int(2)),
    );
    let f = Lu3Trait::new(a);
    let mut pa = a;
    f.p.permute_rows(ref pa);
    assert!(pa == f.l().mul_mat(f.u()));
    let mut lu = f.l().mul_mat(f.u());
    f.p.inv_permute_rows(ref lu);
    assert!(lu == a);
}

// --- benchmarks ---------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__baseline() {
    let p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 });
    let m = black_box(m6());
    assert!(p.p1 != 0 && m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__vector6() {
    let p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 });
    let m = black_box(m6());
    let mut v = m.column(0);
    p.permute_rows(ref v);
    assert!(p.p1 != 0 && v.x != int(0));
}

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__matrix6() {
    let p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 });
    let mut m = black_box(m6());
    p.permute_rows(ref m);
    assert!(p.p1 != 0 && m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__columns_matrix6() {
    let p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 });
    let mut m = black_box(m6());
    p.permute_columns(ref m);
    assert!(p.p1 != 0 && m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__len() {
    let p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 });
    let m = black_box(m6());
    assert!(p.len() == 4 && m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__determinant() {
    let p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 6 });
    let m = black_box(m6());
    let d: Fixed = p.determinant();
    assert!(d == int(1) && m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_perm6_permute_rows__append() {
    let mut p = black_box(Perm6 { p1: 6, p2: 3, p3: 3, p4: 5, p5: 5 });
    let m = black_box(m6());
    p.append_permutation(4, 5);
    assert!(p.p5 == 6 && m.m11 != int(0));
}
