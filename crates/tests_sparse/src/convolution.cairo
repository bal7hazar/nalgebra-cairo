//! `Convolution` (`convolve_full` / `convolve_same` / `convolve_valid`) on `DVector` and the static
//! column vectors, against hand-computed integer results (exact: every sum of integer products
//! is exact in Q32.32).

use fixed::Fixed;
use nalgebra::{Convolution, DVector, DVectorTrait, Matrix1, Vector2, Vector3, Vector6};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{int, v2i, v3i, v6i};
use crate::helpers::{dvi, ints};

#[test]
fn test_convolve_dvector() {
    let x = dvi(array![1, 2, 3, 4].span());
    let k = dvi(array![1, 0, -1].span());
    assert!(x.convolve_full(k).as_slice() == ints(array![1, 2, 2, 2, -3, -4].span()));
    assert!(x.convolve_same(k).as_slice() == ints(array![2, 2, 2, -3].span()));
    assert!(x.convolve_valid(k).as_slice() == ints(array![2, 2].span()));
}

#[test]
fn test_convolve_dvector_longer() {
    let x = dvi(array![3, -1, 2, 5, 0, 1, 4].span());
    let k = dvi(array![2, 1, -3, 1].span());
    assert!(
        x.convolve_full(k).as_slice() == ints(array![6, 1, -6, 18, -2, -11, 14, 1, -11, 4].span()),
    );
    assert!(x.convolve_same(k).as_slice() == ints(array![-6, 18, -2, -11, 14, 1, -11].span()));
    assert!(x.convolve_valid(k).as_slice() == ints(array![18, -2, -11, 14].span()));
}

#[test]
fn test_convolve_static() {
    let x = v3i(1, black_box(2), 3);
    let k = v2i(2, black_box(5));
    assert!(x.convolve_full(k).as_slice() == ints(array![2, 9, 16, 15].span()));
    let same: Vector3<Fixed> = x.convolve_same(k);
    assert!(same == v3i(2, 9, 16));
    assert!(x.convolve_valid(k).as_slice() == ints(array![9, 16].span()));
    // A dynamic kernel on a static vector.
    assert!(
        x.convolve_full(dvi(array![2, 5].span())).as_slice() == ints(array![2, 9, 16, 15].span()),
    );
}

#[test]
fn test_convolve_vector6_full_kernel() {
    let x = v6i(1, 2, 3, 4, 5, black_box(6));
    assert!(
        x
            .convolve_full(x)
            .as_slice() == ints(array![1, 4, 10, 20, 35, 56, 70, 76, 73, 60, 36].span()),
    );
    let same: Vector6<Fixed> = x.convolve_same(x);
    assert!(same == v6i(35, 56, 70, 76, 73, 60));
    assert!(x.convolve_valid(x).as_slice() == ints(array![56].span()));
}

#[test]
fn test_convolve_matrix1() {
    let x = Matrix1 { x: black_box(int(7)) };
    let k = Matrix1 { x: int(3) };
    assert!(x.convolve_full(k).as_slice() == ints(array![21].span()));
    // Upstream's offset of one: the only term falls outside.
    assert!(x.convolve_same(k) == Matrix1 { x: int(0) });
    assert!(x.convolve_valid(k).as_slice() == ints(array![21].span()));
}

#[test]
fn test_convolve_rounding() {
    // One floor per output: raw 92681 squared is 1.99999.. raw units, so h*h + h*h floored once
    // is 3 raw units (two floored products would give 2).
    let h = Fixed { raw: 92681 };
    let x: DVector<Fixed> = black_box(array![h, h]).into();
    let full = x.convolve_full(x);
    assert!(*full.as_slice()[1] == Fixed { raw: 3 });
}

#[test]
#[should_panic(expected: 'nalgebra: convolution kernel')]
fn test_convolve_kernel_too_long() {
    let _ = v2i(1, black_box(2)).convolve_full(v3i(1, 2, 3));
}

#[test]
#[should_panic(expected: 'nalgebra: convolution kernel')]
fn test_convolve_empty_kernel() {
    let k: DVector<Fixed> = black_box(array![]).into();
    let _ = dvi(array![1, 2].span()).convolve_valid(k);
}

#[test]
#[should_panic(expected: 'nalgebra: convolution kernel')]
fn test_convolve_same_kernel_too_long() {
    let _: Vector2<Fixed> = v2i(1, black_box(2)).convolve_same(v3i(1, 2, 3));
}
