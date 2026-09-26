//! `axpy_cs` (`AxpyCs`) on `DVector` and the static column vectors.

use fixed::Fixed;
use nalgebra::sparse::AxpyCs;
use nalgebra::{DVector, DVectorTrait, Matrix1};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{int, v3i};
use crate::helpers::{cs, dvi, ints};

#[test]
fn test_axpy_cs_dvector() {
    let mut y = dvi(array![1, 2, 3, 4].span());
    let x = cs(4, 1, array![(1, 0, 5), (3, 0, -1)].span());
    y.axpy_cs(black_box(int(2)), x, int(3));
    assert!(y.as_slice() == ints(array![3, 16, 9, 10].span()));
}

#[test]
fn test_axpy_cs_beta_zero() {
    // Only the pattern of `x` is written, like upstream.
    let mut y = dvi(array![1, 2, 3, 4].span());
    let x = cs(4, 1, array![(1, 0, 5), (3, 0, -1)].span());
    y.axpy_cs(black_box(int(2)), x, int(0));
    assert!(y.as_slice() == ints(array![1, 10, 3, -2].span()));
}

#[test]
fn test_axpy_cs_static() {
    let mut y = v3i(1, black_box(2), 3);
    y.axpy_cs(int(1), cs(3, 1, array![(0, 0, 4)].span()), int(2));
    assert!(y == v3i(6, 4, 6));
    let mut z = Matrix1 { x: black_box(int(5)) };
    z.axpy_cs(int(3), cs(1, 1, array![(0, 0, 2)].span()), int(0));
    assert!(z == Matrix1 { x: int(6) });
}

#[test]
fn test_axpy_cs_fused() {
    // alpha * x + beta * y floored once: 92681^2 raw is 1.99999.. raw units, twice is 3 raw units.
    let h = Fixed { raw: 92681 };
    let mut y: DVector<Fixed> = black_box(array![h]).into();
    let x: DVector<Fixed> = array![h].into();
    y.axpy_cs(h, x.into(), h);
    assert!(y.as_slice() == array![Fixed { raw: 3 }].span());
}

#[test]
#[should_panic(expected: 'nalgebra: dimension mismatch')]
fn test_axpy_cs_dimension_mismatch() {
    let mut y = v3i(1, black_box(2), 3);
    y.axpy_cs(int(1), cs(2, 1, array![].span()), int(1));
}
