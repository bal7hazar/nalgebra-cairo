//! Tests and gas benchmarks of upstream's `#[doc(hidden)]` building blocks in
//! `nalgebra::linalg`: `gauss_step` / `gauss_step_swap` (they replay `Lu3::new` / `Lu6::new` bit
//! for bit when driven with the same pivots), `try_invert_to`, and `reflection_axis_mut`.

use fixed::Fixed;
use nalgebra::linalg::{gauss_step, gauss_step_swap, reflection_axis_mut, try_invert_to};
use nalgebra::{
    Lu3Trait, Lu6Trait, Matrix1, Matrix1Trait, Matrix3, Matrix3Trait, Matrix6, Vector2,
    Vector2Trait, Vector6, Vector6Trait,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{fx, int, m3, m6, ulp_diff};
use simba::scalar::Real;

fn a3() -> Matrix3<Fixed> {
    black_box(
        m3(
            [
                [-2414118097, 417657389, 4595817171], [6298117444, -2725477524, -1338161353],
                [2614037897, 2690897069, 3051067169],
            ],
        ),
    )
}

fn a6() -> Matrix6<Fixed> {
    black_box(
        m6(
            [
                [-2527254097, 4325213708, -1213189640, 4008510819, 2405075077, 466428599],
                [-700750028, -3536112306, 1731465811, 244560054, 2632174365, 1874080084],
                [-4134312449, -1036601848, -444052936, -1797900266, -2730917964, 404854629],
                [-617580144, 580405440, -3475076872, -924247759, -985644408, 938584824],
                [-1076219818, -1905997652, -1595683778, 2067806730, -3245944919, 3717734456],
                [-308280913, -120179493, 2005545225, -1090342151, 932840990, 2332476429],
            ],
        ),
    )
}

/// `Lu3::new` = the pivoted elimination by `gauss_step` / `gauss_step_swap` (0-based steps), the
/// columns before the step swapped by the caller like upstream's `LU::new`: bit for bit.
#[test]
fn test_gauss_steps_replay_lu3() {
    let f = Lu3Trait::new(a3());
    // `a3` pivots row 2 at step 1 (`p1 = 2`) and row 3 at step 2 (`p2 = 3`).
    assert!(f.p.p1 == 2 && f.p.p2 == 3);
    let mut m = a3();
    gauss_step_swap(ref m, m.m21, 0, 1);
    // Step 2 swaps rows 2 and 3 in the columns before it (the caller's part), then eliminates.
    let t = m.m21;
    m.m21 = m.m31;
    m.m31 = t;
    gauss_step_swap(ref m, m.m32, 1, 2);
    gauss_step(ref m, m.m33, 2);
    assert!(m == f.lu);
}

/// The same on the 6x6 (`p = (3, 3, 4, 5, 5)`).
#[test]
fn test_gauss_steps_replay_lu6() {
    let f = Lu6Trait::new(a6());
    assert!(f.p.p1 == 3 && f.p.p2 == 3 && f.p.p3 == 4 && f.p.p4 == 5 && f.p.p5 == 5);
    let mut m = a6();
    gauss_step_swap(ref m, m.m31, 0, 2);
    // Step 2 has `p2 = 3`: rows 2 and 3 in column 1 first.
    let t = m.m21;
    m.m21 = m.m31;
    m.m31 = t;
    gauss_step_swap(ref m, m.m32, 1, 2);
    // Step 3 has `p3 = 4`: rows 3 and 4 in columns 1..2 first.
    let (t1, t2) = (m.m31, m.m32);
    m.m31 = m.m41;
    m.m32 = m.m42;
    m.m41 = t1;
    m.m42 = t2;
    gauss_step_swap(ref m, m.m43, 2, 3);
    let (t1, t2, t3) = (m.m41, m.m42, m.m43);
    m.m41 = m.m51;
    m.m42 = m.m52;
    m.m43 = m.m53;
    m.m51 = t1;
    m.m52 = t2;
    m.m53 = t3;
    gauss_step_swap(ref m, m.m54, 3, 4);
    gauss_step(ref m, m.m55, 4);
    gauss_step(ref m, m.m66, 5);
    assert!(m == f.lu);
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_gauss_step_out_of_bounds_panics() {
    let mut m = a3();
    gauss_step(ref m, int(1), 3);
}

#[test]
#[should_panic(expected: 'nalgebra: index out of bounds')]
fn test_gauss_step_swap_pivot_not_below_panics() {
    let mut m = a3();
    gauss_step_swap(ref m, int(1), 1, 1);
}

/// A 1x1 has no elimination to do; its swap has no valid pivot.
#[test]
fn test_gauss_step_matrix1() {
    let mut m: Matrix1<Fixed> = Matrix1Trait::new(int(3));
    gauss_step(ref m, int(3), 0);
    assert!(m == Matrix1Trait::new(int(3)));
}

/// `try_invert_to` writes the LU inverse (`Lu3::try_inverse`) and reports singularity.
#[test]
fn test_try_invert_to() {
    let mut out: Matrix3<Fixed> = Matrix3Trait::identity();
    assert!(try_invert_to(a3(), ref out));
    assert!(Option::Some(out) == Lu3Trait::new(a3()).try_inverse());
    let z = Matrix3Trait::new(
        int(1), int(2), int(3), int(2), int(4), int(6), int(0), int(1), int(1),
    );
    let before = out;
    assert!(!try_invert_to(z, ref out));
    assert!(out == before);
}

/// `(3, 4)`: `|v| = 5`, the axis is `(3 + 5, 4) / |(8, 4)|` and `r = -5`; reflecting `v` about
/// it gives `(r, 0)` within the rounding.
#[test]
fn test_reflection_axis_vector2() {
    let mut v: Vector2<Fixed> = black_box(Vector2Trait::new(int(3), int(4)));
    let (r, ok) = reflection_axis_mut(ref v);
    assert!(ok && r == int(-5));
    let n = Real::norm2(int(8), int(4));
    assert!(v == Vector2Trait::new(Real::div(int(8), n), Real::div(int(4), n)));
    // `v0 - 2 (u . v0) u` = `(-5, 0)`.
    let d = Real::sum_prod2(v.x, int(3), v.y, int(4));
    let rx = int(3) - (d + d) * v.x;
    let ry = int(4) - (d + d) * v.y;
    assert!(ulp_diff(rx, int(-5)) <= 8 && ulp_diff(ry, int(0)) <= 8);
    // Negative leading component: `r = +|v|`.
    let mut w: Vector2<Fixed> = black_box(Vector2Trait::new(int(-3), int(4)));
    let (r, ok) = reflection_axis_mut(ref w);
    assert!(ok && r == int(5));
}

/// A zero column is left as it is: `(0, false)`.
#[test]
fn test_reflection_axis_zero() {
    let mut v: Vector6<Fixed> = black_box(
        Vector6Trait::new(int(0), int(0), int(0), int(0), int(0), int(0)),
    );
    let (r, ok) = reflection_axis_mut(ref v);
    assert!(!ok && r == int(0));
    assert!(v == Vector6Trait::new(int(0), int(0), int(0), int(0), int(0), int(0)));
}

/// A real-valued 6-vector: the axis is a unit vector within a few ulp and `|r| = |v|`.
#[test]
fn test_reflection_axis_vector6() {
    let v0: Vector6<Fixed> = black_box(
        Vector6Trait::new(
            fx(-2527254097),
            fx(4325213708),
            fx(-1213189640),
            fx(4008510819),
            fx(2405075077),
            fx(466428599),
        ),
    );
    let mut v = v0;
    let (r, ok) = reflection_axis_mut(ref v);
    assert!(ok && r == v0.norm());
    assert!(ulp_diff(v.norm(), int(1)) <= 8);
}

// --- benchmarks ---------------------------------------------------------------------------------

#[test]
#[inline(never)]
fn bench_gauss_step__baseline() {
    let m = a6();
    assert!(m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_gauss_step__matrix6_step0() {
    let mut m = a6();
    gauss_step(ref m, m.m11, 0);
    assert!(m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_gauss_step__matrix6_swap0() {
    let mut m = a6();
    gauss_step_swap(ref m, m.m31, 0, 2);
    assert!(m.m11 != int(0));
}

#[test]
#[inline(never)]
fn bench_reflection_axis__baseline() {
    let v = black_box(
        Vector6Trait::new(
            fx(-2527254097),
            fx(4325213708),
            fx(-1213189640),
            fx(4008510819),
            fx(2405075077),
            fx(466428599),
        ),
    );
    assert!(v.x != int(0));
}

#[test]
#[inline(never)]
fn bench_reflection_axis__vector6() {
    let mut v = black_box(
        Vector6Trait::new(
            fx(-2527254097),
            fx(4325213708),
            fx(-1213189640),
            fx(4008510819),
            fx(2405075077),
            fx(466428599),
        ),
    );
    let (_, ok) = reflection_axis_mut(ref v);
    assert!(ok && v.x != int(0));
}
