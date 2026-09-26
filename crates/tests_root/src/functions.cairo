//! The crate-root functions of upstream's `lib.rs` (`nalgebra::zero` .. `is_convertible`),
//! operands through `black_box`.

use fixed::Fixed;
use nalgebra::root::Ordering;
use nalgebra::{
    Affine2, Affine2Trait, Affine3, Affine3Trait, Matrix2x3, Matrix2x3Trait, Matrix3, Matrix3Trait,
    Matrix4, Point1, Point2, Point3, Point4, Point5, Point6, Projective2, Projective2Trait,
    Projective3, Projective3Trait, Rotation2, Rotation2AngleTrait, Transform2, Transform2Trait,
    Transform3, Transform3Trait, UnitComplex, Vector2, Vector3, Vector3Trait, abs, center, clamp,
    convert, convert_ref, convert_ref_unchecked, convert_unchecked, distance, distance_squared, inf,
    inf_sup, is_convertible, max, min, one, partial_clamp, partial_cmp, partial_ge, partial_gt,
    partial_le, partial_lt, partial_max, partial_min, partial_sort2, sup, try_convert,
    try_convert_ref, wrap, zero,
};
use nalgebra_testing::black_box;
use nalgebra_tests_utils::{ONE_RAW, fx, int, m3i, m4i};

/// The integer `v` as a `Fixed`, through `black_box`.
fn f(v: i64) -> Fixed {
    black_box(int(v))
}

fn v2(x: i64, y: i64) -> Vector2<Fixed> {
    black_box(Vector2 { x: int(x), y: int(y) })
}

fn v3(x: i64, y: i64, z: i64) -> Vector3<Fixed> {
    black_box(Vector3 { x: int(x), y: int(y), z: int(z) })
}

/// A homogeneous 3x3 matrix whose last row is NOT `(0, 0, 1)`: not an `Affine2`.
fn not_affine3() -> Matrix3<Fixed> {
    black_box(m3i([[1, 2, 3], [4, 5, 6], [1, 0, 1]]))
}

fn affine3() -> Matrix3<Fixed> {
    black_box(m3i([[1, 2, 3], [0, 1, 6], [0, 0, 1]]))
}

fn not_affine4() -> Matrix4<Fixed> {
    black_box(m4i([[1, 2, 3, 4], [5, 6, 7, 8], [9, 1, 2, 3], [1, 0, 0, 1]]))
}

// --- identities --------------------------------------------------------------------------------

#[test]
fn test_zero_scalar() {
    let z: Fixed = zero();
    assert_eq!(z, int(0));
    let n: u32 = zero();
    assert_eq!(n, 0);
}

#[test]
fn test_one_scalar_and_square() {
    let o: Fixed = one();
    assert_eq!(o, int(1));
    let m: Matrix3<Fixed> = one();
    assert_eq!(m, Matrix3Trait::identity());
}

// --- orderings ----------------------------------------------------------------------------------

#[test]
fn test_wrap_inside_and_bounds() {
    assert_eq!(wrap(f(5), f(0), f(10)), int(5));
    assert_eq!(wrap(f(0), f(0), f(10)), int(0));
    assert_eq!(wrap(f(10), f(0), f(10)), int(10));
}

#[test]
fn test_wrap_below() {
    assert_eq!(wrap(f(-3), f(0), f(10)), int(7));
    // Three additions of the width.
    assert_eq!(wrap(f(-23), f(0), f(10)), int(7));
    // Ends in [min, max): -10 + 10 is min.
    assert_eq!(wrap(f(-10), f(0), f(10)), int(0));
}

#[test]
fn test_wrap_above() {
    assert_eq!(wrap(f(13), f(0), f(10)), int(3));
    assert_eq!(wrap(f(35), f(0), f(10)), int(5));
    // Ends in (min, max]: 20 - 10 is max.
    assert_eq!(wrap(f(20), f(0), f(10)), int(10));
}

#[test]
fn test_wrap_integers() {
    assert_eq!(wrap(black_box(-1_i32), 0, 360), 359);
    assert_eq!(wrap(black_box(725_i32), 0, 360), 5);
}

#[test]
#[should_panic(expected: 'nalgebra: invalid wrap bounds')]
fn test_wrap_empty_range() {
    wrap(f(1), f(5), f(5));
}

#[test]
fn test_clamp_scalar() {
    assert_eq!(clamp(f(5), f(0), f(10)), int(5));
    assert_eq!(clamp(f(-5), f(0), f(10)), int(0));
    assert_eq!(clamp(f(15), f(0), f(10)), int(10));
    assert_eq!(clamp(f(0), f(0), f(10)), int(0));
    assert_eq!(clamp(f(10), f(0), f(10)), int(10));
}

#[test]
fn test_clamp_partial_order() {
    // Upstream's tests on the component-wise partial order: (1, 5) > (0, 0) but not < (3, 3).
    assert_eq!(clamp(v2(1, 5), v2(0, 0), v2(3, 3)), v2(3, 3));
    assert_eq!(clamp(v2(1, 2), v2(0, 0), v2(3, 3)), v2(1, 2));
    // (-1, 2) is not > (0, 0).
    assert_eq!(clamp(v2(-1, 2), v2(0, 0), v2(3, 3)), v2(0, 0));
}

#[test]
fn test_min_max() {
    assert_eq!(min(f(2), f(3)), int(2));
    assert_eq!(min(f(3), f(2)), int(2));
    assert_eq!(max(f(2), f(3)), int(3));
    assert_eq!(max(f(3), f(2)), int(3));
    assert_eq!(max(black_box(7_u32), 7), 7);
}

#[test]
fn test_abs() {
    assert_eq!(abs(f(-4)), int(4));
    assert_eq!(abs(f(4)), int(4));
    assert_eq!(abs(black_box(fx(-1))), fx(1));
}

#[test]
fn test_inf_sup() {
    let a: Matrix2x3<Fixed> = black_box(Matrix2x3Trait::new(f(1), f(5), f(-2), f(4), f(0), f(9)));
    let b: Matrix2x3<Fixed> = black_box(Matrix2x3Trait::new(f(3), f(2), f(-7), f(4), f(1), f(8)));
    let lo = Matrix2x3Trait::new(int(1), int(2), int(-7), int(4), int(0), int(8));
    let hi = Matrix2x3Trait::new(int(3), int(5), int(-2), int(4), int(1), int(9));
    assert_eq!(inf(a, b), lo);
    assert_eq!(sup(a, b), hi);
    assert_eq!(inf_sup(a, b), (lo, hi));
    // The same as the methods.
    let u = v3(1, -2, 3);
    let w = v3(0, 4, 3);
    assert_eq!(inf(u, w), u.inf(w));
    assert_eq!(sup(u, w), u.sup(w));
}

#[test]
fn test_partial_cmp_scalar() {
    assert_eq!(partial_cmp(f(1), f(2)), Option::Some(Ordering::Less));
    assert_eq!(partial_cmp(f(2), f(2)), Option::Some(Ordering::Equal));
    assert_eq!(partial_cmp(f(3), f(2)), Option::Some(Ordering::Greater));
}

#[test]
fn test_partial_cmp_matrices() {
    // Upstream's `PartialOrd for Matrix`: Less when every component is <= and one is <.
    assert_eq!(partial_cmp(v2(1, 2), v2(1, 3)), Option::Some(Ordering::Less));
    assert_eq!(partial_cmp(v2(1, 3), v2(1, 2)), Option::Some(Ordering::Greater));
    assert_eq!(partial_cmp(v2(1, 2), v2(1, 2)), Option::Some(Ordering::Equal));
    assert_eq!(partial_cmp(v2(1, 2), v2(2, 1)), Option::None);
}

#[test]
fn test_partial_comparisons() {
    assert!(partial_lt(f(1), f(2)));
    assert!(!partial_lt(f(2), f(2)));
    assert!(partial_le(f(2), f(2)));
    assert!(partial_gt(f(3), f(2)));
    assert!(partial_ge(f(2), f(2)));
    assert!(!partial_ge(f(1), f(2)));
    // Not comparable: every comparison is false.
    let (a, b) = (v2(1, 2), v2(2, 1));
    assert!(!partial_lt(a, b) && !partial_le(a, b) && !partial_gt(a, b) && !partial_ge(a, b));
}

#[test]
fn test_partial_min_max() {
    assert_eq!(partial_min(f(1), f(2)), Option::Some(int(1)));
    assert_eq!(partial_min(f(3), f(2)), Option::Some(int(2)));
    assert_eq!(partial_max(f(1), f(2)), Option::Some(int(2)));
    assert_eq!(partial_max(f(3), f(2)), Option::Some(int(3)));
    assert_eq!(partial_min(v2(1, 2), v2(1, 3)), Option::Some(v2(1, 2)));
    assert_eq!(partial_max(v2(1, 2), v2(1, 3)), Option::Some(v2(1, 3)));
    assert_eq!(partial_min(v2(1, 2), v2(2, 1)), Option::None);
    assert_eq!(partial_max(v2(1, 2), v2(2, 1)), Option::None);
}

#[test]
fn test_partial_clamp() {
    assert_eq!(partial_clamp(f(5), f(0), f(10)), Option::Some(int(5)));
    assert_eq!(partial_clamp(f(-5), f(0), f(10)), Option::Some(int(0)));
    assert_eq!(partial_clamp(f(15), f(0), f(10)), Option::Some(int(10)));
    assert_eq!(partial_clamp(v2(-1, -1), v2(0, 0), v2(5, 5)), Option::Some(v2(0, 0)));
    assert_eq!(partial_clamp(v2(6, 5), v2(0, 0), v2(5, 5)), Option::Some(v2(5, 5)));
    assert_eq!(partial_clamp(v2(1, 5), v2(0, 0), v2(5, 5)), Option::Some(v2(1, 5)));
    // Not comparable with `min`, then with `max`.
    assert_eq!(partial_clamp(v2(-1, 2), v2(0, 0), v2(5, 5)), Option::None);
    assert_eq!(partial_clamp(v2(6, 2), v2(0, 0), v2(5, 5)), Option::None);
}

#[test]
fn test_partial_sort2() {
    assert_eq!(partial_sort2(f(3), f(1)), Option::Some((int(1), int(3))));
    assert_eq!(partial_sort2(f(1), f(3)), Option::Some((int(1), int(3))));
    assert_eq!(partial_sort2(f(2), f(2)), Option::Some((int(2), int(2))));
    assert_eq!(partial_sort2(v2(1, 2), v2(2, 1)), Option::None);
}

// --- points -------------------------------------------------------------------------------------

#[test]
fn test_point1_metric() {
    let p = black_box(Point1 { x: int(1) });
    let q = black_box(Point1 { x: int(4) });
    assert_eq!(center(p, q), Point1 { x: fx(5 * ONE_RAW / 2) });
    assert_eq!(distance(p, q), int(3));
    assert_eq!(distance(q, p), int(3));
    assert_eq!(distance_squared(p, q), int(9));
    // The midpoint is floored: 0 and 1 ulp give 0.
    assert_eq!(center(black_box(Point1 { x: fx(0) }), Point1 { x: fx(1) }), Point1 { x: fx(0) });
}

#[test]
fn test_point2_metric() {
    let p = black_box(Point2 { x: int(0), y: int(0) });
    let q = black_box(Point2 { x: int(3), y: int(4) });
    assert_eq!(center(p, q), Point2 { x: fx(3 * ONE_RAW / 2), y: int(2) });
    assert_eq!(distance(p, q), int(5));
    assert_eq!(distance_squared(p, q), int(25));
}

#[test]
fn test_point3_metric() {
    let p = black_box(Point3 { x: int(1), y: int(2), z: int(3) });
    let q = black_box(Point3 { x: int(3), y: int(4), z: int(4) });
    assert_eq!(center(p, q), Point3 { x: int(2), y: int(3), z: fx(7 * ONE_RAW / 2) });
    assert_eq!(distance(p, q), int(3));
    assert_eq!(distance_squared(p, q), int(9));
}

#[test]
fn test_point4_metric() {
    let p = black_box(Point4 { x: int(0), y: int(0), z: int(0), w: int(0) });
    let q = black_box(Point4 { x: int(1), y: int(-1), z: int(1), w: int(-1) });
    assert_eq!(
        center(p, q),
        Point4 { x: fx(ONE_RAW / 2), y: fx(-ONE_RAW / 2), z: fx(ONE_RAW / 2), w: fx(-ONE_RAW / 2) },
    );
    assert_eq!(distance(p, q), int(2));
    assert_eq!(distance_squared(p, q), int(4));
}

#[test]
fn test_point5_metric() {
    let p = black_box(Point5 { x: int(1), y: int(1), z: int(1), w: int(1), a: int(1) });
    let q = black_box(Point5 { x: int(3), y: int(2), z: int(-1), w: int(1), a: int(1) });
    assert_eq!(
        center(p, q), Point5 { x: int(2), y: fx(3 * ONE_RAW / 2), z: int(0), w: int(1), a: int(1) },
    );
    assert_eq!(distance(p, q), int(3));
    assert_eq!(distance_squared(p, q), int(9));
}

#[test]
fn test_point6_metric() {
    let p = black_box(Point6 { x: int(0), y: int(0), z: int(0), w: int(0), a: int(0), b: int(0) });
    let q = black_box(Point6 { x: int(1), y: int(2), z: int(2), w: int(4), a: int(0), b: int(0) });
    assert_eq!(
        center(p, q),
        Point6 { x: fx(ONE_RAW / 2), y: int(1), z: int(1), w: int(2), a: int(0), b: int(0) },
    );
    assert_eq!(distance(p, q), int(5));
    assert_eq!(distance_squared(p, q), int(25));
}

#[test]
#[should_panic]
fn test_distance_squared_overflow() {
    let p = black_box(Point2 { x: int(0), y: int(0) });
    let q = black_box(Point2 { x: int(100000), y: int(0) });
    distance_squared(p, q);
}

// --- conversions --------------------------------------------------------------------------------

#[test]
fn test_convert_into() {
    let r: Rotation2<Fixed> = black_box(Rotation2AngleTrait::new(fx(ONE_RAW / 3)));
    let c: UnitComplex<Fixed> = convert(r);
    assert_eq!(c, r.into());
    let c2: UnitComplex<Fixed> = convert_ref(@r);
    assert_eq!(c2, c);
    let a: Affine2<Fixed> = convert(r);
    assert_eq!(a, r.into());
}

#[test]
fn test_try_convert() {
    let good: Option<Affine2<Fixed>> = try_convert(affine3());
    assert_eq!(good, Option::Some(Affine2Trait::from_matrix_unchecked(affine3())));
    let bad: Option<Affine2<Fixed>> = try_convert(not_affine3());
    assert!(bad.is_none());
    let m = affine3();
    let by_ref: Option<Affine2<Fixed>> = try_convert_ref(@m);
    assert_eq!(by_ref, good);
    // An `Into` pair always converts (corelib's `TryInto` from `Into`).
    let r: Rotation2<Fixed> = black_box(Rotation2AngleTrait::new(fx(ONE_RAW / 3)));
    let c: Option<UnitComplex<Fixed>> = try_convert(r);
    assert_eq!(c, Option::Some(r.into()));
}

#[test]
fn test_is_convertible() {
    assert!(is_convertible::<_, Affine2<Fixed>>(@affine3()));
    assert!(!is_convertible::<_, Affine2<Fixed>>(@not_affine3()));
    assert!(is_convertible::<_, Projective2<Fixed>>(@affine3()));
    let r: Rotation2<Fixed> = black_box(Rotation2AngleTrait::new(fx(ONE_RAW / 3)));
    assert!(is_convertible::<_, UnitComplex<Fixed>>(@r));
}

#[test]
fn test_convert_unchecked_2d() {
    // No check: a matrix that is not affine becomes an `Affine2` all the same.
    let m = not_affine3();
    let a: Affine2<Fixed> = convert_unchecked(m);
    assert_eq!(a.into_inner(), m);
    let t: Transform2<Fixed> = convert_unchecked(m);
    assert_eq!(t.into_inner(), m);
    let p: Projective2<Fixed> = convert_unchecked(m);
    assert_eq!(p.into_inner(), m);
    let a2: Affine2<Fixed> = convert_unchecked(t);
    assert_eq!(a2, a);
    let a3: Affine2<Fixed> = convert_unchecked(p);
    assert_eq!(a3, a);
    let p2: Projective2<Fixed> = convert_unchecked(t);
    assert_eq!(p2, p);
    let by_ref: Affine2<Fixed> = convert_ref_unchecked(@m);
    assert_eq!(by_ref, a);
}

#[test]
fn test_convert_unchecked_3d() {
    let m = not_affine4();
    let a: Affine3<Fixed> = convert_unchecked(m);
    assert_eq!(a.into_inner(), m);
    let t: Transform3<Fixed> = convert_unchecked(m);
    assert_eq!(t.into_inner(), m);
    let p: Projective3<Fixed> = convert_unchecked(m);
    assert_eq!(p.into_inner(), m);
    let a2: Affine3<Fixed> = convert_unchecked(t);
    assert_eq!(a2, a);
    let a3: Affine3<Fixed> = convert_unchecked(p);
    assert_eq!(a3, a);
    let p2: Projective3<Fixed> = convert_unchecked(t);
    assert_eq!(p2, p);
}
