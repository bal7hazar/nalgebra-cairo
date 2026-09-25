//! WP 8.4-R: the in-place forms of `Quaternion`, `UnitQuaternion`, `UnitComplex`, `Isometry2/3` and
//! `Similarity2/3` (`*=`, `/=`, `conjugate_mut`, `normalize_mut`, `try_inverse_mut`,
//! `inverse_mut`, the scaling mutators), each checked bit for bit against its by-value form, and
//! the `Unit` constructors of `UnitQuaternion`, `UnitComplex` and `Unit<Vector>` that complete
//! upstream's `Unit` (`new_and_get`, `try_new_and_get`, `try_new`, `unwrap`, ...) with their
//! `None` and panic cases.
//!
//! The module is named `quaternion_*` on purpose: the `tests_geometry quaternions` CI shard
//! selects tests by module prefix and would not run a module named otherwise.

use fixed::Fixed;
use nalgebra::base::matrix1::Matrix1;
use nalgebra::base::unit::{Unit, UnitTrait};
use nalgebra::base::vector2::Vector2;
use nalgebra::base::vector3::Vector3;
use nalgebra::base::vector6::Vector6;
use nalgebra::geometry::isometry2::Isometry2Trait;
use nalgebra::geometry::isometry3::Isometry3Trait;
use nalgebra::geometry::quaternion::QuaternionTrait;
use nalgebra::geometry::similarity2::Similarity2Trait;
use nalgebra::geometry::similarity3::Similarity3Trait;
use nalgebra::geometry::unit_complex::UnitComplexTrait;
use nalgebra::geometry::unit_quaternion::UnitQuaternionTrait;
use nalgebra_tests_utils::{fx, int, iso2, iso3, q, qi, r2, sim2, sim3, uc, uq};
use simba::scalar::Real;

const HALF: i64 = 0x80000000;

// --- Quaternion

#[test]
fn test_quaternion_assign_operators_match_operators() {
    let (a, b) = (qi(1, 2, -3, 4), q(0x180000000, -0x7FFFFFFF, 0x12345678, -0x2AAAAAAB));
    let mut x = a;
    x += b;
    assert!(x == a + b);
    let mut x = a;
    x -= b;
    assert!(x == a - b);
    let mut x = a;
    x *= b;
    assert!(x == a * b);
    let k = fx(0x2AAAAAAB);
    let mut x = b;
    x *= k;
    assert!(x == b.scale(k));
    let mut x = b;
    x /= k;
    assert!(x == b.unscale(k));
    // Exact integer case: (1 + 2i - 3j + 4k) / 2.
    let mut x = a;
    x /= int(2);
    assert!(x == q(HALF, 0x100000000, -0x180000000, 0x200000000));
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_quaternion_div_assign_zero_panics() {
    let mut x = qi(1, 2, 3, 4);
    x /= Real::zero();
}

#[test]
fn test_quaternion_conjugate_mut_normalize_mut_try_inverse_mut() {
    let a = q(0x180000000, -0x7FFFFFFF, 0x12345678, -0x2AAAAAAB);
    let mut x = a;
    x.conjugate_mut();
    assert!(x == a.conjugate());
    x.conjugate_mut();
    assert!(x == a);

    let mut x = a;
    let n = x.normalize_mut();
    assert!(x == a.normalize() && n == a.norm());
    // 3-4-0-0 has the exact norm 5.
    let mut x = qi(3, 4, 0, 0);
    assert!(x.normalize_mut() == int(5));

    let mut x = a;
    assert!(x.try_inverse_mut());
    assert!(Some(x) == a.try_inverse());
    // The zero quaternion is not invertible and is left unchanged.
    let mut z = qi(0, 0, 0, 0);
    assert!(!z.try_inverse_mut());
    assert!(z == qi(0, 0, 0, 0));
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_quaternion_normalize_mut_zero_panics() {
    let mut x = qi(0, 0, 0, 0);
    let _ = x.normalize_mut();
}

// --- UnitQuaternion

#[test]
fn test_unit_quaternion_assign_operators_match_operators() {
    let a = uq(0x7DDDDDDD, 0x1AAAAAAA, -0x2BBBBBBB, 0x3CCCCCCC);
    let b = uq(0x6BBBBBBB, -0x4CCCCCCC, 0x2AAAAAAA, 0x1FFFFFFF);
    let mut x = a;
    x *= b;
    assert!(x == a * b);
    let mut x = a;
    x /= b;
    assert!(x == a / b);
    let r = b.to_rotation_matrix();
    let mut x = a;
    x *= r;
    assert!(x == a.mul_rotation(r));
    let mut x = a;
    x /= r;
    assert!(x == a.div_rotation(r));
    let mut x = a;
    x.conjugate_mut();
    assert!(x == a.conjugate() && x == a.inverse());
}

#[test]
fn test_unit_quaternion_new_and_get_try_new_and_get_unwrap() {
    let v = qi(1, 2, -3, 4);
    let (u, n) = UnitQuaternionTrait::<Fixed>::new_and_get(v);
    assert!(u == UnitQuaternionTrait::new_normalize(v) && n == v.norm());
    assert!(u.unwrap() == u.into_inner() && u.unwrap() == u.quaternion);
    assert!(UnitQuaternionTrait::<Fixed>::try_new_and_get(v, int(1)) == Some((u, n)));
    assert!(UnitQuaternionTrait::<Fixed>::try_new_and_get(v, int(10)).is_none());
    // `min_norm` is exclusive: a norm equal to it gives `None`.
    assert!(UnitQuaternionTrait::<Fixed>::try_new_and_get(v, n).is_none());
    assert!(UnitQuaternionTrait::<Fixed>::try_new(v, n).is_none());
    assert!(UnitQuaternionTrait::<Fixed>::try_new_and_get(qi(0, 0, 0, 0), Real::zero()).is_none());
    // Exact: norm 5 along i.
    let (u, n) = UnitQuaternionTrait::<Fixed>::new_and_get(qi(0, 5, 0, 0));
    assert!(u == uq(0, 0x100000000, 0, 0) && n == int(5));
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_unit_quaternion_new_and_get_zero_panics() {
    let _ = UnitQuaternionTrait::<Fixed>::new_and_get(qi(0, 0, 0, 0));
}

// --- UnitComplex

#[test]
fn test_unit_complex_assign_operators_match_operators() {
    let a = uc(0xCCCCCCCD, 0x99999999);
    let b = uc(0x99999999, -0xCCCCCCCD);
    let mut x = a;
    x *= b;
    assert!(x == a * b);
    let mut x = a;
    x /= b;
    assert!(x == a / b);
    let r = r2([[0x99999999, 0x66666666], [-0x66666666, 0x99999999]]);
    let mut x = a;
    x *= r;
    assert!(x == a.mul_rotation(r));
    let mut x = a;
    x /= r;
    assert!(x == a.div_rotation(r));
    let mut x = a;
    x.conjugate_mut();
    assert!(x == a.conjugate() && x == a.inverse());
}

#[test]
fn test_unit_complex_unit_constructors() {
    let v = Vector2 { x: int(3), y: int(4) };
    let (u, n) = UnitComplexTrait::<Fixed>::new_and_get(v);
    assert!(n == int(5));
    assert!(u == UnitComplexTrait::new_normalize(v) && u == UnitComplexTrait::from_complex(v));
    // 3/5 and 4/5 rounded to nearest.
    assert!(u == uc(0x9999999A, 0xCCCCCCCD));
    assert!(UnitComplexTrait::<Fixed>::try_new(v, int(1)) == Some(u));
    assert!(UnitComplexTrait::<Fixed>::try_new_and_get(v, int(1)) == Some((u, n)));
    assert!(UnitComplexTrait::<Fixed>::try_new(v, int(5)).is_none());
    assert!(UnitComplexTrait::<Fixed>::try_new_and_get(v, int(5)).is_none());
    assert!(
        UnitComplexTrait::<Fixed>::try_new_and_get(Vector2 { x: int(0), y: int(0) }, int(0))
            .is_none(),
    );
    let w = UnitComplexTrait::<Fixed>::new_unchecked(Vector2 { x: fx(7), y: fx(-9) });
    assert!(w == uc(7, -9));
    assert!(w.into_inner() == Vector2 { x: fx(7), y: fx(-9) } && w.unwrap() == w.into_inner());
    assert!(w.into_inner() == w.complex());
}

#[test]
#[should_panic(expected: ('Fixed: division by zero',))]
fn test_unit_complex_new_normalize_zero_panics() {
    let _ = UnitComplexTrait::<Fixed>::new_normalize(Vector2 { x: int(0), y: int(0) });
}

// --- Unit<Vector>

#[test]
fn test_unit_vector_unwrap_matches_into_inner() {
    let u: Unit<Vector3<Fixed>> = UnitTrait::new_unchecked(
        Vector3 { x: int(0), y: int(0), z: int(1) },
    );
    assert!(u.unwrap() == u.into_inner() && u.unwrap() == u.value);
    let u1: Unit<Matrix1<Fixed>> = UnitTrait::new_unchecked(Matrix1 { x: int(1) });
    assert!(u1.unwrap() == Matrix1 { x: int(1) });
    let v6 = Vector6 { x: int(1), y: int(0), z: int(0), w: int(0), a: int(0), b: int(0) };
    let u6: Unit<Vector6<Fixed>> = UnitTrait::new_unchecked(v6);
    assert!(u6.unwrap() == v6);
}

// --- Isometry, Similarity

#[test]
fn test_isometry_inverse_mut_matches_inverse() {
    let a = iso2(0x123456789, -0x2BCDEF01, 0xCCCCCCCD, 0x99999999);
    let mut x = a;
    x.inverse_mut();
    assert!(x == a.inverse());
    let b = iso3(
        (0x123456789, -0x2BCDEF01, 0x1FFFFFFF), (0x7DDDDDDD, 0x1AAAAAAA, -0x2BBBBBBB, 0x3CCCCCCC),
    );
    let mut y = b;
    y.inverse_mut();
    assert!(y == b.inverse());
}

#[test]
fn test_similarity_inverse_mut_and_scaling_mutators_match_by_value() {
    let a = sim2(0x123456789, -0x2BCDEF01, 0xCCCCCCCD, 0x99999999, 0x180000000);
    let mut x = a;
    x.inverse_mut();
    assert!(x == a.inverse());
    let mut x = a;
    x.prepend_scaling_mut(fx(0x2AAAAAAB));
    assert!(x == a.prepend_scaling(fx(0x2AAAAAAB)));
    let mut x = a;
    x.append_scaling_mut(fx(0x2AAAAAAB));
    assert!(x == a.append_scaling(fx(0x2AAAAAAB)));

    let b = sim3(
        (0x123456789, -0x2BCDEF01, 0x1FFFFFFF),
        (0x7DDDDDDD, 0x1AAAAAAA, -0x2BBBBBBB, 0x3CCCCCCC),
        0x180000000,
    );
    let mut y = b;
    y.inverse_mut();
    assert!(y == b.inverse());
    let mut y = b;
    y.prepend_scaling_mut(fx(-0x2AAAAAAB));
    assert!(y == b.prepend_scaling(fx(-0x2AAAAAAB)));
    let mut y = b;
    y.append_scaling_mut(fx(-0x2AAAAAAB));
    assert!(y == b.append_scaling(fx(-0x2AAAAAAB)));
    // Exact: scale 2 then 3 gives 6, and the translation is scaled by 3.
    let mut z = sim3((1, 2, 3), (0x100000000, 0, 0, 0), 2 * 0x100000000);
    z.append_scaling_mut(int(3));
    assert!(z.scaling == int(6));
    assert!(z.isometry.translation.vector == Vector3 { x: fx(3), y: fx(6), z: fx(9) });
}

#[test]
#[should_panic(expected: ('nalgebra: zero scale',))]
fn test_similarity3_append_scaling_mut_zero_panics() {
    let mut z = sim3((1, 2, 3), (0x100000000, 0, 0, 0), 0x100000000);
    z.append_scaling_mut(Real::zero());
}

#[test]
#[should_panic(expected: ('nalgebra: zero scale',))]
fn test_similarity2_prepend_scaling_mut_zero_panics() {
    let mut z = sim2(1, 2, 0x100000000, 0, 0x100000000);
    z.prepend_scaling_mut(Real::zero());
}
