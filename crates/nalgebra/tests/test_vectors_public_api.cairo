//! The vector API as a downstream crate sees it: only `nalgebra::` and `simba::` paths.

use fixed::Fixed;
use nalgebra::{Vector2, Vector3, Vector4};

fn f(v: i32) -> Fixed {
    simba::scalar::Real::from_int(v)
}

/// Operators and conversions resolve with the types alone in scope: their impls live in the module
/// of each vector type.
#[test]
fn test_api_operators_and_conversions_need_no_import() {
    let a = Vector3 { x: f(1), y: f(-2), z: f(3) };
    let b: Vector3<Fixed> = (f(4), f(5), f(-6)).into();
    assert!(a + b == Vector3 { x: f(5), y: f(3), z: f(-3) });
    assert!(a - b == Vector3 { x: f(-3), y: f(-7), z: f(9) });
    assert!(-a == Vector3 { x: f(-1), y: f(2), z: f(-3) });
    let mut c = a;
    c += b;
    c -= a;
    assert!(c == b);
    c *= f(2);
    assert!(c == Vector3 { x: f(8), y: f(10), z: f(-12) });
    c /= f(-2);
    assert!(c == -b);
    let [x, y]: [Fixed; 2] = Vector2 { x: f(7), y: f(8) }.into();
    assert!(x == f(7) && y == f(8));
    let d: Vector4<Fixed> = [f(1), f(2), f(3), f(4)].into();
    let (_, _, _, w): (Fixed, Fixed, Fixed, Fixed) = d.into();
    assert!(w == f(4));
    assert!(d == Vector4 { x: f(1), y: f(2), z: f(3), w: f(4) });
    assert!(Default::default() == Vector2 { x: f(0), y: f(0) });
}

mod with_traits {
    use nalgebra::Vector3;
    use nalgebra::base::vector2::Vector2Trait;
    use nalgebra::base::vector3::Vector3Trait;
    use nalgebra::base::vector4::Vector4Trait;
    use simba::prelude::*;
    use super::f;

    /// The three vector traits and the scalar trait share method names (`abs`, `min`, `max`,
    /// `abs_diff_eq`, `lerp`...): everything resolves on the type of the receiver.
    #[test]
    fn test_api_traits_coexist() {
        let v2 = Vector2Trait::new(f(3), f(-4));
        let v3 = v2.push(f(12));
        let v4 = v3.push(f(0));
        assert!(v2.norm() == f(5) && v3.norm() == f(13) && v4.norm() == f(13));
        assert!(v2.abs().min() == f(3) && v3.min() == f(-4) && v4.amax() == f(12));
        assert!(f(-4).abs().min(f(3)) == f(3));
        assert!(v4.xyz() == v3 && v4.xy() == v2 && v3.xy() == v2);
        assert!(v3.abs_diff_eq(v3, 0) && f(1).abs_diff_eq(f(1), 0));
        assert!(v2.perp(Vector2Trait::y()) == f(3));
        assert!(v3.cross(Vector3Trait::z()) == Vector3 { x: f(-4), y: f(-3), z: f(0) });
        assert!(v4.dot(Vector4Trait::repeat(f(1))) == f(11));
        assert!(Vector2Trait::<Fixed>::zeros().is_zero());
    }

    /// A rigid-body style step: reflect a velocity on a plane, cap it, integrate a position.
    #[test]
    fn test_api_physics_expression() {
        let normal = Vector3 { x: f(0), y: f(5), z: f(0) }.normalize();
        let velocity = Vector3 { x: f(3), y: f(-4), z: f(12) };
        let reflected = velocity - normal.scale(velocity.dot(normal) * Real::TWO);
        assert!(reflected == Vector3 { x: f(3), y: f(4), z: f(12) });
        let capped = reflected.cap_magnitude(Real::from_ratio(13, 2));
        assert!(capped == reflected.scale(Real::HALF));
        let position = Vector3Trait::zeros() + capped.scale(Real::from_ratio(1, 4));
        assert!(
            position == Vector3 {
                x: Real::from_ratio(3, 8), y: Real::HALF, z: Real::from_ratio(3, 2),
            },
        );
        assert!(position.try_normalize(f(100)).is_none());
        let (u, w) = normal.orthonormal_basis();
        assert!(u.cross(w) == normal);
    }

    /// Downstream code can stay generic over the scalar with the documented bounds.
    #[generate_trait]
    impl ReflectImpl<
        T,
        +Real<T>,
        +Add<T>,
        +Sub<T>,
        +Mul<T>,
        +Div<T>,
        +Neg<T>,
        +PartialEq<T>,
        +PartialOrd<T>,
        +Copy<T>,
        +Drop<T>,
    > of ReflectTrait<T> {
        fn reflect(self: Vector3<T>, normal: Vector3<T>) -> Vector3<T> {
            self - normal.scale(self.dot(normal) * Real::TWO)
        }
    }

    #[test]
    fn test_api_generic_downstream_code() {
        let v = Vector3 { x: f(3), y: f(-4), z: f(12) };
        assert!(v.reflect(Vector3Trait::y()) == Vector3 { x: f(3), y: f(4), z: f(12) });
    }

    #[test]
    #[should_panic(expected: 'Fixed: division by zero')]
    fn test_api_stable_panic_message() {
        let _ = nalgebra_testing::black_box(Vector4Trait::<Fixed>::zeros()).normalize();
    }
}
