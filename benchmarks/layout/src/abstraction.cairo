//! Question 3 (part 1) — call style: trait method vs free function vs operator trait, generic vs
//! concrete. (The generic/concrete x inline-policy matrix lives in the generated
//! `abstraction_gen` / `inline_gen` test modules.)

use crate::fixed::Fixed;
use crate::vec3::Vec3;

/// Concrete free function.
pub fn vec3_add(a: Vec3<Fixed>, b: Vec3<Fixed>) -> Vec3<Fixed> {
    Vec3 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

/// Generic free function.
pub fn vec3_add_generic<T, +Add<T>, +Drop<T>>(a: Vec3<T>, b: Vec3<T>) -> Vec3<T> {
    Vec3 { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
}

pub fn vec3_dot(a: Vec3<Fixed>, b: Vec3<Fixed>) -> Fixed {
    a.x * b.x + a.y * b.y + a.z * b.z
}

pub fn vec3_dot_generic<T, +Add<T>, +Mul<T>, +Drop<T>>(a: Vec3<T>, b: Vec3<T>) -> T {
    a.x * b.x + a.y * b.y + a.z * b.z
}

/// Operator traits, default inlining. NB: corelib binary operators are homogeneous
/// (`Add<T>: fn add(lhs: T, rhs: T) -> T`), so `v + v`, `v - v`, `-v`, `m * m`, `q * q` can be
/// operators but `m * v` and `v * k` cannot.
pub impl Vec3Add<T, +Add<T>, +Drop<T>> of Add<Vec3<T>> {
    fn add(lhs: Vec3<T>, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 { x: lhs.x + rhs.x, y: lhs.y + rhs.y, z: lhs.z + rhs.z }
    }
}

pub impl Vec3Sub<T, +Sub<T>, +Drop<T>> of Sub<Vec3<T>> {
    #[inline(always)]
    fn sub(lhs: Vec3<T>, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 { x: lhs.x - rhs.x, y: lhs.y - rhs.y, z: lhs.z - rhs.z }
    }
}

#[cfg(test)]
mod tests {
    use crate::fixed::{Fixed, input};
    use crate::vec3::{Vec3, Vec3Ops};
    use crate::vec3_variants_gen::Vec3GenAlways;
    use super::{Vec3Add, Vec3Sub, vec3_add, vec3_add_generic, vec3_dot, vec3_dot_generic};

    fn lit(n: i64) -> Fixed {
        crate::fixed::Scalar::from_int(n)
    }

    fn inputs() -> (Vec3<Fixed>, Vec3<Fixed>) {
        (
            Vec3 { x: input(1), y: input(-2), z: input(3) },
            Vec3 { x: input(4), y: input(5), z: input(-6) },
        )
    }

    #[test]
    #[inline(never)]
    fn bench_call_add__baseline() {
        let (a, _) = inputs();
        assert!(a.x == lit(1) && a.y == lit(-2) && a.z == lit(3));
    }

    #[test]
    #[inline(never)]
    fn bench_call_add__trait_method() {
        let (a, b) = inputs();
        let r = Vec3Ops::add(a, b);
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3));
    }

    #[test]
    #[inline(never)]
    fn bench_call_add__free_fn() {
        let (a, b) = inputs();
        let r = vec3_add(a, b);
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3));
    }

    #[test]
    #[inline(never)]
    fn bench_call_add__free_fn_generic() {
        let (a, b) = inputs();
        let r = vec3_add_generic(a, b);
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3));
    }

    #[test]
    #[inline(never)]
    fn bench_call_add__operator() {
        let (a, b) = inputs();
        let r = a + b;
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3));
    }

    /// `Sub` is declared `#[inline(always)]`; a - (-b) is not available so use sub of negated
    /// inputs: a - b' with b' = -b gives the same result as a + b.
    #[test]
    #[inline(never)]
    fn bench_call_add__operator_inline_always() {
        let a = Vec3 { x: input::<Fixed>(1), y: input(-2), z: input(3) };
        let b = Vec3 { x: input::<Fixed>(-4), y: input(-5), z: input(6) };
        let r = a - b;
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3));
    }

    #[test]
    #[inline(never)]
    fn bench_call_add__trait_method_inline_always() {
        let a = Vec3GenAlways { x: input::<Fixed>(1), y: input(-2), z: input(3) };
        let b = Vec3GenAlways { x: input::<Fixed>(4), y: input(5), z: input(-6) };
        let r = a.add(b);
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3));
    }

    #[test]
    #[inline(never)]
    fn bench_call_dot__baseline() {
        let (a, _) = inputs();
        assert!(a.x == lit(1));
    }

    #[test]
    #[inline(never)]
    fn bench_call_dot__trait_method() {
        let (a, b) = inputs();
        assert!(a.dot(b) == lit(-24));
    }

    #[test]
    #[inline(never)]
    fn bench_call_dot__free_fn() {
        let (a, b) = inputs();
        assert!(vec3_dot(a, b) == lit(-24));
    }

    #[test]
    #[inline(never)]
    fn bench_call_dot__free_fn_generic() {
        let (a, b) = inputs();
        assert!(vec3_dot_generic(a, b) == lit(-24));
    }
}
