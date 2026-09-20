//! Question 1 (continued) — Vec2 / Vec4 scaling of the static vs dynamic gap, and the cost of the
//! different ways to read one component.

use crate::vec3::{DVecLoop, Vec3Ops};

#[derive(Copy, Drop)]
pub struct Vec2<T> {
    pub x: T,
    pub y: T,
}

#[derive(Copy, Drop)]
pub struct Vec4<T> {
    pub x: T,
    pub y: T,
    pub z: T,
    pub w: T,
}

#[derive(Copy, Drop)]
pub struct Vec4A<T> {
    pub data: [T; 4],
}

#[generate_trait]
pub impl Vec2Impl<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>> of Vec2Trait<T> {
    fn add(self: Vec2<T>, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 { x: self.x + rhs.x, y: self.y + rhs.y }
    }
    fn dot(self: Vec2<T>, rhs: Vec2<T>) -> T {
        self.x * rhs.x + self.y * rhs.y
    }
}

#[generate_trait]
pub impl Vec4Impl<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>> of Vec4Trait<T> {
    fn add(self: Vec4<T>, rhs: Vec4<T>) -> Vec4<T> {
        Vec4 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z, w: self.w + rhs.w }
    }
    fn dot(self: Vec4<T>, rhs: Vec4<T>) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
}

#[generate_trait]
pub impl Vec4AImpl<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>> of Vec4ATrait<T> {
    fn add(self: Vec4A<T>, rhs: Vec4A<T>) -> Vec4A<T> {
        let [ax, ay, az, aw] = self.data;
        let [bx, by, bz, bw] = rhs.data;
        Vec4A { data: [ax + bx, ay + by, az + bz, aw + bw] }
    }
    fn dot(self: Vec4A<T>, rhs: Vec4A<T>) -> T {
        let [ax, ay, az, aw] = self.data;
        let [bx, by, bz, bw] = rhs.data;
        ax * bx + ay * by + az * bz + aw * bw
    }
}

/// Dynamic vector of any length: the loop-based `add` / `dot` of `vec3::SpanLoopVec3`.
pub fn dvec<T, +Drop<T>>(data: Array<T>) -> DVecLoop<T> {
    DVecLoop { data: data.span() }
}

#[cfg(test)]
mod tests {
    use crate::fixed::{Fixed, Scalar, input};
    use crate::vec3::{DVecLoop, Vec3, Vec3A, Vec3T};
    use super::*;

    fn lit<T, +Scalar<T>>(n: i64) -> T {
        Scalar::<T>::from_int(n)
    }

    // ------------------------------------------------------------------ Vec2, Q32.32 and felt
    fn in2<T, +Scalar<T>, +Drop<T>>() -> (T, T, T, T) {
        (input(1), input(-2), input(4), input(5))
    }

    #[test]
    #[inline(never)]
    fn bench_vec2_add_fixed__baseline() {
        let (ax, ay, _, _) = in2::<Fixed>();
        assert!(ax == lit(1) && ay == lit(-2));
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_add_fixed__struct() {
        let (ax, ay, bx, by) = in2::<Fixed>();
        let r = Vec2 { x: ax, y: ay }.add(Vec2 { x: bx, y: by });
        assert!(r.x == lit(5) && r.y == lit(3));
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_add_fixed__span_loop() {
        let (ax, ay, bx, by) = in2::<Fixed>();
        let r = dvec(array![ax, ay]).add(dvec(array![bx, by])).data;
        assert!(*r[0] == lit(5) && *r[1] == lit(3));
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_dot_fixed__baseline() {
        let (ax, _, _, _) = in2::<Fixed>();
        assert!(ax == lit(1));
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_dot_fixed__struct() {
        let (ax, ay, bx, by) = in2::<Fixed>();
        assert!(Vec2 { x: ax, y: ay }.dot(Vec2 { x: bx, y: by }) == lit(-6));
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_dot_fixed__span_loop() {
        let (ax, ay, bx, by) = in2::<Fixed>();
        assert!(dvec(array![ax, ay]).dot(dvec(array![bx, by])) == lit(-6));
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_dot_felt__baseline() {
        let (ax, _, _, _) = in2::<felt252>();
        assert!(ax == 1);
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_dot_felt__struct() {
        let (ax, ay, bx, by) = in2::<felt252>();
        assert!(Vec2 { x: ax, y: ay }.dot(Vec2 { x: bx, y: by }) == -6);
    }
    #[test]
    #[inline(never)]
    fn bench_vec2_dot_felt__span_loop() {
        let (ax, ay, bx, by) = in2::<felt252>();
        assert!(dvec(array![ax, ay]).dot(dvec(array![bx, by])) == -6);
    }

    // ------------------------------------------------------------------ Vec4, Q32.32 and felt
    fn in4<T, +Scalar<T>, +Drop<T>>() -> (T, T, T, T, T, T, T, T) {
        (input(1), input(-2), input(3), input(-4), input(4), input(5), input(-6), input(7))
    }

    #[test]
    #[inline(never)]
    fn bench_vec4_add_fixed__baseline() {
        let (ax, ay, az, aw, _, _, _, _) = in4::<Fixed>();
        assert!(ax == lit(1) && ay == lit(-2) && az == lit(3) && aw == lit(-4));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_add_fixed__struct() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<Fixed>();
        let r = Vec4 { x: ax, y: ay, z: az, w: aw }.add(Vec4 { x: bx, y: by, z: bz, w: bw });
        assert!(r.x == lit(5) && r.y == lit(3) && r.z == lit(-3) && r.w == lit(3));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_add_fixed__array() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<Fixed>();
        let [x, y, z, w] = Vec4A { data: [ax, ay, az, aw] }
            .add(Vec4A { data: [bx, by, bz, bw] })
            .data;
        assert!(x == lit(5) && y == lit(3) && z == lit(-3) && w == lit(3));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_add_fixed__span_loop() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<Fixed>();
        let r = dvec(array![ax, ay, az, aw]).add(dvec(array![bx, by, bz, bw])).data;
        assert!(*r[0] == lit(5) && *r[1] == lit(3) && *r[2] == lit(-3) && *r[3] == lit(3));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_fixed__baseline() {
        let (ax, _, _, _, _, _, _, _) = in4::<Fixed>();
        assert!(ax == lit(1));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_fixed__struct() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<Fixed>();
        let a = Vec4 { x: ax, y: ay, z: az, w: aw };
        assert!(a.dot(Vec4 { x: bx, y: by, z: bz, w: bw }) == lit(-52));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_fixed__array() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<Fixed>();
        let a = Vec4A { data: [ax, ay, az, aw] };
        assert!(a.dot(Vec4A { data: [bx, by, bz, bw] }) == lit(-52));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_fixed__span_loop() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<Fixed>();
        assert!(dvec(array![ax, ay, az, aw]).dot(dvec(array![bx, by, bz, bw])) == lit(-52));
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_felt__baseline() {
        let (ax, _, _, _, _, _, _, _) = in4::<felt252>();
        assert!(ax == 1);
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_felt__struct() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<felt252>();
        let a = Vec4 { x: ax, y: ay, z: az, w: aw };
        assert!(a.dot(Vec4 { x: bx, y: by, z: bz, w: bw }) == -52);
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_felt__array() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<felt252>();
        let a = Vec4A { data: [ax, ay, az, aw] };
        assert!(a.dot(Vec4A { data: [bx, by, bz, bw] }) == -52);
    }
    #[test]
    #[inline(never)]
    fn bench_vec4_dot_felt__span_loop() {
        let (ax, ay, az, aw, bx, by, bz, bw) = in4::<felt252>();
        assert!(dvec(array![ax, ay, az, aw]).dot(dvec(array![bx, by, bz, bw])) == -52);
    }

    // ------------------------------------------------------------------ component access (felt)
    fn in3() -> (felt252, felt252, felt252) {
        (input(1), input(-2), input(3))
    }

    #[test]
    #[inline(never)]
    fn bench_access_y__baseline() {
        let (_x, y, _z) = in3();
        assert!(y == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_y__struct_field() {
        let (x, y, z) = in3();
        let v = Vec3 { x, y, z };
        assert!(v.y == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_y__tuple_destructure() {
        let (x, y, z) = in3();
        let v = Vec3T { data: (x, y, z) };
        let (_, vy, _) = v.data;
        assert!(vy == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_y__array_destructure() {
        let (x, y, z) = in3();
        let v = Vec3A { data: [x, y, z] };
        let [_, vy, _] = v.data;
        assert!(vy == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_y__array_span_index() {
        let (x, y, z) = in3();
        let v = Vec3A { data: [x, y, z] };
        assert!(*v.data.span()[1] == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_y__span_index() {
        let (x, y, z) = in3();
        let v: DVecLoop<felt252> = dvec(array![x, y, z]);
        assert!(*v.data[1] == -2);
    }
    /// Runtime (dynamic) index into a static layout: `match` on the index vs span indexing.
    fn component(v: Vec3<felt252>, i: usize) -> felt252 {
        match i {
            0 => v.x,
            1 => v.y,
            2 => v.z,
            _ => core::panic_with_felt252('index out of bounds'),
        }
    }

    #[test]
    #[inline(never)]
    fn bench_access_dyn_index__baseline() {
        let (_x, y, _z) = in3();
        let _i: usize = harness::black_box(1);
        assert!(y == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_dyn_index__struct_match() {
        let (x, y, z) = in3();
        let i: usize = harness::black_box(1);
        assert!(component(Vec3 { x, y, z }, i) == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_dyn_index__array_span() {
        let (x, y, z) = in3();
        let i: usize = harness::black_box(1);
        assert!(*Vec3A { data: [x, y, z] }.data.span()[i] == -2);
    }
    #[test]
    #[inline(never)]
    fn bench_access_dyn_index__span() {
        let (x, y, z) = in3();
        let i: usize = harness::black_box(1);
        assert!(*dvec(array![x, y, z]).data[i] == -2);
    }
}
