//! Question 1 — vector layout. One trait, six layouts, generic over the scalar.
//!
//! * `Vec3<T>`      : struct with named fields
//! * `Vec3A<T>`     : newtype over a fixed-size array `[T; 3]`
//! * `Vec3T<T>`     : newtype over a tuple `(T, T, T)`
//! * `DVecIdx<T>`   : `Span<T>`, fixed dimension known by the code, unrolled with `span[i]`
//! * `DVecBox<T>`   : `Span<T>`, read through `multi_pop_front::<3>()` (one bounds check)
//! * `DVecLoop<T>`  : `Span<T>`, truly dynamic dimension, loops (orion / alexandria style)

pub trait Vec3Ops<V, T> {
    fn new(x: T, y: T, z: T) -> V;
    fn x(self: V) -> T;
    fn y(self: V) -> T;
    fn z(self: V) -> T;
    fn add(self: V, rhs: V) -> V;
    fn sub(self: V, rhs: V) -> V;
    fn scale(self: V, k: T) -> V;
    fn neg(self: V) -> V;
    fn dot(self: V, rhs: V) -> T;
    fn cross(self: V, rhs: V) -> V;
    fn equals(self: V, rhs: V) -> bool;
}

// ---------------------------------------------------------------------------------------------
// struct { x, y, z }
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop, Debug, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

pub impl StructVec3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Vec3Ops<Vec3<T>, T> {
    fn new(x: T, y: T, z: T) -> Vec3<T> {
        Vec3 { x, y, z }
    }
    fn x(self: Vec3<T>) -> T {
        self.x
    }
    fn y(self: Vec3<T>) -> T {
        self.y
    }
    fn z(self: Vec3<T>) -> T {
        self.z
    }
    fn add(self: Vec3<T>, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
    fn sub(self: Vec3<T>, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
    fn scale(self: Vec3<T>, k: T) -> Vec3<T> {
        Vec3 { x: self.x * k, y: self.y * k, z: self.z * k }
    }
    fn neg(self: Vec3<T>) -> Vec3<T> {
        Vec3 { x: -self.x, y: -self.y, z: -self.z }
    }
    fn dot(self: Vec3<T>, rhs: Vec3<T>) -> T {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    fn cross(self: Vec3<T>, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.y * rhs.z - self.z * rhs.y,
            y: self.z * rhs.x - self.x * rhs.z,
            z: self.x * rhs.y - self.y * rhs.x,
        }
    }
    fn equals(self: Vec3<T>, rhs: Vec3<T>) -> bool {
        self.x == rhs.x && self.y == rhs.y && self.z == rhs.z
    }
}

// ---------------------------------------------------------------------------------------------
// [T; 3]
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct Vec3A<T> {
    pub data: [T; 3],
}

pub impl ArrayVec3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Vec3Ops<Vec3A<T>, T> {
    fn new(x: T, y: T, z: T) -> Vec3A<T> {
        Vec3A { data: [x, y, z] }
    }
    fn x(self: Vec3A<T>) -> T {
        let [x, _, _] = self.data;
        x
    }
    fn y(self: Vec3A<T>) -> T {
        let [_, y, _] = self.data;
        y
    }
    fn z(self: Vec3A<T>) -> T {
        let [_, _, z] = self.data;
        z
    }
    fn add(self: Vec3A<T>, rhs: Vec3A<T>) -> Vec3A<T> {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        Vec3A { data: [ax + bx, ay + by, az + bz] }
    }
    fn sub(self: Vec3A<T>, rhs: Vec3A<T>) -> Vec3A<T> {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        Vec3A { data: [ax - bx, ay - by, az - bz] }
    }
    fn scale(self: Vec3A<T>, k: T) -> Vec3A<T> {
        let [ax, ay, az] = self.data;
        Vec3A { data: [ax * k, ay * k, az * k] }
    }
    fn neg(self: Vec3A<T>) -> Vec3A<T> {
        let [ax, ay, az] = self.data;
        Vec3A { data: [-ax, -ay, -az] }
    }
    fn dot(self: Vec3A<T>, rhs: Vec3A<T>) -> T {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        ax * bx + ay * by + az * bz
    }
    fn cross(self: Vec3A<T>, rhs: Vec3A<T>) -> Vec3A<T> {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        Vec3A { data: [ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx] }
    }
    fn equals(self: Vec3A<T>, rhs: Vec3A<T>) -> bool {
        let [ax, ay, az] = self.data;
        let [bx, by, bz] = rhs.data;
        ax == bx && ay == by && az == bz
    }
}

// ---------------------------------------------------------------------------------------------
// (T, T, T)
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct Vec3T<T> {
    pub data: (T, T, T),
}

pub impl TupleVec3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Vec3Ops<Vec3T<T>, T> {
    fn new(x: T, y: T, z: T) -> Vec3T<T> {
        Vec3T { data: (x, y, z) }
    }
    fn x(self: Vec3T<T>) -> T {
        let (x, _, _) = self.data;
        x
    }
    fn y(self: Vec3T<T>) -> T {
        let (_, y, _) = self.data;
        y
    }
    fn z(self: Vec3T<T>) -> T {
        let (_, _, z) = self.data;
        z
    }
    fn add(self: Vec3T<T>, rhs: Vec3T<T>) -> Vec3T<T> {
        let (ax, ay, az) = self.data;
        let (bx, by, bz) = rhs.data;
        Vec3T { data: (ax + bx, ay + by, az + bz) }
    }
    fn sub(self: Vec3T<T>, rhs: Vec3T<T>) -> Vec3T<T> {
        let (ax, ay, az) = self.data;
        let (bx, by, bz) = rhs.data;
        Vec3T { data: (ax - bx, ay - by, az - bz) }
    }
    fn scale(self: Vec3T<T>, k: T) -> Vec3T<T> {
        let (ax, ay, az) = self.data;
        Vec3T { data: (ax * k, ay * k, az * k) }
    }
    fn neg(self: Vec3T<T>) -> Vec3T<T> {
        let (ax, ay, az) = self.data;
        Vec3T { data: (-ax, -ay, -az) }
    }
    fn dot(self: Vec3T<T>, rhs: Vec3T<T>) -> T {
        let (ax, ay, az) = self.data;
        let (bx, by, bz) = rhs.data;
        ax * bx + ay * by + az * bz
    }
    fn cross(self: Vec3T<T>, rhs: Vec3T<T>) -> Vec3T<T> {
        let (ax, ay, az) = self.data;
        let (bx, by, bz) = rhs.data;
        Vec3T { data: (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx) }
    }
    fn equals(self: Vec3T<T>, rhs: Vec3T<T>) -> bool {
        let (ax, ay, az) = self.data;
        let (bx, by, bz) = rhs.data;
        ax == bx && ay == by && az == bz
    }
}

// ---------------------------------------------------------------------------------------------
// Span<T>, dimension known statically, `span[i]` access
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct DVecIdx<T> {
    pub data: Span<T>,
}

pub impl SpanIdxVec3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Vec3Ops<DVecIdx<T>, T> {
    fn new(x: T, y: T, z: T) -> DVecIdx<T> {
        DVecIdx { data: array![x, y, z].span() }
    }
    fn x(self: DVecIdx<T>) -> T {
        *self.data[0]
    }
    fn y(self: DVecIdx<T>) -> T {
        *self.data[1]
    }
    fn z(self: DVecIdx<T>) -> T {
        *self.data[2]
    }
    fn add(self: DVecIdx<T>, rhs: DVecIdx<T>) -> DVecIdx<T> {
        let (a, b) = (self.data, rhs.data);
        DVecIdx { data: array![*a[0] + *b[0], *a[1] + *b[1], *a[2] + *b[2]].span() }
    }
    fn sub(self: DVecIdx<T>, rhs: DVecIdx<T>) -> DVecIdx<T> {
        let (a, b) = (self.data, rhs.data);
        DVecIdx { data: array![*a[0] - *b[0], *a[1] - *b[1], *a[2] - *b[2]].span() }
    }
    fn scale(self: DVecIdx<T>, k: T) -> DVecIdx<T> {
        let a = self.data;
        DVecIdx { data: array![*a[0] * k, *a[1] * k, *a[2] * k].span() }
    }
    fn neg(self: DVecIdx<T>) -> DVecIdx<T> {
        let a = self.data;
        DVecIdx { data: array![-*a[0], -*a[1], -*a[2]].span() }
    }
    fn dot(self: DVecIdx<T>, rhs: DVecIdx<T>) -> T {
        let (a, b) = (self.data, rhs.data);
        *a[0] * *b[0] + *a[1] * *b[1] + *a[2] * *b[2]
    }
    fn cross(self: DVecIdx<T>, rhs: DVecIdx<T>) -> DVecIdx<T> {
        let (a, b) = (self.data, rhs.data);
        let (ax, ay, az) = (*a[0], *a[1], *a[2]);
        let (bx, by, bz) = (*b[0], *b[1], *b[2]);
        DVecIdx { data: array![ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx].span() }
    }
    fn equals(self: DVecIdx<T>, rhs: DVecIdx<T>) -> bool {
        let (a, b) = (self.data, rhs.data);
        *a[0] == *b[0] && *a[1] == *b[1] && *a[2] == *b[2]
    }
}

// ---------------------------------------------------------------------------------------------
// Span<T>, dimension known statically, `multi_pop_front::<3>()` access (single bounds check)
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct DVecBox<T> {
    pub data: Span<T>,
}

#[inline]
fn unpack3<T, +Copy<T>, +Drop<T>>(span: Span<T>) -> (T, T, T) {
    let mut span = span;
    let [x, y, z] = (*span.multi_pop_front::<3>().expect('dim mismatch')).unbox();
    (x, y, z)
}

pub impl SpanBoxVec3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Vec3Ops<DVecBox<T>, T> {
    fn new(x: T, y: T, z: T) -> DVecBox<T> {
        DVecBox { data: array![x, y, z].span() }
    }
    fn x(self: DVecBox<T>) -> T {
        let (x, _, _) = unpack3(self.data);
        x
    }
    fn y(self: DVecBox<T>) -> T {
        let (_, y, _) = unpack3(self.data);
        y
    }
    fn z(self: DVecBox<T>) -> T {
        let (_, _, z) = unpack3(self.data);
        z
    }
    fn add(self: DVecBox<T>, rhs: DVecBox<T>) -> DVecBox<T> {
        let (ax, ay, az) = unpack3(self.data);
        let (bx, by, bz) = unpack3(rhs.data);
        DVecBox { data: array![ax + bx, ay + by, az + bz].span() }
    }
    fn sub(self: DVecBox<T>, rhs: DVecBox<T>) -> DVecBox<T> {
        let (ax, ay, az) = unpack3(self.data);
        let (bx, by, bz) = unpack3(rhs.data);
        DVecBox { data: array![ax - bx, ay - by, az - bz].span() }
    }
    fn scale(self: DVecBox<T>, k: T) -> DVecBox<T> {
        let (ax, ay, az) = unpack3(self.data);
        DVecBox { data: array![ax * k, ay * k, az * k].span() }
    }
    fn neg(self: DVecBox<T>) -> DVecBox<T> {
        let (ax, ay, az) = unpack3(self.data);
        DVecBox { data: array![-ax, -ay, -az].span() }
    }
    fn dot(self: DVecBox<T>, rhs: DVecBox<T>) -> T {
        let (ax, ay, az) = unpack3(self.data);
        let (bx, by, bz) = unpack3(rhs.data);
        ax * bx + ay * by + az * bz
    }
    fn cross(self: DVecBox<T>, rhs: DVecBox<T>) -> DVecBox<T> {
        let (ax, ay, az) = unpack3(self.data);
        let (bx, by, bz) = unpack3(rhs.data);
        DVecBox { data: array![ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx].span() }
    }
    fn equals(self: DVecBox<T>, rhs: DVecBox<T>) -> bool {
        let (ax, ay, az) = unpack3(self.data);
        let (bx, by, bz) = unpack3(rhs.data);
        ax == bx && ay == by && az == bz
    }
}

// ---------------------------------------------------------------------------------------------
// Span<T>, dynamic dimension, loops
// ---------------------------------------------------------------------------------------------

#[derive(Copy, Drop)]
pub struct DVecLoop<T> {
    pub data: Span<T>,
}

pub impl SpanLoopVec3<
    T, +Add<T>, +Sub<T>, +Mul<T>, +Neg<T>, +PartialEq<T>, +Copy<T>, +Drop<T>,
> of Vec3Ops<DVecLoop<T>, T> {
    fn new(x: T, y: T, z: T) -> DVecLoop<T> {
        DVecLoop { data: array![x, y, z].span() }
    }
    fn x(self: DVecLoop<T>) -> T {
        *self.data[0]
    }
    fn y(self: DVecLoop<T>) -> T {
        *self.data[1]
    }
    fn z(self: DVecLoop<T>) -> T {
        *self.data[2]
    }
    fn add(self: DVecLoop<T>, rhs: DVecLoop<T>) -> DVecLoop<T> {
        let (mut a, mut b) = (self.data, rhs.data);
        assert(a.len() == b.len(), 'dim mismatch');
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x + *b.pop_front().unwrap());
        }
        DVecLoop { data: out.span() }
    }
    fn sub(self: DVecLoop<T>, rhs: DVecLoop<T>) -> DVecLoop<T> {
        let (mut a, mut b) = (self.data, rhs.data);
        assert(a.len() == b.len(), 'dim mismatch');
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x - *b.pop_front().unwrap());
        }
        DVecLoop { data: out.span() }
    }
    fn scale(self: DVecLoop<T>, k: T) -> DVecLoop<T> {
        let mut a = self.data;
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(*x * k);
        }
        DVecLoop { data: out.span() }
    }
    fn neg(self: DVecLoop<T>) -> DVecLoop<T> {
        let mut a = self.data;
        let mut out = array![];
        while let Some(x) = a.pop_front() {
            out.append(-*x);
        }
        DVecLoop { data: out.span() }
    }
    fn dot(self: DVecLoop<T>, rhs: DVecLoop<T>) -> T {
        let (mut a, mut b) = (self.data, rhs.data);
        assert(a.len() == b.len() && a.len() != 0, 'dim mismatch');
        let mut acc = *a.pop_front().unwrap() * *b.pop_front().unwrap();
        while let Some(x) = a.pop_front() {
            acc = acc + *x * *b.pop_front().unwrap();
        }
        acc
    }
    fn cross(self: DVecLoop<T>, rhs: DVecLoop<T>) -> DVecLoop<T> {
        let (a, b) = (self.data, rhs.data);
        assert(a.len() == 3 && b.len() == 3, 'dim mismatch');
        // Cyclic-index loop, as a dimension-agnostic library would write it.
        let mut out = array![];
        for i in 0..3_usize {
            let j = (i + 1) % 3;
            let k = (i + 2) % 3;
            out.append(*a[j] * *b[k] - *a[k] * *b[j]);
        }
        DVecLoop { data: out.span() }
    }
    fn equals(self: DVecLoop<T>, rhs: DVecLoop<T>) -> bool {
        let (mut a, mut b) = (self.data, rhs.data);
        if a.len() != b.len() {
            return false;
        }
        let mut equal = true;
        while let Some(x) = a.pop_front() {
            if *x != *b.pop_front().unwrap() {
                equal = false;
                break;
            }
        }
        equal
    }
}

// ---------------------------------------------------------------------------------------------
// Generic benchmark bodies (monomorphised per layout x scalar by the generated tests).
// Inputs: a = (1, -2, 3), b = (4, 5, -6), k = 7, all opaque.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
pub mod runners {
    use crate::fixed::{Scalar, input};
    use super::Vec3Ops;

    #[inline]
    fn inputs<T, +Scalar<T>, +Drop<T>>() -> (T, T, T, T, T, T, T) {
        (input(1), input(-2), input(3), input(4), input(5), input(-6), input(7))
    }

    #[inline]
    fn lit<T, +Scalar<T>>(n: i64) -> T {
        Scalar::<T>::from_int(n)
    }

    /// Baseline of every group asserting three scalars.
    #[inline(never)]
    pub fn baseline3<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>() {
        let (ax, ay, az, _, _, _, _) = inputs::<T>();
        assert!(ax == lit(1) && ay == lit(-2) && az == lit(3));
    }

    /// Baseline of every group asserting one scalar / one bool.
    #[inline(never)]
    pub fn baseline1<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>() {
        let (ax, _, _, _, _, _, _) = inputs::<T>();
        assert!(ax == lit(1));
    }

    #[inline(never)]
    pub fn construct<
        V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>,
    >() {
        let (ax, ay, az, _, _, _, _) = inputs::<T>();
        let r: V = Vec3Ops::new(ax, ay, az);
        assert!(r.x() == lit(1) && r.y() == lit(-2) && r.z() == lit(3));
    }

    #[inline(never)]
    pub fn add<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, bx, by, bz, _) = inputs::<T>();
        let r: V = Vec3Ops::new(ax, ay, az).add(Vec3Ops::new(bx, by, bz));
        assert!(r.x() == lit(5) && r.y() == lit(3) && r.z() == lit(-3));
    }

    #[inline(never)]
    pub fn sub<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, bx, by, bz, _) = inputs::<T>();
        let r: V = Vec3Ops::new(ax, ay, az).sub(Vec3Ops::new(bx, by, bz));
        assert!(r.x() == lit(-3) && r.y() == lit(-7) && r.z() == lit(9));
    }

    #[inline(never)]
    pub fn scale<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, _, _, _, k) = inputs::<T>();
        let r: V = Vec3Ops::new(ax, ay, az).scale(k);
        assert!(r.x() == lit(7) && r.y() == lit(-14) && r.z() == lit(21));
    }

    #[inline(never)]
    pub fn neg<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, _, _, _, _) = inputs::<T>();
        let r: V = Vec3Ops::new(ax, ay, az).neg();
        assert!(r.x() == lit(-1) && r.y() == lit(2) && r.z() == lit(-3));
    }

    #[inline(never)]
    pub fn cross<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, bx, by, bz, _) = inputs::<T>();
        let r: V = Vec3Ops::new(ax, ay, az).cross(Vec3Ops::new(bx, by, bz));
        assert!(r.x() == lit(-3) && r.y() == lit(18) && r.z() == lit(13));
    }

    #[inline(never)]
    pub fn dot<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, bx, by, bz, _) = inputs::<T>();
        let a: V = Vec3Ops::new(ax, ay, az);
        assert!(a.dot(Vec3Ops::new(bx, by, bz)) == lit(-24));
    }

    /// Equality of two equal vectors (worst case: every component compared).
    #[inline(never)]
    pub fn equals<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, _, _, _, _) = inputs::<T>();
        let (cx, cy, cz, _, _, _, _) = inputs::<T>();
        let a: V = Vec3Ops::new(ax, ay, az);
        assert!(a.equals(Vec3Ops::new(cx, cy, cz)));
    }

    #[inline(never)]
    pub fn equals_baseline<T, +Scalar<T>, +PartialEq<T>, +Drop<T>>() {
        let (ax, _, _, _, _, _, _) = inputs::<T>();
        let (_, _, _, _, _, _, _) = inputs::<T>();
        assert!(ax == lit(1));
    }

    /// Chained expression typical of an integrator: (a + b*k - a).cross(b).dot(a).
    #[inline(never)]
    pub fn chain<V, T, +Vec3Ops<V, T>, +Scalar<T>, +PartialEq<T>, +Drop<T>, +Drop<V>, +Copy<V>>() {
        let (ax, ay, az, bx, by, bz, k) = inputs::<T>();
        let a: V = Vec3Ops::new(ax, ay, az);
        let b: V = Vec3Ops::new(bx, by, bz);
        // (a + b*k - b) = a + 6b = (25, 28, -33); cross a = ...
        let c = a.add(b.scale(k)).sub(b);
        let d = c.cross(a);
        // c x a = (28*3 - (-33)(-2), (-33)*1 - 25*3, 25*(-2) - 28*1) = (18, -108, -78)
        assert!(d.dot(b) == lit(0) && d.x() == lit(18) && d.z() == lit(-78));
    }
}
