//! Representation-agnostic compositions (how a generic linear algebra library would be written).
//! All `#[inline]` so the measured cost is the cost of the scalar operators only.

#[inline]
pub fn dot3<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>>(
    ax: T, ay: T, az: T, bx: T, by: T, bz: T,
) -> T {
    ax * bx + ay * by + az * bz
}

#[inline]
pub fn cross3<T, +Sub<T>, +Mul<T>, +Copy<T>, +Drop<T>>(
    ax: T, ay: T, az: T, bx: T, by: T, bz: T,
) -> (T, T, T) {
    (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
}

/// a + (b - a) * t
#[inline]
pub fn lerp<T, +Add<T>, +Sub<T>, +Mul<T>, +Copy<T>, +Drop<T>>(a: T, b: T, t: T) -> T {
    a + (b - a) * t
}

/// a * b + c
#[inline]
pub fn mul_add<T, +Add<T>, +Mul<T>, +Copy<T>, +Drop<T>>(a: T, b: T, c: T) -> T {
    a * b + c
}
