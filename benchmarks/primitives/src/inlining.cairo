//! Inlining: the same function with `#[inline(always)]`, `#[inline(never)]` and the default
//! heuristic, called 1x and 8x (chained so that calls cannot be merged).
//!
//! Groups: `inline_small_x1`, `inline_small_x8` (one-liner `a * b + c`), `inline_medium_x1`,
//! `inline_medium_x8` (Q32.32 dot product of two 3-vectors, ~15 libfuncs), `inline_large_x1`,
//! `inline_large_x4` (four dot products, calls summed).

use core::num::traits::WideMul;

#[inline(always)]
pub fn muladd_always(a: u64, b: u64, c: u64) -> u64 {
    a * b + c
}

#[inline(never)]
pub fn muladd_never(a: u64, b: u64, c: u64) -> u64 {
    a * b + c
}

pub fn muladd_default(a: u64, b: u64, c: u64) -> u64 {
    a * b + c
}

const ONE: u128 = 0x100000000;

#[inline(always)]
pub fn dot3_always(ax: u64, ay: u64, az: u64, bx: u64, by: u64, bz: u64) -> u64 {
    ((ax.wide_mul(bx) + ay.wide_mul(by) + az.wide_mul(bz)) / ONE).try_into().unwrap()
}

#[inline(never)]
pub fn dot3_never(ax: u64, ay: u64, az: u64, bx: u64, by: u64, bz: u64) -> u64 {
    ((ax.wide_mul(bx) + ay.wide_mul(by) + az.wide_mul(bz)) / ONE).try_into().unwrap()
}

pub fn dot3_default(ax: u64, ay: u64, az: u64, bx: u64, by: u64, bz: u64) -> u64 {
    ((ax.wide_mul(bx) + ay.wide_mul(by) + az.wide_mul(bz)) / ONE).try_into().unwrap()
}

/// Large: four dot products (~60 libfuncs once flattened).
#[inline(always)]
pub fn large_always(a: u64, b: u64, c: u64) -> u64 {
    let x = dot3_always(a, b, c, a, b, c);
    let y = dot3_always(b, c, a, a, b, c);
    let z = dot3_always(c, a, b, a, b, c);
    dot3_always(x, y, z, x, y, z)
}

#[inline(never)]
pub fn large_never(a: u64, b: u64, c: u64) -> u64 {
    let x = dot3_always(a, b, c, a, b, c);
    let y = dot3_always(b, c, a, a, b, c);
    let z = dot3_always(c, a, b, a, b, c);
    dot3_always(x, y, z, x, y, z)
}

pub fn large_default(a: u64, b: u64, c: u64) -> u64 {
    let x = dot3_always(a, b, c, a, b, c);
    let y = dot3_always(b, c, a, a, b, c);
    let z = dot3_always(c, a, b, a, b, c);
    dot3_always(x, y, z, x, y, z)
}

#[cfg(test)]
mod tests {
    use harness::black_box;
    use super::*;

    // r -> r * 3 + 1, eight times from 1: 1, 4, 13, 40, 121, 364, 1093, 3280, 9841
    #[test]
    #[inline(never)]
    fn bench_inline_small_x1__baseline() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        let _ = (b, c);
        assert!(a == 1);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x1__hand_inlined() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        assert!(a * b + c == 4);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x1__inline_always() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        assert!(muladd_always(a, b, c) == 4);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x1__inline_never() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        assert!(muladd_never(a, b, c) == 4);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x1__inline_default() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        assert!(muladd_default(a, b, c) == 4);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x8__baseline() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        let _ = (b, c);
        assert!(a == 1);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x8__hand_inlined() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        let r = a * b + c;
        let r = r * b + c;
        let r = r * b + c;
        let r = r * b + c;
        let r = r * b + c;
        let r = r * b + c;
        let r = r * b + c;
        let r = r * b + c;
        assert!(r == 9841);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x8__inline_always() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        let r = muladd_always(a, b, c);
        let r = muladd_always(r, b, c);
        let r = muladd_always(r, b, c);
        let r = muladd_always(r, b, c);
        let r = muladd_always(r, b, c);
        let r = muladd_always(r, b, c);
        let r = muladd_always(r, b, c);
        let r = muladd_always(r, b, c);
        assert!(r == 9841);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x8__inline_never() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        let r = muladd_never(a, b, c);
        let r = muladd_never(r, b, c);
        let r = muladd_never(r, b, c);
        let r = muladd_never(r, b, c);
        let r = muladd_never(r, b, c);
        let r = muladd_never(r, b, c);
        let r = muladd_never(r, b, c);
        let r = muladd_never(r, b, c);
        assert!(r == 9841);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_small_x8__inline_default() {
        let (a, b, c): (u64, u64, u64) = (black_box(1), black_box(3), black_box(1));
        let r = muladd_default(a, b, c);
        let r = muladd_default(r, b, c);
        let r = muladd_default(r, b, c);
        let r = muladd_default(r, b, c);
        let r = muladd_default(r, b, c);
        let r = muladd_default(r, b, c);
        let r = muladd_default(r, b, c);
        let r = muladd_default(r, b, c);
        assert!(r == 9841);
    }

    // Q32.32: x = 1.5, y = 2.0; dot((x, y, y), (x, y, y)) = 2.25 + 4 + 4 = 10.25.
    // Chained: r = dot((r, y, y), (x, y, y)) = 1.5 r + 8.
    const X: u64 = 0x180000000;
    const Y: u64 = 0x200000000;

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x1__baseline() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        let _ = y;
        assert!(x == X);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x1__inline_always() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        assert!(dot3_always(x, y, y, x, y, y) == 0xa40000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x1__inline_never() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        assert!(dot3_never(x, y, y, x, y, y) == 0xa40000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x1__inline_default() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        assert!(dot3_default(x, y, y, x, y, y) == 0xa40000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x8__baseline() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        let _ = y;
        assert!(x == X);
    }

    // r0 = 1.5; r_{n+1} = 1.5 r_n + 8 ; r8 = 432.505859375
    const R8: u64 = 0x1b081800000;

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x8__inline_always() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        let r = dot3_always(x, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        let r = dot3_always(r, y, y, x, y, y);
        assert!(r == R8);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x8__inline_never() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        let r = dot3_never(x, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        let r = dot3_never(r, y, y, x, y, y);
        assert!(r == R8);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_medium_x8__inline_default() {
        let (x, y): (u64, u64) = (black_box(X), black_box(Y));
        let r = dot3_default(x, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        let r = dot3_default(r, y, y, x, y, y);
        assert!(r == R8);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x1__baseline() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let _ = (b, c);
        assert!(a == 0x80000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x1__inline_always() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let r = large_always(a, b, c);
        assert!(r == 0x1b6000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x1__inline_never() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let r = large_never(a, b, c);
        assert!(r == 0x1b6000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x1__inline_default() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let r = large_default(a, b, c);
        assert!(r == 0x1b6000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x4__baseline() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let _ = (b, c);
        assert!(a == 0x80000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x4__inline_always() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let r = large_always(a, b, c)
            + large_always(b, c, a)
            + large_always(c, a, b)
            + large_always(a, c, b);
        assert!(r == 0x6d8000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x4__inline_never() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let r = large_never(a, b, c)
            + large_never(b, c, a)
            + large_never(c, a, b)
            + large_never(a, c, b);
        assert!(r == 0x6d8000000);
    }

    #[test]
    #[inline(never)]
    fn bench_inline_large_x4__inline_default() {
        let (a, b, c): (u64, u64, u64) = (
            black_box(0x80000000), black_box(0x40000000), black_box(0xc0000000),
        );
        let r = large_default(a, b, c)
            + large_default(b, c, a)
            + large_default(c, a, b)
            + large_default(a, c, b);
        assert!(r == 0x6d8000000);
    }
}
