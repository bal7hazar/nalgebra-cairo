//! alexandria (math + linalg, registry 0.10.0): integer / numeric primitives that a fixed-point
//! linear algebra library could be tempted to build on. References use corelib only.
//!
//! Same fixed overhead in every test of a group (`inputs` + `check`).

mod integer {
    use alexandria_math::const_pow::pow2;
    use alexandria_math::fast_power::fast_power;
    use alexandria_math::fast_root::fast_sqrt;
    use alexandria_math::opt_math::OptBitShift;
    use alexandria_math::{BitShift, pow};
    use core::num::traits::{Pow, Sqrt};
    use harness::black_box;

    #[inline(never)]
    fn inputs() -> (u64, u64) {
        (black_box(15032385536), black_box(16)) // 3.5 in Q32.32, a shift amount
    }

    #[inline(never)]
    fn check(value: u128, expected: u128) {
        assert!(value == expected);
    }

    // ---- 2^n ----------------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_pow2_u64__baseline() {
        let (_x, n) = inputs();
        check(n.into(), 16);
    }

    #[test]
    #[inline(never)]
    fn bench_pow2_u64__alexandria_pow() {
        let (_x, n) = inputs();
        check(pow(2_u64, n).into(), 65536);
    }

    #[test]
    #[inline(never)]
    fn bench_pow2_u64__alexandria_fast_power() {
        let (_x, n) = inputs();
        check(fast_power(2_u64, n).into(), 65536);
    }

    #[test]
    #[inline(never)]
    fn bench_pow2_u64__alexandria_const_pow2_lut() {
        let (_x, n) = inputs();
        check(pow2(n.try_into().unwrap()), 65536);
    }

    #[test]
    #[inline(never)]
    fn bench_pow2_u64__core_pow() {
        let (_x, n) = inputs();
        let n: u32 = n.try_into().unwrap();
        check(2_u64.pow(n).into(), 65536);
    }

    // ---- shifts -------------------------------------------------------------------------------

    #[test]
    #[inline(never)]
    fn bench_shr_u64__baseline() {
        let (x, _n) = inputs();
        check(x.into(), 15032385536);
    }

    #[test]
    #[inline(never)]
    fn bench_shr_u64__alexandria_bitshift() {
        let (x, n) = inputs();
        check(BitShift::shr(x, n).into(), 229376);
    }

    #[test]
    #[inline(never)]
    fn bench_shr_u64__alexandria_opt_bitshift_lut() {
        let (x, n) = inputs();
        check(OptBitShift::shr(x, n.try_into().unwrap()).into(), 229376);
    }

    /// Shift by a compile-time constant: a division by a literal.
    #[test]
    #[inline(never)]
    fn bench_shr_u64__reference_const_div() {
        let (x, _n) = inputs();
        check((x / 0x10000).into(), 229376);
    }

    #[test]
    #[inline(never)]
    fn bench_shl_u64__baseline() {
        let (x, _n) = inputs();
        check(x.into(), 15032385536);
    }

    #[test]
    #[inline(never)]
    fn bench_shl_u64__alexandria_bitshift() {
        let (x, n) = inputs();
        check(BitShift::shl(x, n).into(), 985162418487296);
    }

    #[test]
    #[inline(never)]
    fn bench_shl_u64__alexandria_opt_bitshift_lut() {
        let (x, n) = inputs();
        check(OptBitShift::shl(x, n.try_into().unwrap()).into(), 985162418487296);
    }

    #[test]
    #[inline(never)]
    fn bench_shl_u64__reference_const_mul() {
        let (x, _n) = inputs();
        check((x * 0x10000).into(), 985162418487296);
    }

    // ---- integer square root --------------------------------------------------------------------

    const X: u128 = 64563604257983430656; // 3.5 * 2^64, i.e. the radicand of a Q32.32 sqrt

    #[inline(never)]
    fn wide_input() -> u128 {
        black_box(X)
    }

    #[inline(never)]
    fn check_near(value: u128, expected: u128) {
        assert!(value == expected || value == expected + 1);
    }

    #[test]
    #[inline(never)]
    fn bench_isqrt_u128__baseline() {
        check_near(wide_input(), X);
    }

    /// Newton-Raphson from x/2 with a caller-chosen iteration count: 40 iterations are needed to
    /// converge on this input (35 are not enough), nothing tells the caller.
    #[test]
    #[inline(never)]
    fn bench_isqrt_u128__alexandria_fast_sqrt_40_iterations() {
        check_near(fast_sqrt(wide_input(), 40), 8035148054);
    }

    #[test]
    #[inline(never)]
    fn bench_isqrt_u128__core_sqrt() {
        let root: u64 = Sqrt::sqrt(wide_input());
        check_near(root.into(), 8035148054);
    }
}

mod linalg {
    use alexandria_linalg::dot::dot;
    use alexandria_linalg::kron::kron;
    use alexandria_linalg::norm::norm;
    use core::num::traits::Sqrt;
    use harness::black_box;
    use origami_algebra::vector::{Vector, VectorTrait};

    #[inline(never)]
    fn inputs() -> (Span<u64>, Span<u64>, [u64; 3], [u64; 3]) {
        let a = black_box([3_u64, 4, 12]);
        let b = black_box([5_u64, 6, 7]);
        (a.span(), b.span(), a, b)
    }

    #[inline(never)]
    fn check(value: u128, expected: u128) {
        assert!(value == expected);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__baseline() {
        let (sa, _sb, _a, _b) = inputs();
        check((*sa[0]).into(), 3);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__alexandria_dot() {
        let (sa, sb, _a, _b) = inputs();
        check(dot(sa, sb).into(), 123);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__origami_vector() {
        let (sa, sb, _a, _b) = inputs();
        let (va, vb): (Vector<u64>, Vector<u64>) = (VectorTrait::new(sa), VectorTrait::new(sb));
        check(va.dot(vb).into(), 123);
    }

    #[test]
    #[inline(never)]
    fn bench_dot3_u64__reference_unrolled() {
        let (_sa, _sb, a, b) = inputs();
        let [ax, ay, az] = a;
        let [bx, by, bz] = b;
        check((ax * bx + ay * by + az * bz).into(), 123);
    }

    #[test]
    #[inline(never)]
    fn bench_norm3_u64__baseline() {
        let (sa, _sb, _a, _b) = inputs();
        check((*sa[0]).into(), 3);
    }

    /// L2 norm, 10 Newton iterations (the value used by alexandria's own tests).
    #[test]
    #[inline(never)]
    fn bench_norm3_u64__alexandria_norm() {
        let (sa, _sb, _a, _b) = inputs();
        check(norm(sa, 2, 10), 13);
    }

    #[test]
    #[inline(never)]
    fn bench_norm3_u64__reference_unrolled() {
        let (_sa, _sb, a, _b) = inputs();
        let [ax, ay, az] = a;
        let root: u32 = Sqrt::sqrt(ax * ax + ay * ay + az * az);
        check(root.into(), 13);
    }

    #[test]
    #[inline(never)]
    fn bench_kron3_u64__baseline() {
        let (sa, _sb, _a, _b) = inputs();
        check((*sa[2]).into(), 12);
    }

    #[test]
    #[inline(never)]
    fn bench_kron3_u64__alexandria_kron() {
        let (sa, sb, _a, _b) = inputs();
        let r = kron(sa, sb).unwrap();
        check((*r[8]).into(), 84);
    }

    #[test]
    #[inline(never)]
    fn bench_kron3_u64__reference_unrolled() {
        let (_sa, _sb, a, b) = inputs();
        let [ax, ay, az] = a;
        let [bx, by, bz] = b;
        let r = [ax * bx, ax * by, ax * bz, ay * bx, ay * by, ay * bz, az * bx, az * by, az * bz];
        let [_, _, _, _, _, _, _, _, last] = r;
        check(last.into(), 84);
    }
}

/// Decimal fixed point (1e18 "wad") on u256, the DeFi convention.
mod wad {
    use alexandria_math::wad_ray_math::{wad_div, wad_mul};
    use harness::black_box;

    const A: u256 = 3500000000000000000; // 3.5
    const B: u256 = 1250000000000000000; // 1.25

    #[inline(never)]
    fn inputs() -> (u256, u256) {
        (black_box(A), black_box(B))
    }

    #[inline(never)]
    fn check(value: u256, expected: u256) {
        assert!(value == expected);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_wad__baseline() {
        let (a, _b) = inputs();
        check(a, A);
    }

    #[test]
    #[inline(never)]
    fn bench_mul_wad__alexandria_wad_mul() {
        let (a, b) = inputs();
        check(wad_mul(a, b), 4375000000000000000);
    }

    #[test]
    #[inline(never)]
    fn bench_div_wad__baseline() {
        let (a, _b) = inputs();
        check(a, A);
    }

    #[test]
    #[inline(never)]
    fn bench_div_wad__alexandria_wad_div() {
        let (a, b) = inputs();
        check(wad_div(a, b), 2800000000000000000);
    }
}
