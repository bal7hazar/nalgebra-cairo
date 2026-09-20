//! Control flow: what does one loop iteration cost, and what does a branch cost?
//!
//! `loop_sum<N>` groups sum the N elements of a `Span<u64>` with every loop flavour and with
//! fully unrolled code; `loop_fixed<N>` does the same from a `[u64; N]`. The per-iteration
//! overhead is `(cost(16) - cost(4)) / 12`, compared between a loop and the unrolled version.
//! `branch_u64` measures `if`/early-return/`match` on a black-boxed condition.

pub fn sum_while(s: Span<u64>) -> u64 {
    let mut sum = 0;
    let mut i = 0;
    while i < s.len() {
        sum += *s[i];
        i += 1;
    }
    sum
}

pub fn sum_for_range(s: Span<u64>) -> u64 {
    let mut sum = 0;
    for i in 0..s.len() {
        sum += *s[i];
    }
    sum
}

pub fn sum_for_iter(s: Span<u64>) -> u64 {
    let mut sum = 0;
    for x in s {
        sum += *x;
    }
    sum
}

/// Accumulate in felt252 (no per-iteration range check), narrow once at the end.
pub fn sum_for_iter_felt(s: Span<u64>) -> u64 {
    let mut sum: felt252 = 0;
    for x in s {
        sum += (*x).into();
    }
    sum.try_into().unwrap()
}

pub fn sum_loop_pop(s: Span<u64>) -> u64 {
    let mut s = s;
    let mut sum = 0;
    loop {
        match s.pop_front() {
            Some(x) => sum += *x,
            None => { break; },
        }
    }
    sum
}

pub fn sum_while_let(s: Span<u64>) -> u64 {
    let mut s = s;
    let mut sum = 0;
    while let Some(x) = s.pop_front() {
        sum += *x;
    }
    sum
}

pub fn sum_rec(s: Span<u64>) -> u64 {
    let mut s = s;
    match s.pop_front() {
        Some(x) => *x + sum_rec(s),
        None => 0,
    }
}

pub fn sum_rec_acc_entry(s: Span<u64>) -> u64 {
    sum_rec_acc(s, 0)
}

fn sum_rec_acc(s: Span<u64>, acc: u64) -> u64 {
    let mut s = s;
    match s.pop_front() {
        Some(x) => sum_rec_acc(s, acc + *x),
        None => acc,
    }
}

pub fn sum_unrolled1(s: Span<u64>) -> u64 {
    *s[0]
}

/// `multi_pop_front` turns the first 1 elements into a boxed fixed-size array: one bounds check.
pub fn sum_multipop1(s: Span<u64>) -> u64 {
    let mut s = s;
    let [x0] = (*s.multi_pop_front::<1>().unwrap()).unbox();
    x0
}

pub fn sum_fixed1(a: [u64; 1]) -> u64 {
    let [x0] = a;
    x0
}

pub fn sum_unrolled4(s: Span<u64>) -> u64 {
    *s[0] + *s[1] + *s[2] + *s[3]
}

/// `multi_pop_front` turns the first 4 elements into a boxed fixed-size array: one bounds check.
pub fn sum_multipop4(s: Span<u64>) -> u64 {
    let mut s = s;
    let [x0, x1, x2, x3] = (*s.multi_pop_front::<4>().unwrap()).unbox();
    x0 + x1 + x2 + x3
}

pub fn sum_fixed4(a: [u64; 4]) -> u64 {
    let [x0, x1, x2, x3] = a;
    x0 + x1 + x2 + x3
}

pub fn sum_unrolled16(s: Span<u64>) -> u64 {
    *s[0]
        + *s[1]
        + *s[2]
        + *s[3]
        + *s[4]
        + *s[5]
        + *s[6]
        + *s[7]
        + *s[8]
        + *s[9]
        + *s[10]
        + *s[11]
        + *s[12]
        + *s[13]
        + *s[14]
        + *s[15]
}

/// `multi_pop_front` turns the first 16 elements into a boxed fixed-size array: one bounds check.
pub fn sum_multipop16(s: Span<u64>) -> u64 {
    let mut s = s;
    let [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15] = (*s
        .multi_pop_front::<16>()
        .unwrap())
        .unbox();
    x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15
}

pub fn sum_fixed16(a: [u64; 16]) -> u64 {
    let [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15] = a;
    x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9 + x10 + x11 + x12 + x13 + x14 + x15
}

// ------------------------------------------------------------------------------------ branches

pub fn select_if(c: bool, a: u64, b: u64) -> u64 {
    if c {
        a
    } else {
        b
    }
}

pub fn select_early_return(c: bool, a: u64, b: u64) -> u64 {
    if c {
        return a;
    }
    b
}

pub fn select_match(c: bool, a: u64, b: u64) -> u64 {
    match c {
        true => a,
        false => b,
    }
}

/// Branchless select: `b + c * (a - b)` in felt252, then narrow.
pub fn select_arith_felt(c: bool, a: u64, b: u64) -> u64 {
    let c: felt252 = c.into();
    let a: felt252 = a.into();
    let b: felt252 = b.into();
    (b + c * (a - b)).try_into().unwrap()
}

/// Four nested conditions vs. one.
pub fn select_nested4(c: bool, a: u64, b: u64) -> u64 {
    if c {
        if a > b {
            if a != 0 {
                if b != 0 {
                    return a;
                }
            }
        }
    }
    b
}

#[cfg(test)]
mod tests {
    use harness::black_box;
    use super::*;

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__baseline() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(s.len() == 1);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__while_index() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_while(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__for_range_index() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_for_range(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__for_span_iter() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_for_iter(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__loop_pop_front() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_loop_pop(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__while_let_pop_front() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_while_let(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__recursion() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_rec(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__recursion_tail_acc() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_rec_acc_entry(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__unrolled_span_index() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_unrolled1(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__unrolled_multi_pop_front() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_multipop1(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum1__for_span_iter_felt_acc() {
        let s: Span<u64> = black_box(array![1000]).span();
        assert!(sum_for_iter_felt(s) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed1__baseline() {
        let a: [u64; 1] = black_box([1000]);
        let _ = a;
        assert!(black_box(1_u32) == 1);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed1__destructure_unrolled() {
        let a: [u64; 1] = black_box([1000]);
        assert!(black_box(1_u32) == 1);
        assert!(sum_fixed1(a) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed1__span_for_iter() {
        let a: [u64; 1] = black_box([1000]);
        assert!(black_box(1_u32) == 1);
        assert!(sum_for_iter(a.span()) == 1000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__baseline() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(s.len() == 4);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__while_index() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_while(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__for_range_index() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_for_range(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__for_span_iter() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_for_iter(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__loop_pop_front() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_loop_pop(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__while_let_pop_front() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_while_let(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__recursion() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_rec(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__recursion_tail_acc() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_rec_acc_entry(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__unrolled_span_index() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_unrolled4(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__unrolled_multi_pop_front() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_multipop4(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum4__for_span_iter_felt_acc() {
        let s: Span<u64> = black_box(array![1000, 2000, 3000, 4000]).span();
        assert!(sum_for_iter_felt(s) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed4__baseline() {
        let a: [u64; 4] = black_box([1000, 2000, 3000, 4000]);
        let _ = a;
        assert!(black_box(4_u32) == 4);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed4__destructure_unrolled() {
        let a: [u64; 4] = black_box([1000, 2000, 3000, 4000]);
        assert!(black_box(4_u32) == 4);
        assert!(sum_fixed4(a) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed4__span_for_iter() {
        let a: [u64; 4] = black_box([1000, 2000, 3000, 4000]);
        assert!(black_box(4_u32) == 4);
        assert!(sum_for_iter(a.span()) == 10000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__baseline() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(s.len() == 16);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__while_index() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_while(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__for_range_index() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_for_range(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__for_span_iter() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_for_iter(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__loop_pop_front() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_loop_pop(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__while_let_pop_front() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_while_let(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__recursion() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_rec(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__recursion_tail_acc() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_rec_acc_entry(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__unrolled_span_index() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_unrolled16(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__unrolled_multi_pop_front() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_multipop16(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_sum16__for_span_iter_felt_acc() {
        let s: Span<u64> = black_box(
            array![
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        )
            .span();
        assert!(sum_for_iter_felt(s) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed16__baseline() {
        let a: [u64; 16] = black_box(
            [
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        );
        let _ = a;
        assert!(black_box(16_u32) == 16);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed16__destructure_unrolled() {
        let a: [u64; 16] = black_box(
            [
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        );
        assert!(black_box(16_u32) == 16);
        assert!(sum_fixed16(a) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_loop_fixed16__span_for_iter() {
        let a: [u64; 16] = black_box(
            [
                1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000,
                14000, 15000, 16000,
            ],
        );
        assert!(black_box(16_u32) == 16);
        assert!(sum_for_iter(a.span()) == 136000);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__baseline() {
        let (c, a, b): (bool, u64, u64) = (black_box(true), black_box(7), black_box(3));
        let _ = (c, b);
        assert!(a == 7);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__if_else() {
        let (c, a, b): (bool, u64, u64) = (black_box(true), black_box(7), black_box(3));
        assert!(select_if(c, a, b) == 7);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__if_else_not_taken() {
        let (c, a, b): (bool, u64, u64) = (black_box(false), black_box(3), black_box(7));
        assert!(select_if(c, a, b) == 7);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__early_return() {
        let (c, a, b): (bool, u64, u64) = (black_box(true), black_box(7), black_box(3));
        assert!(select_early_return(c, a, b) == 7);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__match_bool() {
        let (c, a, b): (bool, u64, u64) = (black_box(true), black_box(7), black_box(3));
        assert!(select_match(c, a, b) == 7);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__branchless_felt_arith() {
        let (c, a, b): (bool, u64, u64) = (black_box(true), black_box(7), black_box(3));
        assert!(select_arith_felt(c, a, b) == 7);
    }

    #[test]
    #[inline(never)]
    fn bench_branch_u64__nested_4_conditions() {
        let (c, a, b): (bool, u64, u64) = (black_box(true), black_box(7), black_box(3));
        assert!(select_nested4(c, a, b) == 7);
    }
}
