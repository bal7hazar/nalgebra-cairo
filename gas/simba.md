# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## simba::fixed::convert::tests

### from_int

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16040 | 100 | x1.00 |

### from_ratio

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 19080 | 2740 | x1.00 |
| `fixed_pp` | 19080 | 2740 | x1.00 |

### from_raw

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 15940 | 0 | - |

### into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_i16` | 16040 | 100 | x1.00 |
| `fixed_i32` | 16040 | 100 | x1.00 |
| `fixed_i8` | 16040 | 100 | x1.00 |
| `fixed_u16` | 16040 | 100 | x1.00 |
| `fixed_u8` | 16040 | 100 | x1.00 |

### to_int

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 17050 | 1110 | x1.00 |
| `fixed_p` | 17050 | 1110 | x1.00 |

### try_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_u128` | 16310 | 370 | x1.00 |
| `fixed_u32` | 16310 | 370 | x1.00 |
| `fixed_u64` | 16310 | 370 | x1.00 |
| `fixed_i128` | 16580 | 640 | x1.73 |
| `fixed_i64` | 16580 | 640 | x1.73 |

## simba::fixed::fused::tests

### diff_prod

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 19090 | 1950 | x1.00 |
| `unfused` | 21480 | 4340 | x2.23 |

### lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18790 | 2050 | x1.00 |
| `unfused` | 19970 | 3230 | x1.58 |

### mul_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18690 | 1950 | x1.00 |
| `unfused` | 19230 | 2490 | x1.28 |

### mul_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18690 | 1950 | x1.00 |
| `unfused` | 19230 | 2490 | x1.28 |

### norm2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18360 | 2020 | x1.00 |
| `unfused` | 22600 | 6260 | x3.10 |

### norm3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18960 | 2220 | x1.00 |
| `unfused` | 25590 | 8850 | x3.99 |

### norm4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 19560 | 2420 | x1.00 |

### norm_squared2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18290 | 1950 | x1.00 |
| `unfused` | 20680 | 4340 | x2.23 |

### norm_squared3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18890 | 2150 | x1.00 |
| `unfused` | 23670 | 6930 | x3.22 |

### norm_squared4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 19490 | 2350 | x1.00 |

### sqr

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 17690 | 1750 | x1.00 |
| `unfused` | 17690 | 1750 | x1.00 |

### sum_prod2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 19090 | 1950 | x1.00 |
| `unfused` | 21480 | 4340 | x2.23 |

### sum_prod3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 20090 | 2150 | x1.00 |
| `unfused` | 24870 | 6930 | x3.22 |

### sum_prod4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 21090 | 2350 | x1.00 |
| `unfused` | 28260 | 9520 | x4.05 |

## simba::fixed::kernels::alternatives::tests

### abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_downcast_p` | 16650 | - | x1.00 |
| `alt_trim_first_n` | 16810 | - | x1.01 |
| `alt_trim_first_p` | 16820 | - | x1.01 |
| `alt_native_p` | 16830 | - | x1.01 |
| `alt_native_n` | 17010 | - | x1.02 |
| `alt_downcast_n` | 17080 | - | x1.03 |

### abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_gt` | 19170 | - | x1.00 |
| `alt_native_lt` | 19270 | - | x1.01 |

### add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_checked_pn` | 16980 | - | x1.00 |
| `alt_native_pn` | 16980 | - | x1.00 |

### ceil

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_neg_floor_n` | 17750 | - | x1.00 |
| `alt_neg_floor_p` | 17750 | - | x1.00 |
| `alt_branch_n` | 19400 | - | x1.09 |
| `alt_branch_p` | 19400 | - | x1.09 |

### div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_trunc_peel_np` | 19680 | - | x1.00 |
| `alt_trunc_peel_pp` | 19700 | - | x1.00 |
| `alt_floor_peel_pp` | 19930 | - | x1.01 |
| `alt_floor_peel_np` | 20280 | - | x1.03 |
| `alt_floor_peel_pn` | 20280 | - | x1.03 |
| `alt_native_trunc_np` | 29220 | - | x1.48 |
| `alt_native_trunc_pp` | 29220 | - | x1.48 |

### floor

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sub_rem_n` | 17420 | - | x1.00 |
| `alt_sub_rem_p` | 17420 | - | x1.00 |
| `alt_native_p` | 19410 | - | x1.11 |
| `alt_native_n` | 20270 | - | x1.16 |

### fract

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_sub_trunc_p` | 18380 | - | x1.00 |
| `alt_sub_trunc_n` | 18460 | - | x1.00 |

### from_int

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_n` | 16580 | - | x1.00 |

### inv_sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_constrain_p` | 18870 | - | x1.00 |
| `alt_two_step_p` | 19610 | - | x1.04 |
| `alt_recip_p` | 20050 | - | x1.06 |

### is_negative

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_n` | 16920 | - | x1.00 |
| `alt_native_p` | 17020 | - | x1.01 |
| `alt_is_ok_n` | 17220 | - | x1.02 |
| `alt_is_ok_p` | 17220 | - | x1.02 |

### is_positive

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_p` | 16920 | - | x1.00 |
| `alt_native_n` | 17020 | - | x1.01 |

### lt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_constrain_np` | 17220 | - | x1.00 |
| `alt_constrain_pn` | 17320 | - | x1.01 |

### min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_constrain_np` | 17110 | - | x1.00 |
| `alt_constrain_pn` | 17120 | - | x1.00 |

### mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_kernel_raw_pn` | 18090 | - | x1.00 |
| `alt_postcheck_pn` | 18190 | - | x1.01 |
| `alt_via_u128_pn` | 18190 | - | x1.01 |
| `alt_native_trunc_pp` | 26600 | - | x1.47 |
| `alt_native_trunc_pn` | 26680 | - | x1.47 |

### neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_n` | 16240 | - | x1.00 |
| `alt_native_p` | 16240 | - | x1.00 |
| `alt_downcast_n` | 16310 | - | x1.00 |
| `alt_downcast_p` | 16310 | - | x1.00 |

### recip

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_div_p` | 18680 | - | x1.00 |
| `alt_div_n` | 18760 | - | x1.00 |

### rem

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_trunc_np` | 20660 | - | x1.00 |
| `alt_native_trunc_pp` | 20660 | - | x1.00 |

### round

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_half_up_n` | 17520 | - | x1.00 |
| `alt_half_up_p` | 17520 | - | x1.00 |

### signum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_p` | 16940 | - | x1.00 |
| `alt_native_z` | 17290 | - | x1.02 |
| `alt_native_n` | 17380 | - | x1.03 |

### sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_p` | 17760 | - | x1.00 |
| `alt_constrain_p` | 17860 | - | x1.01 |

### sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_pn` | 16980 | - | x1.00 |

### sum_prod8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_typed` | 25090 | - | x1.00 |
| `alt_typed_postcheck` | 25190 | - | x1.00 |
| `alt_chunked` | 27480 | - | x1.10 |
| `alt_native_trunc` | 87720 | - | x3.50 |

### to_int

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_p` | 19370 | - | x1.00 |
| `alt_native_n` | 19630 | - | x1.01 |

### trunc

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_floor_fix_p` | 19100 | - | x1.00 |
| `alt_floor_fix_n` | 19600 | - | x1.03 |

### wide_mul_scalar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_kernel_raw` | 18320 | - | x1.00 |
| `alt_sign_split_p` | 19030 | - | x1.04 |
| `alt_corelib_divrem` | 19160 | - | x1.05 |
| `alt_sign_split_n` | 19290 | - | x1.05 |
| `alt_native_trunc` | 20120 | - | x1.10 |

## simba::fixed::math::tests

### abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 16630 | 690 | x1.00 |
| `fixed_n` | 16810 | 870 | x1.26 |

### abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_lt` | 18500 | 1750 | x1.00 |
| `fixed_gt` | 18590 | 1840 | x1.05 |

### ceil

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 17520 | 1580 | x1.00 |
| `fixed_p` | 17520 | 1580 | x1.00 |

### clamp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_below` | 17550 | 810 | x1.00 |
| `fixed_inside` | 18080 | 1340 | x1.65 |
| `fixed_above` | 18090 | 1350 | x1.67 |

### floor

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 17150 | 1210 | x1.00 |
| `fixed_p` | 17150 | 1210 | x1.00 |

### fract

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 17640 | 1700 | x1.00 |
| `fixed_n` | 17720 | 1780 | x1.05 |

### inv_sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 18770 | 2830 | x1.00 |

### is_negative

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16720 | 680 | x1.00 |
| `fixed_p` | 16820 | 780 | x1.15 |

### is_positive

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 16720 | 770 | x1.00 |
| `fixed_n` | 16810 | 860 | x1.12 |

### max

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 17110 | 770 | x1.00 |
| `fixed_np` | 17120 | 780 | x1.01 |

### min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 17110 | 770 | x1.00 |
| `fixed_pn` | 17120 | 780 | x1.01 |

### recip

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 18130 | 2190 | x1.00 |
| `fixed_n` | 18390 | 2450 | x1.12 |

### round

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 17930 | 1990 | x1.00 |
| `fixed_p` | 18090 | 2150 | x1.08 |

### signum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16610 | 670 | x1.00 |
| `fixed_p` | 16720 | 780 | x1.16 |
| `fixed_z` | 16820 | 880 | x1.31 |

### sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 17760 | 1820 | x1.00 |

### trunc

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 17640 | 1700 | x1.00 |
| `fixed_n` | 17720 | 1780 | x1.05 |

## simba::fixed::ops::tests

### add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_nn` | 16980 | 640 | x1.00 |
| `fixed_pn` | 16980 | 640 | x1.00 |
| `fixed_pp` | 16980 | 640 | x1.00 |

### add_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 16980 | 640 | x1.00 |

### div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 19080 | 2740 | x1.00 |
| `fixed_pp` | 19080 | 2740 | x1.00 |
| `fixed_nn` | 19160 | 2820 | x1.03 |
| `fixed_pn` | 19160 | 2820 | x1.03 |

### div_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 19160 | 2820 | x1.00 |

### eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 16940 | 500 | x1.00 |
| `fixed_pn` | 16940 | 500 | x1.00 |
| `fixed_pp` | 16940 | 500 | x1.00 |

### ge

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 17230 | 880 | x1.00 |
| `fixed_np` | 17310 | 960 | x1.09 |
| `fixed_pp` | 17310 | 960 | x1.09 |

### gt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 17220 | 870 | x1.00 |
| `fixed_np` | 17320 | 970 | x1.11 |
| `fixed_pp` | 17320 | 970 | x1.11 |

### le

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 17230 | 790 | x1.00 |
| `fixed_pp` | 17230 | 790 | x1.00 |
| `fixed_pn` | 17310 | 870 | x1.10 |

### lt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 17220 | 780 | x1.00 |
| `fixed_pp` | 17220 | 780 | x1.00 |
| `fixed_pn` | 17320 | 880 | x1.13 |

### mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_nn` | 18090 | 1750 | x1.00 |
| `fixed_pn` | 18090 | 1750 | x1.00 |
| `fixed_pp` | 18090 | 1750 | x1.00 |

### mul_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 18090 | 1750 | x1.00 |

### neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16240 | 300 | x1.00 |
| `fixed_p` | 16240 | 300 | x1.00 |

### rem

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 18350 | 2010 | x1.00 |
| `fixed_pp` | 18350 | 2010 | x1.00 |
| `fixed_nn` | 18420 | 2080 | x1.03 |
| `fixed_pn` | 18420 | 2080 | x1.03 |

### rem_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 18420 | 2080 | x1.00 |

### sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 16980 | 640 | x1.00 |
| `fixed_pp` | 16980 | 640 | x1.00 |

### sub_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 16980 | 640 | x1.00 |

## simba::fixed::transcendental::tests

### acos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg15_large` | 27370 | 11430 | x1.00 |
| `deg15_small` | 27370 | 11430 | x1.00 |

### asin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg11_small` | 25950 | 10010 | x1.00 |
| `deg15_large` | 27360 | 11420 | x1.14 |
| `deg15_small_x` | 27360 | 11420 | x1.14 |
| `deg19_small` | 27760 | 11820 | x1.18 |

### atan

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reciprocal_only` | 28470 | 12530 | x1.00 |
| `above_one_with_division` | 29540 | 13600 | x1.09 |
| `unit_no_division` | 29540 | 13600 | x1.09 |
| `via_atan2` | 31340 | 15400 | x1.23 |

### atan2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg17` | 29730 | 13390 | x1.00 |
| `deg21` | 31340 | 15000 | x1.12 |
| `deg25` | 31740 | 15400 | x1.15 |
| `deg29` | 33150 | 16810 | x1.26 |

### cos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg10` | 28840 | 12900 | x1.00 |

### exp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg6_alt` | 28820 | 12880 | x1.00 |
| `deg8` | 30230 | 14290 | x1.11 |
| `deg8_negative` | 30230 | 14290 | x1.11 |
| `deg10_alt` | 31640 | 15700 | x1.22 |

### ln

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg7_alt` | 27110 | 11170 | x1.00 |
| `deg11` | 28520 | 12580 | x1.13 |
| `deg11_tiny` | 28520 | 12580 | x1.13 |

### sin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `deg7_alt` | 27430 | 11490 | x1.00 |
| `deg11` | 28740 | 12800 | x1.11 |
| `deg11_huge_argument` | 28740 | 12800 | x1.11 |
| `deg11_negative` | 28740 | 12800 | x1.11 |
| `deg11_alt` | 28840 | 12900 | x1.12 |
| `deg9_alt` | 28840 | 12900 | x1.12 |

### sin_cos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shared` | 33060 | 16820 | x1.00 |
| `two_calls` | 42040 | 25800 | x1.53 |

### tan

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_ratio` | 35440 | 19500 | x1.00 |
| `q32_ratio` | 35880 | 19940 | x1.02 |

## simba::fixed::types::tests

### is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_is_zero` | 16340 | 300 | x1.00 |
| `fixed_is_one` | 16540 | 500 | x1.67 |
| `fixed_is_non_zero` | 16640 | 600 | x2.00 |
| `fixed_is_non_one` | 16740 | 700 | x2.33 |

## simba::fixed::wide::tests

### sum_prod4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_kernel` | 21090 | 2350 | x1.00 |
| `wide` | 21090 | 2350 | x1.00 |
| `wide_from_prod` | 21090 | 2350 | x1.00 |

### sum_prod6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 23090 | 2750 | x1.00 |
| `wide_from_prod` | 23090 | 2750 | x1.00 |

### sum_prod8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 25090 | 3150 | x1.00 |
| `wide_from_prod` | 25090 | 3150 | x1.00 |

### triple_product

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_mul_scalar` | 19320 | 1780 | x1.00 |
| `alt_diff_prod_then_mul` | 21340 | 3800 | x2.13 |

### wide_mixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20290 | 2350 | x1.00 |

### wide_mul_scalar

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18320 | 1580 | x1.00 |
| `alt_rescale_then_mul` | 20340 | 3600 | x2.28 |

### wide_norm6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20760 | 2820 | x1.00 |

## simba::scalar::tests

### real_cross3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `generic` | 23790 | 6650 | x1.00 |

### real_dot3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 19290 | 2150 | x1.00 |
| `generic` | 19290 | 2150 | x1.00 |
| `generic_wide` | 19290 | 2150 | x1.00 |

### real_norm3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `generic` | 18560 | 2220 | x1.00 |

