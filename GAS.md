# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_testing

### testing

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_u64` | 15910 | 370 | x1.00 |

## simba

### abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 16630 | 690 | x1.00 |
| `alt_downcast_p` | 16650 | 710 | x1.03 |
| `alt_trim_first_n` | 16810 | 870 | x1.26 |
| `fixed_n` | 16810 | 870 | x1.26 |
| `alt_trim_first_p` | 16820 | 880 | x1.28 |
| `alt_native_p` | 16830 | 890 | x1.29 |
| `alt_native_n` | 17010 | 1070 | x1.55 |
| `alt_downcast_n` | 17080 | 1140 | x1.65 |

### abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_lt` | 18500 | 1750 | x1.00 |
| `fixed_gt` | 18590 | 1840 | x1.05 |
| `alt_native_gt` | 19170 | 2420 | x1.38 |
| `alt_native_lt` | 19270 | 2520 | x1.44 |

### add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_nn` | 16980 | 640 | x1.00 |
| `fixed_pp` | 16980 | 640 | x1.00 |
| `fixed_pn` | 16980 | 640 | x1.00 |
| `alt_native_pn` | 16980 | 640 | x1.00 |
| `alt_checked_pn` | 16980 | 640 | x1.00 |

### add_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 16980 | 640 | x1.00 |

### ceil

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 17520 | 1580 | x1.00 |
| `fixed_p` | 17520 | 1580 | x1.00 |
| `alt_neg_floor_n` | 17750 | 1810 | x1.15 |
| `alt_neg_floor_p` | 17750 | 1810 | x1.15 |
| `alt_branch_n` | 19400 | 3460 | x2.19 |
| `alt_branch_p` | 19400 | 3460 | x2.19 |

### clamp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_below` | 17550 | 810 | x1.00 |
| `fixed_inside` | 18080 | 1340 | x1.65 |
| `fixed_above` | 18090 | 1350 | x1.67 |

### diff_prod

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 19090 | 1950 | x1.00 |
| `unfused` | 21480 | 4340 | x2.23 |

### div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 19080 | 2740 | x1.00 |
| `fixed_pp` | 19080 | 2740 | x1.00 |
| `fixed_nn` | 19160 | 2820 | x1.03 |
| `fixed_pn` | 19160 | 2820 | x1.03 |
| `alt_trunc_peel_np` | 19680 | 3340 | x1.22 |
| `alt_trunc_peel_pp` | 19700 | 3360 | x1.23 |
| `alt_floor_peel_pp` | 19930 | 3590 | x1.31 |
| `alt_floor_peel_np` | 20280 | 3940 | x1.44 |
| `alt_floor_peel_pn` | 20280 | 3940 | x1.44 |
| `alt_native_trunc_pp` | 29220 | 12880 | x4.70 |
| `alt_native_trunc_np` | 29220 | 12880 | x4.70 |

### div_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 19160 | 2820 | x1.00 |

### eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 16940 | 500 | x1.00 |
| `fixed_pp` | 16940 | 500 | x1.00 |
| `fixed_pn` | 16940 | 500 | x1.00 |

### floor

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 17150 | 1210 | x1.00 |
| `fixed_n` | 17150 | 1210 | x1.00 |
| `alt_sub_rem_n` | 17420 | 1480 | x1.22 |
| `alt_sub_rem_p` | 17420 | 1480 | x1.22 |
| `alt_native_p` | 19410 | 3470 | x2.87 |
| `alt_native_n` | 20270 | 4330 | x3.58 |

### fract

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 17640 | 1700 | x1.00 |
| `fixed_n` | 17720 | 1780 | x1.05 |
| `alt_sub_trunc_p` | 18380 | 2440 | x1.44 |
| `alt_sub_trunc_n` | 18460 | 2520 | x1.48 |

### from_int

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16040 | 100 | x1.00 |
| `alt_native_n` | 16580 | 640 | x6.40 |

### from_ratio

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pp` | 19080 | 2740 | x1.00 |
| `fixed_np` | 19080 | 2740 | x1.00 |

### from_raw

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 15940 | 0 | - |

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
| `fixed_pp` | 17320 | 970 | x1.11 |
| `fixed_np` | 17320 | 970 | x1.11 |

### into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_i32` | 16040 | 100 | x1.00 |
| `fixed_i16` | 16040 | 100 | x1.00 |
| `fixed_u16` | 16040 | 100 | x1.00 |
| `fixed_i8` | 16040 | 100 | x1.00 |
| `fixed_u8` | 16040 | 100 | x1.00 |

### inv_sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 18770 | 2830 | x1.00 |
| `alt_constrain_p` | 18870 | 2930 | x1.04 |
| `alt_two_step_p` | 19610 | 3670 | x1.30 |
| `alt_recip_p` | 20050 | 4110 | x1.45 |

### is_negative

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16720 | 680 | x1.00 |
| `fixed_p` | 16820 | 780 | x1.15 |
| `alt_native_n` | 16920 | 880 | x1.29 |
| `alt_native_p` | 17020 | 980 | x1.44 |
| `alt_is_ok_p` | 17220 | 1180 | x1.74 |
| `alt_is_ok_n` | 17220 | 1180 | x1.74 |

### is_positive

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 16720 | 770 | x1.00 |
| `fixed_n` | 16810 | 860 | x1.12 |
| `alt_native_p` | 16920 | 970 | x1.26 |
| `alt_native_n` | 17020 | 1070 | x1.39 |

### is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_is_zero` | 16340 | 300 | x1.00 |
| `fixed_is_one` | 16540 | 500 | x1.67 |
| `fixed_is_non_zero` | 16640 | 600 | x2.00 |
| `fixed_is_non_one` | 16740 | 700 | x2.33 |

### le

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 17230 | 790 | x1.00 |
| `fixed_pp` | 17230 | 790 | x1.00 |
| `fixed_pn` | 17310 | 870 | x1.10 |

### lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18790 | 2050 | x1.00 |
| `unfused` | 19970 | 3230 | x1.58 |

### lt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_constrain_np` | 17220 | 780 | x1.00 |
| `fixed_np` | 17220 | 780 | x1.00 |
| `fixed_pp` | 17220 | 780 | x1.00 |
| `alt_constrain_pn` | 17320 | 880 | x1.13 |
| `fixed_pn` | 17320 | 880 | x1.13 |

### max

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 17110 | 770 | x1.00 |
| `fixed_np` | 17120 | 780 | x1.01 |

### min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_constrain_np` | 17110 | 770 | x1.00 |
| `fixed_np` | 17110 | 770 | x1.00 |
| `alt_constrain_pn` | 17120 | 780 | x1.01 |
| `fixed_pn` | 17120 | 780 | x1.01 |

### mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_kernel_raw_pn` | 18090 | 1750 | x1.00 |
| `fixed_nn` | 18090 | 1750 | x1.00 |
| `fixed_pp` | 18090 | 1750 | x1.00 |
| `fixed_pn` | 18090 | 1750 | x1.00 |
| `alt_via_u128_pn` | 18190 | 1850 | x1.06 |
| `alt_postcheck_pn` | 18190 | 1850 | x1.06 |
| `alt_native_trunc_pp` | 26600 | 10260 | x5.86 |
| `alt_native_trunc_pn` | 26680 | 10340 | x5.91 |

### mul_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18690 | 1950 | x1.00 |
| `unfused` | 19230 | 2490 | x1.28 |

### mul_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 18090 | 1750 | x1.00 |

### mul_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 18690 | 1950 | x1.00 |
| `unfused` | 19230 | 2490 | x1.28 |

### neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16240 | 300 | x1.00 |
| `fixed_p` | 16240 | 300 | x1.00 |
| `alt_native_n` | 16240 | 300 | x1.00 |
| `alt_native_p` | 16240 | 300 | x1.00 |
| `alt_downcast_n` | 16310 | 370 | x1.23 |
| `alt_downcast_p` | 16310 | 370 | x1.23 |

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

### recip

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 18130 | 2190 | x1.00 |
| `fixed_n` | 18390 | 2450 | x1.12 |
| `alt_div_p` | 18680 | 2740 | x1.25 |
| `alt_div_n` | 18760 | 2820 | x1.29 |

### rem

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_np` | 18350 | 2010 | x1.00 |
| `fixed_pp` | 18350 | 2010 | x1.00 |
| `fixed_pn` | 18420 | 2080 | x1.03 |
| `fixed_nn` | 18420 | 2080 | x1.03 |
| `alt_native_trunc_pp` | 20660 | 4320 | x2.15 |
| `alt_native_trunc_np` | 20660 | 4320 | x2.15 |

### rem_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 18420 | 2080 | x1.00 |

### round

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_half_up_p` | 17520 | 1580 | x1.00 |
| `alt_half_up_n` | 17520 | 1580 | x1.00 |
| `fixed_n` | 17930 | 1990 | x1.26 |
| `fixed_p` | 18090 | 2150 | x1.36 |

### signum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 16610 | 670 | x1.00 |
| `fixed_p` | 16720 | 780 | x1.16 |
| `fixed_z` | 16820 | 880 | x1.31 |
| `alt_native_p` | 16940 | 1000 | x1.49 |
| `alt_native_z` | 17290 | 1350 | x2.01 |
| `alt_native_n` | 17380 | 1440 | x2.15 |

### sqr

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed` | 17690 | 1750 | x1.00 |
| `unfused` | 17690 | 1750 | x1.00 |

### sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_p` | 17760 | 1820 | x1.00 |
| `fixed_p` | 17760 | 1820 | x1.00 |
| `alt_constrain_p` | 17860 | 1920 | x1.05 |

### sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_native_pn` | 16980 | 640 | x1.00 |
| `fixed_pn` | 16980 | 640 | x1.00 |
| `fixed_pp` | 16980 | 640 | x1.00 |

### sub_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_pn` | 16980 | 640 | x1.00 |

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
| `fused_kernel` | 21090 | 2350 | x1.00 |
| `wide_from_prod` | 21090 | 2350 | x1.00 |
| `wide` | 21090 | 2350 | x1.00 |
| `unfused` | 28260 | 9520 | x4.05 |

### sum_prod6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 23090 | 2750 | x1.00 |
| `wide_from_prod` | 23090 | 2750 | x1.00 |

### sum_prod8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_typed` | 25090 | 3150 | x1.00 |
| `wide` | 25090 | 3150 | x1.00 |
| `wide_from_prod` | 25090 | 3150 | x1.00 |
| `alt_typed_postcheck` | 25190 | 3250 | x1.03 |
| `alt_chunked` | 27480 | 5540 | x1.76 |
| `alt_native_trunc` | 87720 | 65780 | x20.88 |

### to_int

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_n` | 17050 | 1110 | x1.00 |
| `fixed_p` | 17050 | 1110 | x1.00 |
| `alt_native_p` | 19370 | 3430 | x3.09 |
| `alt_native_n` | 19630 | 3690 | x3.32 |

### trunc

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_p` | 17640 | 1700 | x1.00 |
| `fixed_n` | 17720 | 1780 | x1.05 |
| `alt_floor_fix_p` | 19100 | 3160 | x1.86 |
| `alt_floor_fix_n` | 19600 | 3660 | x2.15 |

### try_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fixed_u64` | 16310 | 370 | x1.00 |
| `fixed_u128` | 16310 | 370 | x1.00 |
| `fixed_u32` | 16310 | 370 | x1.00 |
| `fixed_i128` | 16580 | 640 | x1.73 |
| `fixed_i64` | 16580 | 640 | x1.73 |

### wide_mixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20290 | 2350 | x1.00 |

### wide_norm6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20760 | 2820 | x1.00 |

