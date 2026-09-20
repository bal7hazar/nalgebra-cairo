# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## scalar::tests::algo

### acos_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sqrt_poly9_typed_m0p3` | 32450 | 16510 | x1.00 |
| `sqrt_poly7_m0p3` | 39310 | 23370 | x1.42 |
| `via_atan2_m0p3` | 52130 | 36190 | x2.19 |

### atan2_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `poly15_typed_small` | 32660 | 16320 | x1.00 |
| `poly15_typed_swap_neg` | 32660 | 16320 | x1.00 |
| `poly19_typed_small` | 35180 | 18840 | x1.15 |
| `poly19_typed_swap_neg` | 35180 | 18840 | x1.15 |
| `poly15_small` | 46880 | 30540 | x1.87 |
| `poly15_swap_neg` | 46880 | 30540 | x1.87 |
| `reduced_poly9_small` | 47760 | 31420 | x1.93 |
| `reduced_poly9_swap_neg` | 47760 | 31420 | x1.93 |

### cos_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `poly11_typed_m2p5` | 28540 | 12600 | x1.00 |
| `poly9_fused_m2p5` | 33170 | 17230 | x1.37 |

### inv_sqrt_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sqrt_then_const_udiv_1234` | 18870 | 2930 | x1.00 |
| `sqrt_then_signed_div_1234` | 20350 | 4410 | x1.51 |
| `newton_mul_only_1234` | 64450 | 48510 | x16.56 |

### normalize3_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `length_inv96_3mul_mix` | 27490 | 9750 | x1.00 |
| `length_inv32_3mul_mix` | 27860 | 10120 | x1.04 |
| `length_3div_mix` | 31440 | 13700 | x1.41 |

### sin_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `poly9_typed_0p5` | 27130 | 11190 | x1.00 |
| `poly9_typed_m2p5` | 27130 | 11190 | x1.00 |
| `lut256_array_0p5` | 28120 | 12180 | x1.09 |
| `lut256_array_m2p5` | 28120 | 12180 | x1.09 |
| `lut64_array_0p5` | 28120 | 12180 | x1.09 |
| `lut64_array_m2p5` | 28120 | 12180 | x1.09 |
| `poly11_typed_0p5` | 28440 | 12500 | x1.12 |
| `poly11_typed_m2p5` | 28440 | 12500 | x1.12 |
| `lut64_match_0p5` | 29400 | 13460 | x1.20 |
| `lut64_match_m2p5` | 29400 | 13460 | x1.20 |
| `poly7_fused_0p5` | 31020 | 15080 | x1.35 |
| `poly7_fused_m2p5` | 31020 | 15080 | x1.35 |
| `poly9_fused_0p5` | 33070 | 17130 | x1.53 |
| `poly9_fused_m2p5` | 33070 | 17130 | x1.53 |
| `poly11_fused_0p5` | 35120 | 19180 | x1.71 |
| `poly11_fused_m2p5` | 35120 | 19180 | x1.71 |
| `poly9_unfused_0p5` | 36030 | 20090 | x1.80 |
| `poly9_unfused_m2p5` | 36030 | 20090 | x1.80 |
| `taylor_loop_cubit_style_0p5` | 98500 | 82560 | x7.38 |
| `taylor_loop_cubit_style_m2p5` | 99080 | 83140 | x7.43 |
| `cordic20_unrolled_0p5` | 154580 | 138640 | x12.39 |
| `cordic20_unrolled_m2p5` | 154950 | 139010 | x12.42 |

### sqrt_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `corelib_0p02` | 17860 | 1920 | x1.00 |
| `corelib_1234` | 17860 | 1920 | x1.00 |
| `newton_seeded_unrolled_0p02` | 47080 | 31140 | x16.22 |
| `newton_seeded_unrolled_1234` | 47080 | 31140 | x16.22 |
| `via_inv_sqrt_newton_0p02` | 66400 | 50460 | x26.28 |
| `via_inv_sqrt_newton_1234` | 66400 | 50460 | x26.28 |
| `newton_loop_0p02` | 181140 | 165200 | x86.04 |
| `newton_loop_1234` | 220100 | 204160 | x106.33 |

## scalar::tests::rep_felt

### abs_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 17430 | 1590 | x1.00 |
| `felt_p` | 17750 | 1910 | x1.20 |

### add4_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pn` | 16640 | 400 | x1.00 |

### add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_nn` | 16340 | 100 | x1.00 |
| `felt_pn` | 16340 | 100 | x1.00 |
| `felt_pp` | 16340 | 100 | x1.00 |

### cross3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_lazy_mix` | 25800 | 7160 | x1.00 |
| `felt_mix` | 32160 | 13520 | x1.89 |

### div_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pp` | 21090 | 4850 | x1.00 |
| `felt_pn` | 21160 | 4920 | x1.01 |

### dot3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_lazy_mix` | 20360 | 2520 | x1.00 |
| `felt_mix` | 24600 | 6760 | x2.68 |

### eq_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_p` | 16850 | 500 | x1.00 |

### from_int_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 15940 | 100 | x1.00 |

### length3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_mix` | 25220 | 8580 | x1.00 |

### lerp_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_mix` | 18960 | 2320 | x1.00 |
| `felt_lazy_mix` | 19060 | 2420 | x1.04 |

### lt_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_nn` | 18140 | 1790 | x1.00 |
| `felt_np` | 18140 | 1790 | x1.00 |
| `felt_pp` | 18140 | 1790 | x1.00 |

### mul4_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pn` | 25020 | 8780 | x1.00 |

### mul_add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_mix` | 18860 | 2220 | x1.00 |
| `felt_lazy_mix` | 18960 | 2320 | x1.05 |

### mul_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_soft_nn` | 17720 | 1480 | x1.00 |
| `felt_soft_pn` | 17720 | 1480 | x1.00 |
| `felt_soft_pp` | 17720 | 1480 | x1.00 |
| `felt_nn` | 18360 | 2120 | x1.43 |
| `felt_pn` | 18360 | 2120 | x1.43 |
| `felt_pp` | 18360 | 2120 | x1.43 |
| `felt_plain_nn` | 18830 | 2590 | x1.75 |
| `felt_plain_pn` | 18830 | 2590 | x1.75 |
| `felt_plain_pp` | 18830 | 2590 | x1.75 |

### neg_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_p` | 15940 | 100 | x1.00 |

### sqrt_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_p` | 17560 | 1720 | x1.00 |

### sub_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pn` | 16340 | 100 | x1.00 |
| `felt_pp` | 16340 | 100 | x1.00 |

### to_int_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 17590 | 1650 | x1.00 |
| `felt_p` | 17590 | 1650 | x1.00 |

## scalar::tests::rep_i128

### abs_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 16830 | 890 | x1.00 |
| `native_n` | 17010 | 1070 | x1.20 |

### add4_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 18120 | 1780 | x1.00 |

### add_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 16710 | 370 | x1.00 |
| `native_pn` | 16710 | 370 | x1.00 |
| `native_pp` | 16710 | 370 | x1.00 |

### cross3_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 72790 | 53850 | x1.00 |

### div_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pp` | 29840 | 13500 | x1.00 |
| `native_pn` | 30100 | 13760 | x1.02 |

### dot3_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_lazy_mix` | 31100 | 13160 | x1.00 |
| `native_mix` | 44890 | 26950 | x2.05 |

### eq_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 16850 | 500 | x1.00 |

### from_int_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_n` | 16310 | 370 | x1.00 |

### length3_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 51170 | 34430 | x1.00 |

### lerp_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 26370 | 9630 | x1.00 |

### lt_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 17220 | 870 | x1.00 |
| `native_np` | 17220 | 870 | x1.00 |
| `native_pp` | 17220 | 870 | x1.00 |

### mul4_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 51960 | 35620 | x1.00 |

### mul_add_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 25900 | 9160 | x1.00 |

### mul_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_nn` | 20840 | 4500 | x1.00 |
| `bounded_pn` | 20840 | 4500 | x1.00 |
| `bounded_pp` | 20840 | 4500 | x1.00 |
| `native_pp` | 24770 | 8430 | x1.87 |
| `native_pn` | 25030 | 8690 | x1.93 |
| `native_nn` | 25310 | 8970 | x1.99 |

### neg_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 16240 | 300 | x1.00 |

### sqrt_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 23300 | 7360 | x1.00 |

### sub_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 16710 | 370 | x1.00 |
| `native_pp` | 16710 | 370 | x1.00 |

### to_int_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 19840 | 3900 | x1.00 |
| `native_n` | 20100 | 4160 | x1.07 |

## scalar::tests::rep_i32

### abs_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 16830 | 890 | x1.00 |
| `native_n` | 17010 | 1070 | x1.20 |

### add4_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 19200 | 2860 | x1.00 |

### add_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 16980 | 640 | x1.00 |
| `native_pn` | 16980 | 640 | x1.00 |
| `native_pp` | 16980 | 640 | x1.00 |

### cross3_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 36760 | 17820 | x1.00 |

### div_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 21200 | 4860 | x1.00 |
| `native_pp` | 21200 | 4860 | x1.00 |

### dot3_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 27100 | 9160 | x1.00 |

### eq_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 16850 | 500 | x1.00 |

### from_int_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_n` | 16580 | 640 | x1.00 |

### length3_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 27660 | 10920 | x1.00 |

### lerp_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 20740 | 4000 | x1.00 |

### lt_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 17220 | 870 | x1.00 |
| `native_np` | 17220 | 870 | x1.00 |
| `native_pp` | 17220 | 870 | x1.00 |

### mul4_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 26560 | 10220 | x1.00 |

### mul_add_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 20000 | 3260 | x1.00 |

### mul_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 18780 | 2440 | x1.00 |
| `native_pp` | 18780 | 2440 | x1.00 |
| `native_pn` | 18860 | 2520 | x1.03 |

### neg_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 16240 | 300 | x1.00 |

### sqrt_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 17760 | 1820 | x1.00 |

### sub_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 16980 | 640 | x1.00 |
| `native_pp` | 16980 | 640 | x1.00 |

### to_int_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 19370 | 3430 | x1.00 |
| `native_n` | 19630 | 3690 | x1.08 |

## scalar::tests::rep_i64

### abs_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_bi_p` | 16650 | 710 | x1.00 |
| `bounded_p` | 16830 | 890 | x1.25 |
| `native_p` | 16830 | 890 | x1.25 |
| `bounded_n` | 17010 | 1070 | x1.51 |
| `native_n` | 17010 | 1070 | x1.51 |
| `bounded_bi_n` | 17080 | 1140 | x1.61 |

### add4_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_pn` | 19200 | 2860 | x1.00 |
| `native_pn` | 19200 | 2860 | x1.00 |

### add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_bi_nn` | 16980 | 640 | x1.00 |
| `bounded_bi_pn` | 16980 | 640 | x1.00 |
| `bounded_bi_pp` | 16980 | 640 | x1.00 |
| `bounded_nn` | 16980 | 640 | x1.00 |
| `bounded_pn` | 16980 | 640 | x1.00 |
| `bounded_pp` | 16980 | 640 | x1.00 |
| `native_nn` | 16980 | 640 | x1.00 |
| `native_pn` | 16980 | 640 | x1.00 |
| `native_pp` | 16980 | 640 | x1.00 |

### cross3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 25590 | 6650 | x1.00 |
| `bounded_mix` | 33060 | 14120 | x2.12 |
| `native_mix` | 39580 | 20640 | x3.10 |

### div_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_pp` | 19910 | 3570 | x1.00 |
| `bounded_pn` | 19980 | 3640 | x1.02 |
| `native_pn` | 21670 | 5330 | x1.49 |
| `native_pp` | 21670 | 5330 | x1.49 |

### dot3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 20190 | 2250 | x1.00 |
| `bounded_mix` | 25170 | 7230 | x3.21 |
| `native_mix` | 28510 | 10570 | x4.70 |

### eq_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_p` | 16850 | 500 | x1.00 |
| `native_p` | 16850 | 500 | x1.00 |

### from_int_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_n` | 16040 | 100 | x1.00 |
| `native_n` | 16580 | 640 | x6.40 |

### length3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 18960 | 2220 | x1.00 |
| `bounded_mix` | 25990 | 9250 | x4.17 |
| `native_mix` | 29070 | 12330 | x5.55 |

### lerp_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 18890 | 2150 | x1.00 |
| `bounded_mix` | 20070 | 3330 | x1.55 |
| `native_mix` | 21210 | 4470 | x2.08 |

### lt_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_nn` | 17220 | 870 | x1.00 |
| `bounded_np` | 17220 | 870 | x1.00 |
| `bounded_pp` | 17220 | 870 | x1.00 |
| `native_nn` | 17220 | 870 | x1.00 |
| `native_np` | 17220 | 870 | x1.00 |
| `native_pp` | 17220 | 870 | x1.00 |

### mul4_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_pn` | 24040 | 7700 | x1.00 |
| `native_pn` | 28440 | 12100 | x1.57 |

### mul_add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 18790 | 2050 | x1.00 |
| `bounded_mix` | 19330 | 2590 | x1.26 |
| `native_mix` | 20470 | 3730 | x1.82 |

### mul_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_precheck_nn` | 18090 | 1750 | x1.00 |
| `bounded_precheck_pn` | 18090 | 1750 | x1.00 |
| `bounded_precheck_pp` | 18090 | 1750 | x1.00 |
| `bounded_nn` | 18190 | 1850 | x1.06 |
| `bounded_pn` | 18190 | 1850 | x1.06 |
| `bounded_pp` | 18190 | 1850 | x1.06 |
| `bounded_u128_nn` | 18190 | 1850 | x1.06 |
| `bounded_u128_pn` | 18190 | 1850 | x1.06 |
| `bounded_u128_pp` | 18190 | 1850 | x1.06 |
| `bounded_trunc_nn` | 18780 | 2440 | x1.39 |
| `bounded_trunc_pp` | 18780 | 2440 | x1.39 |
| `bounded_trunc_pn` | 18860 | 2520 | x1.44 |
| `native_nn` | 19250 | 2910 | x1.66 |
| `native_pp` | 19250 | 2910 | x1.66 |
| `native_pn` | 19330 | 2990 | x1.71 |

### neg_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_p` | 16240 | 300 | x1.00 |
| `native_p` | 16240 | 300 | x1.00 |
| `bounded_bi_p` | 16310 | 370 | x1.23 |

### sqrt_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 17760 | 1820 | x1.00 |
| `bounded_p` | 17860 | 1920 | x1.05 |

### sub_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_bi_pn` | 16980 | 640 | x1.00 |
| `bounded_bi_pp` | 16980 | 640 | x1.00 |
| `bounded_pn` | 16980 | 640 | x1.00 |
| `bounded_pp` | 16980 | 640 | x1.00 |
| `native_pn` | 16980 | 640 | x1.00 |
| `native_pp` | 16980 | 640 | x1.00 |

### to_int_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_n` | 17050 | 1110 | x1.00 |
| `bounded_p` | 17050 | 1110 | x1.00 |
| `native_p` | 19370 | 3430 | x3.09 |
| `native_n` | 19630 | 3690 | x3.32 |

## scalar::tests::rep_sm32

### abs_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 16640 | -200 | - |
| `signmag_p` | 16640 | -200 | - |

### add4_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 25060 | 7610 | x1.00 |

### add_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_nn` | 19060 | 1620 | x1.00 |
| `signmag_pp` | 19240 | 1800 | x1.11 |
| `signmag_pn` | 19880 | 2440 | x1.51 |

### cross3_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 39500 | 17140 | x1.00 |

### div_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 19220 | 1780 | x1.00 |
| `signmag_pn` | 19230 | 1790 | x1.01 |

### dot3_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 28750 | 8900 | x1.00 |

### eq_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 17650 | 900 | x1.00 |

### from_int_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 18860 | 2410 | x1.00 |

### length3_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 27460 | 9420 | x1.00 |

### lerp_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 24710 | 6670 | x1.00 |

### lt_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_np` | 17930 | 1080 | x1.00 |
| `signmag_nn` | 18930 | 2080 | x1.93 |
| `signmag_pp` | 19030 | 2180 | x2.02 |

### mul4_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 24260 | 6820 | x1.00 |

### mul_add_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 22080 | 4030 | x1.00 |

### mul_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 19220 | 1780 | x1.00 |
| `signmag_pn` | 19230 | 1790 | x1.01 |
| `signmag_nn` | 19320 | 1880 | x1.06 |

### neg_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 17350 | 600 | x1.00 |

### sqrt_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 18020 | 1180 | x1.00 |

### sub_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 19640 | 2200 | x1.00 |
| `signmag_pp` | 20280 | 2840 | x1.29 |

### to_int_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 18220 | 2280 | x1.00 |
| `signmag_n` | 18590 | 2650 | x1.16 |

## scalar::tests::rep_sm64

### abs_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 16640 | -200 | - |
| `signmag_p` | 16640 | -200 | - |

### add4_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 25060 | 7610 | x1.00 |

### add_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_nn` | 19060 | 1620 | x1.00 |
| `signmag_pp` | 19240 | 1800 | x1.11 |
| `signmag_pn` | 19880 | 2440 | x1.51 |

### cross3_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 42320 | 19960 | x1.00 |

### div_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 19690 | 2250 | x1.00 |
| `signmag_pn` | 19700 | 2260 | x1.00 |

### dot3_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 30160 | 10310 | x1.00 |

### eq_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 17650 | 900 | x1.00 |

### from_int_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 18860 | 2410 | x1.00 |

### length3_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 28870 | 10830 | x1.00 |

### lerp_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 25180 | 7140 | x1.00 |

### lt_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_np` | 17930 | 1080 | x1.00 |
| `signmag_nn` | 18930 | 2080 | x1.93 |
| `signmag_pp` | 19030 | 2180 | x2.02 |

### mul4_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 26140 | 8700 | x1.00 |

### mul_add_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 22550 | 4500 | x1.00 |

### mul_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 19690 | 2250 | x1.00 |
| `signmag_pn` | 19700 | 2260 | x1.00 |
| `signmag_nn` | 19790 | 2350 | x1.04 |

### neg_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 17350 | 600 | x1.00 |

### sqrt_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 18020 | 1180 | x1.00 |

### sub_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 19640 | 2200 | x1.00 |
| `signmag_pp` | 20280 | 2840 | x1.29 |

### to_int_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 18220 | 2280 | x1.00 |
| `signmag_n` | 18590 | 2650 | x1.16 |

