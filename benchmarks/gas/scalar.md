# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## scalar::tests::algo

### acos_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sqrt_poly9_typed_m0p3` | 24620 | 16510 | x1.00 |
| `sqrt_poly7_m0p3` | 31480 | 23370 | x1.42 |
| `via_atan2_m0p3` | 44300 | 36190 | x2.19 |

### atan2_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `poly15_typed_small` | 24830 | 16320 | x1.00 |
| `poly15_typed_swap_neg` | 24830 | 16320 | x1.00 |
| `poly19_typed_small` | 27350 | 18840 | x1.15 |
| `poly19_typed_swap_neg` | 27350 | 18840 | x1.15 |
| `poly15_small` | 39050 | 30540 | x1.87 |
| `poly15_swap_neg` | 39050 | 30540 | x1.87 |
| `reduced_poly9_small` | 39930 | 31420 | x1.93 |
| `reduced_poly9_swap_neg` | 39930 | 31420 | x1.93 |

### cos_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `poly11_typed_m2p5` | 20710 | 12600 | x1.00 |
| `poly9_fused_m2p5` | 25340 | 17230 | x1.37 |

### inv_sqrt_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sqrt_then_const_udiv_1234` | 11040 | 2930 | x1.00 |
| `sqrt_then_signed_div_1234` | 12520 | 4410 | x1.51 |
| `newton_mul_only_1234` | 56620 | 48510 | x16.56 |

### normalize3_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `length_inv96_3mul_mix` | 19660 | 9750 | x1.00 |
| `length_inv32_3mul_mix` | 20030 | 10120 | x1.04 |
| `length_3div_mix` | 23610 | 13700 | x1.41 |

### sin_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `poly9_typed_0p5` | 19300 | 11190 | x1.00 |
| `poly9_typed_m2p5` | 19300 | 11190 | x1.00 |
| `lut256_array_0p5` | 20290 | 12180 | x1.09 |
| `lut256_array_m2p5` | 20290 | 12180 | x1.09 |
| `lut64_array_0p5` | 20290 | 12180 | x1.09 |
| `lut64_array_m2p5` | 20290 | 12180 | x1.09 |
| `poly11_typed_0p5` | 20610 | 12500 | x1.12 |
| `poly11_typed_m2p5` | 20610 | 12500 | x1.12 |
| `lut64_match_0p5` | 21570 | 13460 | x1.20 |
| `lut64_match_m2p5` | 21570 | 13460 | x1.20 |
| `poly7_fused_0p5` | 23190 | 15080 | x1.35 |
| `poly7_fused_m2p5` | 23190 | 15080 | x1.35 |
| `poly9_fused_0p5` | 25240 | 17130 | x1.53 |
| `poly9_fused_m2p5` | 25240 | 17130 | x1.53 |
| `poly11_fused_0p5` | 27290 | 19180 | x1.71 |
| `poly11_fused_m2p5` | 27290 | 19180 | x1.71 |
| `poly9_unfused_0p5` | 28200 | 20090 | x1.80 |
| `poly9_unfused_m2p5` | 28200 | 20090 | x1.80 |
| `taylor_loop_cubit_style_0p5` | 90670 | 82560 | x7.38 |
| `taylor_loop_cubit_style_m2p5` | 91250 | 83140 | x7.43 |
| `cordic20_unrolled_0p5` | 146750 | 138640 | x12.39 |
| `cordic20_unrolled_m2p5` | 147120 | 139010 | x12.42 |

### sqrt_algo

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `corelib_0p02` | 10030 | 1920 | x1.00 |
| `corelib_1234` | 10030 | 1920 | x1.00 |
| `newton_seeded_unrolled_0p02` | 39250 | 31140 | x16.22 |
| `newton_seeded_unrolled_1234` | 39250 | 31140 | x16.22 |
| `via_inv_sqrt_newton_0p02` | 58570 | 50460 | x26.28 |
| `via_inv_sqrt_newton_1234` | 58570 | 50460 | x26.28 |
| `newton_loop_0p02` | 173310 | 165200 | x86.04 |
| `newton_loop_1234` | 212270 | 204160 | x106.33 |

## scalar::tests::rep_felt

### abs_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 9600 | 1590 | x1.00 |
| `felt_p` | 9920 | 1910 | x1.20 |

### add4_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pn` | 8810 | 400 | x1.00 |

### add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_nn` | 8510 | 100 | x1.00 |
| `felt_pn` | 8510 | 100 | x1.00 |
| `felt_pp` | 8510 | 100 | x1.00 |

### cross3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_lazy_mix` | 17970 | 7160 | x1.00 |
| `felt_mix` | 24330 | 13520 | x1.89 |

### div_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pp` | 13260 | 4850 | x1.00 |
| `felt_pn` | 13330 | 4920 | x1.01 |

### dot3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_lazy_mix` | 12530 | 2520 | x1.00 |
| `felt_mix` | 16770 | 6760 | x2.68 |

### eq_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_p` | 9020 | 500 | x1.00 |

### from_int_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 8110 | 100 | x1.00 |

### length3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_mix` | 17390 | 8580 | x1.00 |

### lerp_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_mix` | 11130 | 2320 | x1.00 |
| `felt_lazy_mix` | 11230 | 2420 | x1.04 |

### lt_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_nn` | 10310 | 1790 | x1.00 |
| `felt_np` | 10310 | 1790 | x1.00 |
| `felt_pp` | 10310 | 1790 | x1.00 |

### mul4_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pn` | 17190 | 8780 | x1.00 |

### mul_add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_mix` | 11030 | 2220 | x1.00 |
| `felt_lazy_mix` | 11130 | 2320 | x1.05 |

### mul_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_soft_nn` | 9890 | 1480 | x1.00 |
| `felt_soft_pn` | 9890 | 1480 | x1.00 |
| `felt_soft_pp` | 9890 | 1480 | x1.00 |
| `felt_nn` | 10530 | 2120 | x1.43 |
| `felt_pn` | 10530 | 2120 | x1.43 |
| `felt_pp` | 10530 | 2120 | x1.43 |
| `felt_plain_nn` | 11000 | 2590 | x1.75 |
| `felt_plain_pn` | 11000 | 2590 | x1.75 |
| `felt_plain_pp` | 11000 | 2590 | x1.75 |

### neg_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_p` | 8110 | 100 | x1.00 |

### sqrt_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_p` | 9730 | 1720 | x1.00 |

### sub_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_pn` | 8510 | 100 | x1.00 |
| `felt_pp` | 8510 | 100 | x1.00 |

### to_int_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `felt_n` | 9760 | 1650 | x1.00 |
| `felt_p` | 9760 | 1650 | x1.00 |

## scalar::tests::rep_i128

### abs_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 9000 | 890 | x1.00 |
| `native_n` | 9180 | 1070 | x1.20 |

### add4_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 10290 | 1780 | x1.00 |

### add_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 8880 | 370 | x1.00 |
| `native_pn` | 8880 | 370 | x1.00 |
| `native_pp` | 8880 | 370 | x1.00 |

### cross3_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 64960 | 53850 | x1.00 |

### div_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pp` | 22010 | 13500 | x1.00 |
| `native_pn` | 22270 | 13760 | x1.02 |

### dot3_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_lazy_mix` | 23270 | 13160 | x1.00 |
| `native_mix` | 37060 | 26950 | x2.05 |

### eq_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 9020 | 500 | x1.00 |

### from_int_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_n` | 8480 | 370 | x1.00 |

### length3_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 43340 | 34430 | x1.00 |

### lerp_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 18540 | 9630 | x1.00 |

### lt_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 9390 | 870 | x1.00 |
| `native_np` | 9390 | 870 | x1.00 |
| `native_pp` | 9390 | 870 | x1.00 |

### mul4_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 44130 | 35620 | x1.00 |

### mul_add_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 18070 | 9160 | x1.00 |

### mul_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_nn` | 13010 | 4500 | x1.00 |
| `bounded_pn` | 13010 | 4500 | x1.00 |
| `bounded_pp` | 13010 | 4500 | x1.00 |
| `native_pp` | 16940 | 8430 | x1.87 |
| `native_pn` | 17200 | 8690 | x1.93 |
| `native_nn` | 17480 | 8970 | x1.99 |

### neg_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 8410 | 300 | x1.00 |

### sqrt_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 15470 | 7360 | x1.00 |

### sub_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 8880 | 370 | x1.00 |
| `native_pp` | 8880 | 370 | x1.00 |

### to_int_i128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 12010 | 3900 | x1.00 |
| `native_n` | 12270 | 4160 | x1.07 |

## scalar::tests::rep_i32

### abs_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 9000 | 890 | x1.00 |
| `native_n` | 9180 | 1070 | x1.20 |

### add4_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 11370 | 2860 | x1.00 |

### add_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 9150 | 640 | x1.00 |
| `native_pn` | 9150 | 640 | x1.00 |
| `native_pp` | 9150 | 640 | x1.00 |

### cross3_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 28930 | 17820 | x1.00 |

### div_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 13370 | 4860 | x1.00 |
| `native_pp` | 13370 | 4860 | x1.00 |

### dot3_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 19270 | 9160 | x1.00 |

### eq_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 9020 | 500 | x1.00 |

### from_int_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_n` | 8750 | 640 | x1.00 |

### length3_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 19830 | 10920 | x1.00 |

### lerp_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 12910 | 4000 | x1.00 |

### lt_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 9390 | 870 | x1.00 |
| `native_np` | 9390 | 870 | x1.00 |
| `native_pp` | 9390 | 870 | x1.00 |

### mul4_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 18730 | 10220 | x1.00 |

### mul_add_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_mix` | 12170 | 3260 | x1.00 |

### mul_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_nn` | 10950 | 2440 | x1.00 |
| `native_pp` | 10950 | 2440 | x1.00 |
| `native_pn` | 11030 | 2520 | x1.03 |

### neg_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 8410 | 300 | x1.00 |

### sqrt_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 9930 | 1820 | x1.00 |

### sub_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_pn` | 9150 | 640 | x1.00 |
| `native_pp` | 9150 | 640 | x1.00 |

### to_int_i32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 11540 | 3430 | x1.00 |
| `native_n` | 11800 | 3690 | x1.08 |

## scalar::tests::rep_i64

### abs_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_bi_p` | 8820 | 710 | x1.00 |
| `bounded_p` | 9000 | 890 | x1.25 |
| `native_p` | 9000 | 890 | x1.25 |
| `bounded_n` | 9180 | 1070 | x1.51 |
| `native_n` | 9180 | 1070 | x1.51 |
| `bounded_bi_n` | 9250 | 1140 | x1.61 |

### add4_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_pn` | 11370 | 2860 | x1.00 |
| `native_pn` | 11370 | 2860 | x1.00 |

### add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_bi_nn` | 9150 | 640 | x1.00 |
| `bounded_bi_pn` | 9150 | 640 | x1.00 |
| `bounded_bi_pp` | 9150 | 640 | x1.00 |
| `bounded_nn` | 9150 | 640 | x1.00 |
| `bounded_pn` | 9150 | 640 | x1.00 |
| `bounded_pp` | 9150 | 640 | x1.00 |
| `native_nn` | 9150 | 640 | x1.00 |
| `native_pn` | 9150 | 640 | x1.00 |
| `native_pp` | 9150 | 640 | x1.00 |

### cross3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 17760 | 6650 | x1.00 |
| `bounded_mix` | 25230 | 14120 | x2.12 |
| `native_mix` | 31750 | 20640 | x3.10 |

### div_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_pp` | 12080 | 3570 | x1.00 |
| `bounded_pn` | 12150 | 3640 | x1.02 |
| `native_pn` | 13840 | 5330 | x1.49 |
| `native_pp` | 13840 | 5330 | x1.49 |

### dot3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 12360 | 2250 | x1.00 |
| `bounded_mix` | 17340 | 7230 | x3.21 |
| `native_mix` | 20680 | 10570 | x4.70 |

### eq_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_p` | 9020 | 500 | x1.00 |
| `native_p` | 9020 | 500 | x1.00 |

### from_int_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_n` | 8210 | 100 | x1.00 |
| `native_n` | 8750 | 640 | x6.40 |

### length3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 11130 | 2220 | x1.00 |
| `bounded_mix` | 18160 | 9250 | x4.17 |
| `native_mix` | 21240 | 12330 | x5.55 |

### lerp_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 11060 | 2150 | x1.00 |
| `bounded_mix` | 12240 | 3330 | x1.55 |
| `native_mix` | 13380 | 4470 | x2.08 |

### lt_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_nn` | 9390 | 870 | x1.00 |
| `bounded_np` | 9390 | 870 | x1.00 |
| `bounded_pp` | 9390 | 870 | x1.00 |
| `native_nn` | 9390 | 870 | x1.00 |
| `native_np` | 9390 | 870 | x1.00 |
| `native_pp` | 9390 | 870 | x1.00 |

### mul4_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_pn` | 16210 | 7700 | x1.00 |
| `native_pn` | 20610 | 12100 | x1.57 |

### mul_add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_fused_mix` | 10960 | 2050 | x1.00 |
| `bounded_mix` | 11500 | 2590 | x1.26 |
| `native_mix` | 12640 | 3730 | x1.82 |

### mul_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_precheck_nn` | 10260 | 1750 | x1.00 |
| `bounded_precheck_pn` | 10260 | 1750 | x1.00 |
| `bounded_precheck_pp` | 10260 | 1750 | x1.00 |
| `bounded_nn` | 10360 | 1850 | x1.06 |
| `bounded_pn` | 10360 | 1850 | x1.06 |
| `bounded_pp` | 10360 | 1850 | x1.06 |
| `bounded_u128_nn` | 10360 | 1850 | x1.06 |
| `bounded_u128_pn` | 10360 | 1850 | x1.06 |
| `bounded_u128_pp` | 10360 | 1850 | x1.06 |
| `bounded_trunc_nn` | 10950 | 2440 | x1.39 |
| `bounded_trunc_pp` | 10950 | 2440 | x1.39 |
| `bounded_trunc_pn` | 11030 | 2520 | x1.44 |
| `native_nn` | 11420 | 2910 | x1.66 |
| `native_pp` | 11420 | 2910 | x1.66 |
| `native_pn` | 11500 | 2990 | x1.71 |

### neg_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_p` | 8410 | 300 | x1.00 |
| `native_p` | 8410 | 300 | x1.00 |
| `bounded_bi_p` | 8480 | 370 | x1.23 |

### sqrt_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `native_p` | 9930 | 1820 | x1.00 |
| `bounded_p` | 10030 | 1920 | x1.05 |

### sub_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_bi_pn` | 9150 | 640 | x1.00 |
| `bounded_bi_pp` | 9150 | 640 | x1.00 |
| `bounded_pn` | 9150 | 640 | x1.00 |
| `bounded_pp` | 9150 | 640 | x1.00 |
| `native_pn` | 9150 | 640 | x1.00 |
| `native_pp` | 9150 | 640 | x1.00 |

### to_int_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `bounded_n` | 9220 | 1110 | x1.00 |
| `bounded_p` | 9220 | 1110 | x1.00 |
| `native_p` | 11540 | 3430 | x3.09 |
| `native_n` | 11800 | 3690 | x3.32 |

## scalar::tests::rep_sm32

### abs_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 8810 | -200 | - |
| `signmag_p` | 8810 | -200 | - |

### add4_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 17230 | 7610 | x1.00 |

### add_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_nn` | 11230 | 1620 | x1.00 |
| `signmag_pp` | 11410 | 1800 | x1.11 |
| `signmag_pn` | 12050 | 2440 | x1.51 |

### cross3_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 31670 | 17140 | x1.00 |

### div_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 11390 | 1780 | x1.00 |
| `signmag_pn` | 11400 | 1790 | x1.01 |

### dot3_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 20920 | 8900 | x1.00 |

### eq_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 9820 | 900 | x1.00 |

### from_int_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 11030 | 2410 | x1.00 |

### length3_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 19630 | 9420 | x1.00 |

### lerp_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 16880 | 6670 | x1.00 |

### lt_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_np` | 10100 | 1080 | x1.00 |
| `signmag_nn` | 11100 | 2080 | x1.93 |
| `signmag_pp` | 11200 | 2180 | x2.02 |

### mul4_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 16430 | 6820 | x1.00 |

### mul_add_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 14250 | 4030 | x1.00 |

### mul_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 11390 | 1780 | x1.00 |
| `signmag_pn` | 11400 | 1790 | x1.01 |
| `signmag_nn` | 11490 | 1880 | x1.06 |

### neg_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 9520 | 600 | x1.00 |

### sqrt_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 10190 | 1180 | x1.00 |

### sub_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 11810 | 2200 | x1.00 |
| `signmag_pp` | 12450 | 2840 | x1.29 |

### to_int_sm32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 10390 | 2280 | x1.00 |
| `signmag_n` | 10760 | 2650 | x1.16 |

## scalar::tests::rep_sm64

### abs_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 8810 | -200 | - |
| `signmag_p` | 8810 | -200 | - |

### add4_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 17230 | 7610 | x1.00 |

### add_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_nn` | 11230 | 1620 | x1.00 |
| `signmag_pp` | 11410 | 1800 | x1.11 |
| `signmag_pn` | 12050 | 2440 | x1.51 |

### cross3_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 34490 | 19960 | x1.00 |

### div_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 11860 | 2250 | x1.00 |
| `signmag_pn` | 11870 | 2260 | x1.00 |

### dot3_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 22330 | 10310 | x1.00 |

### eq_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 9820 | 900 | x1.00 |

### from_int_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_n` | 11030 | 2410 | x1.00 |

### length3_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 21040 | 10830 | x1.00 |

### lerp_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 17350 | 7140 | x1.00 |

### lt_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_np` | 10100 | 1080 | x1.00 |
| `signmag_nn` | 11100 | 2080 | x1.93 |
| `signmag_pp` | 11200 | 2180 | x2.02 |

### mul4_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 18310 | 8700 | x1.00 |

### mul_add_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_mix` | 14720 | 4500 | x1.00 |

### mul_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pp` | 11860 | 2250 | x1.00 |
| `signmag_pn` | 11870 | 2260 | x1.00 |
| `signmag_nn` | 11960 | 2350 | x1.04 |

### neg_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 9520 | 600 | x1.00 |

### sqrt_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 10190 | 1180 | x1.00 |

### sub_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_pn` | 11810 | 2200 | x1.00 |
| `signmag_pp` | 12450 | 2840 | x1.29 |

### to_int_sm64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `signmag_p` | 10390 | 2280 | x1.00 |
| `signmag_n` | 10760 | 2650 | x1.16 |

