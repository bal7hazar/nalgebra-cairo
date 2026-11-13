# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## libs::bench_alexandria::integer

### isqrt_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `core_sqrt` | 9590 | 1180 | x1.00 |
| `alexandria_fast_sqrt_40_iterations` | 749180 | 740770 | x627.77 |

### pow2_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_const_pow2_lut` | 10450 | 1540 | x1.00 |
| `core_pow` | 22860 | 13950 | x9.06 |
| `alexandria_pow` | 26320 | 17410 | x11.31 |
| `alexandria_fast_power` | 100030 | 91120 | x59.17 |

### shl_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_const_mul` | 9380 | 470 | x1.00 |
| `alexandria_opt_bitshift_lut` | 13873 | 4963 | x10.56 |
| `alexandria_bitshift` | 29073 | 20163 | x42.90 |

### shr_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_const_div` | 9920 | 1010 | x1.00 |
| `alexandria_opt_bitshift_lut` | 12230 | 3320 | x3.29 |
| `alexandria_bitshift` | 27530 | 18620 | x18.44 |

## libs::bench_alexandria::linalg

### dot3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_unrolled` | 14160 | 2150 | x1.00 |
| `origami_vector` | 23310 | 11300 | x5.26 |
| `alexandria_dot` | 23810 | 11800 | x5.49 |

### kron3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_unrolled` | 16040 | 3460 | x1.00 |
| `alexandria_kron` | 47750 | 35170 | x10.16 |

### norm3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_unrolled` | 15340 | 3330 | x1.00 |
| `alexandria_norm` | 227830 | 215820 | x64.81 |

## libs::bench_alexandria::wad

### div_wad

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_wad_div` | 39740 | 29730 | x1.00 |

### mul_wad

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_wad_mul` | 32710 | 22700 | x1.00 |

## libs::bench_mat

### det3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 140280 | 19570 | x1.00 |
| `origami_matrix_of_reference` | 378410 | 257700 | x13.17 |
| `origami_matrix_of_cubit` | 400430 | 279720 | x14.29 |

### inv3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 303990 | 98490 | x1.00 |
| `origami_matrix_of_reference` | 1247080 | 1041580 | x10.58 |
| `origami_matrix_of_cubit` | 1281600 | 1076100 | x10.93 |

### matadd3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 219450 | 13950 | x1.00 |
| `origami_matrix_of_reference` | 285350 | 79850 | x5.72 |
| `origami_matrix_of_cubit` | 315830 | 110330 | x7.91 |
| `orion_tensor_fp32x32` | 999710 | 794210 | x56.93 |

### matmul3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 266340 | 60840 | x1.00 |
| `reference` | 323220 | 117720 | x1.93 |
| `origami_matrix_of_reference` | 507640 | 302140 | x4.97 |
| `orion_tensor_fp32x32` | 554540 | 349040 | x5.74 |
| `origami_matrix_of_cubit` | 564510 | 359010 | x5.90 |

### transpose3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 205800 | 300 | x1.00 |
| `origami_matrix_of_cubit` | 262810 | 57310 | x191.03 |

## libs::bench_mat::matvec

### matvec3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 280570 | 20300 | x1.00 |
| `orion_mutmatrix_fp32x32` | 487050 | 226780 | x11.17 |

## libs::bench_mat::q16

### matmul3_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 234530 | 61470 | x1.00 |
| `orion_tensor_fp16x16` | 509210 | 336150 | x5.47 |

## libs::bench_q16

### add_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 13350 | 740 | x1.00 |
| `orion_fp16x16` | 16660 | 4050 | x5.47 |

### div_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 13890 | 1280 | x1.00 |
| `reference` | 17570 | 4960 | x3.88 |

### lt_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 11690 | 680 | x1.00 |
| `reference` | 11880 | 870 | x1.28 |

### mul_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 13890 | 1280 | x1.00 |
| `reference` | 15230 | 2620 | x2.05 |

### sqrt_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 14960 | 2350 | x1.00 |

## libs::bench_q32

### add_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 14950 | 740 | x1.00 |
| `cubit` | 18260 | 4050 | x5.47 |
| `cubit_same_sign` | 18260 | 4050 | x5.47 |
| `orion_fp32x32` | 18260 | 4050 | x5.47 |

### div_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 15960 | 1750 | x1.00 |
| `orion_fp32x32` | 15960 | 1750 | x1.00 |
| `reference` | 19640 | 5430 | x3.10 |

### eq_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 13210 | 600 | x1.00 |
| `cubit` | 13220 | 610 | x1.02 |

### lt_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 13290 | 680 | x1.00 |
| `reference` | 13480 | 870 | x1.28 |
| `reference_same_sign` | 13480 | 870 | x1.28 |
| `cubit_same_sign` | 14490 | 1880 | x2.76 |

### mul_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 15960 | 1750 | x1.00 |
| `orion_fp32x32` | 15960 | 1750 | x1.00 |
| `reference_bounded` | 16160 | 1950 | x1.11 |
| `reference` | 17300 | 3090 | x1.77 |

### neg_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 14410 | 200 | x1.00 |
| `cubit` | 14510 | 300 | x1.50 |

### sqrt_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 16130 | 1920 | x1.00 |
| `cubit` | 19320 | 5110 | x2.66 |
| `orion_fp32x32` | 19320 | 5110 | x2.66 |

### sub_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 14950 | 740 | x1.00 |
| `cubit` | 18770 | 4560 | x6.16 |
| `cubit_same_sign` | 18860 | 4650 | x6.28 |

## libs::bench_q64

### add_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 13080 | 470 | x1.00 |
| `cubit` | 14850 | 2240 | x4.77 |

### div_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 24190 | 11580 | x1.00 |
| `reference` | 28450 | 15840 | x1.37 |

### lt_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 11690 | 680 | x1.00 |
| `reference` | 11880 | 870 | x1.28 |

### mul_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 23890 | 11280 | x1.00 |
| `orion_fp64x64` | 23890 | 11280 | x1.00 |
| `reference` | 26780 | 14170 | x1.26 |

### sqrt_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 21490 | 8880 | x1.00 |
| `reference` | 21970 | 9360 | x1.05 |

## libs::bench_trig::dec8

### sin_dec8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_fast_sin` | 33360 | 23170 | x1.00 |

## libs::bench_trig::q16

### sin_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 35510 | 25290 | x1.00 |
| `orion_fp16x16_fast` | 35510 | 25290 | x1.00 |
| `orion_fp16x16_taylor` | 128070 | 117850 | x4.66 |

## libs::bench_trig::q32

### acos_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 102310 | 88610 | x1.00 |
| `cubit` | 128760 | 115060 | x1.30 |

### atan_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 70990 | 57290 | x1.00 |
| `cubit` | 97440 | 83740 | x1.46 |

### cos_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 48770 | 35070 | x1.00 |
| `cubit` | 148920 | 135220 | x3.86 |

### exp_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 78570 | 64870 | x1.00 |

### ln_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 83960 | 70260 | x1.00 |

### sin_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 44020 | 30320 | x1.00 |
| `reference_poly` | 47000 | 33300 | x1.10 |
| `cubit` | 144170 | 130470 | x4.30 |

### tan_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 83040 | 69340 | x1.00 |
| `cubit` | 282840 | 269140 | x3.88 |

## libs::bench_trig::q64

### sin_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 62820 | 52600 | x1.00 |
| `cubit` | 470550 | 460330 | x8.75 |

## libs::bench_vec

### add3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 38880 | 4550 | x1.00 |
| `cubit_vec3` | 46480 | 12150 | x2.67 |
| `origami_vector_of_cubit` | 60340 | 26010 | x5.72 |

### build_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_vec3` | 10010 | 0 | - |
| `origami_vector` | 11210 | 1200 | - |
| `orion_tensor` | 16320 | 6310 | - |
| `orion_tensor_build_and_read` | 23640 | 13630 | - |

### cross3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 51060 | 13210 | x1.00 |
| `reference` | 60340 | 22490 | x1.70 |
| `cubit_vec3` | 66680 | 28830 | x2.18 |

### dot2_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 39920 | 5590 | x1.00 |
| `cubit_vec2` | 42180 | 7850 | x1.40 |
| `origami_vec2_of_cubit` | 42180 | 7850 | x1.40 |

### dot3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_bounded` | 36680 | 2350 | x1.00 |
| `reference_fused` | 40690 | 6360 | x2.71 |
| `reference` | 47010 | 12680 | x5.40 |
| `cubit_vec3` | 49780 | 15450 | x6.57 |
| `origami_vector_of_reference` | 54280 | 19950 | x8.49 |
| `alexandria_dot_of_reference` | 54780 | 20450 | x8.70 |
| `origami_vector_of_cubit` | 62810 | 28480 | x12.12 |
| `alexandria_dot_of_cubit` | 63210 | 28880 | x12.29 |
| `orion_tensor_fp32x32` | 72150 | 37820 | x16.09 |

### norm3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 39690 | 5360 | x1.00 |
| `reference` | 48930 | 14600 | x2.72 |
| `cubit_vec3` | 55190 | 20860 | x3.89 |

