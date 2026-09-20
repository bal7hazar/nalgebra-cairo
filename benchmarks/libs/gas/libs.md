# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## libs::bench_alexandria::integer

### isqrt_u128

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `core_sqrt` | 17420 | 1180 | x1.00 |
| `alexandria_fast_sqrt_40_iterations` | 757010 | 740770 | x627.77 |

### pow2_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_const_pow2_lut` | 18280 | 1540 | x1.00 |
| `core_pow` | 30690 | 13950 | x9.06 |
| `alexandria_pow` | 34150 | 17410 | x11.31 |
| `alexandria_fast_power` | 107860 | 91120 | x59.17 |

### shl_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_const_mul` | 17210 | 470 | x1.00 |
| `alexandria_opt_bitshift_lut` | 21603 | 4863 | x10.35 |
| `alexandria_bitshift` | 36803 | 20063 | x42.69 |

### shr_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_const_div` | 17750 | 1010 | x1.00 |
| `alexandria_opt_bitshift_lut` | 20060 | 3320 | x3.29 |
| `alexandria_bitshift` | 35360 | 18620 | x18.44 |

## libs::bench_alexandria::linalg

### dot3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_unrolled` | 21990 | 2150 | x1.00 |
| `origami_vector` | 31140 | 11300 | x5.26 |
| `alexandria_dot` | 31640 | 11800 | x5.49 |

### kron3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_unrolled` | 23870 | 3460 | x1.00 |
| `alexandria_kron` | 55580 | 35170 | x10.16 |

### norm3_u64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_unrolled` | 23170 | 3330 | x1.00 |
| `alexandria_norm` | 235660 | 215820 | x64.81 |

## libs::bench_alexandria::wad

### div_wad

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_wad_div` | 47570 | 29730 | x1.00 |

### mul_wad

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_wad_mul` | 40540 | 22700 | x1.00 |

## libs::bench_mat

### det3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 148110 | 19570 | x1.00 |
| `origami_matrix_of_reference` | 386240 | 257700 | x13.17 |
| `origami_matrix_of_cubit` | 408260 | 279720 | x14.29 |

### inv3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 311820 | 98490 | x1.00 |
| `origami_matrix_of_reference` | 1254910 | 1041580 | x10.58 |
| `origami_matrix_of_cubit` | 1289430 | 1076100 | x10.93 |

### matadd3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 227280 | 13950 | x1.00 |
| `origami_matrix_of_reference` | 293180 | 79850 | x5.72 |
| `origami_matrix_of_cubit` | 323660 | 110330 | x7.91 |
| `orion_tensor_fp32x32` | 1007540 | 794210 | x56.93 |

### matmul3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 274170 | 60840 | x1.00 |
| `reference` | 331050 | 117720 | x1.93 |
| `origami_matrix_of_reference` | 515470 | 302140 | x4.97 |
| `orion_tensor_fp32x32` | 562370 | 349040 | x5.74 |
| `origami_matrix_of_cubit` | 572340 | 359010 | x5.90 |

### transpose3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 213630 | 300 | x1.00 |
| `origami_matrix_of_cubit` | 270640 | 57310 | x191.03 |

## libs::bench_mat::matvec

### matvec3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 288300 | 20300 | x1.00 |
| `orion_mutmatrix_fp32x32` | 494780 | 226780 | x11.17 |

## libs::bench_mat::q16

### matmul3_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 242360 | 61470 | x1.00 |
| `orion_tensor_fp16x16` | 517040 | 336150 | x5.47 |

## libs::bench_q16

### add_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 21180 | 740 | x1.00 |
| `orion_fp16x16` | 24490 | 4050 | x5.47 |

### div_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 21720 | 1280 | x1.00 |
| `reference` | 25400 | 4960 | x3.88 |

### lt_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 19520 | 680 | x1.00 |
| `reference` | 19710 | 870 | x1.28 |

### mul_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 21720 | 1280 | x1.00 |
| `reference` | 23060 | 2620 | x2.05 |

### sqrt_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 22790 | 2350 | x1.00 |

## libs::bench_q32

### add_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 22780 | 740 | x1.00 |
| `cubit` | 26090 | 4050 | x5.47 |
| `cubit_same_sign` | 26090 | 4050 | x5.47 |
| `orion_fp32x32` | 26090 | 4050 | x5.47 |

### div_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 23790 | 1750 | x1.00 |
| `orion_fp32x32` | 23790 | 1750 | x1.00 |
| `reference` | 27470 | 5430 | x3.10 |

### eq_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 21040 | 600 | x1.00 |
| `cubit` | 21050 | 610 | x1.02 |

### lt_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 21120 | 680 | x1.00 |
| `reference` | 21310 | 870 | x1.28 |
| `reference_same_sign` | 21310 | 870 | x1.28 |
| `cubit_same_sign` | 22320 | 1880 | x2.76 |

### mul_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 23790 | 1750 | x1.00 |
| `orion_fp32x32` | 23790 | 1750 | x1.00 |
| `reference_bounded` | 23990 | 1950 | x1.11 |
| `reference` | 25130 | 3090 | x1.77 |

### neg_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 22240 | 200 | x1.00 |
| `cubit` | 22340 | 300 | x1.50 |

### sqrt_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 23960 | 1920 | x1.00 |
| `cubit` | 27150 | 5110 | x2.66 |
| `orion_fp32x32` | 27150 | 5110 | x2.66 |

### sub_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 22780 | 740 | x1.00 |
| `cubit` | 26600 | 4560 | x6.16 |
| `cubit_same_sign` | 26690 | 4650 | x6.28 |

## libs::bench_q64

### add_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 20910 | 470 | x1.00 |
| `cubit` | 22680 | 2240 | x4.77 |

### div_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 32020 | 11580 | x1.00 |
| `reference` | 36280 | 15840 | x1.37 |

### lt_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 19520 | 680 | x1.00 |
| `reference` | 19710 | 870 | x1.28 |

### mul_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 31720 | 11280 | x1.00 |
| `orion_fp64x64` | 31720 | 11280 | x1.00 |
| `reference` | 34610 | 14170 | x1.26 |

### sqrt_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 29320 | 8880 | x1.00 |
| `reference` | 29800 | 9360 | x1.05 |

## libs::bench_trig::dec8

### sin_dec8

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alexandria_fast_sin` | 41190 | 23170 | x1.00 |

## libs::bench_trig::q16

### sin_q16

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orion_fp16x16` | 43340 | 25290 | x1.00 |
| `orion_fp16x16_fast` | 43340 | 25290 | x1.00 |
| `orion_fp16x16_taylor` | 135900 | 117850 | x4.66 |

## libs::bench_trig::q32

### acos_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 110140 | 88610 | x1.00 |
| `cubit` | 136590 | 115060 | x1.30 |

### atan_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 78820 | 57290 | x1.00 |
| `cubit` | 105270 | 83740 | x1.46 |

### cos_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 56600 | 35070 | x1.00 |
| `cubit` | 156750 | 135220 | x3.86 |

### exp_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 86400 | 64870 | x1.00 |

### ln_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit` | 91790 | 70260 | x1.00 |

### sin_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 51850 | 30320 | x1.00 |
| `reference_poly` | 54830 | 33300 | x1.10 |
| `cubit` | 152000 | 130470 | x4.30 |

### tan_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 90870 | 69340 | x1.00 |
| `cubit` | 290670 | 269140 | x3.88 |

## libs::bench_trig::q64

### sin_q64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_fast` | 70650 | 52600 | x1.00 |
| `cubit` | 478380 | 460330 | x8.75 |

## libs::bench_vec

### add3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference` | 46710 | 4550 | x1.00 |
| `cubit_vec3` | 54310 | 12150 | x2.67 |
| `origami_vector_of_cubit` | 68170 | 26010 | x5.72 |

### build_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cubit_vec3` | 17840 | 0 | - |
| `origami_vector` | 19040 | 1200 | - |
| `orion_tensor` | 24150 | 6310 | - |
| `orion_tensor_build_and_read` | 31470 | 13630 | - |

### cross3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 58890 | 13210 | x1.00 |
| `reference` | 68170 | 22490 | x1.70 |
| `cubit_vec3` | 74510 | 28830 | x2.18 |

### dot2_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 47750 | 5590 | x1.00 |
| `cubit_vec2` | 50010 | 7850 | x1.40 |
| `origami_vec2_of_cubit` | 50010 | 7850 | x1.40 |

### dot3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_bounded` | 44510 | 2350 | x1.00 |
| `reference_fused` | 48520 | 6360 | x2.71 |
| `reference` | 54840 | 12680 | x5.40 |
| `cubit_vec3` | 57610 | 15450 | x6.57 |
| `origami_vector_of_reference` | 62110 | 19950 | x8.49 |
| `alexandria_dot_of_reference` | 62610 | 20450 | x8.70 |
| `origami_vector_of_cubit` | 70640 | 28480 | x12.12 |
| `alexandria_dot_of_cubit` | 71040 | 28880 | x12.29 |
| `orion_tensor_fp32x32` | 79980 | 37820 | x16.09 |

### norm3_q32

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reference_fused` | 47520 | 5360 | x1.00 |
| `reference` | 56760 | 14600 | x2.72 |
| `cubit_vec3` | 63020 | 20860 | x3.89 |

