# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## layout::abstraction::tests

### call_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator_inline_always` | 20260 | 2120 | x1.00 |
| `trait_method_inline_always` | 20260 | 2120 | x1.00 |
| `free_fn` | 22590 | 4450 | x2.10 |
| `free_fn_generic` | 22590 | 4450 | x2.10 |
| `operator` | 22590 | 4450 | x2.10 |
| `trait_method` | 22590 | 4450 | x2.10 |

### call_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `free_fn` | 52170 | 34630 | x1.00 |
| `free_fn_generic` | 52170 | 34630 | x1.00 |
| `trait_method` | 52170 | 34630 | x1.00 |

## layout::abstraction_gen

### abs_vec3_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 20960 | 2220 | x1.00 |
| `generic_inline_always` | 20960 | 2220 | x1.00 |
| `concrete_default` | 23290 | 4550 | x2.05 |
| `concrete_inline_never` | 23290 | 4550 | x2.05 |
| `generic_default` | 23290 | 4550 | x2.05 |
| `generic_inline_never` | 23290 | 4550 | x2.05 |

### abs_vec3_chain_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 151320 | 132580 | x1.00 |
| `generic_inline_always` | 151320 | 132580 | x1.00 |
| `concrete_default` | 162110 | 143370 | x1.08 |
| `concrete_inline_never` | 162110 | 143370 | x1.08 |
| `generic_default` | 162110 | 143370 | x1.08 |
| `generic_inline_never` | 162110 | 143370 | x1.08 |

### abs_vec3_cross_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 83280 | 64540 | x1.00 |
| `generic_inline_always` | 83280 | 64540 | x1.00 |
| `concrete_default` | 85330 | 66590 | x1.03 |
| `concrete_inline_never` | 85330 | 66590 | x1.03 |
| `generic_default` | 85330 | 66590 | x1.03 |
| `generic_inline_never` | 85330 | 66590 | x1.03 |

### abs_vec3_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 50860 | 32720 | x1.00 |
| `generic_inline_always` | 50860 | 32720 | x1.00 |
| `concrete_default` | 52870 | 34730 | x1.06 |
| `concrete_inline_never` | 52870 | 34730 | x1.06 |
| `generic_default` | 52870 | 34730 | x1.06 |
| `generic_inline_never` | 52870 | 34730 | x1.06 |

## layout::composite::tests

### inertia_world

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_symmetric` | 311730 | 291190 | x1.00 |
| `two_mat_mat` | 613300 | 592760 | x2.04 |

### iso_compose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quat` | 377230 | 355890 | x1.00 |
| `mat3` | 428510 | 407170 | x1.14 |

### iso_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mat3` | 127200 | 107460 | x1.00 |
| `quat` | 198080 | 178340 | x1.66 |

### m_mt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `symmetric_6_dots` | 229720 | 210380 | x1.00 |
| `transpose_then_mat_mat` | 315870 | 296530 | x1.41 |

### normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_only` | 59190 | 42450 | x1.00 |
| `three_divisions` | 99660 | 82920 | x1.95 |
| `one_division_three_muls` | 101170 | 84430 | x1.99 |

### quat_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hamilton` | 196790 | 175950 | x1.00 |

### quat_rotate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 191730 | 173190 | x1.00 |
| `via_mat3` | 226060 | 207520 | x1.20 |
| `sandwich` | 372940 | 354400 | x2.05 |

### quat_rotate_x4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_mat3_once` | 533650 | 508810 | x1.00 |
| `expanded` | 717900 | 693060 | x1.36 |

### quat_to_mat3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `convert` | 124830 | 106290 | x1.00 |

### skew_mul_mat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross_per_column` | 220210 | 199670 | x1.00 |
| `skew_then_mat_mat` | 317570 | 297030 | x1.49 |

### skew_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross` | 84630 | 66490 | x1.00 |
| `skew_then_mat_vec` | 119070 | 100930 | x1.52 |

### sym3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sym6` | 258970 | 240830 | x1.00 |
| `full9` | 362470 | 344330 | x1.43 |

### sym3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sym6` | 119370 | 100030 | x1.00 |
| `full9` | 119670 | 100330 | x1.00 |

## layout::inline_gen

### inline_mat3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fields_malways` | 370970 | 344830 | x1.00 |
| `fields_mdefault` | 370970 | 344830 | x1.00 |
| `cols_malways` | 380190 | 354050 | x1.03 |
| `cols_mdefault` | 380190 | 354050 | x1.03 |
| `fields_mnever` | 409400 | 383260 | x1.11 |
| `cols_mnever` | 418620 | 392480 | x1.14 |

### inline_mat3_mul_mat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fields_malways` | 323170 | 297030 | x1.00 |
| `fields_mdefault` | 323170 | 297030 | x1.00 |
| `cols_malways` | 355490 | 329350 | x1.11 |
| `cols_mdefault` | 355490 | 329350 | x1.11 |
| `fields_mnever` | 372580 | 346440 | x1.17 |
| `cols_mnever` | 390230 | 364090 | x1.23 |

### inline_mat3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fields_malways` | 125170 | 100430 | x1.00 |
| `fields_mdefault` | 125170 | 100430 | x1.00 |
| `cols_malways` | 133290 | 108550 | x1.08 |
| `cols_mdefault` | 133290 | 108550 | x1.08 |
| `fields_mnever` | 141640 | 116900 | x1.16 |
| `cols_mnever` | 144870 | 120130 | x1.20 |

### inline_vec3_chain

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valways_malways` | 151320 | 132580 | x1.00 |
| `valways_mdefault` | 151320 | 132580 | x1.00 |
| `vdefault_malways` | 162110 | 143370 | x1.08 |
| `vdefault_mdefault` | 162110 | 143370 | x1.08 |
| `vnever_malways` | 162110 | 143370 | x1.08 |
| `vnever_mdefault` | 162110 | 143370 | x1.08 |
| `valways_mnever` | 172720 | 153980 | x1.16 |
| `vdefault_mnever` | 181940 | 163200 | x1.23 |
| `vnever_mnever` | 183240 | 164500 | x1.24 |

### inline_vec3_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valways_malways` | 83280 | 64540 | x1.00 |
| `valways_mdefault` | 83280 | 64540 | x1.00 |
| `vdefault_malways` | 85330 | 66590 | x1.03 |
| `vdefault_mdefault` | 85330 | 66590 | x1.03 |
| `vnever_malways` | 85330 | 66590 | x1.03 |
| `vnever_mdefault` | 85330 | 66590 | x1.03 |
| `valways_mnever` | 93980 | 75240 | x1.17 |
| `vdefault_mnever` | 96310 | 77570 | x1.20 |
| `vnever_mnever` | 96310 | 77570 | x1.20 |

### inline_vec3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valways_malways` | 50860 | 32720 | x1.00 |
| `valways_mdefault` | 50860 | 32720 | x1.00 |
| `vdefault_malways` | 52870 | 34730 | x1.06 |
| `vdefault_mdefault` | 52870 | 34730 | x1.06 |
| `vnever_malways` | 52870 | 34730 | x1.06 |
| `vnever_mdefault` | 52870 | 34730 | x1.06 |
| `valways_mnever` | 56130 | 37990 | x1.16 |
| `vdefault_mnever` | 58360 | 40220 | x1.23 |
| `vnever_mnever` | 58360 | 40220 | x1.23 |

## layout::lazy::tests

### lazy_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_felt_floor` | 29880 | 11740 | x1.00 |
| `lazy_felt` | 31250 | 13110 | x1.12 |
| `lazy_i128` | 75350 | 57210 | x4.87 |
| `eager` | 84630 | 66490 | x5.66 |

### lazy_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_bounded_floor` | 19790 | 2250 | x1.00 |
| `lazy_felt_floor` | 20530 | 2990 | x1.33 |
| `lazy_felt` | 23630 | 6090 | x2.71 |
| `lazy_felt_generic_api` | 23630 | 6090 | x2.71 |
| `lazy_wide_mul` | 23800 | 6260 | x2.78 |
| `eager_bounded_floor` | 27000 | 9460 | x4.20 |
| `eager_wide_mul` | 30120 | 12580 | x5.59 |
| `eager_felt` | 31230 | 13690 | x6.08 |
| `lazy_i128` | 45850 | 28310 | x12.58 |
| `eager` | 52170 | 34630 | x15.39 |

### lazy_mul_mat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_bounded_floor` | 50120 | 25780 | x1.00 |
| `lazy_felt_floor` | 57420 | 33080 | x1.28 |
| `lazy_felt` | 62810 | 38470 | x1.49 |
| `lazy_wide_mul` | 65940 | 41600 | x1.61 |
| `eager_bounded_floor` | 94740 | 70400 | x2.73 |
| `eager_wide_mul` | 122820 | 98480 | x3.82 |
| `dynamic_lazy_felt` | 231370 | 207030 | x8.03 |
| `lazy_i128` | 264390 | 240050 | x9.31 |
| `eager` | 321270 | 296930 | x11.52 |
| `dynamic_eager` | 476420 | 452080 | x17.54 |

### lazy_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_bounded_floor` | 35420 | 10080 | x1.00 |
| `lazy_felt_floor` | 38280 | 12940 | x1.28 |
| `lazy_felt` | 39650 | 14310 | x1.42 |
| `lazy_wide_mul` | 40560 | 15220 | x1.51 |
| `eager_bounded_floor` | 50160 | 24820 | x2.46 |
| `eager_wide_mul` | 59520 | 34180 | x3.39 |
| `lazy_i128` | 106910 | 81570 | x8.09 |
| `eager` | 125670 | 100330 | x9.95 |

## layout::mat3_gen

### mat3_add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 26540 | 1100 | x1.00 |
| `array9` | 26540 | 1100 | x1.00 |
| `cols` | 26540 | 1100 | x1.00 |
| `fields` | 26540 | 1100 | x1.00 |
| `dyn_seq` | 51310 | 25870 | x23.52 |
| `dyn_index` | 58500 | 33060 | x30.05 |

### mat3_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 37330 | 11190 | x1.00 |
| `array9` | 37330 | 11190 | x1.00 |
| `fields` | 37330 | 11190 | x1.00 |
| `cols` | 40190 | 14050 | x1.26 |
| `dyn_seq` | 58670 | 32530 | x2.91 |
| `dyn_index` | 65860 | 39720 | x3.55 |

### mat3_add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 37330 | 11190 | x1.00 |
| `array9` | 37330 | 11190 | x1.00 |
| `fields` | 37330 | 11190 | x1.00 |
| `cols` | 40190 | 14050 | x1.26 |
| `dyn_seq` | 57970 | 31830 | x2.84 |
| `dyn_index` | 65160 | 39020 | x3.49 |

### mat3_construct_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 25440 | 0 | - |
| `array9` | 25440 | 0 | - |
| `cols` | 25440 | 0 | - |
| `fields` | 25440 | 0 | - |
| `dyn_seq` | 28010 | 2570 | - |
| `dyn_index` | 35000 | 9560 | - |

### mat3_construct_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 26140 | 0 | - |
| `array9` | 26140 | 0 | - |
| `cols` | 26140 | 0 | - |
| `fields` | 26140 | 0 | - |
| `dyn_seq` | 28710 | 2570 | - |
| `dyn_index` | 35700 | 9560 | - |

### mat3_construct_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 26140 | 0 | - |
| `array9` | 26140 | 0 | - |
| `cols` | 26140 | 0 | - |
| `fields` | 26140 | 0 | - |
| `dyn_seq` | 28710 | 2570 | - |
| `dyn_index` | 35700 | 9560 | - |

### mat3_determinant_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 25440 | 1400 | x1.00 |
| `array9` | 25440 | 1400 | x1.00 |
| `cols` | 25440 | 1400 | x1.00 |
| `fields` | 25440 | 1400 | x1.00 |
| `dyn_seq` | 28010 | 3970 | x2.84 |
| `dyn_index` | 51740 | 27700 | x19.79 |

### mat3_determinant_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 123430 | 99290 | x1.00 |
| `array9` | 123430 | 99290 | x1.00 |
| `fields` | 123430 | 99290 | x1.00 |
| `dyn_seq` | 125100 | 100960 | x1.02 |
| `cols` | 125460 | 101320 | x1.02 |
| `dyn_index` | 146630 | 122490 | x1.23 |

### mat3_determinant_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 37030 | 12890 | x1.00 |
| `array9` | 37030 | 12890 | x1.00 |
| `fields` | 37030 | 12890 | x1.00 |
| `dyn_seq` | 38700 | 14560 | x1.13 |
| `cols` | 39060 | 14920 | x1.16 |
| `dyn_index` | 61730 | 37590 | x2.92 |

### mat3_inverse_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 36440 | 11000 | x1.00 |
| `array9` | 36440 | 11000 | x1.00 |
| `cols` | 36440 | 11000 | x1.00 |
| `fields` | 36440 | 11000 | x1.00 |
| `dyn_seq` | 40680 | 15240 | x1.39 |
| `dyn_index` | 206900 | 181460 | x16.50 |

### mat3_inverse_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 370970 | 344830 | x1.00 |
| `array9` | 370970 | 344830 | x1.00 |
| `fields` | 370970 | 344830 | x1.00 |
| `dyn_seq` | 375210 | 349070 | x1.01 |
| `cols` | 380190 | 354050 | x1.03 |
| `dyn_index` | 600350 | 574210 | x1.67 |

### mat3_inverse_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 91500 | 65360 | x1.00 |
| `array9` | 91500 | 65360 | x1.00 |
| `fields` | 91500 | 65360 | x1.00 |
| `dyn_seq` | 95740 | 69600 | x1.06 |
| `cols` | 100720 | 74580 | x1.14 |
| `dyn_index` | 268450 | 242310 | x3.71 |

### mat3_mul_mat_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 30140 | 4700 | x1.00 |
| `array9` | 30140 | 4700 | x1.00 |
| `cols` | 30140 | 4700 | x1.00 |
| `fields` | 30140 | 4700 | x1.00 |
| `dyn_seq` | 190560 | 165120 | x35.13 |
| `dyn_index` | 238050 | 212610 | x45.24 |

### mat3_mul_mat_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 323170 | 297030 | x1.00 |
| `array9` | 323170 | 297030 | x1.00 |
| `fields` | 323170 | 297030 | x1.00 |
| `cols` | 355490 | 329350 | x1.11 |
| `dyn_seq` | 478020 | 451880 | x1.52 |
| `dyn_index` | 524670 | 498530 | x1.68 |

### mat3_mul_mat_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 63970 | 37830 | x1.00 |
| `array9` | 63970 | 37830 | x1.00 |
| `fields` | 63970 | 37830 | x1.00 |
| `cols` | 81620 | 55480 | x1.47 |
| `dyn_seq` | 222960 | 196820 | x5.20 |
| `dyn_index` | 270450 | 244310 | x6.46 |

### mat3_mul_vec_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 25940 | 1500 | x1.00 |
| `array9` | 25940 | 1500 | x1.00 |
| `cols` | 25940 | 1500 | x1.00 |
| `fields` | 25940 | 1500 | x1.00 |
| `dyn_seq` | 81390 | 56950 | x37.97 |
| `dyn_index` | 110270 | 85830 | x57.22 |

### mat3_mul_vec_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 125170 | 100430 | x1.00 |
| `array9` | 125170 | 100430 | x1.00 |
| `fields` | 125170 | 100430 | x1.00 |
| `cols` | 133290 | 108550 | x1.08 |
| `dyn_seq` | 177410 | 152670 | x1.52 |
| `dyn_index` | 205930 | 181190 | x1.80 |

### mat3_mul_vec_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 38770 | 14030 | x1.00 |
| `array9` | 38770 | 14030 | x1.00 |
| `fields` | 38770 | 14030 | x1.00 |
| `cols` | 42000 | 17260 | x1.23 |
| `dyn_seq` | 92190 | 67450 | x4.81 |
| `dyn_index` | 121070 | 96330 | x6.87 |

### mat3_scale_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 26340 | 900 | x1.00 |
| `array9` | 26340 | 900 | x1.00 |
| `cols` | 26340 | 900 | x1.00 |
| `fields` | 26340 | 900 | x1.00 |
| `dyn_seq` | 46210 | 20770 | x23.08 |
| `dyn_index` | 53400 | 27960 | x31.07 |

### mat3_scale_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 122530 | 96390 | x1.00 |
| `array9` | 122530 | 96390 | x1.00 |
| `fields` | 122530 | 96390 | x1.00 |
| `cols` | 125590 | 99450 | x1.03 |
| `dyn_seq` | 138550 | 112410 | x1.17 |
| `dyn_index` | 145740 | 119600 | x1.24 |

### mat3_scale_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 36330 | 10190 | x1.00 |
| `array9` | 36330 | 10190 | x1.00 |
| `cols` | 36330 | 10190 | x1.00 |
| `fields` | 36330 | 10190 | x1.00 |
| `dyn_seq` | 52670 | 26530 | x2.60 |
| `dyn_index` | 59860 | 33720 | x3.31 |

### mat3_transpose_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 25440 | 0 | - |
| `array9` | 25440 | 0 | - |
| `cols` | 25440 | 0 | - |
| `fields` | 25440 | 0 | - |
| `dyn_seq` | 75300 | 49860 | - |
| `dyn_index` | 87620 | 62180 | - |

### mat3_transpose_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 26140 | 0 | - |
| `array9` | 26140 | 0 | - |
| `cols` | 26140 | 0 | - |
| `fields` | 26140 | 0 | - |
| `dyn_seq` | 76700 | 50560 | - |
| `dyn_index` | 89020 | 62880 | - |

### mat3_transpose_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 26140 | 0 | - |
| `array9` | 26140 | 0 | - |
| `cols` | 26140 | 0 | - |
| `fields` | 26140 | 0 | - |
| `dyn_seq` | 76000 | 49860 | - |
| `dyn_index` | 88320 | 62180 | - |

## layout::matn_gen

### matmul_n2_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 19940 | 1200 | x1.00 |
| `dyn_index` | 93420 | 74680 | x62.23 |
| `dyn_seq` | 95770 | 77030 | x64.19 |

### matmul_n2_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 107250 | 88210 | x1.00 |
| `dyn_index` | 177360 | 158320 | x1.79 |
| `dyn_seq` | 180070 | 161030 | x1.83 |

### matmul_n2_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 30450 | 11410 | x1.00 |
| `dyn_index` | 102200 | 83160 | x7.29 |
| `dyn_seq` | 104550 | 85510 | x7.49 |

### matmul_n3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 28240 | 4600 | x1.00 |
| `dyn_static_kernel` | 36650 | 13010 | x2.83 |
| `dyn_seq` | 184660 | 161020 | x35.00 |
| `dyn_index` | 228860 | 205220 | x44.61 |

### matmul_n3_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 321070 | 296730 | x1.00 |
| `dyn_static_kernel` | 326980 | 302640 | x1.02 |
| `dyn_seq` | 472040 | 447700 | x1.51 |
| `dyn_index` | 515620 | 491280 | x1.66 |

### matmul_n3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 61870 | 37530 | x1.00 |
| `dyn_static_kernel` | 67780 | 43440 | x1.16 |
| `dyn_seq` | 216960 | 192620 | x5.13 |
| `dyn_index` | 261160 | 236820 | x6.31 |

### matmul_n4_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 41940 | 11400 | x1.00 |
| `dyn_seq` | 324370 | 293830 | x25.77 |
| `dyn_index` | 476200 | 445660 | x39.09 |

### matmul_n4_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 735650 | 703810 | x1.00 |
| `dyn_seq` | 1008530 | 976690 | x1.39 |
| `dyn_index` | 1159260 | 1127420 | x1.60 |

### matmul_n4_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 121250 | 89410 | x1.00 |
| `dyn_seq` | 403950 | 372110 | x4.16 |
| `dyn_index` | 555780 | 523940 | x5.86 |

### matmul_n6_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 97540 | 47200 | x1.00 |
| `dyn_seq` | 803330 | 752990 | x15.95 |
| `dyn_index` | 1445340 | 1395000 | x29.56 |

### matmul_n6_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 2481340 | 2427900 | x1.00 |
| `dyn_seq` | 3122610 | 3069170 | x1.26 |
| `dyn_index` | 3761840 | 3708400 | x1.53 |

### matmul_n6_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 407740 | 354300 | x1.00 |
| `dyn_seq` | 1081670 | 1028230 | x2.90 |
| `dyn_index` | 1723680 | 1670240 | x4.71 |

## layout::passing::tests

### pass_iso7

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot_x1` | 19040 | 1000 | x1.00 |
| `value_x1` | 19040 | 1000 | x1.00 |
| `box_x1` | 19640 | 1600 | x1.60 |
| `box_x3` | 20840 | 2800 | x2.80 |
| `snapshot_x3` | 21040 | 3000 | x3.00 |
| `value_x3` | 21040 | 3000 | x3.00 |

### pass_mat3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot_x1` | 20040 | 1200 | x1.00 |
| `value_x1` | 20040 | 1200 | x1.00 |
| `box_x1` | 20640 | 1800 | x1.50 |
| `box_x3` | 21840 | 3000 | x2.50 |
| `snapshot_x3` | 22440 | 3600 | x3.00 |
| `value_x3` | 22440 | 3600 | x3.00 |

### pass_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `value_default_inline_x3` | 21940 | 300 | x1.00 |
| `snapshot_default_inline_x3` | 22140 | 500 | x1.67 |
| `snapshot_x1` | 23540 | 1900 | x6.33 |
| `value_x1` | 23540 | 1900 | x6.33 |
| `box_x1` | 24140 | 2500 | x8.33 |
| `box_x3` | 25340 | 3700 | x12.33 |
| `snapshot_x3` | 27340 | 5700 | x19.00 |
| `value_x3` | 27340 | 5700 | x19.00 |

### pass_mulvec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot` | 120870 | 100330 | x1.00 |
| `value` | 120870 | 100330 | x1.00 |

### pass_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot_x1` | 17040 | 600 | x1.00 |
| `value_x1` | 17040 | 600 | x1.00 |
| `box_x1` | 17640 | 1200 | x2.00 |
| `snapshot_x3` | 18240 | 1800 | x3.00 |
| `value_x3` | 18240 | 1800 | x3.00 |
| `box_x3` | 18840 | 2400 | x4.00 |

### return_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `value` | 17340 | 1800 | x1.00 |
| `boxed` | 17940 | 2400 | x1.33 |

## layout::vec3_gen

### vec3_add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18740 | 300 | x1.00 |
| `struct` | 18740 | 300 | x1.00 |
| `tuple` | 18740 | 300 | x1.00 |
| `span_box` | 23790 | 5350 | x17.83 |
| `span_index` | 27260 | 8820 | x29.40 |
| `span_loop` | 30560 | 12120 | x40.40 |

### vec3_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 23290 | 4550 | x1.00 |
| `struct` | 23290 | 4550 | x1.00 |
| `tuple` | 23290 | 4550 | x1.00 |
| `span_box` | 27040 | 8300 | x1.82 |
| `span_index` | 29910 | 11170 | x2.45 |
| `span_loop` | 33080 | 14340 | x3.15 |

### vec3_add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 23290 | 4550 | x1.00 |
| `struct` | 23290 | 4550 | x1.00 |
| `tuple` | 23290 | 4550 | x1.00 |
| `span_box` | 27040 | 8300 | x1.82 |
| `span_index` | 29910 | 11170 | x2.45 |
| `span_loop` | 32780 | 14040 | x3.09 |

### vec3_chain_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 20640 | 2200 | x1.00 |
| `struct` | 20640 | 2200 | x1.00 |
| `tuple` | 20640 | 2200 | x1.00 |
| `span_box` | 32310 | 13870 | x6.30 |
| `span_index` | 47070 | 28630 | x13.01 |
| `span_loop` | 84430 | 65990 | x30.00 |

### vec3_chain_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 162110 | 143370 | x1.00 |
| `struct` | 162110 | 143370 | x1.00 |
| `tuple` | 162110 | 143370 | x1.00 |
| `span_box` | 169380 | 150640 | x1.05 |
| `span_index` | 180840 | 162100 | x1.13 |
| `span_loop` | 215310 | 196570 | x1.37 |

### vec3_chain_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 44780 | 26040 | x1.00 |
| `struct` | 44780 | 26040 | x1.00 |
| `tuple` | 44780 | 26040 | x1.00 |
| `span_box` | 54680 | 35940 | x1.38 |
| `span_index` | 66140 | 47400 | x1.82 |
| `span_loop` | 101650 | 82910 | x3.18 |

### vec3_construct_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18440 | 0 | - |
| `struct` | 18440 | 0 | - |
| `tuple` | 18440 | 0 | - |
| `span_index` | 20680 | 2240 | - |
| `span_loop` | 20680 | 2240 | - |
| `span_box` | 20750 | 2310 | - |

### vec3_construct_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18740 | 0 | - |
| `struct` | 18740 | 0 | - |
| `tuple` | 18740 | 0 | - |
| `span_index` | 20980 | 2240 | - |
| `span_loop` | 20980 | 2240 | - |
| `span_box` | 21050 | 2310 | - |

### vec3_construct_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18740 | 0 | - |
| `struct` | 18740 | 0 | - |
| `tuple` | 18740 | 0 | - |
| `span_index` | 20980 | 2240 | - |
| `span_loop` | 20980 | 2240 | - |
| `span_box` | 21050 | 2310 | - |

### vec3_cross_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19340 | 900 | x1.00 |
| `struct` | 19340 | 900 | x1.00 |
| `tuple` | 19340 | 900 | x1.00 |
| `span_box` | 24390 | 5950 | x6.61 |
| `span_index` | 27860 | 9420 | x10.47 |
| `span_loop` | 49150 | 30710 | x34.12 |

### vec3_cross_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 85330 | 66590 | x1.00 |
| `struct` | 85330 | 66590 | x1.00 |
| `tuple` | 85330 | 66590 | x1.00 |
| `span_box` | 89380 | 70640 | x1.06 |
| `span_index` | 92250 | 73510 | x1.10 |
| `span_loop` | 111890 | 93150 | x1.40 |

### vec3_cross_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 27730 | 8990 | x1.00 |
| `struct` | 27730 | 8990 | x1.00 |
| `tuple` | 27730 | 8990 | x1.00 |
| `span_box` | 31480 | 12740 | x1.42 |
| `span_index` | 34350 | 15610 | x1.74 |
| `span_loop` | 55210 | 36470 | x4.06 |

### vec3_dot_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18540 | 500 | x1.00 |
| `struct` | 18540 | 500 | x1.00 |
| `tuple` | 18540 | 500 | x1.00 |
| `span_box` | 21280 | 3240 | x6.48 |
| `span_index` | 24520 | 6480 | x12.96 |
| `span_loop` | 26250 | 8210 | x16.42 |

### vec3_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 52870 | 34730 | x1.00 |
| `struct` | 52870 | 34730 | x1.00 |
| `tuple` | 52870 | 34730 | x1.00 |
| `span_box` | 54410 | 36270 | x1.04 |
| `span_index` | 56950 | 38810 | x1.12 |
| `span_loop` | 60470 | 42330 | x1.22 |

### vec3_dot_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 24070 | 5930 | x1.00 |
| `struct` | 24070 | 5930 | x1.00 |
| `tuple` | 24070 | 5930 | x1.00 |
| `span_box` | 26210 | 8070 | x1.36 |
| `span_index` | 28750 | 10610 | x1.79 |
| `span_loop` | 32050 | 13910 | x2.35 |

### vec3_equals_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 21240 | 400 | x1.00 |
| `struct` | 21240 | 400 | x1.00 |
| `tuple` | 21240 | 400 | x1.00 |
| `span_box` | 23980 | 3140 | x7.85 |
| `span_index` | 27920 | 7080 | x17.70 |
| `span_loop` | 30220 | 9380 | x23.45 |

### vec3_equals_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 21540 | 600 | x1.00 |
| `struct` | 21540 | 600 | x1.00 |
| `tuple` | 21540 | 600 | x1.00 |
| `span_box` | 24280 | 3340 | x5.57 |
| `span_index` | 28120 | 7180 | x11.97 |
| `span_loop` | 30520 | 9580 | x15.97 |

### vec3_equals_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 21540 | 600 | x1.00 |
| `struct` | 21540 | 600 | x1.00 |
| `tuple` | 21540 | 600 | x1.00 |
| `span_box` | 24280 | 3340 | x5.57 |
| `span_index` | 28120 | 7180 | x11.97 |
| `span_loop` | 30520 | 9580 | x15.97 |

### vec3_neg_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18740 | 300 | x1.00 |
| `struct` | 18740 | 300 | x1.00 |
| `tuple` | 18740 | 300 | x1.00 |
| `span_box` | 22420 | 3980 | x13.27 |
| `span_index` | 23220 | 4780 | x15.93 |
| `span_loop` | 28260 | 9820 | x32.73 |

### vec3_neg_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19640 | 900 | x1.00 |
| `struct` | 19640 | 900 | x1.00 |
| `tuple` | 19640 | 900 | x1.00 |
| `span_box` | 23320 | 4580 | x5.09 |
| `span_index` | 24120 | 5380 | x5.98 |
| `span_loop` | 29160 | 10420 | x11.58 |

### vec3_neg_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19640 | 900 | x1.00 |
| `struct` | 19640 | 900 | x1.00 |
| `tuple` | 19640 | 900 | x1.00 |
| `span_box` | 23320 | 4580 | x5.09 |
| `span_index` | 24120 | 5380 | x5.98 |
| `span_loop` | 29160 | 10420 | x11.58 |

### vec3_scale_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18740 | 300 | x1.00 |
| `struct` | 18740 | 300 | x1.00 |
| `tuple` | 18740 | 300 | x1.00 |
| `span_box` | 22420 | 3980 | x13.27 |
| `span_index` | 23220 | 4780 | x15.93 |
| `span_loop` | 28660 | 10220 | x34.07 |

### vec3_scale_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 51890 | 33150 | x1.00 |
| `struct` | 51890 | 33150 | x1.00 |
| `tuple` | 51890 | 33150 | x1.00 |
| `span_box` | 54570 | 35830 | x1.08 |
| `span_index` | 56170 | 37430 | x1.13 |
| `span_loop` | 59720 | 40980 | x1.24 |

### vec3_scale_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 20960 | 2220 | x1.00 |
| `struct` | 20960 | 2220 | x1.00 |
| `tuple` | 20960 | 2220 | x1.00 |
| `span_box` | 25970 | 7230 | x3.26 |
| `span_index` | 27570 | 8830 | x3.98 |
| `span_loop` | 30880 | 12140 | x5.47 |

### vec3_sub_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18740 | 300 | x1.00 |
| `struct` | 18740 | 300 | x1.00 |
| `tuple` | 18740 | 300 | x1.00 |
| `span_box` | 23790 | 5350 | x17.83 |
| `span_index` | 27260 | 8820 | x29.40 |
| `span_loop` | 30560 | 12120 | x40.40 |

### vec3_sub_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 23290 | 4550 | x1.00 |
| `struct` | 23290 | 4550 | x1.00 |
| `tuple` | 23290 | 4550 | x1.00 |
| `span_box` | 27040 | 8300 | x1.82 |
| `span_index` | 29910 | 11170 | x2.45 |
| `span_loop` | 32780 | 14040 | x3.09 |

### vec3_sub_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 23290 | 4550 | x1.00 |
| `struct` | 23290 | 4550 | x1.00 |
| `tuple` | 23290 | 4550 | x1.00 |
| `span_box` | 27040 | 8300 | x1.82 |
| `span_index` | 29910 | 11170 | x2.45 |
| `span_loop` | 32780 | 14040 | x3.09 |

## layout::vecn::tests

### access_dyn_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct_match` | 17350 | 710 | x1.00 |
| `span` | 17910 | 1270 | x1.79 |
| `array_span` | 18210 | 1570 | x2.21 |

### access_y

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array_destructure` | 16240 | 0 | - |
| `struct_field` | 16240 | 0 | - |
| `tuple_destructure` | 16240 | 0 | - |
| `span_index` | 17410 | 1170 | - |
| `array_span_index` | 17910 | 1670 | - |

### vec2_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18420 | 1380 | x1.00 |
| `span_loop` | 27300 | 10260 | x7.43 |

### vec2_dot_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 16940 | 300 | x1.00 |
| `span_loop` | 22580 | 5940 | x19.80 |

### vec2_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 40090 | 23350 | x1.00 |
| `span_loop` | 45920 | 29180 | x1.25 |

### vec4_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 24730 | 5490 | x1.00 |
| `struct` | 24730 | 5490 | x1.00 |
| `span_loop` | 37360 | 18120 | x3.30 |

### vec4_dot_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18940 | 700 | x1.00 |
| `struct` | 18940 | 700 | x1.00 |
| `span_loop` | 28520 | 10280 | x14.69 |

### vec4_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 64250 | 45910 | x1.00 |
| `struct` | 64250 | 45910 | x1.00 |
| `span_loop` | 73620 | 55280 | x1.20 |

