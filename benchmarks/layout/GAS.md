# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## layout::abstraction::tests

### call_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator_inline_always` | 12430 | 2120 | x1.00 |
| `trait_method_inline_always` | 12430 | 2120 | x1.00 |
| `free_fn` | 14760 | 4450 | x2.10 |
| `free_fn_generic` | 14760 | 4450 | x2.10 |
| `operator` | 14760 | 4450 | x2.10 |
| `trait_method` | 14760 | 4450 | x2.10 |

### call_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `free_fn` | 44340 | 34630 | x1.00 |
| `free_fn_generic` | 44340 | 34630 | x1.00 |
| `trait_method` | 44340 | 34630 | x1.00 |

## layout::abstraction_gen

### abs_vec3_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 13130 | 2220 | x1.00 |
| `generic_inline_always` | 13130 | 2220 | x1.00 |
| `concrete_default` | 15460 | 4550 | x2.05 |
| `concrete_inline_never` | 15460 | 4550 | x2.05 |
| `generic_default` | 15460 | 4550 | x2.05 |
| `generic_inline_never` | 15460 | 4550 | x2.05 |

### abs_vec3_chain_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 143490 | 132580 | x1.00 |
| `generic_inline_always` | 143490 | 132580 | x1.00 |
| `concrete_default` | 154280 | 143370 | x1.08 |
| `concrete_inline_never` | 154280 | 143370 | x1.08 |
| `generic_default` | 154280 | 143370 | x1.08 |
| `generic_inline_never` | 154280 | 143370 | x1.08 |

### abs_vec3_cross_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 75450 | 64540 | x1.00 |
| `generic_inline_always` | 75450 | 64540 | x1.00 |
| `concrete_default` | 77500 | 66590 | x1.03 |
| `concrete_inline_never` | 77500 | 66590 | x1.03 |
| `generic_default` | 77500 | 66590 | x1.03 |
| `generic_inline_never` | 77500 | 66590 | x1.03 |

### abs_vec3_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `concrete_inline_always` | 43030 | 32720 | x1.00 |
| `generic_inline_always` | 43030 | 32720 | x1.00 |
| `concrete_default` | 45040 | 34730 | x1.06 |
| `concrete_inline_never` | 45040 | 34730 | x1.06 |
| `generic_default` | 45040 | 34730 | x1.06 |
| `generic_inline_never` | 45040 | 34730 | x1.06 |

## layout::composite::tests

### inertia_world

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_symmetric` | 303900 | 291190 | x1.00 |
| `two_mat_mat` | 605470 | 592760 | x2.04 |

### iso_compose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quat` | 369400 | 355890 | x1.00 |
| `mat3` | 420680 | 407170 | x1.14 |

### iso_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mat3` | 119370 | 107460 | x1.00 |
| `quat` | 190250 | 178340 | x1.66 |

### m_mt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `symmetric_6_dots` | 221890 | 210380 | x1.00 |
| `transpose_then_mat_mat` | 308040 | 296530 | x1.41 |

### normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_only` | 51360 | 42450 | x1.00 |
| `three_divisions` | 91830 | 82920 | x1.95 |
| `one_division_three_muls` | 93340 | 84430 | x1.99 |

### quat_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hamilton` | 188960 | 175950 | x1.00 |

### quat_rotate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 183900 | 173190 | x1.00 |
| `via_mat3` | 218230 | 207520 | x1.20 |
| `sandwich` | 365110 | 354400 | x2.05 |

### quat_rotate_x4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_mat3_once` | 525820 | 508810 | x1.00 |
| `expanded` | 710070 | 693060 | x1.36 |

### quat_to_mat3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `convert` | 117000 | 106290 | x1.00 |

### skew_mul_mat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross_per_column` | 212380 | 199670 | x1.00 |
| `skew_then_mat_mat` | 309740 | 297030 | x1.49 |

### skew_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross` | 76800 | 66490 | x1.00 |
| `skew_then_mat_vec` | 111240 | 100930 | x1.52 |

### sym3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sym6` | 251140 | 240830 | x1.00 |
| `full9` | 354640 | 344330 | x1.43 |

### sym3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sym6` | 111540 | 100030 | x1.00 |
| `full9` | 111840 | 100330 | x1.00 |

## layout::inline_gen

### inline_mat3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fields_malways` | 363140 | 344830 | x1.00 |
| `fields_mdefault` | 363140 | 344830 | x1.00 |
| `cols_malways` | 372360 | 354050 | x1.03 |
| `cols_mdefault` | 372360 | 354050 | x1.03 |
| `fields_mnever` | 401570 | 383260 | x1.11 |
| `cols_mnever` | 410790 | 392480 | x1.14 |

### inline_mat3_mul_mat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fields_malways` | 315340 | 297030 | x1.00 |
| `fields_mdefault` | 315340 | 297030 | x1.00 |
| `cols_malways` | 347660 | 329350 | x1.11 |
| `cols_mdefault` | 347660 | 329350 | x1.11 |
| `fields_mnever` | 364750 | 346440 | x1.17 |
| `cols_mnever` | 382400 | 364090 | x1.23 |

### inline_mat3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fields_malways` | 117340 | 100430 | x1.00 |
| `fields_mdefault` | 117340 | 100430 | x1.00 |
| `cols_malways` | 125460 | 108550 | x1.08 |
| `cols_mdefault` | 125460 | 108550 | x1.08 |
| `fields_mnever` | 133810 | 116900 | x1.16 |
| `cols_mnever` | 137040 | 120130 | x1.20 |

### inline_vec3_chain

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valways_malways` | 143490 | 132580 | x1.00 |
| `valways_mdefault` | 143490 | 132580 | x1.00 |
| `vdefault_malways` | 154280 | 143370 | x1.08 |
| `vdefault_mdefault` | 154280 | 143370 | x1.08 |
| `vnever_malways` | 154280 | 143370 | x1.08 |
| `vnever_mdefault` | 154280 | 143370 | x1.08 |
| `valways_mnever` | 164890 | 153980 | x1.16 |
| `vdefault_mnever` | 174110 | 163200 | x1.23 |
| `vnever_mnever` | 175410 | 164500 | x1.24 |

### inline_vec3_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valways_malways` | 75450 | 64540 | x1.00 |
| `valways_mdefault` | 75450 | 64540 | x1.00 |
| `vdefault_malways` | 77500 | 66590 | x1.03 |
| `vdefault_mdefault` | 77500 | 66590 | x1.03 |
| `vnever_malways` | 77500 | 66590 | x1.03 |
| `vnever_mdefault` | 77500 | 66590 | x1.03 |
| `valways_mnever` | 86150 | 75240 | x1.17 |
| `vdefault_mnever` | 88480 | 77570 | x1.20 |
| `vnever_mnever` | 88480 | 77570 | x1.20 |

### inline_vec3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valways_malways` | 43030 | 32720 | x1.00 |
| `valways_mdefault` | 43030 | 32720 | x1.00 |
| `vdefault_malways` | 45040 | 34730 | x1.06 |
| `vdefault_mdefault` | 45040 | 34730 | x1.06 |
| `vnever_malways` | 45040 | 34730 | x1.06 |
| `vnever_mdefault` | 45040 | 34730 | x1.06 |
| `valways_mnever` | 48300 | 37990 | x1.16 |
| `vdefault_mnever` | 50530 | 40220 | x1.23 |
| `vnever_mnever` | 50530 | 40220 | x1.23 |

## layout::lazy::tests

### lazy_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_felt_floor` | 22050 | 11740 | x1.00 |
| `lazy_felt` | 23420 | 13110 | x1.12 |
| `lazy_i128` | 67520 | 57210 | x4.87 |
| `eager` | 76800 | 66490 | x5.66 |

### lazy_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_bounded_floor` | 11960 | 2250 | x1.00 |
| `lazy_felt_floor` | 12700 | 2990 | x1.33 |
| `lazy_felt` | 15800 | 6090 | x2.71 |
| `lazy_felt_generic_api` | 15800 | 6090 | x2.71 |
| `lazy_wide_mul` | 15970 | 6260 | x2.78 |
| `eager_bounded_floor` | 19170 | 9460 | x4.20 |
| `eager_wide_mul` | 22290 | 12580 | x5.59 |
| `eager_felt` | 23400 | 13690 | x6.08 |
| `lazy_i128` | 38020 | 28310 | x12.58 |
| `eager` | 44340 | 34630 | x15.39 |

### lazy_mul_mat

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_bounded_floor` | 42290 | 25780 | x1.00 |
| `lazy_felt_floor` | 49590 | 33080 | x1.28 |
| `lazy_felt` | 54980 | 38470 | x1.49 |
| `lazy_wide_mul` | 58110 | 41600 | x1.61 |
| `eager_bounded_floor` | 86910 | 70400 | x2.73 |
| `eager_wide_mul` | 114990 | 98480 | x3.82 |
| `dynamic_lazy_felt` | 223540 | 207030 | x8.03 |
| `lazy_i128` | 256560 | 240050 | x9.31 |
| `eager` | 313440 | 296930 | x11.52 |
| `dynamic_eager` | 468590 | 452080 | x17.54 |

### lazy_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lazy_bounded_floor` | 27590 | 10080 | x1.00 |
| `lazy_felt_floor` | 30450 | 12940 | x1.28 |
| `lazy_felt` | 31820 | 14310 | x1.42 |
| `lazy_wide_mul` | 32730 | 15220 | x1.51 |
| `eager_bounded_floor` | 42330 | 24820 | x2.46 |
| `eager_wide_mul` | 51690 | 34180 | x3.39 |
| `lazy_i128` | 99080 | 81570 | x8.09 |
| `eager` | 117840 | 100330 | x9.95 |

## layout::mat3_gen

### mat3_add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18710 | 1100 | x1.00 |
| `array9` | 18710 | 1100 | x1.00 |
| `cols` | 18710 | 1100 | x1.00 |
| `fields` | 18710 | 1100 | x1.00 |
| `dyn_seq` | 43480 | 25870 | x23.52 |
| `dyn_index` | 50670 | 33060 | x30.05 |

### mat3_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 29500 | 11190 | x1.00 |
| `array9` | 29500 | 11190 | x1.00 |
| `fields` | 29500 | 11190 | x1.00 |
| `cols` | 32360 | 14050 | x1.26 |
| `dyn_seq` | 50840 | 32530 | x2.91 |
| `dyn_index` | 58030 | 39720 | x3.55 |

### mat3_add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 29500 | 11190 | x1.00 |
| `array9` | 29500 | 11190 | x1.00 |
| `fields` | 29500 | 11190 | x1.00 |
| `cols` | 32360 | 14050 | x1.26 |
| `dyn_seq` | 50140 | 31830 | x2.84 |
| `dyn_index` | 57330 | 39020 | x3.49 |

### mat3_construct_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 17610 | 0 | - |
| `array9` | 17610 | 0 | - |
| `cols` | 17610 | 0 | - |
| `fields` | 17610 | 0 | - |
| `dyn_seq` | 20180 | 2570 | - |
| `dyn_index` | 27170 | 9560 | - |

### mat3_construct_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18310 | 0 | - |
| `array9` | 18310 | 0 | - |
| `cols` | 18310 | 0 | - |
| `fields` | 18310 | 0 | - |
| `dyn_seq` | 20880 | 2570 | - |
| `dyn_index` | 27870 | 9560 | - |

### mat3_construct_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18310 | 0 | - |
| `array9` | 18310 | 0 | - |
| `cols` | 18310 | 0 | - |
| `fields` | 18310 | 0 | - |
| `dyn_seq` | 20880 | 2570 | - |
| `dyn_index` | 27870 | 9560 | - |

### mat3_determinant_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 17610 | 1400 | x1.00 |
| `array9` | 17610 | 1400 | x1.00 |
| `cols` | 17610 | 1400 | x1.00 |
| `fields` | 17610 | 1400 | x1.00 |
| `dyn_seq` | 20180 | 3970 | x2.84 |
| `dyn_index` | 43910 | 27700 | x19.79 |

### mat3_determinant_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 115600 | 99290 | x1.00 |
| `array9` | 115600 | 99290 | x1.00 |
| `fields` | 115600 | 99290 | x1.00 |
| `dyn_seq` | 117270 | 100960 | x1.02 |
| `cols` | 117630 | 101320 | x1.02 |
| `dyn_index` | 138800 | 122490 | x1.23 |

### mat3_determinant_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 29200 | 12890 | x1.00 |
| `array9` | 29200 | 12890 | x1.00 |
| `fields` | 29200 | 12890 | x1.00 |
| `dyn_seq` | 30870 | 14560 | x1.13 |
| `cols` | 31230 | 14920 | x1.16 |
| `dyn_index` | 53900 | 37590 | x2.92 |

### mat3_inverse_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 28610 | 11000 | x1.00 |
| `array9` | 28610 | 11000 | x1.00 |
| `cols` | 28610 | 11000 | x1.00 |
| `fields` | 28610 | 11000 | x1.00 |
| `dyn_seq` | 32850 | 15240 | x1.39 |
| `dyn_index` | 199070 | 181460 | x16.50 |

### mat3_inverse_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 363140 | 344830 | x1.00 |
| `array9` | 363140 | 344830 | x1.00 |
| `fields` | 363140 | 344830 | x1.00 |
| `dyn_seq` | 367380 | 349070 | x1.01 |
| `cols` | 372360 | 354050 | x1.03 |
| `dyn_index` | 592520 | 574210 | x1.67 |

### mat3_inverse_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 83670 | 65360 | x1.00 |
| `array9` | 83670 | 65360 | x1.00 |
| `fields` | 83670 | 65360 | x1.00 |
| `dyn_seq` | 87910 | 69600 | x1.06 |
| `cols` | 92890 | 74580 | x1.14 |
| `dyn_index` | 260620 | 242310 | x3.71 |

### mat3_mul_mat_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 22310 | 4700 | x1.00 |
| `array9` | 22310 | 4700 | x1.00 |
| `cols` | 22310 | 4700 | x1.00 |
| `fields` | 22310 | 4700 | x1.00 |
| `dyn_seq` | 182730 | 165120 | x35.13 |
| `dyn_index` | 230220 | 212610 | x45.24 |

### mat3_mul_mat_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 315340 | 297030 | x1.00 |
| `array9` | 315340 | 297030 | x1.00 |
| `fields` | 315340 | 297030 | x1.00 |
| `cols` | 347660 | 329350 | x1.11 |
| `dyn_seq` | 470190 | 451880 | x1.52 |
| `dyn_index` | 516840 | 498530 | x1.68 |

### mat3_mul_mat_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 56140 | 37830 | x1.00 |
| `array9` | 56140 | 37830 | x1.00 |
| `fields` | 56140 | 37830 | x1.00 |
| `cols` | 73790 | 55480 | x1.47 |
| `dyn_seq` | 215130 | 196820 | x5.20 |
| `dyn_index` | 262620 | 244310 | x6.46 |

### mat3_mul_vec_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18110 | 1500 | x1.00 |
| `array9` | 18110 | 1500 | x1.00 |
| `cols` | 18110 | 1500 | x1.00 |
| `fields` | 18110 | 1500 | x1.00 |
| `dyn_seq` | 73560 | 56950 | x37.97 |
| `dyn_index` | 102440 | 85830 | x57.22 |

### mat3_mul_vec_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 117340 | 100430 | x1.00 |
| `array9` | 117340 | 100430 | x1.00 |
| `fields` | 117340 | 100430 | x1.00 |
| `cols` | 125460 | 108550 | x1.08 |
| `dyn_seq` | 169580 | 152670 | x1.52 |
| `dyn_index` | 198100 | 181190 | x1.80 |

### mat3_mul_vec_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 30940 | 14030 | x1.00 |
| `array9` | 30940 | 14030 | x1.00 |
| `fields` | 30940 | 14030 | x1.00 |
| `cols` | 34170 | 17260 | x1.23 |
| `dyn_seq` | 84360 | 67450 | x4.81 |
| `dyn_index` | 113240 | 96330 | x6.87 |

### mat3_scale_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18510 | 900 | x1.00 |
| `array9` | 18510 | 900 | x1.00 |
| `cols` | 18510 | 900 | x1.00 |
| `fields` | 18510 | 900 | x1.00 |
| `dyn_seq` | 38380 | 20770 | x23.08 |
| `dyn_index` | 45570 | 27960 | x31.07 |

### mat3_scale_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 114700 | 96390 | x1.00 |
| `array9` | 114700 | 96390 | x1.00 |
| `fields` | 114700 | 96390 | x1.00 |
| `cols` | 117760 | 99450 | x1.03 |
| `dyn_seq` | 130720 | 112410 | x1.17 |
| `dyn_index` | 137910 | 119600 | x1.24 |

### mat3_scale_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 28500 | 10190 | x1.00 |
| `array9` | 28500 | 10190 | x1.00 |
| `cols` | 28500 | 10190 | x1.00 |
| `fields` | 28500 | 10190 | x1.00 |
| `dyn_seq` | 44840 | 26530 | x2.60 |
| `dyn_index` | 52030 | 33720 | x3.31 |

### mat3_transpose_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 17610 | 0 | - |
| `array9` | 17610 | 0 | - |
| `cols` | 17610 | 0 | - |
| `fields` | 17610 | 0 | - |
| `dyn_seq` | 67470 | 49860 | - |
| `dyn_index` | 79790 | 62180 | - |

### mat3_transpose_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18310 | 0 | - |
| `array9` | 18310 | 0 | - |
| `cols` | 18310 | 0 | - |
| `fields` | 18310 | 0 | - |
| `dyn_seq` | 68870 | 50560 | - |
| `dyn_index` | 81190 | 62880 | - |

### mat3_transpose_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array3x3` | 18310 | 0 | - |
| `array9` | 18310 | 0 | - |
| `cols` | 18310 | 0 | - |
| `fields` | 18310 | 0 | - |
| `dyn_seq` | 68170 | 49860 | - |
| `dyn_index` | 80490 | 62180 | - |

## layout::matn_gen

### matmul_n2_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 12110 | 1200 | x1.00 |
| `dyn_index` | 85590 | 74680 | x62.23 |
| `dyn_seq` | 87940 | 77030 | x64.19 |

### matmul_n2_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 99420 | 88210 | x1.00 |
| `dyn_index` | 169530 | 158320 | x1.79 |
| `dyn_seq` | 172240 | 161030 | x1.83 |

### matmul_n2_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 22620 | 11410 | x1.00 |
| `dyn_index` | 94370 | 83160 | x7.29 |
| `dyn_seq` | 96720 | 85510 | x7.49 |

### matmul_n3_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 20410 | 4600 | x1.00 |
| `dyn_static_kernel` | 28820 | 13010 | x2.83 |
| `dyn_seq` | 176830 | 161020 | x35.00 |
| `dyn_index` | 221030 | 205220 | x44.61 |

### matmul_n3_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 313240 | 296730 | x1.00 |
| `dyn_static_kernel` | 319150 | 302640 | x1.02 |
| `dyn_seq` | 464210 | 447700 | x1.51 |
| `dyn_index` | 507790 | 491280 | x1.66 |

### matmul_n3_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 54040 | 37530 | x1.00 |
| `dyn_static_kernel` | 59950 | 43440 | x1.16 |
| `dyn_seq` | 209130 | 192620 | x5.13 |
| `dyn_index` | 253330 | 236820 | x6.31 |

### matmul_n4_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 34110 | 11400 | x1.00 |
| `dyn_seq` | 316540 | 293830 | x25.77 |
| `dyn_index` | 468370 | 445660 | x39.09 |

### matmul_n4_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 727820 | 703810 | x1.00 |
| `dyn_seq` | 1000700 | 976690 | x1.39 |
| `dyn_index` | 1151430 | 1127420 | x1.60 |

### matmul_n4_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 113420 | 89410 | x1.00 |
| `dyn_seq` | 396120 | 372110 | x4.16 |
| `dyn_index` | 547950 | 523940 | x5.86 |

### matmul_n6_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 89710 | 47200 | x1.00 |
| `dyn_seq` | 795500 | 752990 | x15.95 |
| `dyn_index` | 1437510 | 1395000 | x29.56 |

### matmul_n6_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 2473510 | 2427900 | x1.00 |
| `dyn_seq` | 3114780 | 3069170 | x1.26 |
| `dyn_index` | 3754010 | 3708400 | x1.53 |

### matmul_n6_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `static` | 399910 | 354300 | x1.00 |
| `dyn_seq` | 1073840 | 1028230 | x2.90 |
| `dyn_index` | 1715850 | 1670240 | x4.71 |

## layout::passing::tests

### pass_iso7

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot_x1` | 11210 | 1000 | x1.00 |
| `value_x1` | 11210 | 1000 | x1.00 |
| `box_x1` | 11810 | 1600 | x1.60 |
| `box_x3` | 13010 | 2800 | x2.80 |
| `snapshot_x3` | 13210 | 3000 | x3.00 |
| `value_x3` | 13210 | 3000 | x3.00 |

### pass_mat3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot_x1` | 12210 | 1200 | x1.00 |
| `value_x1` | 12210 | 1200 | x1.00 |
| `box_x1` | 12810 | 1800 | x1.50 |
| `box_x3` | 14010 | 3000 | x2.50 |
| `snapshot_x3` | 14610 | 3600 | x3.00 |
| `value_x3` | 14610 | 3600 | x3.00 |

### pass_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `value_default_inline_x3` | 14110 | 300 | x1.00 |
| `snapshot_default_inline_x3` | 14310 | 500 | x1.67 |
| `snapshot_x1` | 15710 | 1900 | x6.33 |
| `value_x1` | 15710 | 1900 | x6.33 |
| `box_x1` | 16310 | 2500 | x8.33 |
| `box_x3` | 17510 | 3700 | x12.33 |
| `snapshot_x3` | 19510 | 5700 | x19.00 |
| `value_x3` | 19510 | 5700 | x19.00 |

### pass_mulvec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot` | 113040 | 100330 | x1.00 |
| `value` | 113040 | 100330 | x1.00 |

### pass_vec3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `snapshot_x1` | 9210 | 600 | x1.00 |
| `value_x1` | 9210 | 600 | x1.00 |
| `box_x1` | 9810 | 1200 | x2.00 |
| `snapshot_x3` | 10410 | 1800 | x3.00 |
| `value_x3` | 10410 | 1800 | x3.00 |
| `box_x3` | 11010 | 2400 | x4.00 |

### return_mat4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `value` | 9510 | 1800 | x1.00 |
| `boxed` | 10110 | 2400 | x1.33 |

## layout::vec3_gen

### vec3_add_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10910 | 300 | x1.00 |
| `struct` | 10910 | 300 | x1.00 |
| `tuple` | 10910 | 300 | x1.00 |
| `span_box` | 15960 | 5350 | x17.83 |
| `span_index` | 19430 | 8820 | x29.40 |
| `span_loop` | 22730 | 12120 | x40.40 |

### vec3_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 15460 | 4550 | x1.00 |
| `struct` | 15460 | 4550 | x1.00 |
| `tuple` | 15460 | 4550 | x1.00 |
| `span_box` | 19210 | 8300 | x1.82 |
| `span_index` | 22080 | 11170 | x2.45 |
| `span_loop` | 25250 | 14340 | x3.15 |

### vec3_add_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 15460 | 4550 | x1.00 |
| `struct` | 15460 | 4550 | x1.00 |
| `tuple` | 15460 | 4550 | x1.00 |
| `span_box` | 19210 | 8300 | x1.82 |
| `span_index` | 22080 | 11170 | x2.45 |
| `span_loop` | 24950 | 14040 | x3.09 |

### vec3_chain_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 12810 | 2200 | x1.00 |
| `struct` | 12810 | 2200 | x1.00 |
| `tuple` | 12810 | 2200 | x1.00 |
| `span_box` | 24480 | 13870 | x6.30 |
| `span_index` | 39240 | 28630 | x13.01 |
| `span_loop` | 76600 | 65990 | x30.00 |

### vec3_chain_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 154280 | 143370 | x1.00 |
| `struct` | 154280 | 143370 | x1.00 |
| `tuple` | 154280 | 143370 | x1.00 |
| `span_box` | 161550 | 150640 | x1.05 |
| `span_index` | 173010 | 162100 | x1.13 |
| `span_loop` | 207480 | 196570 | x1.37 |

### vec3_chain_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 36950 | 26040 | x1.00 |
| `struct` | 36950 | 26040 | x1.00 |
| `tuple` | 36950 | 26040 | x1.00 |
| `span_box` | 46850 | 35940 | x1.38 |
| `span_index` | 58310 | 47400 | x1.82 |
| `span_loop` | 93820 | 82910 | x3.18 |

### vec3_construct_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10610 | 0 | - |
| `struct` | 10610 | 0 | - |
| `tuple` | 10610 | 0 | - |
| `span_index` | 12850 | 2240 | - |
| `span_loop` | 12850 | 2240 | - |
| `span_box` | 12920 | 2310 | - |

### vec3_construct_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10910 | 0 | - |
| `struct` | 10910 | 0 | - |
| `tuple` | 10910 | 0 | - |
| `span_index` | 13150 | 2240 | - |
| `span_loop` | 13150 | 2240 | - |
| `span_box` | 13220 | 2310 | - |

### vec3_construct_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10910 | 0 | - |
| `struct` | 10910 | 0 | - |
| `tuple` | 10910 | 0 | - |
| `span_index` | 13150 | 2240 | - |
| `span_loop` | 13150 | 2240 | - |
| `span_box` | 13220 | 2310 | - |

### vec3_cross_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 11510 | 900 | x1.00 |
| `struct` | 11510 | 900 | x1.00 |
| `tuple` | 11510 | 900 | x1.00 |
| `span_box` | 16560 | 5950 | x6.61 |
| `span_index` | 20030 | 9420 | x10.47 |
| `span_loop` | 41320 | 30710 | x34.12 |

### vec3_cross_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 77500 | 66590 | x1.00 |
| `struct` | 77500 | 66590 | x1.00 |
| `tuple` | 77500 | 66590 | x1.00 |
| `span_box` | 81550 | 70640 | x1.06 |
| `span_index` | 84420 | 73510 | x1.10 |
| `span_loop` | 104060 | 93150 | x1.40 |

### vec3_cross_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19900 | 8990 | x1.00 |
| `struct` | 19900 | 8990 | x1.00 |
| `tuple` | 19900 | 8990 | x1.00 |
| `span_box` | 23650 | 12740 | x1.42 |
| `span_index` | 26520 | 15610 | x1.74 |
| `span_loop` | 47380 | 36470 | x4.06 |

### vec3_dot_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10710 | 500 | x1.00 |
| `struct` | 10710 | 500 | x1.00 |
| `tuple` | 10710 | 500 | x1.00 |
| `span_box` | 13450 | 3240 | x6.48 |
| `span_index` | 16690 | 6480 | x12.96 |
| `span_loop` | 18420 | 8210 | x16.42 |

### vec3_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 45040 | 34730 | x1.00 |
| `struct` | 45040 | 34730 | x1.00 |
| `tuple` | 45040 | 34730 | x1.00 |
| `span_box` | 46580 | 36270 | x1.04 |
| `span_index` | 49120 | 38810 | x1.12 |
| `span_loop` | 52640 | 42330 | x1.22 |

### vec3_dot_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16240 | 5930 | x1.00 |
| `struct` | 16240 | 5930 | x1.00 |
| `tuple` | 16240 | 5930 | x1.00 |
| `span_box` | 18380 | 8070 | x1.36 |
| `span_index` | 20920 | 10610 | x1.79 |
| `span_loop` | 24220 | 13910 | x2.35 |

### vec3_equals_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 13410 | 400 | x1.00 |
| `struct` | 13410 | 400 | x1.00 |
| `tuple` | 13410 | 400 | x1.00 |
| `span_box` | 16150 | 3140 | x7.85 |
| `span_index` | 20090 | 7080 | x17.70 |
| `span_loop` | 22390 | 9380 | x23.45 |

### vec3_equals_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 13710 | 600 | x1.00 |
| `struct` | 13710 | 600 | x1.00 |
| `tuple` | 13710 | 600 | x1.00 |
| `span_box` | 16450 | 3340 | x5.57 |
| `span_index` | 20290 | 7180 | x11.97 |
| `span_loop` | 22690 | 9580 | x15.97 |

### vec3_equals_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 13710 | 600 | x1.00 |
| `struct` | 13710 | 600 | x1.00 |
| `tuple` | 13710 | 600 | x1.00 |
| `span_box` | 16450 | 3340 | x5.57 |
| `span_index` | 20290 | 7180 | x11.97 |
| `span_loop` | 22690 | 9580 | x15.97 |

### vec3_neg_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10910 | 300 | x1.00 |
| `struct` | 10910 | 300 | x1.00 |
| `tuple` | 10910 | 300 | x1.00 |
| `span_box` | 14590 | 3980 | x13.27 |
| `span_index` | 15390 | 4780 | x15.93 |
| `span_loop` | 20430 | 9820 | x32.73 |

### vec3_neg_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 11810 | 900 | x1.00 |
| `struct` | 11810 | 900 | x1.00 |
| `tuple` | 11810 | 900 | x1.00 |
| `span_box` | 15490 | 4580 | x5.09 |
| `span_index` | 16290 | 5380 | x5.98 |
| `span_loop` | 21330 | 10420 | x11.58 |

### vec3_neg_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 11810 | 900 | x1.00 |
| `struct` | 11810 | 900 | x1.00 |
| `tuple` | 11810 | 900 | x1.00 |
| `span_box` | 15490 | 4580 | x5.09 |
| `span_index` | 16290 | 5380 | x5.98 |
| `span_loop` | 21330 | 10420 | x11.58 |

### vec3_scale_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10910 | 300 | x1.00 |
| `struct` | 10910 | 300 | x1.00 |
| `tuple` | 10910 | 300 | x1.00 |
| `span_box` | 14590 | 3980 | x13.27 |
| `span_index` | 15390 | 4780 | x15.93 |
| `span_loop` | 20830 | 10220 | x34.07 |

### vec3_scale_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 44060 | 33150 | x1.00 |
| `struct` | 44060 | 33150 | x1.00 |
| `tuple` | 44060 | 33150 | x1.00 |
| `span_box` | 46740 | 35830 | x1.08 |
| `span_index` | 48340 | 37430 | x1.13 |
| `span_loop` | 51890 | 40980 | x1.24 |

### vec3_scale_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 13130 | 2220 | x1.00 |
| `struct` | 13130 | 2220 | x1.00 |
| `tuple` | 13130 | 2220 | x1.00 |
| `span_box` | 18140 | 7230 | x3.26 |
| `span_index` | 19740 | 8830 | x3.98 |
| `span_loop` | 23050 | 12140 | x5.47 |

### vec3_sub_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10910 | 300 | x1.00 |
| `struct` | 10910 | 300 | x1.00 |
| `tuple` | 10910 | 300 | x1.00 |
| `span_box` | 15960 | 5350 | x17.83 |
| `span_index` | 19430 | 8820 | x29.40 |
| `span_loop` | 22730 | 12120 | x40.40 |

### vec3_sub_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 15460 | 4550 | x1.00 |
| `struct` | 15460 | 4550 | x1.00 |
| `tuple` | 15460 | 4550 | x1.00 |
| `span_box` | 19210 | 8300 | x1.82 |
| `span_index` | 22080 | 11170 | x2.45 |
| `span_loop` | 24950 | 14040 | x3.09 |

### vec3_sub_i64

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 15460 | 4550 | x1.00 |
| `struct` | 15460 | 4550 | x1.00 |
| `tuple` | 15460 | 4550 | x1.00 |
| `span_box` | 19210 | 8300 | x1.82 |
| `span_index` | 22080 | 11170 | x2.45 |
| `span_loop` | 24950 | 14040 | x3.09 |

## layout::vecn::tests

### access_dyn_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct_match` | 9520 | 710 | x1.00 |
| `span` | 10080 | 1270 | x1.79 |
| `array_span` | 10380 | 1570 | x2.21 |

### access_y

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array_destructure` | 8410 | 0 | - |
| `struct_field` | 8410 | 0 | - |
| `tuple_destructure` | 8410 | 0 | - |
| `span_index` | 9580 | 1170 | - |
| `array_span_index` | 10080 | 1670 | - |

### vec2_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10590 | 1380 | x1.00 |
| `span_loop` | 19470 | 10260 | x7.43 |

### vec2_dot_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 9110 | 300 | x1.00 |
| `span_loop` | 14750 | 5940 | x19.80 |

### vec2_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 32260 | 23350 | x1.00 |
| `span_loop` | 38090 | 29180 | x1.25 |

### vec4_add_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16900 | 5490 | x1.00 |
| `struct` | 16900 | 5490 | x1.00 |
| `span_loop` | 29530 | 18120 | x3.30 |

### vec4_dot_felt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 11110 | 700 | x1.00 |
| `struct` | 11110 | 700 | x1.00 |
| `span_loop` | 20690 | 10280 | x14.69 |

### vec4_dot_fixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 56420 | 45910 | x1.00 |
| `struct` | 56420 | 45910 | x1.00 |
| `span_loop` | 65790 | 55280 | x1.20 |

