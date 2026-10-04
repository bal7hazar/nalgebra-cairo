# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_base::matrix2::tests

### matrix2_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 13980 | 3770 | x1.00 |

### matrix2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 16320 | 6600 | x1.00 |

### matrix2_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 14470 | 3260 | x1.00 |
| `operator` | 14470 | 3260 | x1.00 |

### matrix2_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 17830 | 6620 | x1.00 |

### matrix2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 10490 | 1780 | x1.00 |

### matrix2_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 9410 | 200 | x1.00 |

### matrix2_from_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10810 | 400 | x1.00 |

### matrix2_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 9610 | -200 | - |

### matrix2_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 9410 | -200 | - |

### matrix2_from_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10810 | 400 | x1.00 |

### matrix2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 9010 | -200 | - |

### matrix2_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 14920 | 6200 | x1.00 |

### matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 18630 | 7420 | x1.00 |
| `fused` | 18630 | 7420 | x1.00 |

### matrix2_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13470 | 3660 | x1.00 |

### matrix2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 11410 | 1200 | x1.00 |

### matrix2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10210 | 0 | - |

### matrix2_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 11130 | 2420 | x1.00 |

### matrix2_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 10890 | 2180 | x1.00 |

### matrix2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 17230 | 6620 | x1.00 |

### matrix2_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 14470 | 3260 | x1.00 |
| `operator` | 14470 | 3260 | x1.00 |

### matrix2_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18630 | 7420 | x1.00 |
| `transpose_mul` | 18630 | 7420 | x1.00 |

### matrix2_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13470 | 3660 | x1.00 |
| `transpose_mul_vec` | 13470 | 3660 | x1.00 |

### matrix2_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 9350 | 640 | x1.00 |

### matrix2_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10610 | 400 | x1.00 |

### matrix2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `prescaled_det_ge_half` | 44180 | 33970 | x1.00 |
| `prescaled_norm_gt_one` | 44180 | 33970 | x1.00 |
| `prescaled_small` | 44180 | 33970 | x1.00 |

### matrix2_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 42990 | 34270 | x1.00 |

### matrix2_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8410 | -800 | - |

## nalgebra_tests_base::matrix3::tests

### matrix3_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 23060 | 9350 | x1.00 |

### matrix3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 25570 | 13850 | x1.00 |

### matrix3_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 23170 | 7460 | x1.00 |
| `operator` | 23170 | 7460 | x1.00 |

### matrix3_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 30730 | 15020 | x1.00 |

### matrix3_cross_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 13110 | 600 | x1.00 |

### matrix3_cross_matrix_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `materialised` | 33430 | 18920 | x1.00 |

### matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 17330 | 7620 | x1.00 |
| `alt_triple_products` | 25100 | 15390 | x2.02 |

### matrix3_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 11010 | 300 | x1.00 |

### matrix3_from_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 15010 | 900 | x1.00 |

### matrix3_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 11610 | -900 | - |

### matrix3_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 11210 | -900 | - |

### matrix3_from_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 15010 | 900 | x1.00 |

### matrix3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 10810 | -900 | - |

### matrix3_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 22670 | 12950 | x1.00 |

### matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 34330 | 18620 | x1.00 |
| `fused` | 34330 | 18620 | x1.00 |

### matrix3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 17650 | 6140 | x1.00 |

### matrix3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 16410 | 2700 | x1.00 |

### matrix3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 13710 | 0 | - |

### matrix3_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 15250 | 5540 | x1.00 |

### matrix3_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 12890 | 3180 | x1.00 |

### matrix3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 29130 | 15020 | x1.00 |

### matrix3_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 23170 | 7460 | x1.00 |
| `operator` | 23170 | 7460 | x1.00 |

### matrix3_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 34330 | 18620 | x1.00 |
| `transpose_mul` | 34330 | 18620 | x1.00 |

### matrix3_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 17650 | 6140 | x1.00 |
| `transpose_mul_vec` | 17650 | 6140 | x1.00 |

### matrix3_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 11090 | 1380 | x1.00 |

### matrix3_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 14610 | 900 | x1.00 |

### matrix3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `prescaled_det_ge_half` | 63130 | 49420 | x1.00 |
| `prescaled_norm_gt_one` | 68940 | 55230 | x1.12 |
| `prescaled_small` | 85880 | 72170 | x1.46 |

### matrix3_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 40750 | 31030 | x1.00 |

### matrix3_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 9910 | -1800 | - |

## nalgebra_tests_base::matrix4::tests

### matrix4_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 36800 | 16190 | x1.00 |

### matrix4_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 38520 | 24000 | x1.00 |

### matrix4_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 37350 | 13340 | x1.00 |
| `operator` | 37350 | 13340 | x1.00 |

### matrix4_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 50790 | 26780 | x1.00 |

### matrix4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `minors2x2` | 39680 | 28570 | x1.00 |
| `alt_row_cofactors` | 58320 | 47210 | x1.65 |

### matrix4_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 13010 | 400 | x1.00 |

### matrix4_from_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22810 | 1600 | x1.00 |

### matrix4_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 16610 | -1600 | - |

### matrix4_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 16010 | -1600 | - |

### matrix4_from_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22810 | 1600 | x1.00 |

### matrix4_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 18810 | 1600 | x1.00 |

### matrix4_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 33520 | 22400 | x1.00 |

### matrix4_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 66920 | 42910 | x1.00 |
| `fused` | 66920 | 42910 | x1.00 |

### matrix4_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 22630 | 9020 | x1.00 |

### matrix4_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 25410 | 4800 | x1.00 |

### matrix4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22210 | 1600 | x1.00 |

### matrix4_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 18050 | 6940 | x1.00 |

### matrix4_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 15690 | 4580 | x1.00 |

### matrix4_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 47790 | 26780 | x1.00 |

### matrix4_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 37350 | 13340 | x1.00 |
| `operator` | 37350 | 13340 | x1.00 |

### matrix4_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 66920 | 42910 | x1.00 |
| `transpose_mul` | 66920 | 42910 | x1.00 |

### matrix4_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 22630 | 9020 | x1.00 |
| `transpose_mul_vec` | 22630 | 9020 | x1.00 |

### matrix4_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 13230 | 2120 | x1.00 |

### matrix4_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22210 | 1600 | x1.00 |

### matrix4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `prescaled_det_ge_half` | 217430 | 196820 | x1.00 |
| `prescaled_norm_gt_one` | 217430 | 196820 | x1.00 |
| `prescaled_small` | 217430 | 196820 | x1.00 |

### matrix4_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 206640 | 195520 | x1.00 |

### matrix4_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 12010 | -5200 | - |

## nalgebra_tests_base::matrix6::benches

### matrix6_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 78630 | 42020 | x1.00 |

### matrix6_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_differs` | 23860 | 1750 | x1.00 |
| `alt_blocks_first_differs` | 36560 | 14450 | x8.26 |
| `all_compared` | 70200 | 48090 | x27.48 |
| `alt_blocks` | 77510 | 55400 | x31.66 |
| `alt_not_inlined` | 95430 | 73320 | x41.90 |
| `alt_not_inlined_first_differs` | 95630 | 73520 | x42.01 |

### matrix6_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 74150 | 30140 | x1.00 |
| `operator` | 74150 | 30140 | x1.00 |

### matrix6_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 18210 | 600 | x1.00 |

### matrix6_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_diagonal_element` | 37010 | 3600 | x1.00 |
| `identity` | 37010 | 3600 | x1.00 |
| `zeros` | 37010 | 3600 | x1.00 |

### matrix6_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 25810 | -4800 | - |

### matrix6_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_differs` | 17910 | 3200 | x1.00 |
| `alt_blocks_first_differs` | 28260 | 13550 | x4.23 |
| `all_compared` | 62800 | 48090 | x15.03 |
| `alt_blocks` | 66510 | 51800 | x16.19 |
| `alt_not_inlined` | 84430 | 69720 | x21.79 |
| `alt_not_inlined_first_differs` | 84630 | 69920 | x21.85 |

### matrix6_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 152920 | 108910 | x1.00 |
| `fused` | 152920 | 108910 | x1.00 |
| `alt_blocks` | 236140 | 192130 | x1.76 |

### matrix6_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 34990 | 15980 | x1.00 |
| `alt_blocks` | 54540 | 35530 | x2.22 |

### matrix6_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 47410 | 10800 | x1.00 |

### matrix6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 33210 | 3600 | x1.00 |

### matrix6_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 97390 | 60380 | x1.00 |

### matrix6_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 74150 | 30140 | x1.00 |
| `operator` | 74150 | 30140 | x1.00 |

### matrix6_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul` | 152920 | 108910 | x1.00 |
| `fused` | 152920 | 108910 | x1.00 |

### matrix6_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_vec` | 34990 | 15980 | x1.00 |
| `fused` | 34990 | 15980 | x1.00 |

### matrix6_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 18710 | 3600 | x1.00 |

### matrix6_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 40210 | 3600 | x1.00 |

## nalgebra_tests_base::point2::benches

### point2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 12350 | 3030 | x1.00 |
| `within` | 13850 | 4530 | x1.50 |

### point2_add_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_assign` | 10990 | 1580 | x1.00 |
| `add_vector` | 10990 | 1580 | x1.00 |

### point2_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_lerp` | 13270 | 3860 | x1.00 |
| `alt_add_scale` | 14150 | 4740 | x1.23 |
| `alt_add_div` | 17030 | 7620 | x1.97 |

### point2_coords

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 9010 | 200 | x1.00 |
| `into_vector` | 9010 | 200 | x1.00 |

### point2_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unfused` | 14390 | 5480 | x1.00 |

### point2_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10210 | 200 | x1.00 |
| `from_coordinates` | 10210 | 200 | x1.00 |
| `vector` | 10210 | 200 | x1.00 |

### point2_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 15720 | 6410 | x1.00 |
| `from_homogeneous` | 16140 | 6830 | x1.07 |

### point2_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 11260 | 1850 | x1.00 |
| `sup` | 11260 | 1850 | x1.00 |
| `inf_sup` | 14210 | 4800 | x2.59 |

### point2_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 8310 | 0 | - |

### point2_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 13670 | 3860 | x1.00 |

### point2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 9410 | 600 | x1.00 |

### point2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 9210 | 200 | x1.00 |

### point2_origin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `origin` | 9210 | 200 | x1.00 |

### point2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 12470 | 3260 | x1.00 |
| `scale` | 12470 | 3260 | x1.00 |

### point2_sub_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_point` | 10990 | 1580 | x1.00 |

### point2_sub_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_assign` | 10990 | 1580 | x1.00 |
| `sub_vector` | 10990 | 1580 | x1.00 |

### point2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 9610 | 300 | x1.00 |

### point2_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 15840 | 6630 | x1.00 |
| `unscale` | 15840 | 6630 | x1.00 |

## nalgebra_tests_base::point3::benches

### point3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 12880 | 3160 | x1.00 |
| `within` | 16260 | 6540 | x2.07 |

### point3_add_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_assign` | 12730 | 2420 | x1.00 |
| `add_vector` | 12730 | 2420 | x1.00 |

### point3_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_lerp` | 16150 | 5840 | x1.00 |
| `alt_add_scale` | 17470 | 7160 | x1.23 |
| `alt_add_div` | 21520 | 11210 | x1.92 |

### point3_coords

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 9810 | 300 | x1.00 |
| `into_vector` | 9810 | 300 | x1.00 |

### point3_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unfused` | 17950 | 8640 | x1.00 |

### point3_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 11410 | 300 | x1.00 |
| `from_coordinates` | 11410 | 300 | x1.00 |
| `vector` | 11410 | 300 | x1.00 |

### point3_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 18100 | 8090 | x1.00 |
| `from_homogeneous` | 19720 | 9710 | x1.20 |

### point3_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 13140 | 2830 | x1.00 |
| `sup` | 13140 | 2830 | x1.00 |
| `inf_sup` | 17570 | 7260 | x2.57 |

### point3_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 8510 | 0 | - |

### point3_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 16550 | 5840 | x1.00 |

### point3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 10410 | 900 | x1.00 |

### point3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 10210 | 300 | x1.00 |

### point3_origin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `origin` | 9910 | 300 | x1.00 |

### point3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 14850 | 4940 | x1.00 |
| `scale` | 14850 | 4940 | x1.00 |

### point3_sub_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_point` | 12730 | 2420 | x1.00 |

### point3_sub_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_assign` | 12730 | 2420 | x1.00 |
| `sub_vector` | 12730 | 2420 | x1.00 |

### point3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 10410 | 400 | x1.00 |

### point3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 19420 | 9510 | x1.00 |
| `unscale` | 19420 | 9510 | x1.00 |

### point3_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 9210 | 200 | x1.00 |

## nalgebra_tests_base::unit::benches

### unit2_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `x_axis` | 9210 | 200 | x1.00 |
| `y_axis` | 9210 | 200 | x1.00 |

### unit2_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 11290 | 1780 | x1.00 |

### unit2_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 18360 | 8750 | x1.00 |

### unit2_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 15730 | 6920 | x1.00 |

### unit3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 12880 | 3160 | x1.00 |
| `within` | 16260 | 6540 | x2.07 |

### unit3_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `x_axis` | 9910 | 300 | x1.00 |
| `y_axis` | 9910 | 300 | x1.00 |
| `z_axis` | 9910 | 300 | x1.00 |

### unit3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 12090 | 1980 | x1.00 |
| `alt_unfused` | 16530 | 6420 | x3.24 |

### unit3_into_inner

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_inner` | 9510 | 0 | - |

### unit3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 10410 | 900 | x1.00 |

### unit3_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 22140 | 11830 | x1.00 |
| `new_and_get` | 22440 | 12130 | x1.03 |
| `try_new` | 22810 | 12500 | x1.06 |
| `try_new_and_get` | 23610 | 13300 | x1.12 |

### unit3_new_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_unchecked` | 9510 | 0 | - |

### unit3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `renormalize` | 21340 | 11830 | x1.00 |

### unit3_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 18310 | 8800 | x1.00 |
| `alt_upstream` | 19150 | 9640 | x1.10 |
| `alt_mul_add` | 19750 | 10240 | x1.16 |
| `alt_scale` | 19990 | 10480 | x1.19 |
| `alt_lerp` | 21250 | 11740 | x1.33 |
| `alt_exact` | 21340 | 11830 | x1.34 |

### unit3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale` | 14850 | 4940 | x1.00 |

### unit4_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `w_axis` | 10610 | 400 | x1.00 |
| `x_axis` | 10610 | 400 | x1.00 |
| `y_axis` | 10610 | 400 | x1.00 |
| `z_axis` | 10610 | 400 | x1.00 |

### unit4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 12890 | 2180 | x1.00 |

### unit4_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 25790 | 14780 | x1.00 |

### unit4_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 20890 | 10680 | x1.00 |

## nalgebra_tests_base::vector2::benches

### vector2_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 10780 | 1970 | x1.00 |

### vector2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs_diff_eq` | 12110 | 3190 | x1.00 |

### vector2_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 10990 | 1580 | x1.00 |
| `add_assign` | 10990 | 1580 | x1.00 |

### vector2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 47700 | 38790 | x1.00 |
| `half_angle` | 70110 | 61200 | x1.58 |

### vector2_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 12880 | 3670 | x1.00 |
| `capped` | 18690 | 9480 | x2.58 |
| `alt_normalize` | 22090 | 12880 | x3.51 |

### vector2_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 16310 | 6900 | x1.00 |

### vector2_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 12670 | 3260 | x1.00 |

### vector2_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 10690 | 1780 | x1.00 |
| `alt_unfused` | 12910 | 4000 | x2.25 |

### vector2_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_element` | 9210 | 200 | x1.00 |
| `repeat` | 9210 | 200 | x1.00 |
| `x` | 9210 | 200 | x1.00 |
| `y` | 9210 | 200 | x1.00 |
| `zeros` | 9210 | 200 | x1.00 |

### vector2_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 9610 | 200 | x1.00 |

### vector2_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imin` | 9080 | 770 | x1.00 |
| `imax` | 9090 | 780 | x1.01 |
| `iamax` | 10950 | 2640 | x3.43 |
| `iamin` | 10960 | 2650 | x3.44 |

### vector2_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 11260 | 1850 | x1.00 |
| `sup` | 11260 | 1850 | x1.00 |

### vector2_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 14210 | 3800 | x1.00 |

### vector2_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 8310 | 0 | - |

### vector2_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 8710 | 300 | x1.00 |

### vector2_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 13670 | 3860 | x1.00 |

### vector2_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 12410 | 3500 | x1.00 |

### vector2_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 9090 | 780 | x1.00 |
| `min` | 9090 | 780 | x1.00 |
| `amax` | 10950 | 2640 | x3.38 |
| `amin` | 10950 | 2640 | x3.38 |

### vector2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 9410 | 600 | x1.00 |

### vector2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 9210 | 200 | x1.00 |

### vector2_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude` | 10330 | 2020 | x1.00 |
| `norm` | 10330 | 2020 | x1.00 |

### vector2_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude_squared` | 10090 | 1780 | x1.00 |
| `norm_squared` | 10090 | 1780 | x1.00 |

### vector2_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 16840 | 8030 | x1.00 |
| `unscale` | 17560 | 8750 | x1.09 |
| `alt_recip_sqrt` | 18520 | 9710 | x1.21 |

### vector2_perp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `perp` | 10690 | 1780 | x1.00 |

### vector2_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 9710 | 0 | - |
| `push` | 10010 | 300 | - |

### vector2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 12470 | 3260 | x1.00 |
| `scale` | 12470 | 3260 | x1.00 |

### vector2_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub` | 10990 | 1580 | x1.00 |
| `sub_assign` | 10990 | 1580 | x1.00 |

### vector2_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 8950 | 640 | x1.00 |

### vector2_try_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 12740 | 3230 | x1.00 |
| `some` | 18930 | 9420 | x2.92 |

### vector2_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 15120 | 5910 | x1.00 |
| `div_assign` | 15840 | 6630 | x1.12 |
| `unscale` | 15840 | 6630 | x1.12 |

## nalgebra_tests_base::vector3::benches

### vector3_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 12380 | 2870 | x1.00 |

### vector3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs_diff_eq` | 13850 | 4530 | x1.00 |

### vector3_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 12730 | 2420 | x1.00 |
| `add_assign` | 12730 | 2420 | x1.00 |

### vector3_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 48430 | 39120 | x1.00 |
| `half_angle` | 81050 | 71740 | x1.83 |

### vector3_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 14080 | 4170 | x1.00 |
| `capped` | 21370 | 11460 | x2.75 |
| `alt_normalize` | 27550 | 17640 | x4.23 |

### vector3_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 20440 | 10130 | x1.00 |

### vector3_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 15250 | 4940 | x1.00 |

### vector3_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross` | 15850 | 5540 | x1.00 |

### vector3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 11290 | 1980 | x1.00 |
| `alt_unfused` | 15730 | 6420 | x3.24 |

### vector3_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_element` | 9910 | 300 | x1.00 |
| `repeat` | 9910 | 300 | x1.00 |
| `x` | 9910 | 300 | x1.00 |
| `y` | 9910 | 300 | x1.00 |
| `z` | 9910 | 300 | x1.00 |
| `zeros` | 9910 | 300 | x1.00 |

### vector3_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 10610 | 300 | x1.00 |

### vector3_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imax` | 10260 | 1750 | x1.00 |
| `imin` | 10260 | 1750 | x1.00 |
| `iamax` | 12920 | 4410 | x2.52 |
| `iamin` | 12940 | 4430 | x2.53 |

### vector3_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 13140 | 2830 | x1.00 |
| `sup` | 13140 | 2830 | x1.00 |

### vector3_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 17570 | 5760 | x1.00 |

### vector3_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 8510 | 0 | - |

### vector3_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 8910 | 300 | x1.00 |

### vector3_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 16550 | 5840 | x1.00 |

### vector3_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 13750 | 4440 | x1.00 |

### vector3_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 10160 | 1650 | x1.00 |
| `min` | 10160 | 1650 | x1.00 |
| `amax` | 12820 | 4310 | x2.61 |
| `amin` | 12820 | 4310 | x2.61 |

### vector3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 10410 | 900 | x1.00 |

### vector3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 10210 | 300 | x1.00 |

### vector3_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude` | 10730 | 2220 | x1.00 |
| `norm` | 10730 | 2220 | x1.00 |

### vector3_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude_squared` | 10490 | 1980 | x1.00 |
| `norm_squared` | 10490 | 1980 | x1.00 |

### vector3_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 19420 | 9910 | x1.00 |
| `alt_recip_sqrt` | 21100 | 11590 | x1.17 |
| `unscale` | 21340 | 11830 | x1.19 |
| `alt_per_element_div` | 21690 | 12180 | x1.23 |

### vector3_orthonormal_basis_zneg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_upstream` | 28950 | 17940 | x1.00 |

### vector3_orthonormal_basis_zpos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_upstream` | 29220 | 18210 | x1.00 |

### vector3_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 10510 | 100 | x1.00 |
| `push` | 10810 | 400 | x4.00 |

### vector3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 14850 | 4940 | x1.00 |
| `scale` | 14850 | 4940 | x1.00 |

### vector3_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub` | 12730 | 2420 | x1.00 |
| `sub_assign` | 12730 | 2420 | x1.00 |

### vector3_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 9890 | 1380 | x1.00 |

### vector3_try_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 13710 | 3500 | x1.00 |
| `some` | 22710 | 12500 | x3.57 |

### vector3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 17500 | 7590 | x1.00 |
| `div_assign` | 19420 | 9510 | x1.25 |
| `unscale` | 19420 | 9510 | x1.25 |

### vector3_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 9210 | 200 | x1.00 |

## nalgebra_tests_base::vector4::benches

### vector4_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 14250 | 4040 | x1.00 |

### vector4_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs_diff_eq` | 15590 | 5870 | x1.00 |

### vector4_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 14470 | 3260 | x1.00 |
| `add_assign` | 14470 | 3260 | x1.00 |

### vector4_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 49700 | 39990 | x1.00 |
| `half_angle` | 91690 | 81980 | x2.05 |

### vector4_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 15280 | 4670 | x1.00 |
| `capped` | 24050 | 13440 | x2.88 |
| `alt_normalize` | 33150 | 22540 | x4.83 |

### vector4_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 24840 | 13630 | x1.00 |

### vector4_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 17830 | 6620 | x1.00 |

### vector4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 11890 | 2180 | x1.00 |
| `alt_unfused` | 18550 | 8840 | x4.06 |

### vector4_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_element` | 10610 | 400 | x1.00 |
| `repeat` | 10610 | 400 | x1.00 |
| `w` | 10610 | 400 | x1.00 |
| `x` | 10610 | 400 | x1.00 |
| `y` | 10610 | 400 | x1.00 |
| `z` | 10610 | 400 | x1.00 |
| `zeros` | 10610 | 400 | x1.00 |

### vector4_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 11610 | 400 | x1.00 |

### vector4_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imin` | 11430 | 2720 | x1.00 |
| `imax` | 11440 | 2730 | x1.00 |
| `iamax` | 15170 | 6460 | x2.38 |
| `iamin` | 15170 | 6460 | x2.38 |

### vector4_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 15010 | 3800 | x1.00 |
| `sup` | 15010 | 3800 | x1.00 |

### vector4_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 20910 | 7700 | x1.00 |

### vector4_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 8710 | 0 | - |

### vector4_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 9110 | 300 | x1.00 |

### vector4_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 19430 | 7820 | x1.00 |

### vector4_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 15090 | 5380 | x1.00 |

### vector4_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `min` | 11230 | 2520 | x1.00 |
| `max` | 11240 | 2530 | x1.00 |
| `amax` | 14970 | 6260 | x2.48 |
| `amin` | 14970 | 6260 | x2.48 |

### vector4_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 11410 | 1200 | x1.00 |

### vector4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 11210 | 400 | x1.00 |

### vector4_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude` | 11130 | 2420 | x1.00 |
| `norm` | 11130 | 2420 | x1.00 |

### vector4_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude_squared` | 10890 | 2180 | x1.00 |
| `norm_squared` | 10890 | 2180 | x1.00 |

### vector4_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 22000 | 11790 | x1.00 |
| `alt_recip_sqrt` | 23680 | 13470 | x1.14 |
| `unscale` | 25260 | 15050 | x1.28 |

### vector4_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 17230 | 6620 | x1.00 |
| `scale` | 17230 | 6620 | x1.00 |

### vector4_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub` | 14470 | 3260 | x1.00 |
| `sub_assign` | 14470 | 3260 | x1.00 |

### vector4_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 10830 | 2120 | x1.00 |

### vector4_try_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 14660 | 3750 | x1.00 |
| `some` | 26630 | 15720 | x4.19 |

### vector4_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 19880 | 9270 | x1.00 |
| `div_assign` | 23140 | 12530 | x1.35 |
| `unscale` | 23140 | 12530 | x1.35 |

### vector4_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 9410 | 200 | x1.00 |

### vector4_xyz

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xyz` | 10010 | 300 | x1.00 |

## nalgebra_tests_base::vector6::benches

### vector6_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 17450 | 5840 | x1.00 |

### vector6_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 17690 | 7580 | x1.00 |

### vector6_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 17950 | 4940 | x1.00 |
| `operator` | 17950 | 4940 | x1.00 |

### vector6_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 22990 | 9980 | x1.00 |

### vector6_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 13090 | 2580 | x1.00 |
| `alt_two_sum_prod3` | 15310 | 4800 | x1.86 |
| `alt_unfused` | 24190 | 13680 | x5.30 |

### vector6_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `zeros` | 12010 | 600 | x1.00 |

### vector6_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 18760 | 5750 | x1.00 |
| `sup` | 18760 | 5750 | x1.00 |

### vector6_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25190 | 11780 | x1.00 |

### vector6_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 13410 | 1800 | x1.00 |

### vector6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 11210 | 600 | x1.00 |

### vector6_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_via_norm_squared` | 13610 | 4500 | x1.00 |
| `wide_sqrt` | 14050 | 4940 | x1.10 |

### vector6_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 11690 | 2580 | x1.00 |
| `alt_two_blocks` | 13910 | 4800 | x1.86 |

### vector6_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unscale_by_norm` | 34680 | 23070 | x1.00 |

### vector6_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 21990 | 9980 | x1.00 |
| `scale` | 21990 | 9980 | x1.00 |

### vector6_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 17950 | 4940 | x1.00 |
| `operator` | 17950 | 4940 | x1.00 |

### vector6_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 12710 | 3600 | x1.00 |

### vector6_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 30040 | 18030 | x1.00 |
| `unscale` | 30040 | 18030 | x1.00 |

