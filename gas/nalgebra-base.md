# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::base::matrix2::tests

### matrix2_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 21810 | 3770 | x1.00 |

### matrix2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 24150 | 6600 | x1.00 |

### matrix2_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 22300 | 3260 | x1.00 |
| `operator` | 22300 | 3260 | x1.00 |

### matrix2_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 18840 | 800 | x1.00 |

### matrix2_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 17240 | 200 | x1.00 |

### matrix2_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 25660 | 6620 | x1.00 |

### matrix2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18320 | 1780 | x1.00 |

### matrix2_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17240 | 200 | x1.00 |

### matrix2_from_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18640 | 400 | x1.00 |

### matrix2_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17440 | -200 | - |

### matrix2_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17240 | -200 | - |

### matrix2_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 24860 | 6620 | x1.00 |

### matrix2_from_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18640 | 400 | x1.00 |

### matrix2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16840 | -200 | - |

### matrix2_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 22750 | 6200 | x1.00 |

### matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 28990 | 9950 | x1.00 |
| `fused` | 28990 | 9950 | x1.00 |

### matrix2_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 25210 | 7670 | x1.00 |
| `generic` | 27790 | 10250 | x1.34 |

### matrix2_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21300 | 3660 | x1.00 |

### matrix2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 19240 | 1200 | x1.00 |

### matrix2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18040 | 0 | - |

### matrix2_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18960 | 2420 | x1.00 |

### matrix2_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18720 | 2180 | x1.00 |

### matrix2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 25060 | 6620 | x1.00 |

### matrix2_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 22300 | 3260 | x1.00 |
| `operator` | 22300 | 3260 | x1.00 |

### matrix2_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 28990 | 9950 | x1.00 |
| `transpose_mul` | 28990 | 9950 | x1.00 |

### matrix2_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21300 | 3660 | x1.00 |
| `transpose_mul_vec` | 21300 | 3660 | x1.00 |

### matrix2_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 17180 | 640 | x1.00 |

### matrix2_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18440 | 400 | x1.00 |

### matrix2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 33330 | 15290 | x1.00 |
| `alt_div_n` | 39610 | 21570 | x1.41 |
| `alt_div` | 40380 | 22340 | x1.46 |
| `prescaled_det_ge_half` | 52010 | 33970 | x2.22 |
| `prescaled_norm_gt_one` | 52010 | 33970 | x2.22 |
| `prescaled_small` | 52010 | 33970 | x2.22 |

### matrix2_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 50820 | 34270 | x1.00 |

### matrix2_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16240 | -800 | - |

## nalgebra::base::matrix3::tests

### matrix3_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 30890 | 9350 | x1.00 |

### matrix3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 33400 | 13850 | x1.00 |

### matrix3_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 31000 | 7460 | x1.00 |
| `operator` | 31000 | 7460 | x1.00 |

### matrix3_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 38360 | 16820 | x1.00 |

### matrix3_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 18840 | 300 | x1.00 |

### matrix3_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 38560 | 15020 | x1.00 |

### matrix3_cross_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 20940 | 600 | x1.00 |

### matrix3_cross_matrix_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 42090 | 19750 | x1.00 |
| `materialised` | 45090 | 22750 | x1.15 |

### matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 27890 | 10350 | x1.00 |
| `alt_triple_products` | 32930 | 15390 | x1.49 |

### matrix3_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18840 | 300 | x1.00 |

### matrix3_from_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22840 | 900 | x1.00 |

### matrix3_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 19440 | -900 | - |

### matrix3_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 19040 | -900 | - |

### matrix3_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 36160 | 15020 | x1.00 |

### matrix3_from_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22840 | 900 | x1.00 |

### matrix3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 18640 | -900 | - |

### matrix3_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 30500 | 12950 | x1.00 |

### matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 45690 | 22150 | x1.00 |
| `fused` | 45690 | 22150 | x1.00 |

### matrix3_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 35050 | 15010 | x1.00 |
| `generic` | 42790 | 22750 | x1.52 |

### matrix3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25480 | 6140 | x1.00 |

### matrix3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 24240 | 2700 | x1.00 |

### matrix3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 21540 | 0 | - |

### matrix3_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 23080 | 5540 | x1.00 |

### matrix3_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20720 | 3180 | x1.00 |

### matrix3_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 18840 | 300 | x1.00 |

### matrix3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 36960 | 15020 | x1.00 |

### matrix3_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 31000 | 7460 | x1.00 |
| `operator` | 31000 | 7460 | x1.00 |

### matrix3_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 45690 | 22150 | x1.00 |
| `transpose_mul` | 45690 | 22150 | x1.00 |

### matrix3_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25480 | 6140 | x1.00 |
| `transpose_mul_vec` | 25480 | 6140 | x1.00 |

### matrix3_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 18920 | 1380 | x1.00 |

### matrix3_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 22440 | 900 | x1.00 |

### matrix3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 70620 | 49080 | x1.00 |
| `alt_div_n` | 88400 | 66860 | x1.36 |
| `alt_div` | 91470 | 69930 | x1.42 |
| `prescaled_det_ge_half` | 109900 | 88360 | x1.80 |
| `prescaled_norm_gt_one` | 109900 | 88360 | x1.80 |
| `prescaled_small` | 109900 | 88360 | x1.80 |

### matrix3_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 106210 | 88660 | x1.00 |

### matrix3_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 17740 | -1800 | - |

## nalgebra::base::matrix4::tests

### matrix4_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 44630 | 16190 | x1.00 |

### matrix4_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 46350 | 24000 | x1.00 |

### matrix4_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 45180 | 13340 | x1.00 |
| `operator` | 45180 | 13340 | x1.00 |

### matrix4_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 93290 | 64850 | x1.00 |

### matrix4_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 20840 | 400 | x1.00 |

### matrix4_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 58620 | 26780 | x1.00 |

### matrix4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `minors2x2` | 47510 | 28570 | x1.00 |
| `alt_row_cofactors` | 66150 | 47210 | x1.65 |

### matrix4_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 20840 | 400 | x1.00 |

### matrix4_from_columns

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 30640 | 1600 | x1.00 |

### matrix4_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 24440 | -1600 | - |

### matrix4_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 23840 | -1600 | - |

### matrix4_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 53820 | 26780 | x1.00 |

### matrix4_from_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 30640 | 1600 | x1.00 |

### matrix4_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 26640 | 1600 | x1.00 |

### matrix4_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 41350 | 22400 | x1.00 |

### matrix4_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 74750 | 42910 | x1.00 |
| `fused` | 74750 | 42910 | x1.00 |

### matrix4_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30460 | 9020 | x1.00 |

### matrix4_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 33240 | 4800 | x1.00 |

### matrix4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 30040 | 1600 | x1.00 |

### matrix4_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 25880 | 6940 | x1.00 |

### matrix4_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 23520 | 4580 | x1.00 |

### matrix4_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 20840 | 400 | x1.00 |

### matrix4_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 55620 | 26780 | x1.00 |

### matrix4_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 45180 | 13340 | x1.00 |
| `operator` | 45180 | 13340 | x1.00 |

### matrix4_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 74750 | 42910 | x1.00 |
| `transpose_mul` | 74750 | 42910 | x1.00 |

### matrix4_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30460 | 9020 | x1.00 |
| `transpose_mul_vec` | 30460 | 9020 | x1.00 |

### matrix4_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 21060 | 2120 | x1.00 |

### matrix4_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 30040 | 1600 | x1.00 |

### matrix4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 156230 | 127790 | x1.00 |
| `alt_div_n` | 190110 | 161670 | x1.27 |
| `alt_div` | 196400 | 167960 | x1.31 |
| `prescaled_det_ge_half` | 225260 | 196820 | x1.54 |
| `prescaled_norm_gt_one` | 225260 | 196820 | x1.54 |
| `prescaled_small` | 225260 | 196820 | x1.54 |

### matrix4_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 214470 | 195520 | x1.00 |

### matrix4_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 19840 | -5200 | - |

## nalgebra::base::matrix6::benches

### matrix6_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 86460 | 42020 | x1.00 |

### matrix6_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_differs` | 31690 | 1750 | x1.00 |
| `alt_blocks_first_differs` | 44390 | 14450 | x8.26 |
| `all_compared` | 78030 | 48090 | x27.48 |
| `alt_blocks` | 85340 | 55400 | x31.66 |
| `alt_not_inlined` | 103260 | 73320 | x41.90 |
| `alt_not_inlined_first_differs` | 103460 | 73520 | x42.01 |

### matrix6_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 81980 | 30140 | x1.00 |
| `operator` | 81980 | 30140 | x1.00 |

### matrix6_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 26040 | 600 | x1.00 |

### matrix6_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_diagonal_element` | 44840 | 3600 | x1.00 |
| `identity` | 44840 | 3600 | x1.00 |
| `zeros` | 44840 | 3600 | x1.00 |

### matrix6_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 33640 | -4800 | - |

### matrix6_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_differs` | 25740 | 3200 | x1.00 |
| `alt_blocks_first_differs` | 36090 | 13550 | x4.23 |
| `all_compared` | 70630 | 48090 | x15.03 |
| `alt_blocks` | 74340 | 51800 | x16.19 |
| `alt_not_inlined` | 92260 | 69720 | x21.79 |
| `alt_not_inlined_first_differs` | 92460 | 69920 | x21.85 |

### matrix6_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 160750 | 108910 | x1.00 |
| `fused` | 160750 | 108910 | x1.00 |
| `alt_blocks` | 272210 | 220370 | x2.02 |

### matrix6_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 48750 | 21910 | x1.00 |
| `alt_blocks` | 62370 | 35530 | x1.62 |

### matrix6_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 55240 | 10800 | x1.00 |

### matrix6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 41040 | 3600 | x1.00 |

### matrix6_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 105220 | 60380 | x1.00 |

### matrix6_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 81980 | 30140 | x1.00 |
| `operator` | 81980 | 30140 | x1.00 |

### matrix6_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul` | 160750 | 108910 | x1.00 |
| `fused` | 160750 | 108910 | x1.00 |

### matrix6_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_vec` | 48750 | 21910 | x1.00 |
| `fused` | 48750 | 21910 | x1.00 |

### matrix6_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 26540 | 3600 | x1.00 |

### matrix6_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 48040 | 3600 | x1.00 |

## nalgebra::base::point2::benches

### point2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 20180 | 3030 | x1.00 |
| `within` | 21680 | 4530 | x1.50 |

### point2_add_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_assign` | 18820 | 1580 | x1.00 |
| `add_vector` | 18820 | 1580 | x1.00 |

### point2_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 20900 | 3660 | x1.00 |
| `alt_lerp` | 21100 | 3860 | x1.05 |
| `alt_add_scale` | 21980 | 4740 | x1.30 |
| `alt_add_div` | 24860 | 7620 | x2.08 |

### point2_coords

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 16840 | 200 | x1.00 |
| `into_vector` | 16840 | 200 | x1.00 |

### point2_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 20240 | 3500 | x1.00 |
| `alt_sqrt` | 21920 | 5180 | x1.48 |

### point2_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20000 | 3260 | x1.00 |
| `alt_unfused` | 22220 | 5480 | x1.68 |

### point2_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18040 | 200 | x1.00 |
| `from_coordinates` | 18040 | 200 | x1.00 |
| `vector` | 18040 | 200 | x1.00 |

### point2_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 23550 | 6410 | x1.00 |
| `from_homogeneous` | 23970 | 6830 | x1.07 |

### point2_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 19090 | 1850 | x1.00 |
| `sup` | 19090 | 1850 | x1.00 |
| `inf_sup` | 22040 | 4800 | x2.59 |

### point2_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16140 | 0 | - |

### point2_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 21500 | 3860 | x1.00 |

### point2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 17240 | 600 | x1.00 |

### point2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 17040 | 200 | x1.00 |

### point2_origin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `origin` | 17040 | 200 | x1.00 |

### point2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 20300 | 3260 | x1.00 |
| `scale` | 20300 | 3260 | x1.00 |

### point2_sub_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_point` | 18820 | 1580 | x1.00 |

### point2_sub_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_assign` | 18820 | 1580 | x1.00 |
| `sub_vector` | 18820 | 1580 | x1.00 |

### point2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 17440 | 300 | x1.00 |

### point2_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 23670 | 6630 | x1.00 |
| `unscale` | 23670 | 6630 | x1.00 |

## nalgebra::base::point3::benches

### point3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 20710 | 3160 | x1.00 |
| `within` | 24090 | 6540 | x2.07 |

### point3_add_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_assign` | 20560 | 2420 | x1.00 |
| `add_vector` | 20560 | 2420 | x1.00 |

### point3_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 23680 | 5540 | x1.00 |
| `alt_lerp` | 23980 | 5840 | x1.05 |
| `alt_add_scale` | 25300 | 7160 | x1.29 |
| `alt_add_div` | 29350 | 11210 | x2.02 |

### point3_coords

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 17640 | 300 | x1.00 |
| `into_vector` | 17640 | 300 | x1.00 |

### point3_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 21580 | 4440 | x1.00 |
| `alt_sqrt` | 23260 | 6120 | x1.38 |

### point3_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21340 | 4200 | x1.00 |
| `alt_unfused` | 25780 | 8640 | x2.06 |

### point3_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19240 | 300 | x1.00 |
| `from_coordinates` | 19240 | 300 | x1.00 |
| `vector` | 19240 | 300 | x1.00 |

### point3_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 25930 | 8090 | x1.00 |
| `from_homogeneous` | 27550 | 9710 | x1.20 |

### point3_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 20970 | 2830 | x1.00 |
| `sup` | 20970 | 2830 | x1.00 |
| `inf_sup` | 25400 | 7260 | x2.57 |

### point3_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16340 | 0 | - |

### point3_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 24380 | 5840 | x1.00 |

### point3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 18240 | 900 | x1.00 |

### point3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 18040 | 300 | x1.00 |

### point3_origin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `origin` | 17740 | 300 | x1.00 |

### point3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 22680 | 4940 | x1.00 |
| `scale` | 22680 | 4940 | x1.00 |

### point3_sub_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_point` | 20560 | 2420 | x1.00 |

### point3_sub_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub_assign` | 20560 | 2420 | x1.00 |
| `sub_vector` | 20560 | 2420 | x1.00 |

### point3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 18240 | 400 | x1.00 |

### point3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 27250 | 9510 | x1.00 |
| `unscale` | 27250 | 9510 | x1.00 |

### point3_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 17040 | 200 | x1.00 |

## nalgebra::base::sym_matrix2::tests

### sym_matrix2_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 32730 | 14590 | x1.00 |
| `generic` | 38440 | 20300 | x1.39 |

### sym_matrix2_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18240 | 400 | x1.00 |

## nalgebra::base::sym_matrix3::tests

### sym_matrix3_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 51270 | 30430 | x1.00 |
| `generic` | 65840 | 45000 | x1.48 |

### sym_matrix3_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 21840 | 900 | x1.00 |

## nalgebra::base::unit::benches

### unit2_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `x_axis` | 17040 | 200 | x1.00 |
| `y_axis` | 17040 | 200 | x1.00 |

### unit2_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 19120 | 1780 | x1.00 |

### unit2_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 26190 | 8750 | x1.00 |

### unit2_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 23560 | 6920 | x1.00 |

### unit3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 20710 | 3160 | x1.00 |
| `within` | 24090 | 6540 | x2.07 |

### unit3_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `x_axis` | 17740 | 300 | x1.00 |
| `y_axis` | 17740 | 300 | x1.00 |
| `z_axis` | 17740 | 300 | x1.00 |

### unit3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 19920 | 1980 | x1.00 |
| `alt_unfused` | 24360 | 6420 | x3.24 |

### unit3_into_inner

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into_inner` | 17340 | 0 | - |

### unit3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 18240 | 900 | x1.00 |

### unit3_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 29970 | 11830 | x1.00 |
| `new_and_get` | 30270 | 12130 | x1.03 |
| `try_new` | 30640 | 12500 | x1.06 |
| `try_new_and_get` | 31440 | 13300 | x1.12 |

### unit3_new_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_unchecked` | 17340 | 0 | - |

### unit3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `renormalize` | 29170 | 11830 | x1.00 |

### unit3_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 26140 | 8800 | x1.00 |
| `alt_upstream` | 26980 | 9640 | x1.10 |
| `alt_mul_add` | 27580 | 10240 | x1.16 |
| `alt_scale` | 27820 | 10480 | x1.19 |
| `alt_lerp` | 29080 | 11740 | x1.33 |
| `alt_exact` | 29170 | 11830 | x1.34 |

### unit3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale` | 22680 | 4940 | x1.00 |

### unit4_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `w_axis` | 18440 | 400 | x1.00 |
| `x_axis` | 18440 | 400 | x1.00 |
| `y_axis` | 18440 | 400 | x1.00 |
| `z_axis` | 18440 | 400 | x1.00 |

### unit4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 20720 | 2180 | x1.00 |

### unit4_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 33620 | 14780 | x1.00 |

### unit4_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 28720 | 10680 | x1.00 |

## nalgebra::base::vector2::benches

### vector2_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 18610 | 1970 | x1.00 |

### vector2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs_diff_eq` | 19940 | 3190 | x1.00 |

### vector2_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 18820 | 1580 | x1.00 |
| `add_assign` | 18820 | 1580 | x1.00 |

### vector2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 55530 | 38790 | x1.00 |
| `half_angle` | 77940 | 61200 | x1.58 |

### vector2_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 20710 | 3670 | x1.00 |
| `capped` | 26520 | 9480 | x2.58 |
| `alt_normalize` | 29920 | 12880 | x3.51 |

### vector2_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 24140 | 6900 | x1.00 |

### vector2_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 20500 | 3260 | x1.00 |

### vector2_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18520 | 1780 | x1.00 |
| `alt_unfused` | 20740 | 4000 | x2.25 |

### vector2_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_element` | 17040 | 200 | x1.00 |
| `repeat` | 17040 | 200 | x1.00 |
| `x` | 17040 | 200 | x1.00 |
| `y` | 17040 | 200 | x1.00 |
| `zeros` | 17040 | 200 | x1.00 |

### vector2_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 17440 | 200 | x1.00 |

### vector2_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imin` | 16910 | 770 | x1.00 |
| `imax` | 16920 | 780 | x1.01 |
| `iamax` | 18780 | 2640 | x3.43 |
| `iamin` | 18790 | 2650 | x3.44 |

### vector2_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 19090 | 1850 | x1.00 |
| `sup` | 19090 | 1850 | x1.00 |

### vector2_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 22040 | 3800 | x1.00 |

### vector2_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16140 | 0 | - |

### vector2_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 16540 | 300 | x1.00 |

### vector2_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 21500 | 3860 | x1.00 |

### vector2_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 20240 | 3500 | x1.00 |

### vector2_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 16920 | 780 | x1.00 |
| `min` | 16920 | 780 | x1.00 |
| `amax` | 18780 | 2640 | x3.38 |
| `amin` | 18780 | 2640 | x3.38 |

### vector2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 17240 | 600 | x1.00 |

### vector2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 17040 | 200 | x1.00 |

### vector2_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude` | 18160 | 2020 | x1.00 |
| `norm` | 18160 | 2020 | x1.00 |

### vector2_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude_squared` | 17920 | 1780 | x1.00 |
| `norm_squared` | 17920 | 1780 | x1.00 |

### vector2_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 24670 | 8030 | x1.00 |
| `unscale` | 25390 | 8750 | x1.09 |
| `alt_recip_sqrt` | 26350 | 9710 | x1.21 |

### vector2_perp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `perp` | 18520 | 1780 | x1.00 |

### vector2_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 17540 | 0 | - |
| `push` | 17840 | 300 | - |

### vector2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 20300 | 3260 | x1.00 |
| `scale` | 20300 | 3260 | x1.00 |

### vector2_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub` | 18820 | 1580 | x1.00 |
| `sub_assign` | 18820 | 1580 | x1.00 |

### vector2_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 16780 | 640 | x1.00 |

### vector2_try_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 20570 | 3230 | x1.00 |
| `some` | 26760 | 9420 | x2.92 |

### vector2_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 22950 | 5910 | x1.00 |
| `div_assign` | 23670 | 6630 | x1.12 |
| `unscale` | 23670 | 6630 | x1.12 |

## nalgebra::base::vector3::benches

### vector3_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 20210 | 2870 | x1.00 |

### vector3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs_diff_eq` | 21680 | 4530 | x1.00 |

### vector3_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 20560 | 2420 | x1.00 |
| `add_assign` | 20560 | 2420 | x1.00 |

### vector3_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 56260 | 39120 | x1.00 |
| `half_angle` | 88880 | 71740 | x1.83 |

### vector3_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 21910 | 4170 | x1.00 |
| `capped` | 29200 | 11460 | x2.75 |
| `alt_normalize` | 35380 | 17640 | x4.23 |

### vector3_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 28270 | 10130 | x1.00 |

### vector3_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 23080 | 4940 | x1.00 |

### vector3_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross` | 23680 | 5540 | x1.00 |

### vector3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19120 | 1980 | x1.00 |
| `alt_unfused` | 23560 | 6420 | x3.24 |

### vector3_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_element` | 17740 | 300 | x1.00 |
| `repeat` | 17740 | 300 | x1.00 |
| `x` | 17740 | 300 | x1.00 |
| `y` | 17740 | 300 | x1.00 |
| `z` | 17740 | 300 | x1.00 |
| `zeros` | 17740 | 300 | x1.00 |

### vector3_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18440 | 300 | x1.00 |

### vector3_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imax` | 18090 | 1750 | x1.00 |
| `imin` | 18090 | 1750 | x1.00 |
| `iamax` | 20750 | 4410 | x2.52 |
| `iamin` | 20770 | 4430 | x2.53 |

### vector3_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 20970 | 2830 | x1.00 |
| `sup` | 20970 | 2830 | x1.00 |

### vector3_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 25400 | 5760 | x1.00 |

### vector3_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16340 | 0 | - |

### vector3_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 16740 | 300 | x1.00 |

### vector3_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 24380 | 5840 | x1.00 |

### vector3_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 21580 | 4440 | x1.00 |

### vector3_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 17990 | 1650 | x1.00 |
| `min` | 17990 | 1650 | x1.00 |
| `amax` | 20650 | 4310 | x2.61 |
| `amin` | 20650 | 4310 | x2.61 |

### vector3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 18240 | 900 | x1.00 |

### vector3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 18040 | 300 | x1.00 |

### vector3_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude` | 18560 | 2220 | x1.00 |
| `norm` | 18560 | 2220 | x1.00 |

### vector3_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude_squared` | 18320 | 1980 | x1.00 |
| `norm_squared` | 18320 | 1980 | x1.00 |

### vector3_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 27250 | 9910 | x1.00 |
| `alt_recip_sqrt` | 28930 | 11590 | x1.17 |
| `unscale` | 29170 | 11830 | x1.19 |
| `alt_per_element_div` | 29520 | 12180 | x1.23 |

### vector3_orthonormal_basis_zneg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 34890 | 16050 | x1.00 |
| `alt_upstream` | 36780 | 17940 | x1.12 |

### vector3_orthonormal_basis_zpos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 34730 | 15890 | x1.00 |
| `alt_upstream` | 37050 | 18210 | x1.15 |

### vector3_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 18340 | 100 | x1.00 |
| `push` | 18640 | 400 | x4.00 |

### vector3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 22680 | 4940 | x1.00 |
| `scale` | 22680 | 4940 | x1.00 |

### vector3_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub` | 20560 | 2420 | x1.00 |
| `sub_assign` | 20560 | 2420 | x1.00 |

### vector3_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 17720 | 1380 | x1.00 |

### vector3_try_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 21540 | 3500 | x1.00 |
| `some` | 30540 | 12500 | x3.57 |

### vector3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 25330 | 7590 | x1.00 |
| `div_assign` | 27250 | 9510 | x1.25 |
| `unscale` | 27250 | 9510 | x1.25 |

### vector3_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 17040 | 200 | x1.00 |

## nalgebra::base::vector4::benches

### vector4_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 22080 | 4040 | x1.00 |

### vector4_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs_diff_eq` | 23420 | 5870 | x1.00 |

### vector4_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 22300 | 3260 | x1.00 |
| `add_assign` | 22300 | 3260 | x1.00 |

### vector4_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 57530 | 39990 | x1.00 |
| `half_angle` | 99520 | 81980 | x2.05 |

### vector4_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 23110 | 4670 | x1.00 |
| `capped` | 31880 | 13440 | x2.88 |
| `alt_normalize` | 40980 | 22540 | x4.83 |

### vector4_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 32670 | 13630 | x1.00 |

### vector4_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 25660 | 6620 | x1.00 |

### vector4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19720 | 2180 | x1.00 |
| `alt_unfused` | 26380 | 8840 | x4.06 |

### vector4_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_element` | 18440 | 400 | x1.00 |
| `repeat` | 18440 | 400 | x1.00 |
| `w` | 18440 | 400 | x1.00 |
| `x` | 18440 | 400 | x1.00 |
| `y` | 18440 | 400 | x1.00 |
| `z` | 18440 | 400 | x1.00 |
| `zeros` | 18440 | 400 | x1.00 |

### vector4_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19440 | 400 | x1.00 |

### vector4_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imin` | 19260 | 2720 | x1.00 |
| `imax` | 19270 | 2730 | x1.00 |
| `iamax` | 23000 | 6460 | x2.38 |
| `iamin` | 23000 | 6460 | x2.38 |

### vector4_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 22840 | 3800 | x1.00 |
| `sup` | 22840 | 3800 | x1.00 |

### vector4_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 28740 | 7700 | x1.00 |

### vector4_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16540 | 0 | - |

### vector4_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 16940 | 300 | x1.00 |

### vector4_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 27260 | 7820 | x1.00 |

### vector4_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 22920 | 5380 | x1.00 |

### vector4_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `min` | 19060 | 2520 | x1.00 |
| `max` | 19070 | 2530 | x1.00 |
| `amax` | 22800 | 6260 | x2.48 |
| `amin` | 22800 | 6260 | x2.48 |

### vector4_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 19240 | 1200 | x1.00 |

### vector4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 19040 | 400 | x1.00 |

### vector4_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude` | 18960 | 2420 | x1.00 |
| `norm` | 18960 | 2420 | x1.00 |

### vector4_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `magnitude_squared` | 18720 | 2180 | x1.00 |
| `norm_squared` | 18720 | 2180 | x1.00 |

### vector4_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 29830 | 11790 | x1.00 |
| `alt_recip_sqrt` | 31510 | 13470 | x1.14 |
| `unscale` | 33090 | 15050 | x1.28 |

### vector4_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 25060 | 6620 | x1.00 |
| `scale` | 25060 | 6620 | x1.00 |

### vector4_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sub` | 22300 | 3260 | x1.00 |
| `sub_assign` | 22300 | 3260 | x1.00 |

### vector4_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 18660 | 2120 | x1.00 |

### vector4_try_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 22490 | 3750 | x1.00 |
| `some` | 34460 | 15720 | x4.19 |

### vector4_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 27710 | 9270 | x1.00 |
| `div_assign` | 30970 | 12530 | x1.35 |
| `unscale` | 30970 | 12530 | x1.35 |

### vector4_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 17240 | 200 | x1.00 |

### vector4_xyz

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xyz` | 17840 | 300 | x1.00 |

## nalgebra::base::vector6::benches

### vector6_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 25280 | 5840 | x1.00 |

### vector6_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 25520 | 7580 | x1.00 |

### vector6_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 25780 | 4940 | x1.00 |
| `operator` | 25780 | 4940 | x1.00 |

### vector6_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 30820 | 9980 | x1.00 |

### vector6_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20920 | 2580 | x1.00 |
| `alt_two_sum_prod3` | 23140 | 4800 | x1.86 |
| `alt_unfused` | 32020 | 13680 | x5.30 |

### vector6_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `zeros` | 19840 | 600 | x1.00 |

### vector6_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 26590 | 5750 | x1.00 |
| `sup` | 26590 | 5750 | x1.00 |

### vector6_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 33020 | 11780 | x1.00 |

### vector6_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 21240 | 1800 | x1.00 |

### vector6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 19040 | 600 | x1.00 |

### vector6_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_via_norm_squared` | 21440 | 4500 | x1.00 |
| `wide_sqrt` | 21880 | 4940 | x1.10 |

### vector6_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 19520 | 2580 | x1.00 |
| `alt_two_blocks` | 21740 | 4800 | x1.86 |

### vector6_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unscale_by_norm` | 42510 | 23070 | x1.00 |

### vector6_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 29820 | 9980 | x1.00 |
| `scale` | 29820 | 9980 | x1.00 |

### vector6_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 25780 | 4940 | x1.00 |
| `operator` | 25780 | 4940 | x1.00 |

### vector6_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 20540 | 3600 | x1.00 |

### vector6_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 37870 | 18030 | x1.00 |
| `unscale` | 37870 | 18030 | x1.00 |

