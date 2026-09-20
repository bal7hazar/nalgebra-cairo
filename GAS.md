# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::base::matrix2::tests

### matrix2_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 21680 | 3640 | x1.00 |

### matrix2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 23850 | 6300 | x1.00 |

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
| `products` | 26340 | 7300 | x1.00 |

### matrix2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18490 | 1950 | x1.00 |

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
| `products` | 25540 | 7300 | x1.00 |

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
| `all_compared` | 22450 | 5900 | x1.00 |

### matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 29300 | 10260 | x1.00 |
| `fused` | 29300 | 10260 | x1.00 |

### matrix2_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 25350 | 7810 | x1.00 |
| `generic` | 28100 | 10560 | x1.35 |

### matrix2_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21640 | 4000 | x1.00 |

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
| `fused` | 18890 | 2350 | x1.00 |

### matrix2_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 17240 | 200 | x1.00 |

### matrix2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 25740 | 7300 | x1.00 |

### matrix2_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 22300 | 3260 | x1.00 |
| `operator` | 22300 | 3260 | x1.00 |

### matrix2_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 29300 | 10260 | x1.00 |
| `transpose_mul` | 29300 | 10260 | x1.00 |

### matrix2_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21640 | 4000 | x1.00 |
| `transpose_mul_vec` | 21640 | 4000 | x1.00 |

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
| `alt_recip` | 32500 | 14460 | x1.00 |
| `alt_div` | 34370 | 16330 | x1.13 |
| `prescaled_det_ge_half` | 50880 | 32840 | x2.27 |
| `prescaled_norm_gt_one` | 50880 | 32840 | x2.27 |
| `prescaled_small` | 50880 | 32840 | x2.27 |

### matrix2_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 49690 | 33140 | x1.00 |

### matrix2_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16240 | -800 | - |

## nalgebra::base::matrix3::tests

### matrix3_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 30350 | 8810 | x1.00 |

### matrix3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 32600 | 13050 | x1.00 |

### matrix3_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 31000 | 7460 | x1.00 |
| `operator` | 31000 | 7460 | x1.00 |

### matrix3_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 39890 | 18350 | x1.00 |

### matrix3_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 18840 | 300 | x1.00 |

### matrix3_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 40090 | 16550 | x1.00 |

### matrix3_cross_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 20940 | 600 | x1.00 |

### matrix3_cross_matrix_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 43250 | 20910 | x1.00 |
| `materialised` | 46250 | 23910 | x1.14 |

### matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 28200 | 10660 | x1.00 |
| `alt_triple_products` | 33750 | 16210 | x1.52 |

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
| `products` | 37690 | 16550 | x1.00 |

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
| `all_compared` | 29700 | 12150 | x1.00 |

### matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 46850 | 23310 | x1.00 |
| `fused` | 46850 | 23310 | x1.00 |

### matrix3_mul_cross_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 43250 | 20910 | x1.00 |
| `materialised` | 46250 | 23910 | x1.14 |

### matrix3_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 35700 | 15660 | x1.00 |
| `generic` | 43950 | 23910 | x1.53 |

### matrix3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25990 | 6650 | x1.00 |

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
| `wide` | 20960 | 3420 | x1.00 |

### matrix3_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20890 | 3350 | x1.00 |

### matrix3_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 18840 | 300 | x1.00 |

### matrix3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 38490 | 16550 | x1.00 |

### matrix3_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 31000 | 7460 | x1.00 |
| `operator` | 31000 | 7460 | x1.00 |

### matrix3_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 46850 | 23310 | x1.00 |
| `transpose_mul` | 46850 | 23310 | x1.00 |

### matrix3_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25990 | 6650 | x1.00 |
| `transpose_mul_vec` | 25990 | 6650 | x1.00 |

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
| `alt_recip` | 72310 | 50770 | x1.00 |
| `alt_div` | 79030 | 57490 | x1.13 |
| `prescaled_det_ge_half` | 110480 | 88940 | x1.75 |
| `prescaled_norm_gt_one` | 110480 | 88940 | x1.75 |
| `prescaled_small` | 110480 | 88940 | x1.75 |

### matrix3_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 106790 | 89240 | x1.00 |

### matrix3_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 17740 | -1800 | - |

## nalgebra::base::matrix4::tests

### matrix4_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 43840 | 15400 | x1.00 |

### matrix4_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 44850 | 22500 | x1.00 |

### matrix4_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 45180 | 13340 | x1.00 |
| `operator` | 45180 | 13340 | x1.00 |

### matrix4_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 97850 | 69410 | x1.00 |

### matrix4_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 20840 | 400 | x1.00 |

### matrix4_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 61340 | 29500 | x1.00 |

### matrix4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `minors2x2` | 49350 | 30410 | x1.00 |
| `alt_row_cofactors` | 67190 | 48250 | x1.59 |

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
| `products` | 56540 | 29500 | x1.00 |

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
| `all_compared` | 39850 | 20900 | x1.00 |

### matrix4_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 77100 | 45260 | x1.00 |
| `fused` | 77100 | 45260 | x1.00 |

### matrix4_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 31140 | 9700 | x1.00 |

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
| `wide` | 23760 | 4820 | x1.00 |

### matrix4_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 23690 | 4750 | x1.00 |

### matrix4_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 20840 | 400 | x1.00 |

### matrix4_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 58340 | 29500 | x1.00 |

### matrix4_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 45180 | 13340 | x1.00 |
| `operator` | 45180 | 13340 | x1.00 |

### matrix4_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 77100 | 45260 | x1.00 |
| `transpose_mul` | 77100 | 45260 | x1.00 |

### matrix4_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 31140 | 9700 | x1.00 |
| `transpose_mul_vec` | 31140 | 9700 | x1.00 |

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
| `alt_recip` | 163670 | 135230 | x1.00 |
| `alt_div` | 177180 | 148740 | x1.10 |
| `prescaled_det_ge_half` | 234680 | 206240 | x1.53 |
| `prescaled_norm_gt_one` | 234680 | 206240 | x1.53 |
| `prescaled_small` | 234680 | 206240 | x1.53 |

### matrix4_try_inverse_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 223890 | 204940 | x1.00 |

### matrix4_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 19840 | -5200 | - |

## nalgebra::base::matrix6::benches

### matrix6_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `componentwise` | 82860 | 38420 | x1.00 |

### matrix6_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 82140 | 52200 | x1.00 |

### matrix6_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 82880 | 30140 | x1.00 |
| `operator` | 82880 | 30140 | x1.00 |

### matrix6_blocks

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_blocks` | 26940 | 0 | - |

### matrix6_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 26040 | 600 | x1.00 |

### matrix6_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `zeros` | 45740 | 3600 | x1.00 |
| `from_diagonal_element` | 48440 | 6300 | x1.75 |
| `identity` | 48440 | 6300 | x1.75 |

### matrix6_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 35440 | -3900 | - |

### matrix6_is_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 74740 | 50400 | x1.00 |

### matrix6_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 167400 | 114660 | x1.00 |
| `fused` | 167400 | 114660 | x1.00 |
| `alt_blocks` | 282590 | 229850 | x2.00 |

### matrix6_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 49400 | 22560 | x1.00 |
| `alt_blocks` | 64610 | 37770 | x1.67 |

### matrix6_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 55240 | 10800 | x1.00 |

### matrix6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 43740 | 6300 | x1.00 |

### matrix6_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `blocks` | 111340 | 66500 | x1.00 |

### matrix6_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 82880 | 30140 | x1.00 |
| `operator` | 82880 | 30140 | x1.00 |

### matrix6_tr_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul` | 167400 | 114660 | x1.00 |
| `fused` | 167400 | 114660 | x1.00 |

### matrix6_tr_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_vec` | 49400 | 22560 | x1.00 |
| `fused` | 49400 | 22560 | x1.00 |

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
| `outside` | 19360 | 2210 | x1.00 |
| `within` | 20140 | 2990 | x1.35 |

### point2_add_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_assign` | 18820 | 1580 | x1.00 |
| `add_vector` | 18820 | 1580 | x1.00 |

### point2_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 21240 | 4000 | x1.00 |
| `alt_lerp` | 21440 | 4200 | x1.05 |
| `alt_add_scale` | 22320 | 5080 | x1.27 |
| `alt_add_div` | 22920 | 5680 | x1.42 |

### point2_coords

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 16840 | 200 | x1.00 |
| `into_vector` | 16840 | 200 | x1.00 |

### point2_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 20240 | 3500 | x1.00 |
| `alt_sqrt` | 22090 | 5350 | x1.53 |

### point2_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20170 | 3430 | x1.00 |
| `alt_unfused` | 22560 | 5820 | x1.70 |

### point2_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 18040 | 200 | x1.00 |
| `from_coordinates` | 18040 | 200 | x1.00 |
| `tuple` | 18040 | 200 | x1.00 |
| `vector` | 18040 | 200 | x1.00 |

### point2_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_homogeneous` | 23220 | 6080 | x1.00 |
| `alt_recip` | 23530 | 6390 | x1.05 |

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
| `tuple` | 16140 | 0 | - |

### point2_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 21840 | 4200 | x1.00 |

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

### point2_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `push` | 17840 | 300 | x1.00 |

### point2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 20640 | 3600 | x1.00 |
| `scale` | 20640 | 3600 | x1.00 |

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
| `div_assign` | 22620 | 5580 | x1.00 |
| `unscale` | 22620 | 5580 | x1.00 |

## nalgebra::base::point3::benches

### point3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 19840 | 2290 | x1.00 |
| `within` | 21780 | 4230 | x1.85 |

### point3_add_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_assign` | 20560 | 2420 | x1.00 |
| `add_vector` | 20560 | 2420 | x1.00 |

### point3_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 24190 | 6050 | x1.00 |
| `alt_lerp` | 24490 | 6350 | x1.05 |
| `alt_add_scale` | 25810 | 7670 | x1.27 |
| `alt_add_div` | 26710 | 8570 | x1.42 |

### point3_coords

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 17640 | 300 | x1.00 |
| `into_vector` | 17640 | 300 | x1.00 |

### point3_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 21580 | 4440 | x1.00 |
| `alt_sqrt` | 23430 | 6290 | x1.42 |

### point3_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21510 | 4370 | x1.00 |
| `alt_unfused` | 26290 | 9150 | x2.09 |

### point3_from

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 19240 | 300 | x1.00 |
| `from_coordinates` | 19240 | 300 | x1.00 |
| `tuple` | 19240 | 300 | x1.00 |
| `vector` | 19240 | 300 | x1.00 |

### point3_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 26080 | 8240 | x1.00 |
| `from_homogeneous` | 26760 | 8920 | x1.08 |

### point3_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sup` | 20960 | 2820 | x1.00 |
| `inf` | 20970 | 2830 | x1.00 |
| `inf_sup` | 25390 | 7250 | x2.57 |

### point3_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16340 | 0 | - |
| `tuple` | 16340 | 0 | - |

### point3_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 24890 | 6350 | x1.00 |

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
| `mul_assign` | 23190 | 5450 | x1.00 |
| `scale` | 23190 | 5450 | x1.00 |

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
| `div_assign` | 26160 | 8420 | x1.00 |
| `unscale` | 26160 | 8420 | x1.00 |

### point3_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 17040 | 200 | x1.00 |

## nalgebra::base::sym_matrix2::tests

### sym_matrix2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 21710 | 4560 | x1.00 |

### sym_matrix2_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 20560 | 2420 | x1.00 |
| `operator` | 20560 | 2420 | x1.00 |
| `generic` | 21300 | 3160 | x1.31 |

### sym_matrix2_add_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 19420 | 1680 | x1.00 |

### sym_matrix2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `generic` | 18290 | 1950 | x1.00 |
| `structured` | 18290 | 1950 | x1.00 |

### sym_matrix2_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17040 | 200 | x1.00 |

### sym_matrix2_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17140 | 0 | - |

### sym_matrix2_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 16940 | 0 | - |

### sym_matrix2_from_matrix_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17840 | 300 | x1.00 |

### sym_matrix2_from_outer_self

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 22590 | 5450 | x1.00 |
| `generic` | 24340 | 7200 | x1.32 |

### sym_matrix2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16540 | 0 | - |

### sym_matrix2_inverse_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 45680 | 28340 | x1.00 |

### sym_matrix2_mul_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 29000 | 10160 | x1.00 |
| `generic` | 29100 | 10260 | x1.01 |

### sym_matrix2_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `generic` | 21440 | 4000 | x1.00 |
| `structured` | 21440 | 4000 | x1.00 |

### sym_matrix2_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 18240 | 900 | x1.00 |

### sym_matrix2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17340 | 0 | - |

### sym_matrix2_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18760 | 2420 | x1.00 |

### sym_matrix2_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 33550 | 15410 | x1.00 |
| `generic` | 39060 | 20920 | x1.36 |

### sym_matrix2_quadform_sym

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 34650 | 16310 | x1.00 |
| `generic` | 39260 | 20920 | x1.28 |

### sym_matrix2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 23190 | 5450 | x1.00 |
| `generic` | 24940 | 7200 | x1.32 |

### sym_matrix2_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 20560 | 2420 | x1.00 |
| `operator` | 20560 | 2420 | x1.00 |

### sym_matrix2_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18240 | 400 | x1.00 |

### sym_matrix2_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 16980 | 640 | x1.00 |

### sym_matrix2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 46080 | 28740 | x1.00 |
| `structured_prescaled` | 46080 | 28740 | x1.00 |
| `generic` | 50480 | 33140 | x1.15 |

### sym_matrix2_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 15940 | -600 | - |

## nalgebra::base::sym_matrix3::tests

### sym_matrix3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 27350 | 9000 | x1.00 |

### sym_matrix3_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 25780 | 4940 | x1.00 |
| `operator` | 25780 | 4940 | x1.00 |
| `generic` | 28000 | 7160 | x1.45 |

### sym_matrix3_add_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 22560 | 2720 | x1.00 |

### sym_matrix3_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 31640 | 12200 | x1.00 |
| `generic` | 37490 | 18050 | x1.48 |

### sym_matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 27300 | 10360 | x1.00 |
| `generic` | 27600 | 10660 | x1.03 |

### sym_matrix3_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18240 | 300 | x1.00 |

### sym_matrix3_from_diagonal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18540 | -300 | - |

### sym_matrix3_from_diagonal_element

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18140 | -300 | - |

### sym_matrix3_from_matrix_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 20640 | 600 | x1.00 |

### sym_matrix3_from_outer_self

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 29840 | 11000 | x1.00 |
| `generic` | 35090 | 16250 | x1.48 |

### sym_matrix3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 17740 | -300 | - |

### sym_matrix3_inverse_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 84580 | 65140 | x1.00 |

### sym_matrix3_mul_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 45950 | 23010 | x1.00 |
| `generic` | 46250 | 23310 | x1.01 |

### sym_matrix3_mul_vec

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `generic` | 25390 | 6650 | x1.00 |
| `structured` | 25390 | 6650 | x1.00 |

### sym_matrix3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 21240 | 1800 | x1.00 |

### sym_matrix3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 19440 | 0 | - |

### sym_matrix3_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20360 | 3420 | x1.00 |

### sym_matrix3_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 53450 | 32610 | x1.00 |
| `generic` | 68160 | 47320 | x1.45 |

### sym_matrix3_quadform_sym

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 57950 | 36510 | x1.00 |
| `generic` | 68760 | 47320 | x1.30 |

### sym_matrix3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 30840 | 11000 | x1.00 |
| `generic` | 36090 | 16250 | x1.48 |

### sym_matrix3_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 25780 | 4940 | x1.00 |
| `operator` | 25780 | 4940 | x1.00 |

### sym_matrix3_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 21840 | 900 | x1.00 |

### sym_matrix3_trace

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 18320 | 1380 | x1.00 |

### sym_matrix3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 84980 | 65540 | x1.00 |
| `structured_prescaled` | 84980 | 65540 | x1.00 |
| `generic` | 108980 | 89540 | x1.37 |

### sym_matrix3_zeros

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16840 | -1200 | - |

## nalgebra::base::unit::benches

### unit2_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `x_axis` | 17040 | 200 | x1.00 |
| `y_axis` | 17040 | 200 | x1.00 |

### unit2_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 19290 | 1950 | x1.00 |

### unit2_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 25140 | 7700 | x1.00 |

### unit2_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 24240 | 7600 | x1.00 |

### unit3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `outside` | 19840 | 2290 | x1.00 |
| `within` | 21780 | 4230 | x1.85 |

### unit3_axes

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `x_axis` | 17740 | 300 | x1.00 |
| `y_axis` | 17740 | 300 | x1.00 |
| `z_axis` | 17740 | 300 | x1.00 |

### unit3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `dot` | 20090 | 2150 | x1.00 |
| `dot_vector` | 20090 | 2150 | x1.00 |
| `alt_unfused` | 24870 | 6930 | x3.22 |

### unit3_into_inner

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `as_ref` | 17340 | 0 | - |
| `into_inner` | 17340 | 0 | - |

### unit3_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 18240 | 900 | x1.00 |

### unit3_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 28880 | 10740 | x1.00 |
| `new_and_get` | 29180 | 11040 | x1.03 |
| `try_new` | 29850 | 11710 | x1.09 |
| `try_new_and_get` | 30650 | 12510 | x1.16 |

### unit3_new_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_unchecked` | 17340 | 0 | - |

### unit3_orthonormal_basis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `orthonormal_basis` | 34380 | 15540 | x1.00 |

### unit3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `renormalize` | 28080 | 10740 | x1.00 |

### unit3_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 26990 | 9650 | x1.00 |
| `alt_upstream` | 27830 | 10490 | x1.09 |
| `alt_exact` | 28080 | 10740 | x1.11 |
| `alt_mul_add` | 28430 | 11090 | x1.15 |
| `alt_scale` | 28670 | 11330 | x1.17 |
| `alt_lerp` | 29930 | 12590 | x1.30 |

### unit3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale` | 23190 | 5450 | x1.00 |

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
| `dot` | 20890 | 2350 | x1.00 |

### unit4_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_normalize` | 32620 | 13780 | x1.00 |

### unit4_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fma_factor` | 29740 | 11700 | x1.00 |

## nalgebra::base::vector2::benches

### vector2_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 18500 | 1860 | x1.00 |

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
| `alt_acos` | 39450 | 22710 | x1.00 |
| `half_angle` | 58030 | 41290 | x1.82 |

### vector2_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 20620 | 3580 | x1.00 |
| `capped` | 26570 | 9530 | x2.66 |
| `alt_normalize` | 29410 | 12370 | x3.46 |

### vector2_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 22900 | 5660 | x1.00 |

### vector2_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 20840 | 3600 | x1.00 |

### vector2_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18690 | 1950 | x1.00 |
| `alt_unfused` | 21080 | 4340 | x2.23 |

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
| `tuple` | 17440 | 200 | x1.00 |

### vector2_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imin` | 16910 | 770 | x1.00 |
| `imax` | 16920 | 780 | x1.01 |
| `iamax` | 18670 | 2530 | x3.29 |
| `iamin` | 18680 | 2540 | x3.30 |

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
| `tuple` | 16140 | 0 | - |

### vector2_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 16540 | 300 | x1.00 |

### vector2_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 21840 | 4200 | x1.00 |

### vector2_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 20240 | 3500 | x1.00 |

### vector2_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 16910 | 770 | x1.00 |
| `min` | 16920 | 780 | x1.01 |
| `amin` | 18670 | 2530 | x3.29 |
| `amax` | 18680 | 2540 | x3.30 |

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
| `magnitude_squared` | 18090 | 1950 | x1.00 |
| `norm_squared` | 18090 | 1950 | x1.00 |

### vector2_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unscale` | 24340 | 7700 | x1.00 |
| `alt_recip` | 24650 | 8010 | x1.04 |
| `alt_inv_sqrt` | 25220 | 8580 | x1.11 |

### vector2_perp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `perp` | 18690 | 1950 | x1.00 |

### vector2_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 17540 | 0 | - |
| `push` | 17840 | 300 | - |

### vector2_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 20640 | 3600 | x1.00 |
| `scale` | 20640 | 3600 | x1.00 |

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
| `none` | 20400 | 3060 | x1.00 |
| `some` | 26010 | 8670 | x2.83 |

### vector2_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 22620 | 5580 | x1.00 |
| `unscale` | 22620 | 5580 | x1.00 |
| `alt_recip` | 22930 | 5890 | x1.06 |

## nalgebra::base::vector3::benches

### vector3_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 20090 | 2750 | x1.00 |

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
| `alt_acos` | 40450 | 23310 | x1.00 |
| `half_angle` | 66550 | 49410 | x2.12 |

### vector3_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 21820 | 4080 | x1.00 |
| `capped` | 29420 | 11680 | x2.86 |
| `alt_normalize` | 35100 | 17360 | x4.25 |

### vector3_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 26640 | 8500 | x1.00 |

### vector3_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 23590 | 5450 | x1.00 |

### vector3_cross

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross` | 24190 | 6050 | x1.00 |

### vector3_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19290 | 2150 | x1.00 |
| `alt_unfused` | 24070 | 6930 | x3.22 |

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
| `tuple` | 18440 | 300 | x1.00 |

### vector3_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imax` | 18090 | 1750 | x1.00 |
| `imin` | 18090 | 1750 | x1.00 |
| `iamax` | 20630 | 4290 | x2.45 |
| `iamin` | 20650 | 4310 | x2.46 |

### vector3_inf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sup` | 20960 | 2820 | x1.00 |
| `inf` | 20970 | 2830 | x1.00 |

### vector3_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf_sup` | 25390 | 5750 | x1.00 |

### vector3_into

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `array` | 16340 | 0 | - |
| `tuple` | 16340 | 0 | - |

### vector3_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 16740 | 300 | x1.00 |

### vector3_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 24890 | 6350 | x1.00 |

### vector3_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 21580 | 4440 | x1.00 |

### vector3_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 17990 | 1650 | x1.00 |
| `min` | 17990 | 1650 | x1.00 |
| `amin` | 20530 | 4190 | x2.54 |
| `amax` | 20550 | 4210 | x2.55 |

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
| `magnitude_squared` | 18490 | 2150 | x1.00 |
| `norm_squared` | 18490 | 2150 | x1.00 |

### vector3_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 27400 | 10060 | x1.00 |
| `alt_inv_sqrt` | 27970 | 10630 | x1.06 |
| `unscale` | 28080 | 10740 | x1.07 |

### vector3_orthonormal_basis_zneg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 34450 | 15610 | x1.00 |
| `alt_upstream` | 36330 | 17490 | x1.12 |

### vector3_orthonormal_basis_zpos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 34380 | 15540 | x1.00 |
| `alt_upstream` | 36330 | 17490 | x1.13 |

### vector3_push

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `to_homogeneous` | 18340 | 100 | x1.00 |
| `push` | 18640 | 400 | x4.00 |

### vector3_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 23190 | 5450 | x1.00 |
| `scale` | 23190 | 5450 | x1.00 |

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
| `none` | 21280 | 3240 | x1.00 |
| `some` | 29750 | 11710 | x3.61 |

### vector3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 25480 | 7740 | x1.00 |
| `div_assign` | 26160 | 8420 | x1.09 |
| `unscale` | 26160 | 8420 | x1.09 |

### vector3_xy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `xy` | 17040 | 200 | x1.00 |

## nalgebra::base::vector4::benches

### vector4_abs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `abs` | 21860 | 3820 | x1.00 |

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
| `alt_acos` | 41450 | 23910 | x1.00 |
| `half_angle` | 75070 | 57530 | x2.41 |

### vector4_cap_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unchanged` | 23020 | 4580 | x1.00 |
| `capped` | 32270 | 13830 | x3.02 |
| `alt_normalize` | 40790 | 22350 | x4.88 |

### vector4_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_div` | 30380 | 11340 | x1.00 |

### vector4_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 26340 | 7300 | x1.00 |

### vector4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19890 | 2350 | x1.00 |
| `alt_unfused` | 27060 | 9520 | x4.05 |

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
| `tuple` | 19440 | 400 | x1.00 |

### vector4_imin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `imin` | 19260 | 2720 | x1.00 |
| `imax` | 19270 | 2730 | x1.00 |
| `iamax` | 22780 | 6240 | x2.29 |
| `iamin` | 22780 | 6240 | x2.29 |

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
| `tuple` | 16540 | 0 | - |

### vector4_is_zero

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `is_zero` | 16940 | 300 | x1.00 |

### vector4_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp` | 27940 | 8500 | x1.00 |

### vector4_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `metric_distance` | 22920 | 5380 | x1.00 |

### vector4_min

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `max` | 19060 | 2520 | x1.00 |
| `min` | 19060 | 2520 | x1.00 |
| `amin` | 22580 | 6040 | x2.40 |
| `amax` | 22590 | 6050 | x2.40 |

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
| `magnitude_squared` | 18890 | 2350 | x1.00 |
| `norm_squared` | 18890 | 2350 | x1.00 |

### vector4_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 30150 | 12110 | x1.00 |
| `alt_inv_sqrt` | 30720 | 12680 | x1.05 |
| `unscale` | 31820 | 13780 | x1.14 |

### vector4_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 25740 | 7300 | x1.00 |
| `scale` | 25740 | 7300 | x1.00 |

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
| `none` | 22160 | 3420 | x1.00 |
| `some` | 33490 | 14750 | x4.31 |

### vector4_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 28030 | 9590 | x1.00 |
| `div_assign` | 29700 | 11260 | x1.17 |
| `unscale` | 29700 | 11260 | x1.17 |

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
| `componentwise` | 25040 | 5600 | x1.00 |

### vector6_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 25820 | 7580 | x1.00 |

### vector6_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 26680 | 4940 | x1.00 |
| `operator` | 26680 | 4940 | x1.00 |

### vector6_blocks

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_blocks` | 18240 | 0 | - |

### vector6_component_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_mul` | 32740 | 11000 | x1.00 |

### vector6_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 21690 | 2750 | x1.00 |
| `alt_two_sum_prod3` | 24080 | 5140 | x1.87 |
| `alt_unfused` | 33640 | 14700 | x5.35 |

### vector6_fill

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `zeros` | 20440 | 900 | x1.00 |

### vector6_inf_sup

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inf` | 27490 | 5750 | x1.00 |
| `sup` | 27490 | 5750 | x1.00 |

### vector6_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 34940 | 12800 | x1.00 |

### vector6_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `operator` | 21240 | 1800 | x1.00 |

### vector6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 19340 | 900 | x1.00 |

### vector6_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide_sqrt` | 19760 | 2820 | x1.00 |
| `alt_via_norm_squared` | 21610 | 4670 | x1.66 |

### vector6_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 19690 | 2750 | x1.00 |
| `alt_two_blocks` | 22080 | 5140 | x1.87 |

### vector6_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unscale_by_norm` | 39300 | 19860 | x1.00 |

### vector6_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 30840 | 11000 | x1.00 |
| `scale` | 30840 | 11000 | x1.00 |

### vector6_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `assign` | 26380 | 4940 | x1.00 |
| `operator` | 26380 | 4940 | x1.00 |

### vector6_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum` | 20540 | 3600 | x1.00 |

### vector6_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div_assign` | 36780 | 16940 | x1.00 |
| `unscale` | 36780 | 16940 | x1.00 |

## nalgebra::geometry::quaternion::benches

### quaternion_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 19860 | 2220 | x1.00 |

### quaternion_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 22300 | 3260 | x1.00 |

### quaternion_as_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18440 | 400 | x1.00 |

### quaternion_conjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 19040 | 1000 | x1.00 |

### quaternion_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19890 | 2350 | x1.00 |

### quaternion_from_imag

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17940 | 100 | x1.00 |

### quaternion_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18640 | 400 | x1.00 |

### quaternion_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16540 | -500 | - |

### quaternion_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 27940 | 8500 | x1.00 |

### quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_wide` | 30900 | 11860 | x1.00 |
| `alt_sum_prod4` | 31800 | 12760 | x1.08 |
| `alt_unfused` | 57820 | 38780 | x3.27 |

### quaternion_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 19240 | 1200 | x1.00 |

### quaternion_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 19040 | 400 | x1.00 |

### quaternion_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm4` | 18960 | 2420 | x1.00 |

### quaternion_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 18890 | 2350 | x1.00 |

### quaternion_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 30150 | 12110 | x1.00 |
| `divisions` | 31820 | 13780 | x1.14 |

### quaternion_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 25740 | 7300 | x1.00 |

### quaternion_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 22300 | 3260 | x1.00 |

### quaternion_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 31480 | 13440 | x1.00 |
| `divisions` | 33150 | 15110 | x1.12 |

### quaternion_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 29700 | 11260 | x1.00 |

### quaternion_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17840 | 300 | x1.00 |

## nalgebra::geometry::rotation2::benches

### rotation2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 19850 | 2210 | x1.00 |

### rotation2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 31940 | 15400 | x1.00 |

### rotation2_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_columns` | 36840 | 19300 | x1.00 |
| `alt_product_then_angle` | 43300 | 25760 | x1.33 |

### rotation2_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize_first_column` | 26340 | 8300 | x1.00 |

### rotation2_from_matrix_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 18040 | 0 | - |

### rotation2_from_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20040 | 2400 | x1.00 |

### rotation2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 17240 | -200 | - |

### rotation2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 19640 | 1600 | x1.00 |

### rotation2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transposed` | 21640 | 4000 | x1.00 |

### rotation2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_inverse_then_transform` | 21640 | 4000 | x1.00 |
| `transposed` | 21640 | 4000 | x1.00 |

### rotation2_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 16840 | 300 | x1.00 |

### rotation2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_complex` | 23640 | 4600 | x1.00 |
| `matrix` | 29300 | 10260 | x2.23 |

### rotation2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 34860 | 17420 | x1.00 |

### rotation2_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 53110 | 34670 | x1.00 |

### rotation2_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 26340 | 8300 | x1.00 |

### rotation2_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 32800 | 14560 | x1.00 |
| `alt_unit_complex` | 32900 | 14660 | x1.01 |

### rotation2_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 59910 | 41270 | x1.00 |

### rotation2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 16540 | 0 | - |

### rotation2_to_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 17840 | 800 | x1.00 |

### rotation2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unit_complex` | 21640 | 4000 | x1.00 |
| `matrix` | 21640 | 4000 | x1.00 |

### rotation2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unit_complex` | 26340 | 8700 | x1.00 |
| `matrix` | 26340 | 8700 | x1.00 |

## nalgebra::geometry::rotation3::benches

### rotation3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `entries` | 32690 | 13050 | x1.00 |

### rotation3_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_trace` | 36290 | 18750 | x1.00 |
| `alt_quaternion` | 62670 | 45130 | x2.41 |

### rotation3_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `antisymmetric_part` | 32570 | 14030 | x1.00 |

### rotation3_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `asin_atan2` | 64100 | 46560 | x1.00 |

### rotation3_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `two_normalizations` | 57460 | 36320 | x1.00 |

### rotation3_from_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rodrigues` | 69810 | 49070 | x1.00 |
| `alt_quaternion` | 70350 | 49610 | x1.01 |

### rotation3_from_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_sin_cos` | 92660 | 71920 | x1.00 |

### rotation3_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_and_rodrigues` | 80590 | 60250 | x1.00 |

### rotation3_from_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 44070 | 23530 | x1.00 |

### rotation3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 18340 | -1200 | - |

### rotation3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 22440 | 900 | x1.00 |

### rotation3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `tr_mul_vec` | 25990 | 6650 | x1.00 |

### rotation3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 58960 | 37820 | x1.00 |

### rotation3_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 21540 | 0 | - |

### rotation3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix_product` | 46850 | 23310 | x1.00 |

### rotation3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 60710 | 39170 | x1.00 |
| `alt_quaternion` | 91790 | 70250 | x1.79 |
| `alt_newton` | 99860 | 78320 | x2.00 |

### rotation3_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic_quaternion` | 95610 | 74470 | x1.00 |
| `alt_axis_angle` | 128640 | 107500 | x1.44 |

### rotation3_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_times_angle` | 58770 | 40230 | x1.00 |

### rotation3_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_rodrigues` | 130390 | 109250 | x1.00 |

### rotation3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 17540 | 0 | - |

### rotation3_to_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 42610 | 23570 | x1.00 |

### rotation3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_vec` | 25990 | 6650 | x1.00 |

### rotation3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_vec` | 25990 | 6650 | x1.00 |

## nalgebra::geometry::unit_complex::benches

### unit_complex_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 18870 | 2030 | x1.00 |

### unit_complex_accessors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `complex` | 16840 | 200 | x1.00 |
| `cos_sin_angle` | 17240 | 600 | x3.00 |

### unit_complex_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 31540 | 15400 | x1.00 |

### unit_complex_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_atan2` | 36040 | 19300 | x1.00 |

### unit_complex_append_axisangle_linearized

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_renormalize_fast` | 29040 | 12000 | x1.00 |
| `renormalize` | 29140 | 12100 | x1.01 |
| `alt_sin_cos` | 37960 | 20920 | x1.74 |

### unit_complex_from_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 17240 | 200 | x1.00 |

### unit_complex_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16340 | -100 | - |
| `from_cos_sin_unchecked` | 16340 | -100 | - |

### unit_complex_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate` | 18040 | 1400 | x1.00 |

### unit_complex_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21240 | 4000 | x1.00 |

### unit_complex_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21240 | 4000 | x1.00 |
| `alt_inverse_then_transform` | 21540 | 4300 | x1.07 |

### unit_complex_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21240 | 4000 | x1.00 |

### unit_complex_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 33460 | 17020 | x1.00 |

### unit_complex_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 51310 | 34270 | x1.00 |

### unit_complex_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_and_divisions` | 24340 | 7700 | x1.00 |

### unit_complex_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_add` | 24240 | 7600 | x1.00 |
| `alt_exact` | 24340 | 7700 | x1.01 |
| `alt_lerp` | 24640 | 8000 | x1.05 |
| `alt_mul_add_each` | 24640 | 8000 | x1.05 |
| `alt_literal` | 25080 | 8440 | x1.11 |

### unit_complex_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 31300 | 14060 | x1.00 |
| `alt_normalized_inputs` | 45140 | 27900 | x1.98 |
| `alt_atan2` | 69860 | 52620 | x3.74 |

### unit_complex_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21240 | 4000 | x1.00 |

### unit_complex_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 57410 | 39770 | x1.00 |

### unit_complex_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `angle_then_compose` | 61570 | 43930 | x1.00 |

### unit_complex_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 16440 | 300 | x1.00 |

### unit_complex_to_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 16740 | 600 | x1.00 |

### unit_complex_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21240 | 4000 | x1.00 |

### unit_complex_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25940 | 8700 | x1.00 |

## nalgebra::geometry::unit_quaternion::benches

### unit_quaternion_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 19850 | 2210 | x1.00 |

### unit_quaternion_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 33260 | 16720 | x1.00 |
| `atan2` | 38000 | 21460 | x1.28 |

### unit_quaternion_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product_and_atan2` | 51560 | 34020 | x1.00 |

### unit_quaternion_append_axisangle_linearized

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reduced_product` | 50390 | 31550 | x1.00 |
| `alt_exact` | 65720 | 46880 | x1.49 |

### unit_quaternion_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_new` | 32760 | 15220 | x1.00 |

### unit_quaternion_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19890 | 2350 | x1.00 |

### unit_quaternion_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `five_entries` | 75470 | 58930 | x1.00 |

### unit_quaternion_from_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 44220 | 25980 | x1.00 |

### unit_quaternion_from_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_sin_cos` | 91410 | 73170 | x1.00 |

### unit_quaternion_from_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 42610 | 23570 | x1.00 |

### unit_quaternion_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_division` | 52760 | 34920 | x1.00 |
| `alt_unit_axis` | 58400 | 40560 | x1.16 |

### unit_quaternion_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16540 | -500 | - |

### unit_quaternion_imag

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 17840 | 300 | x1.00 |

### unit_quaternion_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate` | 19040 | 1000 | x1.00 |

### unit_quaternion_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate` | 42170 | 23830 | x1.00 |

### unit_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hamilton` | 30900 | 11860 | x1.00 |

### unit_quaternion_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 31820 | 13780 | x1.00 |

### unit_quaternion_new_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18040 | 0 | - |

### unit_quaternion_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 41820 | 22380 | x1.00 |

### unit_quaternion_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_angle` | 84150 | 65710 | x1.00 |

### unit_quaternion_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 31820 | 13780 | x1.00 |

### unit_quaternion_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 29740 | 11700 | x1.00 |
| `alt_exact` | 31820 | 13780 | x1.18 |

### unit_quaternion_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 68080 | 49440 | x1.00 |
| `alt_axis_angle` | 104400 | 85760 | x1.73 |

### unit_quaternion_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_product` | 31500 | 12460 | x1.00 |

### unit_quaternion_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_factor` | 48370 | 30830 | x1.00 |
| `divisions` | 54010 | 36470 | x1.18 |

### unit_quaternion_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos` | 104400 | 85760 | x1.00 |

### unit_quaternion_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_sin` | 101380 | 81940 | x1.00 |

### unit_quaternion_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 40070 | 23530 | x1.00 |

### unit_quaternion_to_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_one_minus` | 43470 | 22930 | x1.00 |
| `fused` | 44070 | 23530 | x1.03 |

### unit_quaternion_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 41870 | 23530 | x1.00 |

### unit_quaternion_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 41570 | 23230 | x1.00 |
| `alt_sandwich` | 43060 | 24720 | x1.06 |
| `alt_via_matrix` | 50680 | 32340 | x1.39 |

### unit_quaternion_transform_vector_x2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_matrix` | 56210 | 37870 | x1.00 |
| `expanded` | 65920 | 47580 | x1.26 |

### unit_quaternion_transform_vector_x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_matrix` | 62780 | 44440 | x1.00 |
| `expanded` | 89250 | 70910 | x1.60 |

## nalgebra::linalg::cholesky::benches

### cholesky2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 19940 | 3600 | x1.00 |

### cholesky2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 33450 | 16110 | x1.00 |
| `triangular` | 34420 | 17080 | x1.06 |

### cholesky2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### cholesky2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 29690 | 12350 | x1.00 |

### cholesky2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 35120 | 17680 | x1.00 |
| `alt_recip` | 35700 | 18260 | x1.03 |

### cholesky3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 22390 | 5450 | x1.00 |

### cholesky3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 52450 | 33010 | x1.00 |
| `triangular` | 55360 | 35920 | x1.09 |

### cholesky3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### cholesky3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 44920 | 25480 | x1.00 |

### cholesky3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 47060 | 28320 | x1.00 |
| `alt_recip` | 48150 | 29410 | x1.04 |

### cholesky4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 25040 | 7300 | x1.00 |

### cholesky4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 86200 | 58960 | x1.00 |
| `triangular` | 92020 | 64780 | x1.10 |

### cholesky4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### cholesky4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 68120 | 44680 | x1.00 |

### cholesky4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 59700 | 39460 | x1.00 |
| `alt_recip` | 61300 | 41060 | x1.04 |

### cholesky6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 30940 | 11000 | x1.00 |

### cholesky6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 173690 | 132250 | x1.00 |
| `triangular` | 188240 | 146800 | x1.11 |

### cholesky6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 45940 | 3600 | x1.00 |

### cholesky6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 137530 | 102090 | x1.00 |

### cholesky6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 87080 | 63240 | x1.00 |
| `alt_recip` | 89700 | 65860 | x1.04 |

## nalgebra::linalg::ldlt::benches

### ldlt2_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 17040 | 0 | - |

### ldlt2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 18090 | 1750 | x1.00 |

### ldlt2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 29950 | 12610 | x1.00 |
| `triangular` | 30920 | 13580 | x1.08 |

### ldlt2_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 17940 | 100 | x1.00 |

### ldlt2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 24070 | 6730 | x1.00 |
| `alt_products` | 26120 | 8780 | x1.30 |

### ldlt2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 29040 | 11600 | x1.00 |
| `alt_recip` | 32000 | 14560 | x1.26 |

### ldlt3_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 18240 | 0 | - |

### ldlt3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 20540 | 3600 | x1.00 |

### ldlt3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 47200 | 27760 | x1.00 |
| `triangular` | 50110 | 30670 | x1.10 |

### ldlt3_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 20940 | 0 | - |

### ldlt3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 37010 | 17570 | x1.00 |
| `alt_products` | 42560 | 23120 | x1.32 |

### ldlt3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 38160 | 19420 | x1.00 |
| `alt_recip` | 42600 | 23860 | x1.23 |

### ldlt4_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 19640 | 0 | - |

### ldlt4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 23190 | 5450 | x1.00 |

### ldlt4_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 79200 | 51960 | x1.00 |
| `triangular` | 85020 | 57780 | x1.11 |

### ldlt4_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 28840 | 1600 | x1.00 |

### ldlt4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 57720 | 34280 | x1.00 |
| `alt_products` | 68820 | 45380 | x1.32 |

### ldlt4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 47980 | 27740 | x1.00 |
| `alt_recip` | 53900 | 33660 | x1.21 |

### ldlt6_d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 23340 | 0 | - |

### ldlt6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `diagonal_product` | 29390 | 9150 | x1.00 |

### ldlt6_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 163490 | 121750 | x1.00 |
| `triangular` | 178040 | 136300 | x1.12 |

### ldlt6_l

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 46240 | 3600 | x1.00 |

### ldlt6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `factorize` | 122450 | 86710 | x1.00 |
| `alt_products` | 150200 | 114460 | x1.32 |

### ldlt6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 70020 | 45880 | x1.00 |
| `alt_recip` | 78900 | 54760 | x1.19 |

## nalgebra::linalg::lu::lu2::tests

### lu2_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 19690 | 2950 | x1.00 |

### lu2_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 18340 | 100 | x1.00 |
| `u` | 18340 | 100 | x1.00 |

### lu2_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 17450 | 700 | x1.00 |

### lu2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 25270 | 7230 | x1.00 |
| `pivot` | 28590 | 10550 | x1.46 |

### lu2_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 16740 | 0 | - |

### lu2_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 18640 | 800 | x1.00 |

### lu2_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 20440 | 1200 | x1.00 |

### lu2_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 32380 | 14240 | x1.00 |
| `alt_recip` | 34900 | 16760 | x1.18 |

### lu2_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 31890 | 14540 | x1.00 |

### lu2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `columns` | 37720 | 19180 | x1.00 |
| `alt_recip` | 38400 | 19860 | x1.04 |
| `alt_solve_columns` | 50620 | 32080 | x1.67 |

## nalgebra::linalg::lu::lu3::tests

### lu3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 25820 | 7880 | x1.00 |

### lu3_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 21940 | 0 | - |
| `u` | 21940 | 0 | - |

### lu3_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 18750 | 800 | x1.00 |

### lu3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 44810 | 23270 | x1.00 |
| `pivot` | 54800 | 33260 | x1.43 |

### lu3_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 18640 | 0 | - |

### lu3_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 21480 | 1740 | x1.00 |

### lu3_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 27320 | 3380 | x1.00 |

### lu3_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 43600 | 23560 | x1.00 |
| `alt_recip` | 47600 | 27560 | x1.17 |

### lu3_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 42610 | 23860 | x1.00 |

### lu3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 67190 | 44950 | x1.00 |
| `columns` | 68470 | 46230 | x1.03 |
| `alt_solve_columns` | 97820 | 75580 | x1.68 |

### lu3_vs_matrix3_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 28200 | 10660 | x1.00 |
| `lu` | 58780 | 41240 | x3.87 |

### lu3_vs_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lu` | 101430 | 79590 | x1.00 |
| `cofactors` | 110780 | 88940 | x1.12 |

## nalgebra::linalg::lu::lu4::tests

### lu4_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 30680 | 11140 | x1.00 |

### lu4_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 30640 | 1600 | x1.00 |
| `u` | 30640 | 1600 | x1.00 |

### lu4_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 20450 | 900 | x1.00 |

### lu4_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 83120 | 54680 | x1.00 |
| `pivot` | 104430 | 75990 | x1.39 |

### lu4_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 20840 | 0 | - |

### lu4_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 25330 | 3290 | x1.00 |

### lu4_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 45140 | 12700 | x1.00 |

### lu4_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 56520 | 34180 | x1.00 |
| `alt_recip` | 62000 | 39660 | x1.16 |

### lu4_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 55030 | 34480 | x1.00 |

### lu4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 115190 | 84250 | x1.00 |
| `columns` | 120810 | 89870 | x1.07 |
| `alt_solve_columns` | 174260 | 143320 | x1.70 |

## nalgebra::linalg::lu::lu6::benches

### lu6_determinant

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 42800 | 18860 | x1.00 |

### lu6_factors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `l` | 49940 | 3600 | x1.00 |
| `u` | 49940 | 3600 | x1.00 |

### lu6_is_invertible

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pivots` | 25050 | 1100 | x1.00 |

### lu6_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_no_pivot` | 218750 | 174310 | x1.00 |
| `pivot` | 283850 | 239410 | x1.37 |

### lu6_p

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `field` | 26440 | 0 | - |

### lu6_permute

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 42640 | 14800 | x1.00 |

### lu6_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpositions` | 97640 | 44800 | x1.00 |

### lu6_solve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `substitution` | 93160 | 65020 | x1.00 |
| `alt_recip` | 101600 | 73460 | x1.13 |

### lu6_solve_singular

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `none` | 90670 | 65320 | x1.00 |

### lu6_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 262640 | 213300 | x1.00 |
| `columns` | 282760 | 233420 | x1.09 |
| `alt_solve_columns` | 451860 | 402520 | x1.89 |

### matrix6_lu

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `determinant` | 277710 | 254770 | x1.00 |
| `solve` | 328070 | 305130 | x1.20 |
| `try_inverse` | 521570 | 498630 | x1.96 |

## nalgebra::linalg::symmetric_eigen2::tests

### symmetric_eigen2_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 23810 | 7470 | x1.00 |

### symmetric_eigen2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 40870 | 24530 | x1.00 |

### symmetric_eigen2_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 50640 | 15410 | x1.00 |

## nalgebra::linalg::symmetric_eigen3::tests

### symmetric_eigen3_eigenvalues

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `without_eigenvectors` | 320410 | 303470 | x1.00 |
| `via_new` | 567710 | 550770 | x1.81 |

### symmetric_eigen3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `jacobi_3_sweeps` | 442130 | 425190 | x1.00 |
| `no_renormalisation` | 545250 | 528310 | x1.24 |
| `jacobi_4_sweeps` | 567710 | 550770 | x1.30 |
| `diagonal_input` | 568170 | 551230 | x1.30 |
| `jacobi_5_sweeps` | 693290 | 676350 | x1.59 |
| `jacobi_6_sweeps` | 818870 | 801930 | x1.89 |

### symmetric_eigen3_recompose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quadform` | 601480 | 32610 | x1.00 |

### symmetric_eigen3_sweep

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_rotations_without_eigenvectors` | 92200 | 73460 | x1.00 |
| `three_rotations` | 144120 | 125380 | x1.71 |

## nalgebra_testing::tests

### testing

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add_u64` | 15910 | 370 | x1.00 |

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

### wide_mixed

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wide` | 20290 | 2350 | x1.00 |

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

