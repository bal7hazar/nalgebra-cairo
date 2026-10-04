# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry::isometry2::benches

### isometry2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 15600 | 5880 | x1.00 |

### isometry2_append_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose_rotate` | 18230 | 7420 | x1.00 |

### isometry2_append_rotation_wrt_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 14870 | 3860 | x1.00 |

### isometry2_append_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shift_rotate` | 23240 | 11830 | x1.00 |

### isometry2_append_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 12790 | 1780 | x1.00 |

### isometry2_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 10810 | 400 | x1.00 |

### isometry2_from_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 9910 | 100 | x1.00 |

### isometry2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 9610 | -600 | - |

### isometry2_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 20110 | 8900 | x1.00 |
| `alt_inverse_then_mul` | 23290 | 12080 | x1.36 |

### isometry2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 14470 | 4260 | x1.00 |

### isometry2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract_rotate` | 14950 | 5140 | x1.00 |

### isometry2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 13470 | 3660 | x1.00 |

### isometry2_lerp_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 90060 | 78450 | x1.00 |

### isometry2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 19030 | 7820 | x1.00 |

### isometry2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 41610 | 31400 | x1.00 |

### isometry2_prepend_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 14870 | 3860 | x1.00 |

### isometry2_prepend_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate_add` | 15270 | 4260 | x1.00 |

### isometry2_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 40410 | 30800 | x1.00 |

### isometry2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3` | 9210 | 500 | x1.00 |

### isometry2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13870 | 4060 | x1.00 |
| `alt_rotate_then_add` | 15150 | 5340 | x1.32 |

### isometry2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 13470 | 3660 | x1.00 |

### isometry2_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 9910 | 100 | x1.00 |

## nalgebra_tests_geometry::isometry3::benches

### isometry3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 21420 | 10500 | x1.00 |

### isometry3_append_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose_rotate` | 44900 | 31590 | x1.00 |

### isometry3_append_rotation_wrt_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 22930 | 9320 | x1.00 |

### isometry3_append_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shift_rotate` | 48820 | 34710 | x1.00 |

### isometry3_append_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 16330 | 2820 | x1.00 |

### isometry3_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 89780 | 76670 | x1.00 |

### isometry3_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 13210 | 700 | x1.00 |

### isometry3_from_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 11310 | -200 | - |

### isometry3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 11110 | -1200 | - |

### isometry3_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 44890 | 30980 | x1.00 |
| `alt_conjugate_then_mul` | 48920 | 35010 | x1.13 |
| `alt_inverse_then_mul` | 63910 | 50000 | x1.61 |

### isometry3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 32950 | 20640 | x1.00 |

### isometry3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract_rotate` | 32970 | 21860 | x1.00 |

### isometry3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 30750 | 19640 | x1.00 |

### isometry3_lerp_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_sin` | 144860 | 130550 | x1.00 |

### isometry3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 111850 | 98740 | x1.00 |

### isometry3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 43270 | 29360 | x1.00 |

### isometry3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 62980 | 50670 | x1.00 |

### isometry3_prepend_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 22930 | 9320 | x1.00 |

### isometry3_prepend_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate_add` | 36880 | 23370 | x1.00 |

### isometry3_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 61280 | 49770 | x1.00 |

### isometry3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 31980 | 22670 | x1.00 |

### isometry3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 31350 | 20240 | x1.00 |
| `alt_rotate_then_add` | 35700 | 24590 | x1.21 |
| `alt_rotation_matrix` | 44980 | 33870 | x1.67 |

### isometry3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 30750 | 19640 | x1.00 |

### isometry3_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 11310 | -200 | - |

## nalgebra_tests_geometry::quaternion::benches

### quaternion_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 14510 | 4700 | x1.00 |

### quaternion_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 14470 | 3260 | x1.00 |

### quaternion_as_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10610 | 400 | x1.00 |

### quaternion_conj_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_conjugate_then_mul` | 21130 | 9920 | x1.00 |

### quaternion_conjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 11210 | 1000 | x1.00 |

### quaternion_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 11890 | 2180 | x1.00 |

### quaternion_from_imag

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10110 | 100 | x1.00 |

### quaternion_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10810 | 400 | x1.00 |

### quaternion_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8710 | -500 | - |

### quaternion_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19430 | 7820 | x1.00 |

### quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_wide` | 20230 | 9020 | x1.00 |
| `alt_sum_prod4` | 23660 | 12450 | x1.38 |
| `alt_unfused` | 47270 | 36060 | x4.00 |

### quaternion_neg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 11410 | 1200 | x1.00 |

### quaternion_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 11210 | 400 | x1.00 |

### quaternion_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm4` | 11130 | 2420 | x1.00 |

### quaternion_norm_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 10890 | 2180 | x1.00 |

### quaternion_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 22000 | 11790 | x1.00 |
| `divisions` | 24990 | 14780 | x1.25 |

### quaternion_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 17230 | 6620 | x1.00 |

### quaternion_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 14470 | 3260 | x1.00 |

### quaternion_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 23160 | 12950 | x1.00 |
| `divisions` | 26120 | 15910 | x1.23 |

### quaternion_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 22870 | 12260 | x1.00 |

### quaternion_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10010 | 300 | x1.00 |

## nalgebra_tests_geometry::quaternion::benches_ext

### quaternion_acos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 160390 | 150180 | x1.00 |

### quaternion_acosh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 169660 | 159450 | x1.00 |

### quaternion_asin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 171630 | 161420 | x1.00 |

### quaternion_asinh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 134140 | 123930 | x1.00 |

### quaternion_atan

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 164230 | 154020 | x1.00 |

### quaternion_atanh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 182450 | 172240 | x1.00 |

### quaternion_cast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 10210 | 0 | - |

### quaternion_cos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 104560 | 94350 | x1.00 |
| `alt_fixed` | 114190 | 103980 | x1.10 |

### quaternion_cosh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 105130 | 94920 | x1.00 |
| `alt_fixed` | 118130 | 107920 | x1.14 |

### quaternion_exp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 78130 | 67920 | x1.00 |

### quaternion_exp_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 78130 | 67920 | x1.00 |

### quaternion_from_array

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 10610 | 400 | x1.00 |

### quaternion_from_polar_decomposition

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 50760 | 39950 | x1.00 |

### quaternion_half

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_scale_half` | 16830 | 6620 | x1.00 |
| `div4` | 21480 | 11270 | x1.70 |

### quaternion_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 8710 | 0 | - |

### quaternion_inner

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reduced` | 21560 | 10350 | x1.00 |
| `alt_products` | 50150 | 38940 | x3.76 |

### quaternion_is_pure

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compare` | 9110 | 300 | x1.00 |

### quaternion_left_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 42100 | 30890 | x1.00 |

### quaternion_ln

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 83980 | 73770 | x1.00 |
| `alt_acos` | 85580 | 75370 | x1.02 |

### quaternion_magnitude

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm4` | 11130 | 2420 | x1.00 |

### quaternion_magnitude_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 10890 | 2180 | x1.00 |

### quaternion_one

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8710 | -500 | - |

### quaternion_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cross` | 16550 | 5340 | x1.00 |

### quaternion_polar_decomposition

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 61930 | 51220 | x1.00 |

### quaternion_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exp_ln` | 158320 | 148110 | x1.00 |

### quaternion_project

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `right_div` | 55680 | 44470 | x1.00 |

### quaternion_pure

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10310 | 100 | x1.00 |

### quaternion_reject

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `right_div` | 50570 | 39360 | x1.00 |

### quaternion_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 30360 | 20550 | x1.00 |

### quaternion_right_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_inverse_then_mul` | 36310 | 25100 | x1.00 |
| `fused` | 45230 | 34020 | x1.36 |

### quaternion_sin

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 104260 | 94050 | x1.00 |
| `alt_fixed` | 113990 | 103780 | x1.10 |

### quaternion_sinh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 105130 | 94920 | x1.00 |
| `alt_fixed` | 118130 | 107920 | x1.14 |
| `alt_exp_difference` | 167350 | 157140 | x1.66 |

### quaternion_sqrt

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 45720 | 35510 | x1.00 |
| `alt_powf` | 158320 | 148110 | x4.17 |

### quaternion_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_mul` | 19230 | 9020 | x1.00 |
| `reduced` | 20160 | 9950 | x1.10 |

### quaternion_tan

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 232830 | 222620 | x1.00 |

### quaternion_tanh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `upstream` | 234270 | 224060 | x1.00 |

### quaternion_ulps_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 22050 | 12240 | x1.00 |

### real_cosh_sinh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 38500 | 29890 | x1.00 |
| `two_exp` | 55680 | 47070 | x1.57 |
| `alt_fixed` | 66150 | 57540 | x1.93 |

## nalgebra_tests_geometry::quaternion_inplace::benches

### quaternion_div_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_by_value` | 20370 | 11060 | x1.00 |
| `div_assign` | 20370 | 11060 | x1.00 |

### quaternion_mul_assign

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_by_value` | 18330 | 9020 | x1.00 |
| `mul_assign` | 18330 | 9020 | x1.00 |

### quaternion_normalize_mut

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_by_value` | 22490 | 13180 | x1.00 |
| `normalize_mut` | 22490 | 13180 | x1.00 |

### quaternion_try_inverse_mut

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_inverse_mut` | 22620 | 13310 | x1.00 |
| `alt_by_value` | 23720 | 14410 | x1.08 |

## nalgebra_tests_geometry::rotation2::benches

### rotation2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 12020 | 2210 | x1.00 |

### rotation2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 38140 | 29430 | x1.00 |

### rotation2_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_columns` | 42700 | 32990 | x1.00 |
| `alt_product_then_angle` | 49190 | 39480 | x1.20 |

### rotation2_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 24990 | 14780 | x1.00 |

### rotation2_from_matrix_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 10210 | 0 | - |

### rotation2_from_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 12210 | 2400 | x1.00 |

### rotation2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 9410 | -200 | - |

### rotation2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 11810 | 1600 | x1.00 |

### rotation2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transposed` | 13470 | 3660 | x1.00 |

### rotation2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_inverse_then_transform` | 13470 | 3660 | x1.00 |
| `transposed` | 13470 | 3660 | x1.00 |

### rotation2_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 9010 | 300 | x1.00 |

### rotation2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_complex` | 15470 | 4260 | x1.00 |
| `matrix` | 21160 | 9950 | x2.34 |

### rotation2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 41210 | 31600 | x1.00 |

### rotation2_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 73320 | 62710 | x1.00 |

### rotation2_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_first_column` | 19090 | 8880 | x1.00 |
| `closed_form_limit` | 25190 | 14980 | x1.69 |

### rotation2_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 27470 | 17060 | x1.00 |
| `alt_unit_complex` | 27670 | 17260 | x1.01 |

### rotation2_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 79580 | 68770 | x1.00 |

### rotation2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 8710 | 0 | - |

### rotation2_to_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 10010 | 800 | x1.00 |

### rotation2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unit_complex` | 13470 | 3660 | x1.00 |
| `matrix` | 13470 | 3660 | x1.00 |

### rotation2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unit_complex` | 13470 | 3660 | x1.00 |
| `matrix` | 13470 | 3660 | x1.00 |

## nalgebra_tests_geometry::rotation3::benches

### rotation3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `entries` | 25660 | 13850 | x1.00 |

### rotation3_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_trace` | 43340 | 33630 | x1.00 |
| `alt_quaternion` | 73280 | 63570 | x1.89 |

### rotation3_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `antisymmetric_part` | 25530 | 14820 | x1.00 |

### rotation3_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `asin_atan2` | 100470 | 90760 | x1.00 |

### rotation3_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `two_normalizations` | 57480 | 44170 | x1.00 |

### rotation3_from_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_quaternion` | 73400 | 60490 | x1.00 |
| `rodrigues` | 73980 | 61070 | x1.01 |

### rotation3_from_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_sin_cos` | 126040 | 113130 | x1.00 |

### rotation3_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_and_rodrigues` | 89310 | 76800 | x1.00 |

### rotation3_from_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 35080 | 22370 | x1.00 |

### rotation3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 10510 | -1200 | - |

### rotation3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 14610 | 900 | x1.00 |

### rotation3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `tr_mul_vec` | 17650 | 6140 | x1.00 |

### rotation3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 58980 | 45670 | x1.00 |

### rotation3_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 13710 | 0 | - |

### rotation3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix_product` | 34330 | 18620 | x1.00 |

### rotation3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_gram_schmidt` | 60560 | 46850 | x1.00 |
| `alt_newton` | 87790 | 74080 | x1.58 |
| `closed_form` | 514470 | 500760 | x10.69 |

### rotation3_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic_quaternion` | 100050 | 86740 | x1.00 |
| `alt_axis_angle` | 161230 | 147920 | x1.71 |

### rotation3_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_times_angle` | 70130 | 59420 | x1.00 |

### rotation3_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_rodrigues` | 162910 | 149600 | x1.00 |

### rotation3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 9710 | 0 | - |

### rotation3_to_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 39190 | 27980 | x1.00 |

### rotation3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_vec` | 17650 | 6140 | x1.00 |

### rotation3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_vec` | 17650 | 6140 | x1.00 |

## nalgebra_tests_geometry::similarity2::benches

### similarity2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 21610 | 11600 | x1.00 |
| `alt_reciprocal` | 26780 | 16770 | x1.45 |

### similarity2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 20130 | 10120 | x1.00 |
| `alt_reciprocal` | 25300 | 15290 | x1.51 |

### similarity2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_translate` | 17430 | 7420 | x1.00 |
| `alt_rotate_scale_add` | 20840 | 10830 | x1.46 |

### similarity2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_after_rotate` | 17030 | 7020 | x1.00 |

## nalgebra_tests_geometry::similarity3::benches

### similarity3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_reciprocal` | 47920 | 36610 | x1.00 |
| `division` | 48780 | 37470 | x1.02 |

### similarity3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 40290 | 28980 | x1.00 |
| `alt_reciprocal` | 45700 | 34390 | x1.19 |

### similarity3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_translate` | 39720 | 28410 | x1.00 |
| `alt_rotate_scale_add` | 41040 | 29730 | x1.05 |

### similarity3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_after_rotate` | 35990 | 24680 | x1.00 |

## nalgebra_tests_geometry::translation2::benches

### translation2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 12120 | 3200 | x1.00 |

### translation2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8410 | -400 | - |

### translation2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `negate` | 9410 | 600 | x1.00 |

### translation2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract` | 10990 | 1580 | x1.00 |
| `alt_inverse_then_transform` | 11590 | 2180 | x1.38 |

### translation2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 10990 | 1580 | x1.00 |

### translation2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_components` | 8810 | 0 | - |
| `from_vector` | 8810 | 0 | - |

### translation2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3` | 8310 | 0 | - |

### translation2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 10990 | 1580 | x1.00 |

## nalgebra_tests_geometry::translation3::benches

### translation3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 13860 | 4540 | x1.00 |

### translation3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8910 | -600 | - |

### translation3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `negate` | 10410 | 900 | x1.00 |

### translation3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract` | 12730 | 2420 | x1.00 |
| `alt_inverse_then_transform` | 13630 | 3320 | x1.37 |

### translation3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 12730 | 2420 | x1.00 |

### translation3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_components` | 9510 | 0 | - |
| `from_vector` | 9510 | 0 | - |

### translation3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 8510 | 0 | - |

### translation3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 12730 | 2420 | x1.00 |

## nalgebra_tests_geometry::unit_complex::benches

### unit_complex_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `all_compared` | 11040 | 2030 | x1.00 |

### unit_complex_accessors

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `complex` | 9010 | 200 | x1.00 |
| `cos_sin_angle` | 9410 | 600 | x3.00 |

### unit_complex_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 37740 | 29430 | x1.00 |

### unit_complex_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_atan2` | 41900 | 32990 | x1.00 |

### unit_complex_from_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 9410 | 200 | x1.00 |

### unit_complex_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8510 | -100 | - |
| `from_cos_sin_unchecked` | 8510 | -100 | - |

### unit_complex_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate` | 10210 | 1400 | x1.00 |

### unit_complex_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |

### unit_complex_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |
| `alt_inverse_then_transform` | 13370 | 3960 | x1.08 |

### unit_complex_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |

### unit_complex_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 39810 | 31200 | x1.00 |

### unit_complex_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 71520 | 62310 | x1.00 |

### unit_complex_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_and_divisions` | 17290 | 8480 | x1.00 |

### unit_complex_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_add` | 15730 | 6920 | x1.00 |
| `alt_mul_add_each` | 16130 | 7320 | x1.06 |
| `alt_lerp` | 16330 | 7520 | x1.09 |
| `alt_literal` | 16570 | 7760 | x1.12 |
| `alt_exact` | 17290 | 8480 | x1.23 |

### unit_complex_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 26070 | 16660 | x1.00 |
| `alt_atan2` | 91600 | 82190 | x4.93 |

### unit_complex_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |

### unit_complex_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 77280 | 67470 | x1.00 |

### unit_complex_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `angle_then_compose` | 81470 | 71660 | x1.00 |

### unit_complex_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 8610 | 300 | x1.00 |

### unit_complex_to_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expand` | 8910 | 600 | x1.00 |

### unit_complex_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |

### unit_complex_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |

## nalgebra_tests_geometry::unit_complex::benches_ext

### unit_complex_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 39410 | 30600 | x1.00 |

### unit_complex_cast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 8810 | 0 | - |

### unit_complex_default

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8110 | -100 | - |

### unit_complex_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13070 | 3660 | x1.00 |

### unit_complex_div_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `kernels` | 13470 | 3660 | x1.00 |

### unit_complex_from_basis_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 9410 | 200 | x1.00 |

### unit_complex_from_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize` | 17560 | 8750 | x1.00 |

### unit_complex_from_complex_and_get

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize` | 18160 | 8850 | x1.00 |

### unit_complex_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 20760 | 11550 | x1.00 |
| `alt_iterate` | 170180 | 160970 | x13.94 |

### unit_complex_from_matrix_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `muller_4` | 170180 | 160970 | x1.00 |

### unit_complex_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 39810 | 31200 | x1.00 |

### unit_complex_into_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 9610 | -200 | - |

### unit_complex_into_similarity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10210 | -100 | - |

### unit_complex_inverse_transform_unit_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_transform_vector` | 13070 | 3660 | x1.00 |

### unit_complex_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 18230 | 7420 | x1.00 |

### unit_complex_mul_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `kernels` | 13470 | 3660 | x1.00 |

### unit_complex_mul_similarity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 19030 | 7520 | x1.00 |

### unit_complex_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 14270 | 3860 | x1.00 |

### unit_complex_one

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8110 | -100 | - |

### unit_complex_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 21000 | 11990 | x1.00 |

### unit_complex_rotation_between_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 26070 | 16660 | x1.00 |

### unit_complex_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 37740 | 29430 | x1.00 |

### unit_complex_scaled_rotation_between_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2` | 77280 | 67470 | x1.00 |

### unit_complex_transform_unit_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transform_vector` | 13070 | 3660 | x1.00 |

### unit_complex_ulps_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 14990 | 5980 | x1.00 |

## nalgebra_tests_geometry::unit_quaternion::benches

### unit_quaternion_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 14490 | 4680 | x1.00 |

### unit_quaternion_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 40880 | 32170 | x1.00 |
| `atan2` | 44200 | 35490 | x1.10 |

### unit_quaternion_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product_and_atan2` | 54820 | 45110 | x1.00 |

### unit_quaternion_append_axisangle_linearized

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reduced_product` | 46780 | 35770 | x1.00 |
| `alt_exact` | 70100 | 59090 | x1.65 |

### unit_quaternion_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_new` | 29180 | 19470 | x1.00 |

### unit_quaternion_conj_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_conjugate_then_mul` | 21130 | 9920 | x1.00 |

### unit_quaternion_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 11890 | 2180 | x1.00 |

### unit_quaternion_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `five_entries` | 110990 | 102280 | x1.00 |

### unit_quaternion_from_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 48130 | 37720 | x1.00 |

### unit_quaternion_from_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_sin_cos` | 124620 | 114210 | x1.00 |

### unit_quaternion_from_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 39190 | 27980 | x1.00 |

### unit_quaternion_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_division` | 59980 | 49970 | x1.00 |
| `alt_unit_axis` | 68660 | 58650 | x1.17 |

### unit_quaternion_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8710 | -500 | - |

### unit_quaternion_imag

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10010 | 300 | x1.00 |

### unit_quaternion_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate` | 11210 | 1000 | x1.00 |

### unit_quaternion_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `folded` | 30150 | 19640 | x1.00 |
| `alt_conjugate_then_transform` | 33480 | 22970 | x1.17 |

### unit_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hamilton` | 20230 | 9020 | x1.00 |

### unit_quaternion_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 24990 | 14780 | x1.00 |

### unit_quaternion_new_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10210 | 0 | - |

### unit_quaternion_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 34850 | 23240 | x1.00 |

### unit_quaternion_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_angle` | 107400 | 96790 | x1.00 |

### unit_quaternion_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 25530 | 15320 | x1.00 |

### unit_quaternion_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 20890 | 10680 | x1.00 |
| `alt_exact` | 25530 | 15320 | x1.43 |

### unit_quaternion_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 73680 | 62870 | x1.00 |
| `alt_axis_angle` | 137190 | 126380 | x2.01 |

### unit_quaternion_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_product` | 21130 | 9920 | x1.00 |

### unit_quaternion_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_factor` | 56050 | 46340 | x1.00 |
| `divisions` | 64420 | 54710 | x1.18 |

### unit_quaternion_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos` | 137190 | 126380 | x1.00 |

### unit_quaternion_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_sin` | 126750 | 115140 | x1.00 |

### unit_quaternion_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 31080 | 22370 | x1.00 |

### unit_quaternion_to_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_one_minus` | 34480 | 21770 | x1.00 |
| `fused` | 35080 | 22370 | x1.03 |

### unit_quaternion_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 30150 | 19640 | x1.00 |

### unit_quaternion_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 30150 | 19640 | x1.00 |
| `alt_sandwich` | 31280 | 20770 | x1.06 |
| `alt_via_matrix` | 41550 | 31040 | x1.58 |

### unit_quaternion_transform_vector_x2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_matrix` | 46200 | 35690 | x1.00 |
| `expanded` | 50730 | 40220 | x1.13 |

### unit_quaternion_transform_vector_x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_matrix` | 52260 | 41750 | x1.00 |
| `expanded` | 70290 | 59780 | x1.43 |

## nalgebra_tests_geometry::unit_quaternion::benches_ext

### unit_quaternion_cast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `into` | 10210 | 0 | - |

### unit_quaternion_default

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8710 | -500 | - |

### unit_quaternion_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_inverse_then_mul` | 21130 | 9920 | x1.00 |
| `fused` | 23360 | 12150 | x1.22 |

### unit_quaternion_div_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 49330 | 36020 | x1.00 |
| `alt_inverse_then_mul` | 62710 | 49400 | x1.37 |

### unit_quaternion_div_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `convert_then_mul_conj` | 67490 | 40330 | x1.00 |

### unit_quaternion_div_similarity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 70000 | 55990 | x1.00 |
| `alt_inverse_then_mul` | 82780 | 68770 | x1.23 |

### unit_quaternion_exp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quaternion_exp` | 78130 | 67920 | x1.00 |

### unit_quaternion_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotation3` | 83060 | 72250 | x1.00 |

### unit_quaternion_from_basis_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 54240 | 28080 | x1.00 |

### unit_quaternion_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 501470 | 490260 | x1.00 |
| `alt_iterate` | 2350880 | 2339670 | x4.77 |

### unit_quaternion_from_matrix_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `muller_8` | 908910 | 897700 | x1.00 |

### unit_quaternion_from_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize` | 25530 | 15320 | x1.00 |

### unit_quaternion_from_scaled_axis_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new_eps` | 60550 | 50540 | x1.00 |

### unit_quaternion_into_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 11510 | -200 | - |

### unit_quaternion_into_similarity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 12110 | -100 | - |

### unit_quaternion_inverse_transform_unit_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_transform_vector` | 30150 | 19640 | x1.00 |

### unit_quaternion_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `quaternion_lerp` | 19030 | 7820 | x1.00 |

### unit_quaternion_ln

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scaled_axis` | 65020 | 54810 | x1.00 |

### unit_quaternion_look_at_lh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `face_towards` | 84060 | 73250 | x1.00 |

### unit_quaternion_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `face_towards` | 84660 | 73850 | x1.00 |

### unit_quaternion_mean_of

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `squarings` | 481820 | 469510 | x1.00 |

### unit_quaternion_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 42070 | 28760 | x1.00 |

### unit_quaternion_mul_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `convert_then_mul` | 64360 | 37200 | x1.00 |

### unit_quaternion_mul_similarity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 42870 | 28860 | x1.00 |

### unit_quaternion_mul_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 32550 | 20040 | x1.00 |

### unit_quaternion_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 59980 | 49970 | x1.00 |

### unit_quaternion_new_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `threshold` | 60550 | 50540 | x1.00 |

### unit_quaternion_new_observer_frames

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `face_towards` | 83060 | 72250 | x1.00 |

### unit_quaternion_one

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 8710 | -500 | - |

### unit_quaternion_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 30660 | 20850 | x1.00 |

### unit_quaternion_rotation_between_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 34250 | 23440 | x1.00 |

### unit_quaternion_scaled_rotation_between_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos` | 99570 | 88360 | x1.00 |

### unit_quaternion_to_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `euler_angles` | 111690 | 101980 | x1.00 |

### unit_quaternion_transform_unit_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transform_vector` | 30150 | 19640 | x1.00 |

### unit_quaternion_ulps_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 21840 | 12030 | x1.00 |

