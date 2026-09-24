# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::geometry::isometry2::benches

### isometry2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 23430 | 5880 | x1.00 |

### isometry2_append_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose_rotate` | 26060 | 7420 | x1.00 |

### isometry2_append_rotation_wrt_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 22700 | 3860 | x1.00 |

### isometry2_append_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shift_rotate` | 31070 | 11830 | x1.00 |

### isometry2_append_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 20620 | 1780 | x1.00 |

### isometry2_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 18640 | 400 | x1.00 |

### isometry2_from_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 17740 | 100 | x1.00 |

### isometry2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 17440 | -600 | - |

### isometry2_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 30470 | 11430 | x1.00 |
| `alt_inverse_then_mul` | 33750 | 14710 | x1.29 |

### isometry2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 22700 | 4660 | x1.00 |

### isometry2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract_rotate` | 22780 | 5140 | x1.00 |

### isometry2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 21300 | 3660 | x1.00 |

### isometry2_lerp_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 40890 | 21450 | x1.00 |

### isometry2_lerp_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 97890 | 78450 | x1.00 |

### isometry2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 29390 | 10350 | x1.00 |

### isometry2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 49440 | 31400 | x1.00 |

### isometry2_prepend_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 22700 | 3860 | x1.00 |

### isometry2_prepend_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate_add` | 23100 | 4260 | x1.00 |

### isometry2_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 26920 | 8680 | x1.00 |

### isometry2_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 25360 | 7120 | x1.00 |

### isometry2_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 48240 | 30800 | x1.00 |

### isometry2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3` | 17040 | 500 | x1.00 |

### isometry2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21700 | 4060 | x1.00 |
| `alt_rotate_then_add` | 22980 | 5340 | x1.32 |

### isometry2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 21300 | 3660 | x1.00 |

### isometry2_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 17740 | 100 | x1.00 |

## nalgebra::geometry::isometry3::benches

### isometry3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 28680 | 9930 | x1.00 |

### isometry3_append_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose_rotate` | 55560 | 34420 | x1.00 |

### isometry3_append_rotation_wrt_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 33690 | 12250 | x1.00 |

### isometry3_append_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shift_rotate` | 61480 | 39540 | x1.00 |

### isometry3_append_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 24160 | 2820 | x1.00 |

### isometry3_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 97610 | 76670 | x1.00 |

### isometry3_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 21040 | 700 | x1.00 |

### isometry3_from_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 19140 | -200 | - |

### isometry3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 18940 | -1200 | - |

### isometry3_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 60380 | 38640 | x1.00 |
| `alt_conjugate_then_mul` | 60980 | 39240 | x1.02 |
| `alt_inverse_then_mul` | 80430 | 58690 | x1.52 |

### isometry3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 44110 | 23970 | x1.00 |

### isometry3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract_rotate` | 45130 | 26190 | x1.00 |

### isometry3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 41010 | 22070 | x1.00 |

### isometry3_lerp_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 58640 | 36500 | x1.00 |

### isometry3_lerp_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_sin` | 155960 | 133820 | x1.00 |

### isometry3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 120980 | 100040 | x1.00 |

### isometry3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 57060 | 35320 | x1.00 |

### isometry3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 70810 | 50670 | x1.00 |

### isometry3_prepend_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 33690 | 12250 | x1.00 |

### isometry3_prepend_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate_add` | 45010 | 23670 | x1.00 |

### isometry3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 35220 | 15080 | x1.00 |

### isometry3_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 31120 | 10980 | x1.00 |

### isometry3_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 69110 | 49770 | x1.00 |

### isometry3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 39810 | 22670 | x1.00 |

### isometry3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 42210 | 23270 | x1.00 |
| `alt_rotate_then_add` | 45960 | 27020 | x1.16 |
| `alt_rotation_matrix` | 52810 | 33870 | x1.46 |

### isometry3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 41010 | 22070 | x1.00 |

### isometry3_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 19140 | -200 | - |

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

### quaternion_conj_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30590 | 11550 | x1.00 |
| `alt_conjugate_then_mul` | 31190 | 12150 | x1.05 |

### quaternion_conjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `neg` | 19040 | 1000 | x1.00 |

### quaternion_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19720 | 2180 | x1.00 |

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
| `fused` | 27260 | 7820 | x1.00 |

### quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_wide` | 30590 | 11550 | x1.00 |
| `alt_sum_prod4` | 31490 | 12450 | x1.08 |
| `alt_unfused` | 55100 | 36060 | x3.12 |

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
| `fused` | 18720 | 2180 | x1.00 |

### quaternion_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 29830 | 11790 | x1.00 |
| `divisions` | 32820 | 14780 | x1.25 |

### quaternion_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 25060 | 6620 | x1.00 |

### quaternion_sub

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `op` | 22300 | 3260 | x1.00 |

### quaternion_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 30990 | 12950 | x1.00 |
| `divisions` | 33950 | 15910 | x1.23 |

### quaternion_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 30700 | 12260 | x1.00 |

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
| `atan2` | 45970 | 29430 | x1.00 |

### rotation2_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_columns` | 50530 | 32990 | x1.00 |
| `alt_product_then_angle` | 57020 | 39480 | x1.20 |

### rotation2_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `normalize_first_column` | 26920 | 8880 | x1.00 |

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
| `transposed` | 21300 | 3660 | x1.00 |

### rotation2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_inverse_then_transform` | 21300 | 3660 | x1.00 |
| `transposed` | 21300 | 3660 | x1.00 |

### rotation2_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `accessor` | 16840 | 300 | x1.00 |

### rotation2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_complex` | 23300 | 4260 | x1.00 |
| `matrix` | 28990 | 9950 | x2.34 |

### rotation2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 49040 | 31600 | x1.00 |

### rotation2_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 81150 | 62710 | x1.00 |

### rotation2_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `first_column` | 26920 | 8880 | x1.00 |

### rotation2_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 35300 | 17060 | x1.00 |
| `alt_unit_complex` | 35500 | 17260 | x1.01 |

### rotation2_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 87410 | 68770 | x1.00 |

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
| `alt_unit_complex` | 21300 | 3660 | x1.00 |
| `matrix` | 21300 | 3660 | x1.00 |

### rotation2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_unit_complex` | 21300 | 3660 | x1.00 |
| `matrix` | 21300 | 3660 | x1.00 |

## nalgebra::geometry::rotation3::benches

### rotation3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `entries` | 33490 | 13850 | x1.00 |

### rotation3_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_trace` | 51170 | 33630 | x1.00 |
| `alt_quaternion` | 81110 | 63570 | x1.89 |

### rotation3_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `antisymmetric_part` | 33360 | 14820 | x1.00 |

### rotation3_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `asin_atan2` | 108300 | 90760 | x1.00 |

### rotation3_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `two_normalizations` | 65310 | 44170 | x1.00 |

### rotation3_from_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rodrigues` | 81810 | 61070 | x1.00 |
| `alt_quaternion` | 83060 | 62320 | x1.02 |

### rotation3_from_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_sin_cos` | 133870 | 113130 | x1.00 |

### rotation3_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_and_rodrigues` | 97140 | 76800 | x1.00 |

### rotation3_from_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 42910 | 22370 | x1.00 |

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
| `tr_mul_vec` | 25480 | 6140 | x1.00 |

### rotation3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 66810 | 45670 | x1.00 |

### rotation3_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 21540 | 0 | - |

### rotation3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix_product` | 45690 | 22150 | x1.00 |

### rotation3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 68390 | 46850 | x1.00 |
| `alt_quaternion` | 94820 | 73280 | x1.56 |
| `alt_newton` | 101550 | 80010 | x1.71 |

### rotation3_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic_quaternion` | 107880 | 86740 | x1.00 |
| `alt_axis_angle` | 169060 | 147920 | x1.71 |

### rotation3_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_times_angle` | 77960 | 59420 | x1.00 |

### rotation3_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_rodrigues` | 170740 | 149600 | x1.00 |

### rotation3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 17540 | 0 | - |

### rotation3_to_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 47020 | 27980 | x1.00 |

### rotation3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_vec` | 25480 | 6140 | x1.00 |

### rotation3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_vec` | 25480 | 6140 | x1.00 |

## nalgebra::geometry::similarity2::benches

### similarity2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 29440 | 11600 | x1.00 |
| `alt_reciprocal` | 34610 | 16770 | x1.45 |

### similarity2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 27960 | 10120 | x1.00 |
| `alt_reciprocal` | 33130 | 15290 | x1.51 |

### similarity2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_translate` | 25260 | 7420 | x1.00 |
| `alt_rotate_scale_add` | 28670 | 10830 | x1.46 |

### similarity2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_after_rotate` | 24860 | 7020 | x1.00 |

## nalgebra::geometry::similarity3::benches

### similarity3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_reciprocal` | 60080 | 40940 | x1.00 |
| `division` | 60940 | 41800 | x1.02 |

### similarity3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 50550 | 31410 | x1.00 |
| `alt_reciprocal` | 55960 | 36820 | x1.17 |

### similarity3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_translate` | 49980 | 30840 | x1.00 |
| `alt_rotate_scale_add` | 51300 | 32160 | x1.04 |

### similarity3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_after_rotate` | 46250 | 27110 | x1.00 |

## nalgebra::geometry::translation2::benches

### translation2_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 19950 | 3200 | x1.00 |

### translation2_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16240 | -400 | - |

### translation2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `negate` | 17240 | 600 | x1.00 |

### translation2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract` | 18820 | 1580 | x1.00 |
| `alt_inverse_then_transform` | 19420 | 2180 | x1.38 |

### translation2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 18820 | 1580 | x1.00 |

### translation2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_components` | 16640 | 0 | - |
| `from_vector` | 16640 | 0 | - |

### translation2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3` | 16140 | 0 | - |

### translation2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 18820 | 1580 | x1.00 |

## nalgebra::geometry::translation3::benches

### translation3_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps` | 21690 | 4540 | x1.00 |

### translation3_identity

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `const` | 16740 | -600 | - |

### translation3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `negate` | 18240 | 900 | x1.00 |

### translation3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract` | 20560 | 2420 | x1.00 |
| `alt_inverse_then_transform` | 21460 | 3320 | x1.37 |

### translation3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 20560 | 2420 | x1.00 |

### translation3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_components` | 17340 | 0 | - |
| `from_vector` | 17340 | 0 | - |

### translation3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 16340 | 0 | - |

### translation3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 20560 | 2420 | x1.00 |

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
| `atan2` | 45570 | 29430 | x1.00 |

### unit_complex_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_atan2` | 49730 | 32990 | x1.00 |

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
| `fused` | 20900 | 3660 | x1.00 |

### unit_complex_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20900 | 3660 | x1.00 |
| `alt_inverse_then_transform` | 21200 | 3960 | x1.08 |

### unit_complex_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20900 | 3660 | x1.00 |

### unit_complex_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 47640 | 31200 | x1.00 |

### unit_complex_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 79350 | 62310 | x1.00 |

### unit_complex_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm_and_divisions` | 25120 | 8480 | x1.00 |

### unit_complex_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_add` | 23560 | 6920 | x1.00 |
| `alt_mul_add_each` | 23960 | 7320 | x1.06 |
| `alt_lerp` | 24160 | 7520 | x1.09 |
| `alt_literal` | 24400 | 7760 | x1.12 |
| `alt_exact` | 25120 | 8480 | x1.23 |

### unit_complex_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 33900 | 16660 | x1.00 |
| `alt_normalized_inputs` | 45820 | 28580 | x1.72 |
| `alt_atan2` | 99430 | 82190 | x4.93 |

### unit_complex_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20900 | 3660 | x1.00 |

### unit_complex_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 85110 | 67470 | x1.00 |

### unit_complex_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `angle_then_compose` | 89300 | 71660 | x1.00 |

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
| `fused` | 20900 | 3660 | x1.00 |

### unit_complex_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20900 | 3660 | x1.00 |

## nalgebra::geometry::unit_quaternion::benches

### unit_quaternion_abs_diff_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `components` | 19850 | 2210 | x1.00 |

### unit_quaternion_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 48710 | 32170 | x1.00 |
| `atan2` | 52030 | 35490 | x1.10 |

### unit_quaternion_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `product_and_atan2` | 65280 | 47740 | x1.00 |

### unit_quaternion_append_axisangle_linearized

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `reduced_product` | 54610 | 35770 | x1.00 |
| `alt_exact` | 80460 | 61620 | x1.72 |

### unit_quaternion_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `try_new` | 37010 | 19470 | x1.00 |

### unit_quaternion_conj_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 30590 | 11550 | x1.00 |
| `alt_conjugate_then_mul` | 31190 | 12150 | x1.05 |

### unit_quaternion_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 19720 | 2180 | x1.00 |

### unit_quaternion_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `five_entries` | 118820 | 102280 | x1.00 |

### unit_quaternion_from_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 58090 | 39850 | x1.00 |

### unit_quaternion_from_euler_angles

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `three_sin_cos` | 132450 | 114210 | x1.00 |

### unit_quaternion_from_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd` | 47020 | 27980 | x1.00 |

### unit_quaternion_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `one_division` | 67810 | 49970 | x1.00 |
| `alt_unit_axis` | 76490 | 58650 | x1.17 |

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
| `folded` | 40410 | 22070 | x1.00 |
| `alt_conjugate_then_transform` | 41010 | 22670 | x1.03 |

### unit_quaternion_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `hamilton` | 30590 | 11550 | x1.00 |

### unit_quaternion_new_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `divisions` | 32820 | 14780 | x1.00 |

### unit_quaternion_new_unchecked

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18040 | 0 | - |

### unit_quaternion_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 42680 | 23240 | x1.00 |

### unit_quaternion_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_angle` | 116130 | 97690 | x1.00 |

### unit_quaternion_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 33360 | 15320 | x1.00 |

### unit_quaternion_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 28720 | 10680 | x1.00 |
| `alt_exact` | 33360 | 15320 | x1.43 |

### unit_quaternion_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `algebraic` | 81510 | 62870 | x1.00 |
| `alt_axis_angle` | 146520 | 127880 | x2.03 |

### unit_quaternion_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_product` | 31190 | 12150 | x1.00 |

### unit_quaternion_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_factor` | 63880 | 46340 | x1.00 |
| `divisions` | 72250 | 54710 | x1.18 |

### unit_quaternion_scaled_rotation_between

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos` | 146520 | 127880 | x1.00 |

### unit_quaternion_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_sin` | 143690 | 124250 | x1.00 |

### unit_quaternion_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 38910 | 22370 | x1.00 |

### unit_quaternion_to_rotation_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_one_minus` | 42310 | 21770 | x1.00 |
| `fused` | 42910 | 22370 | x1.03 |

### unit_quaternion_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 40710 | 22370 | x1.00 |

### unit_quaternion_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `expanded` | 40410 | 22070 | x1.00 |
| `alt_sandwich` | 42440 | 24100 | x1.09 |
| `alt_via_matrix` | 49380 | 31040 | x1.41 |

### unit_quaternion_transform_vector_x2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_matrix` | 54030 | 35690 | x1.00 |
| `expanded` | 63600 | 45260 | x1.27 |

### unit_quaternion_transform_vector_x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `via_matrix` | 60090 | 41750 | x1.00 |
| `expanded` | 85770 | 67430 | x1.62 |

