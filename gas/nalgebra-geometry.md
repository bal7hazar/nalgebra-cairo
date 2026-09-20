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
| `compose_rotate` | 26740 | 8100 | x1.00 |

### isometry2_append_rotation_wrt_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 23040 | 4200 | x1.00 |

### isometry2_append_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shift_rotate` | 31380 | 12140 | x1.00 |

### isometry2_append_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 20620 | 1780 | x1.00 |

### isometry2_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 18640 | 400 | x1.00 |

### isometry2_from_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 17440 | -200 | - |

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
| `direct` | 30780 | 11740 | x1.00 |
| `alt_inverse_then_mul` | 34400 | 15360 | x1.31 |

### isometry2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 23040 | 5000 | x1.00 |

### isometry2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract_rotate` | 23120 | 5480 | x1.00 |

### isometry2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 21640 | 4000 | x1.00 |

### isometry2_lerp_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 37960 | 18520 | x1.00 |

### isometry2_lerp_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `atan2_sin_cos` | 70130 | 50690 | x1.00 |

### isometry2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 29700 | 10660 | x1.00 |

### isometry2_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 35260 | 17220 | x1.00 |

### isometry2_prepend_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 23040 | 4200 | x1.00 |

### isometry2_prepend_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate_add` | 23440 | 4600 | x1.00 |

### isometry2_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 26140 | 7900 | x1.00 |

### isometry2_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 26040 | 7800 | x1.00 |

### isometry2_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sin_cos` | 34060 | 16620 | x1.00 |

### isometry2_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix3` | 17040 | 500 | x1.00 |

### isometry2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 22040 | 4400 | x1.00 |
| `alt_rotate_then_add` | 23320 | 5680 | x1.29 |

### isometry2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 21640 | 4000 | x1.00 |

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
| `compose_rotate` | 57030 | 35890 | x1.00 |

### isometry3_append_rotation_wrt_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 34000 | 12560 | x1.00 |

### isometry3_append_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shift_rotate` | 62950 | 41010 | x1.00 |

### isometry3_append_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 24160 | 2820 | x1.00 |

### isometry3_face_towards

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 85350 | 64410 | x1.00 |

### isometry3_from_parts

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 21040 | 700 | x1.00 |

### isometry3_from_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `pure` | 19340 | -200 | - |

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
| `direct` | 63050 | 41310 | x1.00 |
| `alt_inverse_then_mul` | 85660 | 63920 | x1.55 |

### isometry3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 47170 | 27030 | x1.00 |

### isometry3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract_rotate` | 46890 | 27950 | x1.00 |

### isometry3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `conjugate_rotate` | 42770 | 23830 | x1.00 |

### isometry3_lerp_nlerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lerp_normalize` | 53750 | 31610 | x1.00 |

### isometry3_lerp_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `acos_sin` | 113790 | 91650 | x1.00 |

### isometry3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `gram_schmidt` | 109880 | 88940 | x1.00 |

### isometry3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 58530 | 36790 | x1.00 |

### isometry3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 55760 | 35620 | x1.00 |

### isometry3_prepend_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `compose` | 34000 | 12560 | x1.00 |

### isometry3_prepend_translation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate_add` | 46170 | 24830 | x1.00 |

### isometry3_renormalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `exact` | 34220 | 14080 | x1.00 |

### isometry3_renormalize_fast

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `newton` | 32140 | 12000 | x1.00 |

### isometry3_rotation

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_scaled_axis` | 54060 | 34720 | x1.00 |

### isometry3_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix4` | 40970 | 23830 | x1.00 |

### isometry3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 43370 | 24430 | x1.00 |
| `alt_rotate_then_add` | 44690 | 25750 | x1.05 |
| `alt_rotation_matrix` | 54480 | 35540 | x1.45 |

### isometry3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotate` | 42170 | 23230 | x1.00 |

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

## nalgebra::geometry::similarity2::benches

### similarity2_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 44720 | 25980 | x1.00 |
| `alt_inverse_then_mul` | 53510 | 34770 | x1.34 |

### similarity2_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 29000 | 11160 | x1.00 |
| `alt_reciprocal` | 33610 | 15770 | x1.41 |

### similarity2_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 27520 | 9680 | x1.00 |
| `alt_reciprocal` | 32130 | 14290 | x1.48 |

### similarity2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_translate` | 25940 | 8100 | x1.00 |
| `alt_rotate_scale_add` | 29550 | 11710 | x1.45 |

### similarity2_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_after_rotate` | 25540 | 7700 | x1.00 |

## nalgebra::geometry::similarity3::benches

### similarity3_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 82820 | 61980 | x1.00 |
| `alt_inverse_then_mul` | 119750 | 98910 | x1.60 |

### similarity3_inverse_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 58450 | 39310 | x1.00 |
| `alt_reciprocal` | 60670 | 41530 | x1.06 |

### similarity3_inverse_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `division` | 51490 | 32350 | x1.00 |
| `alt_reciprocal` | 56550 | 37410 | x1.16 |

### similarity3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_translate` | 51280 | 32140 | x1.00 |
| `alt_rotate_scale_add` | 53170 | 34030 | x1.06 |

### similarity3_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `scale_after_rotate` | 47920 | 28780 | x1.00 |

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

