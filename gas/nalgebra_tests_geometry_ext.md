# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_ext::abstract_rotation::benches

### abstract_rotation_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 48990 | 19740 | x1.00 |
| `trait` | 48990 | 19740 | x1.00 |

## nalgebra_tests_geometry_ext::inplace::benches

### point3_apply

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_map` | 8810 | 700 | x1.00 |
| `apply` | 8810 | 700 | x1.00 |

### rotation2_mul_assign_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 60950 | 10050 | x1.00 |
| `alt_by_value` | 61050 | 10150 | x1.01 |

## nalgebra_tests_geometry_ext::point::benches

### point3_from_slice

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `checked` | 12750 | 2740 | x1.00 |

### point3_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 8510 | 0 | - |

### point3_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_wise` | 43760 | 34850 | x1.00 |

### point4_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div4` | 22870 | 12460 | x1.00 |

### point4_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 8710 | 0 | - |

### point4_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 24550 | 7920 | x1.00 |

### point4_partial_ord

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 10390 | 1080 | x1.00 |

### point4_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps_eq` | 29150 | 19840 | x1.00 |
| `component_wise` | 55540 | 46230 | x2.33 |

### point4_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div4` | 30280 | 11370 | x1.00 |

### point6_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 33070 | 11880 | x1.00 |

## nalgebra_tests_geometry_ext::rotation2::benches

### rotation2_from_matrix_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 34410 | 12650 | x1.00 |
| `iterate_8` | 334730 | 312970 | x24.74 |

### rotation2_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 62020 | 31700 | x1.00 |

### rotation2_mul_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 101010 | 3760 | x1.00 |
| `first_column` | 101010 | 3760 | x1.00 |

### rotation2_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `four_kernels` | 109930 | 7520 | x1.00 |
| `div` | 112460 | 10050 | x1.34 |

### rotation2_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `through_unit_complex` | 239410 | 72360 | x1.00 |

## nalgebra_tests_geometry_ext::rotation3::benches

### rotation3_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_trace` | 528770 | 32230 | x1.00 |
| `alt_rotation_to_angle` | 571830 | 75290 | x2.34 |

### rotation3_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_then_angle` | 345670 | 53950 | x1.00 |

### rotation3_euler_angles_ordered

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shuster_markley` | 672020 | 216170 | x1.00 |

### rotation3_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 1060950 | 529700 | x1.00 |
| `iterate_8` | 1893970 | 1362720 | x2.57 |
| `alt_matrix_iterate_8` | 2086130 | 1554880 | x2.94 |

### rotation3_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 236070 | 0 | - |

### rotation3_look_at_lh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `face_towards_transposed` | 99310 | 45170 | x1.00 |

### rotation3_mul_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd_then_hamilton` | 312570 | 37200 | x1.00 |
| `div` | 319230 | 43860 | x1.18 |

### rotation3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rodrigues` | 163470 | 76900 | x1.00 |

### rotation3_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_angle` | 471870 | 115900 | x1.00 |

### rotation3_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps_eq` | 281910 | 44140 | x1.00 |
| `component_wise` | 340900 | 103130 | x2.34 |

### rotation3_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 505070 | 18720 | x1.00 |
| `product_with_transpose` | 505070 | 18720 | x1.00 |

### rotation3_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `through_quaternions` | 872350 | 201910 | x1.00 |
| `try_slerp` | 873250 | 202810 | x1.00 |

## nalgebra_tests_geometry_ext::translation::benches

### translation2_mul_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 8810 | 300 | x1.00 |

### translation3_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract` | 12730 | 2420 | x1.00 |
| `relative_eq` | 46780 | 36470 | x15.07 |

### translation3_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_unit_quaternion` | 9710 | 0 | - |
| `add_translation` | 12730 | 3020 | - |

### translation4_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix5` | 8310 | 0 | - |

### translation4_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 14470 | 3260 | x1.00 |

### translation6_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 14450 | 4340 | x1.00 |

