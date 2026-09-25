# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_ext::abstract_rotation::benches

### abstract_rotation_transform_vector

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `direct` | 62580 | 22170 | x1.00 |
| `trait` | 62580 | 22170 | x1.00 |

## nalgebra_tests_geometry_ext::inplace::benches

### point3_apply

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_map` | 16640 | 700 | x1.00 |
| `apply` | 16640 | 700 | x1.00 |

### rotation2_mul_assign_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_assign` | 68780 | 10050 | x1.00 |
| `alt_by_value` | 68880 | 10150 | x1.01 |

## nalgebra_tests_geometry_ext::point::benches

### point3_from_slice

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `checked` | 20580 | 2740 | x1.00 |

### point3_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 16340 | 0 | - |

### point3_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `component_wise` | 51590 | 34850 | x1.00 |

### point4_from_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div4` | 30700 | 12460 | x1.00 |

### point4_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 16540 | 0 | - |

### point4_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 32380 | 7920 | x1.00 |

### point4_partial_ord

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `le` | 18220 | 1080 | x1.00 |

### point4_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps_eq` | 36980 | 19840 | x1.00 |
| `component_wise` | 63370 | 46230 | x2.33 |

### point4_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div4` | 38110 | 11370 | x1.00 |

### point6_lerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 40900 | 11880 | x1.00 |

## nalgebra_tests_geometry_ext::rotation2::benches

### rotation2_from_matrix_eps

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 42240 | 12650 | x1.00 |
| `iterate_8` | 342560 | 312970 | x24.74 |

### rotation2_from_scaled_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `new` | 69850 | 31700 | x1.00 |

### rotation2_mul_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 108840 | 3760 | x1.00 |
| `first_column` | 108840 | 3760 | x1.00 |

### rotation2_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `four_kernels` | 117760 | 7520 | x1.00 |
| `div` | 120290 | 10050 | x1.34 |

### rotation2_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `through_unit_complex` | 247240 | 72360 | x1.00 |

## nalgebra_tests_geometry_ext::rotation3::benches

### rotation3_angle_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused_trace` | 536600 | 32230 | x1.00 |
| `alt_rotation_to_angle` | 584320 | 79950 | x2.48 |

### rotation3_axis_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_then_angle` | 353500 | 53950 | x1.00 |

### rotation3_euler_angles_ordered

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shuster_markley` | 694920 | 226260 | x1.00 |

### rotation3_from_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `closed_form` | 1068580 | 529600 | x1.00 |
| `iterate_8` | 1943680 | 1404700 | x2.65 |
| `alt_matrix_iterate_8` | 2164840 | 1625860 | x3.07 |

### rotation3_index

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `match` | 243900 | 0 | - |

### rotation3_look_at_lh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `face_towards_transposed` | 107140 | 45170 | x1.00 |

### rotation3_mul_unit_quaternion

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `shepperd_then_hamilton` | 325860 | 39730 | x1.00 |
| `div` | 327060 | 40930 | x1.03 |

### rotation3_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rodrigues` | 171300 | 76900 | x1.00 |

### rotation3_powf

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `axis_angle` | 479700 | 115900 | x1.00 |

### rotation3_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `ulps_eq` | 289740 | 44140 | x1.00 |
| `component_wise` | 348730 | 103130 | x2.34 |

### rotation3_rotation_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `div` | 520860 | 22250 | x1.00 |
| `product_with_transpose` | 520860 | 22250 | x1.00 |

### rotation3_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `through_quaternions` | 886720 | 205180 | x1.00 |
| `try_slerp` | 888220 | 206680 | x1.01 |

## nalgebra_tests_geometry_ext::translation::benches

### translation2_mul_unit_complex

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `wrap` | 16640 | 300 | x1.00 |

### translation3_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `subtract` | 20560 | 2420 | x1.00 |
| `relative_eq` | 54610 | 36470 | x15.07 |

### translation3_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_unit_quaternion` | 17540 | 0 | - |
| `add_translation` | 20560 | 3020 | - |

### translation4_to_homogeneous

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix5` | 16140 | 0 | - |

### translation4_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 22300 | 3260 | x1.00 |

### translation6_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `add` | 22280 | 4340 | x1.00 |

