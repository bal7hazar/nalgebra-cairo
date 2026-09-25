# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_poses::isometry_matrix::benches

### isometry_matrix2_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 42440 | 21600 | x1.00 |

### isometry_matrix2_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 39360 | 18520 | x1.00 |
| `alt_inverse_then_mul` | 42440 | 21600 | x1.17 |

### isometry_matrix2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 24100 | 4660 | x1.00 |

### isometry_matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 38280 | 17440 | x1.00 |

### isometry_matrix2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 22100 | 4060 | x1.00 |
| `alt_rotate_then_add` | 23380 | 5340 | x1.32 |

### isometry_matrix3_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 91980 | 35740 | x1.00 |
| `alt_inverse_then_mul` | 97200 | 40960 | x1.15 |

### isometry_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 46630 | 8040 | x1.00 |

### isometry_matrix3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotation3_frame` | 81100 | 57660 | x1.00 |

### isometry_matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 90360 | 34120 | x1.00 |

### isometry_matrix3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 41730 | 6840 | x1.00 |
| `alt_rotate_then_add` | 43650 | 8760 | x1.28 |

## nalgebra_tests_geometry_poses::poses::benches

### isometry2_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 33750 | 14710 | x1.00 |

### isometry2_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 23100 | 4860 | x1.00 |
| `alt_rotate_then_add` | 24380 | 6140 | x1.26 |

### isometry3_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 80430 | 58690 | x1.00 |

### isometry3_look_at_lh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unit_quaternion_frame` | 120980 | 100040 | x1.00 |

### isometry3_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 44610 | 24270 | x1.00 |
| `alt_rotate_then_add` | 45930 | 25590 | x1.05 |

### quaternion_exp_real

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `identity` | 85960 | 67920 | x1.00 |

### rotation3_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix` | 69980 | 29390 | x1.00 |

### similarity2_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 31020 | 11280 | x1.00 |

### similarity3_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 113280 | 90640 | x1.00 |

## nalgebra_tests_geometry_poses::similarity_matrix::benches

### similarity_matrix2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose_div` | 34280 | 14140 | x1.00 |

### similarity_matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 44220 | 22480 | x1.00 |

### similarity_matrix2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 25660 | 7420 | x1.00 |

### similarity_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose_div` | 59960 | 20670 | x1.00 |

### similarity_matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 97980 | 40840 | x1.00 |

### similarity_matrix3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 46970 | 11880 | x1.00 |

