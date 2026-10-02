# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_geometry_poses::isometry_matrix::benches

### isometry_matrix2_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 34610 | 21600 | x1.00 |

### isometry_matrix2_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 31530 | 18520 | x1.00 |
| `alt_inverse_then_mul` | 34610 | 21600 | x1.17 |

### isometry_matrix2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 16270 | 4660 | x1.00 |

### isometry_matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 30450 | 17440 | x1.00 |

### isometry_matrix2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 14270 | 4060 | x1.00 |
| `alt_rotate_then_add` | 15550 | 5340 | x1.32 |

### isometry_matrix3_inv_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 84150 | 35740 | x1.00 |
| `alt_inverse_then_mul` | 89370 | 40960 | x1.15 |

### isometry_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose` | 38800 | 8040 | x1.00 |

### isometry_matrix3_look_at_rh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `rotation3_frame` | 73270 | 57660 | x1.00 |

### isometry_matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 82530 | 34120 | x1.00 |

### isometry_matrix3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 33900 | 6840 | x1.00 |
| `alt_rotate_then_add` | 35820 | 8760 | x1.28 |

## nalgebra_tests_geometry_poses::poses::benches

### isometry2_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 25920 | 14710 | x1.00 |

### isometry2_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 15270 | 4860 | x1.00 |
| `alt_rotate_then_add` | 16550 | 6140 | x1.26 |

### isometry3_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 72600 | 58690 | x1.00 |

### isometry3_look_at_lh

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `unit_quaternion_frame` | 113150 | 100040 | x1.00 |

### isometry3_rotation_wrt_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 36780 | 24270 | x1.00 |
| `alt_rotate_then_add` | 38100 | 25590 | x1.05 |

### quaternion_exp_real

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `identity` | 78130 | 67920 | x1.00 |

### rotation3_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix` | 62150 | 29390 | x1.00 |

### similarity2_mul_isometry

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 23190 | 11280 | x1.00 |

### similarity3_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `inverse_then_mul` | 105450 | 90640 | x1.00 |

## nalgebra_tests_geometry_poses::similarity_matrix::benches

### similarity_matrix2_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose_div` | 26450 | 14140 | x1.00 |

### similarity_matrix2_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 36390 | 22480 | x1.00 |

### similarity_matrix2_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 17830 | 7420 | x1.00 |

### similarity_matrix3_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `transpose_div` | 52130 | 20670 | x1.00 |

### similarity_matrix3_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `composition` | 90150 | 40840 | x1.00 |

### similarity_matrix3_transform_point

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 39140 | 11880 | x1.00 |

