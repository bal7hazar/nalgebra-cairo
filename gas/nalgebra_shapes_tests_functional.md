# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_functional::benches

### matrix2x3_mul_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_mul_mat` | 89040 | 12550 | x1.00 |
| `library` | 89040 | 12550 | x1.00 |

### matrix2x3_zip_map

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 90050 | 6340 | x1.00 |

### matrix3_apply_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_norm` | 48180 | 5640 | x1.00 |
| `library` | 48180 | 5640 | x1.00 |

### matrix3_map

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 88530 | 8660 | x1.00 |

### matrix3_swap

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 88830 | 9360 | x1.00 |

### matrix3_swap_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_pair_match` | 85670 | 6200 | x1.00 |
| `library` | 86610 | 7140 | x1.15 |

### matrix3_transpose_mut

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 81370 | 1900 | x1.00 |

### matrix4_apply

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 145550 | 15240 | x1.00 |

### matrix4_fill_lower_triangle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 136330 | 6020 | x1.00 |
| `alt_per_component` | 146400 | 16090 | x2.67 |

### matrix4_fill_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 136520 | 6210 | x1.00 |
| `alt_per_component` | 143510 | 13200 | x2.13 |

### matrix6_set_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 316620 | 21310 | x1.00 |

### row_vector3_apply_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_amax` | 45070 | 7320 | x1.00 |
| `library` | 45070 | 7320 | x1.00 |

### vector3_normalize_mut

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 50550 | 12600 | x1.00 |

### vector4_fold

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 29350 | 2960 | x1.00 |

