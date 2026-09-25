# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_functional::benches

### matrix2x3_mul_to

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_mul_mat` | 96870 | 12550 | x1.00 |
| `library` | 96870 | 12550 | x1.00 |

### matrix2x3_zip_map

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 97880 | 6340 | x1.00 |

### matrix3_apply_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_norm` | 56010 | 5640 | x1.00 |
| `library` | 56010 | 5640 | x1.00 |

### matrix3_map

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 96360 | 8660 | x1.00 |

### matrix3_swap

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 96660 | 9360 | x1.00 |

### matrix3_swap_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_pair_match` | 93500 | 6200 | x1.00 |
| `library` | 94440 | 7140 | x1.15 |

### matrix3_transpose_mut

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 89200 | 1900 | x1.00 |

### matrix4_apply

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 153380 | 15240 | x1.00 |

### matrix4_fill_lower_triangle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 144160 | 6020 | x1.00 |
| `alt_per_component` | 154230 | 16090 | x2.67 |

### matrix4_fill_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 144350 | 6210 | x1.00 |
| `alt_per_component` | 151340 | 13200 | x2.13 |

### matrix6_set_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 324450 | 21310 | x1.00 |

### row_vector3_apply_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_amax` | 52900 | 7320 | x1.00 |
| `library` | 52900 | 7320 | x1.00 |

### vector3_normalize_mut

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 58380 | 12600 | x1.00 |

### vector4_fold

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 37180 | 2960 | x1.00 |

