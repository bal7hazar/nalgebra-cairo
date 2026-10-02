# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_norms::benches

### matrix2x3_one_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 42460 | 9850 | x1.00 |

### matrix2x3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 78040 | 18930 | x1.00 |
| `alt_per_component_div` | 88750 | 29640 | x1.57 |

### matrix3_from_row_slice

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 48540 | 1500 | x1.00 |
| `alt_span_index` | 56900 | 9860 | x6.57 |

### matrix3_index_linear

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 44010 | 1470 | x1.00 |

### matrix3_index_pair

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 44550 | 2010 | x1.00 |

### matrix3_lp_norm3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 258660 | 216120 | x1.00 |

### matrix3_partial_cmp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 80490 | 2820 | x1.00 |

### matrix3_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 95590 | 17920 | x1.00 |

### matrix3x2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 178010 | 120400 | x1.00 |

### matrix3x4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 102510 | 5180 | x1.00 |

### matrix4_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 245380 | 57370 | x1.00 |

### matrix4_symmetric_part

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 144490 | 14580 | x1.00 |

### matrix4x2_cmpy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 135950 | 31780 | x1.00 |

### matrix5_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 109770 | 14270 | x1.00 |

### matrix6_amax

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 198460 | 66550 | x1.00 |

### matrix6_iamax_full

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 205260 | 72850 | x1.00 |

### matrix6_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 429800 | 159090 | x1.00 |
| `alt_div9` | 430870 | 160160 | x1.01 |

### row_vector3_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 42690 | 4940 | x1.00 |

### vector4_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 204430 | 141180 | x1.00 |
| `library` | 218250 | 155000 | x1.10 |

### vector5_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 72880 | 21090 | x1.00 |

### vector6_argmax

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 38190 | 5080 | x1.00 |

