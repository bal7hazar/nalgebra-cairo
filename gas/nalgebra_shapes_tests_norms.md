# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_norms::benches

### matrix2x3_one_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 50290 | 9850 | x1.00 |

### matrix2x3_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 85870 | 18930 | x1.00 |
| `alt_per_component_div` | 96580 | 29640 | x1.57 |

### matrix3_from_row_slice

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 56370 | 1500 | x1.00 |
| `alt_span_index` | 64730 | 9860 | x6.57 |

### matrix3_index_linear

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 51840 | 1470 | x1.00 |

### matrix3_index_pair

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 52380 | 2010 | x1.00 |

### matrix3_lp_norm3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 266490 | 216120 | x1.00 |

### matrix3_partial_cmp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 88320 | 2820 | x1.00 |

### matrix3_relative_eq

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 103420 | 17920 | x1.00 |

### matrix3x2_angle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 185840 | 120400 | x1.00 |

### matrix3x4_dot

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 110340 | 5180 | x1.00 |

### matrix4_component_div

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 253210 | 57370 | x1.00 |

### matrix4_symmetric_part

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 152320 | 14580 | x1.00 |

### matrix4x2_cmpy

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 143780 | 31780 | x1.00 |

### matrix5_norm

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 117600 | 14270 | x1.00 |

### matrix6_amax

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 206290 | 66550 | x1.00 |

### matrix6_iamax_full

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 213090 | 72850 | x1.00 |

### matrix6_unscale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 437630 | 159090 | x1.00 |
| `alt_div9` | 438700 | 160160 | x1.01 |

### row_vector3_metric_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 50520 | 4940 | x1.00 |

### vector4_slerp

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_acos` | 212260 | 141180 | x1.00 |
| `library` | 226080 | 155000 | x1.10 |

### vector5_normalize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 80710 | 21090 | x1.00 |

### vector6_argmax

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 46020 | 5080 | x1.00 |

