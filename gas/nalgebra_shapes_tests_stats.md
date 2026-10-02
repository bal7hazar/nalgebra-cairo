# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_stats::benches

### matrix2x3_row_variance_tr

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 86340 | 38460 | x1.00 |

### matrix3_column_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 101610 | 43800 | x1.00 |

### matrix3_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 52280 | 9740 | x1.00 |

### matrix3_product

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 55980 | 13440 | x1.00 |

### matrix3_row_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 77330 | 19520 | x1.00 |
| `alt_div_each` | 77530 | 19720 | x1.01 |

### matrix3_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 45820 | 3280 | x1.00 |
| `alt_checked` | 50990 | 8450 | x2.58 |

### matrix3_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 66080 | 23540 | x1.00 |

### matrix6_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 149750 | 17840 | x1.00 |

### matrix6_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 140590 | 8680 | x1.00 |
| `alt_checked` | 163040 | 31130 | x3.59 |

### matrix6_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 188930 | 57020 | x1.00 |

### row_vector2_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 18240 | 740 | x1.00 |
| `alt_wide` | 19380 | 1880 | x2.54 |

### row_vector3_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 24160 | 1480 | x1.00 |
| `alt_wide` | 24760 | 2080 | x1.41 |

### row_vector4_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 28210 | 2220 | x1.00 |
| `alt_wide` | 28270 | 2280 | x1.03 |

### vector6_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 41450 | 8840 | x1.00 |

### vector6_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 52430 | 19820 | x1.00 |

