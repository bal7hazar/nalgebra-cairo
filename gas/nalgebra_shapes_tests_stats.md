# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_stats::benches

### matrix2x3_row_variance_tr

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 94170 | 38460 | x1.00 |

### matrix3_column_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 109440 | 43800 | x1.00 |

### matrix3_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 60110 | 9740 | x1.00 |

### matrix3_product

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 63810 | 13440 | x1.00 |

### matrix3_row_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 85160 | 19520 | x1.00 |
| `alt_div_each` | 85360 | 19720 | x1.01 |

### matrix3_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 53650 | 3280 | x1.00 |
| `alt_checked` | 58820 | 8450 | x2.58 |

### matrix3_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 73910 | 23540 | x1.00 |

### matrix6_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 157580 | 17840 | x1.00 |

### matrix6_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 148420 | 8680 | x1.00 |
| `alt_checked` | 170870 | 31130 | x3.59 |

### matrix6_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 196760 | 57020 | x1.00 |

### row_vector2_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 26070 | 740 | x1.00 |
| `alt_wide` | 27210 | 1880 | x2.54 |

### row_vector3_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 31990 | 1480 | x1.00 |
| `alt_wide` | 32590 | 2080 | x1.41 |

### row_vector4_sum

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 36040 | 2220 | x1.00 |
| `alt_wide` | 36100 | 2280 | x1.03 |

### vector6_mean

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 49280 | 8840 | x1.00 |

### vector6_variance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 60260 | 19820 | x1.00 |

