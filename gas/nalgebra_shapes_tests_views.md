# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_views::benches

### matrix2_kronecker_matrix3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 267830 | 68510 | x1.00 |

### matrix2x4_fixed_resize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_direct` | 78460 | 1900 | x1.00 |
| `library` | 78460 | 1900 | x1.00 |

### matrix3_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 60520 | 2310 | x1.00 |

### matrix3_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 60520 | 2310 | x1.00 |

### matrix3_upper_triangle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 80470 | 1000 | x1.00 |

### matrix4_fixed_view

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 89200 | 3810 | x1.00 |
| `alt_composed` | 90100 | 4710 | x1.24 |
| `alt_padded` | 93930 | 8540 | x2.24 |

### matrix4_insert_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 155450 | 9900 | x1.00 |

### matrix4_remove_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 118690 | 4820 | x1.00 |

### matrix4_row_part

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 84590 | 3210 | x1.00 |

### matrix4_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_fixed_rows` | 103440 | 4010 | x1.00 |
| `library` | 103440 | 4010 | x1.00 |

### matrix4_select_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 106430 | 6300 | x1.00 |

### matrix6_fixed_resize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 173840 | 4600 | x1.00 |

### matrix6_fixed_view

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 176840 | 7200 | x1.00 |
| `alt_composed` | 187740 | 18100 | x2.51 |

### vector3_zyx

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 38650 | 700 | x1.00 |

### vector6_fixed_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 50480 | 2200 | x1.00 |

