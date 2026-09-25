# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_views::benches

### matrix2_kronecker_matrix3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 275660 | 68510 | x1.00 |

### matrix2x4_fixed_resize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_direct` | 86290 | 1900 | x1.00 |
| `library` | 86290 | 1900 | x1.00 |

### matrix3_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 68350 | 2310 | x1.00 |

### matrix3_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 68350 | 2310 | x1.00 |

### matrix3_upper_triangle

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 88300 | 1000 | x1.00 |

### matrix4_fixed_view

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 97030 | 3810 | x1.00 |
| `alt_composed` | 97930 | 4710 | x1.24 |
| `alt_padded` | 101760 | 8540 | x2.24 |

### matrix4_insert_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 163280 | 9900 | x1.00 |

### matrix4_remove_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 126520 | 4820 | x1.00 |

### matrix4_row_part

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 92420 | 3210 | x1.00 |

### matrix4_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_fixed_rows` | 111270 | 4010 | x1.00 |
| `library` | 111270 | 4010 | x1.00 |

### matrix4_select_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 114260 | 6300 | x1.00 |

### matrix6_fixed_resize

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 181670 | 4600 | x1.00 |

### matrix6_fixed_view

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 184670 | 7200 | x1.00 |
| `alt_composed` | 195570 | 18100 | x2.51 |

### vector3_zyx

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 46480 | 700 | x1.00 |

### vector6_fixed_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 58310 | 2200 | x1.00 |

