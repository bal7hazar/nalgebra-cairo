# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_core::benches

### matrix2x3_mul_matrix3x2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 89040 | 12550 | x1.00 |

### matrix2x3_mul_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 62630 | 5160 | x1.00 |

### matrix2x3_tr_mul_matrix2x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_mat` | 115690 | 21150 | x1.00 |
| `tr_mul` | 115690 | 21150 | x1.00 |

### matrix3x2_mul_matrix2x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 115690 | 21150 | x1.00 |

### matrix3x2_tr_mul_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_mat` | 62630 | 5160 | x1.00 |
| `tr_mul` | 62630 | 5160 | x1.00 |

### matrix4_mul_matrix4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 234320 | 46310 | x1.00 |
| `operator` | 234320 | 46310 | x1.00 |

### matrix5_mul_matrix5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 357310 | 76330 | x1.00 |
| `operator` | 357310 | 76330 | x1.00 |

### matrix5_mul_vector5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 159910 | 20230 | x1.00 |

### matrix5_tr_mul_vector5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_mat` | 159910 | 20230 | x1.00 |
| `tr_mul` | 159910 | 20230 | x1.00 |

### matrix6x4_mul_matrix4x6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 412280 | 97110 | x1.00 |

### row_vector3_mul_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 46910 | 2780 | x1.00 |

### row_vector6_mul_vector6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_nested` | 67970 | 3980 | x1.00 |
| `mul_mat` | 67970 | 3980 | x1.00 |

### vector3_mul_row_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 92830 | 18150 | x1.00 |

