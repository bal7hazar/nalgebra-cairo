# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_shapes_tests_core::benches

### matrix2x3_mul_matrix3x2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 96870 | 12550 | x1.00 |

### matrix2x3_mul_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 70460 | 5160 | x1.00 |

### matrix2x3_tr_mul_matrix2x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_mat` | 123520 | 21150 | x1.00 |
| `tr_mul` | 123520 | 21150 | x1.00 |

### matrix3x2_mul_matrix2x3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 123520 | 21150 | x1.00 |

### matrix3x2_tr_mul_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_mat` | 70460 | 5160 | x1.00 |
| `tr_mul` | 70460 | 5160 | x1.00 |

### matrix4_mul_matrix4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 242150 | 46310 | x1.00 |
| `operator` | 242150 | 46310 | x1.00 |

### matrix5_mul_matrix5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 365140 | 76330 | x1.00 |
| `operator` | 365140 | 76330 | x1.00 |

### matrix5_mul_vector5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 167740 | 20230 | x1.00 |

### matrix5_tr_mul_vector5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_transpose_then_mul_mat` | 167740 | 20230 | x1.00 |
| `tr_mul` | 167740 | 20230 | x1.00 |

### matrix6x4_mul_matrix4x6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 420110 | 97110 | x1.00 |

### row_vector3_mul_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 54740 | 2780 | x1.00 |

### row_vector6_mul_vector6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_nested` | 75800 | 3980 | x1.00 |
| `mul_mat` | 75800 | 3980 | x1.00 |

### vector3_mul_row_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `mul_mat` | 100660 | 18150 | x1.00 |

