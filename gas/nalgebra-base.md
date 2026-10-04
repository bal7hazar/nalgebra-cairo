# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::base::matrix2::tests

### matrix2_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 11010 | - | x1.00 |

### matrix2_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 9410 | - | x1.00 |

### matrix2_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 17030 | - | x1.00 |

### matrix2_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `generic` | 17030 | 7320 | x1.00 |
| `structured` | 17380 | 7670 | x1.05 |

### matrix2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 25500 | - | x1.00 |
| `alt_div_n` | 31780 | - | x1.25 |
| `alt_div` | 32550 | - | x1.28 |

## nalgebra::base::matrix3::benches_views

### matrix3_column_internal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `column2` | 11410 | 300 | x1.00 |
| `column` | 12320 | 1210 | x4.03 |

### matrix3_row_internal

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `row2` | 11410 | 300 | x1.00 |
| `row` | 12320 | 1210 | x4.03 |

## nalgebra::base::matrix3::tests

### matrix3_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 30530 | - | x1.00 |

### matrix3_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 11010 | - | x1.00 |

### matrix3_cross_matrix_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 34260 | - | x1.00 |

### matrix3_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 28330 | - | x1.00 |

### matrix3_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 27220 | 15010 | x1.00 |
| `generic` | 30530 | 18320 | x1.22 |

### matrix3_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 11010 | - | x1.00 |

### matrix3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 60060 | - | x1.00 |
| `alt_div_n` | 77840 | - | x1.30 |
| `alt_div` | 80910 | - | x1.35 |

## nalgebra::base::matrix4::tests

### matrix4_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 85460 | - | x1.00 |

### matrix4_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 13010 | - | x1.00 |

### matrix4_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 45990 | - | x1.00 |

### matrix4_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 13010 | - | x1.00 |

### matrix4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 148400 | - | x1.00 |
| `alt_div_n` | 182280 | - | x1.23 |
| `alt_div` | 188570 | - | x1.27 |

## nalgebra::base::point2::benches

### point2_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 13070 | - | x1.00 |

### point2_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 12410 | - | x1.00 |
| `alt_sqrt` | 14090 | - | x1.14 |

### point2_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 12170 | - | x1.00 |

## nalgebra::base::point3::benches

### point3_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 15850 | - | x1.00 |

### point3_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 13750 | - | x1.00 |
| `alt_sqrt` | 15430 | - | x1.12 |

### point3_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 13510 | - | x1.00 |

## nalgebra::base::sym_matrix2::tests

### sym_matrix2_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 24900 | 14590 | x1.00 |
| `generic` | 24950 | 14640 | x1.00 |

### sym_matrix2_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 10410 | 400 | x1.00 |

## nalgebra::base::sym_matrix3::tests

### sym_matrix3_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 43440 | 30430 | x1.00 |
| `generic` | 48550 | 35540 | x1.17 |

### sym_matrix3_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 14010 | 900 | x1.00 |

## nalgebra::base::vector3::benches

### vector3_orthonormal_basis_zneg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 27060 | - | x1.00 |

### vector3_orthonormal_basis_zpos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 26900 | - | x1.00 |

