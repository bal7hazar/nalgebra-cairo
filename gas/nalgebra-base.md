# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra::base::matrix2::tests

### matrix2_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 18840 | - | x1.00 |

### matrix2_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 17240 | - | x1.00 |

### matrix2_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 24860 | - | x1.00 |

### matrix2_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 25210 | 7670 | x1.00 |
| `generic` | 27790 | 10250 | x1.34 |

### matrix2_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 33330 | - | x1.00 |
| `alt_div_n` | 39610 | - | x1.19 |
| `alt_div` | 40380 | - | x1.21 |

## nalgebra::base::matrix3::tests

### matrix3_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 38360 | - | x1.00 |

### matrix3_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 18840 | - | x1.00 |

### matrix3_cross_matrix_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 42090 | - | x1.00 |

### matrix3_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 36160 | - | x1.00 |

### matrix3_mul_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 35050 | 15010 | x1.00 |
| `generic` | 42790 | 22750 | x1.52 |

### matrix3_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 18840 | - | x1.00 |

### matrix3_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 70620 | - | x1.00 |
| `alt_div_n` | 88400 | - | x1.25 |
| `alt_div` | 91470 | - | x1.30 |

## nalgebra::base::matrix4::tests

### matrix4_adjugate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cofactors` | 93290 | - | x1.00 |

### matrix4_column

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 20840 | - | x1.00 |

### matrix4_from_outer

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `products` | 53820 | - | x1.00 |

### matrix4_row

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `second` | 20840 | - | x1.00 |

### matrix4_try_inverse

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_recip` | 156230 | - | x1.00 |
| `alt_div_n` | 190110 | - | x1.22 |
| `alt_div` | 196400 | - | x1.26 |

## nalgebra::base::point2::benches

### point2_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 20900 | - | x1.00 |

### point2_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 20240 | - | x1.00 |
| `alt_sqrt` | 21920 | - | x1.08 |

### point2_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 20000 | - | x1.00 |

## nalgebra::base::point3::benches

### point3_center

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `sum_prod2` | 23680 | - | x1.00 |

### point3_distance

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `norm` | 21580 | - | x1.00 |
| `alt_sqrt` | 23260 | - | x1.08 |

### point3_distance_squared

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `fused` | 21340 | - | x1.00 |

## nalgebra::base::sym_matrix2::tests

### sym_matrix2_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 32730 | 14590 | x1.00 |
| `generic` | 38440 | 20300 | x1.39 |

### sym_matrix2_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 18240 | 400 | x1.00 |

## nalgebra::base::sym_matrix3::tests

### sym_matrix3_quadform

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `structured` | 51270 | 30430 | x1.00 |
| `generic` | 65840 | 45000 | x1.48 |

### sym_matrix3_to_matrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `struct` | 21840 | 900 | x1.00 |

## nalgebra::base::vector3::benches

### vector3_orthonormal_basis_zneg

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 34890 | - | x1.00 |

### vector3_orthonormal_basis_zpos

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `duff` | 34730 | - | x1.00 |

