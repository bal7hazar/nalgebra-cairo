# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_factor::givens

### givens_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cancel_y` | 31110 | 14670 | x1.00 |
| `new` | 31780 | 15340 | x1.05 |

### givens_rotate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix2x3` | 31350 | 13710 | x1.00 |
| `rows_matrix3x2` | 31350 | 13710 | x1.00 |

## nalgebra_tests_linalg_factor::permutation

### perm6_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `append` | 25340 | 1200 | x1.00 |
| `len` | 27550 | 3410 | x2.84 |
| `determinant` | 27750 | 3610 | x3.01 |
| `vector6` | 35340 | 11200 | x9.33 |
| `columns_matrix6` | 65340 | 41200 | x34.33 |
| `matrix6` | 65340 | 41200 | x34.33 |

## nalgebra_tests_linalg_factor::steps

### gauss_step

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix6_step0` | 98240 | 75400 | x1.00 |
| `matrix6_swap0` | 98240 | 75400 | x1.00 |

### reflection_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector6` | 58240 | 41400 | x1.00 |

