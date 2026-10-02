# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_factor::givens

### givens_new

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `cancel_y` | 23280 | 14670 | x1.00 |
| `new` | 23950 | 15340 | x1.05 |

### givens_rotate

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix2x3` | 23520 | 13710 | x1.00 |
| `rows_matrix3x2` | 23520 | 13710 | x1.00 |

## nalgebra_tests_linalg_factor::permutation

### perm6_permute_rows

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `append` | 17510 | 1200 | x1.00 |
| `len` | 19720 | 3410 | x2.84 |
| `determinant` | 19920 | 3610 | x3.01 |
| `vector6` | 27510 | 11200 | x9.33 |
| `columns_matrix6` | 57510 | 41200 | x34.33 |
| `matrix6` | 57510 | 41200 | x34.33 |

## nalgebra_tests_linalg_factor::steps

### gauss_step

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `matrix6_step0` | 90410 | 75400 | x1.00 |
| `matrix6_swap0` | 90410 | 75400 | x1.00 |

### reflection_axis

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `vector6` | 50410 | 41400 | x1.00 |

