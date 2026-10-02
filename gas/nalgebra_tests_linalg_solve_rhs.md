# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_solve_rhs::solve::benches

### matrix2_solve_matrix2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lower_unchecked` | 26630 | 16820 | x1.00 |
| `lower` | 27330 | 17520 | x1.04 |

### matrix3_solve_matrix3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_columns` | 53440 | 41630 | x1.00 |
| `lower_unchecked` | 66220 | 54410 | x1.31 |
| `lower` | 66520 | 54710 | x1.31 |

### matrix4_solve_matrix4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_columns` | 92240 | 77630 | x1.00 |
| `lower_unchecked` | 112430 | 97820 | x1.26 |
| `lower` | 112830 | 98220 | x1.27 |

### matrix6_solve_matrix6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_columns` | 209110 | 186500 | x1.00 |
| `lower_unchecked` | 249610 | 227000 | x1.22 |
| `lower` | 250210 | 227600 | x1.22 |

