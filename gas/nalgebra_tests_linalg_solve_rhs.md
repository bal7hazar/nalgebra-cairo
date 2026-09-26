# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_solve_rhs::solve::benches

### matrix2_solve_matrix2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `lower_unchecked` | 34460 | 16820 | x1.00 |
| `lower` | 35160 | 17520 | x1.04 |

### matrix3_solve_matrix3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_columns` | 61270 | 41630 | x1.00 |
| `lower_unchecked` | 74050 | 54410 | x1.31 |
| `lower` | 74350 | 54710 | x1.31 |

### matrix4_solve_matrix4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_columns` | 100070 | 77630 | x1.00 |
| `lower_unchecked` | 120260 | 97820 | x1.26 |
| `lower` | 120660 | 98220 | x1.27 |

### matrix6_solve_matrix6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `alt_columns` | 216940 | 186500 | x1.00 |
| `lower_unchecked` | 257440 | 227000 | x1.22 |
| `lower` | 258040 | 227600 | x1.22 |

