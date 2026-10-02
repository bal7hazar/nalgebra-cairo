# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_solve::solve::benches

### matrix2_solve_vector2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 14410 | 5000 | x1.00 |
| `lower_unchecked` | 17920 | 8510 | x1.70 |
| `lower` | 18320 | 8910 | x1.78 |
| `tr_lower` | 18320 | 8910 | x1.78 |
| `upper` | 18320 | 8910 | x1.78 |

### matrix3_solve_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 20440 | 9830 | x1.00 |
| `lower_unchecked` | 24600 | 13990 | x1.42 |
| `tr_lower` | 24930 | 14320 | x1.46 |
| `upper` | 24930 | 14320 | x1.46 |
| `lower` | 25200 | 14590 | x1.48 |

### matrix4_solve_vector4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 27340 | 15130 | x1.00 |
| `lower_unchecked` | 31610 | 19400 | x1.28 |
| `tr_lower` | 32140 | 19930 | x1.32 |
| `upper` | 32140 | 19930 | x1.32 |
| `lower` | 32410 | 20200 | x1.34 |

### matrix6_solve_vector6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 42670 | 26060 | x1.00 |
| `lower_unchecked` | 47700 | 31090 | x1.19 |
| `lower` | 48900 | 32290 | x1.24 |
| `tr_lower` | 48900 | 32290 | x1.24 |
| `upper` | 48900 | 32290 | x1.24 |

