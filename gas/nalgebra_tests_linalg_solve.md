# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_linalg_solve::solve::benches

### matrix2_solve_vector2

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 22240 | 5000 | x1.00 |
| `lower_unchecked` | 25750 | 8510 | x1.70 |
| `lower` | 26150 | 8910 | x1.78 |
| `tr_lower` | 26150 | 8910 | x1.78 |
| `upper` | 26150 | 8910 | x1.78 |

### matrix3_solve_vector3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 28270 | 9830 | x1.00 |
| `lower_unchecked` | 32430 | 13990 | x1.42 |
| `tr_lower` | 32760 | 14320 | x1.46 |
| `upper` | 32760 | 14320 | x1.46 |
| `lower` | 33030 | 14590 | x1.48 |

### matrix4_solve_vector4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 35170 | 15130 | x1.00 |
| `lower_unchecked` | 39440 | 19400 | x1.28 |
| `tr_lower` | 39970 | 19930 | x1.32 |
| `upper` | 39970 | 19930 | x1.32 |
| `lower` | 40240 | 20200 | x1.34 |

### matrix6_solve_vector6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `with_diag` | 50500 | 26060 | x1.00 |
| `lower_unchecked` | 55530 | 31090 | x1.19 |
| `lower` | 56730 | 32290 | x1.24 |
| `tr_lower` | 56730 | 32290 | x1.24 |
| `upper` | 56730 | 32290 | x1.24 |

