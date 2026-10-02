# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_root::benches

### center_point3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `root` | 15440 | 6010 | x1.00 |
| `coords` | 17060 | 7630 | x1.27 |

### distance_point3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 14540 | 5110 | x1.00 |
| `root` | 14540 | 5110 | x1.00 |

### distance_point5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 19560 | 9110 | x1.00 |
| `root` | 19560 | 9110 | x1.00 |

### dmatrix_macro3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_row_slice` | 73390 | 63480 | x1.00 |
| `macro` | 73390 | 63480 | x1.00 |
| `alt_usize_check` | 73690 | 63780 | x1.00 |

### inf_matrix3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `method` | 23210 | 7120 | x1.00 |
| `root` | 23210 | 7120 | x1.00 |

### matrix_macro3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `macro` | 18470 | 8560 | x1.00 |
| `new` | 18470 | 8560 | x1.00 |

### product_matrix3_4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chain` | 81860 | 67650 | x1.00 |
| `snapshots` | 95100 | 80890 | x1.20 |
| `owned_corelib` | 119560 | 105350 | x1.56 |

### stack_2x2_blocks

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `macro` | 15870 | 3700 | x1.00 |
| `new` | 15870 | 3700 | x1.00 |

### sum_dmatrix3_4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chain` | 87390 | 66980 | x1.00 |
| `library` | 94450 | 74040 | x1.11 |

### sum_matrix3_4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chain` | 34590 | 20380 | x1.00 |
| `alt_unrolled4` | 48260 | 34050 | x1.67 |
| `alt_unrolled2` | 48530 | 34320 | x1.68 |
| `library` | 48530 | 34320 | x1.68 |
| `snapshots` | 49330 | 35120 | x1.72 |
| `upstream_from_zero` | 52240 | 38030 | x1.87 |
| `alt_fold` | 61960 | 47750 | x2.34 |

