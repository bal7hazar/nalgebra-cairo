# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_root::benches

### center_point3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `root` | 23270 | 6010 | x1.00 |
| `coords` | 24890 | 7630 | x1.27 |

### distance_point3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 22370 | 5110 | x1.00 |
| `root` | 22370 | 5110 | x1.00 |

### distance_point5

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `coords` | 27390 | 9110 | x1.00 |
| `root` | 27390 | 9110 | x1.00 |

### dmatrix_macro3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `from_row_slice` | 81220 | 63480 | x1.00 |
| `macro` | 81220 | 63480 | x1.00 |
| `alt_usize_check` | 81520 | 63780 | x1.00 |

### inf_matrix3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `method` | 31040 | 7120 | x1.00 |
| `root` | 31040 | 7120 | x1.00 |

### matrix_macro3

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `macro` | 26300 | 8560 | x1.00 |
| `new` | 26300 | 8560 | x1.00 |

### product_matrix3_4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chain` | 89690 | 67650 | x1.00 |
| `snapshots` | 102930 | 80890 | x1.20 |
| `owned_corelib` | 127390 | 105350 | x1.56 |

### stack_2x2_blocks

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `macro` | 23700 | 3700 | x1.00 |
| `new` | 23700 | 3700 | x1.00 |

### sum_dmatrix3_4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chain` | 95220 | 66980 | x1.00 |
| `library` | 102280 | 74040 | x1.11 |

### sum_matrix3_4

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `chain` | 42420 | 20380 | x1.00 |
| `alt_unrolled4` | 56090 | 34050 | x1.67 |
| `alt_unrolled2` | 56360 | 34320 | x1.68 |
| `library` | 56360 | 34320 | x1.68 |
| `snapshots` | 57160 | 35120 | x1.72 |
| `upstream_from_zero` | 60070 | 38030 | x1.87 |
| `alt_fold` | 69790 | 47750 | x2.34 |

