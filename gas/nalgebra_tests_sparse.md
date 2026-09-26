# Gas report

Sierra gas (`l2_gas`) per benchmark; `net` = raw - group baseline.

## nalgebra_tests_sparse::benches

### axpy_cs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 335740 | 104970 | x1.00 |

### convolve

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `valid` | 250240 | 135150 | x1.00 |
| `same` | 357390 | 242300 | x1.79 |
| `full` | 418710 | 303620 | x2.25 |

### convolve_vector6

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `same` | 110020 | 93000 | x1.00 |
| `full` | 137840 | 120820 | x1.30 |

### cs_add

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 7137130 | 660010 | x1.00 |

### cs_cholesky

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `symbolic` | 8407190 | 3578450 | x1.00 |
| `library` | 10399680 | 5570940 | x1.56 |
| `alt_left_looking` | 12903520 | 8074780 | x2.26 |

### cs_cholesky_lap1d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 4276540 | 2614490 | x1.00 |

### cs_cholesky_numeric

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 10395580 | 1988390 | x1.00 |

### cs_from_dmatrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 6742890 | 1002450 | x1.00 |

### cs_from_triplet

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 4828740 | 2161100 | x1.00 |

### cs_into_dmatrix

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 5740440 | 911700 | x1.00 |

### cs_mul

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 7713990 | 4403560 | x1.00 |

### cs_mul_lap2d

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 12374650 | 7545910 | x1.00 |

### cs_scale

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 5046910 | 217470 | x1.00 |

### cs_solve_lower

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 12874980 | 2398010 | x1.00 |

### cs_solve_lower_cs

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 12900820 | 2464380 | x1.00 |

### cs_tr_solve_lower

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 11104270 | 627300 | x1.00 |

### cs_transpose

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 6646580 | 1817840 | x1.00 |
| `alt_packed_keys` | 6972820 | 2144080 | x1.18 |
| `alt_dict` | 8870920 | 4042180 | x2.22 |

### matrix_market

| variant | raw | net | vs best |
|---|---:|---:|---:|
| `library` | 2778370 | 2761160 | x1.00 |

